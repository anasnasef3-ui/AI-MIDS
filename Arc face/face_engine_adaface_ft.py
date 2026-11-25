"""
face_engine_adaface_ft.py - Face embedding engine using fine-tuned AdaFace IR-101 exported to ONNX.

Backbone:
    - Detection: SCRFD-10G (from antelopev2 / buffalo_l)
    - Recognition: Custom fine-tuned AdaFace IR-101 (aimids_adaface_ir101_ft.onnx, 512-d embeddings)

Drop-in replacement for FaceEngineAntelopeV2 and FaceEngine:
    from face_engine_adaface_ft import FaceEngineAdaFaceFT
    engine = FaceEngineAdaFaceFT(use_gpu=False, use_tta=True)
"""
import os
import cv2
import numpy as np
from insightface.utils import face_align

try:
    from face_engine import create_face_analysis, create_ort_session, _app_uses_cuda
except ImportError:
    from .face_engine import create_face_analysis, create_ort_session, _app_uses_cuda

DEFAULT_ONNX_FILENAME = "aimids_adaface_ir101_ft.onnx"


def resolve_onnx_path(onnx_path=None):
    """
    Resolves the ONNX model path in strict priority order:
      1. Explicit argument `onnx_path` (raises FileNotFoundError if missing)
      2. Environment variable `AIMIDS_ONNX_PATH` (raises FileNotFoundError if missing)
      3. Arc face/pretrained: Main files/Arc face/pretrained/aimids_adaface_ir101_ft.onnx
      4. Local Arc face directory: Main files/Arc face/aimids_adaface_ir101_ft.onnx
      5. Project checkpoints directory: checkpoints/aimids_adaface_ir101_ft.onnx
    Raises FileNotFoundError with clear details if no candidate file exists on disk.
    """
    if onnx_path:
        p = os.path.abspath(onnx_path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Explicit ONNX model path not found: {p}")
        return p

    env_path = os.environ.get("AIMIDS_ONNX_PATH")
    if env_path and env_path.strip():
        p = os.path.abspath(env_path.strip())
        if not os.path.isfile(p):
            raise FileNotFoundError(f"AIMIDS_ONNX_PATH is set but file not found: {p}")
        return p

    loc_pretrained = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained", DEFAULT_ONNX_FILENAME)
    if os.path.isfile(loc_pretrained):
        return os.path.abspath(loc_pretrained)

    loc1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_ONNX_FILENAME)
    if os.path.isfile(loc1):
        return os.path.abspath(loc1)

    loc2 = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints", DEFAULT_ONNX_FILENAME)
    )
    if os.path.isfile(loc2):
        return loc2

    searched = [loc_pretrained, loc1, loc2]
    raise FileNotFoundError(
        f"Fine-tuned AdaFace ONNX model '{DEFAULT_ONNX_FILENAME}' not found. "
        f"Searched locations: {searched}. "
        f"Set AIMIDS_ONNX_PATH or place the model file in one of the searched locations."
    )


class FaceEngineAdaFaceFT:
    model_name = "aimids_ft"  # tags every embedding produced by this engine

    def __init__(self, onnx_path=None, insightface_home="~/.insightface", use_gpu=True,
                 det_size=(640, 640), use_tta=False, det_app=None):
        """
        Initializes FaceEngineAdaFaceFT with custom fine-tuned AdaFace IR-101 ONNX model.

        Args:
            onnx_path: Explicit path to .onnx file (defaults to resolve_onnx_path())
            insightface_home: Root directory for insightface models.
            use_gpu: True for CUDA, False for CPU.
            det_size: Detection image scale (width, height).
            use_tta: If True, uses Flip Test-Time Augmentation (averages original and horizontally flipped face embeddings).
            det_app: Optional pre-initialized FaceAnalysis detection instance.
        """
        self.onnx_path = resolve_onnx_path(onnx_path)
        self.use_tta = bool(use_tta)
        self.ort_session = create_ort_session(self.onnx_path, use_gpu=use_gpu)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        if det_app is not None:
            self.app = det_app
        else:
            try:
                self.app = create_face_analysis(
                    name="antelopev2",
                    root=insightface_home,
                    allowed_modules=["detection"],
                    use_gpu=use_gpu,
                    det_thresh=0.4,
                    det_size=det_size,
                )
            except Exception:
                self.app = create_face_analysis(
                    name="buffalo_l",
                    root=insightface_home,
                    allowed_modules=["detection"],
                    use_gpu=use_gpu,
                    det_thresh=0.4,
                    det_size=det_size,
                )
        self.use_gpu = (
            "CUDAExecutionProvider" in self.ort_session.get_providers()
            or _app_uses_cuda(self.app)
        )

    @staticmethod
    def chip_bgr_to_rgb_chw(chip_bgr):
        """
        Converts 112x112 BGR uint8 crop to normalized RGB float32 CHW tensor in [-1.0, 1.0].
        """
        if chip_bgr.shape[0] != 112 or chip_bgr.shape[1] != 112:
            chip_bgr = cv2.resize(chip_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(chip_bgr, cv2.COLOR_BGR2RGB)
        arr = (rgb.astype(np.float32) / 127.5) - 1.0
        chw = np.transpose(arr, (2, 0, 1))
        return np.ascontiguousarray(chw, dtype=np.float32)

    def _embed_chip(self, chip_bgr):
        """Runs the ONNX model on a single 112x112 aligned face chip and returns a unit vector."""
        inp = self.chip_bgr_to_rgb_chw(chip_bgr)[None, ...]
        out = self.ort_session.run([self.output_name], {self.input_name: inp})[0]
        emb = out.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb

    @staticmethod
    def _read(src):
        if isinstance(src, np.ndarray):
            return src
        # np.fromfile so filenames with special/Unicode characters also work on Windows
        img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src)
        return img

    def _faces(self, img):
        faces = self.app.get(img)
        pad = 0
        img_for_crop = img
        if not faces:  # tight crops (face fills the frame) need a border to be detected
            pad = int(0.5 * max(img.shape[:2]))
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            faces = self.app.get(padded)
            img_for_crop = padded
        return faces, pad, img_for_crop

    def _pack(self, f, pad, img_for_crop):
        aimg = face_align.norm_crop(img_for_crop, landmark=f.kps, image_size=112)
        if not self.use_tta:
            embedding = self._embed_chip(aimg)
        else:
            aimg_flip = cv2.flip(aimg, 1)
            emb1 = self._embed_chip(aimg)
            emb2 = self._embed_chip(aimg_flip)
            comb = emb1 + emb2
            norm = float(np.linalg.norm(comb))
            embedding = (comb / norm).astype(np.float32) if norm > 0 else comb.astype(np.float32)

        return {
            "embedding": embedding,
            "bbox": [float(v) - pad for v in f.bbox],
            "kps": (np.asarray(f.kps, dtype=np.float32) - pad).tolist(),  # 5 points: eyes, nose, mouth corners
            "det_score": float(f.det_score),
        }

    def embed_all(self, src):
        """Every face in the photo, largest first. Empty list if none."""
        img = self._read(src)
        faces, pad, img_for_crop = self._faces(img)
        faces = sorted(faces, key=lambda q: -(q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
        return [self._pack(f, pad, img_for_crop) for f in faces]

    def embed(self, src):
        """Largest face only, plus n_faces. None if no face is found."""
        found = self.embed_all(src)
        if not found:
            return None
        best = dict(found[0])
        best["n_faces"] = len(found)
        return best

    @staticmethod
    def similarity(a, b):
        return float(np.dot(a, b))
