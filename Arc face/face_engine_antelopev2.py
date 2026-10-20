"""
face_engine_antelopev2.py - Face embedding engine using InsightFace antelopev2.

Backbone:
    - Detection: SCRFD-10G (scrfd_10g_bnkps.onnx)
    - Recognition: Glint360K ResNet-100 (glintr100.onnx, 512-d embeddings)

Drop-in replacement for FaceEngine:
    from face_engine_antelopev2 import FaceEngineAntelopeV2
    engine = FaceEngineAntelopeV2(use_gpu=False, use_tta=False)

    # With optional Flip Test-Time Augmentation for higher cross-age accuracy:
    engine_tta = FaceEngineAntelopeV2(use_gpu=False, use_tta=True)
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align


class FaceEngineAntelopeV2:
    def __init__(self, insightface_home="~/.insightface", use_gpu=True, det_size=(640, 640), use_tta=False):
        """
        Initializes FaceEngine with the antelopev2 model pack (Glint360K ResNet-100).
        
        Args:
            insightface_home: Root directory for models.
            use_gpu: True for CUDA, False for CPU.
            det_size: Detection image scale (width, height).
            use_tta: If True, uses Flip Test-Time Augmentation (averages original and horizontally
                     flipped face embeddings) for higher matching stability across age gaps.
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.use_tta = use_tta
        self.app = FaceAnalysis(
            name="antelopev2",
            root=insightface_home,
            allowed_modules=["detection", "recognition"],
            providers=providers
        )
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_thresh=0.4, det_size=det_size)
        self.rec_model = self.app.models.get("recognition")

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
        if not self.use_tta or self.rec_model is None:
            embedding = f.normed_embedding.astype(np.float32)
        else:
            # Flip-TTA: compute aligned face and its horizontal reflection
            aimg = face_align.norm_crop(img_for_crop, landmark=f.kps, image_size=112)
            aimg_flip = cv2.flip(aimg, 1)
            emb1 = self.rec_model.get_feat(aimg)
            emb2 = self.rec_model.get_feat(aimg_flip)
            comb = emb1 + emb2
            embedding = (comb / np.linalg.norm(comb)).flatten().astype(np.float32)

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
