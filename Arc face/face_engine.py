"""
face_engine.py - the ONE place that turns a photo into face vectors for AI-MIDS.

    from face_engine import FaceEngine
    engine = FaceEngine()                 # load once at startup (slow), reuse for every request

    r = engine.embed("photo.jpg")         # path, or a BGR numpy image. Largest face only.
    if r is None: ...                     # no face found -> ask for another photo
    r["embedding"]                        # 512 floats, length 1.0 (this is what goes into the index)
    r["n_faces"]                          # how many faces were in the photo

    faces = engine.embed_all("group.jpg") # EVERY face, largest first: [{"embedding", "bbox", "det_score"}, ...]
                                          # bbox = [x1, y1, x2, y2] in the original photo's pixels

Rule for the whole system: only vectors from this module go into the search index.
"""
import importlib.util
import os
import sys
import warnings

import numpy as np
import cv2
from insightface.app import FaceAnalysis

_CUDA_DLL_HANDLES = []
_CUDA_DLL_DIRS_ADDED = set()


def ensure_cuda_dll_paths():
    """
    On Windows, registers PyTorch's bundled CUDA 12 / cuDNN 9 DLL directory
    (`torch/lib`) with the Windows DLL loader and PATH before ONNX Runtime
    initializes CUDAExecutionProvider. Safe to call multiple times.
    """
    if os.name != "nt":
        return []

    candidate_dirs = []
    if "torch" in sys.modules and getattr(sys.modules["torch"], "__file__", None):
        candidate_dirs.append(os.path.join(os.path.dirname(sys.modules["torch"].__file__), "lib"))
    else:
        spec = importlib.util.find_spec("torch")
        if spec and spec.submodule_search_locations:
            candidate_dirs.append(os.path.join(list(spec.submodule_search_locations)[0], "lib"))

    added = []
    path_entries = [os.path.normcase(os.path.abspath(p)) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        norm_d = os.path.normcase(os.path.abspath(d))
        if norm_d not in _CUDA_DLL_DIRS_ADDED:
            if hasattr(os, "add_dll_directory"):
                try:
                    _CUDA_DLL_HANDLES.append(os.add_dll_directory(d))
                except OSError:
                    pass
            _CUDA_DLL_DIRS_ADDED.add(norm_d)
        if norm_d not in path_entries:
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            path_entries.insert(0, norm_d)
        added.append(d)

    ort = sys.modules.get("onnxruntime")
    if ort is not None and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True)
        except Exception:
            pass
    return added


def get_ort_providers(use_gpu=True):
    """Returns ONNX Runtime execution providers list, registering CUDA DLL paths when use_gpu=True."""
    if use_gpu:
        ensure_cuda_dll_paths()
        cuda_opts = {
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
            "arena_extend_strategy": "kSameAsRequested",
        }
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _app_uses_cuda(app):
    for model in getattr(app, "models", {}).values():
        sess = getattr(model, "session", None)
        if sess is not None and "CUDAExecutionProvider" in sess.get_providers():
            return True
    return False


def create_face_analysis(name, root="~/.insightface", allowed_modules=None, use_gpu=True,
                         det_thresh=0.4, det_size=(640, 640)):
    """Initializes and prepares InsightFace FaceAnalysis with graceful GPU -> CPU fallback."""
    providers = get_ort_providers(use_gpu=use_gpu)
    try:
        app = FaceAnalysis(
            name=name,
            root=root,
            allowed_modules=allowed_modules,
            providers=providers,
        )
        app.prepare(ctx_id=0 if use_gpu else -1, det_thresh=det_thresh, det_size=det_size)
        return app
    except Exception as e:
        if not use_gpu:
            raise
        warnings.warn(f"CUDA FaceAnalysis init failed for '{name}' ({e}); falling back to CPUExecutionProvider.")
        app = FaceAnalysis(
            name=name,
            root=root,
            allowed_modules=allowed_modules,
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_thresh=det_thresh, det_size=det_size)
        return app


def create_ort_session(onnx_path, use_gpu=True):
    """Creates an ONNX Runtime InferenceSession with graceful GPU -> CPU fallback."""
    import onnxruntime as ort
    providers = get_ort_providers(use_gpu=use_gpu)
    try:
        return ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        if not use_gpu:
            raise
        warnings.warn(f"CUDA InferenceSession failed for '{onnx_path}' ({e}); falling back to CPUExecutionProvider.")
        return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


class FaceEngine:
    model_name = "buffalo_l"   # tags every embedding this engine produces, so the store never mixes models

    def __init__(self, insightface_home="~/.insightface", use_gpu=True, det_size=(640, 640)):
        self.app = create_face_analysis(
            name="buffalo_l",
            root=insightface_home,
            allowed_modules=["detection", "recognition"],
            use_gpu=use_gpu,
            det_thresh=0.4,
            det_size=det_size,
        )
        self.use_gpu = _app_uses_cuda(self.app)

    @staticmethod
    def _read(src):
        if isinstance(src, np.ndarray):
            return src
        # np.fromfile so filenames with special characters also work on Windows
        img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src)
        return img

    def _faces(self, img):
        faces = self.app.get(img)
        pad = 0
        if not faces:  # tight crops (face fills the frame) need a border to be detected
            pad = int(0.5 * max(img.shape[:2]))
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            faces = self.app.get(img)
        return faces, pad

    @staticmethod
    def _pack(f, pad):
        return {
            "embedding": f.normed_embedding.astype(np.float32),
            "bbox": [float(v) - pad for v in f.bbox],
            "kps": (np.asarray(f.kps, dtype=np.float32) - pad).tolist(),  # 5 points: eyes, nose, mouth corners
            "det_score": float(f.det_score),
        }

    def embed_all(self, src):
        """Every face in the photo, largest first. Empty list if none."""
        faces, pad = self._faces(self._read(src))
        faces = sorted(faces, key=lambda q: -(q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
        return [self._pack(f, pad) for f in faces]

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