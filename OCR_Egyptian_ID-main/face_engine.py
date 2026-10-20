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
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self, insightface_home="~/.insightface", use_gpu=True, det_size=(640, 640)):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", root=insightface_home,
                                allowed_modules=["detection", "recognition"], providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_thresh=0.4, det_size=det_size)

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