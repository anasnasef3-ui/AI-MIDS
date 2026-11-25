"""
face_engine_custom.py - Cross-age dual-branch face embedding engine & AS-Norm utilities.

Combines two complementary backbones under a single SCRFD-10G detector pass:
    - Branch A (512-d): InsightFace antelopev2 (Glint360K ResNet-100, glintr100)
    - Branch B (512-d): Fine-tuned AdaFace IR-101 ONNX (aimids_adaface_ir101_ft.onnx)

For each detected face, the 112x112 chip is aligned once, embedded with optional
Flip-TTA on both branches, weighted by [sqrt(w_a) * emb_a, sqrt(w_b) * emb_b], and
L2-normalized into a 1024-d float32 unit vector.

Also provides pure-NumPy Adaptive Score Normalization (AS-Norm) helpers for
cohort-normalized evaluation and FaceStore similarity calibration.
"""
import cv2
import numpy as np
from insightface.utils import face_align

try:
    from face_engine import ensure_cuda_dll_paths
    from face_engine_antelopev2 import FaceEngineAntelopeV2
    from face_engine_adaface_ft import FaceEngineAdaFaceFT, resolve_onnx_path
    from photo_quality import SMALL_FACE_PX, frontalness
except ImportError:
    from .face_engine import ensure_cuda_dll_paths
    from .face_engine_antelopev2 import FaceEngineAntelopeV2
    from .face_engine_adaface_ft import FaceEngineAdaFaceFT, resolve_onnx_path
    from .photo_quality import SMALL_FACE_PX, frontalness


EMBEDDING_DIM = 1024
DEFAULT_W_A = 0.55
DEFAULT_W_B = 0.45
MIN_QUALITY = 0.25
MAX_QUALITY = 1.0


def compute_face_quality(aligned_chip_bgr, kps, bbox):
    """
    Computes a bounded per-face quality scalar in [0.25, 1.0] from:
      - Laplacian variance sharpness on the 112x112 aligned BGR chip
      - Eye-nose frontalness via photo_quality.frontalness(kps)
      - Bounding-box height normalized by 80 px (photo_quality.SMALL_FACE_PX), capped at 1.0
    """
    gray = cv2.cvtColor(aligned_chip_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    q_sharp = min(max(sharp / 100.0, 0.0), 1.0)

    # frontalness(kps) is 0.5 when straight-on, 0.0 or 1.0 when side-on.
    front = frontalness(kps)
    q_front = min(max(1.0 - 2.0 * abs(front - 0.5), 0.0), 1.0)

    bbox_h = float(bbox[3]) - float(bbox[1])
    q_size = min(max(bbox_h / float(SMALL_FACE_PX), 0.0), 1.0)

    raw_q = 0.4 * q_sharp + 0.3 * q_front + 0.3 * q_size
    return float(min(max(raw_q, MIN_QUALITY), MAX_QUALITY))


class FaceEngineCrossAge:
    model_name = "aimids_crossage"
    embedding_dim = EMBEDDING_DIM

    def __init__(
        self,
        onnx_path=None,
        insightface_home="~/.insightface",
        use_gpu=True,
        det_size=(640, 640),
        use_tta=True,
        w_a=DEFAULT_W_A,
        w_b=DEFAULT_W_B,
    ):
        """
        Initializes the cross-age dual-branch engine with a single shared SCRFD-10G detector.

        Args:
            onnx_path: Optional explicit path to aimids_adaface_ir101_ft.onnx.
            insightface_home: Root directory for InsightFace models.
            use_gpu: True for CUDA, False for CPU.
            det_size: Detection input resolution (width, height).
            use_tta: Whether to apply horizontal Flip-TTA on both branches.
            w_a: Weight assigned to Branch A (antelopev2) before square-root scaling.
            w_b: Weight assigned to Branch B (aimids_ft) before square-root scaling.
        """
        w_a = float(w_a)
        w_b = float(w_b)
        if w_a < 0.0 or w_b < 0.0 or (w_a + w_b) <= 0.0:
            raise ValueError(f"Branch weights must be non-negative with positive sum, got w_a={w_a}, w_b={w_b}")

        # Resolve ONNX path before loading heavy InsightFace packs so missing weights fail fast.
        resolved_onnx = resolve_onnx_path(onnx_path)
        if use_gpu:
            ensure_cuda_dll_paths()

        self.use_tta = bool(use_tta)
        self.w_a = w_a
        self.w_b = w_b
        self._sqrt_wa = float(np.sqrt(w_a))
        self._sqrt_wb = float(np.sqrt(w_b))

        self.antelope = FaceEngineAntelopeV2(
            insightface_home=insightface_home,
            use_gpu=use_gpu,
            det_size=det_size,
            use_tta=self.use_tta,
        )
        self.app = self.antelope.app
        self.adaface = FaceEngineAdaFaceFT(
            onnx_path=resolved_onnx,
            insightface_home=insightface_home,
            use_gpu=use_gpu,
            det_size=det_size,
            use_tta=self.use_tta,
            det_app=self.antelope.app,
        )
        self.use_gpu = bool(self.antelope.use_gpu or self.adaface.use_gpu)

    @staticmethod
    def _read(src):
        return FaceEngineAntelopeV2._read(src)

    def _faces(self, img):
        return self.antelope._faces(img)

    @staticmethod
    def _l2_normalize(vec):
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm > 1e-12:
            v = v / norm
        return v.astype(np.float32)

    def _embed_branch_a(self, f, aimg, aimg_flip):
        if not self.use_tta or self.antelope.rec_model is None:
            return self._l2_normalize(f.normed_embedding)
        emb1 = np.asarray(self.antelope.rec_model.get_feat(aimg), dtype=np.float32).reshape(-1)
        emb2 = np.asarray(self.antelope.rec_model.get_feat(aimg_flip), dtype=np.float32).reshape(-1)
        return self._l2_normalize(emb1 + emb2)

    def _embed_branch_b(self, aimg, aimg_flip):
        emb1 = self.adaface._embed_chip(aimg)
        if not self.use_tta:
            return self._l2_normalize(emb1)
        emb2 = self.adaface._embed_chip(aimg_flip)
        return self._l2_normalize(emb1 + emb2)

    def _pack(self, f, pad, img_for_crop):
        aimg = face_align.norm_crop(img_for_crop, landmark=f.kps, image_size=112)
        aimg_flip = cv2.flip(aimg, 1) if self.use_tta else None

        emb_a = self._embed_branch_a(f, aimg, aimg_flip)
        emb_b = self._embed_branch_b(aimg, aimg_flip)

        fused = np.concatenate([self._sqrt_wa * emb_a, self._sqrt_wb * emb_b]).astype(np.float32)
        embedding = self._l2_normalize(fused)

        bbox = [float(v) - pad for v in f.bbox]
        kps = (np.asarray(f.kps, dtype=np.float32) - pad).tolist()
        quality = compute_face_quality(aimg, kps, bbox)

        return {
            "embedding": embedding,
            "bbox": bbox,
            "kps": kps,
            "det_score": float(f.det_score),
            "quality": quality,
            "dim": self.embedding_dim,
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
        va = np.asarray(a, dtype=np.float32).reshape(-1)
        vb = np.asarray(b, dtype=np.float32).reshape(-1)
        return float(np.dot(va, vb))


def _normalize_rows(mat):
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def compute_asnorm_scores(
    probe_vec,
    candidate_vecs,
    cohort_matrix,
    candidate_raw_sims=None,
    top_k=50,
    std_floor=0.05,
    candidate_ids=None,
    cohort_ids=None,
    normalized_inputs=False,
):
    """
    Computes symmetric Adaptive S-Norm (AS-Norm) z-scores in pure NumPy.

    Args:
        probe_vec: 1D query embedding vector of shape (D,).
        candidate_vecs: 2D array of candidate embeddings of shape (N, D).
        cohort_matrix: 2D imposter cohort matrix of shape (M, D).
        candidate_raw_sims: Optional precomputed cosine similarities of shape (N,).
        top_k: Number of highest-similarity cohort members used per adaptive stats.
        std_floor: Minimum standard deviation floor for numerical stability.
        candidate_ids: Optional 1D sequence of length N identifying candidates.
            When provided with cohort_ids, any cohort row matching candidate_ids[i]
            is excluded from both probe and candidate adaptive stats for row i.
        cohort_ids: Optional 1D sequence of length M identifying cohort rows.
        normalized_inputs: If True, assumes probe_vec, candidate_vecs, and
            cohort_matrix are already L2-normalized float32 arrays.

    Returns:
        1D float32 array of raw AS-Norm z-scores of shape (N,).
    """
    if (candidate_ids is None) != (cohort_ids is None):
        raise ValueError("candidate_ids and cohort_ids must both be provided or both be None")

    p = np.asarray(probe_vec, dtype=np.float32).reshape(-1)
    if not normalized_inputs:
        p_norm = float(np.linalg.norm(p))
        if p_norm > 1e-12:
            p = p / p_norm

    cand = np.asarray(candidate_vecs, dtype=np.float32)
    if cand.size == 0:
        return np.empty((0,), dtype=np.float32)
    if cand.ndim == 1:
        cand = cand.reshape(1, -1)
    if cand.shape[1] != p.shape[0]:
        raise ValueError(f"Dimension mismatch: probe_vec has {p.shape[0]}, candidate_vecs has {cand.shape[1]}")
    if not normalized_inputs:
        cand = _normalize_rows(cand)

    if candidate_raw_sims is None:
        raw_sims = (cand @ p).astype(np.float32)
    else:
        raw_sims = np.asarray(candidate_raw_sims, dtype=np.float32).reshape(-1)
        if raw_sims.shape[0] != cand.shape[0]:
            raise ValueError(
                f"candidate_raw_sims length ({raw_sims.shape[0]}) != candidate_vecs count ({cand.shape[0]})"
            )

    cohort = np.asarray(cohort_matrix, dtype=np.float32)
    if cohort.size == 0 or int(top_k) <= 0:
        return raw_sims.astype(np.float32)
    if cohort.ndim == 1:
        cohort = cohort.reshape(1, -1)
    if cohort.shape[1] != p.shape[0]:
        raise ValueError(f"Dimension mismatch: probe_vec has {p.shape[0]}, cohort_matrix has {cohort.shape[1]}")
    if not normalized_inputs:
        cohort = _normalize_rows(cohort)

    n = cand.shape[0]
    m = cohort.shape[0]
    k_req = int(top_k)
    k_eff = min(k_req, m)
    floor = float(max(std_floor, 1e-6))

    probe_cohort = cohort @ p
    cand_cohort = cand @ cohort.T

    if candidate_ids is not None:
        cand_ids = np.asarray(candidate_ids).reshape(-1)
        coh_ids = np.asarray(cohort_ids).reshape(-1)
        if cand_ids.shape[0] != n:
            raise ValueError(f"candidate_ids length ({cand_ids.shape[0]}) != candidate_vecs count ({n})")
        if coh_ids.shape[0] != m:
            raise ValueError(f"cohort_ids length ({coh_ids.shape[0]}) != cohort_matrix count ({m})")
        exclude = cand_ids[:, None] == coh_ids[None, :]
        row_has_excl = np.any(exclude, axis=1)
    else:
        exclude = None
        row_has_excl = None

    if row_has_excl is None or not np.any(row_has_excl):
        top_p = np.partition(probe_cohort, -k_eff)[-k_eff:] if k_eff < m else probe_cohort
        mu_p = float(np.mean(top_p))
        sigma_p = max(float(np.std(top_p)), floor)

        top_c = np.partition(cand_cohort, -k_eff, axis=1)[:, -k_eff:] if k_eff < m else cand_cohort
        mu_c = np.mean(top_c, axis=1)
        sigma_c = np.maximum(np.std(top_c, axis=1), floor)

        z_p = (raw_sims - mu_p) / sigma_p
        z_c = (raw_sims - mu_c) / sigma_c
        return (0.5 * (z_p + z_c)).astype(np.float32)

    valid_counts = m - np.sum(exclude, axis=1)
    empty_rows = valid_counts == 0

    top_p_all = np.partition(probe_cohort, -k_eff)[-k_eff:] if k_eff < m else probe_cohort
    mu_p = np.full(n, float(np.mean(top_p_all)), dtype=np.float64)
    sigma_p = np.full(n, max(float(np.std(top_p_all)), floor), dtype=np.float64)

    for i in np.flatnonzero(row_has_excl & ~empty_rows):
        valid_p = probe_cohort[~exclude[i]]
        m_i = valid_p.size
        k_i = min(k_req, m_i)
        top_p_i = np.partition(valid_p, -k_i)[-k_i:] if k_i < m_i else valid_p
        mu_p[i] = float(np.mean(top_p_i))
        sigma_p[i] = max(float(np.std(top_p_i)), floor)

    min_valid_nonempty = int(np.min(valid_counts[~empty_rows])) if np.any(~empty_rows) else 0
    if min_valid_nonempty >= k_req and k_req < m:
        # Fast vectorized path: mask excluded cohort entries with -inf so they never enter top-k_req.
        cand_cohort[exclude] = -np.inf
        top_c = np.partition(cand_cohort, -k_req, axis=1)[:, -k_req:]
        if np.any(empty_rows):
            top_c[empty_rows] = 0.0
        mu_c = np.mean(top_c, axis=1)
        sigma_c = np.maximum(np.std(top_c, axis=1), floor)
    else:
        mu_c = np.zeros(n, dtype=np.float64)
        sigma_c = np.full(n, floor, dtype=np.float64)
        no_excl_mask = ~row_has_excl
        if np.any(no_excl_mask):
            sub_c = cand_cohort[no_excl_mask]
            top_c_no = np.partition(sub_c, -k_eff, axis=1)[:, -k_eff:] if k_eff < m else sub_c
            mu_c[no_excl_mask] = np.mean(top_c_no, axis=1)
            sigma_c[no_excl_mask] = np.maximum(np.std(top_c_no, axis=1), floor)
        for i in np.flatnonzero(row_has_excl & ~empty_rows):
            valid_c = cand_cohort[i, ~exclude[i]]
            m_i = valid_c.size
            k_i = min(k_req, m_i)
            top_c_i = np.partition(valid_c, -k_i)[-k_i:] if k_i < m_i else valid_c
            mu_c[i] = float(np.mean(top_c_i))
            sigma_c[i] = max(float(np.std(top_c_i)), floor)

    z_p = (raw_sims - mu_p) / sigma_p
    z_c = (raw_sims - mu_c) / sigma_c
    out = (0.5 * (z_p + z_c)).astype(np.float32)
    if np.any(empty_rows):
        out[empty_rows] = raw_sims[empty_rows]
    return out


def calibrate_asnorm_to_cosine(
    asnorm_scores,
    raw_sims,
    z_center=0.0,
    z_scale=8.0,
    asnorm_weight=0.5,
):
    """
    Smoothly maps AS-Norm z-scores and raw cosine similarities into [-1.0, 1.0]
    while preserving monotonicity and FaceStore tier thresholds (STRONG >= 0.36,
    POSSIBLE >= 0.26).
    """
    z = np.asarray(asnorm_scores, dtype=np.float32).reshape(-1)
    raw = np.asarray(raw_sims, dtype=np.float32).reshape(-1)
    if z.shape[0] != raw.shape[0]:
        raise ValueError(f"Shape mismatch: asnorm_scores ({z.shape[0]}) != raw_sims ({raw.shape[0]})")
    if z.size == 0:
        return np.empty((0,), dtype=np.float32)

    scale = max(float(z_scale), 1e-6)
    w = float(np.clip(asnorm_weight, 0.0, 1.0))

    raw_clipped = np.clip(raw, -1.0, 1.0)
    z_mapped = np.tanh((z - float(z_center)) / scale)
    calibrated = (1.0 - w) * raw_clipped + w * z_mapped
    return np.clip(calibrated, -1.0, 1.0).astype(np.float32)
