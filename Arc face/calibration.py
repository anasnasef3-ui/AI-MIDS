"""
calibration.py - 4-Layer 1:N Score Normalization, QMF Calibration, Open-Set Conformal/EVT
Tiering, COVE 1:1 Verification, and Background Triage Advisories for AI-MIDS.

This module is strictly opt-in: legacy FaceStore.search(..., calibrated=False) and
AIMIDS_CALIBRATION=legacy execute their original code paths unchanged.

Layers:
  1. ExternalCohortASNorm: fixed external balanced cohort (M, D) with precomputed
     enrollment stats (mu_e, sigma_e) -> O(MD + N) symmetric AS-Norm from N=1 upward.
  2. QMFCalibrator: prior-weighted logistic regression in LLR space combining
     [asnorm_z, raw_sim, q_min, age_gap_scaled, has_age_gap], plus Bayesian posterior
     match_prob, monotonic cosine-range mapping, and calibration diagnostics (C_llr,
     PAV C_llr_min, ECE, Brier).
  3. OpenSetCalibrator: Split Conformal non-mated max-of-N curve + Generalized Pareto
     Distribution (GPD POT) tail model with N_eff(N) scaling for N-adaptive tiers
     ("STRONG", "possible", "weak") and per-hit false-alarm p-values p_nonmate(s; N).
  4. CoveCalibrator & compute_bg_zscore: identity-level 1:1 Split Conformal verifier
     (COVE, arXiv:2610.35277) and trimmed s_{3..50} background rank z-score advisories.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import genpareto


SCHEMA_VERSION = 2
DEFAULT_TOP_K = 200
DEFAULT_STD_FLOOR = 0.05
DEFAULT_PRIOR_PI = 0.5
MIN_QUALITY = 0.25
MAX_QUALITY = 1.0


def _l2_normalize_vec(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.all(np.isfinite(arr)) or norm < 1e-12:
        raise ValueError("invalid embedding (zero-length or non-finite vector)")
    return (arr / norm).astype(np.float32)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return arr.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if not np.all(np.isfinite(arr)) or np.any(norms < 1e-12):
        raise ValueError("invalid embedding matrix (zero-length or non-finite rows)")
    if float(np.max(np.abs(norms - 1.0))) < 1e-6:
        return arr.astype(np.float32)
    return (arr / norms).astype(np.float32)


def compute_cohort_hash(cohort_vecs: np.ndarray) -> str:
    """Deterministic short SHA-256 tag identifying a fixed external cohort matrix."""
    arr = np.ascontiguousarray(np.asarray(cohort_vecs, dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def compute_proj_hash(proj_basis: Optional[np.ndarray]) -> str:
    """Deterministic short SHA-256 tag identifying an orthonormal age-subspace projection basis."""
    if proj_basis is None:
        return "none"
    arr = np.asarray(proj_basis, dtype=np.float32)
    if arr.size == 0:
        return "none"
    normed = np.ascontiguousarray(_l2_normalize_rows(arr))
    return hashlib.sha256(normed.tobytes()).hexdigest()[:16]


def compute_age_subspace_basis(
    vecs: np.ndarray,
    identities: Sequence[str],
    is_young: Sequence[bool],
    n_components: int = 5,
) -> np.ndarray:
    """
    Computes top-k orthonormal right singular vectors (k, dim) float32 of
    mean(young_i) - mean(adult_i) across paired identities (e.g. external LAG).
    """
    mat = _l2_normalize_rows(np.asarray(vecs, dtype=np.float32))
    if mat.ndim != 2 or mat.shape[0] < 1 or mat.shape[1] < 1:
        raise ValueError(f"vecs must be a non-empty 2D matrix, got shape {mat.shape}")
    if len(identities) != mat.shape[0] or len(is_young) != mat.shape[0]:
        raise ValueError("vecs, identities, and is_young must have identical lengths")

    k_req = int(n_components)
    if k_req <= 0:
        return np.empty((0, mat.shape[1]), dtype=np.float32)

    by_id: Dict[str, Dict[bool, list]] = {}
    for idx, (ident, young_flag) in enumerate(zip(identities, is_young)):
        key = str(ident)
        bucket = by_id.setdefault(key, {True: [], False: []})
        bucket[bool(young_flag)].append(idx)

    diffs = []
    for key in sorted(by_id.keys()):
        y_idxs = by_id[key][True]
        a_idxs = by_id[key][False]
        if y_idxs and a_idxs:
            m_y = np.mean(mat[y_idxs], axis=0, dtype=np.float64)
            m_a = np.mean(mat[a_idxs], axis=0, dtype=np.float64)
            diffs.append(m_y - m_a)

    if not diffs:
        raise ValueError("compute_age_subspace_basis requires at least one paired (young, adult) identity")

    diff_mat = np.stack(diffs).astype(np.float64)
    _, _, vt = np.linalg.svd(diff_mat, full_matrices=False)
    k_eff = min(k_req, vt.shape[0])
    return np.ascontiguousarray(_l2_normalize_rows(vt[:k_eff].astype(np.float32)))


def project_subspace(vecs: np.ndarray, proj_basis: Optional[np.ndarray]) -> np.ndarray:
    """Projects out rows of proj_basis (X - (X @ B.T) @ B) and L2-renormalizes."""
    arr = np.asarray(vecs, dtype=np.float32)
    is_1d = arr.ndim == 1
    if is_1d:
        normed = _l2_normalize_vec(arr).reshape(1, -1)
    else:
        if arr.shape[0] == 0:
            return arr.astype(np.float32)
        normed = _l2_normalize_rows(arr)

    if proj_basis is not None:
        basis = np.asarray(proj_basis, dtype=np.float32)
        if basis.ndim == 1:
            basis = basis.reshape(1, -1)
        if basis.size > 0:
            if basis.shape[1] != normed.shape[1]:
                raise ValueError(
                    f"Projection basis dim ({basis.shape[1]}) != embedding dim ({normed.shape[1]})"
                )
            proj = normed - (normed @ basis.T) @ basis
            normed = _l2_normalize_rows(proj)

    return normed[0] if is_1d else normed


def extract_face_quality(engine: Any, img: Optional[np.ndarray], face_dict: Optional[Dict[str, Any]]) -> float:
    """
    Computes bounded per-face quality in [0.25, 1.0] without altering engine._pack().

    If face_dict already carries a finite 'quality' scalar (e.g., FaceEngineCrossAge),
    returns it clamped to [0.25, 1.0]. Otherwise aligns a 112x112 BGR crop from img
    using face_dict['kps'] and evaluates face_engine_custom.compute_face_quality.
    """
    if not face_dict:
        return 1.0
    existing = face_dict.get("quality")
    if existing is not None:
        try:
            q_val = float(existing)
            if np.isfinite(q_val):
                return float(min(max(q_val, MIN_QUALITY), MAX_QUALITY))
        except (TypeError, ValueError):
            pass

    if img is None or "kps" not in face_dict or "bbox" not in face_dict:
        return 1.0

    try:
        from insightface.utils import face_align

        try:
            from face_engine_custom import compute_face_quality
        except ImportError:
            from .face_engine_custom import compute_face_quality

        kps = np.asarray(face_dict["kps"], dtype=np.float32)
        aimg = face_align.norm_crop(img, landmark=kps, image_size=112)
        q_val = float(compute_face_quality(aimg, kps, face_dict["bbox"]))
        if not np.isfinite(q_val):
            return 1.0
        return float(min(max(q_val, MIN_QUALITY), MAX_QUALITY))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Layer 1: Fixed External Balanced AS-Norm
# ---------------------------------------------------------------------------


class ExternalCohortASNorm:
    """
    Fixed external cohort Adaptive S-Norm (AS-Norm) with precomputed enrollment stats.

    Given an external balanced cohort C of shape (M, D) disjoint from gallery and probe
    identities, enrollment stats (mu_e, sigma_e) are computed once per photo/centroid
    and probe stats (mu_p, sigma_p) are computed once per search query in O(MD), making
    all-candidate AS-Norm scoring O(N) with zero N >= 80 discontinuity and zero rank-200 cutoff.
    """

    def __init__(
        self,
        cohort_vecs: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        std_floor: float = DEFAULT_STD_FLOOR,
        cohort_hash: Optional[str] = None,
    ):
        mat = _l2_normalize_rows(np.asarray(cohort_vecs, dtype=np.float32))
        if mat.ndim != 2 or mat.shape[0] < 1 or mat.shape[1] < 1:
            raise ValueError(f"cohort_vecs must be a non-empty 2D matrix, got shape {mat.shape}")
        self.cohort_vecs = mat
        self.m, self.dim = int(mat.shape[0]), int(mat.shape[1])
        self.requested_top_k = int(top_k)
        self.top_k = max(1, min(int(top_k), self.m))
        self.std_floor = float(max(std_floor, 1e-6))
        self.cohort_hash = str(cohort_hash) if cohort_hash else compute_cohort_hash(self.cohort_vecs)

    def enroll_stats(self, vecs: np.ndarray, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes (mu_e, sigma_e) for enrollment vectors in bounded memory batches.

        Returns two 1D float32 arrays of shape (N,).
        """
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {arr.shape[1]}")

        normed = _l2_normalize_rows(arr)
        n = normed.shape[0]
        bs = max(1, int(batch_size))
        mu_out = np.empty(n, dtype=np.float32)
        sigma_out = np.empty(n, dtype=np.float32)
        k_eff = self.top_k

        for start in range(0, n, bs):
            end = min(start + bs, n)
            sims = normed[start:end] @ self.cohort_vecs.T
            if k_eff < self.m:
                top = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
            else:
                top = sims
            mu_out[start:end] = np.mean(top, axis=1, dtype=np.float64).astype(np.float32)
            std_b = np.std(top, axis=1, dtype=np.float64)
            sigma_out[start:end] = np.maximum(std_b, self.std_floor).astype(np.float32)

        return mu_out, sigma_out

    def probe_stats(self, q: np.ndarray) -> Tuple[float, float]:
        """Computes (mu_p, sigma_p) for a single query embedding in O(MD)."""
        q_norm = _l2_normalize_vec(q)
        if q_norm.shape[0] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {q_norm.shape[0]}")
        sims = self.cohort_vecs @ q_norm
        top = np.partition(sims, -self.top_k)[-self.top_k:] if self.top_k < self.m else sims
        mu_p = float(np.mean(top, dtype=np.float64))
        sigma_p = max(float(np.std(top, dtype=np.float64)), self.std_floor)
        return mu_p, sigma_p

    def score_batch(
        self,
        q: np.ndarray,
        cand_vecs: Optional[np.ndarray] = None,
        cand_mu: Optional[np.ndarray] = None,
        cand_sigma: Optional[np.ndarray] = None,
        raw_sims: Optional[np.ndarray] = None,
        probe_stats: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Computes symmetric AS-Norm z-scores for N candidates against query q.

        When cand_mu, cand_sigma, raw_sims, and probe_stats are supplied, runs in O(N)
        with zero cohort dot products. Equivalent to brute-force compute_asnorm_scores
        against the fixed external cohort.
        """
        if raw_sims is None:
            if cand_vecs is None:
                raise ValueError("Either raw_sims or cand_vecs must be provided")
            q_norm = _l2_normalize_vec(q)
            c_norm = _l2_normalize_rows(cand_vecs)
            if c_norm.shape[0] == 0:
                return np.empty((0,), dtype=np.float32)
            if c_norm.shape[1] != self.dim or q_norm.shape[0] != self.dim:
                raise ValueError(f"Dimension mismatch with cohort dim {self.dim}")
            raw_arr = (c_norm @ q_norm).astype(np.float64)
        else:
            raw_arr = np.asarray(raw_sims, dtype=np.float64).reshape(-1)
            if raw_arr.size == 0:
                return np.empty((0,), dtype=np.float32)

        if cand_mu is None or cand_sigma is None:
            if cand_vecs is None:
                raise ValueError("Either (cand_mu, cand_sigma) or cand_vecs must be provided")
            mu_e, sigma_e = self.enroll_stats(cand_vecs)
        else:
            mu_e = np.asarray(cand_mu, dtype=np.float64).reshape(-1)
            sigma_e = np.asarray(cand_sigma, dtype=np.float64).reshape(-1)

        if mu_e.shape[0] != raw_arr.shape[0] or sigma_e.shape[0] != raw_arr.shape[0]:
            raise ValueError(
                f"Candidate count mismatch: raw_sims={raw_arr.shape[0]}, "
                f"cand_mu={mu_e.shape[0]}, cand_sigma={sigma_e.shape[0]}"
            )

        if probe_stats is None:
            mu_p, sigma_p = self.probe_stats(q)
        else:
            mu_p, sigma_p = float(probe_stats[0]), max(float(probe_stats[1]), self.std_floor)

        z_p = (raw_arr - mu_p) / sigma_p
        z_c = (raw_arr - mu_e.astype(np.float64)) / np.maximum(sigma_e.astype(np.float64), self.std_floor)
        return (0.5 * (z_p + z_c)).astype(np.float32)


# ---------------------------------------------------------------------------
# Layer 2: QMF Logistic LLR Calibration & Diagnostics
# ---------------------------------------------------------------------------


def compute_cllr(llrs: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes empirical cross-entropy / cost of log-likelihood ratio (C_llr, bits).

    Reference: Brummer & du Preez (Computer Speech & Language, 2006).
    An uninformative system (LLR == 0) scores 1.0 bit; well-calibrated systems << 1.0.
    """
    s = np.asarray(llrs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int32).reshape(-1)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("compute_cllr requires both positive (y=1) and negative (y=0) trials")

    ln2 = float(np.log(2.0))
    # Numerically overflow-safe softplus(-pos) and softplus(+neg)
    loss_pos = float(np.mean(np.logaddexp(0.0, -pos))) / ln2
    loss_neg = float(np.mean(np.logaddexp(0.0, neg))) / ln2
    return 0.5 * (loss_pos + loss_neg)


def pav_cllr_min(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Non-parametric minimum achievable C_llr (C_llr_min) via Pool-Adjacent-Violators (PAV).

    Finds the optimal monotonic posterior mapping on the evaluation scores without
    parametric assumptions, providing the calibration ceiling (C_llr - C_llr_min = C_llr_cal).
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    n_tot = s.size
    n_pos = float(np.sum(y == 1.0))
    n_neg = float(np.sum(y == 0.0))
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError("pav_cllr_min requires both positive and negative trials")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    # Pre-group exact ties into single blocks before running PAV
    blocks = []  # list of [sum_y, count]
    i = 0
    while i < n_tot:
        j = i + 1
        while j < n_tot and s_sorted[j] == s_sorted[i]:
            j += 1
        blocks.append([float(np.sum(y_sorted[i:j])), float(j - i)])
        i = j

    # Pool adjacent violators (must be monotonically non-decreasing in positive rate)
    merged = []
    for blk in blocks:
        merged.append(blk)
        while len(merged) >= 2:
            s_prev, c_prev = merged[-2]
            s_curr, c_curr = merged[-1]
            if (s_prev / c_prev) >= (s_curr / c_curr):
                merged[-2][0] = s_prev + s_curr
                merged[-2][1] = c_prev + c_curr
                merged.pop()
            else:
                break

    prior_odds_log = float(np.log(n_pos / n_neg))
    ln2 = float(np.log(2.0))
    sum_pos_loss = 0.0
    sum_neg_loss = 0.0

    for s_b, c_b in merged:
        p_hat = s_b / c_b
        n_b_pos = s_b
        n_b_neg = c_b - s_b
        if p_hat <= 0.0:
            # Pure negative block -> LLR = -inf -> 0 loss on negatives
            continue
        if p_hat >= 1.0:
            # Pure positive block -> LLR = +inf -> 0 loss on positives
            continue
        llr_b = float(np.log(p_hat / (1.0 - p_hat))) - prior_odds_log
        if n_b_pos > 0:
            sum_pos_loss += n_b_pos * float(np.logaddexp(0.0, -llr_b)) / ln2
        if n_b_neg > 0:
            sum_neg_loss += n_b_neg * float(np.logaddexp(0.0, llr_b)) / ln2

    return 0.5 * ((sum_pos_loss / n_pos) + (sum_neg_loss / n_neg))


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score (mean squared error between predicted probabilities and {0, 1} labels)."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    return float(np.mean((p - y) ** 2))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) over equal-width probability bins."""
    p = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = float(p.size)
    for b in range(int(n_bins)):
        lo, hi = edges[b], edges[b + 1]
        if b == 0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p > lo) & (p <= hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            acc = float(np.mean(y[mask]))
            conf = float(np.mean(p[mask]))
            ece += (cnt / n) * abs(acc - conf)
    return float(ece)


class QMFCalibrator:
    """
    Quality Measure Function (QMF) logistic calibrator in log-likelihood-ratio (LLR) space.

    Feature order:
        [asnorm_z, raw_sim, q_min, age_gap_scaled, has_age_gap]

    When age_gap is None or NaN, age_gap_scaled = 0.0 and has_age_gap = 0.0 so the
    calibrator cleanly marginalizes over missing age metadata.
    """

    FEATURE_NAMES = ("asnorm_z", "raw_sim", "q_min", "age_gap_scaled", "has_age_gap")

    def __init__(
        self,
        weights: Sequence[float],
        bias: float,
        age_center: float = 20.0,
        age_scale: float = 20.0,
        cosine_shift: float = 1.5,
        cosine_scale: float = 3.5,
        default_prior: float = DEFAULT_PRIOR_PI,
    ):
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != 5:
            raise ValueError(f"QMFCalibrator expects 5 weights, got {w.shape[0]}")
        self.weights = w
        self.bias = float(bias)
        self.age_center = float(age_center)
        self.age_scale = max(float(age_scale), 1e-6)
        self.cosine_shift = float(cosine_shift)
        self.cosine_scale = max(float(cosine_scale), 1e-6)
        self.default_prior = float(np.clip(default_prior, 1e-6, 1.0 - 1e-6))

    def build_features(
        self,
        asnorm_z: Union[float, np.ndarray],
        raw_sim: Union[float, np.ndarray],
        q_min: Union[float, np.ndarray] = 1.0,
        age_gap: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Builds (N, 5) float64 feature matrix and returns (X, scalar_input_flag)."""
        is_scalar = np.ndim(asnorm_z) == 0 and np.ndim(raw_sim) == 0 and np.ndim(q_min) == 0 and (
            age_gap is None or np.ndim(age_gap) == 0
        )
        z = np.asarray(asnorm_z, dtype=np.float64).reshape(-1)
        r = np.asarray(raw_sim, dtype=np.float64).reshape(-1)
        if r.size == 1 and z.size > 1:
            r = np.full_like(z, float(r[0]))
        elif z.size == 1 and r.size > 1:
            z = np.full_like(r, float(z[0]))
        if z.shape[0] != r.shape[0]:
            raise ValueError(f"Shape mismatch: asnorm_z={z.shape[0]}, raw_sim={r.shape[0]}")

        n = z.shape[0]
        q_arr = np.asarray(q_min, dtype=np.float64).reshape(-1)
        if q_arr.size == 1:
            q_arr = np.full(n, float(q_arr[0]), dtype=np.float64)
        elif q_arr.shape[0] != n:
            raise ValueError(f"Shape mismatch: q_min={q_arr.shape[0]} vs N={n}")
        q_arr = np.where(np.isfinite(q_arr), np.clip(q_arr, MIN_QUALITY, MAX_QUALITY), 1.0)

        if age_gap is None:
            gap_scaled = np.zeros(n, dtype=np.float64)
            has_gap = np.zeros(n, dtype=np.float64)
        else:
            g_raw = np.asarray(age_gap, dtype=np.float64).reshape(-1)
            if g_raw.size == 1:
                g_raw = np.full(n, float(g_raw[0]), dtype=np.float64)
            elif g_raw.shape[0] != n:
                raise ValueError(f"Shape mismatch: age_gap={g_raw.shape[0]} vs N={n}")
            valid = np.isfinite(g_raw) & (g_raw >= 0.0)
            has_gap = valid.astype(np.float64)
            gap_scaled = np.where(valid, (g_raw - self.age_center) / self.age_scale, 0.0)

        X = np.column_stack([z, r, q_arr, gap_scaled, has_gap])
        return X, is_scalar

    def predict_llr(
        self,
        asnorm_z: Union[float, np.ndarray],
        raw_sim: Union[float, np.ndarray],
        q_min: Union[float, np.ndarray] = 1.0,
        age_gap: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Predicts calibrated log-likelihood ratio (LLR)."""
        X, is_scalar = self.build_features(asnorm_z, raw_sim, q_min=q_min, age_gap=age_gap)
        llr = X @ self.weights + self.bias
        return float(llr[0]) if is_scalar else llr

    def predict_prob(
        self,
        llr: Union[float, np.ndarray],
        pi: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Converts LLR to Bayesian posterior match probability at prior pi."""
        p = self.default_prior if pi is None else float(np.clip(pi, 1e-9, 1.0 - 1e-9))
        log_prior_odds = float(logit(p))
        arr = np.asarray(llr, dtype=np.float64)
        probs = expit(arr + log_prior_odds)
        return float(probs) if np.ndim(llr) == 0 else probs

    def llr_to_cosine(self, llr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Monotonic mapping from unbounded LLR into cosine-comparable range (0, 1) subset [-1, 1].
        """
        arr = np.asarray(llr, dtype=np.float64)
        scores = np.clip(expit((arr - self.cosine_shift) / self.cosine_scale), 0.0, 1.0)
        return float(scores) if np.ndim(llr) == 0 else scores

    @classmethod
    def fit(
        cls,
        asnorm_z: np.ndarray,
        raw_sim: np.ndarray,
        labels: np.ndarray,
        q_min: Union[float, np.ndarray] = 1.0,
        age_gap: Optional[Union[float, np.ndarray]] = None,
        l2_reg: float = 1e-2,
        age_center: Optional[float] = None,
        age_scale: float = 20.0,
        positive_indices: Tuple[int, ...] = (0, 1),
        default_prior: float = DEFAULT_PRIOR_PI,
    ) -> "QMFCalibrator":
        """
        Fits prior-weighted (BOSARIS pi_eff=0.5) logistic regression in LLR space
        via bounded L-BFGS-B so similarity coefficients stay non-negative.
        """
        y = np.asarray(labels, dtype=np.int32).reshape(-1)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("QMFCalibrator.fit requires both positive and negative trials")

        if age_center is None:
            if age_gap is not None:
                g_arr = np.asarray(age_gap, dtype=np.float64).reshape(-1)
                valid_g = g_arr[np.isfinite(g_arr) & (g_arr >= 0.0)]
                age_center = float(np.mean(valid_g)) if valid_g.size > 0 else 20.0
            else:
                age_center = 20.0

        tmp = cls(
            weights=np.zeros(5, dtype=np.float64),
            bias=0.0,
            age_center=age_center,
            age_scale=age_scale,
            default_prior=default_prior,
        )
        X, _ = tmp.build_features(asnorm_z, raw_sim, q_min=q_min, age_gap=age_gap)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Feature count ({X.shape[0]}) != label count ({y.shape[0]})")

        # Prior-weighted sample weights (equal 0.5 mass on targets and non-targets)
        sample_w = np.where(y == 1, 0.5 / float(n_pos), 0.5 / float(n_neg))
        reg = max(float(l2_reg), 0.0)

        def _obj(theta: np.ndarray) -> Tuple[float, np.ndarray]:
            w = theta[:-1]
            b = theta[-1]
            logits = X @ w + b
            # Binary cross-entropy weighted by sample_w
            # loss_i = logaddexp(0, logits) - y * logits
            loss_i = np.logaddexp(0.0, logits) - y * logits
            loss = float(np.sum(sample_w * loss_i)) + 0.5 * reg * float(np.sum(w * w))

            probs = expit(logits)
            diff = sample_w * (probs - y)
            grad_w = (X.T @ diff) + reg * w
            grad_b = float(np.sum(diff))
            grad = np.concatenate([grad_w, [grad_b]])
            return loss, grad

        x0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, -2.0], dtype=np.float64)
        bounds = []
        for idx in range(5):
            if idx == 0:
                bounds.append((1e-3, None))
            elif idx in positive_indices:
                bounds.append((0.0, None))
            else:
                bounds.append((None, None))
        bounds.append((None, None))  # intercept

        res = minimize(_obj, x0, jac=True, method="L-BFGS-B", bounds=bounds)
        w_opt = res.x[:-1]
        b_opt = float(res.x[-1])

        # Calibrate cosine-range display mapping smoothly around imp/gen midpoints
        llrs = X @ w_opt + b_opt
        imp_med = float(np.median(llrs[y == 0]))
        gen_med = float(np.median(llrs[y == 1]))
        # Place 0.50 cosine-display around strong/possible boundary (~60% from imp_med toward gen_med)
        span = max(gen_med - imp_med, 4.0)
        cos_shift = float(imp_med + 0.60 * span)
        cos_scale = float(max(span / 4.2, 1.8))

        return cls(
            weights=w_opt,
            bias=b_opt,
            age_center=age_center,
            age_scale=age_scale,
            cosine_shift=cos_shift,
            cosine_scale=cos_scale,
            default_prior=default_prior,
        )

    def evaluate(
        self,
        asnorm_z: np.ndarray,
        raw_sim: np.ndarray,
        labels: np.ndarray,
        q_min: Union[float, np.ndarray] = 1.0,
        age_gap: Optional[Union[float, np.ndarray]] = None,
        pi: float = DEFAULT_PRIOR_PI,
    ) -> Dict[str, float]:
        """Computes C_llr, PAV C_llr_min, C_llr_cal, Brier, and ECE on a trial set."""
        y = np.asarray(labels, dtype=np.int32).reshape(-1)
        llrs = self.predict_llr(asnorm_z, raw_sim, q_min=q_min, age_gap=age_gap)
        probs = self.predict_prob(llrs, pi=pi)
        cllr = compute_cllr(llrs, y)
        cllr_min = pav_cllr_min(llrs, y)
        return {
            "cllr": float(cllr),
            "cllr_min": float(cllr_min),
            "cllr_cal": float(max(0.0, cllr - cllr_min)),
            "brier": compute_brier(probs, y),
            "ece": compute_ece(probs, y),
        }


def fit_qmf(
    asnorm_z: np.ndarray,
    raw_sim: np.ndarray,
    labels: np.ndarray,
    q_min: Union[float, np.ndarray] = 1.0,
    age_gap: Optional[Union[float, np.ndarray]] = None,
    l2_reg: float = 1e-2,
    age_center: Optional[float] = None,
    age_scale: float = 20.0,
    positive_indices: Tuple[int, ...] = (0, 1),
    default_prior: float = DEFAULT_PRIOR_PI,
) -> QMFCalibrator:
    """Functional alias for QMFCalibrator.fit."""
    return QMFCalibrator.fit(
        asnorm_z=asnorm_z,
        raw_sim=raw_sim,
        labels=labels,
        q_min=q_min,
        age_gap=age_gap,
        l2_reg=l2_reg,
        age_center=age_center,
        age_scale=age_scale,
        positive_indices=positive_indices,
        default_prior=default_prior,
    )


# ---------------------------------------------------------------------------
# Layer 3: Open-Set N-Scaled Calibration (Split Conformal + GPD POT)
# ---------------------------------------------------------------------------


class OpenSetCalibrator:
    """
    Gallery-size (N) adaptive open-set threshold & p-value calibrator.

    Combines:
      - Empirical Split Conformal non-mated max-of-N distributions on n_grid
      - Generalized Pareto Distribution (GPD Peaks-Over-Threshold) tail fit
        (xi, sigma_u, u, p_u) on base person-level non-mated scores with an
        empirically fitted multiplicity law N_eff(N) = c * N^gamma.

    Supports both finite-sample conformal guarantees within observed N and smooth
    monotonic extrapolation to large galleries (N = 100,000+) and deep tails.
    """

    def __init__(
        self,
        n_grid: np.ndarray,
        max_scores: np.ndarray,
        u: float,
        p_u: float,
        xi: float,
        sigma_u: float,
        n_eff_c: float = 1.0,
        n_eff_gamma: float = 1.0,
        base_scores: Optional[np.ndarray] = None,
    ):
        ng = np.asarray(n_grid, dtype=np.int64).reshape(-1)
        ms = np.asarray(max_scores, dtype=np.float32)
        if ng.size == 0 or ms.ndim != 2 or ms.shape[0] != ng.size or ms.shape[1] < 1:
            raise ValueError("n_grid (K,) and max_scores (K, M) must be non-empty and aligned")
        order = np.argsort(ng)
        self.n_grid = ng[order]
        # Each row sorted ascending and enforced non-decreasing across increasing N
        self.max_scores = np.maximum.accumulate(
            np.sort(ms[order], axis=1).astype(np.float32), axis=0
        )
        self.m_calib = int(self.max_scores.shape[1])

        self.u = float(u)
        self.p_u = float(np.clip(p_u, 1e-6, 1.0))
        self.xi = float(xi)
        self.sigma_u = max(float(sigma_u), 1e-6)
        self.n_eff_c = max(float(n_eff_c), 1e-3)
        self.n_eff_gamma = max(float(n_eff_gamma), 0.1)

        if base_scores is not None and len(base_scores) > 0:
            self.base_scores = np.sort(np.asarray(base_scores, dtype=np.float32).reshape(-1))
        else:
            self.base_scores = self.max_scores[0].copy()

        # Precompute boundary shifts so N < n_lo and N > n_hi bridge Split Conformal & GPD seamlessly
        alphas_ref = [a for a in (0.01, 0.02, 0.05, 0.08) if a >= 1.0 / float(self.m_calib + 1)]
        if not alphas_ref:
            alphas_ref = [1.0 / float(self.m_calib + 1)]
        n_lo = int(self.n_grid[0])
        n_hi = int(self.n_grid[-1])
        self._delta_lo = float(
            min(self._conformal_tau_for_row(self.max_scores[0], a) - self._gpd_threshold(n_lo, a) for a in alphas_ref)
        )
        self._delta_top = float(
            max(self._conformal_tau_for_row(self.max_scores[-1], a) - self._gpd_threshold(n_hi, a) for a in alphas_ref)
        )
        self._row_cache: Dict[int, np.ndarray] = {}
        self._tau_cache: Dict[Tuple[int, float, float], Tuple[float, float]] = {}

    def n_eff(self, n_gallery: int) -> float:
        """Effective multiplicity N_eff(N) accounting for multi-photo & correlation."""
        n = max(1, int(n_gallery))
        if n == 1:
            return 1.0
        return max(1.0, float(self.n_eff_c * (float(n) ** self.n_eff_gamma)))

    def _s1_tail(self, score: float) -> float:
        """Base single-person non-mated survival P(Z_1 > score)."""
        z = float(score)
        if z >= self.u:
            y = z - self.u
            if abs(self.xi) < 1e-6:
                surv = self.p_u * float(np.exp(-y / self.sigma_u))
            else:
                inner = max(0.0, 1.0 + self.xi * (y / self.sigma_u))
                if inner <= 0.0:
                    return 0.0
                surv = self.p_u * float(inner ** (-1.0 / self.xi))
            return float(np.clip(surv, 0.0, 1.0))

        # Body (z < u): smooth empirical survival interpolated from base_scores
        bs = self.base_scores
        idx = int(np.searchsorted(bs, z, side="left"))
        emp_surv = float(len(bs) - idx) / float(len(bs))
        return float(np.clip(max(emp_surv, self.p_u), self.p_u, 1.0))

    def _inv_s1(self, p1: float) -> float:
        """Inverse of _s1_tail: score z such that S_1(z) == p1."""
        p = float(np.clip(p1, 1e-12, 1.0 - 1e-12))
        if p <= self.p_u:
            ratio = p / self.p_u
            if abs(self.xi) < 1e-6:
                return float(self.u - self.sigma_u * np.log(ratio))
            return float(self.u + (self.sigma_u / self.xi) * (ratio ** (-self.xi) - 1.0))
        q = float(np.clip(1.0 - p, 0.0, 1.0))
        return float(np.quantile(self.base_scores, q))

    def _gpd_threshold(self, n_gallery: int, alpha: float) -> float:
        a = float(np.clip(alpha, 1e-9, 1.0 - 1e-9))
        neff = self.n_eff(n_gallery)
        # p1 = 1 - (1 - alpha)^(1 / neff) = -expm1(log1p(-alpha) / neff)
        p1 = float(-np.expm1(np.log1p(-a) / neff))
        return self._inv_s1(p1)

    def _gpd_p_nonmate(self, score: float, n_gallery: int) -> float:
        s1 = self._s1_tail(score)
        if s1 <= 0.0:
            return 0.0
        if s1 >= 1.0:
            return 1.0
        neff = self.n_eff(n_gallery)
        return float(np.clip(-np.expm1(neff * np.log1p(-s1)), 0.0, 1.0))

    def _conformal_idx(self, alpha: float) -> Optional[int]:
        m = self.m_calib
        a = float(np.clip(alpha, 1e-9, 1.0 - 1e-9))
        idx = int(np.ceil((1.0 - a) * (m + 1)))
        if idx >= m:
            return None
        return max(0, idx)

    def _conformal_tau_for_row(self, row: np.ndarray, alpha: float) -> float:
        idx = self._conformal_idx(alpha)
        if idx is None:
            return float(row[-1])
        m = int(row.shape[0])
        while idx < m and idx > 0 and float(row[idx]) <= float(row[idx - 1]):
            idx += 1
        if idx >= m:
            return float(row[-1]) + 1e-6
        return float(row[idx])

    def _row_for_n(self, n_gallery: int) -> np.ndarray:
        """Returns sorted ascending conformal max-score vector interpolated in log(N) inside [n_lo, n_hi]."""
        n = int(n_gallery)
        cached = self._row_cache.get(n)
        if cached is not None:
            return cached
        if len(self.n_grid) == 1 or n <= int(self.n_grid[0]):
            res = self.max_scores[0]
        elif n >= int(self.n_grid[-1]):
            res = self.max_scores[-1]
        else:
            log_n = float(np.log(float(n)))
            log_grid = np.log(self.n_grid.astype(np.float64))
            hi = int(np.searchsorted(self.n_grid, n, side="left"))
            if int(self.n_grid[hi]) == n:
                res = self.max_scores[hi]
            else:
                lo = hi - 1
                w = (log_n - float(log_grid[lo])) / max(float(log_grid[hi] - log_grid[lo]), 1e-9)
                res = (1.0 - w) * self.max_scores[lo].astype(np.float64) + w * self.max_scores[hi].astype(np.float64)
        if len(self._row_cache) > 64:
            self._row_cache.clear()
        self._row_cache[n] = res
        return res

    def _conformal_p_for_row(self, row: np.ndarray, score: float, n_gallery: int) -> float:
        m = self.m_calib
        pos = int(np.searchsorted(row, score, side="left"))
        n_ge = m - pos
        p_conf = float(1 + n_ge) / float(m + 1)
        if n_ge >= 2:
            return float(np.clip(p_conf, 0.0, 1.0))

        # Smooth GPD decay beyond second-highest empirical calibration score
        p_gpd = self._gpd_p_nonmate(score, n_gallery)
        if p_gpd <= 0.0:
            return 0.0
        anchor_z = float(row[max(0, m - 2)])
        if score <= anchor_z:
            return float(np.clip(p_conf, 0.0, 1.0))
        anchor_p_conf = 2.0 / float(m + 1)
        anchor_p_gpd = max(self._gpd_p_nonmate(anchor_z, n_gallery), 1e-12)
        tail_val = anchor_p_conf * (p_gpd / anchor_p_gpd)
        return float(np.clip(min(p_conf, tail_val), 0.0, 1.0))

    def tier_thresholds(
        self,
        n_gallery: int,
        alpha_strong: float = 0.01,
        alpha_possible: float = 0.08,
    ) -> Tuple[float, float]:
        """
        Returns (tau_strong, tau_possible) calibrated for gallery size n_gallery.

        Monotonic in both n_gallery and -alpha; smoothly bridges Split Conformal
        empirical curves (for N within n_grid) and GPD POT extrapolation (beyond n_grid).
        """
        n = max(1, int(n_gallery))
        a_s = float(np.clip(alpha_strong, 1e-6, 0.5))
        a_p = float(np.clip(max(alpha_possible, a_s), 1e-6, 0.9))
        cache_key = (n, a_s, a_p)
        cached_tau = self._tau_cache.get(cache_key)
        if cached_tau is not None:
            return cached_tau
        n_lo = int(self.n_grid[0])
        n_hi = int(self.n_grid[-1])

        def _single_tau(alpha: float) -> float:
            idx = self._conformal_idx(alpha)
            if idx is None:
                # alpha < 1 / (m + 1): anchor GPD tail above conformal ceiling
                ceil_alpha = 1.0 / float(self.m_calib + 1)
                row_clamped = self._row_for_n(min(max(n, n_lo), n_hi))
                base = float(row_clamped[-1])
                gpd_n = self._gpd_threshold(n, alpha)
                gpd_ceil = self._gpd_threshold(min(max(n, n_lo), n_hi), ceil_alpha)
                return float(max(base, gpd_n + max(0.0, base - gpd_ceil)))

            if n < n_lo:
                tau_lo = self._conformal_tau_for_row(self.max_scores[0], alpha)
                return float(min(tau_lo, self._gpd_threshold(n, alpha) + self._delta_lo))
            if n <= n_hi:
                row_n = self._row_for_n(n)
                return self._conformal_tau_for_row(row_n, alpha)
            tau_hi = self._conformal_tau_for_row(self.max_scores[-1], alpha)
            return float(max(tau_hi, self._gpd_threshold(n, alpha) + self._delta_top))

        tau_s = _single_tau(a_s)
        tau_p = _single_tau(a_p)
        if tau_s < tau_p:
            tau_s = tau_p
        out_taus = (float(tau_s), float(tau_p))
        if len(self._tau_cache) > 64:
            self._tau_cache.clear()
        self._tau_cache[cache_key] = out_taus
        return out_taus

    def p_nonmate(self, score: float, n_gallery: int) -> float:
        """
        Estimated probability that an open-set non-mated search of size n_gallery
        produces a top score >= score.

        Monotonically non-increasing in score and non-decreasing in n_gallery.
        """
        z = float(score)
        n = max(1, int(n_gallery))
        n_lo = int(self.n_grid[0])
        n_hi = int(self.n_grid[-1])

        if n < n_lo:
            p_lo = self._conformal_p_for_row(self.max_scores[0], z, n_lo)
            p_gpd_lo = self._gpd_p_nonmate(z - self._delta_lo, n)
            return float(np.clip(min(p_lo, p_gpd_lo), 0.0, 1.0))

        if n <= n_hi:
            row_n = self._row_for_n(n)
            return self._conformal_p_for_row(row_n, z, n)

        p_hi = self._conformal_p_for_row(self.max_scores[-1], z, n_hi)
        p_gpd_hi = self._gpd_p_nonmate(z - self._delta_top, n)
        return float(np.clip(max(p_hi, p_gpd_hi), 0.0, 1.0))

    def classify(
        self,
        score: float,
        n_gallery: int,
        alpha_strong: float = 0.01,
        alpha_possible: float = 0.08,
    ) -> str:
        """Classifies a score into 'STRONG', 'possible', or 'weak' at gallery size n_gallery."""
        tau_s, tau_p = self.tier_thresholds(n_gallery, alpha_strong=alpha_strong, alpha_possible=alpha_possible)
        s = float(score)
        if s >= tau_s:
            return "STRONG"
        if s >= tau_p:
            return "possible"
        return "weak"

    @classmethod
    def fit(
        cls,
        n_grid: Sequence[int],
        max_scores_by_n: Sequence[np.ndarray],
        base_nonmate_scores: np.ndarray,
        tail_quantile: float = 0.90,
    ) -> "OpenSetCalibrator":
        """
        Fits GPD Peaks-Over-Threshold on base non-mated person scores and calibrates
        effective multiplicity N_eff(N) = c * N^gamma against empirical max-of-N tails.
        """
        ng = np.asarray(n_grid, dtype=np.int64).reshape(-1)
        ms = np.stack([np.sort(np.asarray(row, dtype=np.float32).reshape(-1)) for row in max_scores_by_n])
        base = np.sort(np.asarray(base_nonmate_scores, dtype=np.float32).reshape(-1))
        if base.size < 20:
            raise ValueError("OpenSetCalibrator.fit needs at least 20 base non-mated scores")

        q_u = float(np.clip(tail_quantile, 0.70, 0.98))
        u = float(np.quantile(base, q_u))
        exceed = base[base > u] - u
        if exceed.size < 15:
            u = float(np.quantile(base, 0.80))
            exceed = base[base > u] - u
        p_u = float(np.mean(base > u))
        p_u = float(np.clip(p_u, 1e-4, 0.5))

        try:
            xi_fit, _loc, sigma_fit = genpareto.fit(exceed.astype(np.float64), floc=0.0)
            xi = float(np.clip(xi_fit, -0.40, 0.50))
            sigma_u = max(float(sigma_fit), 1e-3)
        except Exception:
            xi = 0.0
            sigma_u = max(float(np.mean(exceed)), 1e-3)

        # Fit N_eff(N) = c * N^gamma via log-log regression on upper quantiles of max_scores
        tmp = cls(ng, ms, u=u, p_u=p_u, xi=xi, sigma_u=sigma_u, base_scores=base)
        xs, ys = [], []
        for row_idx, n_val in enumerate(tmp.n_grid):
            if n_val <= 1:
                continue
            row = tmp.max_scores[row_idx]
            for alpha in (0.02, 0.05, 0.08, 0.12, 0.20):
                q_val = float(np.quantile(row, 1.0 - alpha))
                s1 = tmp._s1_tail(q_val)
                if 1e-8 < s1 < 0.25:
                    n_eff_emp = float(np.log(1.0 - alpha) / np.log(1.0 - s1))
                    if n_eff_emp > 0.5:
                        xs.append(np.log(float(n_val)))
                        ys.append(np.log(n_eff_emp))

        if len(xs) >= 3 and len(set(xs)) >= 2:
            X_fit = np.column_stack([np.ones(len(xs)), np.array(xs)])
            coef, _, _, _ = np.linalg.lstsq(X_fit, np.array(ys), rcond=None)
            c_fit = float(np.clip(np.exp(coef[0]), 0.25, 3.0))
            gamma_fit = float(np.clip(coef[1], 0.65, 1.05))
        else:
            c_fit, gamma_fit = 1.0, 1.0

        # Downsample base_scores cleanly to at most 8,192 quantiles to keep bundle compact
        if base.size > 8192:
            idx_sub = np.linspace(0, base.size - 1, num=8192, dtype=np.int64)
            base_compact = base[idx_sub]
        else:
            base_compact = base

        return cls(
            n_grid=tmp.n_grid,
            max_scores=tmp.max_scores,
            u=u,
            p_u=p_u,
            xi=xi,
            sigma_u=sigma_u,
            n_eff_c=c_fit,
            n_eff_gamma=gamma_fit,
            base_scores=base_compact,
        )


# ---------------------------------------------------------------------------
# Layer 3b / 4: COVE 1:1 Identity-Level Conformal & Background Z-Score
# ---------------------------------------------------------------------------


class CoveCalibrator:
    """
    Identity-level Split Conformal 1:1 verifier (COVE, Khazem 2026, arXiv:2610.35277).

    Ranks a 1:1 verification trial score against all binom(m, 2) pairwise scores
    between m disjoint calibration identities, achieving O(1/m^2) threshold resolution
    without hub/wolf pair correlation leakage.
    """

    def __init__(self, sorted_pair_scores: np.ndarray, n_identities: int):
        arr = np.sort(np.asarray(sorted_pair_scores, dtype=np.float32).reshape(-1))
        if arr.size == 0:
            raise ValueError("CoveCalibrator requires non-empty pair scores")
        self.sorted_scores = arr
        self.n_pairs = int(arr.size)
        self.n_identities = int(n_identities)

    @classmethod
    def fit(cls, embeddings: np.ndarray, identity_ids: Sequence[Any]) -> "CoveCalibrator":
        """
        Builds COVE calibrator from embeddings and identity IDs (uses 1 representative
        vector per unique identity, forming all binom(m, 2) cross-identity pairs).
        """
        vecs = _l2_normalize_rows(np.asarray(embeddings, dtype=np.float32))
        if vecs.shape[0] != len(identity_ids):
            raise ValueError("embeddings and identity_ids must have equal length")

        seen = {}
        rep_vecs = []
        for idx, ident in enumerate(identity_ids):
            if ident not in seen:
                seen[ident] = len(rep_vecs)
                rep_vecs.append(vecs[idx])

        m = len(rep_vecs)
        if m < 3:
            raise ValueError("CoveCalibrator.fit requires at least 3 distinct identities")

        mat = np.stack(rep_vecs).astype(np.float32)
        sim_mat = mat @ mat.T
        iu, ju = np.triu_indices(m, k=1)
        pair_scores = sim_mat[iu, ju].astype(np.float32)
        return cls(pair_scores, n_identities=m)

    def p_value(self, score: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Identity-level conformal false-accept p-value (1 + #{S_ij >= score}) / (K + 1)."""
        pos = np.searchsorted(self.sorted_scores, score, side="left")
        n_ge = self.n_pairs - pos
        p = (1.0 + n_ge) / float(self.n_pairs + 1)
        return float(p) if np.ndim(score) == 0 else p.astype(np.float64)

    def threshold(self, alpha: float = 0.001) -> float:
        """Smallest threshold tau with finite-sample conformal p_value(tau) <= alpha."""
        a = float(np.clip(alpha, 1e-9, 1.0 - 1e-9))
        k = self.n_pairs
        idx = int(np.ceil((1.0 - a) * (k + 1)))
        if idx >= k:
            return float(self.sorted_scores[-1] + 1e-6)
        idx = max(0, idx)
        # Step right past any tie so searchsorted(..., 'left') leaves <= k - idx items >= tau
        while idx < k and idx > 0 and self.sorted_scores[idx] == self.sorted_scores[idx - 1]:
            idx += 1
        if idx >= k:
            return float(self.sorted_scores[-1] + 1e-6)
        return float(self.sorted_scores[idx])

    def verify(self, score: float, alpha_accept: float = 0.001, alpha_review: float = 0.01) -> Dict[str, Any]:
        """Evaluates a 1:1 score with distribution-free FAR bounds."""
        p = float(self.p_value(float(score)))
        if p <= alpha_accept:
            dec = "ACCEPT"
        elif p <= alpha_review:
            dec = "REVIEW"
        else:
            dec = "REJECT"
        return {"score": float(score), "p_cove": p, "decision": dec}


def compute_bg_zscore(
    sorted_scores: Sequence[float],
    std_floor: float = DEFAULT_STD_FLOOR,
    z_standout: float = 4.0,
    gap_z_ambig: float = 1.0,
) -> Dict[str, Any]:
    """
    Computes Rank-1-vs-Rank-2 margin (s1 - s2) and trimmed background rank z-score z_bg
    over s_{3..50} (falling back to s_{2..} if fewer than 4 items).

    Skipping s_2 prevents cross-dataset duplicate entries or twins from collapsing z_bg.
    Emits strictly non-gating advisory flags ('standout', 'ambiguous_or_duplicate').
    """
    arr = np.asarray(list(sorted_scores), dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return {"margin_12": None, "z_bg": None, "mu_bg": None, "sigma_bg": None, "advisory_flags": []}
    if n == 1:
        return {"margin_12": None, "z_bg": None, "mu_bg": None, "sigma_bg": None, "advisory_flags": []}

    scores = np.sort(arr)[::-1]
    s1 = float(scores[0])
    s2 = float(scores[1])
    margin_12 = float(s1 - s2)

    bg = scores[2:50] if n >= 4 else scores[1:]
    mu_bg = float(np.mean(bg))
    sigma_bg = max(float(np.std(bg)), float(max(std_floor, 1e-6)))
    z_bg = float((s1 - mu_bg) / sigma_bg)

    flags = []
    if z_bg >= float(z_standout):
        flags.append("standout")
    if (margin_12 / sigma_bg) < float(gap_z_ambig):
        flags.append("ambiguous_or_duplicate")

    return {
        "margin_12": margin_12,
        "z_bg": z_bg,
        "mu_bg": mu_bg,
        "sigma_bg": sigma_bg,
        "advisory_flags": flags,
    }


# ---------------------------------------------------------------------------
# Bundle Persistence (.npz + .json metadata)
# ---------------------------------------------------------------------------


def _sanitize_onnx_stem(onnx_path: str) -> str:
    """Returns a filesystem-safe tag from an ONNX filename stem."""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_").lower()
    return clean or "custom"


def resolve_bundle_path(model_name: str, pretrained_dir: Optional[str] = None) -> str:
    """Resolves path to Arc face/pretrained/calib_<model>.npz (with AIMIDS_CALIB_DIR / AIMIDS_ONNX_PATH support)."""
    clean_model = str(model_name).strip()
    env_dir = os.environ.get("AIMIDS_CALIB_DIR", "").strip()
    if pretrained_dir is not None:
        base_dir = os.path.abspath(pretrained_dir)
    elif env_dir:
        base_dir = os.path.abspath(env_dir)
    else:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")

    if clean_model == "aimids_ft":
        onnx_env = os.environ.get("AIMIDS_ONNX_PATH", "").strip()
        if onnx_env:
            stem = _sanitize_onnx_stem(onnx_env)
            return os.path.join(base_dir, f"calib_{stem}.npz")
    return os.path.join(base_dir, f"calib_{clean_model}.npz")


class CalibrationBundle:
    """Serializes and loads all 4 calibration layers for one embedding engine."""

    def __init__(
        self,
        model_name: str,
        dim: int,
        asnorm: ExternalCohortASNorm,
        qmf: QMFCalibrator,
        openset: OpenSetCalibrator,
        cove: Optional[CoveCalibrator] = None,
        proj_basis: Optional[np.ndarray] = None,
        photo_hybrid_weight: float = 1.0,
        schema_version: int = SCHEMA_VERSION,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = str(model_name)
        self.dim = int(dim)
        self.schema_version = int(schema_version)
        self.cohort_hash = asnorm.cohort_hash
        self.asnorm = asnorm
        self.qmf = qmf
        self.openset = openset
        self.cove = cove
        if proj_basis is not None:
            pb = np.asarray(proj_basis, dtype=np.float32)
            if pb.ndim == 1:
                pb = pb.reshape(1, -1)
            if pb.size == 0:
                self.proj_basis = None
            else:
                if pb.ndim != 2 or pb.shape[1] != self.dim:
                    raise ValueError(
                        f"proj_basis must have shape (k, {self.dim}), got {pb.shape}"
                    )
                self.proj_basis = np.ascontiguousarray(_l2_normalize_rows(pb))
        else:
            self.proj_basis = None
        self.proj_hash = compute_proj_hash(self.proj_basis)
        w_f = float(photo_hybrid_weight)
        if not np.isfinite(w_f):
            raise ValueError("photo_hybrid_weight must be finite")
        self.photo_hybrid_weight = float(np.clip(w_f, 0.0, 1.0))
        self.metadata = dict(metadata or {})

    def apply_subspace(self, vecs: np.ndarray) -> np.ndarray:
        """Projects out age-correlated subspace (X - (X @ B.T) @ B, L2-renormalized)."""
        arr = np.asarray(vecs, dtype=np.float32)
        expected_dim = arr.shape[0] if arr.ndim == 1 else (arr.shape[1] if arr.ndim == 2 else -1)
        if expected_dim != self.dim:
            raise ValueError(f"Embedding dim ({expected_dim}) != bundle dim ({self.dim})")
        return project_subspace(arr, self.proj_basis)

    @staticmethod
    def _meta_path(npz_path: str) -> str:
        stem, _ = os.path.splitext(npz_path)
        return stem + ".json"

    def save(self, npz_path: str) -> Tuple[str, str]:
        """Writes compressed .npz arrays and sibling .json metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
        payload = {
            "schema_version": np.array([self.schema_version], dtype=np.int64),
            "model_name": np.array([self.model_name]),
            "dim": np.array([self.dim], dtype=np.int64),
            "cohort_hash": np.array([self.cohort_hash]),
            "proj_hash": np.array([self.proj_hash]),
            "photo_hybrid_weight": np.array([self.photo_hybrid_weight], dtype=np.float64),
            "cohort_vecs": self.asnorm.cohort_vecs.astype(np.float32),
            "asnorm_top_k": np.array([self.asnorm.top_k], dtype=np.int64),
            "asnorm_std_floor": np.array([self.asnorm.std_floor], dtype=np.float64),
            "qmf_weights": self.qmf.weights.astype(np.float64),
            "qmf_bias": np.array([self.qmf.bias], dtype=np.float64),
            "qmf_age_center": np.array([self.qmf.age_center], dtype=np.float64),
            "qmf_age_scale": np.array([self.qmf.age_scale], dtype=np.float64),
            "qmf_cosine_shift": np.array([self.qmf.cosine_shift], dtype=np.float64),
            "qmf_cosine_scale": np.array([self.qmf.cosine_scale], dtype=np.float64),
            "qmf_default_prior": np.array([self.qmf.default_prior], dtype=np.float64),
            "os_n_grid": self.openset.n_grid.astype(np.int64),
            "os_max_scores": self.openset.max_scores.astype(np.float32),
            "os_u": np.array([self.openset.u], dtype=np.float64),
            "os_p_u": np.array([self.openset.p_u], dtype=np.float64),
            "os_xi": np.array([self.openset.xi], dtype=np.float64),
            "os_sigma_u": np.array([self.openset.sigma_u], dtype=np.float64),
            "os_n_eff_c": np.array([self.openset.n_eff_c], dtype=np.float64),
            "os_n_eff_gamma": np.array([self.openset.n_eff_gamma], dtype=np.float64),
            "os_base_scores": self.openset.base_scores.astype(np.float32),
        }
        if self.proj_basis is not None:
            payload["proj_basis"] = self.proj_basis.astype(np.float32)
        if self.cove is not None:
            payload["cove_sorted_scores"] = self.cove.sorted_scores.astype(np.float32)
            payload["cove_n_identities"] = np.array([self.cove.n_identities], dtype=np.int64)

        np.savez_compressed(npz_path, **payload)

        meta_out = {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "dim": self.dim,
            "cohort_hash": self.cohort_hash,
            "proj_hash": self.proj_hash,
            "proj_k": int(self.proj_basis.shape[0]) if self.proj_basis is not None else 0,
            "photo_hybrid_weight": float(self.photo_hybrid_weight),
            "cohort_size": self.asnorm.m,
            "asnorm_top_k": self.asnorm.top_k,
            "asnorm_std_floor": self.asnorm.std_floor,
            "qmf_weights": {k: float(v) for k, v in zip(QMFCalibrator.FEATURE_NAMES, self.qmf.weights)},
            "qmf_bias": float(self.qmf.bias),
            "openset_n_grid": [int(x) for x in self.openset.n_grid],
            "openset_m_calib": self.openset.m_calib,
            "openset_gpd": {
                "u": float(self.openset.u),
                "p_u": float(self.openset.p_u),
                "xi": float(self.openset.xi),
                "sigma_u": float(self.openset.sigma_u),
                "n_eff_c": float(self.openset.n_eff_c),
                "n_eff_gamma": float(self.openset.n_eff_gamma),
            },
            "cove_n_identities": self.cove.n_identities if self.cove else None,
            "cove_n_pairs": self.cove.n_pairs if self.cove else None,
            **self.metadata,
        }
        meta_path = self._meta_path(npz_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)
        return npz_path, meta_path

    @classmethod
    def load(cls, npz_path: str) -> "CalibrationBundle":
        """Loads and validates CalibrationBundle from .npz and optional sibling .json."""
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Calibration bundle not found: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as d:
            schema = int(d["schema_version"][0])
            if schema not in (1, 2):
                raise ValueError(f"Unsupported calibration schema version {schema} (expected 1 or {SCHEMA_VERSION})")

            model_name = str(d["model_name"][0])
            dim = int(d["dim"][0])
            cohort_hash = str(d["cohort_hash"][0])

            raw_cohort = d["cohort_vecs"]
            actual_hash = compute_cohort_hash(raw_cohort)
            if actual_hash != cohort_hash:
                raise ValueError(f"Cohort hash mismatch in {npz_path}: expected {cohort_hash}, got {actual_hash}")
            asnorm = ExternalCohortASNorm(
                cohort_vecs=raw_cohort,
                top_k=int(d["asnorm_top_k"][0]),
                std_floor=float(d["asnorm_std_floor"][0]),
                cohort_hash=cohort_hash,
            )
            if asnorm.dim != dim:
                raise ValueError(f"Bundle dim mismatch: header {dim} vs cohort {asnorm.dim}")

            proj_basis: Optional[np.ndarray] = None
            if "proj_basis" in d:
                proj_basis = np.asarray(d["proj_basis"], dtype=np.float32)
                if proj_basis.size == 0:
                    proj_basis = None
            if "proj_hash" in d:
                expected_phash = str(d["proj_hash"][0])
                actual_phash = compute_proj_hash(proj_basis)
                if actual_phash != expected_phash:
                    raise ValueError(
                        f"Projection basis hash mismatch in {npz_path}: expected {expected_phash}, got {actual_phash}"
                    )

            photo_hybrid_weight = (
                float(d["photo_hybrid_weight"][0]) if "photo_hybrid_weight" in d else 1.0
            )

            qmf = QMFCalibrator(
                weights=d["qmf_weights"],
                bias=float(d["qmf_bias"][0]),
                age_center=float(d["qmf_age_center"][0]),
                age_scale=float(d["qmf_age_scale"][0]),
                cosine_shift=float(d["qmf_cosine_shift"][0]),
                cosine_scale=float(d["qmf_cosine_scale"][0]),
                default_prior=float(d["qmf_default_prior"][0]),
            )

            openset = OpenSetCalibrator(
                n_grid=d["os_n_grid"],
                max_scores=d["os_max_scores"],
                u=float(d["os_u"][0]),
                p_u=float(d["os_p_u"][0]),
                xi=float(d["os_xi"][0]),
                sigma_u=float(d["os_sigma_u"][0]),
                n_eff_c=float(d["os_n_eff_c"][0]),
                n_eff_gamma=float(d["os_n_eff_gamma"][0]),
                base_scores=d["os_base_scores"],
            )

            cove = None
            if "cove_sorted_scores" in d:
                cove = CoveCalibrator(
                    sorted_pair_scores=d["cove_sorted_scores"],
                    n_identities=int(d["cove_n_identities"][0]),
                )

        meta = {}
        meta_path = cls._meta_path(npz_path)
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        return cls(
            model_name=model_name,
            dim=dim,
            asnorm=asnorm,
            qmf=qmf,
            openset=openset,
            cove=cove,
            proj_basis=proj_basis,
            photo_hybrid_weight=photo_hybrid_weight,
            schema_version=schema,
            metadata=meta,
        )
