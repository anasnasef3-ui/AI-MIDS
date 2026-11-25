"""
strict_eval.py - Strict, Leakage-Audited, Out-of-Fold Evaluation Suite for AI-MIDS.

Evaluates full official protocols (NO --limit flag) across:
  - agedb30_official : Official 6,000-pair .mat protocol reading images straight from
                       03_Protocol_Images.tar in memory (ignoring .pts files).
  - calfw            : Official 6,000-pair 10-fold pairs_CALFW.txt protocol.
  - fgnet            : Child (<=12y) -> Adult (>12y) 1:N retrieval with age-gap buckets.
  - caf              : Child (<=12y) -> Adult (>12y) 1:N retrieval across 3 exclusion scopes
                       (fully_unseen, model_strict, cohort_only); skipped for scratch ONNX.
  - ecaf             : Child -> Adult 1:N retrieval split into ecaf_official (test=True)
                       and ecaf_all, auditing cohort, CACD, and known-ID overlap.

Enforces strict integrity guards (GuardError -> exit code 2):
  - Full protocol size/fold balance and zero missing tar stems.
  - Failed-detection verification pairs kept in denominator (score=-1.0), aborted if > 5%.
  - Retrieval path overlap guard, SHA-256 exact-duplicate exclusion, cross-ID collision audit,
    cosine >= 0.97 near-duplicate exclusion (reporting both clean and with_near_duplicates),
    and failed-detection probes/subjects kept in denominator as guaranteed misses.
  - Live cache re-embedding integrity check (cos >= 0.995 on deterministic sample).
  - Primary headline scope minimum gallery size (>= 50 IDs).
  - Strict out-of-fold threshold selection, Wilson 95% CIs, identity-cluster bootstrap CIs,
    and paired exact McNemar comparisons (strict_compare.json).
"""

from __future__ import annotations

import os

# Must be set before importing numpy / faiss / onnxruntime to keep Windows virtual commit bounded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import gc
import gzip
import hashlib
import io
import itertools
import json
import platform
import random
import re
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ARC_DIR = os.path.dirname(os.path.abspath(__file__))
if not sys.path or os.path.normcase(os.path.abspath(sys.path[0])) != os.path.normcase(ARC_DIR):
    if ARC_DIR in sys.path:
        sys.path.remove(ARC_DIR)
    sys.path.insert(0, ARC_DIR)

if hasattr(sys.stdout, "reconfigure") and getattr(sys.stdout, "encoding", None):
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import faiss
import numpy as np
import scipy
import scipy.io
from scipy.stats import binomtest

from bench_datasets import (
    collect_known_dataset_identities,
    load_cache,
    load_caf_gallery,
    load_calfw,
    load_fgnet_gallery,
    load_lag_cohort_identities,
    load_quality_cache,
    normalize_identity,
    parse_caf_images,
    parse_ecaf_images,
    save_cache,
)
from calibration import CalibrationBundle, extract_face_quality, resolve_bundle_path
from store import FaceStore


MAIN_DIR = os.path.dirname(ARC_DIR)
ASSETS_DIR = r"D:\GRAD\Assets"
DEFAULT_AGEDB_MAT = os.path.join(ASSETS_DIR, "AgeDB", "protocols", "04_FINAL_protocol_30_years.mat")
DEFAULT_AGEDB_TAR = os.path.join(ASSETS_DIR, "AgeDB", "protocols", "03_Protocol_Images.tar")
DEFAULT_CALFW_ROOT = os.path.join(ASSETS_DIR, "calfw")
DEFAULT_FGNET_ROOT = os.path.join(ASSETS_DIR, "FGNET", "images")
DEFAULT_CAF_ROOT = os.path.join(ASSETS_DIR, "CAF")
DEFAULT_ECAF_ROOT = os.path.join(ASSETS_DIR, "ECAF")
DEFAULT_CACD_ZIP = os.path.join(os.path.dirname(MAIN_DIR), "training", "custom_model-20261114T121928Z-1-001.zip")
DEFAULT_CACD_MEMBER = "custom_model/split_manifest_v2.csv.gz"

ALLOWED_ENGINES = ("aimids_crossage", "antelopev2", "aimids_ft", "buffalo_l")
ALLOWED_SUITES = ("agedb30_official", "calfw", "fgnet", "caf", "ecaf")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
NEAR_DUP_COS_THRESHOLD = 0.97
CACHE_VERIFY_MIN_COS = 0.995
MAX_VERIFY_FAILED_RATE = 0.05
MIN_HEADLINE_IDS = 50


class GuardError(RuntimeError):
    """Raised when a strict data-integrity, protocol, or cache check fails."""


# ---------------------------------------------------------------------------
# Path, ONNX, Hashing & Provenance Helpers
# ---------------------------------------------------------------------------


def sanitize_onnx_stem(onnx_path: str) -> str:
    """Returns a filesystem-safe tag from an ONNX filename stem."""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_").lower()
    return clean or "custom"


def resolve_custom_onnx(onnx_arg: Optional[str]) -> Optional[str]:
    """Resolves and validates an explicit --onnx path relative to cwd or Arc face/."""
    if not onnx_arg:
        return None
    cand_cwd = os.path.abspath(onnx_arg)
    if os.path.isfile(cand_cwd):
        return cand_cwd
    cand_arc = os.path.abspath(os.path.join(ARC_DIR, onnx_arg))
    if os.path.isfile(cand_arc):
        return cand_arc
    raise FileNotFoundError(f"ONNX model not found: {onnx_arg} (checked '{cand_cwd}' and '{cand_arc}')")


def should_skip_caf(onnx_path: Optional[str], engine_name: Optional[str] = None) -> bool:
    """Returns True when an ONNX path (or AIMIDS_ONNX_PATH for custom engines) indicates a scratch model."""
    candidates = []
    if onnx_path:
        candidates.append(str(onnx_path))
    if engine_name in ("aimids_ft", "aimids_crossage"):
        env_onnx = os.environ.get("AIMIDS_ONNX_PATH", "").strip()
        if env_onnx:
            candidates.append(env_onnx)
    return any("scratch" in c.lower() for c in candidates)


def get_cache_path(
    cache_dir: str,
    dataset_name: str,
    engine_name: str,
    use_tta: bool,
    onnx_path: Optional[str] = None,
) -> str:
    """Matches evaluate.get_cache_path() for standard datasets."""
    onnx_tag = f"_{sanitize_onnx_stem(onnx_path)}" if onnx_path else ""
    tta_tag = "_tta" if use_tta else ""
    fname = f"bench_cache_{dataset_name.lower()}_{engine_name.lower()}{onnx_tag}{tta_tag}.npz"
    return os.path.join(cache_dir, fname)


def get_agedb_official_cache_path(
    cache_dir: str,
    engine_name: str,
    use_tta: bool,
    onnx_path: Optional[str] = None,
) -> str:
    """Builds the isolated cache path for agedb30_official tar embeddings."""
    onnx_tag = f"_{sanitize_onnx_stem(onnx_path)}" if onnx_path else ""
    tta_tag = "_tta" if use_tta else ""
    fname = f"strict_cache_agedb30_official_{engine_name.lower()}{onnx_tag}{tta_tag}.npz"
    return os.path.join(cache_dir, fname)


def resolve_calibration_bundle(engine_name: str, onnx_path: Optional[str] = None) -> Optional[CalibrationBundle]:
    """Loads CalibrationBundle for standard calibrated engines; returns None for custom ONNX or missing bundle."""
    if onnx_path is not None:
        return None
    bundle_path = resolve_bundle_path(engine_name)
    if os.path.isfile(bundle_path):
        return CalibrationBundle.load(bundle_path)
    return None


def compute_file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Computes lowercase hex SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def sha256_file_if_exists(path: Optional[str]) -> Optional[str]:
    """Returns SHA-256 hex digest if file exists, else None."""
    if not path or not os.path.isfile(path):
        return None
    return compute_file_sha256(path)


def compute_manifest_sha256(records: Iterable[str]) -> str:
    """Computes deterministic SHA-256 over an ordered/sorted sequence of manifest strings."""
    h = hashlib.sha256()
    for rec in sorted(records):
        h.update(rec.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def _clamp_quality(q_val: Any) -> float:
    if q_val is None:
        return 1.0
    try:
        qf = float(q_val)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(qf):
        return 1.0
    return float(min(max(qf, 0.25), 1.0))


def _safe_unit_vec(vec: Any) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return None
    return (arr / norm).astype(np.float32)


def load_engine_lazy(
    engine_name: str,
    use_gpu: bool = True,
    use_tta: bool = True,
    onnx_path: Optional[str] = None,
) -> Any:
    """Lazily imports and initializes the requested recognition engine."""
    from face_engine import FaceEngine, ensure_cuda_dll_paths
    from face_engine_adaface_ft import FaceEngineAdaFaceFT
    from face_engine_antelopev2 import FaceEngineAntelopeV2
    from face_engine_custom import FaceEngineCrossAge

    name = engine_name.lower()
    if onnx_path and name not in ("aimids_ft", "aimids_crossage"):
        raise ValueError(f"--onnx is only supported with aimids_ft or aimids_crossage, got '{engine_name}'")

    if use_gpu:
        ensure_cuda_dll_paths()

    if name == "aimids_crossage":
        print(f">> Loading aimids_crossage (GPU={use_gpu}, Flip-TTA={use_tta})...")
        return FaceEngineCrossAge(onnx_path=onnx_path, use_gpu=use_gpu, use_tta=use_tta)
    if name == "antelopev2":
        print(f">> Loading antelopev2 (GPU={use_gpu}, Flip-TTA={use_tta})...")
        return FaceEngineAntelopeV2(use_gpu=use_gpu, use_tta=use_tta)
    if name == "aimids_ft":
        print(f">> Loading aimids_ft (GPU={use_gpu}, Flip-TTA={use_tta})...")
        return FaceEngineAdaFaceFT(onnx_path=onnx_path, use_gpu=use_gpu, use_tta=use_tta)
    if name == "buffalo_l":
        print(f">> Loading buffalo_l (GPU={use_gpu}, Flip-TTA=False)...")
        return FaceEngine(use_gpu=use_gpu)
    raise ValueError(f"Unsupported engine '{engine_name}'")


def get_engine_providers(engine: Any) -> Dict[str, List[str]]:
    """Extracts active ONNX Runtime providers from an instantiated engine's sessions."""
    out: Dict[str, List[str]] = {}
    if engine is None:
        return out

    def _collect_app(prefix: str, app: Any) -> None:
        models = getattr(app, "models", None)
        if isinstance(models, dict):
            for mod_name, mod in models.items():
                sess = getattr(mod, "session", None)
                if sess is not None and hasattr(sess, "get_providers"):
                    out[f"{prefix}.{mod_name}"] = list(sess.get_providers())

    _collect_app("app", getattr(engine, "app", None))
    if hasattr(engine, "antelope"):
        _collect_app("antelope", getattr(engine.antelope, "app", None))
    ort_sess = getattr(engine, "ort_session", None)
    if ort_sess is not None and hasattr(ort_sess, "get_providers"):
        out["ort_session"] = list(ort_sess.get_providers())
    if hasattr(engine, "adaface"):
        ada_sess = getattr(engine.adaface, "ort_session", None)
        if ada_sess is not None and hasattr(ada_sess, "get_providers"):
            out["adaface.ort_session"] = list(ada_sess.get_providers())
    return out


def collect_provenance(
    args: argparse.Namespace,
    engines: Sequence[str],
    onnx_path: Optional[str],
    protocol_hashes: Dict[str, Any],
    engine_providers: Dict[str, Any],
) -> Dict[str, Any]:
    """Collects full reproducibility and binary/bundle/protocol provenance."""
    git_head = "unknown"
    dirty_count = None
    try:
        head_proc = subprocess.run(
            ["git", "-C", MAIN_DIR, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if head_proc.returncode == 0:
            git_head = head_proc.stdout.strip()
        stat_proc = subprocess.run(
            ["git", "-C", MAIN_DIR, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if stat_proc.returncode == 0:
            dirty_count = len([ln for ln in stat_proc.stdout.splitlines() if ln.strip()])
    except Exception:
        pass

    try:
        import onnxruntime as ort
        ort_ver = ort.__version__
    except Exception:
        ort_ver = "not_loaded"

    antelope_dir = os.path.expanduser("~/.insightface/models/antelopev2")
    adaface_default = os.path.join(ARC_DIR, "pretrained", "aimids_adaface_ir101_ft.onnx")
    glint_onnx = os.path.join(antelope_dir, "glintr100.onnx")
    scrfd_onnx = os.path.join(antelope_dir, "scrfd_10g_bnkps.onnx")

    bundle_hashes = {}
    for eng in engines:
        bpath = resolve_bundle_path(eng) if onnx_path is None else None
        bundle_hashes[eng] = {
            "path": bpath if (bpath and os.path.isfile(bpath)) else None,
            "sha256": sha256_file_if_exists(bpath),
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "git_head": git_head,
        "git_dirty_file_count": dirty_count,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": ort_ver,
            "scipy": scipy.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
        },
        "engine_providers": engine_providers,
        "model_artifacts": {
            "aimids_adaface_ir101_ft_onnx": {
                "path": adaface_default,
                "sha256": sha256_file_if_exists(adaface_default),
            },
            "custom_onnx": {
                "path": onnx_path,
                "sha256": sha256_file_if_exists(onnx_path),
            },
            "glintr100_onnx": {
                "path": glint_onnx,
                "sha256": sha256_file_if_exists(glint_onnx),
            },
            "scrfd_10g_bnkps_onnx": {
                "path": scrfd_onnx,
                "sha256": sha256_file_if_exists(scrfd_onnx),
            },
        },
        "calibration_bundles": bundle_hashes,
        "protocol_hashes": protocol_hashes,
    }


# ---------------------------------------------------------------------------
# Protocol Loaders & Identity Audits
# ---------------------------------------------------------------------------


def load_cacd_finetune_identities(
    zip_path: str = DEFAULT_CACD_ZIP,
    member_name: str = DEFAULT_CACD_MEMBER,
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Reads CACD identities from custom_model/split_manifest_v2.csv.gz in memory.
    Guards that CACD splits are non-empty and a subset of {'train', 'validation'}.
    """
    if not os.path.isfile(zip_path):
        raise GuardError(f"CACD split manifest zip not found: {zip_path}")

    splits_seen: Set[str] = set()
    per_split_ids: Dict[str, Set[str]] = defaultdict(set)
    all_ids: Set[str] = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        if member_name not in zf.namelist():
            raise GuardError(f"Manifest member '{member_name}' missing in {zip_path}")
        with zf.open(member_name) as gf:
            with gzip.GzipFile(fileobj=gf) as gz:
                reader = csv.DictReader(io.TextIOWrapper(gz, encoding="utf-8", errors="replace"))
                for row in reader:
                    if row.get("dataset") != "CACD":
                        continue
                    sp = str(row.get("split") or "").strip()
                    splits_seen.add(sp)
                    norm = normalize_identity(row.get("identity", ""))
                    if norm and sp in ("train", "validation"):
                        per_split_ids[sp].add(norm)
                        all_ids.add(norm)

    if not splits_seen:
        raise GuardError(f"No CACD rows found in {zip_path}:{member_name}")
    unexpected = splits_seen - {"train", "validation"}
    if unexpected:
        raise GuardError(f"Unexpected CACD splits found (expected subset of train/validation): {sorted(unexpected)}")
    if not all_ids:
        raise GuardError("CACD normalized identity exclusion set is empty")

    per_split_counts = {k: len(v) for k, v in sorted(per_split_ids.items())}
    meta = {
        "zip_path": zip_path,
        "member_name": member_name,
        "splits_seen": sorted(splits_seen),
        "per_split_identities": per_split_counts,
        "total_identities": len(all_ids),
    }
    print(f"[CACD] Exclusion set: {len(all_ids)} IDs across splits {sorted(splits_seen)} ({per_split_counts})")
    return all_ids, meta


def _extract_mat_stem(entry: Any) -> str:
    """Unwraps MATLAB struct 'name' field across nested 0-d/1-d/2-d numpy arrays."""
    val = entry["name"]
    while isinstance(val, np.ndarray):
        if val.size == 0:
            raise GuardError("Empty stem array in AgeDB-30 .mat protocol")
        val = val.flat[0]
    return str(val).strip()


def load_agedb30_official(
    mat_path: str = DEFAULT_AGEDB_MAT,
    tar_path: str = DEFAULT_AGEDB_TAR,
) -> Tuple[List[Tuple[str, str, int, int]], Dict[str, tarfile.TarInfo], Dict[str, Any]]:
    """
    Loads the official 6,000-pair AgeDB-30 .mat protocol and validates tar image coverage.
    Does NOT extract 03_Protocol_Images.tar; ignores non-image files (.pts).
    Returns:
        pairs: list of ("agedbtar://<stem1>", "agedbtar://<stem2>", is_same, fold_idx)
        tar_map: {stem: TarInfo}
        meta: protocol stats
    """
    if not os.path.isfile(mat_path):
        raise GuardError(f"Official AgeDB-30 .mat not found: {mat_path}")
    if not os.path.isfile(tar_path):
        raise GuardError(f"Official AgeDB-30 tar not found: {tar_path}")

    data = scipy.io.loadmat(mat_path)
    if "splits" not in data:
        raise GuardError(f"'splits' missing in {mat_path}")
    splits = data["splits"]
    if splits.shape[0] != 10:
        raise GuardError(f"Expected 10 folds in {mat_path}, got shape {splits.shape}")

    raw_pairs: List[Tuple[str, str, int, int]] = []
    needed_stems: Set[str] = set()
    fold_counts = []

    for fold_idx in range(10):
        s = splits[fold_idx, 0]
        mat_pairs = s["pairs"][0, 0]
        annot = np.asarray(s["annot"][0, 0]).reshape(-1)
        n_pairs = mat_pairs.shape[1]
        if annot.shape[0] != n_pairs:
            raise GuardError(f"AgeDB-30 fold {fold_idx} annot mismatch: {annot.shape[0]} vs {n_pairs}")
        n_gen = 0
        n_imp = 0
        for i in range(n_pairs):
            stem1 = _extract_mat_stem(mat_pairs[0, i])
            stem2 = _extract_mat_stem(mat_pairs[1, i])
            is_same = 1 if int(annot[i]) == 1 else 0
            if is_same == 1:
                n_gen += 1
            else:
                n_imp += 1
            needed_stems.add(stem1)
            needed_stems.add(stem2)
            raw_pairs.append((f"agedbtar://{stem1}", f"agedbtar://{stem2}", is_same, fold_idx))
        if n_gen != 300 or n_imp != 300:
            raise GuardError(
                f"AgeDB-30 fold {fold_idx} balance guard failed: expected 300 genuine + 300 impostor, got {n_gen} + {n_imp}"
            )
        fold_counts.append({"fold": fold_idx, "genuine": n_gen, "impostor": n_imp})

    if len(raw_pairs) != 6000:
        raise GuardError(f"AgeDB-30 pair count guard failed: expected 6000, got {len(raw_pairs)}")
    total_gen = sum(1 for p in raw_pairs if p[2] == 1)
    total_imp = sum(1 for p in raw_pairs if p[2] == 0)
    if total_gen != 3000 or total_imp != 3000:
        raise GuardError(f"AgeDB-30 class balance guard failed: {total_gen} gen, {total_imp} imp")

    tar_map: Dict[str, tarfile.TarInfo] = {}
    ignored_non_images = 0
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            ext = os.path.splitext(member.name)[1].lower()
            if ext not in IMAGE_EXTS:
                ignored_non_images += 1
                continue
            stem = os.path.splitext(os.path.basename(member.name))[0]
            if stem in tar_map:
                raise GuardError(f"Duplicate image stem '{stem}' inside {tar_path}")
            tar_map[stem] = member

    missing_stems = sorted(needed_stems - set(tar_map.keys()))
    if missing_stems:
        raise GuardError(
            f"AgeDB-30 .mat stems missing from {tar_path}: {len(missing_stems)} missing (sample: {missing_stems[:5]})"
        )

    meta = {
        "mat_path": mat_path,
        "tar_path": tar_path,
        "total_pairs": len(raw_pairs),
        "unique_protocol_stems": len(needed_stems),
        "tar_image_count": len(tar_map),
        "tar_non_image_ignored": ignored_non_images,
        "fold_counts": fold_counts,
    }
    print(
        f"[AgeDB-30 Official] Verified 6000 pairs (3000+3000), {len(needed_stems)} unique stems, "
        f"0 missing in tar ({ignored_non_images} non-image entries ignored)."
    )
    return raw_pairs, tar_map, meta


def load_calfw_official(root: str = DEFAULT_CALFW_ROOT) -> Tuple[List[Tuple[str, str, int, int]], Dict[str, Any]]:
    """Loads full 6,000-pair CALFW 10-fold protocol and enforces strict pair/fold guards."""
    pairs, desc = load_calfw(root=root, limit=None)
    if len(pairs) != 6000:
        raise GuardError(f"CALFW pair count guard failed: expected 6000, got {len(pairs)}")
    total_gen = sum(1 for p in pairs if p[2] == 1)
    total_imp = sum(1 for p in pairs if p[2] == 0)
    if total_gen != 3000 or total_imp != 3000:
        raise GuardError(f"CALFW balance guard failed: {total_gen} gen, {total_imp} imp")
    for f in range(10):
        fg = sum(1 for p in pairs if p[3] == f and p[2] == 1)
        fi = sum(1 for p in pairs if p[3] == f and p[2] == 0)
        if fg != 300 or fi != 300:
            raise GuardError(f"CALFW fold {f} balance guard failed: {fg} gen, {fi} imp (expected 300+300)")

    unique_paths = {p[0] for p in pairs} | {p[1] for p in pairs}
    meta = {
        "root": root,
        "description": desc,
        "total_pairs": len(pairs),
        "unique_images": len(unique_paths),
    }
    return pairs, meta


def validate_headline_scope_size(scope_key: str, gallery_by_id: Dict[str, Any], min_ids: int = MIN_HEADLINE_IDS) -> None:
    """Enforces minimum gallery identity count on primary headline scopes."""
    n = len(gallery_by_id)
    if n < min_ids:
        raise GuardError(f"Headline scope '{scope_key}' has only {n} gallery IDs (< {min_ids} minimum)")


def _compute_filtering_breakdown(
    eligible_ids: Set[str],
    cohort_ids: Set[str],
    cacd_ids: Set[str],
    known_ids: Set[str],
) -> Dict[str, Any]:
    rem_cohort = eligible_ids - cohort_ids
    removed_cohort = len(eligible_ids & cohort_ids)
    rem_model_strict = rem_cohort - cacd_ids
    removed_cacd = len(rem_cohort & cacd_ids)
    rem_fully_unseen = rem_model_strict - known_ids
    removed_known = len(rem_model_strict & known_ids)
    return {
        "total_unfiltered": len(eligible_ids),
        "removed_cohort": removed_cohort,
        "removed_cacd": removed_cacd,
        "removed_known": removed_known,
        "raw_overlap_cohort": len(eligible_ids & cohort_ids),
        "raw_overlap_cacd": len(eligible_ids & cacd_ids),
        "raw_overlap_known": len(eligible_ids & known_ids),
        "remaining_cohort_only": len(rem_cohort),
        "remaining_model_strict": len(rem_model_strict),
        "remaining_fully_unseen": len(rem_fully_unseen),
    }


def _scope_filter_summary(breakdown: Dict[str, Any], scope_name: str, remaining_count: int) -> Dict[str, Any]:
    return {
        "total_unfiltered": breakdown["total_unfiltered"],
        "removed_cohort": breakdown["removed_cohort"],
        "removed_cacd": breakdown["removed_cacd"] if scope_name in ("model_strict", "fully_unseen") else 0,
        "removed_known": breakdown["removed_known"] if scope_name == "fully_unseen" else 0,
        "remaining": remaining_count,
    }


def build_caf_dataset(
    root: str = DEFAULT_CAF_ROOT,
    cohort_ids: Optional[Set[str]] = None,
    cacd_ids: Optional[Set[str]] = None,
    known_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Loads CAF (split='child_to_adult' ONLY) across scopes:
      - fully_unseen (headline)
      - model_strict
      - cohort_only
    """
    use_default_loaders = cohort_ids is None and cacd_ids is None and known_ids is None
    eff_cohort = set(cohort_ids) if cohort_ids is not None else load_lag_cohort_identities()
    eff_cacd = set(cacd_ids) if cacd_ids is not None else load_cacd_finetune_identities()[0]
    eff_known = set(known_ids) if known_ids is not None else collect_known_dataset_identities(include_demo_dbs=True)

    identities = parse_caf_images(root=root)
    eligible: Dict[str, Tuple[List[Tuple[int, str]], List[Dict[str, Any]]]] = {}
    for norm_id in sorted(identities.keys()):
        info = identities[norm_id]
        items = info["photos"]
        if len(items) < 2:
            continue
        gal_items = [(age, p) for age, p in items if age <= 12]
        probe_items = [(age, p) for age, p in items if age > 12]
        if gal_items and probe_items:
            probes_list = [
                {
                    "subject_id": norm_id,
                    "norm_id": norm_id,
                    "display_name": info["display_name"],
                    "age": age,
                    "path": path,
                }
                for age, path in probe_items
            ]
            eligible[norm_id] = (gal_items, probes_list)

    breakdown = _compute_filtering_breakdown(set(eligible.keys()), eff_cohort, eff_cacd, eff_known)

    scopes: Dict[str, Dict[str, Any]] = {}
    scope_specs = [
        ("fully_unseen", True, True, eff_cacd, True),
        ("model_strict", True, False, eff_cacd, False),
        ("cohort_only", True, False, None, False),
    ]
    all_paths: Set[str] = set()
    for scope_name, ex_coh, ex_kn, ex_cacd, is_headline in scope_specs:
        if use_default_loaders:
            gal, pr = load_caf_gallery(
                root=root,
                exclude_cohort=ex_coh,
                exclude_known=ex_kn,
                exclude_ids=ex_cacd,
                split="child_to_adult",
            )
        else:
            blocked = set()
            if ex_coh:
                blocked |= eff_cohort
            if ex_kn:
                blocked |= eff_known
            if ex_cacd:
                blocked |= ex_cacd
            gal = {}
            pr = []
            for norm_id in sorted(eligible.keys()):
                if norm_id in blocked:
                    continue
                g_items, p_items = eligible[norm_id]
                gal[norm_id] = list(g_items)
                pr.extend(p_items)

        for items in gal.values():
            for _, p in items:
                all_paths.add(p)
        for p in pr:
            all_paths.add(p["path"])

        if is_headline:
            validate_headline_scope_size(f"caf:{scope_name}", gal)

        scopes[scope_name] = {
            "gallery_by_id": gal,
            "probes": pr,
            "is_headline": is_headline,
            "filtering": _scope_filter_summary(breakdown, scope_name, len(gal)),
        }

    return {
        "all_paths": sorted(all_paths),
        "id_filtering": breakdown,
        "scopes": scopes,
    }


def build_ecaf_datasets(
    root: str = DEFAULT_ECAF_ROOT,
    cohort_ids: Optional[Set[str]] = None,
    cacd_ids: Optional[Set[str]] = None,
    known_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Parses ECAF once and builds both sub-datasets:
      - ecaf_official (official_test_only=True): model_strict (headline), cohort_only, fully_unseen
      - ecaf_all (official_test_only=False): model_strict (headline), fully_unseen
    """
    eff_cohort = set(cohort_ids) if cohort_ids is not None else load_lag_cohort_identities()
    eff_cacd = set(cacd_ids) if cacd_ids is not None else load_cacd_finetune_identities()[0]
    eff_known = set(known_ids) if known_ids is not None else collect_known_dataset_identities(include_demo_dbs=True)

    identities = parse_ecaf_images(root=root)

    def _build_subdataset(official_only: bool, label: str, scope_defs: List[Tuple[str, Set[str], bool]]) -> Dict[str, Any]:
        eligible: Dict[str, Tuple[List[Tuple[Optional[int], str]], List[Dict[str, Any]]]] = {}
        for norm_id in sorted(identities.keys()):
            info = identities[norm_id]
            if official_only and not info["official_test"]:
                continue
            if not info["child"] or not info["adult"]:
                continue
            gal_items = [(None, fp) for _, fp in info["child"]]
            probe_items = [
                {
                    "subject_id": norm_id,
                    "norm_id": norm_id,
                    "display_name": info["display_name"],
                    "age": None,
                    "path": fp,
                }
                for _, fp in info["adult"]
            ]
            eligible[norm_id] = (gal_items, probe_items)

        breakdown = _compute_filtering_breakdown(set(eligible.keys()), eff_cohort, eff_cacd, eff_known)
        scopes: Dict[str, Dict[str, Any]] = {}

        for scope_name, blocked_set, is_headline in scope_defs:
            gal: Dict[str, List[Tuple[Optional[int], str]]] = {}
            pr: List[Dict[str, Any]] = []
            for norm_id in sorted(eligible.keys()):
                if norm_id in blocked_set:
                    continue
                g_items, p_items = eligible[norm_id]
                gal[norm_id] = list(g_items)
                pr.extend(p_items)

            if is_headline:
                validate_headline_scope_size(f"{label}:{scope_name}", gal)

            scope_meta: Dict[str, Any] = {
                "gallery_by_id": gal,
                "probes": pr,
                "is_headline": is_headline,
                "filtering": _scope_filter_summary(breakdown, scope_name, len(gal)),
            }
            if official_only and scope_name == "model_strict":
                overlap_n = len(set(gal.keys()) & eff_known)
                scope_meta["known_overlap_count"] = overlap_n
                scope_meta["known_overlap_note"] = (
                    f"{overlap_n}/{len(gal)} official IDs appear in AgeDB/CALFW "
                    "(affecting QMF/OpenSet calibration fit, not aimids_ft training)"
                )
            if official_only and scope_name == "fully_unseen":
                scope_meta["small_gallery_warning"] = len(gal) < MIN_HEADLINE_IDS

            scopes[scope_name] = scope_meta

        return {
            "id_filtering": breakdown,
            "scopes": scopes,
        }

    all_paths: Set[str] = set()
    for info in identities.values():
        if info["child"] and info["adult"]:
            for _, fp in info["child"]:
                all_paths.add(fp)
            for _, fp in info["adult"]:
                all_paths.add(fp)

    ecaf_official = _build_subdataset(
        official_only=True,
        label="ecaf_official",
        scope_defs=[
            ("model_strict", eff_cohort | eff_cacd, True),
            ("cohort_only", eff_cohort, False),
            ("fully_unseen", eff_cohort | eff_cacd | eff_known, False),
        ],
    )
    ecaf_all = _build_subdataset(
        official_only=False,
        label="ecaf_all",
        scope_defs=[
            ("model_strict", eff_cohort | eff_cacd, True),
            ("fully_unseen", eff_cohort | eff_cacd | eff_known, False),
        ],
    )

    return {
        "all_paths": sorted(all_paths),
        "ecaf_official": ecaf_official,
        "ecaf_all": ecaf_all,
    }


# ---------------------------------------------------------------------------
# Image Reading, Embedding Cache & Live Integrity Check
# ---------------------------------------------------------------------------


def read_bgr_image(
    key: str,
    tar_handle: Optional[tarfile.TarFile] = None,
    tar_map: Optional[Dict[str, tarfile.TarInfo]] = None,
) -> Optional[np.ndarray]:
    """Reads a BGR uint8 image from disk or an open AgeDB protocol tar archive."""
    if key.startswith("agedbtar://"):
        stem = key[len("agedbtar://"):]
        if tar_handle is None or tar_map is None or stem not in tar_map:
            raise FileNotFoundError(f"Tar member missing for key '{key}'")
        member = tar_map[stem]
        extracted = tar_handle.extractfile(member)
        if extracted is None:
            return None
        raw = extracted.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    raw_bytes = np.fromfile(key, dtype=np.uint8)
    return cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)


def ensure_strict_embeddings(
    keys: Iterable[str],
    cache_path: str,
    get_engine_fn: Callable[[], Any],
    tar_handle: Optional[tarfile.TarFile] = None,
    tar_map: Optional[Dict[str, tarfile.TarInfo]] = None,
) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, float]]:
    """Loads disk cache and embeds only missing keys (supports both file paths and agedbtar:// keys)."""
    unique_keys = sorted(set(keys))
    cache_dict = load_cache(cache_path)
    quality_dict = load_quality_cache(cache_path)

    missing = [k for k in unique_keys if k not in cache_dict]
    if not missing:
        print(f">> Cache hit: all {len(unique_keys)} images loaded from {os.path.basename(cache_path)}")
        return cache_dict, quality_dict

    parent_dir = os.path.dirname(os.path.abspath(cache_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    engine = get_engine_fn()
    device_label = "GPU" if getattr(engine, "use_gpu", False) else "CPU"
    print(f">> Embedding {len(missing)} / {len(unique_keys)} uncached images on {device_label}...")
    t0 = time.time()

    try:
        for idx, key in enumerate(missing, 1):
            try:
                img = read_bgr_image(key, tar_handle=tar_handle, tar_map=tar_map)
                if img is None:
                    cache_dict[key] = None
                else:
                    emb_res = engine.embed(img)
                    if emb_res is not None and "embedding" in emb_res and emb_res["embedding"] is not None:
                        cache_dict[key] = np.asarray(emb_res["embedding"], dtype=np.float32).reshape(-1)
                        quality_dict[key] = float(extract_face_quality(engine, img, emb_res))
                    else:
                        cache_dict[key] = None
            except Exception:
                cache_dict[key] = None

            if idx % 100 == 0 or idx == len(missing):
                elapsed = time.time() - t0
                speed = elapsed / idx
                eta_sec = (len(missing) - idx) * speed
                print(
                    f"  [{idx}/{len(missing)}] ({idx / len(missing) * 100:5.1f}%) | "
                    f"{speed:.3f}s/img | Elapsed: {elapsed / 60:4.1f}m | ETA: {eta_sec / 60:4.1f}m",
                    flush=True,
                )
                save_cache(cache_path, cache_dict, quality_dict=quality_dict)
    finally:
        save_cache(cache_path, cache_dict, quality_dict=quality_dict)

    return cache_dict, quality_dict


def verify_cache_integrity(
    keys: Iterable[str],
    cache_dict: Dict[str, Optional[np.ndarray]],
    get_engine_fn: Callable[[], Any],
    verify_sample: int = 32,
    seed: int = 0,
    min_cos_threshold: float = CACHE_VERIFY_MIN_COS,
    tar_handle: Optional[tarfile.TarFile] = None,
    tar_map: Optional[Dict[str, tarfile.TarInfo]] = None,
) -> Dict[str, Any]:
    """
    Deterministically samples up to `verify_sample` cached non-None embeddings, re-reads
    their source images (including agedbtar://), re-embeds live, and asserts cos >= 0.995.
    """
    if verify_sample <= 0:
        return {"n_checked": 0, "min_cos": None, "mean_cos": None, "offending": []}

    valid_keys = sorted(k for k in set(keys) if cache_dict.get(k) is not None)
    if not valid_keys:
        raise GuardError("Cache integrity check failed: 0 non-None embeddings available to verify")

    k_sample = min(int(verify_sample), len(valid_keys))
    rng = random.Random(seed)
    sampled = rng.sample(valid_keys, k_sample)

    engine = get_engine_fn()
    cos_vals: List[float] = []
    offending: List[Dict[str, Any]] = []

    for key in sampled:
        img = read_bgr_image(key, tar_handle=tar_handle, tar_map=tar_map)
        live_res = engine.embed(img) if img is not None else None
        if live_res is None or "embedding" not in live_res or live_res["embedding"] is None:
            offending.append({"key": key, "cos": None, "reason": "live_embed_none"})
            continue

        c_vec = _safe_unit_vec(cache_dict.get(key))
        l_vec = _safe_unit_vec(live_res["embedding"])
        if c_vec is None or l_vec is None or c_vec.shape != l_vec.shape:
            offending.append({"key": key, "cos": None, "reason": "invalid_shape_or_norm"})
            continue

        cos = float(np.dot(c_vec, l_vec))
        cos_vals.append(cos)
        if not np.isfinite(cos) or cos < min_cos_threshold:
            offending.append({"key": key, "cos": cos, "reason": f"cos_below_{min_cos_threshold}"})

    result = {
        "n_checked": k_sample,
        "min_cos": float(min(cos_vals)) if cos_vals else None,
        "mean_cos": float(np.mean(cos_vals)) if cos_vals else None,
        "offending": offending,
    }
    if offending:
        raise GuardError(
            f"Cache integrity check failed ({len(offending)}/{k_sample} offending, "
            f"threshold={min_cos_threshold}): {offending[:3]}"
        )
    print(
        f">> Cache integrity verified: {k_sample}/{k_sample} live re-embeds passed "
        f"(min_cos={result['min_cos']:.6f}, mean_cos={result['mean_cos']:.6f})"
    )
    return result


# ---------------------------------------------------------------------------
# Pure Statistical & Evaluation Functions
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score confidence interval in [0.0, 1.0] proportion units."""
    if n <= 0:
        return (0.0, 0.0)
    p_hat = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p_hat + z2 / (2.0 * float(n))) / denom
    rad = (z / denom) * float(np.sqrt((p_hat * (1.0 - p_hat) / float(n)) + (z2 / (4.0 * float(n) * float(n)))))
    return (float(max(0.0, center - rad)), float(min(1.0, center + rad)))


def cluster_bootstrap_ci(
    subject_ids: Sequence[Any],
    hit1_flags: Sequence[bool],
    hit10_flags: Sequence[bool],
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[List[float], List[float]]:
    """
    Identity-cluster bootstrap 95% CIs (in % units) for Rank-1 and Top-10 retrieval.
    Resamples unique subject_ids with replacement, weighting each subject's probes/hits.
    """
    n_total = len(subject_ids)
    if n_total == 0:
        return ([0.0, 0.0], [0.0, 0.0])

    clusters: Dict[Any, List[int]] = {}
    for sid, h1, h10 in zip(subject_ids, hit1_flags, hit10_flags):
        entry = clusters.get(sid)
        if entry is None:
            entry = [0, 0, 0]
            clusters[sid] = entry
        entry[0] += 1
        if h1:
            entry[1] += 1
        if h10:
            entry[2] += 1

    vals = np.array(list(clusters.values()), dtype=np.float64)
    c_n = vals[:, 0]
    c_r1 = vals[:, 1]
    c_t10 = vals[:, 2]
    k_clusters = len(clusters)

    pt_r1 = float(np.sum(c_r1) / np.sum(c_n)) * 100.0
    pt_t10 = float(np.sum(c_t10) / np.sum(c_n)) * 100.0
    if n_boot <= 0 or k_clusters <= 1:
        return ([pt_r1, pt_r1], [pt_t10, pt_t10])

    rng = np.random.default_rng(seed)
    pvals = np.full(k_clusters, 1.0 / float(k_clusters), dtype=np.float64)
    weights = rng.multinomial(k_clusters, pvals, size=int(n_boot)).astype(np.float64)
    denoms = weights @ c_n
    boot_r1 = ((weights @ c_r1) / denoms) * 100.0
    boot_t10 = ((weights @ c_t10) / denoms) * 100.0
    r1_lo, r1_hi = np.percentile(boot_r1, [2.5, 97.5])
    t10_lo, t10_hi = np.percentile(boot_t10, [2.5, 97.5])
    return ([float(r1_lo), float(r1_hi)], [float(t10_lo), float(t10_hi)])


def mcnemar_exact(hits_a: Sequence[bool], hits_b: Sequence[bool]) -> Dict[str, Any]:
    """Exact two-sided paired McNemar test using scipy.stats.binomtest."""
    arr_a = np.asarray(list(hits_a), dtype=bool).reshape(-1)
    arr_b = np.asarray(list(hits_b), dtype=bool).reshape(-1)
    if arr_a.shape[0] != arr_b.shape[0]:
        raise ValueError(f"McNemar inputs must match length: {arr_a.shape[0]} vs {arr_b.shape[0]}")

    n = int(arr_a.shape[0])
    if n == 0:
        return {
            "n": 0,
            "acc_a": 0.0,
            "acc_b": 0.0,
            "diff_pct": 0.0,
            "b": 0,
            "c": 0,
            "p_value": 1.0,
            "significant_p05": False,
        }

    acc_a = float(np.mean(arr_a)) * 100.0
    acc_b = float(np.mean(arr_b)) * 100.0
    b = int(np.sum(arr_a & ~arr_b))
    c = int(np.sum(~arr_a & arr_b))
    p_val = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    return {
        "n": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "diff_pct": acc_a - acc_b,
        "b": b,
        "c": c,
        "p_value": p_val,
        "significant_p05": bool(p_val < 0.05),
    }


def evaluate_verification_pairs(
    pairs: Sequence[Tuple[str, str, int, int]],
    cache: Dict[str, Optional[np.ndarray]],
    max_failed_rate: float = MAX_VERIFY_FAILED_RATE,
    num_folds: int = 10,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Runs strict out-of-fold 1:1 raw single-image cosine verification.
    Failed detections (emb is None) get score=-1.0 and stay in the denominator;
    failed_pair_rate > max_failed_rate raises GuardError.
    Returns (summary_dict, oof_correct_bool_array).
    """
    if not pairs:
        raise GuardError("Verification protocol is empty")

    m = len(pairs)
    scores = np.empty(m, dtype=np.float64)
    labels = np.empty(m, dtype=np.int32)
    folds = np.empty(m, dtype=np.int32)
    failed_mask = np.zeros(m, dtype=bool)
    failed_keys: Set[str] = set()

    for idx, (ka, kb, is_same, fold_idx) in enumerate(pairs):
        labels[idx] = 1 if int(is_same) == 1 else 0
        folds[idx] = int(fold_idx)
        va = _safe_unit_vec(cache.get(ka))
        vb = _safe_unit_vec(cache.get(kb))
        if va is None:
            failed_keys.add(ka)
        if vb is None:
            failed_keys.add(kb)
        if va is None or vb is None:
            scores[idx] = -1.0
            failed_mask[idx] = True
        else:
            scores[idx] = float(np.clip(np.dot(va, vb), -1.0, 1.0))

    n_failed_pairs = int(np.sum(failed_mask))
    failed_rate = float(n_failed_pairs) / float(m)
    if failed_rate > max_failed_rate:
        raise GuardError(
            f"Verification failed-detection pair rate {failed_rate:.4%} exceeds {max_failed_rate:.2%} "
            f"({n_failed_pairs}/{m} pairs)"
        )

    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if s_min < -0.20 or s_max > 0.80:
        lo = min(-0.20, float(np.floor(s_min * 1000.0) / 1000.0) - 0.05)
        hi = max(0.80, float(np.ceil(s_max * 1000.0) / 1000.0) + 0.05)
        n_pts = int(round((hi - lo) / 0.001)) + 1
        threshold_grid = np.linspace(lo, hi, max(1001, n_pts))
    else:
        threshold_grid = np.linspace(-0.20, 0.80, 1001)

    oof_correct = np.zeros(m, dtype=bool)
    fold_acc: List[float] = []
    fold_tar: List[float] = []
    fold_achieved_far: List[float] = []
    tau_acc_list: List[float] = []
    tau_far1_list: List[float] = []

    for f in range(num_folds):
        test_mask = folds == f
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise GuardError(f"Empty train or test partition on fold {f}")

        train_scores = scores[train_mask]
        train_labels = labels[train_mask]
        test_scores = scores[test_mask]
        test_labels = labels[test_mask]

        preds = train_scores[None, :] >= threshold_grid[:, None]
        accs = np.mean(preds == (train_labels[None, :] == 1), axis=1)
        max_acc = float(np.max(accs))
        best_idxs = np.flatnonzero(np.abs(accs - max_acc) <= 1e-12)
        tau_acc = float(threshold_grid[int(best_idxs[len(best_idxs) // 2])])

        test_corr = (test_scores >= tau_acc) == (test_labels == 1)
        oof_correct[test_mask] = test_corr
        fold_acc.append(float(np.mean(test_corr)) * 100.0)
        tau_acc_list.append(tau_acc)

        train_imp = train_scores[train_labels == 0]
        if train_imp.size == 0:
            raise GuardError(f"Fold {f} has 0 training impostor pairs")
        sorted_train_imp = np.sort(train_imp)
        idx_far1 = min(int(np.floor(0.99 * len(sorted_train_imp))), len(sorted_train_imp) - 1)
        tau_far1 = float(sorted_train_imp[idx_far1])

        test_gen = test_scores[test_labels == 1]
        test_imp = test_scores[test_labels == 0]
        fold_tar.append(float(np.mean(test_gen >= tau_far1)) * 100.0 if test_gen.size else 0.0)
        fold_achieved_far.append(float(np.mean(test_imp >= tau_far1)) * 100.0 if test_imp.size else 0.0)
        tau_far1_list.append(tau_far1)

    k_corr = int(np.sum(oof_correct))
    ci_lo, ci_hi = wilson_ci(k_corr, m)
    all_gen = scores[labels == 1]
    all_imp = scores[labels == 0]

    summary = {
        "total_pairs": m,
        "n_genuine": int(np.sum(labels == 1)),
        "n_impostor": int(np.sum(labels == 0)),
        "n_failed_images": len(failed_keys),
        "n_failed_pairs": n_failed_pairs,
        "failed_pair_rate": failed_rate,
        "mean_acc": float(np.mean(fold_acc)),
        "std_acc": float(np.std(fold_acc)),
        "pooled_acc": (float(k_corr) / float(m)) * 100.0,
        "pooled_acc_ci95": [ci_lo * 100.0, ci_hi * 100.0],
        "mean_tar_far1": float(np.mean(fold_tar)),
        "std_tar_far1": float(np.std(fold_tar)),
        "mean_achieved_far1": float(np.mean(fold_achieved_far)),
        "gen_strong_pct": float(np.mean(all_gen >= 0.36)) * 100.0 if all_gen.size else 0.0,
        "imp_poss_plus_pct": float(np.mean(all_imp >= 0.26)) * 100.0 if all_imp.size else 0.0,
        "tau_acc_per_fold": tau_acc_list,
        "tau_far1_per_fold": tau_far1_list,
        "fold_acc_per_fold": fold_acc,
        "fold_tar_per_fold": fold_tar,
        "fold_achieved_far_per_fold": fold_achieved_far,
    }
    return summary, oof_correct


def _make_probe_key(subject_id: str, path: str) -> str:
    return f"{subject_id}||{os.path.normcase(os.path.abspath(path))}"


def _summarize_retrieval_view(
    records: Sequence[Dict[str, Any]],
    gallery_ids_total: int,
    gallery_ids_enrolled: int,
    n_exact_dup_excluded: int,
    n_near_dup_excluded: int,
    bootstrap_n: int,
    seed: int,
    compute_age_buckets: bool = False,
) -> Dict[str, Any]:
    n_probes = len(records)
    denom = float(n_probes) if n_probes > 0 else 1.0
    n_failed = sum(1 for r in records if r["failed_probe_detect"])
    n_r1 = sum(1 for r in records if r["hit_1"])
    n_r5 = sum(1 for r in records if r["hit_5"])
    n_r10 = sum(1 for r in records if r["hit_10"])
    n_strong = sum(1 for r in records if r["is_strong"])
    n_poss = sum(1 for r in records if r["is_possible_plus"])

    r1_ci, t10_ci = cluster_bootstrap_ci(
        [r["subject_id"] for r in records],
        [r["hit_1"] for r in records],
        [r["hit_10"] for r in records],
        n_boot=bootstrap_n,
        seed=seed,
    )

    out: Dict[str, Any] = {
        "gallery_ids_total": int(gallery_ids_total),
        "gallery_ids_enrolled": int(gallery_ids_enrolled),
        "probes": int(n_probes),
        "n_failed_probe_detect": int(n_failed),
        "n_exact_dup_excluded": int(n_exact_dup_excluded),
        "n_near_dup_excluded": int(n_near_dup_excluded),
        "rank1": (float(n_r1) / denom) * 100.0 if n_probes else 0.0,
        "top5": (float(n_r5) / denom) * 100.0 if n_probes else 0.0,
        "top10": (float(n_r10) / denom) * 100.0 if n_probes else 0.0,
        "strong": (float(n_strong) / denom) * 100.0 if n_probes else 0.0,
        "possible_plus": (float(n_poss) / denom) * 100.0 if n_probes else 0.0,
        "rank1_ci95": r1_ci,
        "top10_ci95": t10_ci,
    }

    if compute_age_buckets:
        buckets: Dict[str, Dict[str, Any]] = {}
        for bname in ("<10", "10-19", ">=20"):
            b_recs = [r for r in records if r.get("age_bucket") == bname]
            b_n = len(b_recs)
            b_denom = float(b_n) if b_n > 0 else 1.0
            buckets[bname] = {
                "probes": int(b_n),
                "n_failed_probe_detect": sum(1 for r in b_recs if r["failed_probe_detect"]),
                "rank1": (sum(1 for r in b_recs if r["hit_1"]) / b_denom) * 100.0 if b_n else 0.0,
                "top5": (sum(1 for r in b_recs if r["hit_5"]) / b_denom) * 100.0 if b_n else 0.0,
                "top10": (sum(1 for r in b_recs if r["hit_10"]) / b_denom) * 100.0 if b_n else 0.0,
                "strong": (sum(1 for r in b_recs if r["is_strong"]) / b_denom) * 100.0 if b_n else 0.0,
                "possible_plus": (sum(1 for r in b_recs if r["is_possible_plus"]) / b_denom) * 100.0 if b_n else 0.0,
            }
        out["age_gap_buckets"] = buckets

    return out


def evaluate_retrieval_scope(
    gallery_by_id: Dict[str, List[Tuple[Optional[int], str]]],
    probes: Sequence[Dict[str, Any]],
    cache: Dict[str, Optional[np.ndarray]],
    quality_cache: Dict[str, float],
    engine_name: str,
    onnx_path: Optional[str] = None,
    bundle: Optional[CalibrationBundle] = None,
    bootstrap_n: int = 2000,
    seed: int = 0,
    near_dup_threshold: float = NEAR_DUP_COS_THRESHOLD,
    compute_age_buckets: bool = False,
    sha256_cache: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Evaluates one 1:N retrieval scope through FaceStore(":memory:") in both 'raw' and
    (if bundle is provided) 'calibrated' modes, applying path, SHA-256, near-duplicate,
    and failed-detection guards.

    Returns:
        summary: JSON-serializable scope summary with 'cross_id_byte_collisions' and 'modes'
        probe_records_by_mode: {mode: {probe_key: {"hit_1", "is_exact_dup", "is_near_dup"}}}
    """
    hash_memo = sha256_cache if sha256_cache is not None else {}

    def _get_sha(path: str) -> str:
        norm_p = os.path.normcase(os.path.abspath(path))
        digest = hash_memo.get(norm_p)
        if digest is None:
            digest = compute_file_sha256(path)
            hash_memo[norm_p] = digest
        return digest

    # 1. Path overlap guard across the entire scope
    gallery_paths_norm: Set[str] = set()
    for items in gallery_by_id.values():
        for _, path in items:
            gallery_paths_norm.add(os.path.normcase(os.path.abspath(path)))
    for pr in probes:
        p_norm = os.path.normcase(os.path.abspath(pr["path"]))
        if p_norm in gallery_paths_norm:
            raise GuardError(f"Path overlap guard violated: probe path '{pr['path']}' is also in gallery")

    # 2. SHA-256 exact-duplicate & cross-ID collision checks
    gal_hashes_by_id: Dict[str, Set[str]] = defaultdict(set)
    hash_to_subjects: Dict[str, Set[str]] = defaultdict(set)
    hash_to_paths: Dict[str, Set[str]] = defaultdict(set)
    max_gal_age_by_id: Dict[str, Optional[int]] = {}
    valid_gal_embs_by_id: Dict[str, List[np.ndarray]] = defaultdict(list)

    for subj_id, items in gallery_by_id.items():
        ages = [age for age, _ in items if age is not None]
        max_gal_age_by_id[subj_id] = max(ages) if ages else None
        for _, path in items:
            dig = _get_sha(path)
            gal_hashes_by_id[subj_id].add(dig)
            hash_to_subjects[dig].add(str(subj_id))
            hash_to_paths[dig].add(path)
            vec = _safe_unit_vec(cache.get(path))
            if vec is not None:
                valid_gal_embs_by_id[subj_id].append(vec)

    probe_meta: List[Dict[str, Any]] = []
    n_exact_dup_excluded = 0
    n_near_dup_excluded = 0

    for pr in probes:
        subj_id = str(pr["subject_id"])
        p_path = pr["path"]
        p_dig = _get_sha(p_path)
        hash_to_subjects[p_dig].add(subj_id)
        hash_to_paths[p_dig].add(p_path)

        is_exact_dup = p_dig in gal_hashes_by_id.get(subj_id, set())
        if is_exact_dup:
            n_exact_dup_excluded += 1

        p_vec = _safe_unit_vec(cache.get(p_path))
        is_near_dup = False
        if not is_exact_dup and p_vec is not None and valid_gal_embs_by_id.get(subj_id):
            max_cos = max(float(np.dot(p_vec, g_vec)) for g_vec in valid_gal_embs_by_id[subj_id])
            if max_cos >= near_dup_threshold:
                is_near_dup = True
                n_near_dup_excluded += 1

        age_bucket = None
        if compute_age_buckets and pr.get("age") is not None and max_gal_age_by_id.get(subj_id) is not None:
            gap = int(pr["age"]) - int(max_gal_age_by_id[subj_id])  # type: ignore[arg-type]
            if gap < 10:
                age_bucket = "<10"
            elif gap < 20:
                age_bucket = "10-19"
            else:
                age_bucket = ">=20"

        probe_meta.append({
            "probe": pr,
            "probe_key": _make_probe_key(subj_id, p_path),
            "subject_id": subj_id,
            "path": p_path,
            "emb": p_vec,
            "is_exact_dup": is_exact_dup,
            "is_near_dup": is_near_dup,
            "failed_probe_detect": p_vec is None,
            "age_bucket": age_bucket,
        })

    colliding_hashes = sorted(d for d, sids in hash_to_subjects.items() if len(sids) > 1)
    cross_id_collisions = {
        "count": len(colliding_hashes),
        "samples": [
            {
                "sha256": d,
                "subject_ids": sorted(hash_to_subjects[d])[:5],
                "sample_paths": sorted(hash_to_paths[d])[:5],
            }
            for d in colliding_hashes[:10]
        ],
    }

    gallery_ids_total = len(gallery_by_id)
    base_model = f"{engine_name}__{sanitize_onnx_stem(onnx_path)}" if onnx_path else engine_name
    modes_out: Dict[str, Any] = {}
    probe_records_by_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for mode in ("raw", "calibrated"):
        if mode == "calibrated" and bundle is None:
            modes_out["calibrated"] = {"skipped": True, "skip_reason": "No calibration bundle available"}
            continue

        store_model = f"{base_model}__raw" if mode == "raw" else f"{base_model}__cal"
        store = FaceStore(":memory:")
        try:
            if mode == "calibrated" and bundle is not None:
                store.set_calibration_bundle(store_model, bundle)

            pid_map: Dict[str, int] = {}
            for subj_id in sorted(gallery_by_id.keys()):
                pid = store.add_person(str(subj_id))
                added = 0
                for age, path in gallery_by_id[subj_id]:
                    vec = _safe_unit_vec(cache.get(path))
                    if vec is None:
                        continue
                    q_val = _clamp_quality(quality_cache.get(path, 1.0))
                    store.add_photo(
                        pid,
                        vec,
                        photo_path=path,
                        age=age,
                        model=store_model,
                        quality=q_val,
                    )
                    added += 1
                if added > 0:
                    pid_map[subj_id] = pid
                else:
                    store.delete_person(pid)

            if mode == "raw" and store.calibrated_ready(store_model):
                raise GuardError(f"Raw retrieval mode unexpectedly calibrated for '{store_model}'")
            if mode == "calibrated" and not store.calibrated_ready(store_model):
                raise GuardError(f"Calibrated retrieval mode unexpectedly uncalibrated for '{store_model}'")

            gallery_ids_enrolled = len(pid_map)
            evaluated_records: List[Dict[str, Any]] = []
            mode_probe_map: Dict[str, Dict[str, Any]] = {}

            for item in probe_meta:
                pkey = item["probe_key"]
                if item["is_exact_dup"]:
                    mode_probe_map[pkey] = {
                        "hit_1": False,
                        "is_exact_dup": True,
                        "is_near_dup": False,
                    }
                    continue

                emb_p = item["emb"]
                target_pid = pid_map.get(item["subject_id"])
                if emb_p is None or target_pid is None:
                    hit_1 = False
                    hit_5 = False
                    hit_10 = False
                    is_strong = False
                    is_possible_plus = False
                else:
                    q_probe = _clamp_quality(quality_cache.get(item["path"], 1.0))
                    hits = store.search(
                        emb_p,
                        k=10,
                        min_sim=-1.0,
                        model=store_model,
                        probe_quality=q_probe,
                        age_gap=None,
                    )
                    hit_pids = [h["person_id"] for h in hits]
                    hit_1 = bool(hit_pids and hit_pids[0] == target_pid)
                    hit_5 = bool(target_pid in hit_pids[:5])
                    hit_10 = bool(target_pid in hit_pids[:10])
                    target_hit = next((h for h in hits if h["person_id"] == target_pid), None)
                    is_strong = bool(target_hit is not None and target_hit["tier"] == "STRONG")
                    is_possible_plus = bool(
                        target_hit is not None and target_hit["tier"] in ("STRONG", "possible")
                    )

                rec = {
                    "subject_id": item["subject_id"],
                    "probe_key": pkey,
                    "is_near_dup": item["is_near_dup"],
                    "failed_probe_detect": item["failed_probe_detect"],
                    "age_bucket": item["age_bucket"],
                    "hit_1": hit_1,
                    "hit_5": hit_5,
                    "hit_10": hit_10,
                    "is_strong": is_strong,
                    "is_possible_plus": is_possible_plus,
                }
                evaluated_records.append(rec)
                mode_probe_map[pkey] = {
                    "hit_1": hit_1,
                    "is_exact_dup": False,
                    "is_near_dup": item["is_near_dup"],
                }

            clean_records = [r for r in evaluated_records if not r["is_near_dup"]]
            clean_view = _summarize_retrieval_view(
                clean_records,
                gallery_ids_total=gallery_ids_total,
                gallery_ids_enrolled=gallery_ids_enrolled,
                n_exact_dup_excluded=n_exact_dup_excluded,
                n_near_dup_excluded=n_near_dup_excluded,
                bootstrap_n=bootstrap_n,
                seed=seed,
                compute_age_buckets=compute_age_buckets,
            )
            near_dup_view = _summarize_retrieval_view(
                evaluated_records,
                gallery_ids_total=gallery_ids_total,
                gallery_ids_enrolled=gallery_ids_enrolled,
                n_exact_dup_excluded=n_exact_dup_excluded,
                n_near_dup_excluded=0,
                bootstrap_n=bootstrap_n,
                seed=seed,
                compute_age_buckets=compute_age_buckets,
            )

            modes_out[mode] = {
                "skipped": False,
                "clean": clean_view,
                "with_near_duplicates": near_dup_view,
            }
            probe_records_by_mode[mode] = mode_probe_map
        finally:
            store.db.close()

    return {
        "gallery_ids_total": gallery_ids_total,
        "total_probes_raw": len(probes),
        "n_exact_dup_excluded": n_exact_dup_excluded,
        "n_near_dup_excluded": n_near_dup_excluded,
        "cross_id_byte_collisions": cross_id_collisions,
        "modes": modes_out,
    }, probe_records_by_mode


# ---------------------------------------------------------------------------
# Paired Exact McNemar Comparison Builder
# ---------------------------------------------------------------------------


def build_pairwise_comparisons(
    engines: Sequence[str],
    artifacts_by_engine: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Builds strict_compare.json pairwise comparisons across all >= 2 evaluated engines."""
    if len(engines) < 2:
        return {}

    retrieval_targets = (
        ("fgnet", "default"),
        ("caf", "fully_unseen"),
        ("caf", "model_strict"),
        ("ecaf_official", "model_strict"),
        ("ecaf_all", "fully_unseen"),
        ("ecaf_all", "model_strict"),
    )

    comparisons: Dict[str, Any] = {}
    for eng_a, eng_b in itertools.combinations(engines, 2):
        pair_key = f"{eng_a}_vs_{eng_b}"
        art_a = artifacts_by_engine[eng_a]
        art_b = artifacts_by_engine[eng_b]

        verif_cmp: Dict[str, Any] = {}
        for vsuite in ("agedb30_official", "calfw"):
            va = art_a["verification_oof"].get(vsuite)
            vb = art_b["verification_oof"].get(vsuite)
            if va is not None and vb is not None:
                verif_cmp[vsuite] = mcnemar_exact(va, vb)

        ret_cmp: Dict[str, Any] = {}
        for ds_key, scope_key in retrieval_targets:
            rec_a = art_a["retrieval_probes"].get((ds_key, scope_key))
            rec_b = art_b["retrieval_probes"].get((ds_key, scope_key))
            if not rec_a or not rec_b:
                continue

            mode_cmp: Dict[str, Any] = {}
            for mode in ("raw", "calibrated"):
                ma = rec_a.get(mode)
                mb = rec_b.get(mode)
                if ma is None or mb is None:
                    mode_cmp[mode] = {"skipped": True, "reason": f"Mode '{mode}' unavailable for one or both engines"}
                    continue

                common_keys = sorted(set(ma.keys()) & set(mb.keys()))
                clean_keys = [
                    k
                    for k in common_keys
                    if not ma[k]["is_exact_dup"]
                    and not ma[k]["is_near_dup"]
                    and not mb[k]["is_exact_dup"]
                    and not mb[k]["is_near_dup"]
                ]
                hits_a = [ma[k]["hit_1"] for k in clean_keys]
                hits_b = [mb[k]["hit_1"] for k in clean_keys]
                res = mcnemar_exact(hits_a, hits_b)
                res["n_shared_total"] = len(common_keys)
                res["n_excluded_dup_either"] = len(common_keys) - len(clean_keys)
                mode_cmp[mode] = res

            ret_cmp[f"{ds_key}:{scope_key}"] = mode_cmp

        comparisons[pair_key] = {
            "engine_a": eng_a,
            "engine_b": eng_b,
            "verification": verif_cmp,
            "retrieval": ret_cmp,
        }
    return comparisons


# ---------------------------------------------------------------------------
# Suite Runners & CLI Orchestration
# ---------------------------------------------------------------------------


def _run_engine_suites(
    engine_name: str,
    suites: Sequence[str],
    use_gpu: bool,
    onnx_path: Optional[str],
    verify_sample: int,
    bootstrap_n: int,
    seed: int,
    cache_dir: str,
    caf_bundle_data: Optional[Dict[str, Any]],
    ecaf_bundle_data: Optional[Dict[str, Any]],
    agedb_bundle_data: Optional[Tuple[List[Tuple[str, str, int, int]], Dict[str, tarfile.TarInfo], Dict[str, Any]]],
    calfw_bundle_data: Optional[Tuple[List[Tuple[str, str, int, int]], Dict[str, Any]]],
    fgnet_bundle_data: Optional[Tuple[Dict[str, List[Tuple[int, str]]], List[Dict[str, Any]]]],
    sha256_cache: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Executes all requested suites for a single engine and releases engine resources before returning."""
    use_tta = engine_name != "buffalo_l"
    bundle = resolve_calibration_bundle(engine_name, onnx_path=onnx_path)

    engine_holder: Dict[str, Any] = {"engine": None, "providers": {}}

    def get_engine() -> Any:
        if engine_holder["engine"] is None:
            eng = load_engine_lazy(engine_name, use_gpu=use_gpu, use_tta=use_tta, onnx_path=onnx_path)
            engine_holder["engine"] = eng
            engine_holder["providers"] = get_engine_providers(eng)
        return engine_holder["engine"]

    engine_summary: Dict[str, Any] = {
        "engine": engine_name,
        "use_tta": use_tta,
        "requested_gpu": use_gpu,
        "onnx_path": onnx_path,
        "calibration_bundle_loaded": bundle is not None,
        "verification": {},
        "retrieval": {},
        "cache_integrity": {},
    }
    artifacts = {
        "verification_oof": {},
        "retrieval_probes": {},
    }

    try:
        if "agedb30_official" in suites:
            assert agedb_bundle_data is not None
            raw_pairs, tar_map, proto_meta = agedb_bundle_data
            needed_keys = {k for a, b, _, _ in raw_pairs for k in (a, b)}
            cache_path = get_agedb_official_cache_path(cache_dir, engine_name, use_tta, onnx_path=onnx_path)
            with tarfile.open(proto_meta["tar_path"], "r:") as tf:
                cache, _ = ensure_strict_embeddings(
                    needed_keys,
                    cache_path,
                    get_engine,
                    tar_handle=tf,
                    tar_map=tar_map,
                )
                integrity = verify_cache_integrity(
                    needed_keys,
                    cache,
                    get_engine,
                    verify_sample=verify_sample,
                    seed=seed,
                    tar_handle=tf,
                    tar_map=tar_map,
                )
            ver_summary, oof_corr = evaluate_verification_pairs(raw_pairs, cache)
            ver_summary["protocol"] = proto_meta
            engine_summary["verification"]["agedb30_official"] = ver_summary
            engine_summary["cache_integrity"]["agedb30_official"] = integrity
            artifacts["verification_oof"]["agedb30_official"] = oof_corr

        if "calfw" in suites:
            assert calfw_bundle_data is not None
            raw_pairs, proto_meta = calfw_bundle_data
            needed_keys = {k for a, b, _, _ in raw_pairs for k in (a, b)}
            cache_path = get_cache_path(cache_dir, "calfw", engine_name, use_tta, onnx_path=onnx_path)
            cache, _ = ensure_strict_embeddings(needed_keys, cache_path, get_engine)
            integrity = verify_cache_integrity(
                needed_keys,
                cache,
                get_engine,
                verify_sample=verify_sample,
                seed=seed,
            )
            ver_summary, oof_corr = evaluate_verification_pairs(raw_pairs, cache)
            ver_summary["protocol"] = proto_meta
            engine_summary["verification"]["calfw"] = ver_summary
            engine_summary["cache_integrity"]["calfw"] = integrity
            artifacts["verification_oof"]["calfw"] = oof_corr

        if "fgnet" in suites:
            assert fgnet_bundle_data is not None
            fg_gal, fg_probes = fgnet_bundle_data
            validate_headline_scope_size("fgnet:default", fg_gal)
            needed_keys = [p for items in fg_gal.values() for _, p in items] + [pr["path"] for pr in fg_probes]
            cache_path = get_cache_path(cache_dir, "fgnet", engine_name, use_tta, onnx_path=onnx_path)
            cache, q_cache = ensure_strict_embeddings(needed_keys, cache_path, get_engine)
            integrity = verify_cache_integrity(
                needed_keys,
                cache,
                get_engine,
                verify_sample=verify_sample,
                seed=seed,
            )
            scope_res, probe_map = evaluate_retrieval_scope(
                fg_gal,
                fg_probes,
                cache,
                q_cache,
                engine_name=engine_name,
                onnx_path=onnx_path,
                bundle=bundle,
                bootstrap_n=bootstrap_n,
                seed=seed,
                compute_age_buckets=True,
                sha256_cache=sha256_cache,
            )
            engine_summary["retrieval"]["fgnet"] = {"scopes": {"default": scope_res}}
            engine_summary["cache_integrity"]["fgnet"] = integrity
            artifacts["retrieval_probes"][("fgnet", "default")] = probe_map

        if "caf" in suites:
            if should_skip_caf(onnx_path, engine_name=engine_name):
                engine_summary["retrieval"]["caf"] = {
                    "skipped": True,
                    "skip_reason": "CAF is in from-scratch Stage 2 training set",
                }
            else:
                assert caf_bundle_data is not None
                cache_path = get_cache_path(cache_dir, "caf", engine_name, use_tta, onnx_path=onnx_path)
                cache, q_cache = ensure_strict_embeddings(caf_bundle_data["all_paths"], cache_path, get_engine)
                integrity = verify_cache_integrity(
                    caf_bundle_data["all_paths"],
                    cache,
                    get_engine,
                    verify_sample=verify_sample,
                    seed=seed,
                )
                caf_scopes_out: Dict[str, Any] = {}
                for scope_name, sc in caf_bundle_data["scopes"].items():
                    scope_res, probe_map = evaluate_retrieval_scope(
                        sc["gallery_by_id"],
                        sc["probes"],
                        cache,
                        q_cache,
                        engine_name=engine_name,
                        onnx_path=onnx_path,
                        bundle=bundle,
                        bootstrap_n=bootstrap_n,
                        seed=seed,
                        compute_age_buckets=False,
                        sha256_cache=sha256_cache,
                    )
                    scope_res["is_headline"] = sc["is_headline"]
                    scope_res["filtering"] = sc["filtering"]
                    caf_scopes_out[scope_name] = scope_res
                    artifacts["retrieval_probes"][("caf", scope_name)] = probe_map

                engine_summary["retrieval"]["caf"] = {
                    "skipped": False,
                    "id_filtering": caf_bundle_data["id_filtering"],
                    "scopes": caf_scopes_out,
                }
                engine_summary["cache_integrity"]["caf"] = integrity

        if "ecaf" in suites:
            assert ecaf_bundle_data is not None
            cache_path = get_cache_path(cache_dir, "ecaf", engine_name, use_tta, onnx_path=onnx_path)
            cache, q_cache = ensure_strict_embeddings(ecaf_bundle_data["all_paths"], cache_path, get_engine)
            integrity = verify_cache_integrity(
                ecaf_bundle_data["all_paths"],
                cache,
                get_engine,
                verify_sample=verify_sample,
                seed=seed,
            )
            engine_summary["cache_integrity"]["ecaf"] = integrity

            for ds_key in ("ecaf_official", "ecaf_all"):
                sub = ecaf_bundle_data[ds_key]
                sub_scopes_out: Dict[str, Any] = {}
                for scope_name, sc in sub["scopes"].items():
                    scope_res, probe_map = evaluate_retrieval_scope(
                        sc["gallery_by_id"],
                        sc["probes"],
                        cache,
                        q_cache,
                        engine_name=engine_name,
                        onnx_path=onnx_path,
                        bundle=bundle,
                        bootstrap_n=bootstrap_n,
                        seed=seed,
                        compute_age_buckets=False,
                        sha256_cache=sha256_cache,
                    )
                    scope_res["is_headline"] = sc["is_headline"]
                    scope_res["filtering"] = sc["filtering"]
                    for meta_k in ("known_overlap_count", "known_overlap_note", "small_gallery_warning"):
                        if meta_k in sc:
                            scope_res[meta_k] = sc[meta_k]
                    sub_scopes_out[scope_name] = scope_res
                    artifacts["retrieval_probes"][(ds_key, scope_name)] = probe_map

                engine_summary["retrieval"][ds_key] = {
                    "id_filtering": sub["id_filtering"],
                    "scopes": sub_scopes_out,
                }
    finally:
        if engine_holder["engine"] is not None:
            engine_summary["actual_gpu"] = bool(getattr(engine_holder["engine"], "use_gpu", False))
            engine_summary["providers"] = engine_holder["providers"]
            del engine_holder["engine"]
            engine_holder["engine"] = None
            gc.collect()
        else:
            engine_summary["actual_gpu"] = False
            engine_summary["providers"] = {}

    return engine_summary, artifacts


def print_strict_summary(
    engines: Sequence[str],
    summaries: Dict[str, Any],
    comparisons: Dict[str, Any],
) -> None:
    """Prints plain-language summary tables for verification, retrieval, and McNemar tests."""
    print("\n" + "=" * 112)
    print("  AI-MIDS STRICT EVALUATION SUMMARY")
    print("=" * 112)

    for eng in engines:
        s = summaries[eng]
        print(f"\n  Engine: {eng} (Flip-TTA={'ON' if s['use_tta'] else 'OFF'}, GPU={s.get('actual_gpu')})")

        if s["verification"]:
            print("  [1:1 Out-of-Fold Verification]")
            print(
                f"    {'Suite':<18} | {'Pairs':>5} | {'Fail':>4} | {'10-Fold Acc':>16} | "
                f"{'Pooled 95% CI':>17} | {'TAR@FAR1%':>10} | {'Ach FAR':>8}"
            )
            print("    " + "-" * 94)
            for suite_name, v in s["verification"].items():
                ci = v["pooled_acc_ci95"]
                print(
                    f"    {suite_name:<18} | {v['total_pairs']:>5} | {v['n_failed_pairs']:>4} | "
                    f"{v['mean_acc']:>6.2f}% ± {v['std_acc']:>4.2f}% | "
                    f"[{ci[0]:>5.2f}%, {ci[1]:>5.2f}%] | "
                    f"{v['mean_tar_far1']:>9.2f}% | {v['mean_achieved_far1']:>7.2f}%"
                )

        if s["retrieval"]:
            print("  [1:N Child -> Adult Retrieval (clean view)]")
            print(
                f"    {'Dataset:Scope':<28} | {'Mode':<10} | {'IDs':>7} | {'Probes':>6} | "
                f"{'Ex/Nd':>7} | {'Rank-1':>8} | {'Top-10':>8} | {'STRONG':>8} | {'Poss+':>7}"
            )
            print("    " + "-" * 106)
            for ds_key, ds_val in s["retrieval"].items():
                if ds_val.get("skipped"):
                    print(f"    {ds_key:<28} | SKIPPED: {ds_val.get('skip_reason')}")
                    continue
                for scope_key, sc_val in ds_val["scopes"].items():
                    for mode_key, m_val in sc_val["modes"].items():
                        if m_val.get("skipped"):
                            continue
                        cl = m_val["clean"]
                        id_str = f"{cl['gallery_ids_enrolled']}/{cl['gallery_ids_total']}"
                        dup_str = f"{cl['n_exact_dup_excluded']}/{cl['n_near_dup_excluded']}"
                        print(
                            f"    {f'{ds_key}:{scope_key}':<28} | {mode_key:<10} | {id_str:>7} | "
                            f"{cl['probes']:>6} | {dup_str:>7} | {cl['rank1']:>7.2f}% | "
                            f"{cl['top10']:>7.2f}% | {cl['strong']:>7.2f}% | {cl['possible_plus']:>6.2f}%"
                        )

    if comparisons:
        print("\n  [Paired Exact McNemar Comparisons]")
        for pair_key, cmp_data in comparisons.items():
            print(f"    Pair: {pair_key}")
            for v_name, m in cmp_data.get("verification", {}).items():
                print(
                    f"      verif:{v_name:<18} n={m['n']:<4} "
                    f"{cmp_data['engine_a']}={m['acc_a']:.2f}% vs {cmp_data['engine_b']}={m['acc_b']:.2f}% "
                    f"(diff={m['diff_pct']:+.2f}%, b={m['b']}, c={m['c']}, p={m['p_value']:.4g}, sig={m['significant_p05']})"
                )
            for r_name, modes_d in cmp_data.get("retrieval", {}).items():
                for mode_name, m in modes_d.items():
                    if m.get("skipped"):
                        continue
                    print(
                        f"      {r_name}:{mode_name:<10} n={m['n']:<4} "
                        f"{cmp_data['engine_a']}={m['acc_a']:.2f}% vs {cmp_data['engine_b']}={m['acc_b']:.2f}% "
                        f"(diff={m['diff_pct']:+.2f}%, b={m['b']}, c={m['c']}, p={m['p_value']:.4g}, sig={m['significant_p05']})"
                    )
    print("=" * 112 + "\n")


def _parse_csv_arg(raw: str, allowed: Sequence[str], label: str) -> List[str]:
    items = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    if not items:
        raise ValueError(f"--{label} cannot be empty")
    deduped: List[str] = []
    for item in items:
        if item not in allowed:
            raise ValueError(f"Unsupported {label} '{item}'. Allowed: {', '.join(allowed)}")
        if item not in deduped:
            deduped.append(item)
    return deduped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-MIDS Strict Full-Protocol Evaluation Suite (No --limit)"
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="aimids_crossage,antelopev2",
        help="Comma-separated engines (default: aimids_crossage,antelopev2)",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="agedb30_official,calfw,fgnet,caf,ecaf",
        help="Comma-separated suites (default: agedb30_official,calfw,fgnet,caf,ecaf)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Optional explicit .onnx path (for aimids_ft or aimids_crossage)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution instead of CUDA GPU",
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=32,
        help="Number of cached embeddings per suite/engine to re-embed live (default: 32)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Identity-cluster bootstrap replicates for 95%% retrieval CIs (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for integrity sampling and bootstrap (default: 0)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="bench_results/strict",
        help="Output directory for strict JSON reports (default: bench_results/strict relative to Arc face/)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        engines = _parse_csv_arg(args.engines, ALLOWED_ENGINES, "engines")
        suites = _parse_csv_arg(args.suites, ALLOWED_SUITES, "suites")
        onnx_path = resolve_custom_onnx(args.onnx)
        if onnx_path:
            for eng in engines:
                if eng not in ("aimids_ft", "aimids_crossage"):
                    raise ValueError(
                        f"--onnx is only supported with aimids_ft or aimids_crossage, got '{eng}'"
                    )

        out_dir = args.out if os.path.isabs(args.out) else os.path.abspath(os.path.join(ARC_DIR, args.out))
        use_gpu = not args.cpu
        t0 = time.time()

        # Pre-load & validate all protocols + compute protocol SHA-256 hashes before running engines
        protocol_hashes: Dict[str, Any] = {}
        agedb_bundle_data = None
        calfw_bundle_data = None
        fgnet_bundle_data = None
        caf_bundle_data = None
        ecaf_bundle_data = None
        sha256_cache: Dict[str, str] = {}

        if "agedb30_official" in suites:
            agedb_bundle_data = load_agedb30_official()
            raw_pairs, _, proto_meta = agedb_bundle_data
            protocol_hashes["agedb30_official"] = {
                "mat_sha256": sha256_file_if_exists(proto_meta["mat_path"]),
                "tar_sha256": sha256_file_if_exists(proto_meta["tar_path"]),
                "pairs_manifest_sha256": compute_manifest_sha256(
                    f"{f}|{lbl}|{ka}|{kb}" for ka, kb, lbl, f in raw_pairs
                ),
            }

        if "calfw" in suites:
            calfw_bundle_data = load_calfw_official()
            c_pairs, c_meta = calfw_bundle_data
            protocol_hashes["calfw"] = {
                "pairs_calfw_sha256": sha256_file_if_exists(os.path.join(c_meta["root"], "pairs_CALFW.txt")),
                "pairs_manifest_sha256": compute_manifest_sha256(
                    f"{f}|{lbl}|{os.path.basename(ka)}|{os.path.basename(kb)}" for ka, kb, lbl, f in c_pairs
                ),
            }

        if "fgnet" in suites:
            fgnet_bundle_data = load_fgnet_gallery(root=DEFAULT_FGNET_ROOT)
            fg_gal, fg_pr = fgnet_bundle_data
            validate_headline_scope_size("fgnet:default", fg_gal)
            protocol_hashes["fgnet"] = {
                "manifest_sha256": compute_manifest_sha256(
                    [f"G|{sid}|{age}|{os.path.basename(p)}" for sid, items in fg_gal.items() for age, p in items]
                    + [f"P|{pr['subject_id']}|{pr['age']}|{os.path.basename(pr['path'])}" for pr in fg_pr]
                ),
            }

        if "caf" in suites or "ecaf" in suites:
            cacd_ids, cacd_meta = load_cacd_finetune_identities()
            cohort_ids = load_lag_cohort_identities()
            known_ids = collect_known_dataset_identities(include_demo_dbs=True)
            cacd_zip_sha = sha256_file_if_exists(cacd_meta["zip_path"])
        else:
            cacd_ids, cacd_meta, cohort_ids, known_ids, cacd_zip_sha = None, None, None, None, None

        if "caf" in suites:
            if all(should_skip_caf(onnx_path, engine_name=e) for e in engines):
                protocol_hashes["caf"] = {"skipped": True, "skip_reason": "CAF is in from-scratch Stage 2 training set"}
            else:
                caf_bundle_data = build_caf_dataset(
                    root=DEFAULT_CAF_ROOT,
                    cohort_ids=cohort_ids,
                    cacd_ids=cacd_ids,
                    known_ids=known_ids,
                )
                protocol_hashes["caf"] = {
                    "cacd_zip_sha256": cacd_zip_sha,
                    "cacd_manifest_meta": cacd_meta,
                    "manifest_sha256": compute_manifest_sha256(
                        f"{os.path.relpath(p, DEFAULT_CAF_ROOT)}" for p in caf_bundle_data["all_paths"]
                    ),
                }

        if "ecaf" in suites:
            ecaf_bundle_data = build_ecaf_datasets(
                root=DEFAULT_ECAF_ROOT,
                cohort_ids=cohort_ids,
                cacd_ids=cacd_ids,
                known_ids=known_ids,
            )
            name_list_path = next(
                (
                    p
                    for p in (
                        os.path.join(DEFAULT_ECAF_ROOT, "image_data", "name_list.json"),
                        os.path.join(DEFAULT_ECAF_ROOT, "name_list.json"),
                    )
                    if os.path.isfile(p)
                ),
                None,
            )
            protocol_hashes["ecaf"] = {
                "name_list_sha256": sha256_file_if_exists(name_list_path),
                "cacd_zip_sha256": cacd_zip_sha,
                "cacd_manifest_meta": cacd_meta,
                "manifest_sha256": compute_manifest_sha256(
                    f"{os.path.relpath(p, DEFAULT_ECAF_ROOT)}" for p in ecaf_bundle_data["all_paths"]
                ),
            }

        summaries: Dict[str, Any] = {}
        artifacts_by_engine: Dict[str, Any] = {}
        providers_by_engine: Dict[str, Any] = {}

        for eng in engines:
            eng_summary, eng_artifacts = _run_engine_suites(
                engine_name=eng,
                suites=suites,
                use_gpu=use_gpu,
                onnx_path=onnx_path,
                verify_sample=args.verify_sample,
                bootstrap_n=args.bootstrap,
                seed=args.seed,
                cache_dir=ARC_DIR,
                caf_bundle_data=caf_bundle_data,
                ecaf_bundle_data=ecaf_bundle_data,
                agedb_bundle_data=agedb_bundle_data,
                calfw_bundle_data=calfw_bundle_data,
                fgnet_bundle_data=fgnet_bundle_data,
                sha256_cache=sha256_cache,
            )
            summaries[eng] = eng_summary
            artifacts_by_engine[eng] = eng_artifacts
            providers_by_engine[eng] = eng_summary.get("providers", {})

        comparisons = build_pairwise_comparisons(engines, artifacts_by_engine)
        provenance = collect_provenance(
            args=args,
            engines=engines,
            onnx_path=onnx_path,
            protocol_hashes=protocol_hashes,
            engine_providers=providers_by_engine,
        )
        elapsed_sec = time.time() - t0

        os.makedirs(out_dir, exist_ok=True)
        onnx_tag = f"_{sanitize_onnx_stem(onnx_path)}" if onnx_path else ""
        written_files: List[str] = []

        for eng in engines:
            eng_file = os.path.join(out_dir, f"strict_{eng}{onnx_tag}.json")
            with open(eng_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "provenance": provenance,
                        "runtime_sec": elapsed_sec,
                        "result": summaries[eng],
                    },
                    f,
                    indent=2,
                )
            written_files.append(eng_file)

        summary_file = os.path.join(out_dir, "strict_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provenance": provenance,
                    "runtime_sec": elapsed_sec,
                    "engines": summaries,
                },
                f,
                indent=2,
            )
        written_files.append(summary_file)

        if len(engines) >= 2:
            compare_file = os.path.join(out_dir, "strict_compare.json")
            with open(compare_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "provenance": provenance,
                        "comparisons": comparisons,
                    },
                    f,
                    indent=2,
                )
            written_files.append(compare_file)

        print_strict_summary(engines, summaries, comparisons)
        print(f">> Strict evaluation completed in {elapsed_sec:.1f}s ({elapsed_sec / 60:.2f} min).")
        for wf in written_files:
            print(f"   Wrote: {wf}")
        return 0
    except GuardError as exc:
        print(f"[GUARD FAILURE] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
