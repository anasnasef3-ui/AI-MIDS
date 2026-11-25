"""
full_pipeline_eval.py - Unified 3-Stage End-to-End Evaluation Suite for AI-MIDS.

Evaluates 3 recognition engines (`aimids_crossage`, `antelopev2`, `fromscratchv2`)
through the full production pipeline (excluding ID verification / LivenessChecker):
  - Shared external 1,200-face LAG cohort & matched 4-layer calibration fitting
    (`age_gap=None`, identical CALFW folds 0..1 QMF trials & fold 2..3 350 non-mates).
  - Stage 1: Full Missing-Persons FastAPI HTTP integration smoke test per engine.
  - Stage 2: Unified all-5-dataset combined FaceStore gallery evaluation (FG-NET + ECAF
    + CAF + AgeDB + CALFW + 200 unseen strangers) in both `matched_calibrated` and
    `production_bundle` modes, with `fair_4_datasets_no_caf` and `all_5_datasets` views.
  - Stage 3: Live HTTP `POST /api/search` parity verification against in-process
    `FaceStore` on a unified all-5-dataset SQLite DB across 30 single-face probes.
"""

from __future__ import annotations

import os

# Must be set before importing numpy / faiss / onnxruntime to keep Windows virtual commit bounded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARC_DIR = os.path.join(HERE, "Arc face")
for _p in (HERE, ARC_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, HERE)
sys.path.insert(0, ARC_DIR)

if hasattr(sys.stdout, "reconfigure") and getattr(sys.stdout, "encoding", None):
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import gc
import itertools
import json
import mimetypes
import socket
import sqlite3
import subprocess
import tarfile
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from bench_datasets import (
    ASSETS_DIR,
    collect_known_dataset_identities,
    load_cache,
    load_fgnet_gallery,
    load_lag_cohort_identities,
    normalize_identity,
    parse_agedb_images,
    parse_caf_images,
    parse_ecaf_images,
)
from calibration import (
    CalibrationBundle,
    CoveCalibrator,
    ExternalCohortASNorm,
    OpenSetCalibrator,
    QMFCalibrator,
    compute_age_subspace_basis,
    project_subspace,
    resolve_bundle_path,
)
from calibrate_thresholds import (
    LAG_CACHE_PATH,
    LAG_TAR_PATH,
    LFW_CONTAMINATED_NONMATES,
    _CALFW_RE,
    _simulate_nonmate_max_scores,
)
from store import FaceStore
from strict_eval import (
    CACHE_VERIFY_MIN_COS,
    DEFAULT_CAF_ROOT,
    DEFAULT_CALFW_ROOT,
    DEFAULT_ECAF_ROOT,
    DEFAULT_FGNET_ROOT,
    GuardError,
    NEAR_DUP_COS_THRESHOLD,
    _clamp_quality,
    _make_probe_key,
    _safe_unit_vec,
    cluster_bootstrap_ci,
    compute_file_sha256,
    compute_manifest_sha256,
    ensure_strict_embeddings,
    get_cache_path,
    get_engine_providers,
    load_cacd_finetune_identities,
    load_calfw_official,
    load_engine_lazy,
    mcnemar_exact,
    sanitize_onnx_stem,
    sha256_file_if_exists,
    should_skip_caf,
    verify_cache_integrity,
)


SCRATCH_V2_ONNX_PATH = os.path.join(ARC_DIR, "pretrained", "aimids_scratch_ir50_crossage.onnx")
LAG_SCRATCH_CACHE_PATH = os.path.join(ARC_DIR, "pretrained", "lag_cohort_scratch_v2.npz")
DEFAULT_AGEDB_IMAGES_DIR = os.path.join(ASSETS_DIR, "AgeDB", "images")

ENGINE_SPECS: Dict[str, Dict[str, Any]] = {
    "aimids_crossage": {
        "engine_name": "aimids_crossage",
        "store_model": "aimids_crossage",
        "onnx_path": None,
        "dim": 1024,
    },
    "antelopev2": {
        "engine_name": "antelopev2",
        "store_model": "antelopev2",
        "onnx_path": None,
        "dim": 512,
    },
    "fromscratchv2": {
        "engine_name": "aimids_ft",
        "store_model": "aimids_ft",
        "onnx_path": SCRATCH_V2_ONNX_PATH,
        "dim": 512,
    },
}

DEFAULT_ENGINES = ("aimids_crossage", "antelopev2", "fromscratchv2")
DATASET_ORDER = ("fgnet", "ecaf", "caf", "agedb", "calfw")
STAGE2_CALIB_MODES = ("matched_calibrated", "production_bundle")
STAGE3_BUCKETS = ("fgnet", "ecaf", "agedb", "calfw", "stranger")


def parse_engine_keys(raw: str) -> List[str]:
    """Parses and validates --engines CSV list preserving user order."""
    items = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if not items:
        raise ValueError("--engines cannot be empty")
    deduped: List[str] = []
    for item in items:
        if item not in ENGINE_SPECS:
            raise ValueError(f"Unsupported engine '{item}'. Allowed: {', '.join(ENGINE_SPECS.keys())}")
        if item not in deduped:
            deduped.append(item)
    return deduped


def _calfw_norm_from_path(path: str) -> str:
    base = os.path.basename(str(path))
    m = _CALFW_RE.match(base)
    raw = m.group(1) if m else os.path.splitext(base)[0]
    return normalize_identity(raw)


def _calfw_sort_key(path: str) -> Tuple[int, str, str]:
    base = os.path.basename(str(path))
    m = _CALFW_RE.match(base)
    idx = int(m.group(2)) if m else 0
    return (idx, base.lower(), str(path))


def _resolve_prod_bundle_path(spec: Dict[str, Any]) -> str:
    """Resolves the on-disk production CalibrationBundle path without mutating global env."""
    onnx_path = spec.get("onnx_path")
    if spec["store_model"] == "aimids_ft" and onnx_path:
        old_env = os.environ.get("AIMIDS_ONNX_PATH")
        try:
            os.environ["AIMIDS_ONNX_PATH"] = onnx_path
            return resolve_bundle_path("aimids_ft")
        finally:
            if old_env is None:
                os.environ.pop("AIMIDS_ONNX_PATH", None)
            else:
                os.environ["AIMIDS_ONNX_PATH"] = old_env
    return resolve_bundle_path(spec["store_model"])


# ---------------------------------------------------------------------------
# External LAG Cohort Loading & Scratch v2 Cohort Extraction
# ---------------------------------------------------------------------------


def _select_lag_tar_members(
    tf: tarfile.TarFile,
    target_idents: np.ndarray,
    target_young: np.ndarray,
    max_faces: int = 1200,
    seed: int = 42,
) -> List[Tuple[str, bool, tarfile.TarInfo]]:
    """
    Reproduces calibrate_thresholds._extract_lag_embeddings_onnx deterministic member
    selection (seed=42), verifying exact identity & young-flag alignment with lag_cohort_cache.npz.
    Falls back to matching the saved (ident, is_young) multiset if demo DBs ever diverge.
    """
    known = collect_known_dataset_identities()
    rng = np.random.default_rng(seed)

    all_by_ident: Dict[str, Dict[str, List[tarfile.TarInfo]]] = defaultdict(lambda: {"y": [], "a": []})
    filtered_by_ident: Dict[str, Dict[str, List[tarfile.TarInfo]]] = defaultdict(lambda: {"y": [], "a": []})

    for m in tf.getmembers():
        if not m.isfile() or not m.name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        parts = m.name.split("/")
        if len(parts) < 3:
            continue
        ident = parts[1]
        bucket = "y" if "/y/" in m.name else "a"
        all_by_ident[ident][bucket].append(m)
        if normalize_identity(ident) not in known:
            filtered_by_ident[ident][bucket].append(m)

    disjoint_idents = sorted(filtered_by_ident.keys())
    rng.shuffle(disjoint_idents)

    selected: List[Tuple[str, bool, tarfile.TarInfo]] = []
    target_half = max_faces // 2
    young_count = 0
    adult_count = 0

    for ident in disjoint_idents:
        ys = sorted(filtered_by_ident[ident]["y"], key=lambda x: x.name)
        ads = sorted(filtered_by_ident[ident]["a"], key=lambda x: x.name)
        if ys and young_count < target_half:
            selected.append((ident, True, ys[0]))
            young_count += 1
        if ads and adult_count < target_half:
            selected.append((ident, False, ads[0]))
            adult_count += 1
        if len(selected) >= max_faces:
            break

    if len(selected) < max_faces:
        used_names = {s[2].name for s in selected}
        for ident in disjoint_idents:
            for bucket, is_y in (("y", True), ("a", False)):
                for m in sorted(filtered_by_ident[ident][bucket], key=lambda x: x.name):
                    if m.name not in used_names:
                        selected.append((ident, is_y, m))
                        used_names.add(m.name)
                        if len(selected) >= max_faces:
                            break
                if len(selected) >= max_faces:
                    break
            if len(selected) >= max_faces:
                break

    selected.sort(key=lambda s: s[2].offset_data)
    rep_idents = np.array([s[0] for s in selected])
    rep_young = np.array([s[1] for s in selected], dtype=bool)
    if np.array_equal(rep_idents, target_idents) and np.array_equal(rep_young, target_young):
        return selected

    # Fallback: reconstruct exact members matching the saved (ident, is_young) sequence in sorted order
    used_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    fallback: List[Tuple[str, bool, tarfile.TarInfo]] = []
    for ident_raw, is_y_raw in zip(target_idents.tolist(), target_young.tolist()):
        ident = str(ident_raw)
        is_y = bool(is_y_raw)
        bucket = "y" if is_y else "a"
        cands = sorted(all_by_ident[ident][bucket], key=lambda x: x.name)
        idx = used_counts[(ident, bucket)]
        if idx >= len(cands):
            raise GuardError(f"LAG tar missing member for ({ident}, {bucket}, idx={idx})")
        fallback.append((ident, is_y, cands[idx]))
        used_counts[(ident, bucket)] = idx + 1

    fallback.sort(key=lambda s: s[2].offset_data)
    fb_idents = np.array([s[0] for s in fallback])
    fb_young = np.array([s[1] for s in fallback], dtype=bool)
    if not (np.array_equal(fb_idents, target_idents) and np.array_equal(fb_young, target_young)):
        raise GuardError("LAG cohort member reconstruction mismatch against lag_cohort_cache.npz")
    return fallback


def ensure_lag_cohort_for_engines(
    engine_keys: Sequence[str],
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Loads shared 1,200-face external LAG cohort embeddings without modifying lag_cohort_cache.npz.
    Extracts and caches lag_cohort_scratch_v2.npz via FaceEngineAdaFaceFT preprocessing if needed.
    """
    if not os.path.isfile(LAG_CACHE_PATH):
        raise GuardError(f"External LAG cohort cache not found: {LAG_CACHE_PATH}")

    with np.load(LAG_CACHE_PATH, allow_pickle=False) as lag_d:
        target_idents = lag_d["identities"].copy()
        target_young = lag_d["is_young"].copy()
        cohorts: Dict[str, np.ndarray] = {}
        for k in ("antelopev2", "aimids_crossage"):
            if k in lag_d:
                cohorts[k] = lag_d[k].astype(np.float32)

    if "fromscratchv2" in engine_keys:
        loaded_scratch = False
        if os.path.isfile(LAG_SCRATCH_CACHE_PATH):
            with np.load(LAG_SCRATCH_CACHE_PATH, allow_pickle=False) as sd:
                if (
                    "fromscratchv2" in sd
                    and "identities" in sd
                    and "is_young" in sd
                    and np.array_equal(sd["identities"], target_idents)
                    and np.array_equal(sd["is_young"], target_young)
                ):
                    cohorts["fromscratchv2"] = sd["fromscratchv2"].astype(np.float32)
                    loaded_scratch = True
                    print(
                        f">> Loaded cached LAG cohort for fromscratchv2 ({len(cohorts['fromscratchv2'])} faces) "
                        f"from {os.path.basename(LAG_SCRATCH_CACHE_PATH)}"
                    )

        if not loaded_scratch:
            if not os.path.isfile(LAG_TAR_PATH):
                raise GuardError(f"LAG tarball not found for fromscratchv2 cohort extraction: {LAG_TAR_PATH}")
            onnx_path = ENGINE_SPECS["fromscratchv2"]["onnx_path"]
            if not os.path.isfile(onnx_path):
                raise GuardError(f"Scratch v2 ONNX model not found: {onnx_path}")

            from face_engine import create_ort_session
            from face_engine_adaface_ft import FaceEngineAdaFaceFT

            with tarfile.open(LAG_TAR_PATH, "r:gz") as tf:
                selected = _select_lag_tar_members(tf, target_idents, target_young, max_faces=len(target_idents), seed=42)
                imgs: List[np.ndarray] = []
                meta_idents: List[str] = []
                meta_young: List[bool] = []
                for ident, is_y, member in selected:
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        raise GuardError(f"Failed to read LAG tar member {member.name}")
                    img = cv2.imdecode(np.frombuffer(extracted.read(), np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise GuardError(f"Failed to decode LAG image {member.name}")
                    if img.shape[0] != 112 or img.shape[1] != 112:
                        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
                    imgs.append(img)
                    meta_idents.append(ident)
                    meta_young.append(is_y)

            meta_idents_arr = np.array(meta_idents)
            meta_young_arr = np.array(meta_young, dtype=bool)
            if not (np.array_equal(meta_idents_arr, target_idents) and np.array_equal(meta_young_arr, target_young)):
                raise GuardError("Decoded LAG cohort metadata does not match lag_cohort_cache.npz")

            sess = None
            try:
                sess = create_ort_session(onnx_path, use_gpu=use_gpu)
                inp_name = sess.get_inputs()[0].name
                out_name = sess.get_outputs()[0].name
                vecs = np.empty((len(imgs), 512), dtype=np.float32)
                batch_size = 32
                for start in range(0, len(imgs), batch_size):
                    batch_imgs = imgs[start : start + batch_size]
                    b1 = np.stack([FaceEngineAdaFaceFT.chip_bgr_to_rgb_chw(im) for im in batch_imgs], axis=0)
                    b2 = np.stack(
                        [FaceEngineAdaFaceFT.chip_bgr_to_rgb_chw(cv2.flip(im, 1)) for im in batch_imgs], axis=0
                    )
                    e1 = sess.run([out_name], {inp_name: b1})[0]
                    e2 = sess.run([out_name], {inp_name: b2})[0]
                    e1 = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
                    e2 = e2 / np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12)
                    comb = e1 + e2
                    comb = comb / np.maximum(np.linalg.norm(comb, axis=1, keepdims=True), 1e-12)
                    vecs[start : start + batch_size] = comb.astype(np.float32)
            finally:
                if sess is not None:
                    del sess
                gc.collect()

            os.makedirs(os.path.dirname(LAG_SCRATCH_CACHE_PATH), exist_ok=True)
            np.savez_compressed(
                LAG_SCRATCH_CACHE_PATH,
                fromscratchv2=vecs,
                identities=meta_idents_arr,
                is_young=meta_young_arr,
            )
            cohorts["fromscratchv2"] = vecs
            print(f">> Saved fromscratchv2 external LAG cohort ({len(vecs)} faces) -> {LAG_SCRATCH_CACHE_PATH}")

    for ek in engine_keys:
        if ek not in cohorts:
            raise GuardError(f"LAG cohort vectors missing for engine '{ek}'")

    adult_mask = (~np.asarray(target_young, dtype=bool)).astype(bool)
    return {
        "cohorts": cohorts,
        "identities": target_idents,
        "is_young": target_young,
        "adult_mask": adult_mask,
        "adult_masks": {ek: adult_mask for ek in engine_keys},
        "cohort_size": int(len(target_idents)),
    }


# ---------------------------------------------------------------------------
# Deterministic Unified 5-Dataset Partitioning & Deduplication
# ---------------------------------------------------------------------------


def build_unified_5dataset_plan(
    *,
    cohort_ids: Set[str],
    cacd_ids: Set[str],
    calfw_pairs: Sequence[Tuple[str, str, int, int]],
    fgnet_gallery_by_id: Dict[str, List[Tuple[int, str]]],
    fgnet_probes: Sequence[Dict[str, Any]],
    ecaf_identities: Dict[str, Dict[str, Any]],
    caf_identities: Dict[str, Dict[str, Any]],
    agedb_identities: Dict[str, List[Tuple[int, str, str]]],
    agedb_cache_priority: Optional[Set[str]] = None,
    fgnet_all_raw_ids: Optional[Set[str]] = None,
    n_calib_nonmates: int = 350,
    n_strangers: int = 200,
    agedb_min_gap: int = 15,
) -> Dict[str, Any]:
    """
    Deterministically partitions CALFW-only identities into QMF calib, OpenSet calib non-mates,
    unseen strangers, and CALFW gallery, then builds one deduplicated 5-dataset gallery in
    priority order `fgnet -> ecaf -> caf -> agedb -> calfw` with zero cross-split leakage.
    """
    cohort_norm = {normalize_identity(x) for x in cohort_ids if normalize_identity(x)}
    cacd_norm = {normalize_identity(x) for x in cacd_ids if normalize_identity(x)}
    lfw_norm = {normalize_identity(x) for x in LFW_CONTAMINATED_NONMATES if normalize_identity(x)}
    global_excl = cohort_norm | cacd_norm | lfw_norm

    # Normalize & merge ECAF, CAF, AgeDB dictionaries by canonical identity key
    ecaf_by_norm: Dict[str, Dict[str, Any]] = {}
    for raw_k, info in ecaf_identities.items():
        nk = normalize_identity(raw_k)
        if not nk:
            continue
        dst = ecaf_by_norm.setdefault(
            nk,
            {
                "display_name": info.get("display_name", raw_k),
                "child": [],
                "adult": [],
                "official_test": bool(info.get("official_test", False)),
            },
        )
        dst["child"].extend(info.get("child", []))
        dst["adult"].extend(info.get("adult", []))
        dst["official_test"] = bool(dst["official_test"] or info.get("official_test", False))

    caf_by_norm: Dict[str, Dict[str, Any]] = {}
    for raw_k, info in caf_identities.items():
        nk = normalize_identity(raw_k)
        if not nk:
            continue
        dst = caf_by_norm.setdefault(nk, {"display_name": info.get("display_name", raw_k), "photos": []})
        dst["photos"].extend(info.get("photos", []))

    agedb_by_norm: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for raw_k, items in agedb_identities.items():
        nk = normalize_identity(raw_k)
        if not nk:
            continue
        for item in items:
            agedb_by_norm[nk].append((int(item[0]), item[1]))

    # Extract CALFW photos and min fold per canonical identity
    calfw_fold_by_id: Dict[str, int] = {}
    calfw_photo_sets: Dict[str, Set[str]] = defaultdict(set)
    for pa, pb, _is_same, fold in calfw_pairs:
        f_int = int(fold)
        for p in (pa, pb):
            nk = _calfw_norm_from_path(p)
            if not nk:
                continue
            calfw_photo_sets[nk].add(p)
            prev_f = calfw_fold_by_id.get(nk)
            calfw_fold_by_id[nk] = f_int if prev_f is None else min(prev_f, f_int)

    calfw_photos_by_id: Dict[str, List[str]] = {
        nk: sorted(paths, key=_calfw_sort_key) for nk, paths in calfw_photo_sets.items()
    }

    # Union of ALL normalized IDs present anywhere in FG-NET, ECAF, CAF, AgeDB
    other_ds_ids: Set[str] = set()
    for sid in fgnet_gallery_by_id:
        base_sid = sid[6:] if str(sid).startswith("fgnet_") else str(sid)
        other_ds_ids.add(f"fgnet_{base_sid}")
        other_ds_ids.add(normalize_identity(f"fgnet_{base_sid}"))
    if fgnet_all_raw_ids:
        for sid in fgnet_all_raw_ids:
            base_sid = sid[6:] if str(sid).startswith("fgnet_") else str(sid)
            other_ds_ids.add(f"fgnet_{base_sid}")
            other_ds_ids.add(normalize_identity(f"fgnet_{base_sid}"))
    other_ds_ids |= set(ecaf_by_norm.keys())
    other_ds_ids |= set(caf_by_norm.keys())
    other_ds_ids |= set(agedb_by_norm.keys())

    calfw_only_clean_ids = sorted(
        nk
        for nk, photos in calfw_photos_by_id.items()
        if nk not in global_excl and nk not in other_ds_ids and len(photos) >= 2
    )

    # Partition calfw_only_clean_ids into 4 mutually exclusive sets:
    # 1) QMF calibration (folds 0..1)
    qmf_calib_ids = sorted(nk for nk in calfw_only_clean_ids if calfw_fold_by_id[nk] <= 1)
    qmf_calib_set = set(qmf_calib_ids)
    fit_pairs: List[Tuple[str, str, int, int]] = [
        (pa, pb, int(is_same), int(fold))
        for pa, pb, is_same, fold in calfw_pairs
        if int(fold) <= 1 and _calfw_norm_from_path(pa) in qmf_calib_set and _calfw_norm_from_path(pb) in qmf_calib_set
    ]

    # 2) Open-set non-mate calibration pool (folds 2..3 first, deterministic shuffle seed=77)
    rng_nm = np.random.default_rng(77)
    pool_fold23 = [nk for nk in calfw_only_clean_ids if nk not in qmf_calib_set and calfw_fold_by_id[nk] <= 3]
    rng_nm.shuffle(pool_fold23)
    if len(pool_fold23) < n_calib_nonmates:
        pool_rest = [
            nk for nk in calfw_only_clean_ids if nk not in qmf_calib_set and nk not in set(pool_fold23)
        ]
        rng_nm.shuffle(pool_rest)
        pool_combined = pool_fold23 + pool_rest
    else:
        pool_combined = pool_fold23
    calib_nonmate_ids = sorted(pool_combined[: min(int(n_calib_nonmates), len(pool_combined))])
    calib_nonmate_set = set(calib_nonmate_ids)
    calib_nonmate_probes = [(nk, calfw_photos_by_id[nk][-1]) for nk in calib_nonmate_ids]

    # 3) Held-out unseen open-set strangers (seed=99)
    rng_str = np.random.default_rng(99)
    rem_for_strangers = [
        nk for nk in calfw_only_clean_ids if nk not in qmf_calib_set and nk not in calib_nonmate_set
    ]
    rng_str.shuffle(rem_for_strangers)
    stranger_test_ids = sorted(rem_for_strangers[: min(int(n_strangers), len(rem_for_strangers))])
    stranger_set = set(stranger_test_ids)
    stranger_probes = [
        {
            "subject_id": nk,
            "norm_id": nk,
            "display_name": nk,
            "dataset": "stranger",
            "age": None,
            "path": calfw_photos_by_id[nk][-1],
        }
        for nk in stranger_test_ids
    ]

    # 4) Unified deduplicated gallery in priority order: fgnet -> ecaf -> caf -> agedb -> calfw
    gallery_by_id: Dict[str, List[Tuple[Optional[int], str]]] = {}
    dataset_by_id: Dict[str, str] = {}
    mated_probes: List[Dict[str, Any]] = []
    seen_norm_ids: Set[str] = set()

    # FG-NET probes indexed by canonical gallery key
    fg_probes_by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pr in fgnet_probes:
        raw_sid = str(pr["subject_id"])
        base_sid = raw_sid[6:] if raw_sid.startswith("fgnet_") else raw_sid
        fg_probes_by_sid[f"fgnet_{base_sid}"].append(pr)

    for sid in sorted(fgnet_gallery_by_id.keys()):
        base_sid = sid[6:] if str(sid).startswith("fgnet_") else str(sid)
        nk = f"fgnet_{base_sid}"
        nk_norm = normalize_identity(nk)
        if nk in global_excl or nk_norm in global_excl or nk in seen_norm_ids or nk_norm in seen_norm_ids:
            continue
        gal_items = sorted([(int(a), p) for a, p in fgnet_gallery_by_id[sid]], key=lambda x: (x[0], x[1]))
        pr_list = fg_probes_by_sid.get(nk, [])
        if not gal_items or not pr_list:
            continue
        seen_norm_ids.add(nk)
        seen_norm_ids.add(nk_norm)
        gallery_by_id[nk] = gal_items
        dataset_by_id[nk] = "fgnet"
        for pr in sorted(pr_list, key=lambda x: (int(x["age"]) if x.get("age") is not None else -1, x["path"])):
            mated_probes.append({
                "subject_id": nk,
                "norm_id": nk,
                "display_name": nk,
                "dataset": "fgnet",
                "age": pr.get("age"),
                "path": pr["path"],
            })

    for nk in sorted(ecaf_by_norm.keys()):
        if nk in seen_norm_ids or nk in global_excl:
            continue
        info = ecaf_by_norm[nk]
        child_items = sorted(info["child"], key=lambda x: (x[0], x[1]))
        adult_items = sorted(info["adult"], key=lambda x: (x[0], x[1]))
        if not child_items or not adult_items:
            continue
        seen_norm_ids.add(nk)
        gallery_by_id[nk] = [(None, fp) for _, fp in child_items]
        dataset_by_id[nk] = "ecaf"
        for _, fp in adult_items:
            mated_probes.append({
                "subject_id": nk,
                "norm_id": nk,
                "display_name": info["display_name"],
                "dataset": "ecaf",
                "age": None,
                "path": fp,
            })

    for nk in sorted(caf_by_norm.keys()):
        if nk in seen_norm_ids or nk in global_excl:
            continue
        items = sorted(caf_by_norm[nk]["photos"], key=lambda x: (int(x[0]), x[1]))
        if len(items) < 2:
            continue
        gal_items = [(int(age), p) for age, p in items if int(age) <= 12]
        pr_items = [(int(age), p) for age, p in items if int(age) > 12]
        if not gal_items or not pr_items:
            continue
        seen_norm_ids.add(nk)
        gallery_by_id[nk] = gal_items
        dataset_by_id[nk] = "caf"
        for age, p in pr_items:
            mated_probes.append({
                "subject_id": nk,
                "norm_id": nk,
                "display_name": caf_by_norm[nk]["display_name"],
                "dataset": "caf",
                "age": int(age),
                "path": p,
            })

    prio_set = {os.path.normcase(os.path.abspath(p)) for p in (agedb_cache_priority or set())}
    agedb_distractor_ids = 0
    for nk in sorted(agedb_by_norm.keys()):
        if nk in seen_norm_ids or nk in global_excl:
            continue
        raw_items = agedb_by_norm[nk]
        if len(raw_items) < 2:
            continue
        photos = sorted(
            raw_items,
            key=lambda x: (x[0], 0 if os.path.normcase(os.path.abspath(x[1])) in prio_set else 1, x[1]),
        )
        min_a, max_a = photos[0][0], photos[-1][0]
        if (max_a - min_a) >= agedb_min_gap:
            pr_sel = photos[-2:]
            young = [(a, p) for a, p in photos[:-2] if a <= pr_sel[0][0] - agedb_min_gap]
            if not young:
                pr_sel = photos[-1:]
                young = [(a, p) for a, p in photos[:-1] if a <= pr_sel[0][0] - agedb_min_gap]
            gal_sel = young[:4]
            if not gal_sel or not pr_sel:
                continue
            if min(a for a, _ in pr_sel) - max(a for a, _ in gal_sel) < agedb_min_gap:
                raise GuardError(f"AgeDB >= {agedb_min_gap}y gap invariant violated for '{nk}'")
            seen_norm_ids.add(nk)
            gallery_by_id[nk] = [(int(a), p) for a, p in gal_sel]
            dataset_by_id[nk] = "agedb"
            for age, p in pr_sel:
                mated_probes.append({
                    "subject_id": nk,
                    "norm_id": nk,
                    "display_name": nk,
                    "dataset": "agedb",
                    "age": int(age),
                    "path": p,
                })
        else:
            seen_norm_ids.add(nk)
            gallery_by_id[nk] = [(int(a), p) for a, p in photos[:3]]
            dataset_by_id[nk] = "agedb"
            agedb_distractor_ids += 1

    for nk in calfw_only_clean_ids:
        if nk in seen_norm_ids or nk in qmf_calib_set or nk in calib_nonmate_set or nk in stranger_set or nk in global_excl:
            continue
        photos = calfw_photos_by_id[nk]
        if len(photos) < 2:
            continue
        seen_norm_ids.add(nk)
        gallery_by_id[nk] = [(None, p) for p in photos[:-1]]
        dataset_by_id[nk] = "calfw"
        mated_probes.append({
            "subject_id": nk,
            "norm_id": nk,
            "display_name": nk,
            "dataset": "calfw",
            "age": None,
            "path": photos[-1],
        })

    # Disjointness assertions across ALL 6 identity sets (checking canonical normalized forms)
    gallery_ids = set(gallery_by_id.keys())
    gallery_norm_set = {normalize_identity(x) for x in gallery_ids}
    if len(gallery_norm_set) != len(gallery_ids):
        raise GuardError("Duplicate normalized identity detected in unified gallery")

    disjoint_sets = {
        "gallery": gallery_norm_set,
        "qmf_calib": {normalize_identity(x) for x in qmf_calib_set},
        "calib_nonmate": {normalize_identity(x) for x in calib_nonmate_set},
        "stranger_test": {normalize_identity(x) for x in stranger_set},
        "cohort": cohort_norm,
        "cacd": cacd_norm,
    }
    set_names = list(disjoint_sets.keys())
    for i in range(len(set_names)):
        for j in range(i + 1, len(set_names)):
            a_name, b_name = set_names[i], set_names[j]
            overlap = disjoint_sets[a_name] & disjoint_sets[b_name]
            if overlap:
                raise GuardError(
                    f"Identity leakage between '{a_name}' and '{b_name}': "
                    f"{len(overlap)} overlapping IDs (sample={sorted(overlap)[:5]})"
                )

    # Path overlap guard across unified gallery vs mated + stranger probes
    gal_paths_norm = {
        os.path.normcase(os.path.abspath(p)) for items in gallery_by_id.values() for _, p in items
    }
    for pr in list(mated_probes) + list(stranger_probes):
        if os.path.normcase(os.path.abspath(pr["path"])) in gal_paths_norm:
            raise GuardError(f"Path overlap guard violated: '{pr['path']}' appears in both gallery and probes")

    # Gather paths needed per dataset cache tag
    paths_by_cache_tag: Dict[str, Set[str]] = defaultdict(set)
    for subj_id, items in gallery_by_id.items():
        ds = dataset_by_id[subj_id]
        tag = "agedb30" if ds == "agedb" else ds
        for _, p in items:
            paths_by_cache_tag[tag].add(p)
    for pr in mated_probes:
        ds = pr["dataset"]
        tag = "agedb30" if ds == "agedb" else ds
        paths_by_cache_tag[tag].add(pr["path"])
    for pa, pb, _, _ in fit_pairs:
        paths_by_cache_tag["calfw"].add(pa)
        paths_by_cache_tag["calfw"].add(pb)
    for _, p in calib_nonmate_probes:
        paths_by_cache_tag["calfw"].add(p)
    for pr in stranger_probes:
        paths_by_cache_tag["calfw"].add(pr["path"])

    counts: Dict[str, Any] = {"datasets": {}}
    for ds in DATASET_ORDER:
        ds_ids = [sid for sid, d in dataset_by_id.items() if d == ds]
        ds_gal_photos = sum(len(gallery_by_id[sid]) for sid in ds_ids)
        ds_probes = sum(1 for pr in mated_probes if pr["dataset"] == ds)
        ds_info: Dict[str, Any] = {
            "gallery_ids": len(ds_ids),
            "gallery_photos": ds_gal_photos,
            "probes": ds_probes,
        }
        if ds == "agedb":
            ds_info["gallery_only_distractor_ids"] = agedb_distractor_ids
        counts["datasets"][ds] = ds_info

    counts["total_gallery_ids"] = len(gallery_by_id)
    counts["total_gallery_photos"] = sum(len(v) for v in gallery_by_id.values())
    counts["total_mated_probes"] = len(mated_probes)
    counts["qmf_calib_ids"] = len(qmf_calib_ids)
    counts["qmf_fit_pairs"] = len(fit_pairs)
    counts["calib_nonmate_ids"] = len(calib_nonmate_ids)
    counts["stranger_test_ids"] = len(stranger_test_ids)
    counts["cohort_excluded_ids"] = len(cohort_norm)
    counts["cacd_excluded_ids"] = len(cacd_norm)

    return {
        "gallery_by_id": gallery_by_id,
        "dataset_by_id": dataset_by_id,
        "mated_probes": mated_probes,
        "stranger_probes": stranger_probes,
        "qmf_calib_ids": qmf_calib_ids,
        "fit_pairs": fit_pairs,
        "calib_nonmate_ids": calib_nonmate_ids,
        "calib_nonmate_probes": calib_nonmate_probes,
        "stranger_test_ids": stranger_test_ids,
        "paths_by_cache_tag": {k: sorted(v) for k, v in paths_by_cache_tag.items()},
        "counts": counts,
        "disjointness_verified": True,
    }


def load_default_5dataset_inputs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads all real datasets and exclusion sets from D:\\GRAD\\Assets & Arc face caches."""
    cohort_ids = load_lag_cohort_identities()
    cacd_ids, cacd_meta = load_cacd_finetune_identities()
    calfw_pairs, calfw_meta = load_calfw_official(root=DEFAULT_CALFW_ROOT)
    fgnet_gal, fgnet_probes = load_fgnet_gallery(root=DEFAULT_FGNET_ROOT)
    ecaf_identities = parse_ecaf_images(root=DEFAULT_ECAF_ROOT)
    caf_identities = parse_caf_images(root=DEFAULT_CAF_ROOT)
    agedb_identities, _ = parse_agedb_images(DEFAULT_AGEDB_IMAGES_DIR)

    # Also capture all FG-NET subject IDs on disk (even child-only/adult-only) in other_ds_ids
    fgnet_all_raw: Set[str] = set(fgnet_gal.keys())
    for fn in os.listdir(DEFAULT_FGNET_ROOT):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            sid_part = fn.split("A")[0].split("a")[0]
            if sid_part.isdigit():
                fgnet_all_raw.add(sid_part)

    plan = build_unified_5dataset_plan(
        cohort_ids=cohort_ids,
        cacd_ids=cacd_ids,
        calfw_pairs=calfw_pairs,
        fgnet_gallery_by_id=fgnet_gal,
        fgnet_probes=fgnet_probes,
        ecaf_identities=ecaf_identities,
        caf_identities=caf_identities,
        agedb_identities=agedb_identities,
        agedb_cache_priority=None,
        fgnet_all_raw_ids=fgnet_all_raw,
    )
    meta = {
        "cacd_manifest": cacd_meta,
        "calfw_protocol": calfw_meta,
        "plan_manifest_sha256": compute_manifest_sha256(
            [f"G|{sid}|{a}|{os.path.basename(p)}" for sid, items in plan["gallery_by_id"].items() for a, p in items]
            + [f"P|{pr['dataset']}|{pr['subject_id']}|{os.path.basename(pr['path'])}" for pr in plan["mated_probes"]]
            + [f"S|{pr['subject_id']}|{os.path.basename(pr['path'])}" for pr in plan["stranger_probes"]]
        ),
    }
    return plan, meta


# ---------------------------------------------------------------------------
# Embedding Cache Assurance & Single-Face Probe Discovery
# ---------------------------------------------------------------------------


def _collect_stage3_candidate_paths(plan: Dict[str, Any], per_bucket: int = 18) -> Dict[str, List[str]]:
    """Deterministically selects up to `per_bucket` candidate paths per Stage 3 bucket."""
    out: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, Set[str]] = defaultdict(set)
    for pr in sorted(plan["mated_probes"], key=lambda x: (x["dataset"], x["subject_id"], x["path"])):
        ds = pr["dataset"]
        if ds not in ("fgnet", "ecaf", "agedb", "calfw"):
            continue
        if len(out[ds]) < per_bucket and pr["path"] not in seen[ds]:
            out[ds].append(pr["path"])
            seen[ds].add(pr["path"])
    for pr in sorted(plan["stranger_probes"], key=lambda x: (x["subject_id"], x["path"])):
        if len(out["stranger"]) < per_bucket and pr["path"] not in seen["stranger"]:
            out["stranger"].append(pr["path"])
            seen["stranger"].add(pr["path"])
    return dict(out)


def prepare_engine_embeddings_and_stage3_candidates(
    engine_keys: Sequence[str],
    plan: Dict[str, Any],
    use_gpu: bool = True,
    verify_sample: int = 24,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Ensures embeddings + qualities across all 5 datasets for each engine, verifies cache integrity
    against live re-embeds, checks Stage 3 single-face candidates while each engine is loaded,
    and cleanly frees VRAM before returning.
    """
    caches_by_engine: Dict[str, Tuple[Dict[str, Optional[np.ndarray]], Dict[str, float]]] = {}
    integrity_by_engine: Dict[str, Any] = {}
    providers_by_engine: Dict[str, Any] = {}
    actual_gpu_by_engine: Dict[str, bool] = {}
    single_face_flags: Dict[str, Dict[str, bool]] = {}

    stage3_cands = _collect_stage3_candidate_paths(plan, per_bucket=18)
    stage3_cand_paths = sorted({p for paths in stage3_cands.values() for p in paths})

    for ek in engine_keys:
        spec = ENGINE_SPECS[ek]
        eng_name = spec["engine_name"]
        onnx_path = spec["onnx_path"]

        engine_holder: Dict[str, Any] = {"engine": None, "providers": {}}

        def get_engine() -> Any:
            if engine_holder["engine"] is None:
                eng = load_engine_lazy(eng_name, use_gpu=use_gpu, use_tta=True, onnx_path=onnx_path)
                engine_holder["engine"] = eng
                engine_holder["providers"] = get_engine_providers(eng)
            return engine_holder["engine"]

        combined_emb: Dict[str, Optional[np.ndarray]] = {}
        combined_qual: Dict[str, float] = {}
        all_needed: List[str] = []

        try:
            for ds_tag in ("fgnet", "ecaf", "caf", "agedb30", "calfw"):
                needed = plan["paths_by_cache_tag"].get(ds_tag, [])
                if not needed:
                    continue
                cache_path = get_cache_path(ARC_DIR, ds_tag, eng_name, use_tta=True, onnx_path=onnx_path)
                emb_d, qual_d = ensure_strict_embeddings(needed, cache_path, get_engine)
                for k in needed:
                    combined_emb[k] = emb_d.get(k)
                    if k in qual_d:
                        combined_qual[k] = qual_d[k]
                all_needed.extend(needed)

            integrity = verify_cache_integrity(
                all_needed,
                combined_emb,
                get_engine,
                verify_sample=verify_sample,
                seed=seed,
            )

            # Check single-face count on candidate Stage 3 probes using the already-loaded detector
            eng_obj = get_engine()
            s3_flags: Dict[str, bool] = {}
            for p in stage3_cand_paths:
                if _safe_unit_vec(combined_emb.get(p)) is None:
                    s3_flags[p] = False
                    continue
                try:
                    faces = eng_obj.embed_all(p)
                    s3_flags[p] = bool(len(faces) == 1)
                except Exception:
                    s3_flags[p] = False

            caches_by_engine[ek] = (combined_emb, combined_qual)
            integrity_by_engine[ek] = integrity
            single_face_flags[ek] = s3_flags
            providers_by_engine[ek] = engine_holder["providers"]
            actual_gpu_by_engine[ek] = bool(getattr(eng_obj, "use_gpu", False))
        finally:
            if engine_holder["engine"] is not None:
                del engine_holder["engine"]
                engine_holder["engine"] = None
                gc.collect()

    return {
        "caches_by_engine": caches_by_engine,
        "integrity_by_engine": integrity_by_engine,
        "providers_by_engine": providers_by_engine,
        "actual_gpu_by_engine": actual_gpu_by_engine,
        "single_face_flags": single_face_flags,
        "stage3_candidates": stage3_cands,
    }


def select_stage3_probes(
    plan: Dict[str, Any],
    single_face_flags_by_engine: Dict[str, Dict[str, bool]],
    caches_by_engine: Dict[str, Tuple[Dict[str, Optional[np.ndarray]], Dict[str, float]]],
    engine_keys: Sequence[str],
    per_bucket: int = 6,
) -> List[Dict[str, Any]]:
    """Selects 30 deterministic single-face probes (6 fgnet/ecaf/agedb/calfw/stranger)."""
    probe_lookup: Dict[str, Dict[str, Any]] = {}
    for pr in plan["mated_probes"]:
        probe_lookup.setdefault(pr["path"], pr)
    for pr in plan["stranger_probes"]:
        probe_lookup.setdefault(pr["path"], pr)

    cands = _collect_stage3_candidate_paths(plan, per_bucket=18)
    selected: List[Dict[str, Any]] = []
    for bucket in STAGE3_BUCKETS:
        picked = 0
        for p in cands.get(bucket, []):
            ok_all = all(
                single_face_flags_by_engine.get(ek, {}).get(p, False)
                and _safe_unit_vec(caches_by_engine[ek][0].get(p)) is not None
                for ek in engine_keys
            )
            if ok_all:
                pr = probe_lookup[p]
                selected.append({
                    "bucket": bucket,
                    "dataset": pr["dataset"],
                    "subject_id": None if bucket == "stranger" else pr["subject_id"],
                    "path": p,
                })
                picked += 1
                if picked >= per_bucket:
                    break
        if picked < per_bucket:
            raise GuardError(f"Insufficient single-face Stage 3 probes for '{bucket}': found {picked}/{per_bucket}")
    return selected


# ---------------------------------------------------------------------------
# Matched 4-Layer Calibration Bundle Fitting
# ---------------------------------------------------------------------------


def _build_gallery_arrays_for_calib(
    gallery_by_id: Dict[str, List[Tuple[Optional[int], str]]],
    emb_cache: Dict[str, Optional[np.ndarray]],
    qual_cache: Dict[str, float],
    proj_basis: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Builds FaceStore-identical quality + inverse age-density centroids for OpenSet simulation."""
    pids: List[int] = []
    norm_keys: List[str] = []
    cent_vecs: List[np.ndarray] = []
    cent_quals: List[float] = []
    faces_by_pid: Dict[int, List[Tuple[int, Optional[int], np.ndarray, float]]] = {}

    next_pid = 1
    next_fid = 1
    for sid in sorted(gallery_by_id.keys()):
        items = []
        for age, path in gallery_by_id[sid]:
            vec = _safe_unit_vec(emb_cache.get(path))
            if vec is None:
                continue
            if proj_basis is not None:
                vec = project_subspace(vec, proj_basis)
            q_val = _clamp_quality(qual_cache.get(path, 1.0))
            items.append((next_fid, age, vec, q_val))
            next_fid += 1
        if not items:
            continue

        vecs = np.stack([x[2] for x in items]).astype(np.float32)
        ages = [x[1] for x in items]
        quals = [x[3] for x in items]
        weights = np.empty(len(items), dtype=np.float32)
        for i, (age_i, qual_i) in enumerate(zip(ages, quals)):
            if age_i is None:
                w_age = 1.0
            else:
                age_f = float(age_i)
                nearby = sum(
                    1 for j, a_j in enumerate(ages) if j != i and a_j is not None and abs(float(a_j) - age_f) <= 2.0
                )
                w_age = 1.0 / (1.0 + nearby)
            weights[i] = qual_i * w_age
        weighted = np.sum(vecs * weights[:, None], axis=0)
        nrm = float(np.linalg.norm(weighted))
        cvec = (weighted / nrm if nrm > 0 else weighted).astype(np.float32)

        pids.append(next_pid)
        norm_keys.append(sid)
        cent_vecs.append(cvec)
        cent_quals.append(float(max(quals)))
        faces_by_pid[next_pid] = items
        next_pid += 1

    return {
        "pids": np.array(pids, dtype=np.int64),
        "norm_keys": norm_keys,
        "cent_vecs": np.stack(cent_vecs).astype(np.float32),
        "cent_quals": np.array(cent_quals, dtype=np.float32),
        "faces_by_pid": faces_by_pid,
    }


def fit_matched_calibration_bundle(
    engine_keys: Sequence[str],
    plan: Dict[str, Any],
    caches_by_engine: Dict[str, Tuple[Dict[str, Optional[np.ndarray]], Dict[str, float]]],
    cohorts_by_engine: Dict[str, np.ndarray],
    save_paths: Optional[Dict[str, str]] = None,
    cohort_identities: Optional[Sequence[str]] = None,
    cohort_is_young: Optional[Sequence[bool]] = None,
    proj_k: int = 5,
    photo_hybrid_weight: float = 0.35,
) -> Tuple[Dict[str, CalibrationBundle], Dict[str, Any]]:
    """
    Fits matched QMF (age_gap=None), COVE, and OpenSet calibrators on the exact intersected
    CALFW folds 0..1 pairs and fold 2..3 non-mate probes across all active engines.
    Saves only explicit `save_paths` (never overwrites calib_aimids_crossage or calib_antelopev2).
    """
    matched_fit_pairs = [
        (pa, pb, lbl, fold)
        for pa, pb, lbl, fold in plan["fit_pairs"]
        if all(
            _safe_unit_vec(caches_by_engine[ek][0].get(pa)) is not None
            and _safe_unit_vec(caches_by_engine[ek][0].get(pb)) is not None
            for ek in engine_keys
        )
    ]
    if len(matched_fit_pairs) < 10:
        raise GuardError(f"Insufficient intersected QMF fit pairs across engines: {len(matched_fit_pairs)}")

    matched_nm_probes = [
        (nk, p)
        for nk, p in plan["calib_nonmate_probes"]
        if all(_safe_unit_vec(caches_by_engine[ek][0].get(p)) is not None for ek in engine_keys)
    ]
    if len(matched_nm_probes) < 10:
        raise GuardError(f"Insufficient intersected OpenSet non-mate probes: {len(matched_nm_probes)}")

    bundles: Dict[str, CalibrationBundle] = {}
    diagnostics: Dict[str, Any] = {}

    for ek in engine_keys:
        spec = ENGINE_SPECS[ek]
        emb_cache, qual_cache = caches_by_engine[ek]
        full_cohort = cohorts_by_engine[ek]

        proj_basis: Optional[np.ndarray] = None
        if cohort_identities is not None and cohort_is_young is not None:
            young_mask = np.asarray(cohort_is_young, dtype=bool)
            if proj_k > 0:
                proj_basis = compute_age_subspace_basis(
                    full_cohort, cohort_identities, young_mask, n_components=proj_k
                )
            adult_mask = ~young_mask
            asnorm_cohort_raw = full_cohort[adult_mask] if np.any(adult_mask) else full_cohort
            asnorm_cohort = project_subspace(asnorm_cohort_raw, proj_basis)
        else:
            asnorm_cohort = full_cohort

        asnorm = ExternalCohortASNorm(asnorm_cohort, top_k=200, std_floor=0.05)

        embs_a_raw = np.stack([_safe_unit_vec(emb_cache[pa]) for pa, _, _, _ in matched_fit_pairs]).astype(np.float32)
        embs_b_raw = np.stack([_safe_unit_vec(emb_cache[pb]) for _, pb, _, _ in matched_fit_pairs]).astype(np.float32)
        embs_a = project_subspace(embs_a_raw, proj_basis) if proj_basis is not None else embs_a_raw
        embs_b = project_subspace(embs_b_raw, proj_basis) if proj_basis is not None else embs_b_raw
        raw_fit = np.clip(np.sum(embs_a * embs_b, axis=1), -1.0, 1.0).astype(np.float64)
        mu_a, sig_a = asnorm.enroll_stats(embs_a)
        mu_b, sig_b = asnorm.enroll_stats(embs_b)
        z_fit = 0.5 * ((raw_fit - mu_a) / sig_a + (raw_fit - mu_b) / sig_b)
        q_fit = np.array(
            [
                min(_clamp_quality(qual_cache.get(pa, 1.0)), _clamp_quality(qual_cache.get(pb, 1.0)))
                for pa, pb, _, _ in matched_fit_pairs
            ],
            dtype=np.float64,
        )
        y_fit = np.array([lbl for _, _, lbl, _ in matched_fit_pairs], dtype=np.int32)

        # Fit with age_gap=None so production inference matches calibration cleanly without age-gap artifact
        qmf = QMFCalibrator.fit(z_fit, raw_fit, y_fit, q_min=q_fit, age_gap=None, l2_reg=5e-2)
        fit_eval = qmf.evaluate(z_fit, raw_fit, y_fit, q_min=q_fit, age_gap=None)

        # Keep CoveCalibrator on raw vectors for 1:1 /api/verify-id
        cove_vecs: List[np.ndarray] = []
        cove_ids: List[str] = []
        seen_cove: Set[str] = set()
        for idx_p, (pa, pb, _, _) in enumerate(matched_fit_pairs):
            id_a = _calfw_norm_from_path(pa)
            id_b = _calfw_norm_from_path(pb)
            if id_a not in seen_cove:
                seen_cove.add(id_a)
                cove_vecs.append(embs_a_raw[idx_p])
                cove_ids.append(id_a)
            if id_b not in seen_cove:
                seen_cove.add(id_b)
                cove_vecs.append(embs_b_raw[idx_p])
                cove_ids.append(id_b)
        cove = CoveCalibrator.fit(np.stack(cove_vecs), cove_ids) if len(seen_cove) >= 3 else None

        gal_arr = _build_gallery_arrays_for_calib(
            plan["gallery_by_id"], emb_cache, qual_cache, proj_basis=proj_basis
        )
        max_avail_n = int(len(gal_arr["pids"]))
        base_grid = (
            [1, 50, 80, 150, 300, 600, 1000, 2000]
            if max_avail_n >= 100
            else [1, max(2, max_avail_n // 4), max(3, max_avail_n // 2)]
        )
        candidate_ns = sorted({n for n in base_grid if 1 <= n <= max_avail_n} | {max_avail_n})

        nm_vecs_raw = np.stack([_safe_unit_vec(emb_cache[p]) for _, p in matched_nm_probes]).astype(np.float32)
        nm_vecs = project_subspace(nm_vecs_raw, proj_basis) if proj_basis is not None else nm_vecs_raw
        nm_quals = np.array([_clamp_quality(qual_cache.get(p, 1.0)) for _, p in matched_nm_probes], dtype=np.float32)
        nm_keys = [nk for nk, _ in matched_nm_probes]

        cal_llr_nm, _, _, base_nm = _simulate_nonmate_max_scores(
            nm_vecs,
            nm_quals,
            nm_keys,
            gal_arr["pids"],
            gal_arr["cent_vecs"],
            gal_arr["cent_quals"],
            gal_arr["norm_keys"],
            gal_arr["faces_by_pid"],
            asnorm,
            qmf,
            candidate_ns,
            seed=123,
            photo_hybrid_weight=photo_hybrid_weight,
        )
        openset = OpenSetCalibrator.fit(
            candidate_ns,
            [cal_llr_nm[n] for n in candidate_ns],
            base_nm,
            tail_quantile=0.95,
        )

        meta = {
            "cohort_source": "lag_external_tar",
            "cohort_filter": "lag_adult_only" if cohort_is_young is not None else "all",
            "calibration_split": "matched_calfw_folds_01_nonmates_23",
            "fit_trials": len(matched_fit_pairs),
            "nonmate_calib_probes": len(matched_nm_probes),
            "gallery_n_max": max_avail_n,
            "fit_diagnostics": fit_eval,
        }
        if spec.get("onnx_path"):
            meta["onnx_stem"] = sanitize_onnx_stem(spec["onnx_path"])

        bundle = CalibrationBundle(
            model_name=spec["store_model"],
            dim=spec["dim"],
            asnorm=asnorm,
            qmf=qmf,
            openset=openset,
            cove=cove,
            proj_basis=proj_basis,
            photo_hybrid_weight=photo_hybrid_weight,
            metadata=meta,
        )
        bundles[ek] = bundle
        diagnostics[ek] = {
            "fit_trials": len(matched_fit_pairs),
            "nonmate_calib_probes": len(matched_nm_probes),
            "gallery_n_max": max_avail_n,
            "proj_k": int(proj_basis.shape[0]) if proj_basis is not None else 0,
            "photo_hybrid_weight": float(bundle.photo_hybrid_weight),
            "cohort_size": int(asnorm.m),
            "fit_diagnostics": fit_eval,
            "openset_n_grid": candidate_ns,
            "tau_strong_max_n": round(openset.tier_thresholds(max_avail_n)[0], 4),
            "tau_possible_max_n": round(openset.tier_thresholds(max_avail_n)[1], 4),
        }

        if save_paths and ek in save_paths:
            npz_out, json_out = bundle.save(save_paths[ek])
            diagnostics[ek]["saved_npz"] = npz_out
            diagnostics[ek]["saved_json"] = json_out
            print(f">> Saved matched CalibrationBundle for {ek} -> {npz_out}")

    return bundles, diagnostics


# ---------------------------------------------------------------------------
# Stage 1 & Stage 3 HTTP Server Helpers + Workers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _ServerThread(threading.Thread):
    def __init__(self, app: Any, port: int):
        super().__init__(daemon=True)
        import uvicorn

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self) -> None:
        self.server.run()

    def stop(self) -> None:
        self.server.should_exit = True


def _wait_server_ready(base_url: str, timeout_sec: float = 60.0) -> Dict[str, Any]:
    import requests

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/", timeout=1.5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(0.4)
    raise RuntimeError(f"FastAPI server at {base_url} failed to respond within {timeout_sec:.0f}s")


def run_stage1_api_smoke_worker(engine_key: str, use_gpu: bool = True) -> int:
    """
    Runs isolated Missing-Persons + Core FastAPI HTTP smoke test for one engine
    (skips /api/verify-id, /api/verify-id/check-framing, and LivenessChecker).
    """
    if engine_key not in ENGINE_SPECS:
        raise ValueError(f"Unknown engine_key '{engine_key}'")
    spec = ENGINE_SPECS[engine_key]

    tmp_dir = tempfile.mkdtemp(prefix=f"aimids_s1_{engine_key}_")
    tmp_db = os.path.join(tmp_dir, "smoke.db")
    os.environ["AIMIDS_ENGINE"] = spec["engine_name"]
    if spec["onnx_path"]:
        os.environ["AIMIDS_ONNX_PATH"] = spec["onnx_path"]
    else:
        os.environ.pop("AIMIDS_ONNX_PATH", None)
    os.environ["AIMIDS_TTA"] = "1"
    os.environ["AIMIDS_USE_GPU"] = "1" if use_gpu else "0"
    os.environ["AIMIDS_DB_PATH"] = tmp_db

    import requests
    import api

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    th = _ServerThread(api.app, port)
    th.start()

    passed = 0
    total = 0
    failed_checks: List[str] = []
    created_pids: List[int] = []
    preview_files_to_clean: List[str] = []

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
            print(f"  [PASS] {name} {extra}".rstrip())
        else:
            failed_checks.append(f"{name} {extra}".strip())
            print(f"  [FAIL] {name} {extra}".rstrip())

    root_data: Dict[str, Any] = {}
    try:
        root_data = _wait_server_ready(base_url, timeout_sec=60.0)
        check("GET / status online", root_data.get("status") == "online")
        check("GET / engine matches", root_data.get("engine") == spec["engine_name"], f"(got {root_data.get('engine')})")
        check("GET / tta is True", root_data.get("tta") is True)
        if use_gpu:
            check("GET / gpu is True", root_data.get("gpu") is True, f"(gpu={root_data.get('gpu')})")
        check("GET / calibration == calibrated", root_data.get("calibration") == "calibrated")
        check("GET / asnorm_active is True", root_data.get("asnorm_active") is True)

        r_stats0 = requests.get(f"{base_url}/api/stats", timeout=5)
        check("GET /api/stats initially empty", r_stats0.status_code == 200 and r_stats0.json().get("people") == 0)

        r_demo = requests.get(f"{base_url}/demo", timeout=5)
        check("GET /demo serves HTML", r_demo.status_code == 200 and "AI-MIDS" in r_demo.text)

        r_evil_cors = requests.get(f"{base_url}/api/persons", headers={"Origin": "https://evil.example.com"}, timeout=5)
        check("Unknown CORS origin blocked", r_evil_cors.headers.get("access-control-allow-origin") is None)

        r_cross_post = requests.post(
            f"{base_url}/api/persons",
            data={"name": "CrossSite"},
            headers={"Origin": "https://evil.example.com"},
            timeout=5,
        )
        check("Cross-site POST rejected 403", r_cross_post.status_code == 403)

        r_bad_host = requests.get(f"{base_url}/api/stats", headers={"Host": "evil.example.com"}, timeout=5)
        check("Untrusted Host header rejected 400", r_bad_host.status_code == 400)

        # Register Anas Test + Hoz Distractor
        r_anas = requests.post(f"{base_url}/api/persons", data={"name": "Anas Test", "notes": "smoke"}, timeout=5)
        check("POST /api/persons Anas Test 201", r_anas.status_code == 201)
        anas_pid = int(r_anas.json()["person_id"])
        created_pids.append(anas_pid)

        r_hoz = requests.post(f"{base_url}/api/persons", data={"name": "Hoz Distractor", "notes": "smoke"}, timeout=5)
        check("POST /api/persons Hoz Distractor 201", r_hoz.status_code == 201)
        hoz_pid = int(r_hoz.json()["person_id"])
        created_pids.append(hoz_pid)

        anas_dir = os.path.join(ARC_DIR, "my_photos", "anas")
        c1_path = os.path.join(anas_dir, "child1_anas.jpeg")
        c2_path = os.path.join(anas_dir, "child2_anas.jpg")
        grp_path = os.path.join(anas_dir, "portsaid.png")
        now_path = os.path.join(anas_dir, "now.png")

        with open(c1_path, "rb") as f:
            r_p1 = requests.post(
                f"{base_url}/api/persons/{anas_pid}/photos",
                files={"file": ("child1_anas.jpeg", f, "image/jpeg")},
                data={"age": 4},
                timeout=15,
            )
        check("Upload child1_anas 200 ok/warn", r_p1.status_code == 200 and r_p1.json().get("status") in ("ok", "warn"))
        face_id_1 = r_p1.json().get("face_id")

        with open(c2_path, "rb") as f:
            r_p2 = requests.post(
                f"{base_url}/api/persons/{anas_pid}/photos",
                files={"file": ("child2_anas.jpg", f, "image/jpeg")},
                data={"age": 5},
                timeout=15,
            )
        check("Upload child2_anas 200 ok/warn", r_p2.status_code == 200 and r_p2.json().get("status") in ("ok", "warn"))

        with open(grp_path, "rb") as f:
            r_hoz_p = requests.post(
                f"{base_url}/api/persons/{hoz_pid}/photos",
                files={"file": ("portsaid.png", f, "image/png")},
                data={"face_index": 2},
                timeout=15,
            )
        check(
            "Upload portsaid face_index=2 on Hoz 200",
            r_hoz_p.status_code == 200 and r_hoz_p.json().get("status") in ("ok", "warn"),
        )

        # Small/blank image rejected 400
        ok_enc, blank_buf = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
        assert ok_enc
        r_blank = requests.post(
            f"{base_url}/api/persons/{anas_pid}/photos",
            files={"file": ("blank.jpg", blank_buf.tobytes(), "image/jpeg")},
            timeout=10,
        )
        check(
            "Blank image rejected 400 status=reject",
            r_blank.status_code == 400 and r_blank.json().get("detail", {}).get("status") == "reject",
        )

        # Group photo without face_index -> choose_face + preview_url -> re-upload with face_index=0 -> delete temp
        r_tmp_p = requests.post(f"{base_url}/api/persons", data={"name": "Temp Group Check"}, timeout=5)
        tmp_pid = int(r_tmp_p.json()["person_id"])
        created_pids.append(tmp_pid)
        with open(grp_path, "rb") as f:
            r_choose = requests.post(
                f"{base_url}/api/persons/{tmp_pid}/photos",
                files={"file": ("portsaid.png", f, "image/png")},
                timeout=15,
            )
        ch_json = r_choose.json() if r_choose.status_code == 200 else {}
        prev_url = ch_json.get("preview_url", "")
        if prev_url:
            preview_files_to_clean.append(os.path.join(HERE, "uploads", "previews", os.path.basename(prev_url)))
        check(
            "Group photo without face_index returns choose_face + preview_url",
            r_choose.status_code == 200 and ch_json.get("status") == "choose_face" and bool(prev_url),
        )
        with open(grp_path, "rb") as f:
            r_reup = requests.post(
                f"{base_url}/api/persons/{tmp_pid}/photos",
                files={"file": ("portsaid.png", f, "image/png")},
                data={"face_index": 0},
                timeout=15,
            )
        check("Re-upload with face_index=0 succeeds", r_reup.status_code == 200 and r_reup.json().get("status") in ("ok", "warn"))
        r_del_tmp = requests.delete(f"{base_url}/api/persons/{tmp_pid}", timeout=5)
        check("Delete temp person succeeds", r_del_tmp.status_code == 200)
        if tmp_pid in created_pids:
            created_pids.remove(tmp_pid)

        # GET /api/persons, GET /api/persons/{id}, GET /api/photos/{face_id}
        r_list = requests.get(f"{base_url}/api/persons", timeout=5)
        check("GET /api/persons lists Anas", r_list.status_code == 200 and any(p["person_id"] == anas_pid for p in r_list.json()))

        r_det = requests.get(f"{base_url}/api/persons/{anas_pid}", timeout=5)
        det_photos = r_det.json().get("photos", []) if r_det.status_code == 200 else []
        check(
            "GET /api/persons/{id} shows 2 photos with photo_url and no photo_path",
            len(det_photos) == 2 and all("photo_url" in ph and "photo_path" not in ph for ph in det_photos),
        )

        r_img = requests.get(f"{base_url}/api/photos/{face_id_1}", timeout=5)
        check("GET /api/photos/{face_id} 200 image/jpeg", r_img.status_code == 200 and "image/jpeg" in r_img.headers.get("content-type", ""))

        # POST /api/search bounds & calibrated results
        with open(now_path, "rb") as f:
            r_k0 = requests.post(f"{base_url}/api/search", files={"file": ("now.png", f, "image/png")}, data={"k": 0}, timeout=10)
        check("POST /api/search k=0 rejected 400", r_k0.status_code == 400)
        with open(now_path, "rb") as f:
            r_k51 = requests.post(f"{base_url}/api/search", files={"file": ("now.png", f, "image/png")}, data={"k": 51}, timeout=10)
        check("POST /api/search k=51 rejected 400", r_k51.status_code == 400)

        with open(now_path, "rb") as f:
            r_s1 = requests.post(f"{base_url}/api/search", files={"file": ("now.png", f, "image/png")}, data={"k": 10}, timeout=15)
        s1_json = r_s1.json() if r_s1.status_code == 200 else {}
        matches = s1_json.get("results", [{}])[0].get("matches", []) if s1_json.get("results") else []
        top1 = matches[0] if matches else {}
        check(
            "POST /api/search now.png ranks Anas Test #1 with calibrated fields",
            r_s1.status_code == 200
            and s1_json.get("faces_detected", 0) >= 1
            and top1.get("person_id") == anas_pid
            and isinstance(top1.get("calibrated_llr"), (int, float))
            and isinstance(top1.get("match_prob"), (int, float))
            and isinstance(top1.get("p_nonmate"), (int, float))
            and isinstance(top1.get("advisory_flags"), list)
            and top1.get("match_source") in ("centroid", "photo")
            and isinstance(top1.get("centroid_similarity"), (int, float))
            and isinstance(top1.get("best_photo_similarity"), (int, float)),
            f"(top1={top1})",
        )

        with open(grp_path, "rb") as f:
            r_sg = requests.post(f"{base_url}/api/search", files={"file": ("portsaid.png", f, "image/png")}, data={"k": 3}, timeout=15)
        check("POST /api/search group portsaid.png detects 4 faces", r_sg.status_code == 200 and r_sg.json().get("faces_detected") == 4)

        # Delete both persons & verify clean DB + disk
        for pid in list(created_pids):
            r_del = requests.delete(f"{base_url}/api/persons/{pid}", timeout=5)
            check(f"DELETE /api/persons/{pid} 200", r_del.status_code == 200)
            created_pids.remove(pid)

        r_stats_end = requests.get(f"{base_url}/api/stats", timeout=5).json()
        check(
            "Final /api/stats reports 0 people and 0 photos",
            r_stats_end.get("people") == 0 and r_stats_end.get("photos") == 0,
        )
    finally:
        for pid in list(created_pids):
            try:
                requests.delete(f"{base_url}/api/persons/{pid}", timeout=5)
            except Exception:
                pass
        for pf in preview_files_to_clean:
            if os.path.isfile(pf):
                try:
                    os.remove(pf)
                except OSError:
                    pass
        th.stop()
        th.join(timeout=5)
        if getattr(api, "store", None) is not None:
            try:
                api.store.db.close()
            except Exception:
                pass
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = {
        "engine_key": engine_key,
        "engine_name": spec["engine_name"],
        "passed": passed,
        "total": total,
        "all_passed": bool(passed == total and total > 0),
        "calibration": root_data.get("calibration"),
        "gpu": root_data.get("gpu"),
        "tta": root_data.get("tta"),
        "asnorm_active": root_data.get("asnorm_active"),
        "failed_checks": failed_checks,
    }
    print(f"STAGE1_JSON: {json.dumps(summary)}")
    return 0 if summary["all_passed"] else 1


def _run_worker_subprocess(args_list: List[str], marker_prefix: str) -> Dict[str, Any]:
    """Runs a clean isolated worker subprocess of this script and parses its final JSON marker."""
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *args_list],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)

    parsed: Optional[Dict[str, Any]] = None
    for line in proc.stdout.splitlines():
        if line.startswith(marker_prefix):
            parsed = json.loads(line[len(marker_prefix) :].strip())

    if parsed is None:
        raise GuardError(f"Worker {args_list} exited with code {proc.returncode} without {marker_prefix}")
    parsed["exit_code"] = int(proc.returncode)
    return parsed


def run_stage1_subprocess(engine_key: str, use_gpu: bool = True) -> Dict[str, Any]:
    """Executes Stage 1 API smoke test in a clean subprocess for `engine_key`."""
    cli = ["--worker-api-smoke", engine_key]
    if not use_gpu:
        cli.append("--cpu")
    print(f"\n>> [Stage 1] Running FastAPI HTTP smoke test for '{engine_key}' (GPU={use_gpu})...")
    res = _run_worker_subprocess(cli, "STAGE1_JSON: ")
    if not res.get("all_passed") or res.get("exit_code") != 0:
        raise GuardError(f"Stage 1 API smoke failed for {engine_key}: {res}")
    return res


def _build_stage3_sqlite_db(
    db_path: str,
    plan: Dict[str, Any],
    store_model: str,
    emb_cache: Dict[str, Optional[np.ndarray]],
    qual_cache: Dict[str, float],
) -> Dict[str, int]:
    """Writes the unified 5-dataset gallery into a temporary SQLite DB and returns {subject_id: person_id}."""
    st = FaceStore(db_path)
    person_rows: List[Tuple[int, str, str, float]] = []
    face_rows: List[Tuple[int, Optional[str], Optional[int], bytes, str, float]] = []
    pid_map: Dict[str, int] = {}
    next_pid = 1
    now_ts = time.time()

    try:
        for subj_id in sorted(plan["gallery_by_id"].keys()):
            valid_items = []
            for age, path in plan["gallery_by_id"][subj_id]:
                vec = _safe_unit_vec(emb_cache.get(path))
                if vec is not None:
                    valid_items.append((age, path, vec, _clamp_quality(qual_cache.get(path, 1.0))))
            if not valid_items:
                continue
            ds = plan["dataset_by_id"][subj_id]
            person_rows.append((next_pid, str(subj_id), f"dataset={ds}", now_ts))
            for age, path, vec, q_val in valid_items:
                face_rows.append((next_pid, path, age, vec.tobytes(), store_model, float(q_val)))
            pid_map[subj_id] = next_pid
            next_pid += 1

        st.db.executemany("INSERT INTO persons(id, name, notes, created) VALUES (?,?,?,?)", person_rows)
        st.db.executemany(
            "INSERT INTO faces(person_id, photo_path, age, embedding, model, quality) VALUES (?,?,?,?,?,?)",
            face_rows,
        )
        st.db.commit()
    finally:
        st.db.close()

    return pid_map


def run_stage3_http_worker(spec_json_path: str) -> int:
    """
    Loads unified 5-dataset SQLite DB in FastAPI + in-process FaceStore, verifies live
    HTTP `POST /api/search` parity across 30 deterministic single-face probes.
    """
    with open(spec_json_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    os.environ["AIMIDS_ENGINE"] = spec["engine_name"]
    if spec.get("onnx_path"):
        os.environ["AIMIDS_ONNX_PATH"] = spec["onnx_path"]
    else:
        os.environ.pop("AIMIDS_ONNX_PATH", None)
    os.environ["AIMIDS_TTA"] = "1"
    os.environ["AIMIDS_USE_GPU"] = "1" if spec.get("use_gpu", True) else "0"
    os.environ["AIMIDS_DB_PATH"] = spec["db_path"]

    import requests
    import api
    from calibration import extract_face_quality

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    th = _ServerThread(api.app, port)
    th.start()

    ref_store: Optional[FaceStore] = None
    try:
        root = _wait_server_ready(base_url, timeout_sec=90.0)
        if root.get("calibration") != "calibrated" or root.get("asnorm_active") is not True:
            raise GuardError(f"Stage 3 root not calibrated: {root}")
        if root.get("stats", {}).get("people") != spec["expected_people"]:
            raise GuardError(
                f"Stage 3 people count mismatch: HTTP={root.get('stats', {}).get('people')} "
                f"vs expected={spec['expected_people']}"
            )

        ref_store = FaceStore(spec["db_path"])
        store_model = spec["store_model"]
        if not ref_store.calibrated_ready(store_model):
            raise GuardError(f"Reference FaceStore not calibrated for '{store_model}'")

        latencies_ms: List[float] = []
        llr_diffs: List[float] = []
        sim_diffs: List[float] = []
        rank1_parity = 0
        mated_total = 0
        mated_rank1_hits = 0
        failures: List[str] = []

        probe_items = spec["probe_items"]
        for item in probe_items:
            p_path = item["path"]
            img = cv2.imdecode(np.fromfile(p_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                failures.append(f"cv2.imdecode failed for {p_path}")
                continue
            emb_res = api.engine.embed(img)
            if emb_res is None:
                failures.append(f"api.engine.embed None for {p_path}")
                continue
            q_probe = extract_face_quality(api.engine, img, emb_res)
            store_hits = ref_store.search(
                emb_res["embedding"], k=10, min_sim=-1.0, model=store_model, probe_quality=q_probe, age_gap=None
            )

            mime = mimetypes.guess_type(p_path)[0] or "image/jpeg"
            with open(p_path, "rb") as pf:
                raw_bytes = pf.read()

            t0 = time.perf_counter()
            resp = requests.post(
                f"{base_url}/api/search",
                files={"file": (os.path.basename(p_path), raw_bytes, mime)},
                data={"k": 10, "min_sim": -1.0},
                timeout=30,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            if resp.status_code != 200:
                failures.append(f"HTTP {resp.status_code} on {p_path}")
                continue
            rj = resp.json()
            if rj.get("faces_detected") != 1 or not rj.get("results"):
                failures.append(f"faces_detected={rj.get('faces_detected')} on {p_path}")
                continue
            http_matches = rj["results"][0].get("matches", [])
            if not http_matches or not store_hits:
                failures.append(f"Empty matches on {p_path}")
                continue

            h_top = http_matches[0]
            s_top = store_hits[0]
            if h_top.get("person_id") == s_top.get("person_id"):
                rank1_parity += 1
            else:
                failures.append(
                    f"Rank-1 mismatch on {os.path.basename(p_path)}: http={h_top.get('person_id')} vs store={s_top.get('person_id')}"
                )

            llr_diff = abs(float(h_top["calibrated_llr"]) - float(s_top["calibrated_llr"]))
            sim_diff = abs(float(h_top["similarity"]) - float(s_top["similarity"]))
            llr_diffs.append(llr_diff)
            sim_diffs.append(sim_diff)
            if llr_diff >= 1e-2:
                failures.append(f"LLR diff {llr_diff:.5f} >= 1e-2 on {os.path.basename(p_path)}")
            if sim_diff >= 1e-3:
                failures.append(f"Sim diff {sim_diff:.5f} >= 1e-3 on {os.path.basename(p_path)}")

            exp_pid = item.get("expected_person_id")
            if exp_pid is not None:
                mated_total += 1
                if h_top.get("person_id") == exp_pid:
                    mated_rank1_hits += 1

        n_probes = len(probe_items)
        max_llr_diff = float(max(llr_diffs)) if llr_diffs else None
        max_sim_diff = float(max(sim_diffs)) if sim_diffs else None
        all_ok = (
            len(failures) == 0
            and rank1_parity == n_probes
            and max_llr_diff is not None
            and max_llr_diff < 1e-2
            and max_sim_diff is not None
            and max_sim_diff < 1e-3
        )
        out = {
            "engine_key": spec["engine_key"],
            "n_probes": n_probes,
            "expected_people": spec["expected_people"],
            "http_calibration": root.get("calibration"),
            "http_asnorm_active": root.get("asnorm_active"),
            "rank1_parity_count": rank1_parity,
            "rank1_parity_pct": (float(rank1_parity) / max(1, n_probes)) * 100.0,
            "max_llr_diff": max_llr_diff,
            "max_sim_diff": max_sim_diff,
            "http_latency_mean_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else None,
            "http_latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else None,
            "mated_probe_count": mated_total,
            "mated_http_rank1_hits": mated_rank1_hits,
            "all_passed": bool(all_ok),
            "failures": failures[:10],
        }
        print(f"STAGE3_JSON: {json.dumps(out)}")
        return 0 if all_ok else 1
    finally:
        if ref_store is not None:
            try:
                ref_store.db.close()
            except Exception:
                pass
        th.stop()
        th.join(timeout=5)
        if getattr(api, "store", None) is not None:
            try:
                api.store.db.close()
            except Exception:
                pass


def run_stage3_subprocess(
    engine_key: str,
    plan: Dict[str, Any],
    emb_cache: Dict[str, Optional[np.ndarray]],
    qual_cache: Dict[str, float],
    stage3_probes: Sequence[Dict[str, Any]],
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Builds isolated unified SQLite DB and verifies live HTTP parity in a clean subprocess."""
    spec = ENGINE_SPECS[engine_key]
    print(f"\n>> [Stage 3] Verifying live HTTP POST /api/search parity for '{engine_key}' ({len(stage3_probes)} probes)...")
    with tempfile.TemporaryDirectory(prefix=f"aimids_s3_{engine_key}_") as tmp_dir:
        db_path = os.path.join(tmp_dir, "unified_stage3.db")
        spec_path = os.path.join(tmp_dir, "stage3_spec.json")
        pid_map = _build_stage3_sqlite_db(db_path, plan, spec["store_model"], emb_cache, qual_cache)

        probe_payload = [
            {
                "bucket": pr["bucket"],
                "dataset": pr["dataset"],
                "subject_id": pr["subject_id"],
                "expected_person_id": pid_map.get(pr["subject_id"]) if pr["subject_id"] is not None else None,
                "path": pr["path"],
            }
            for pr in stage3_probes
        ]
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "engine_key": engine_key,
                    "engine_name": spec["engine_name"],
                    "store_model": spec["store_model"],
                    "onnx_path": spec["onnx_path"],
                    "use_gpu": use_gpu,
                    "db_path": db_path,
                    "expected_people": len(pid_map),
                    "probe_items": probe_payload,
                },
                f,
            )

        res = _run_worker_subprocess(["--worker-stage3-http", spec_path], "STAGE3_JSON: ")
        if not res.get("all_passed") or res.get("exit_code") != 0:
            raise GuardError(f"Stage 3 live HTTP parity failed for {engine_key}: {res}")
        return res


# ---------------------------------------------------------------------------
# Stage 2: Unified All-5-Dataset Evaluation & Pairwise McNemar Comparison
# ---------------------------------------------------------------------------


def _summarize_stage2_slice(
    records: Sequence[Dict[str, Any]],
    n_exact_dup: int,
    n_near_dup: int,
    bootstrap_n: int,
    seed: int,
    include_age_buckets: bool = False,
) -> Dict[str, Any]:
    n_probes = len(records)
    denom = float(n_probes) if n_probes > 0 else 1.0
    n_failed = sum(1 for r in records if r["failed_probe_detect"])
    n_r1 = sum(1 for r in records if r["hit_1"])
    n_r5 = sum(1 for r in records if r["hit_5"])
    n_r10 = sum(1 for r in records if r["hit_10"])
    n_strong = sum(1 for r in records if r["is_strong"])
    n_poss = sum(1 for r in records if r["is_possible_plus"])

    # Centroid vs photo match share across non-failed queries and across Rank-1 hits
    non_failed = [r for r in records if not r["failed_probe_detect"] and r.get("top1_match_source") is not None]
    r1_hits = [r for r in records if r["hit_1"] and r.get("top1_match_source") is not None]
    top1_cent_share = (
        sum(1 for r in non_failed if r["top1_match_source"] == "centroid") / float(len(non_failed)) * 100.0
        if non_failed
        else 0.0
    )
    r1_cent_share = (
        sum(1 for r in r1_hits if r["top1_match_source"] == "centroid") / float(len(r1_hits)) * 100.0
        if r1_hits
        else 0.0
    )

    r1_ci, t10_ci = cluster_bootstrap_ci(
        [r["subject_id"] for r in records],
        [r["hit_1"] for r in records],
        [r["hit_10"] for r in records],
        n_boot=bootstrap_n,
        seed=seed,
    )

    out: Dict[str, Any] = {
        "probes": int(n_probes),
        "n_failed_probe_detect": int(n_failed),
        "n_exact_dup_excluded": int(n_exact_dup),
        "n_near_dup_excluded": int(n_near_dup),
        "rank1": (float(n_r1) / denom) * 100.0 if n_probes else 0.0,
        "top5": (float(n_r5) / denom) * 100.0 if n_probes else 0.0,
        "top10": (float(n_r10) / denom) * 100.0 if n_probes else 0.0,
        "strong": (float(n_strong) / denom) * 100.0 if n_probes else 0.0,
        "possible_plus": (float(n_poss) / denom) * 100.0 if n_probes else 0.0,
        "rank1_ci95": r1_ci,
        "top10_ci95": t10_ci,
        "top1_centroid_share_pct": round(top1_cent_share, 2),
        "top1_photo_share_pct": round(100.0 - top1_cent_share, 2) if non_failed else 0.0,
        "r1_centroid_share_pct": round(r1_cent_share, 2),
    }

    if include_age_buckets:
        buckets: Dict[str, Any] = {}
        for bname in ("<10", "10-19", ">=20"):
            brecs = [r for r in records if r.get("age_bucket") == bname]
            bn = len(brecs)
            bden = float(bn) if bn > 0 else 1.0
            buckets[bname] = {
                "probes": int(bn),
                "rank1": (sum(1 for r in brecs if r["hit_1"]) / bden) * 100.0 if bn else 0.0,
                "top5": (sum(1 for r in brecs if r["hit_5"]) / bden) * 100.0 if bn else 0.0,
                "top10": (sum(1 for r in brecs if r["hit_10"]) / bden) * 100.0 if bn else 0.0,
                "strong": (sum(1 for r in brecs if r["is_strong"]) / bden) * 100.0 if bn else 0.0,
            }
        out["age_gap_buckets"] = buckets

    return out


def evaluate_unified_gallery(
    plan: Dict[str, Any],
    engine_key: str,
    spec: Dict[str, Any],
    emb_cache: Dict[str, Optional[np.ndarray]],
    qual_cache: Dict[str, float],
    bundles_by_mode: Dict[str, CalibrationBundle],
    bootstrap_n: int = 1000,
    seed: int = 0,
    near_dup_threshold: float = NEAR_DUP_COS_THRESHOLD,
    sha256_cache: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Evaluates one engine against the unified all-5-dataset FaceStore(":memory:") gallery
    plus unseen strangers in each CalibrationBundle mode (`matched_calibrated`, `production_bundle`).
    """
    hash_memo = sha256_cache if sha256_cache is not None else {}

    def _get_sha(path: str) -> str:
        norm_p = os.path.normcase(os.path.abspath(path))
        dig = hash_memo.get(norm_p)
        if dig is None:
            dig = compute_file_sha256(path)
            hash_memo[norm_p] = dig
        return dig

    gallery_by_id = plan["gallery_by_id"]
    mated_probes = plan["mated_probes"]
    stranger_probes = plan["stranger_probes"]

    gal_hashes_by_id: Dict[str, Set[str]] = defaultdict(set)
    valid_gal_embs_by_id: Dict[str, List[np.ndarray]] = defaultdict(list)
    max_gal_age_by_id: Dict[str, Optional[int]] = {}
    hash_to_subjects: Dict[str, Set[str]] = defaultdict(set)

    for subj_id, items in gallery_by_id.items():
        ages = [age for age, _ in items if age is not None]
        max_gal_age_by_id[subj_id] = max(ages) if ages else None
        for _, path in items:
            dig = _get_sha(path)
            gal_hashes_by_id[subj_id].add(dig)
            hash_to_subjects[dig].add(str(subj_id))
            vec = _safe_unit_vec(emb_cache.get(path))
            if vec is not None:
                valid_gal_embs_by_id[subj_id].append(vec)

    probe_meta: List[Dict[str, Any]] = []
    exact_dup_by_ds: Dict[str, int] = defaultdict(int)
    near_dup_by_ds: Dict[str, int] = defaultdict(int)

    for pr in mated_probes:
        subj_id = str(pr["subject_id"])
        ds = str(pr["dataset"])
        p_path = pr["path"]
        p_dig = _get_sha(p_path)
        hash_to_subjects[p_dig].add(subj_id)

        is_exact_dup = p_dig in gal_hashes_by_id.get(subj_id, set())
        if is_exact_dup:
            exact_dup_by_ds[ds] += 1

        p_vec = _safe_unit_vec(emb_cache.get(p_path))
        is_near_dup = False
        if not is_exact_dup and p_vec is not None and valid_gal_embs_by_id.get(subj_id):
            max_cos = max(float(np.dot(p_vec, g_vec)) for g_vec in valid_gal_embs_by_id[subj_id])
            if max_cos >= near_dup_threshold:
                is_near_dup = True
                near_dup_by_ds[ds] += 1

        age_bucket = None
        if ds == "fgnet" and pr.get("age") is not None and max_gal_age_by_id.get(subj_id) is not None:
            gap = int(pr["age"]) - int(max_gal_age_by_id[subj_id])  # type: ignore[arg-type]
            age_bucket = "<10" if gap < 10 else ("10-19" if gap < 20 else ">=20")

        probe_meta.append({
            "probe_key": _make_probe_key(subj_id, p_path),
            "subject_id": subj_id,
            "dataset": ds,
            "path": p_path,
            "emb": p_vec,
            "is_exact_dup": is_exact_dup,
            "is_near_dup": is_near_dup,
            "failed_probe_detect": p_vec is None,
            "age_bucket": age_bucket,
        })

    stranger_meta = [
        {
            "subject_id": str(pr["subject_id"]),
            "path": pr["path"],
            "emb": _safe_unit_vec(emb_cache.get(pr["path"])),
        }
        for pr in stranger_probes
    ]

    caf_in_train = bool(should_skip_caf(spec.get("onnx_path"), engine_name=spec["engine_name"]))
    modes_summary: Dict[str, Any] = {}
    probe_records_by_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for mode_name, bundle in bundles_by_mode.items():
        store_model = f"{spec['store_model']}__{mode_name}"
        store = FaceStore(":memory:")
        try:
            store.set_calibration_bundle(store_model, bundle)
            pid_map: Dict[str, int] = {}
            for subj_id in sorted(gallery_by_id.keys()):
                pid = store.add_person(str(subj_id))
                added = 0
                for age, path in gallery_by_id[subj_id]:
                    vec = _safe_unit_vec(emb_cache.get(path))
                    if vec is None:
                        continue
                    store.add_photo(
                        pid,
                        vec,
                        photo_path=path,
                        age=age,
                        model=store_model,
                        quality=_clamp_quality(qual_cache.get(path, 1.0)),
                    )
                    added += 1
                if added > 0:
                    pid_map[subj_id] = pid
                else:
                    store.delete_person(pid)

            is_cal_ready = bool(store.calibrated_ready(store_model))
            if not is_cal_ready:
                raise GuardError(f"FaceStore calibrated_ready unexpectedly False for '{store_model}'")

            # Warm up precomputed cohort stats once before measuring query latency
            first_vec = next((m["emb"] for m in probe_meta if m["emb"] is not None), None)
            if first_vec is not None and pid_map:
                _ = store.search(first_vec, k=10, min_sim=-1.0, model=store_model, probe_quality=1.0, age_gap=None)

            evaluated_records: List[Dict[str, Any]] = []
            mode_probe_map: Dict[str, Dict[str, Any]] = {}
            total_search_sec = 0.0
            n_searches = 0

            for item in probe_meta:
                pkey = item["probe_key"]
                if item["is_exact_dup"]:
                    mode_probe_map[pkey] = {
                        "dataset": item["dataset"],
                        "subject_id": item["subject_id"],
                        "hit_1": False,
                        "is_exact_dup": True,
                        "is_near_dup": False,
                    }
                    continue

                emb_p = item["emb"]
                target_pid = pid_map.get(item["subject_id"])
                top1_source = None
                if emb_p is None or target_pid is None:
                    hit_1 = hit_5 = hit_10 = is_strong = is_poss = False
                else:
                    q_probe = _clamp_quality(qual_cache.get(item["path"], 1.0))
                    t_q0 = time.perf_counter()
                    hits = store.search(
                        emb_p, k=10, min_sim=-1.0, model=store_model, probe_quality=q_probe, age_gap=None
                    )
                    total_search_sec += time.perf_counter() - t_q0
                    n_searches += 1
                    hit_pids = [h["person_id"] for h in hits]
                    hit_1 = bool(hit_pids and hit_pids[0] == target_pid)
                    hit_5 = bool(target_pid in hit_pids[:5])
                    hit_10 = bool(target_pid in hit_pids[:10])
                    top1_source = hits[0].get("match_source") if hits else None
                    target_hit = next((h for h in hits if h["person_id"] == target_pid), None)
                    is_strong = bool(target_hit is not None and target_hit["tier"] == "STRONG")
                    is_poss = bool(target_hit is not None and target_hit["tier"] in ("STRONG", "possible"))

                rec = {
                    "subject_id": item["subject_id"],
                    "dataset": item["dataset"],
                    "probe_key": pkey,
                    "is_near_dup": item["is_near_dup"],
                    "failed_probe_detect": item["failed_probe_detect"],
                    "age_bucket": item["age_bucket"],
                    "hit_1": hit_1,
                    "hit_5": hit_5,
                    "hit_10": hit_10,
                    "is_strong": is_strong,
                    "is_possible_plus": is_poss,
                    "top1_match_source": top1_source,
                }
                evaluated_records.append(rec)
                mode_probe_map[pkey] = {
                    "dataset": item["dataset"],
                    "subject_id": item["subject_id"],
                    "hit_1": hit_1,
                    "is_exact_dup": False,
                    "is_near_dup": item["is_near_dup"],
                }

            # Evaluate 200 unseen open-set strangers
            s_llrs: List[float] = []
            s_pnon: List[float] = []
            s_strong = 0
            s_poss = 0
            s_failed = 0
            for sm in stranger_meta:
                emb_s = sm["emb"]
                if emb_s is None:
                    s_failed += 1
                    continue
                q_s = _clamp_quality(qual_cache.get(sm["path"], 1.0))
                t_q0 = time.perf_counter()
                shits = store.search(emb_s, k=10, min_sim=-1.0, model=store_model, probe_quality=q_s, age_gap=None)
                total_search_sec += time.perf_counter() - t_q0
                n_searches += 1
                if shits:
                    top_s = shits[0]
                    s_llrs.append(float(top_s["calibrated_llr"]))
                    s_pnon.append(float(top_s["p_nonmate"]))
                    if top_s["tier"] == "STRONG":
                        s_strong += 1
                    if top_s["tier"] in ("STRONG", "possible"):
                        s_poss += 1

            n_str = len(stranger_meta)
            str_denom = float(n_str) if n_str > 0 else 1.0
            stranger_summary = {
                "n_strangers": int(n_str),
                "n_failed_detect": int(s_failed),
                "fpir_strong_pct": round((s_strong / str_denom) * 100.0, 2) if n_str else 0.0,
                "fpir_possible_plus_pct": round((s_poss / str_denom) * 100.0, 2) if n_str else 0.0,
                "mean_max_llr": round(float(np.mean(s_llrs)), 4) if s_llrs else None,
                "mean_p_nonmate": round(float(np.mean(s_pnon)), 4) if s_pnon else None,
            }

            clean_all = [r for r in evaluated_records if not r["is_near_dup"]]
            clean_fair4 = [r for r in clean_all if r["dataset"] != "caf"]

            fair4_ex = sum(exact_dup_by_ds[d] for d in ("fgnet", "ecaf", "agedb", "calfw"))
            fair4_nd = sum(near_dup_by_ds[d] for d in ("fgnet", "ecaf", "agedb", "calfw"))
            all5_ex = sum(exact_dup_by_ds.values())
            all5_nd = sum(near_dup_by_ds.values())

            per_dataset_out: Dict[str, Any] = {}
            for ds in DATASET_ORDER:
                ds_recs = [r for r in clean_all if r["dataset"] == ds]
                if not ds_recs and exact_dup_by_ds[ds] == 0 and near_dup_by_ds[ds] == 0:
                    continue
                ds_sum = _summarize_stage2_slice(
                    ds_recs,
                    n_exact_dup=exact_dup_by_ds[ds],
                    n_near_dup=near_dup_by_ds[ds],
                    bootstrap_n=bootstrap_n,
                    seed=seed,
                    include_age_buckets=(ds == "fgnet"),
                )
                if ds == "caf":
                    ds_sum["caf_in_train_set"] = caf_in_train
                per_dataset_out[ds] = ds_sum

            modes_summary[mode_name] = {
                "calibrated_ready": is_cal_ready,
                "caf_in_train_set": caf_in_train,
                "gallery_ids_total": len(gallery_by_id),
                "gallery_ids_enrolled": len(pid_map),
                "fair_4_datasets_no_caf": _summarize_stage2_slice(
                    clean_fair4, fair4_ex, fair4_nd, bootstrap_n=bootstrap_n, seed=seed
                ),
                "all_5_datasets": {
                    **_summarize_stage2_slice(clean_all, all5_ex, all5_nd, bootstrap_n=bootstrap_n, seed=seed),
                    "caf_in_train_set": caf_in_train,
                },
                "per_dataset": per_dataset_out,
                "stranger_open_set": stranger_summary,
                "latency_ms_per_query": round((total_search_sec * 1000.0) / max(1, n_searches), 2),
            }
            probe_records_by_mode[mode_name] = mode_probe_map
        finally:
            store.db.close()

    colliding = sum(1 for sids in hash_to_subjects.values() if len(sids) > 1)
    summary = {
        "engine_key": engine_key,
        "engine_name": spec["engine_name"],
        "caf_in_train_set": caf_in_train,
        "cross_id_byte_collisions": colliding,
        "modes": modes_summary,
    }
    return summary, probe_records_by_mode


def build_full_compare(
    engines: Sequence[str],
    artifacts_by_engine: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Builds paired exact McNemar comparisons across all evaluated engines, modes, and subsets."""
    if len(engines) < 2:
        return {}

    comparisons: Dict[str, Any] = {}
    subset_defs = (
        ("fair_4_datasets_no_caf", lambda ds: ds != "caf"),
        ("all_5_datasets", lambda ds: True),
        *[(ds, (lambda target: (lambda x: x == target))(ds)) for ds in DATASET_ORDER],
    )

    for eng_a, eng_b in itertools.combinations(engines, 2):
        pair_key = f"{eng_a}_vs_{eng_b}"
        rec_a = artifacts_by_engine[eng_a].get("probe_records", {})
        rec_b = artifacts_by_engine[eng_b].get("probe_records", {})
        mode_out: Dict[str, Any] = {}

        for mode_name in STAGE2_CALIB_MODES:
            ma = rec_a.get(mode_name)
            mb = rec_b.get(mode_name)
            if ma is None or mb is None:
                continue
            common_keys = sorted(set(ma.keys()) & set(mb.keys()))
            sub_out: Dict[str, Any] = {}

            for sub_name, pred in subset_defs:
                sub_keys = [k for k in common_keys if pred(ma[k]["dataset"])]
                clean_keys = [
                    k
                    for k in sub_keys
                    if not ma[k]["is_exact_dup"]
                    and not ma[k]["is_near_dup"]
                    and not mb[k]["is_exact_dup"]
                    and not mb[k]["is_near_dup"]
                ]
                if not sub_keys and sub_name in DATASET_ORDER:
                    continue
                hits_a = [bool(ma[k]["hit_1"]) for k in clean_keys]
                hits_b = [bool(mb[k]["hit_1"]) for k in clean_keys]
                res = mcnemar_exact(hits_a, hits_b)
                res["n_shared_total"] = len(sub_keys)
                res["n_excluded_dup_either"] = len(sub_keys) - len(clean_keys)
                sub_out[sub_name] = res

            mode_out[mode_name] = sub_out

        comparisons[pair_key] = {
            "engine_a": eng_a,
            "engine_b": eng_b,
            "modes": mode_out,
        }

    return comparisons


def print_full_pipeline_summary(
    engines: Sequence[str],
    stage1_res: Dict[str, Any],
    stage2_res: Dict[str, Any],
    stage3_res: Dict[str, Any],
    comparisons: Dict[str, Any],
) -> None:
    """Prints clean plain-language summary tables across Stage 1, Stage 2, Stage 3, and McNemar."""
    print("\n" + "=" * 116)
    print("  AI-MIDS UNIFIED FULL-PIPELINE EVALUATION SUMMARY")
    print("=" * 116)

    print("\n  [Stage 1: FastAPI Missing-Persons HTTP Smoke]")
    for ek in engines:
        s1 = stage1_res.get(ek, {})
        print(
            f"    {ek:<16} | {s1.get('passed')}/{s1.get('total')} checks | "
            f"cal={s1.get('calibration')} | gpu={s1.get('gpu')} | asnorm={s1.get('asnorm_active')}"
        )

    print("\n  [Stage 2: Unified All-5-Datasets Combined Gallery]")
    print(
        f"    {'Engine':<15} | {'Mode':<18} | {'Subset':<22} | {'Probes':>6} | {'Rank-1':>7} | "
        f"{'Top-10':>7} | {'STRONG':>7} | {'StrFP1%':>7} | {'ms/q':>6}"
    )
    print("    " + "-" * 108)
    for ek in engines:
        s2 = stage2_res[ek]
        for mode_name, m_val in s2["modes"].items():
            str_fp = m_val["stranger_open_set"]["fpir_strong_pct"]
            lat = m_val["latency_ms_per_query"]
            for sub_name in ("fair_4_datasets_no_caf", "all_5_datasets"):
                sv = m_val[sub_name]
                tag = f"{sub_name}{'*' if sub_name == 'all_5_datasets' and s2['caf_in_train_set'] else ''}"
                print(
                    f"    {ek:<15} | {mode_name:<18} | {tag:<22} | {sv['probes']:>6} | "
                    f"{sv['rank1']:>6.2f}% | {sv['top10']:>6.2f}% | {sv['strong']:>6.2f}% | "
                    f"{str_fp:>6.2f}% | {lat:>6.2f}"
                )

    print("\n  [Stage 3: Live HTTP POST /api/search Parity (30 Single-Face Probes)]")
    for ek in engines:
        s3 = stage3_res.get(ek, {})
        print(
            f"    {ek:<16} | parity={s3.get('rank1_parity_count')}/{s3.get('n_probes')} "
            f"({s3.get('rank1_parity_pct', 0.0):.1f}%) | max_llr_diff={s3.get('max_llr_diff')} | "
            f"max_sim_diff={s3.get('max_sim_diff')} | http_mean={s3.get('http_latency_mean_ms')} ms"
        )

    if comparisons:
        print("\n  [Paired Exact McNemar (matched_calibrated)]")
        for pair_key, cdata in comparisons.items():
            m_cal = cdata["modes"].get("matched_calibrated", {})
            for sub_name in ("fair_4_datasets_no_caf", "all_5_datasets"):
                if sub_name in m_cal:
                    r = m_cal[sub_name]
                    print(
                        f"    {pair_key:<34} | {sub_name:<22} | n={r['n']:<4} | "
                        f"{cdata['engine_a']}={r['acc_a']:.2f}% vs {cdata['engine_b']}={r['acc_b']:.2f}% "
                        f"(diff={r['diff_pct']:+.2f}%, p={r['p_value']:.4g}, sig={r['significant_p05']})"
                    )
    print("=" * 116 + "\n")


# ---------------------------------------------------------------------------
# CLI & Orchestration
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-MIDS Unified 3-Stage Full-Pipeline Evaluation Suite"
    )
    parser.add_argument(
        "--engines",
        type=str,
        default=",".join(DEFAULT_ENGINES),
        help="Comma-separated engine keys (default: aimids_crossage,antelopev2,fromscratchv2)",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Cluster bootstrap replicates (default: 1000)")
    parser.add_argument("--verify-sample", type=int, default=24, help="Live cache verification sample size (default: 24)")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed (default: 0)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution instead of CUDA GPU")
    parser.add_argument(
        "--out",
        type=str,
        default="Arc face/bench_results/full_pipeline",
        help="Output directory for JSON reports (default: Arc face/bench_results/full_pipeline)",
    )
    parser.add_argument("--worker-api-smoke", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-stage3-http", type=str, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    use_gpu = not args.cpu

    try:
        if args.worker_api_smoke:
            return run_stage1_api_smoke_worker(args.worker_api_smoke, use_gpu=use_gpu)
        if args.worker_stage3_http:
            return run_stage3_http_worker(args.worker_stage3_http)

        t0 = time.time()
        engine_keys = parse_engine_keys(args.engines)
        out_dir = args.out if os.path.isabs(args.out) else os.path.abspath(os.path.join(HERE, args.out))

        # 1. Shared LAG cohort (never overwrites lag_cohort_cache.npz)
        cohort_bundle = ensure_lag_cohort_for_engines(engine_keys, use_gpu=use_gpu)

        # 2. Deterministic deduplicated 5-dataset plan
        plan, plan_meta = load_default_5dataset_inputs()
        print(f">> Unified 5-dataset plan counts: {json.dumps(plan['counts'])}")

        # 3. Embeddings, cache integrity, and single-face Stage 3 candidates
        emb_bundle = prepare_engine_embeddings_and_stage3_candidates(
            engine_keys,
            plan,
            use_gpu=use_gpu,
            verify_sample=args.verify_sample,
            seed=args.seed,
        )
        caches_by_engine = emb_bundle["caches_by_engine"]
        stage3_probes = select_stage3_probes(
            plan, emb_bundle["single_face_flags"], caches_by_engine, engine_keys, per_bucket=6
        )

        # 4. Matched 4-layer calibration fitting (saves ONLY fromscratchv2 bundle)
        save_paths = {}
        if "fromscratchv2" in engine_keys:
            save_paths["fromscratchv2"] = os.path.join(
                ARC_DIR, "pretrained", f"calib_{sanitize_onnx_stem(SCRATCH_V2_ONNX_PATH)}.npz"
            )
        matched_bundles, calib_diag = fit_matched_calibration_bundle(
            engine_keys,
            plan,
            caches_by_engine,
            cohort_bundle["cohorts"],
            save_paths=save_paths,
            cohort_identities=cohort_bundle["identities"],
            cohort_is_young=cohort_bundle["is_young"],
            proj_k=5,
            photo_hybrid_weight=0.35,
        )

        # Load on-disk production bundles (never overwrites aimids_crossage / antelopev2 bundles)
        prod_bundles: Dict[str, CalibrationBundle] = {}
        for ek in engine_keys:
            prod_path = _resolve_prod_bundle_path(ENGINE_SPECS[ek])
            if not os.path.isfile(prod_path):
                raise GuardError(f"Production calibration bundle not found for '{ek}': {prod_path}")
            prod_bundles[ek] = CalibrationBundle.load(prod_path)

        # 5. Stage 1, Stage 2, and Stage 3 per engine
        stage1_results: Dict[str, Any] = {}
        stage2_results: Dict[str, Any] = {}
        stage3_results: Dict[str, Any] = {}
        artifacts_by_engine: Dict[str, Any] = {}
        sha256_cache: Dict[str, str] = {}

        for ek in engine_keys:
            spec = ENGINE_SPECS[ek]
            emb_cache, qual_cache = caches_by_engine[ek]

            stage1_results[ek] = run_stage1_subprocess(ek, use_gpu=use_gpu)

            s2_summary, probe_records = evaluate_unified_gallery(
                plan=plan,
                engine_key=ek,
                spec=spec,
                emb_cache=emb_cache,
                qual_cache=qual_cache,
                bundles_by_mode={
                    "matched_calibrated": matched_bundles[ek],
                    "production_bundle": prod_bundles[ek],
                },
                bootstrap_n=args.bootstrap,
                seed=args.seed,
                sha256_cache=sha256_cache,
            )
            stage2_results[ek] = s2_summary
            artifacts_by_engine[ek] = {"probe_records": probe_records}

            stage3_results[ek] = run_stage3_subprocess(
                ek, plan, emb_cache, qual_cache, stage3_probes, use_gpu=use_gpu
            )

        comparisons = build_full_compare(engine_keys, artifacts_by_engine)
        elapsed_sec = time.time() - t0

        provenance = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cli_args": vars(args),
            "lag_cohort_size": cohort_bundle["cohort_size"],
            "lag_cohort_sha256": sha256_file_if_exists(LAG_CACHE_PATH),
            "lag_scratch_cohort_sha256": sha256_file_if_exists(LAG_SCRATCH_CACHE_PATH),
            "plan_meta": plan_meta,
            "plan_counts": plan["counts"],
            "disjointness_verified": plan["disjointness_verified"],
            "engine_providers": emb_bundle["providers_by_engine"],
            "actual_gpu": emb_bundle["actual_gpu_by_engine"],
            "cache_integrity": emb_bundle["integrity_by_engine"],
            "matched_calibration_fit": calib_diag,
            "stage3_probes": [
                {"bucket": p["bucket"], "dataset": p["dataset"], "file": os.path.basename(p["path"])}
                for p in stage3_probes
            ],
        }

        os.makedirs(out_dir, exist_ok=True)
        written: List[str] = []
        for ek in engine_keys:
            p_eng = os.path.join(out_dir, f"full_pipeline_{ek}.json")
            with open(p_eng, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "provenance": provenance,
                        "runtime_sec": elapsed_sec,
                        "stage1_api_smoke": stage1_results[ek],
                        "stage2_unified_eval": stage2_results[ek],
                        "stage3_http_parity": stage3_results[ek],
                    },
                    f,
                    indent=2,
                )
            written.append(p_eng)

        p_sum = os.path.join(out_dir, "full_pipeline_summary.json")
        with open(p_sum, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provenance": provenance,
                    "runtime_sec": elapsed_sec,
                    "stage1_api_smoke": stage1_results,
                    "stage2_unified_eval": stage2_results,
                    "stage3_http_parity": stage3_results,
                },
                f,
                indent=2,
            )
        written.append(p_sum)

        if len(engine_keys) >= 2:
            p_cmp = os.path.join(out_dir, "full_pipeline_compare.json")
            with open(p_cmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "provenance": provenance,
                        "comparisons": comparisons,
                    },
                    f,
                    indent=2,
                )
            written.append(p_cmp)

        print_full_pipeline_summary(engine_keys, stage1_results, stage2_results, stage3_results, comparisons)
        print(f">> Full pipeline evaluation completed in {elapsed_sec:.1f}s ({elapsed_sec / 60:.2f} min).")
        for wf in written:
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
