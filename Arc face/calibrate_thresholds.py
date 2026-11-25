"""
calibrate_thresholds.py - Offline cohort builder, 4-layer calibrator fitter, and
Before-vs-After A/B benchmark suite for AI-MIDS.

Builds strictly identity-disjoint calibration bundles for `antelopev2` (512-d) and
`aimids_crossage` (1024-d):
  1. External Balanced Cohort (M ~ 1,200 faces across child/young/adult/older ages):
     - Prefers external `LAG-aligned.tar.gz` (zero normalized-name overlap with
       AgeDB, FG-NET, CALFW, and demo DBs) embedded via ONNX with Flip-TTA.
     - Supports `--cohort-source carved` (and automatic fallback under `--cohort-source auto`
       if ONNX cannot load) using strictly identity-disjoint carved embeddings.
  2. Fold-disjoint QMF logistic LLR calibrator (`QMFCalibrator`) audited via C_llr,
     PAV C_llr_min, ECE, and Brier.
  3. Open-set N-scaled Split Conformal + GPD POT calibrator (`OpenSetCalibrator`)
     fitted on real non-mated max-of-N searches across N in [1..3935].
  4. Identity-level 1:1 Split Conformal verifier (`CoveCalibrator`, arXiv:2610.35277).

Outputs:
  - Arc face/pretrained/calib_antelopev2.npz (+ .json)
  - Arc face/pretrained/calib_aimids_crossage.npz (+ .json)
  - Arc face/bench_results/calibration_ab_summary.json

Never modifies demo_aimids.db, demo_crossage.db, or existing bench_results/*.csv files.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gc
import json
import re
import sqlite3
import sys
import tarfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

ARC_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(ARC_DIR)
ASSETS_DIR = r"D:\GRAD\Assets"
sys.path.insert(0, ARC_DIR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from bench_datasets import (
    ALIAS_CANONICAL,
    collect_known_dataset_identities,
    load_benchmark_dataset,
    load_cache,
    load_calfw_local,
    load_quality_cache,
    normalize_identity,
    parse_agedb_images,
)
from calibration import (
    CalibrationBundle,
    CoveCalibrator,
    ExternalCohortASNorm,
    OpenSetCalibrator,
    QMFCalibrator,
    compute_age_subspace_basis,
    compute_cllr,
    compute_brier,
    compute_ece,
    pav_cllr_min,
    project_subspace,
    resolve_bundle_path,
)
from face_engine_custom import calibrate_asnorm_to_cosine, compute_asnorm_scores
from store import FaceStore, STRONG, POSSIBLE


LAG_TAR_PATH = os.path.join(ASSETS_DIR, "LAG-aligned.tar.gz")
LAG_CACHE_PATH = os.path.join(ARC_DIR, "pretrained", "lag_cohort_cache.npz")
AGEDB_DIR = os.path.join(ASSETS_DIR, "AgeDB", "images")
CALFW_DIR = os.path.join(ASSETS_DIR, "calfw")
FGNET_DIR = os.path.join(ASSETS_DIR, "FGNET", "images")

DEMO_DBS = {
    "antelopev2": os.path.join(MAIN_DIR, "demo_aimids.db"),
    "aimids_crossage": os.path.join(MAIN_DIR, "demo_crossage.db"),
}

# Known LFW/CALFW multi-person wrong-crop news photos (same person under two unrelated captions)
LFW_CONTAMINATED_NONMATES = {
    "patrickdennehy", "carltondotson",
    "donnaralston", "kateburton",
    "fredricseaman",
    "maguiserna", "ameliemauresmo",
    "merylstreep", "janefonda",
    "geoffhoon", "jeremygompertz",
    "gradylittle", "kenmacha",
    "pervezmusharraf", "rashidqureshi",
    "barbarabrezigar", "ingridbetancourt",
    "hayleytullett", "paularadcliffe",
    "paulvathis", "wiltonnormanchamberlain",
    "alinaimi", "chakibkhelil",
    "juancarlosferrero", "julienboutter",
    "geovanilapentti", "nicolaslapentti",
    "chenkaige", "tianzhuangzhuang",
    "ivamajoli", "jelenadokic",
    "ianthorpe", "granthackett",
}

_AGEDB_RE = re.compile(r"^(\d+)_(.+?)_(\d+)_([mf])\.jpg$", re.IGNORECASE)
_CALFW_RE = re.compile(r"^(.+?)_(\d{4})\.jpg$", re.IGNORECASE)
_FGNET_RE = re.compile(r"^(\d+)A(\d+)\.jpg$", re.IGNORECASE)


def identity_from_path(path: str) -> Tuple[str, Optional[int]]:
    """Extracts (normalized_identity, optional_age) from an AgeDB, CALFW, or FG-NET file path."""
    base = os.path.basename(str(path))
    m_age = _AGEDB_RE.match(base)
    if m_age:
        return normalize_identity(m_age.group(2)), int(m_age.group(3))
    m_fg = _FGNET_RE.match(base)
    if m_fg:
        return normalize_identity(f"fgnet_{m_fg.group(1)}"), int(m_fg.group(2))
    m_cf = _CALFW_RE.match(base)
    if m_cf:
        return normalize_identity(m_cf.group(1)), None
    stem, _ = os.path.splitext(base)
    return normalize_identity(stem), None


# ---------------------------------------------------------------------------
# Read-only DB & Cache Loaders
# ---------------------------------------------------------------------------


def load_ro_db_gallery(db_path: str, model_name: str) -> Dict[str, Any]:
    """
    Loads persons, faces, and QA-weighted centroids from SQLite in strictly read-only mode.
    Returns dict with person metadata, normalized identity keys, face vectors, and centroids.
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    persons_rows = con.execute("SELECT id, name, notes FROM persons ORDER BY id").fetchall()
    faces_rows = con.execute(
        "SELECT id, person_id, age, embedding, quality FROM faces WHERE model=? ORDER BY id",
        (model_name,),
    ).fetchall()
    con.close()

    p_info = {int(pid): (name, notes) for pid, name, notes in persons_rows}
    faces_by_pid: Dict[int, List[Tuple[int, Optional[int], np.ndarray, float]]] = defaultdict(list)
    for fid, pid, age, blob, qual in faces_rows:
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        q_val = 1.0 if qual is None or not np.isfinite(qual) else float(np.clip(qual, 0.25, 1.0))
        faces_by_pid[int(pid)].append((int(fid), age, vec, q_val))

    # Build centroids identically to FaceStore._update_centroid
    pids = []
    names = []
    norm_keys = []
    cent_vecs = []
    cent_quals = []
    min_ages = []
    max_ages = []

    all_fids = []
    all_fpids = []
    all_fvecs = []
    all_fquals = []
    all_fages = []

    for pid in sorted(faces_by_pid.keys()):
        items = faces_by_pid[pid]
        vecs = np.stack([x[2] for x in items]).astype(np.float32)
        ages = [x[1] for x in items]
        quals = [x[3] for x in items]
        weights = np.empty(len(items), dtype=np.float32)
        for i, (age_i, qual_i) in enumerate(zip(ages, quals)):
            if age_i is None:
                w_age = 1.0
            else:
                age_f = float(age_i)
                nearby = sum(1 for j, a_j in enumerate(ages) if j != i and a_j is not None and abs(float(a_j) - age_f) <= 2.0)
                w_age = 1.0 / (1.0 + nearby)
            weights[i] = qual_i * w_age
        weighted = np.sum(vecs * weights[:, None], axis=0)
        nrm = float(np.linalg.norm(weighted))
        cvec = (weighted / nrm if nrm > 0 else weighted).astype(np.float32)

        pname, _ = p_info.get(pid, (f"Person_{pid}", ""))
        valid_ages = [int(a) for a in ages if a is not None]
        pids.append(pid)
        names.append(pname)
        norm_keys.append(normalize_identity(pname))
        cent_vecs.append(cvec)
        cent_quals.append(float(max(quals)))
        min_ages.append(min(valid_ages) if valid_ages else None)
        max_ages.append(max(valid_ages) if valid_ages else None)

        for fid, age_i, vec_i, q_i in items:
            all_fids.append(fid)
            all_fpids.append(pid)
            all_fvecs.append(vec_i)
            all_fquals.append(q_i)
            all_fages.append(age_i)

    return {
        "pids": np.array(pids, dtype=np.int64),
        "names": names,
        "norm_keys": norm_keys,
        "cent_vecs": np.stack(cent_vecs).astype(np.float32),
        "cent_quals": np.array(cent_quals, dtype=np.float32),
        "min_ages": min_ages,
        "max_ages": max_ages,
        "faces_by_pid": faces_by_pid,
        "fids": np.array(all_fids, dtype=np.int64),
        "fpids": np.array(all_fpids, dtype=np.int64),
        "fvecs": np.stack(all_fvecs).astype(np.float32),
        "fquals": np.array(all_fquals, dtype=np.float32),
        "fages": all_fages,
    }


def load_pair_Caches(model_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Loads cached AgeDB-30 and CALFW embeddings (+ qualities if available) indexed by normalized basename and full path."""
    emb_map: Dict[str, np.ndarray] = {}
    qual_map: Dict[str, float] = {}

    suffix_list = [
        f"bench_cache_agedb30_{model_name}_tta.npz",
        f"bench_cache_calfw_{model_name}_tta.npz",
    ]
    for fn in suffix_list:
        p = os.path.join(ARC_DIR, fn)
        if os.path.isfile(p):
            c = load_cache(p)
            qc = load_quality_cache(p)
            for k, v in c.items():
                if v is not None:
                    vec = np.asarray(v, dtype=np.float32).reshape(-1)
                    nrm = float(np.linalg.norm(vec))
                    if nrm > 1e-12:
                        vec = vec / nrm
                    emb_map[str(k)] = vec
                    emb_map[os.path.basename(str(k)).lower()] = vec
                    if k in qc:
                        qual_map[str(k)] = float(np.clip(qc[k], 0.25, 1.0))
                        qual_map[os.path.basename(str(k)).lower()] = float(np.clip(qc[k], 0.25, 1.0))

    # Cross-populate quality cache from aimids_crossage caches for identical images
    for ds in ("agedb30", "calfw"):
        ca_p = os.path.join(ARC_DIR, f"bench_cache_{ds}_aimids_crossage_tta.npz")
        if os.path.isfile(ca_p):
            qc = load_quality_cache(ca_p)
            for k, qv in qc.items():
                qual_map.setdefault(str(k), float(np.clip(qv, 0.25, 1.0)))
                qual_map.setdefault(os.path.basename(str(k)).lower(), float(np.clip(qv, 0.25, 1.0)))

    return emb_map, qual_map


# ---------------------------------------------------------------------------
# Cohort Extraction (LAG External ONNX or Disjoint Carved Fallback)
# ---------------------------------------------------------------------------


def _extract_lag_embeddings_onnx(max_faces: int = 1200, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Reads LAG-aligned.tar.gz directly in memory, excludes every identity matching AgeDB/CALFW/FG-NET/DB,
    samples a balanced young + adult subset, and embeds with glintr100.onnx + aimids_adaface_ir101_ft.onnx.
    Caches results to pretrained/lag_cohort_cache.npz.
    """
    if os.path.isfile(LAG_CACHE_PATH):
        d = np.load(LAG_CACHE_PATH, allow_pickle=False)
        if "antelopev2" in d and "aimids_crossage" in d and len(d["antelopev2"]) >= 500:
            print(f">> Loaded cached external LAG cohort embeddings from {os.path.basename(LAG_CACHE_PATH)} ({len(d['antelopev2'])} faces)")
            return {
                "antelopev2": d["antelopev2"].astype(np.float32),
                "aimids_crossage": d["aimids_crossage"].astype(np.float32),
                "identities": d["identities"],
                "is_young": d["is_young"],
            }

    if not os.path.isfile(LAG_TAR_PATH):
        raise FileNotFoundError(f"LAG tarball not found: {LAG_TAR_PATH}")

    # Check ONNX availability first before scanning the 76 MB tarball
    import onnxruntime as ort

    glint_path = os.path.expanduser("~/.insightface/models/antelopev2/glintr100.onnx")
    ada_path = os.path.join(ARC_DIR, "pretrained", "aimids_adaface_ir101_ft.onnx")
    if not os.path.isfile(glint_path) or not os.path.isfile(ada_path):
        raise FileNotFoundError(f"Missing ONNX weights: glint={glint_path}, ada={ada_path}")

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    # Smoke-load glintr100 first so memory/commit limits fail fast in <1s
    sess_glint = None
    sess_ada = None
    try:
        sess_glint = ort.InferenceSession(glint_path, sess_options=so, providers=["CPUExecutionProvider"])

        known = collect_known_dataset_identities()
        rng = np.random.default_rng(seed)

        with tarfile.open(LAG_TAR_PATH, "r:gz") as tf:
            by_ident: Dict[str, Dict[str, List[tarfile.TarInfo]]] = defaultdict(lambda: {"y": [], "a": []})
            for m in tf.getmembers():
                if not m.isfile() or not m.name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                parts = m.name.split("/")
                if len(parts) < 3:
                    continue
                ident = parts[1]
                if normalize_identity(ident) in known:
                    continue
                bucket = "y" if "/y/" in m.name else "a"
                by_ident[ident][bucket].append(m)

            disjoint_idents = sorted(by_ident.keys())
            rng.shuffle(disjoint_idents)

            selected: List[Tuple[str, bool, tarfile.TarInfo]] = []
            target_half = max_faces // 2
            young_count = 0
            adult_count = 0

            for ident in disjoint_idents:
                ys = sorted(by_ident[ident]["y"], key=lambda x: x.name)
                ads = sorted(by_ident[ident]["a"], key=lambda x: x.name)
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
                        for m in sorted(by_ident[ident][bucket], key=lambda x: x.name):
                            if m.name not in used_names:
                                selected.append((ident, is_y, m))
                                used_names.add(m.name)
                                if len(selected) >= max_faces:
                                    break
                        if len(selected) >= max_faces:
                            break
                    if len(selected) >= max_faces:
                        break

            # Sort by archive offset_data so gzip reads forward sequentially in 1 pass
            selected.sort(key=lambda s: s[2].offset_data)
            imgs = []
            meta_idents = []
            meta_young = []
            for ident, is_y, member in selected:
                raw = tf.extractfile(member).read()
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[0] != 112 or img.shape[1] != 112:
                    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
                imgs.append(img)
                meta_idents.append(ident)
                meta_young.append(is_y)

        def _run_session(sess, batch_size: int = 8) -> np.ndarray:
            inp_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            vecs = np.empty((len(imgs), 512), dtype=np.float32)
            for start in range(0, len(imgs), batch_size):
                batch_imgs = imgs[start : start + batch_size]
                batch_flips = [cv2.flip(im, 1) for im in batch_imgs]
                b1 = cv2.dnn.blobFromImages(batch_imgs, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                b2 = cv2.dnn.blobFromImages(batch_flips, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                e1 = sess.run([out_name], {inp_name: b1})[0]
                e2 = sess.run([out_name], {inp_name: b2})[0]
                e1 = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
                e2 = e2 / np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12)
                comb = e1 + e2
                comb = comb / np.maximum(np.linalg.norm(comb, axis=1, keepdims=True), 1e-12)
                vecs[start : start + batch_size] = comb.astype(np.float32)
            return vecs

        print(f">> Embedding {len(imgs)} disjoint external LAG faces via glintr100.onnx + aimids_adaface_ir101_ft.onnx...")
        emb_ant = _run_session(sess_glint, batch_size=8)
        del sess_glint
        sess_glint = None
        gc.collect()

        sess_ada = ort.InferenceSession(ada_path, sess_options=so, providers=["CPUExecutionProvider"])
        emb_ada = _run_session(sess_ada, batch_size=8)
        del sess_ada
        sess_ada = None
        gc.collect()
    finally:
        if sess_glint is not None:
            del sess_glint
        if sess_ada is not None:
            del sess_ada
        gc.collect()
    fused = np.concatenate([np.sqrt(0.55) * emb_ant, np.sqrt(0.45) * emb_ada], axis=1).astype(np.float32)
    fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)

    os.makedirs(os.path.dirname(LAG_CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        LAG_CACHE_PATH,
        antelopev2=emb_ant,
        aimids_crossage=fused,
        identities=np.array(meta_idents),
        is_young=np.array(meta_young, dtype=bool),
    )
    print(f">> Saved LAG external cohort cache ({len(imgs)} faces, {len(set(meta_idents))} disjoint identities) -> {LAG_CACHE_PATH}")
    return {
        "antelopev2": emb_ant,
        "aimids_crossage": fused,
        "identities": np.array(meta_idents),
        "is_young": np.array(meta_young, dtype=bool),
    }


def build_carved_balanced_cohort(
    gallery: Dict[str, Any],
    protected_norm_keys: Set[str],
    target_size: int = 1100,
    seed: int = 42,
) -> Tuple[np.ndarray, Set[str], Dict[str, int]]:
    """
    Fallback strictly identity-disjoint balanced cohort carved from gallery identities
    NOT in protected_norm_keys (zero overlap with held-out 1:1 or 1:N test identities).
    Balances faces across age strata (<18, 18-45, >45, CALFW no-age).
    """
    rng = np.random.default_rng(seed)
    eligible_pids_by_key: Dict[str, List[int]] = defaultdict(list)
    for pid, nkey in zip(gallery["pids"].tolist(), gallery["norm_keys"]):
        if nkey not in protected_norm_keys:
            eligible_pids_by_key[nkey].append(pid)

    # Categorize faces by age bin while tracking identity key
    bins: Dict[str, List[Tuple[str, np.ndarray]]] = {"child": [], "adult": [], "older": [], "calfw": []}
    for nkey, pid_list in sorted(eligible_pids_by_key.items()):
        for pid in pid_list:
            for _fid, age_val, vec, _qual in gallery["faces_by_pid"][pid]:
                if age_val is None:
                    bins["calfw"].append((nkey, vec))
                elif age_val < 18:
                    bins["child"].append((nkey, vec))
                elif age_val <= 45:
                    bins["adult"].append((nkey, vec))
                else:
                    bins["older"].append((nkey, vec))

    # Reserve distinct identity keys cleanly across strata
    quotas = {
        "child": min(len(bins["child"]), int(target_size * 0.20)),
        "adult": min(len(bins["adult"]), int(target_size * 0.25)),
        "older": min(len(bins["older"]), int(target_size * 0.25)),
        "calfw": 0,
    }
    used = quotas["child"] + quotas["adult"] + quotas["older"]
    quotas["calfw"] = min(len(bins["calfw"]), max(0, target_size - used))

    cohort_vecs = []
    cohort_keys: Set[str] = set()
    counts = {}

    for bname in ("child", "older", "adult", "calfw"):
        items = list(bins[bname])
        idxs = rng.permutation(len(items))
        chosen = 0
        # Take at most 2 faces per identity to maximize diversity
        per_key_count: Dict[str, int] = defaultdict(int)
        for idx in idxs:
            nkey, vec = items[idx]
            if per_key_count[nkey] >= 2:
                continue
            cohort_vecs.append(vec)
            cohort_keys.add(nkey)
            per_key_count[nkey] += 1
            chosen += 1
            if chosen >= quotas[bname]:
                break
        counts[bname] = chosen

    return np.stack(cohort_vecs).astype(np.float32), cohort_keys, counts


# ---------------------------------------------------------------------------
# 1:1 Pairs Dataset Assembly (AgeDB-30 + CALFW, Strictly Identity-Disjoint)
# ---------------------------------------------------------------------------


def assemble_1to1_trials(
    model_name: str,
    emb_map: Dict[str, np.ndarray],
    qual_map: Dict[str, float],
    excluded_norm_keys: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Assembles real 1:1 verification trials from AgeDB-30 and CALFW caches, keeping
    only pairs whose identities are NOT in excluded_norm_keys.
    """
    agedb_pairs, _ = load_benchmark_dataset("agedb30", limit=None, seed=0)
    calfw_pairs, _ = load_calfw_local(limit=None, seed=0)

    def _collect(raw_list, ds_name):
        out = []
        for path_a, path_b, is_same, fold in raw_list:
            ba = os.path.basename(str(path_a)).lower()
            bb = os.path.basename(str(path_b)).lower()
            ea = emb_map.get(str(path_a), emb_map.get(ba))
            eb = emb_map.get(str(path_b), emb_map.get(bb))
            if ea is None or eb is None:
                continue
            id_a, age_a = identity_from_path(path_a)
            id_b, age_b = identity_from_path(path_b)
            if id_a in excluded_norm_keys or id_b in excluded_norm_keys:
                continue
            qa = qual_map.get(str(path_a), qual_map.get(ba, 1.0))
            qb = qual_map.get(str(path_b), qual_map.get(bb, 1.0))
            age_gap = float(abs(age_a - age_b)) if (age_a is not None and age_b is not None) else np.nan
            out.append({
                "dataset": ds_name,
                "fold": int(fold),
                "is_same": int(is_same),
                "id_a": id_a,
                "id_b": id_b,
                "emb_a": ea,
                "emb_b": eb,
                "q_min": float(min(qa, qb)),
                "age_gap": age_gap,
                "raw_sim": float(np.clip(np.dot(ea, eb), -1.0, 1.0)),
            })
        return out

    return _collect(agedb_pairs, "agedb30"), _collect(calfw_pairs, "calfw")


def eval_1to1_subset(
    trials: List[Dict[str, Any]],
    asnorm: ExternalCohortASNorm,
    qmf: QMFCalibrator,
    cove: Optional[CoveCalibrator] = None,
    proj_basis: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluates Legacy raw vs Calibrated QMF on a 1:1 trial list."""
    if not trials:
        return {}
    embs_a_raw = np.stack([t["emb_a"] for t in trials])
    embs_b_raw = np.stack([t["emb_b"] for t in trials])
    embs_a = project_subspace(embs_a_raw, proj_basis) if proj_basis is not None else embs_a_raw
    embs_b = project_subspace(embs_b_raw, proj_basis) if proj_basis is not None else embs_b_raw
    mu_a, sig_a = asnorm.enroll_stats(embs_a)
    mu_b, sig_b = asnorm.enroll_stats(embs_b)
    raw_cove = np.array([t["raw_sim"] for t in trials], dtype=np.float64)
    raw = (
        np.clip(np.sum(embs_a * embs_b, axis=1), -1.0, 1.0).astype(np.float64)
        if proj_basis is not None
        else raw_cove
    )
    z = 0.5 * ((raw - mu_a) / sig_a + (raw - mu_b) / sig_b)
    q_min = np.array([t["q_min"] for t in trials], dtype=np.float64)
    age_gap = np.array([t["age_gap"] for t in trials], dtype=np.float64)
    labels = np.array([t["is_same"] for t in trials], dtype=np.int32)

    # Fit raw 1D baseline Platt scaling out-of-fold? No: measure raw cosine directly + QMF!
    llrs = qmf.predict_llr(z, raw, q_min=q_min, age_gap=age_gap)
    probs = qmf.predict_prob(llrs, pi=0.5)

    def _acc_and_tar(scores):
        # 5-fold threshold accuracy over the folds present in trials
        present_folds = sorted({t["fold"] for t in trials})
        accs = []
        s_arr = np.asarray(scores, dtype=np.float64)
        grid = np.linspace(float(np.min(s_arr)), float(np.max(s_arr)), 501)
        for tf in present_folds:
            tr_mask = np.array([t["fold"] != tf for t in trials])
            te_mask = ~tr_mask
            if np.any(tr_mask) and np.any(te_mask):
                tr_s, tr_y = s_arr[tr_mask], labels[tr_mask]
                preds = tr_s[None, :] >= grid[:, None]
                tau = float(grid[int(np.argmax(np.mean(preds == tr_y[None, :], axis=1)))])
                accs.append(float(np.mean((s_arr[te_mask] >= tau) == labels[te_mask])) * 100.0)
        imp = s_arr[labels == 0]
        gen = s_arr[labels == 1]
        tau_far1 = float(np.quantile(imp, 0.99)) if imp.size else 0.0
        tar_far1 = float(np.mean(gen >= tau_far1)) * 100.0 if gen.size else 0.0
        return float(np.mean(accs)) if accs else 0.0, tar_far1

    raw_acc, raw_tar1 = _acc_and_tar(raw)
    cal_acc, cal_tar1 = _acc_and_tar(llrs)

    # Naive baseline logistic LLR from raw cosine (for fair C_llr comparison against QMF)
    naive_prob = 1.0 / (1.0 + np.exp(-np.clip(raw * 10.0 - 2.5, -30.0, 30.0)))
    raw_cllr = compute_cllr(raw * 10.0 - 2.5, labels)
    raw_cllr_min = pav_cllr_min(raw, labels)

    cal_cllr = compute_cllr(llrs, labels)
    cal_cllr_min = pav_cllr_min(llrs, labels)

    out = {
        "n_trials": len(trials),
        "n_genuine": int(np.sum(labels == 1)),
        "n_impostor": int(np.sum(labels == 0)),
        "legacy_raw": {
            "acc_pct": round(raw_acc, 2),
            "tar_at_far1_pct": round(raw_tar1, 2),
            "cllr_uncalibrated_logit": round(raw_cllr, 4),
            "pav_cllr_min": round(raw_cllr_min, 4),
            "brier": round(compute_brier(naive_prob, labels), 4),
            "ece": round(compute_ece(naive_prob, labels), 4),
        },
        "calibrated_qmf": {
            "acc_pct": round(cal_acc, 2),
            "tar_at_far1_pct": round(cal_tar1, 2),
            "cllr": round(cal_cllr, 4),
            "pav_cllr_min": round(cal_cllr_min, 4),
            "cllr_cal": round(max(0.0, cal_cllr - cal_cllr_min), 4),
            "brier": round(compute_brier(probs, labels), 4),
            "ece": round(compute_ece(probs, labels), 4),
        },
    }

    if cove is not None:
        imp_raw = raw_cove[labels == 0]
        gen_raw = raw_cove[labels == 1]
        tau_cove_1 = cove.threshold(alpha=0.01)
        tau_cove_01 = cove.threshold(alpha=0.001)
        out["cove_1to1"] = {
            "tau_alpha_1pct": round(tau_cove_1, 4),
            "emp_far_at_1pct": round(float(np.mean(imp_raw >= tau_cove_1)) * 100.0, 3),
            "emp_tar_at_1pct": round(float(np.mean(gen_raw >= tau_cove_1)) * 100.0, 2),
            "tau_alpha_01pct": round(tau_cove_01, 4),
            "emp_far_at_01pct": round(float(np.mean(imp_raw >= tau_cove_01)) * 100.0, 3),
            "emp_tar_at_01pct": round(float(np.mean(gen_raw >= tau_cove_01)) * 100.0, 2),
        }
    return out


# ---------------------------------------------------------------------------
# Fast Vectorized Open-Set 1:N Search Simulator (Exact Legacy vs Calibrated)
# ---------------------------------------------------------------------------


def _simulate_nonmate_max_scores(
    probe_vecs: np.ndarray,
    probe_quals: np.ndarray,
    probe_keys: List[str],
    gallery_pids: np.ndarray,
    gallery_cent_vecs: np.ndarray,
    gallery_cent_quals: np.ndarray,
    gallery_norm_keys: List[str],
    gallery_faces_by_pid: Dict[int, List[Tuple[int, Optional[int], np.ndarray, float]]],
    asnorm: ExternalCohortASNorm,
    qmf: QMFCalibrator,
    n_sizes: List[int],
    seed: int = 123,
    photo_hybrid_weight: float = 1.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """
    Runs exact vectorized 1:N searches over gallery sizes in `n_sizes` for both:
      - Legacy raw_fused: max(best_photo_sim, raw_cent_sim)
      - Legacy asnorm_fused: max(best_photo_sim, legacy_asnorm_cal_cent)
      - New 4-layer calibrated LLR: max(qmf_llr_photo, qmf_llr_cent)
    Also returns base N=1 person-level calibrated LLRs for GPD POT fitting.
    """
    rng = np.random.default_rng(seed)
    # Index gallery persons cleanly (exclude any person whose normalized key is in probe_keys)
    probe_key_set = set(probe_keys)
    valid_g_indices = [i for i, nk in enumerate(gallery_norm_keys) if nk not in probe_key_set]
    rng.shuffle(valid_g_indices)

    # Precompute gallery face arrays ordered by valid_g_indices
    max_n = min(max(n_sizes), len(valid_g_indices))
    sub_idx = np.array(valid_g_indices[:max_n], dtype=np.int64)
    sub_pids = gallery_pids[sub_idx]
    sub_cvecs = gallery_cent_vecs[sub_idx]
    sub_cquals = gallery_cent_quals[sub_idx]
    mu_c_all, sig_c_all = asnorm.enroll_stats(sub_cvecs)

    # Flatten faces belonging to sub_pids, keeping each face's gallery rank position 0..max_n-1
    pid_to_pos = {int(pid): pos for pos, pid in enumerate(sub_pids.tolist())}
    f_pos_list = []
    f_vecs_list = []
    f_quals_list = []
    for pos, pid in enumerate(sub_pids.tolist()):
        for _fid, _age, fvec, fqual in gallery_faces_by_pid[pid]:
            f_pos_list.append(pos)
            f_vecs_list.append(fvec)
            f_quals_list.append(fqual)

    f_pos = np.array(f_pos_list, dtype=np.int64)
    f_vecs = np.stack(f_vecs_list).astype(np.float32)
    f_quals = np.array(f_quals_list, dtype=np.float32)
    mu_f_all, sig_f_all = asnorm.enroll_stats(f_vecs)

    # Probe -> all gallery centroids & faces
    # Chunk probes so peak matrix memory is tiny (< 50 MB)
    m_probes = len(probe_vecs)
    cal_llr_by_n = {n: np.empty(m_probes, dtype=np.float32) for n in n_sizes if n <= max_n}
    leg_raw_by_n = {n: np.empty(m_probes, dtype=np.float32) for n in n_sizes if n <= max_n}
    leg_asn_by_n = {n: np.empty(m_probes, dtype=np.float32) for n in n_sizes if n <= max_n}

    # Legacy AS-Norm cohorts per N (linspace up to 300 from sub_cvecs[:n])
    leg_cohorts = {}
    for n in cal_llr_by_n:
        if n >= 80:
            c_sub = sub_cvecs[:n]
            c_pids = sub_pids[:n]
            if n > 300:
                c_idx = np.linspace(0, n - 1, num=300, dtype=np.int64)
                leg_cohorts[n] = (c_pids[c_idx], c_sub[c_idx])
            else:
                leg_cohorts[n] = (c_pids, c_sub)

    base_samples = []

    for p_i in range(m_probes):
        q = probe_vecs[p_i]
        q_qual = float(probe_quals[p_i])
        p_stats = asnorm.probe_stats(q)

        # Centroid scores across all max_n
        raw_c = (sub_cvecs @ q).astype(np.float32)
        z_c = asnorm.score_batch(q, cand_mu=mu_c_all, cand_sigma=sig_c_all, raw_sims=raw_c, probe_stats=p_stats)
        llr_c = qmf.predict_llr(z_c, raw_c, np.minimum(q_qual, sub_cquals))

        # Photo scores across all faces of max_n
        raw_f = (f_vecs @ q).astype(np.float32)
        z_f = asnorm.score_batch(q, cand_mu=mu_f_all, cand_sigma=sig_f_all, raw_sims=raw_f, probe_stats=p_stats)
        llr_f = qmf.predict_llr(z_f, raw_f, np.minimum(q_qual, f_quals))

        # Per-person max over photos via np.maximum.at
        max_raw_f_per_pos = np.full(max_n, -1.0, dtype=np.float32)
        np.maximum.at(max_raw_f_per_pos, f_pos, raw_f)

        max_llr_f_per_pos = np.full(max_n, -1e9, dtype=np.float64)
        np.maximum.at(max_llr_f_per_pos, f_pos, llr_f)

        w_h = float(np.clip(photo_hybrid_weight, 0.0, 1.0))
        person_llr = llr_c + w_h * np.maximum(0.0, max_llr_f_per_pos - llr_c)
        person_raw = np.maximum(raw_c, max_raw_f_per_pos)

        if p_i < 200:
            take = rng.choice(max_n, size=min(300, max_n), replace=False)
            base_samples.append(person_llr[take].astype(np.float32))

        for n in sorted(cal_llr_by_n.keys()):
            cal_llr_by_n[n][p_i] = float(np.max(person_llr[:n]))
            leg_raw_by_n[n][p_i] = float(np.max(person_raw[:n]))

            if n < 80:
                leg_asn_by_n[n][p_i] = leg_raw_by_n[n][p_i]
            else:
                # Exact replication of store.py legacy AS-Norm: top-200 raw candidates, centroid-only AS-Norm, then max(photo, cal_cent)
                c_pids_coh, c_mat_coh = leg_cohorts[n]
                raw_n = person_raw[:n]
                if n <= 300:
                    rescore_pos = np.arange(n, dtype=np.int64)
                else:
                    n_res = min(n, 200)
                    rescore_pos = np.argpartition(raw_n, -n_res)[-n_res:]
                z_leg = compute_asnorm_scores(
                    q,
                    sub_cvecs[rescore_pos],
                    c_mat_coh,
                    candidate_raw_sims=raw_c[rescore_pos],
                    candidate_ids=sub_pids[rescore_pos],
                    cohort_ids=c_pids_coh,
                    normalized_inputs=True,
                )
                cal_leg = np.clip(calibrate_asnorm_to_cosine(z_leg, raw_c[rescore_pos]), -1.0, 1.0)
                rescored_person = np.maximum(max_raw_f_per_pos[rescore_pos], cal_leg)
                best_n = float(np.max(rescored_person))
                if n > 300:
                    unrescored = np.ones(n, dtype=bool)
                    unrescored[rescore_pos] = False
                    if np.any(unrescored):
                        best_n = max(best_n, float(np.max(raw_n[unrescored])))
                leg_asn_by_n[n][p_i] = best_n

    base_arr = np.concatenate(base_samples) if base_samples else np.empty((0,), dtype=np.float32)
    return cal_llr_by_n, leg_raw_by_n, leg_asn_by_n, base_arr


# ---------------------------------------------------------------------------
# End-to-End Calibration & A/B Pipeline Per Engine
# ---------------------------------------------------------------------------


def calibrate_one_model(
    model_name: str,
    cohort_vecs: np.ndarray,
    cohort_source_tag: str,
    cohort_excluded_keys: Set[str],
    run_ab: bool = True,
    cohort_identities: Optional[Sequence[str]] = None,
    cohort_is_young: Optional[Sequence[bool]] = None,
    proj_k: int = 5,
    photo_hybrid_weight: float = 0.35,
) -> Dict[str, Any]:
    t0 = time.time()
    print(f"\n======================================================================")
    print(f"  Calibrating Engine: {model_name} (cohort={cohort_vecs.shape}, source={cohort_source_tag})")
    print(f"======================================================================")

    db_path = DEMO_DBS[model_name]
    gallery = load_ro_db_gallery(db_path, model_name)
    emb_map, qual_map = load_pair_Caches(model_name)

    proj_basis: Optional[np.ndarray] = None
    if cohort_identities is not None and cohort_is_young is not None:
        young_mask = np.asarray(cohort_is_young, dtype=bool)
        if proj_k > 0:
            proj_basis = compute_age_subspace_basis(
                cohort_vecs, cohort_identities, young_mask, n_components=proj_k
            )
        adult_mask = ~young_mask
        asnorm_cohort_raw = cohort_vecs[adult_mask] if np.any(adult_mask) else cohort_vecs
        asnorm_cohort = project_subspace(asnorm_cohort_raw, proj_basis)
        print(
            f">> Built v2 age-subspace proj_basis={None if proj_basis is None else proj_basis.shape}, "
            f"LAG_ADULT cohort={asnorm_cohort.shape}, photo_hybrid_weight={photo_hybrid_weight:.2f}"
        )
    else:
        asnorm_cohort = cohort_vecs

    asnorm = ExternalCohortASNorm(asnorm_cohort, top_k=200, std_floor=0.05)

    # 1. Assemble strictly cohort-disjoint 1:1 trials (AgeDB-30 + CALFW)
    agedb_trials, calfw_trials = assemble_1to1_trials(model_name, emb_map, qual_map, cohort_excluded_keys)
    all_trials = agedb_trials + calfw_trials
    # Strict identity-disjoint split: folds 0..4 for calibration fit, folds 5..9 for held-out test
    # Note: AgeDB-30 and CALFW fold schemes are per-dataset, so check no identity leaks between fit(0..4) & test(5..9)
    fit_trials = [t for t in all_trials if t["fold"] < 5]
    fit_keys = {t["id_a"] for t in fit_trials} | {t["id_b"] for t in fit_trials}
    test_trials_raw = [t for t in all_trials if t["fold"] >= 5]
    test_trials = [t for t in test_trials_raw if t["id_a"] not in fit_keys and t["id_b"] not in fit_keys]

    print(f">> 1:1 trials (disjoint from cohort): fit={len(fit_trials)}, held-out test={len(test_trials)} "
          f"(agedb={sum(t['dataset']=='agedb30' for t in test_trials)}, calfw={sum(t['dataset']=='calfw' for t in test_trials)})")

    # Compute external AS-Norm z-scores on (projected) fit trials and fit QMFCalibrator
    embs_a_fit_raw = np.stack([t["emb_a"] for t in fit_trials])
    embs_b_fit_raw = np.stack([t["emb_b"] for t in fit_trials])
    embs_a_fit = project_subspace(embs_a_fit_raw, proj_basis) if proj_basis is not None else embs_a_fit_raw
    embs_b_fit = project_subspace(embs_b_fit_raw, proj_basis) if proj_basis is not None else embs_b_fit_raw
    mu_a_f, sig_a_f = asnorm.enroll_stats(embs_a_fit)
    mu_b_f, sig_b_f = asnorm.enroll_stats(embs_b_fit)
    raw_fit = (
        np.clip(np.sum(embs_a_fit * embs_b_fit, axis=1), -1.0, 1.0).astype(np.float64)
        if proj_basis is not None
        else np.array([t["raw_sim"] for t in fit_trials], dtype=np.float64)
    )
    z_fit = 0.5 * ((raw_fit - mu_a_f) / sig_a_f + (raw_fit - mu_b_f) / sig_b_f)
    q_fit = np.array([t["q_min"] for t in fit_trials], dtype=np.float64)
    age_fit = np.array([t["age_gap"] for t in fit_trials], dtype=np.float64)
    y_fit = np.array([t["is_same"] for t in fit_trials], dtype=np.int32)

    qmf = QMFCalibrator.fit(z_fit, raw_fit, y_fit, q_min=q_fit, age_gap=age_fit, l2_reg=5e-2)
    print(">> Fitted QMF weights:", {k: round(float(v), 4) for k, v in zip(QMFCalibrator.FEATURE_NAMES, qmf.weights)},
          f"bias={qmf.bias:.4f}")

    # 2. Fit CoveCalibrator on RAW disjoint calibration identities (for 1:1 /api/verify-id)
    cove_embs = []
    cove_ids = []
    seen_cove = set()
    for t in fit_trials:
        if t["id_a"] not in seen_cove:
            seen_cove.add(t["id_a"])
            cove_embs.append(t["emb_a"])
            cove_ids.append(t["id_a"])
        if t["id_b"] not in seen_cove:
            seen_cove.add(t["id_b"])
            cove_embs.append(t["emb_b"])
            cove_ids.append(t["id_b"])
    cove = CoveCalibrator.fit(np.stack(cove_embs), cove_ids)
    print(f">> Fitted CoveCalibrator: {cove.n_identities} disjoint identities -> {cove.n_pairs:,} pairs")

    # 3. Build disjoint non-mated probe pool (600 unique non-overlapping DB identities) for 1:N Open-Set N-scaling
    rng = np.random.default_rng(77)
    unique_by_key: Dict[str, int] = {}
    for idx, nkey in enumerate(gallery["norm_keys"]):
        if nkey not in cohort_excluded_keys and nkey not in unique_by_key:
            unique_by_key[nkey] = idx

    # Prefer selecting the 600 non-mated probes from clean CALFW-only identities
    # (excluding AgeDB/FGNET and known LFW wrong-crop multi-person captions)
    calfw_only_keys = [
        nk for nk, idx in unique_by_key.items()
        if gallery["names"][idx].startswith("CALFW ") and nk not in LFW_CONTAMINATED_NONMATES
    ]
    rng.shuffle(calfw_only_keys)
    nonmate_keys = calfw_only_keys[:600]
    nonmate_key_set = set(nonmate_keys)

    probe_vecs_nm = []
    probe_quals_nm = []
    for nk in nonmate_keys:
        pid = int(gallery["pids"][unique_by_key[nk]])
        # Pick the newest/last face of this person as probe
        _fid, _age, pvec, pqual = gallery["faces_by_pid"][pid][-1]
        probe_vecs_nm.append(pvec)
        probe_quals_nm.append(pqual)
    probe_vecs_nm_raw = np.stack(probe_vecs_nm).astype(np.float32)
    probe_vecs_nm_arr = (
        project_subspace(probe_vecs_nm_raw, proj_basis) if proj_basis is not None else probe_vecs_nm_raw
    )
    probe_quals_nm_arr = np.array(probe_quals_nm, dtype=np.float32)

    # Deduplicated disjoint gallery (excluding cohort_excluded_keys AND nonmate_keys, 1 person per norm_key)
    dedup_gal_indices = [
        idx for nk, idx in unique_by_key.items()
        if nk not in nonmate_key_set
    ]
    max_avail_n = len(dedup_gal_indices)
    candidate_ns = [n for n in [1, 50, 80, 150, 300, 600, 1000, 2000, 3000, 3935] if n <= max_avail_n]
    if max_avail_n not in candidate_ns:
        candidate_ns.append(max_avail_n)
    candidate_ns = sorted(set(candidate_ns))

    # Build filtered gallery arrays containing 1 person per unique dedup_gal_indices (projected if proj_basis is set)
    dedup_mask_pids = gallery["pids"][dedup_gal_indices]
    dedup_cquals = gallery["cent_quals"][dedup_gal_indices]
    dedup_keys = [gallery["norm_keys"][i] for i in dedup_gal_indices]
    if proj_basis is not None:
        proj_faces_by_pid: Dict[int, List[Tuple[int, Optional[int], np.ndarray, float]]] = {}
        dedup_cvecs_list: List[np.ndarray] = []
        for pid_int in dedup_mask_pids.tolist():
            items = gallery["faces_by_pid"][int(pid_int)]
            raw_f = np.stack([x[2] for x in items]).astype(np.float32)
            proj_f = project_subspace(raw_f, proj_basis)
            ages = [x[1] for x in items]
            quals = [x[3] for x in items]
            weights = np.empty(len(items), dtype=np.float32)
            for i, (age_i, qual_i) in enumerate(zip(ages, quals)):
                if age_i is None:
                    w_age = 1.0
                else:
                    age_f = float(age_i)
                    nearby = sum(
                        1 for j, a_j in enumerate(ages)
                        if j != i and a_j is not None and abs(float(a_j) - age_f) <= 2.0
                    )
                    w_age = 1.0 / (1.0 + nearby)
                weights[i] = qual_i * w_age
            weighted = np.sum(proj_f * weights[:, None], axis=0)
            nrm = float(np.linalg.norm(weighted))
            dedup_cvecs_list.append((weighted / nrm if nrm > 0 else weighted).astype(np.float32))
            proj_faces_by_pid[int(pid_int)] = [
                (items[i][0], items[i][1], proj_f[i], items[i][3]) for i in range(len(items))
            ]
        dedup_cvecs = np.stack(dedup_cvecs_list).astype(np.float32)
        sim_faces_by_pid = proj_faces_by_pid
    else:
        dedup_cvecs = gallery["cent_vecs"][dedup_gal_indices]
        sim_faces_by_pid = gallery["faces_by_pid"]

    cal_llr_nm, leg_raw_nm, leg_asn_nm, base_nm = _simulate_nonmate_max_scores(
        probe_vecs_nm_arr,
        probe_quals_nm_arr,
        nonmate_keys,
        dedup_mask_pids,
        dedup_cvecs,
        dedup_cquals,
        dedup_keys,
        sim_faces_by_pid,
        asnorm,
        qmf,
        candidate_ns,
        seed=123,
        photo_hybrid_weight=photo_hybrid_weight,
    )

    openset_full = OpenSetCalibrator.fit(
        candidate_ns,
        [cal_llr_nm[n] for n in candidate_ns],
        base_nm,
        tail_quantile=0.95,
    )
    print(f">> Fitted OpenSetCalibrator: n_grid={candidate_ns}, GPD(xi={openset_full.xi:.4f}, "
          f"sigma_u={openset_full.sigma_u:.4f}, u={openset_full.u:.4f}, p_u={openset_full.p_u:.4f}), "
          f"N_eff(N)={openset_full.n_eff_c:.3f}*N^{openset_full.n_eff_gamma:.3f}")

    # Save CalibrationBundle (.npz + .json)
    bundle_path = resolve_bundle_path(model_name)
    fit_eval = qmf.evaluate(z_fit, raw_fit, y_fit, q_min=q_fit, age_gap=age_fit)
    bundle = CalibrationBundle(
        model_name=model_name,
        dim=asnorm.dim,
        asnorm=asnorm,
        qmf=qmf,
        openset=openset_full,
        cove=cove,
        proj_basis=proj_basis,
        photo_hybrid_weight=photo_hybrid_weight,
        metadata={
            "cohort_source": cohort_source_tag,
            "cohort_filter": "lag_adult_only" if cohort_is_young is not None else "all",
            "fit_trials": len(fit_trials),
            "fit_diagnostics": fit_eval,
        },
    )
    saved_npz, saved_json = bundle.save(bundle_path)
    print(f">> Saved CalibrationBundle -> {saved_npz} & {saved_json}")

    if not run_ab:
        return {"bundle_path": saved_npz, "meta_path": saved_json}

    # =========================================================================
    # Exhaustive Out-of-Sample A/B Benchmark
    # =========================================================================
    ab: Dict[str, Any] = {
        "model_name": model_name,
        "dim": asnorm.dim,
        "cohort_size": asnorm.m,
        "cohort_source": cohort_source_tag,
        "max_disjoint_gallery_n": max_avail_n,
    }

    # A) 1:1 Held-Out Verification & Calibration Quality
    test_agedb = [t for t in test_trials if t["dataset"] == "agedb30"]
    test_calfw = [t for t in test_trials if t["dataset"] == "calfw"]
    ab["1to1_heldout"] = {
        "agedb30": eval_1to1_subset(test_agedb, asnorm, qmf, cove=cove, proj_basis=proj_basis),
        "calfw": eval_1to1_subset(test_calfw, asnorm, qmf, cove=cove, proj_basis=proj_basis),
        "combined": eval_1to1_subset(test_trials, asnorm, qmf, cove=cove, proj_basis=proj_basis),
    }

    # B) 1:N Open-Set N-Scaling (5-fold strictly out-of-sample CV across the 600 non-mated probes)
    n_report_targets = [n for n in [80, 300, 1000, max_avail_n] if n in cal_llr_nm]
    n_scaling_rows = []
    n_folds = 5
    m_nm = len(nonmate_keys)
    fold_ids = np.arange(m_nm) % n_folds

    for n_val in n_report_targets:
        cal_strong_hits = 0
        cal_poss_hits = 0
        # 5-fold out-of-sample OpenSetCalibrator evaluation
        for f_idx in range(n_folds):
            tr_m = fold_ids != f_idx
            te_m = fold_ids == f_idx
            os_cv = OpenSetCalibrator.fit(
                candidate_ns,
                [cal_llr_nm[nn][tr_m] for nn in candidate_ns],
                base_nm,
                tail_quantile=0.95,
            )
            tau_s, tau_p = os_cv.tier_thresholds(n_val, alpha_strong=0.01, alpha_possible=0.08)
            te_scores = cal_llr_nm[n_val][te_m]
            cal_strong_hits += int(np.sum(te_scores >= tau_s))
            cal_poss_hits += int(np.sum(te_scores >= tau_p))

        raw_arr = leg_raw_nm[n_val]
        asn_arr = leg_asn_nm[n_val]
        tau_s_full, tau_p_full = openset_full.tier_thresholds(n_val, alpha_strong=0.01, alpha_possible=0.08)
        n_scaling_rows.append({
            "N": int(n_val),
            "legacy_raw_fpir_026_pct": round(float(np.mean(raw_arr >= POSSIBLE)) * 100.0, 2),
            "legacy_raw_fpir_036_pct": round(float(np.mean(raw_arr >= STRONG)) * 100.0, 2),
            "legacy_asnorm_fpir_026_pct": round(float(np.mean(asn_arr >= POSSIBLE)) * 100.0, 2),
            "legacy_asnorm_fpir_036_pct": round(float(np.mean(asn_arr >= STRONG)) * 100.0, 2),
            "calibrated_fpir_strong_pct": round(cal_strong_hits / float(m_nm) * 100.0, 2),
            "calibrated_fpir_possible_pct": round(cal_poss_hits / float(m_nm) * 100.0, 2),
            "tau_strong_llr": round(tau_s_full, 4),
            "tau_possible_llr": round(tau_p_full, 4),
        })
    ab["1toN_open_set_scaling"] = n_scaling_rows

    # C) 1:N Severe Age-Gap (>= 30y) Mated Retrieval against large distractor gallery
    # Collect genuine >=30y AgeDB-30 held-out probes (from test_agedb where identity is in dedup_gal_indices)
    dedup_pid_by_key = {dedup_keys[i]: int(dedup_mask_pids[i]) for i in range(len(dedup_keys))}
    gal_pos_by_pid = {int(dedup_mask_pids[i]): i for i in range(len(dedup_keys))}

    # Build temporary in-memory FaceStore over dedup_gal_indices via bulk SQLite insert
    mem_store = FaceStore(":memory:")
    person_rows = []
    face_rows = []
    next_pid = 1
    for gal_idx in dedup_gal_indices:
        orig_pid = int(gallery["pids"][gal_idx])
        pname = gallery["names"][gal_idx]
        gal_pos_by_pid[orig_pid] = next_pid
        person_rows.append((next_pid, pname, "", 0.0))
        for _fid, age_v, fvec, fqual in gallery["faces_by_pid"][orig_pid]:
            face_rows.append((next_pid, None, age_v, fvec.tobytes(), model_name, float(fqual)))
        next_pid += 1
    mem_store.db.executemany("INSERT INTO persons(id, name, notes, created) VALUES (?,?,?,?)", person_rows)
    mem_store.db.executemany(
        "INSERT INTO faces(person_id, photo_path, age, embedding, model, quality) VALUES (?,?,?,?,?,?)",
        face_rows,
    )
    mem_store.db.commit()
    mem_store._rebuild_all()
    mem_store.set_calibration_bundle(model_name, bundle)

    # Warm up calibrated cohort stats cache once
    _ = mem_store.search(dedup_cvecs[0], k=10, model=model_name)

    def _search_store(q_vec, k, min_sim, calibrated_flag, q_qual=1.0):
        if calibrated_flag:
            return mem_store.search(q_vec, k=k, min_sim=min_sim, model=model_name, probe_quality=q_qual)
        with mem_store.lock:
            q_norm = mem_store._norm(q_vec, model=model_name)
            idx = mem_store.indexes[model_name]
            return mem_store._search_cosine_fallback(q_norm, idx, k=k, min_sim=min_sim, model=model_name)

    # Severe age-gap (>=30y) mated queries: use held-out AgeDB probes whose person is in mem_store
    # Build probe list from agedb_trials (genuine pairs with gap >= 30y whose id_a is in dedup_pid_by_key)
    seen_probe_sigs = set()
    severe_queries = []
    for t in test_agedb:
        if t["is_same"] != 1 or not np.isfinite(t["age_gap"]) or t["age_gap"] < 30.0:
            continue
        nk = t["id_a"]
        if nk not in dedup_pid_by_key:
            continue
        orig_pid = dedup_pid_by_key[nk]
        store_pid = gal_pos_by_pid[orig_pid]
        # Avoid trivial exact-embedding duplicate: require probe != every stored gallery face of store_pid
        gal_fvecs = np.stack([x[2] for x in gallery["faces_by_pid"][orig_pid]])
        for cand_emb, cand_q in ((t["emb_a"], t["q_min"]), (t["emb_b"], t["q_min"])):
            max_self = float(np.max(gal_fvecs @ cand_emb))
            if max_self < 0.995:
                sig = (store_pid, round(float(cand_emb[0]), 5), round(float(cand_emb[1]), 5))
                if sig not in seen_probe_sigs:
                    seen_probe_sigs.add(sig)
                    severe_queries.append((store_pid, cand_emb, cand_q, float(t["age_gap"])))
        if len(severe_queries) >= 250:
            break

    # If held-out folds had fewer severe queries (because demo_setup only put older photos in demo_probes),
    # gather held-out manifest probes from bench_cache matching AgeDB/FGNET in mem_store!
    if len(severe_queries) < 100:
        manifest_p = os.path.join(MAIN_DIR, "demo_probes", "manifest.json")
        if os.path.isfile(manifest_p):
            with open(manifest_p, encoding="utf-8") as mf:
                mprobes = json.load(mf)["probes"]
            for mp in mprobes:
                if mp["dataset"] != "agedb":
                    continue
                nk = normalize_identity(mp["expected_name"])
                if nk not in dedup_pid_by_key:
                    continue
                bn = os.path.basename(mp["file"]).lower()
                if bn not in emb_map:
                    continue
                orig_pid = dedup_pid_by_key[nk]
                store_pid = gal_pos_by_pid[orig_pid]
                max_g_age = gallery["max_ages"][unique_by_key[nk]]
                p_age = mp.get("probe_age")
                gap = float(abs(p_age - max_g_age)) if (p_age is not None and max_g_age is not None) else 30.0
                if gap >= 20.0:
                    severe_queries.append((store_pid, emb_map[bn], qual_map.get(bn, 1.0), gap))
                if len(severe_queries) >= 250:
                    break

    def _eval_store_queries(query_items, calibrated_flag):
        t_q0 = time.time()
        r1 = r10 = strong_r1 = poss_r1 = strong_t10 = poss_t10 = 0
        top1_scores = []
        for target_pid, q_emb, q_qual, _gap in query_items:
            hits = _search_store(
                q_emb, k=10, min_sim=-1.0, calibrated_flag=calibrated_flag, q_qual=q_qual,
            )
            if hits:
                top1_scores.append(hits[0]["calibrated_llr"] if calibrated_flag else hits[0]["similarity"])
            found_rank = None
            found_tier = None
            for r_idx, h in enumerate(hits, 1):
                if h["person_id"] == target_pid:
                    found_rank = r_idx
                    found_tier = h["tier"]
                    break
            if found_rank == 1:
                r1 += 1
                if found_tier == "STRONG":
                    strong_r1 += 1
                if found_tier in ("STRONG", "possible"):
                    poss_r1 += 1
            if found_rank is not None:
                r10 += 1
                if found_tier == "STRONG":
                    strong_t10 += 1
                if found_tier in ("STRONG", "possible"):
                    poss_t10 += 1
        dt_ms = (time.time() - t_q0) * 1000.0 / max(1, len(query_items))
        n_q = max(1, len(query_items))
        return {
            "n_queries": len(query_items),
            "rank1_pct": round(r1 / n_q * 100.0, 2),
            "top10_pct": round(r10 / n_q * 100.0, 2),
            "strong_rank1_pct": round(strong_r1 / n_q * 100.0, 2),
            "possible_plus_rank1_pct": round(poss_r1 / n_q * 100.0, 2),
            "strong_top10_pct": round(strong_t10 / n_q * 100.0, 2),
            "possible_plus_top10_pct": round(poss_t10 / n_q * 100.0, 2),
            "latency_ms_per_query": round(dt_ms, 2),
        }

    if severe_queries:
        sev_leg = _eval_store_queries(severe_queries, calibrated_flag=False)
        sev_cal = _eval_store_queries(severe_queries, calibrated_flag=True)
        # Also measure TPIR@FPIR=1% using the exact N=max_avail_n empirical 1% non-mated cutoff!
        tau_emp_1_leg = float(np.quantile(leg_asn_nm[max_avail_n], 0.99))
        tau_emp_1_cal = float(np.quantile(cal_llr_nm[max_avail_n], 0.99))

        def _tpir_at_emp1(calibrated_flag, cutoff):
            hit = 0
            for target_pid, q_emb, q_qual, _gap in severe_queries:
                hits = _search_store(
                    q_emb, k=1, min_sim=-1.0, calibrated_flag=calibrated_flag, q_qual=q_qual,
                )
                if hits and hits[0]["person_id"] == target_pid:
                    sc = hits[0]["calibrated_llr"] if calibrated_flag else hits[0]["similarity"]
                    if sc >= cutoff:
                        hit += 1
            return round(hit / float(len(severe_queries)) * 100.0, 2)

        sev_leg["tpir_at_emp_fpir1_pct"] = _tpir_at_emp1(False, tau_emp_1_leg)
        sev_cal["tpir_at_emp_fpir1_pct"] = _tpir_at_emp1(True, tau_emp_1_cal)
        ab["1toN_severe_age_gap"] = {"legacy_asnorm": sev_leg, "calibrated_4layer": sev_cal}

    # D) End-to-End Demo Gallery Closed-Set Regression (on matched held-back demo_probes against mem_store)
    manifest_p = os.path.join(MAIN_DIR, "demo_probes", "manifest.json")
    demo_queries = []
    if os.path.isfile(manifest_p):
        with open(manifest_p, encoding="utf-8") as mf:
            for mp in json.load(mf)["probes"]:
                nk = normalize_identity(mp["expected_name"])
                if nk not in dedup_pid_by_key:
                    continue
                bn = os.path.basename(mp["file"]).lower()
                if bn not in emb_map:
                    continue
                orig_pid = dedup_pid_by_key[nk]
                demo_queries.append((gal_pos_by_pid[orig_pid], emb_map[bn], qual_map.get(bn, 1.0), 0.0))

    if demo_queries:
        ab["demo_gallery_regression"] = {
            "gallery_persons": len(dedup_gal_indices),
            "legacy_asnorm": _eval_store_queries(demo_queries, calibrated_flag=False),
            "calibrated_4layer": _eval_store_queries(demo_queries, calibrated_flag=True),
        }

    # Also record end-to-end store latency on 100 non-mated probes at N = max_avail_n
    t_lat0 = time.time()
    for p_i in range(min(100, len(probe_vecs_nm_raw))):
        _ = _search_store(probe_vecs_nm_raw[p_i], k=10, min_sim=-1.0, calibrated_flag=False)
    leg_ms = (time.time() - t_lat0) * 1000.0 / min(100, len(probe_vecs_nm_raw))

    t_lat1 = time.time()
    for p_i in range(min(100, len(probe_vecs_nm_raw))):
        _ = mem_store.search(probe_vecs_nm_raw[p_i], k=10, model=model_name, probe_quality=float(probe_quals_nm_arr[p_i]))
    cal_ms = (time.time() - t_lat1) * 1000.0 / min(100, len(probe_vecs_nm_raw))
    ab["store_search_latency_ms"] = {
        "N": len(dedup_gal_indices),
        "legacy_asnorm_ms": round(leg_ms, 2),
        "calibrated_4layer_ms": round(cal_ms, 2),
    }

    ab["runtime_sec"] = round(time.time() - t0, 2)
    mem_store.db.close()
    return ab


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-MIDS 4-Layer Calibration Fitter & A/B Benchmark")
    parser.add_argument("--models", type=str, default="antelopev2,aimids_crossage")
    parser.add_argument("--cohort-source", type=str, default="auto", choices=["auto", "lag", "carved"])
    parser.add_argument("--cohort-size", type=int, default=1200)
    parser.add_argument("--skip-ab", action="store_true")
    parser.add_argument(
        "--output-summary",
        type=str,
        default=os.path.join(ARC_DIR, "bench_results", "calibration_ab_summary.json"),
    )
    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    lag_data = None
    cohort_source_used = args.cohort_source

    if args.cohort_source in ("auto", "lag"):
        try:
            lag_data = _extract_lag_embeddings_onnx(max_faces=args.cohort_size)
            cohort_source_used = "lag_external_tar"
        except Exception as e:
            if args.cohort_source == "lag":
                raise
            print(f">> Notice: LAG ONNX extraction unavailable ({e}); falling back to strictly identity-disjoint carved cohort.")
            cohort_source_used = "carved_disjoint"

    summary: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cohort_source": cohort_source_used,
        "models": {},
    }

    # If using carved fallback, determine a shared cohort identity set so both models stay 100% consistent
    carved_keys: Optional[Set[str]] = None
    if cohort_source_used == "carved_disjoint":
        # Protect all held-out 1:1 test identities (folds >= 2), all CALFW pair identities,
        # and demo manifest AgeDB/FG-NET identities so benchmark coverage stays intact
        protected: Set[str] = set()
        for m in models:
            em, qm = load_pair_Caches(m)
            ag, cf = assemble_1to1_trials(m, em, qm, set())
            for t in ag:
                if t["fold"] >= 2:
                    protected.add(t["id_a"])
                    protected.add(t["id_b"])
            for t in cf:
                protected.add(t["id_a"])
                protected.add(t["id_b"])
        manifest_p = os.path.join(MAIN_DIR, "demo_probes", "manifest.json")
        if os.path.isfile(manifest_p):
            with open(manifest_p, encoding="utf-8") as mf:
                for mp in json.load(mf)["probes"]:
                    if mp["dataset"] in ("agedb", "fgnet"):
                        protected.add(normalize_identity(mp["expected_name"]))
        g0 = load_ro_db_gallery(DEMO_DBS[models[0]], models[0])
        _, carved_keys, _ = build_carved_balanced_cohort(g0, protected, target_size=args.cohort_size)

    for model_name in models:
        if cohort_source_used == "lag_external_tar":
            cohort_vecs = lag_data[model_name]
            cohort_idents = lag_data.get("identities")
            cohort_young = lag_data.get("is_young")
            excluded_keys: Set[str] = set()  # LAG already excluded all known AgeDB/CALFW/FG-NET/DB names
        else:
            g = load_ro_db_gallery(DEMO_DBS[model_name], model_name)
            # Extract all faces belonging to carved_keys (preserving exact balanced count)
            vecs = []
            for pid, nk in zip(g["pids"].tolist(), g["norm_keys"]):
                if nk in carved_keys:
                    for _fid, _age, v, _q in g["faces_by_pid"][pid][:2]:
                        vecs.append(v)
            cohort_vecs = np.stack(vecs).astype(np.float32)
            cohort_idents = None
            cohort_young = None
            excluded_keys = set(carved_keys)

        res = calibrate_one_model(
            model_name=model_name,
            cohort_vecs=cohort_vecs,
            cohort_source_tag=cohort_source_used,
            cohort_excluded_keys=excluded_keys,
            run_ab=not args.skip_ab,
            cohort_identities=cohort_idents,
            cohort_is_young=cohort_young,
        )
        summary["models"][model_name] = res

    if not args.skip_ab:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_summary)), exist_ok=True)
        with open(args.output_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n>> Wrote A/B benchmark summary -> {args.output_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
