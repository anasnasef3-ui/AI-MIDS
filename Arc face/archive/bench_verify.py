"""
bench_verify.py - Reusable Verification Benchmark Tool for AI-MIDS.

What this module does:
1. Takes verification pairs (img_a, img_b, is_same, fold_idx) from standard benchmark datasets.
2. Extracts face embeddings using the project's production engines (antelopev2 or buffalo_l,
   with optional Flip Test-Time Augmentation).
3. Embeds each UNIQUE image only once and caches embeddings to an .npz file so subsequent
   evaluations or threshold tests take seconds rather than hours.
4. Evaluates verification performance using standard 10-fold cross-validation:
   - 10-fold accuracy (best threshold tuned on 9 folds, evaluated on 1 fold).
   - Mean genuine vs. impostor cosine similarities.
   - TAR (True Accept Rate) at strict security thresholds: FAR = 1% and FAR = 0.1%.
   - Project confidence tiers distribution (STRONG >= 0.36, POSSIBLE >= 0.26, WEAK < 0.26).
5. Exports every pair's cosine similarity to a CSV file in 'bench_results/' for threshold calibration.

Usage Examples:
    # 1. Benchmark antelopev2 with Flip-TTA on AgeDB-30 (fast run with 600 pairs):
    python bench_verify.py --dataset agedb30 --engine antelopev2 --tta --limit 600

    # 2. Benchmark buffalo_l for comparison:
    python bench_verify.py --dataset agedb30 --engine buffalo_l --limit 600

    # 3. Full benchmark (6,000 pairs, uses cached embeddings if already embedded):
    python bench_verify.py --dataset agedb30 --engine antelopev2 --tta

Author: AI-MIDS Team / Pair Programming Assistant
"""

import os
import re
import sys
import time
import argparse
import csv
from collections import defaultdict
import numpy as np

# Set environment and imports relative to Arc face directory
ARC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARC_DIR)
os.chdir(ARC_DIR)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from face_engine import FaceEngine
from face_engine_antelopev2 import FaceEngineAntelopeV2
from face_engine_adaface_ft import FaceEngineAdaFaceFT
from face_engine_custom import (
    FaceEngineCrossAge,
    compute_asnorm_scores,
    calibrate_asnorm_to_cosine,
)
from bench_datasets import load_benchmark_dataset


# ==============================================================================
# 1. Embedding Engine & Disk Cache
# ==============================================================================

def load_engine(engine_name, use_tta=False):
    """
    Loads buffalo_l (ResNet-50), antelopev2 (ResNet-100), aimids_ft (fine-tuned AdaFace IR-101),
    or aimids_crossage (1024-d dual-branch antelopev2 + aimids_ft) engine on CPU.
    """
    engine_name = engine_name.lower()
    if engine_name == "antelopev2":
        print(f">> Loading AntelopeV2 engine (ResNet-100, TTA={use_tta})...")
        return FaceEngineAntelopeV2(use_gpu=False, use_tta=use_tta)
    elif engine_name == "buffalo_l":
        print(">> Loading Buffalo_L engine (ResNet-50)...")
        return FaceEngine(use_gpu=False)
    elif engine_name in ("aimids_ft", "adaface_ft"):
        print(f">> Loading Fine-Tuned AdaFace engine (ResNet-101, TTA={use_tta})...")
        return FaceEngineAdaFaceFT(use_gpu=False, use_tta=use_tta)
    elif engine_name in ("aimids_crossage", "crossage"):
        print(f">> Loading Cross-Age Dual-Branch engine (1024-d antelopev2+aimids_ft, TTA={use_tta})...")
        return FaceEngineCrossAge(use_gpu=False, use_tta=use_tta)
    else:
        raise ValueError(
            f"Unknown engine '{engine_name}'. Supported: antelopev2, buffalo_l, aimids_ft, aimids_crossage"
        )


def get_cache_path(cache_dir, dataset_name, engine_name, use_tta):
    """Generates the filename for the embedding cache .npz."""
    tta_suffix = "_tta" if use_tta else ""
    fname = f"bench_cache_{dataset_name.lower()}_{engine_name.lower()}{tta_suffix}.npz"
    return os.path.join(cache_dir, fname)


def load_cache(cache_path):
    """
    Loads cached embeddings from disk.
    Returns: dict mapping file_path -> (embedding_vector or None if no face)
    """
    if not os.path.isfile(cache_path):
        return {}

    try:
        data = np.load(cache_path, allow_pickle=True)
        paths = list(data["paths"])
        embeddings = data["embeddings"]
        has_face = data["has_face"]
        cache = {}
        for p, emb, hf in zip(paths, embeddings, has_face):
            cache[p] = emb if hf else None
        print(f">> Loaded {len(cache)} cached embeddings from: {os.path.basename(cache_path)}")
        return cache
    except Exception as e:
        print(f"Warning: Failed to load cache from {cache_path} ({e}). Starting fresh.")
        return {}


def load_quality_cache(cache_path):
    """Loads optional per-image quality scalars saved alongside embeddings in .npz."""
    if not os.path.isfile(cache_path):
        return {}
    try:
        data = np.load(cache_path, allow_pickle=True)
        if "qualities" not in data:
            return {}
        paths = list(data["paths"])
        qualities = data["qualities"]
        return {p: float(q) for p, q in zip(paths, qualities) if np.isfinite(q)}
    except Exception:
        return {}


def _infer_embedding_dim(cache_dict, default_dim=512):
    """Infers embedding dimension dynamically from the first valid vector in cache_dict."""
    for vec in cache_dict.values():
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return int(arr.size)
    return int(default_dim)


def save_cache(cache_path, cache_dict, quality_dict=None):
    """
    Saves the embedding cache dictionary to a compressed .npz file.
    Infers embedding dimension dynamically (e.g., 512 or 1024) from loaded vectors.
    """
    paths = list(cache_dict.keys())
    dim = _infer_embedding_dim(cache_dict, default_dim=512)
    embeddings = []
    has_face = []
    qualities = []
    zero_vec = np.zeros(dim, dtype=np.float32)

    for p in paths:
        vec = cache_dict[p]
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size != dim:
                raise ValueError(f"Mixed embedding dimensions in cache: expected {dim}, got {arr.size} for {p}")
            embeddings.append(arr)
            has_face.append(True)
        else:
            embeddings.append(zero_vec)
            has_face.append(False)
        if quality_dict is not None:
            q_val = quality_dict.get(p, 1.0)
            qualities.append(float(q_val) if q_val is not None and np.isfinite(q_val) else 1.0)

    emb_arr = np.array(embeddings, dtype=np.float32) if embeddings else np.empty((0, dim), dtype=np.float32)
    payload = {
        "paths": np.array(paths),
        "embeddings": emb_arr,
        "has_face": np.array(has_face, dtype=bool),
    }
    if quality_dict is not None:
        payload["qualities"] = np.array(qualities, dtype=np.float32)

    np.savez_compressed(cache_path, **payload)
    print(f">> Saved {len(paths)} embeddings (dim={dim}) to cache: {os.path.basename(cache_path)}")


def embed_unique_images(image_paths, engine, cache_dict, cache_path, quality_dict=None):
    """
    Embeds all unique images in the dataset once, skipping any already in cache_dict.
    Uses engine.embed() which picks the largest face in each image.
    Updates cache on disk after completion.
    """
    unique_paths = sorted(list(set(image_paths)))
    missing = [p for p in unique_paths if p not in cache_dict]

    if not missing:
        print(f">> All {len(unique_paths)} unique images already present in cache!")
        return cache_dict

    print(f">> Need to embed {len(missing)} / {len(unique_paths)} unique images on CPU...")
    t0 = time.time()
    detected_count = 0
    no_face_count = 0

    try:
        for idx, p in enumerate(missing, 1):
            try:
                emb_res = engine.embed(p)
                if emb_res is not None and "embedding" in emb_res:
                    cache_dict[p] = np.asarray(emb_res["embedding"], dtype=np.float32).reshape(-1)
                    if quality_dict is not None and "quality" in emb_res:
                        quality_dict[p] = float(emb_res["quality"])
                    detected_count += 1
                else:
                    cache_dict[p] = None
                    no_face_count += 1
            except Exception:
                # If image file cannot be read or is corrupted
                cache_dict[p] = None
                no_face_count += 1

            # Progress reporting and checkpoint save every 50 images or on last image
            if idx % 50 == 0 or idx == len(missing):
                elapsed = time.time() - t0
                speed = elapsed / idx
                eta_sec = (len(missing) - idx) * speed
                print(f"  [{idx}/{len(missing)}] ({idx/len(missing)*100:5.1f}%) | "
                      f"Speed: {speed:.3f}s/img | Elapsed: {elapsed/60:4.1f}m | ETA: {eta_sec/60:4.1f}m",
                      flush=True)
                # Periodic checkpoint save to survive power cuts or interruptions
                save_cache(cache_path, cache_dict, quality_dict=quality_dict)
    finally:
        # Guarantee cache is saved to disk even if interrupted
        save_cache(cache_path, cache_dict, quality_dict=quality_dict)
    return cache_dict


# ==============================================================================
# 2. Verification Evaluation & 10-Fold Cross-Validation
# ==============================================================================

_AGEDB_NAME_RE = re.compile(r"^(\d+)_(.+?)_(\d+)_([mf])$", re.IGNORECASE)
_CALFW_NAME_RE = re.compile(r"^(.+?)_(\d+)$")


def extract_identity_from_path(path):
    """Extracts normalized identity string from an AgeDB or CALFW/LFW image filename."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    m_age = _AGEDB_NAME_RE.match(stem)
    if m_age:
        return m_age.group(2).strip().lower()
    m_calfw = _CALFW_NAME_RE.match(stem)
    if m_calfw:
        return m_calfw.group(1).strip().lower()
    return stem.strip().lower()


def load_calfw_local(root=None, limit=None, seed=0):
    """
    Loads local CALFW 10-fold protocol from D:\\GRAD\\Assets\\calfw (or custom root).
    In pairs_CALFW.txt, pairs 0..2999 are genuine (folds 1..10, 300/fold) and
    pairs 3000..5999 are impostor (300/fold), so fold = (idx % 3000) // 300 yields
    300 genuine + 300 impostor per fold with zero cross-fold identity overlap.
    """
    if root is None:
        root = r"D:\GRAD\Assets\calfw"
    pairs_file = os.path.join(root, "pairs_CALFW.txt")
    if not os.path.isfile(pairs_file):
        raise FileNotFoundError(
            f"CALFW pairs file not found at '{pairs_file}'."
        )

    img_lookup = {}
    aligned_dir = os.path.join(root, "aligned images")
    search_root = aligned_dir if os.path.isdir(aligned_dir) else root
    for dirpath, _dirnames, filenames in os.walk(search_root):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_lookup.setdefault(fname, os.path.join(dirpath, fname))

    with open(pairs_file, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    pairs = []
    n_pairs = len(raw_lines) // 2
    for idx in range(n_pairs):
        p1_parts = raw_lines[2 * idx].split()
        p2_parts = raw_lines[2 * idx + 1].split()
        if len(p1_parts) < 2 or len(p2_parts) < 2:
            continue
        p1 = img_lookup.get(os.path.basename(p1_parts[0]))
        p2 = img_lookup.get(os.path.basename(p2_parts[0]))
        if not p1 or not p2:
            continue
        is_same = 1 if int(p1_parts[1]) != 0 else 0
        if n_pairs == 6000:
            fold_idx = (idx % 3000) // 300
        else:
            fold_idx = min(9, (idx * 10) // max(1, n_pairs))
        pairs.append((p1, p2, is_same, fold_idx))

    if limit is not None and limit < len(pairs):
        per_fold = max(2, limit // 10)
        gen_per_fold = per_fold // 2
        imp_per_fold = per_fold - gen_per_fold
        sampled = []
        for f_idx in range(10):
            f_gen = [p for p in pairs if p[3] == f_idx and p[2] == 1]
            f_imp = [p for p in pairs if p[3] == f_idx and p[2] == 0]
            sampled.extend(f_gen[:gen_per_fold])
            sampled.extend(f_imp[:imp_per_fold])
        pairs = sampled

    gen_c = sum(1 for p in pairs if p[2] == 1)
    imp_c = sum(1 for p in pairs if p[2] == 0)
    print(f"[CALFW] Loaded {len(pairs)} pairs ({gen_c} genuine + {imp_c} impostor) across 10 disjoint folds from {pairs_file}")
    return pairs, "Official CALFW 10-Fold Protocol (pairs_CALFW.txt)"


def load_dataset_pairs(dataset_name, root=None, limit=None, seed=0):
    """Delegates to bench_datasets.load_benchmark_dataset with local CALFW fallback."""
    try:
        return load_benchmark_dataset(dataset_name, root=root, limit=limit, seed=seed)
    except FileNotFoundError:
        norm_name = dataset_name.lower().replace("-", "").replace("_", "")
        if norm_name == "calfw":
            return load_calfw_local(root=root, limit=limit, seed=seed)
        raise


def apply_fold_disjoint_asnorm(evaluated_pairs, cache_dict, num_folds=10, top_k=50, std_floor=0.05):
    """
    Applies strictly fold-disjoint AS-Norm to evaluated_pairs in-place:
    For each test fold t, builds the impostor cohort strictly from identities in the
    other (num_folds - 1) folds, explicitly excluding every identity appearing in fold t
    (zero test identity leakage).
    """
    fold_stats = []
    for test_fold in range(num_folds):
        test_pairs = [p for p in evaluated_pairs if p["fold"] == test_fold]
        if not test_pairs:
            continue

        test_identities = set()
        for p in test_pairs:
            test_identities.add(extract_identity_from_path(p["pair_a"]))
            test_identities.add(extract_identity_from_path(p["pair_b"]))

        cohort_paths_by_id = defaultdict(set)
        for p in evaluated_pairs:
            if p["fold"] == test_fold:
                continue
            for path in (p["pair_a"], p["pair_b"]):
                ident = extract_identity_from_path(path)
                if ident in test_identities:
                    continue
                if cache_dict.get(path) is not None:
                    cohort_paths_by_id[ident].add(path)

        # Assert strict disjointness
        leakage = set(cohort_paths_by_id.keys()) & test_identities
        if leakage:
            raise AssertionError(f"Fold {test_fold} AS-Norm cohort leakage detected: {leakage}")

        cohort_vecs = []
        for ident in sorted(cohort_paths_by_id.keys()):
            vecs = [np.asarray(cache_dict[path], dtype=np.float32).reshape(-1)
                    for path in sorted(cohort_paths_by_id[ident])]
            mean_v = np.mean(vecs, axis=0)
            norm = float(np.linalg.norm(mean_v))
            if norm > 1e-8:
                cohort_vecs.append((mean_v / norm).astype(np.float32))

        if not cohort_vecs:
            for p in test_pairs:
                raw_sim = p["cosine"]
                p["raw_cosine"] = raw_sim
                p["asnorm_z"] = raw_sim
                p["asnorm_cal"] = raw_sim
            fold_stats.append((test_fold, 0))
            continue

        cohort_matrix = np.stack(cohort_vecs).astype(np.float32)
        fold_stats.append((test_fold, cohort_matrix.shape[0]))

        for p in test_pairs:
            emb_a = np.asarray(cache_dict[p["pair_a"]], dtype=np.float32).reshape(-1)
            emb_b = np.asarray(cache_dict[p["pair_b"]], dtype=np.float32).reshape(-1)
            raw_sim = float(p["cosine"])
            raw_arr = np.array([raw_sim], dtype=np.float32)
            z_score = float(compute_asnorm_scores(
                emb_a,
                emb_b[None, :],
                cohort_matrix,
                candidate_raw_sims=raw_arr,
                top_k=top_k,
                std_floor=std_floor,
            )[0])
            cal_score = float(calibrate_asnorm_to_cosine(
                np.array([z_score], dtype=np.float32),
                raw_arr,
            )[0])
            p["raw_cosine"] = raw_sim
            p["asnorm_z"] = z_score
            p["asnorm_cal"] = cal_score
            p["cosine"] = z_score

    if fold_stats:
        cohort_sizes = [sz for _, sz in fold_stats]
        print(f">> Applied fold-disjoint AS-Norm (top_k={top_k}, zero test identity overlap): "
              f"cohort size min={min(cohort_sizes)}, max={max(cohort_sizes)} identities/fold")
    return fold_stats


def compute_verification_metrics(evaluated_pairs, num_folds=10):
    """
    Computes standard verification metrics across 10 folds:
    - 10-fold cross-validation accuracy (best threshold selected on 9 folds, tested on remaining 1).
    - Genuine vs impostor score distribution (mean +/- std).
    - TAR (True Accept Rate) @ FAR = 1% (0.01) and FAR = 0.1% (0.001), along with their thresholds.
    - AI-MIDS Project confidence tiers:
      * STRONG:   score >= 0.36
      * POSSIBLE: 0.26 <= score < 0.36
      * WEAK:     score < 0.26
    """
    if not evaluated_pairs:
        raise ValueError("No valid pairs to evaluate!")

    all_scores = np.array([p["cosine"] for p in evaluated_pairs], dtype=np.float32)
    s_min, s_max = float(np.min(all_scores)), float(np.max(all_scores))
    if s_min < -0.20 or s_max > 0.80:
        threshold_grid = np.linspace(s_min - 0.05, s_max + 0.05, 2001)
    else:
        threshold_grid = np.linspace(-0.20, 0.80, 1001)

    fold_accuracies = []
    best_thresholds = []

    for test_fold in range(num_folds):
        train = [p for p in evaluated_pairs if p["fold"] != test_fold]
        test = [p for p in evaluated_pairs if p["fold"] == test_fold]

        if not test:
            continue

        train_sims = np.array([p["cosine"] for p in train], dtype=np.float32)
        train_labels = np.array([p["is_same"] for p in train], dtype=np.int32)

        test_sims = np.array([p["cosine"] for p in test], dtype=np.float32)
        test_labels = np.array([p["is_same"] for p in test], dtype=np.int32)

        # Vectorized threshold tuning on training folds
        preds = train_sims[None, :] >= threshold_grid[:, None]
        accs = np.mean(preds == train_labels[None, :], axis=1)
        best_idx = int(np.argmax(accs))
        best_tau = float(threshold_grid[best_idx])
        best_thresholds.append(best_tau)

        # Evaluate on test fold
        test_acc = float(np.mean((test_sims >= best_tau) == test_labels))
        fold_accuracies.append(test_acc)

    mean_acc = float(np.mean(fold_accuracies)) * 100.0
    std_acc = float(np.std(fold_accuracies)) * 100.0
    mean_tau = float(np.mean(best_thresholds)) if best_thresholds else 0.0

    # Separate genuine and impostor scores
    gen_sims = np.array([p["cosine"] for p in evaluated_pairs if p["is_same"] == 1], dtype=np.float32)
    imp_sims = np.array([p["cosine"] for p in evaluated_pairs if p["is_same"] == 0], dtype=np.float32)

    # TAR @ FAR calculation helper
    def calc_tar_at_far(target_far):
        if len(imp_sims) == 0 or len(gen_sims) == 0:
            return 0.0, 0.0, 0.0
        # Sort impostors ascending: threshold at (1 - target_far) percentile
        sorted_imp = np.sort(imp_sims)
        cutoff_idx = int(np.floor((1.0 - target_far) * len(sorted_imp)))
        cutoff_idx = min(cutoff_idx, len(sorted_imp) - 1)
        tau = float(sorted_imp[cutoff_idx])
        actual_far = float(np.mean(imp_sims >= tau)) * 100.0
        tar = float(np.mean(gen_sims >= tau)) * 100.0
        return tar, tau, actual_far

    tar_far1, tau_far1, act_far1 = calc_tar_at_far(0.01)
    tar_far01, tau_far01, act_far01 = calc_tar_at_far(0.001)

    # Project confidence tiers
    gen_strong = float(np.mean(gen_sims >= 0.36)) * 100.0 if len(gen_sims) else 0.0
    gen_possible = float(np.mean((gen_sims >= 0.26) & (gen_sims < 0.36))) * 100.0 if len(gen_sims) else 0.0
    gen_weak = float(np.mean(gen_sims < 0.26)) * 100.0 if len(gen_sims) else 0.0

    imp_strong = float(np.mean(imp_sims >= 0.36)) * 100.0 if len(imp_sims) else 0.0
    imp_possible = float(np.mean((imp_sims >= 0.26) & (imp_sims < 0.36))) * 100.0 if len(imp_sims) else 0.0
    imp_weak = float(np.mean(imp_sims < 0.26)) * 100.0 if len(imp_sims) else 0.0

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_threshold": mean_tau,
        "num_folds_evaluated": len(fold_accuracies),
        "gen_count": len(gen_sims),
        "gen_mean": float(np.mean(gen_sims)) if len(gen_sims) else 0.0,
        "gen_std": float(np.std(gen_sims)) if len(gen_sims) else 0.0,
        "gen_min": float(np.min(gen_sims)) if len(gen_sims) else 0.0,
        "gen_max": float(np.max(gen_sims)) if len(gen_sims) else 0.0,
        "imp_count": len(imp_sims),
        "imp_mean": float(np.mean(imp_sims)) if len(imp_sims) else 0.0,
        "imp_std": float(np.std(imp_sims)) if len(imp_sims) else 0.0,
        "imp_min": float(np.min(imp_sims)) if len(imp_sims) else 0.0,
        "imp_max": float(np.max(imp_sims)) if len(imp_sims) else 0.0,
        "tar_far1": tar_far1,
        "tau_far1": tau_far1,
        "act_far1": act_far1,
        "tar_far01": tar_far01,
        "tau_far01": tau_far01,
        "act_far01": act_far01,
        "gen_strong": gen_strong,
        "gen_possible": gen_possible,
        "gen_weak": gen_weak,
        "imp_strong": imp_strong,
        "imp_possible": imp_possible,
        "imp_weak": imp_weak,
    }


def save_pairs_csv(csv_path, evaluated_pairs):
    """
    Saves individual verification pair scores to a CSV file.
    Columns: pair_a, pair_b, is_same, cosine, fold (plus raw_cosine, asnorm_cal if AS-Norm ran).
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    has_asnorm = bool(evaluated_pairs and "raw_cosine" in evaluated_pairs[0])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if has_asnorm:
            writer.writerow(["pair_a", "pair_b", "is_same", "cosine", "fold", "raw_cosine", "asnorm_cal"])
            for p in evaluated_pairs:
                writer.writerow([
                    p["pair_a"], p["pair_b"], p["is_same"],
                    f"{p['cosine']:.5f}", p["fold"],
                    f"{p['raw_cosine']:.5f}", f"{p['asnorm_cal']:.5f}",
                ])
        else:
            writer.writerow(["pair_a", "pair_b", "is_same", "cosine", "fold"])
            for p in evaluated_pairs:
                writer.writerow([p["pair_a"], p["pair_b"], p["is_same"], f"{p['cosine']:.5f}", p["fold"]])
    print(f">> Saved detailed pair scores ({len(evaluated_pairs)} rows) to: {csv_path}")


def print_benchmark_report(header_info, metrics, runtime_sec, cal_metrics=None):
    """
    Prints a structured, plain-language evaluation summary for student review.
    """
    print("\n" + "=" * 80)
    print("  AI-MIDS VERIFICATION BENCHMARK REPORT")
    print("=" * 80)
    for k, v in header_info.items():
        print(f"  * {k:<25}: {v}")
    print(f"  * {'Total Runtime':<25}: {runtime_sec:.1f}s ({runtime_sec/60:.2f} min)")
    print("-" * 80)

    print(f"\n[1] 10-Fold Verification Accuracy (Standard Protocol):")
    print(f"    * Mean Accuracy      : {metrics['mean_acc']:.2f}% ± {metrics['std_acc']:.2f}%")
    print(f"    * Optimal Threshold  : {metrics['mean_threshold']:.4f}")

    score_label = "AS-Norm Z-Score Distribution" if cal_metrics is not None else "Cosine Similarity Distribution"
    print(f"\n[2] {score_label}:")
    print(f"    * Genuine Pairs ({metrics['gen_count']})  : Mean {metrics['gen_mean']:.4f} ± {metrics['gen_std']:.4f}  "
          f"(Min: {metrics['gen_min']:.4f}, Max: {metrics['gen_max']:.4f})")
    print(f"    * Impostor Pairs ({metrics['imp_count']}) : Mean {metrics['imp_mean']:.4f} ± {metrics['imp_std']:.4f}  "
          f"(Min: {metrics['imp_min']:.4f}, Max: {metrics['imp_max']:.4f})")

    print(f"\n[3] Security Operating Points (TAR @ FAR):")
    print(f"    * TAR @ FAR = 1.0%   : {metrics['tar_far1']:.2f}%  (Threshold: {metrics['tau_far1']:.4f}, Actual FAR: {metrics['act_far1']:.2f}%)")
    print(f"    * TAR @ FAR = 0.1%   : {metrics['tar_far01']:.2f}%  (Threshold: {metrics['tau_far01']:.4f}, Actual FAR: {metrics['act_far01']:.2f}%)")

    tier_m = cal_metrics if cal_metrics is not None else metrics
    tier_label = "Calibrated AS-Norm Tiers (FaceStore Mapping)" if cal_metrics is not None else "AI-MIDS Project Confidence Tiers Breakdown"
    print(f"\n[4] {tier_label}:")
    print(f"    {'Tier':<20} | {'Score Range':<15} | {'Genuine Pairs':<18} | {'Impostor Pairs':<18}")
    print(f"    {'-'*20} | {'-'*15} | {'-'*18} | {'-'*18}")
    print(f"    {'STRONG':<20} | {'>= 0.36':<15} | {tier_m['gen_strong']:>6.1f}%             | {tier_m['imp_strong']:>6.1f}%")
    print(f"    {'POSSIBLE':<20} | {'0.26 to 0.359':<15} | {tier_m['gen_possible']:>6.1f}%             | {tier_m['imp_possible']:>6.1f}%")
    print(f"    {'WEAK':<20} | {'< 0.26':<15} | {tier_m['gen_weak']:>6.1f}%             | {tier_m['imp_weak']:>6.1f}%")
    if cal_metrics is not None:
        print(f"    * Calibrated 10-Fold Acc: {cal_metrics['mean_acc']:.2f}% ± {cal_metrics['std_acc']:.2f}% | "
              f"TAR@FAR=1%: {cal_metrics['tar_far1']:.2f}% | "
              f"Gen Mean: {cal_metrics['gen_mean']:.4f}, Imp Mean: {cal_metrics['imp_mean']:.4f}")
    print("=" * 80 + "\n")


# ==============================================================================
# 3. Main Benchmark Execution Function
# ==============================================================================

def run_verification_benchmark(dataset_name="agedb30", engine_name="antelopev2",
                               use_tta=False, limit=None, root=None,
                               cache_dir=ARC_DIR, results_dir=None, no_cache=False, seed=0,
                               use_asnorm=False, asnorm_topk=50):
    """
    Executes the verification benchmark end-to-end.
    Returns: (metrics_dict, summary_info_dict, runtime_sec)
    """
    t_start = time.time()
    if results_dir is None:
        results_dir = os.path.join(ARC_DIR, "bench_results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load dataset pairs
    print(f"\n>> Loading benchmark dataset '{dataset_name}'...")
    raw_pairs, protocol_desc = load_dataset_pairs(dataset_name, root=root, limit=limit, seed=seed)
    total_requested_pairs = len(raw_pairs)
    print(f">> Dataset loaded: {total_requested_pairs} pairs across 10 folds.")

    # 2. Extract unique images
    all_images = []
    for p in raw_pairs:
        all_images.append(p[0])
        all_images.append(p[1])
    unique_images = sorted(list(set(all_images)))
    print(f">> Total unique images involved in benchmark: {len(unique_images)}")

    # 3. Load or compute image embeddings
    cache_path = get_cache_path(cache_dir, dataset_name, engine_name, use_tta)
    cache_dict = {} if no_cache else load_cache(cache_path)
    quality_dict = {} if no_cache else load_quality_cache(cache_path)

    # Check which images need embedding
    needed = [p for p in unique_images if p not in cache_dict]
    if needed:
        engine = load_engine(engine_name, use_tta=use_tta)
        cache_dict = embed_unique_images(unique_images, engine, cache_dict, cache_path, quality_dict=quality_dict)
    else:
        print(f">> Fast path: All {len(unique_images)} images found in disk cache!")

    # 4. Score pairs
    evaluated_pairs = []
    skipped_count = 0

    for path_a, path_b, is_same, fold_idx in raw_pairs:
        emb_a = cache_dict.get(path_a)
        emb_b = cache_dict.get(path_b)

        if emb_a is None or emb_b is None:
            # Face not detected in one or both photos
            skipped_count += 1
            continue

        # Cosine similarity between unit-normalized vectors is simple dot product
        sim = float(np.dot(emb_a, emb_b))
        # Numerical guard: clamp to [-1.0, 1.0]
        sim = max(-1.0, min(1.0, sim))

        evaluated_pairs.append({
            "pair_a": path_a,
            "pair_b": path_b,
            "is_same": is_same,
            "cosine": sim,
            "fold": fold_idx
        })

    used_count = len(evaluated_pairs)
    print(f">> Pair scoring complete: {used_count} pairs evaluated, {skipped_count} skipped (no face detected).")

    cal_metrics = None
    if use_asnorm:
        apply_fold_disjoint_asnorm(evaluated_pairs, cache_dict, num_folds=10, top_k=asnorm_topk)
        cal_pairs = [
            {"cosine": p["asnorm_cal"], "is_same": p["is_same"], "fold": p["fold"]}
            for p in evaluated_pairs
        ]
        cal_metrics = compute_verification_metrics(cal_pairs, num_folds=10)

    # 5. Save detailed CSV
    tta_suffix = "_tta" if use_tta else ""
    asnorm_suffix = "_asnorm" if use_asnorm else ""
    csv_name = f"{dataset_name.lower()}_{engine_name.lower()}{tta_suffix}{asnorm_suffix}_pairs.csv"
    csv_path = os.path.join(results_dir, csv_name)
    save_pairs_csv(csv_path, evaluated_pairs)

    # 6. Compute metrics
    metrics = compute_verification_metrics(evaluated_pairs, num_folds=10)
    runtime = time.time() - t_start

    # 7. Print Report
    asnorm_label = f"ON (fold-disjoint, top_k={asnorm_topk})" if use_asnorm else "OFF"
    header_info = {
        "Dataset": dataset_name.upper(),
        "Protocol": protocol_desc,
        "Engine": f"{engine_name} (Flip-TTA: {'ON' if use_tta else 'OFF'})",
        "AS-Norm": asnorm_label,
        "Pairs Requested": total_requested_pairs,
        "Pairs Evaluated": f"{used_count} ({used_count/total_requested_pairs*100:.1f}%)",
        "Pairs Skipped": f"{skipped_count} ({skipped_count/total_requested_pairs*100:.1f}%)",
        "Unique Images": len(unique_images),
        "Scores CSV": csv_path,
    }
    print_benchmark_report(header_info, metrics, runtime, cal_metrics=cal_metrics)

    return metrics, header_info, runtime


# ==============================================================================
# 4. CLI Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="AI-MIDS Verification Benchmark Tool")
    parser.add_argument("--dataset", type=str, default="agedb30",
                        help="Dataset name: agedb30, cacd_vs, calfw, lag (default: agedb30)")
    parser.add_argument("--engine", type=str, default="antelopev2",
                        choices=["antelopev2", "buffalo_l", "aimids_ft", "aimids_crossage"],
                        help="Face recognition engine: antelopev2, buffalo_l, aimids_ft, or aimids_crossage (default: antelopev2)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable Flip Test-Time Augmentation (recommended for production antelopev2)")
    parser.add_argument("--asnorm", action="store_true",
                        help="Apply strictly fold-disjoint AS-Norm (cohort from other 9 folds only)")
    parser.add_argument("--asnorm-topk", type=int, default=50,
                        help="Top-k cohort members for AS-Norm (default: 50)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total number of pairs for faster CPU runs (e.g. 600 or 1000)")
    parser.add_argument("--root", type=str, default=None,
                        help="Custom root directory path for the dataset")
    parser.add_argument("--cache-dir", type=str, default=ARC_DIR,
                        help="Directory to store .npz embedding caches")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory to save pair score CSV files (default: Arc face/bench_results)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Do not read existing .npz cache (force re-embedding)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for deterministic protocol generation (default: 0)")

    args = parser.parse_args()

    try:
        run_verification_benchmark(
            dataset_name=args.dataset,
            engine_name=args.engine,
            use_tta=args.tta,
            limit=args.limit,
            root=args.root,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir,
            no_cache=args.no_cache,
            seed=args.seed,
            use_asnorm=args.asnorm,
            asnorm_topk=args.asnorm_topk,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] Dataset error:\n{e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Benchmark execution failed: {e}\n", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
