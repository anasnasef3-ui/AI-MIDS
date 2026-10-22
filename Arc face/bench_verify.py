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
import sys
import time
import argparse
import csv
import numpy as np

# Set environment and imports relative to Arc face directory
ARC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARC_DIR)
os.chdir(ARC_DIR)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from face_engine import FaceEngine
from face_engine_antelopev2 import FaceEngineAntelopeV2
from bench_datasets import load_benchmark_dataset


# ==============================================================================
# 1. Embedding Engine & Disk Cache
# ==============================================================================

def load_engine(engine_name, use_tta=False):
    """
    Loads either buffalo_l (ResNet-50) or antelopev2 (ResNet-100) engine on CPU.
    """
    engine_name = engine_name.lower()
    if engine_name == "antelopev2":
        print(f">> Loading AntelopeV2 engine (ResNet-100, TTA={use_tta})...")
        return FaceEngineAntelopeV2(use_gpu=False, use_tta=use_tta)
    elif engine_name == "buffalo_l":
        print(">> Loading Buffalo_L engine (ResNet-50)...")
        return FaceEngine(use_gpu=False)
    else:
        raise ValueError(f"Unknown engine '{engine_name}'. Supported: antelopev2, buffalo_l")


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


def save_cache(cache_path, cache_dict):
    """
    Saves the embedding cache dictionary to a compressed .npz file.
    """
    paths = list(cache_dict.keys())
    embeddings = []
    has_face = []
    zero_vec = np.zeros(512, dtype=np.float32)

    for p in paths:
        vec = cache_dict[p]
        if vec is not None:
            embeddings.append(vec)
            has_face.append(True)
        else:
            embeddings.append(zero_vec)
            has_face.append(False)

    np.savez_compressed(
        cache_path,
        paths=np.array(paths),
        embeddings=np.array(embeddings, dtype=np.float32),
        has_face=np.array(has_face, dtype=bool)
    )
    print(f">> Saved {len(paths)} embeddings to cache: {os.path.basename(cache_path)}")


def embed_unique_images(image_paths, engine, cache_dict, cache_path):
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
                    cache_dict[p] = emb_res["embedding"]
                    detected_count += 1
                else:
                    cache_dict[p] = None
                    no_face_count += 1
            except Exception as e:
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
                save_cache(cache_path, cache_dict)
    finally:
        # Guarantee cache is saved to disk even if interrupted
        save_cache(cache_path, cache_dict)
    return cache_dict


# ==============================================================================
# 2. Verification Evaluation & 10-Fold Cross-Validation
# ==============================================================================

def compute_verification_metrics(evaluated_pairs, num_folds=10):
    """
    Computes standard verification metrics across 10 folds:
    - 10-fold cross-validation accuracy (best threshold selected on 9 folds, tested on remaining 1).
    - Genuine vs impostor cosine similarity distribution (mean +/- std).
    - TAR (True Accept Rate) @ FAR = 1% (0.01) and FAR = 0.1% (0.001), along with their thresholds.
    - AI-MIDS Project confidence tiers:
      * STRONG:   cosine >= 0.36
      * POSSIBLE: 0.26 <= cosine < 0.36
      * WEAK:     cosine < 0.26
    """
    if not evaluated_pairs:
        raise ValueError("No valid pairs to evaluate!")

    fold_accuracies = []
    best_thresholds = []
    threshold_grid = np.linspace(-0.20, 0.80, 1001)

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
    Columns: pair_a, pair_b, is_same, cosine, fold
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_a", "pair_b", "is_same", "cosine", "fold"])
        for p in evaluated_pairs:
            writer.writerow([p["pair_a"], p["pair_b"], p["is_same"], f"{p['cosine']:.5f}", p["fold"]])
    print(f">> Saved detailed pair scores ({len(evaluated_pairs)} rows) to: {csv_path}")


def print_benchmark_report(header_info, metrics, runtime_sec):
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

    print(f"\n[2] Cosine Similarity Distribution:")
    print(f"    * Genuine Pairs ({metrics['gen_count']})  : Mean {metrics['gen_mean']:.4f} ± {metrics['gen_std']:.4f}  "
          f"(Min: {metrics['gen_min']:.4f}, Max: {metrics['gen_max']:.4f})")
    print(f"    * Impostor Pairs ({metrics['imp_count']}) : Mean {metrics['imp_mean']:.4f} ± {metrics['imp_std']:.4f}  "
          f"(Min: {metrics['imp_min']:.4f}, Max: {metrics['imp_max']:.4f})")

    print(f"\n[3] Security Operating Points (TAR @ FAR):")
    print(f"    * TAR @ FAR = 1.0%   : {metrics['tar_far1']:.2f}%  (Threshold: {metrics['tau_far1']:.4f}, Actual FAR: {metrics['act_far1']:.2f}%)")
    print(f"    * TAR @ FAR = 0.1%   : {metrics['tar_far01']:.2f}%  (Threshold: {metrics['tau_far01']:.4f}, Actual FAR: {metrics['act_far01']:.2f}%)")

    print(f"\n[4] AI-MIDS Project Confidence Tiers Breakdown:")
    print(f"    {'Tier':<20} | {'Cosine Range':<15} | {'Genuine Pairs':<18} | {'Impostor Pairs':<18}")
    print(f"    {'-'*20} | {'-'*15} | {'-'*18} | {'-'*18}")
    print(f"    {'STRONG':<20} | {'>= 0.36':<15} | {metrics['gen_strong']:>6.1f}%             | {metrics['imp_strong']:>6.1f}%")
    print(f"    {'POSSIBLE':<20} | {'0.26 to 0.359':<15} | {metrics['gen_possible']:>6.1f}%             | {metrics['imp_possible']:>6.1f}%")
    print(f"    {'WEAK':<20} | {'< 0.26':<15} | {metrics['gen_weak']:>6.1f}%             | {metrics['imp_weak']:>6.1f}%")
    print("=" * 80 + "\n")


# ==============================================================================
# 3. Main Benchmark Execution Function
# ==============================================================================

def run_verification_benchmark(dataset_name="agedb30", engine_name="antelopev2",
                               use_tta=False, limit=None, root=None,
                               cache_dir=ARC_DIR, results_dir=None, no_cache=False, seed=0):
    """
    Executes the verification benchmark end-to-end.
    Returns: (metrics_dict, summary_info_dict)
    """
    t_start = time.time()
    if results_dir is None:
        results_dir = os.path.join(ARC_DIR, "bench_results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load dataset pairs
    print(f"\n>> Loading benchmark dataset '{dataset_name}'...")
    raw_pairs, protocol_desc = load_benchmark_dataset(dataset_name, root=root, limit=limit, seed=seed)
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

    # Check which images need embedding
    needed = [p for p in unique_images if p not in cache_dict]
    if needed:
        engine = load_engine(engine_name, use_tta=use_tta)
        cache_dict = embed_unique_images(unique_images, engine, cache_dict, cache_path)
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

    # 5. Save detailed CSV
    tta_suffix = "_tta" if use_tta else ""
    csv_name = f"{dataset_name.lower()}_{engine_name.lower()}{tta_suffix}_pairs.csv"
    csv_path = os.path.join(results_dir, csv_name)
    save_pairs_csv(csv_path, evaluated_pairs)

    # 6. Compute metrics
    metrics = compute_verification_metrics(evaluated_pairs, num_folds=10)
    runtime = time.time() - t_start

    # 7. Print Report
    header_info = {
        "Dataset": dataset_name.upper(),
        "Protocol": protocol_desc,
        "Engine": f"{engine_name} (Flip-TTA: {'ON' if use_tta else 'OFF'})",
        "Pairs Requested": total_requested_pairs,
        "Pairs Evaluated": f"{used_count} ({used_count/total_requested_pairs*100:.1f}%)",
        "Pairs Skipped": f"{skipped_count} ({skipped_count/total_requested_pairs*100:.1f}%)",
        "Unique Images": len(unique_images),
        "Scores CSV": csv_path,
    }
    print_benchmark_report(header_info, metrics, runtime)

    return metrics, header_info, runtime


# ==============================================================================
# 4. CLI Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="AI-MIDS Verification Benchmark Tool")
    parser.add_argument("--dataset", type=str, default="agedb30",
                        help="Dataset name: agedb30, cacd_vs, calfw, lag (default: agedb30)")
    parser.add_argument("--engine", type=str, default="antelopev2",
                        choices=["antelopev2", "buffalo_l"],
                        help="Face recognition engine: antelopev2 or buffalo_l (default: antelopev2)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable Flip Test-Time Augmentation (recommended for production antelopev2)")
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
            seed=args.seed
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] Dataset error:\n{e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Benchmark execution failed: {e}\n", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
