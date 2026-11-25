"""
test_agedb_benchmark.py - Evaluates Face Engines on the AgeDB Benchmark (30+ Year Age Gaps).

Optimized:
1. Filters the gallery to ONLY the identities involved in the test probes + distractors (e.g., 60-100 identities instead of 524 for fast evaluation in < 1 minute).
2. Prints a live progress counter so you always see exactly what photo is currently being processed.
3. Compares buffalo_l (ResNet-50) vs antelopev2 (ResNet-100) on 30+ year aging gaps.
"""
import os
import re
import sys
import time
import numpy as np
import cv2
import faiss

ARC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARC_DIR)
os.chdir(ARC_DIR)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from face_engine import FaceEngine
from face_engine_antelopev2 import FaceEngineAntelopeV2

AGEDB_DIR = r"D:\GRAD\Assets\AgeDB\images"

def parse_agedb(img_dir):
    pattern = re.compile(r"^(\d+)_(.+?)_(\d+)_([mf])\.jpg$", re.IGNORECASE)
    identities = {}
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"AgeDB directory not found: {img_dir}")
        
    for fname in sorted(os.listdir(img_dir)):
        m = pattern.match(fname)
        if m:
            img_id, name, age, gender = m.groups()
            age = int(age)
            full_path = os.path.join(img_dir, fname)
            identities.setdefault(name, []).append((age, full_path, gender))
            
    return identities


def build_protocol(identities, min_gap=30, max_probes=60, distractor_count=40, seed=42):
    """
    Selects 60 probes with >= 30 year gap, their corresponding gallery photos,
    plus distractor identities to keep evaluation fast (~30-45 seconds per model on CPU).
    """
    candidates = []
    for name, photos in sorted(identities.items()):
        photos.sort(key=lambda x: x[0])
        youngest_age, youngest_path, _ = photos[0]
        older = [p for p in photos if (p[0] - youngest_age) >= min_gap]
        if older:
            candidates.append((name, youngest_age, youngest_path, older))
            
    np.random.seed(seed)
    selected_idx = np.random.choice(len(candidates), size=min(max_probes, len(candidates)), replace=False)
    
    gallery = []   # (pid, name, youngest_age, path)
    probes = []    # (pid, name, probe_age, youngest_age, path)
    
    for pid, idx in enumerate(selected_idx, 1):
        name, y_age, y_path, older = candidates[idx]
        gallery.append((pid, name, y_age, y_path))
        # Pick one adult probe with largest gap
        best_probe = max(older, key=lambda x: x[0])
        probes.append((pid, name, best_probe[0], y_age, best_probe[1]))
        
    # Add distractors
    remaining_names = [c[0] for i, c in enumerate(candidates) if i not in selected_idx]
    distractor_idx = np.random.choice(len(remaining_names), size=min(distractor_count, len(remaining_names)), replace=False)
    for i, d_idx in enumerate(distractor_idx, len(gallery) + 1):
        d_name = remaining_names[d_idx]
        d_photos = identities[d_name]
        d_photos.sort(key=lambda x: x[0])
        gallery.append((i, d_name, d_photos[0][0], d_photos[0][1]))
        
    return gallery, probes


def run_benchmark(engine, engine_name, gallery, probes):
    print(f"\n>> Evaluating: {engine_name}...")
    t0 = time.time()
    
    # 1. Gallery
    index = faiss.IndexIDMap(faiss.IndexFlatIP(512))
    pids = []
    vecs = []
    
    print(f"  [1/2] Embedding gallery ({len(gallery)} identities)...", end="", flush=True)
    t_gal = time.time()
    for pid, name, age, path in gallery:
        emb_res = engine.embed(path)
        if emb_res is not None:
            vecs.append(emb_res["embedding"])
            pids.append(pid)
            
    index.add_with_ids(np.stack(vecs), np.array(pids, dtype=np.int64))
    print(f" done in {time.time() - t_gal:.1f}s ({len(pids)} faces indexed).", flush=True)
    
    # 2. Probes
    print(f"  [2/2] Searching {len(probes)} probes (30+ yr gap)...", end="", flush=True)
    t_probe = time.time()
    r1, r5, r10 = 0, 0, 0
    sims_target = []
    age_gaps = []
    total = len(probes)
    
    for pid, name, p_age, g_age, p_path in probes:
        age_gaps.append(p_age - g_age)
        emb_res = engine.embed(p_path)
        if emb_res is None:
            continue
            
        sims, hits = index.search(emb_res["embedding"][None, :], 10)
        hit_ids = list(hits[0])
        
        if pid in hit_ids:
            rank = hit_ids.index(pid) + 1
            sim = float(sims[0][hit_ids.index(pid)])
            sims_target.append(sim)
            
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
            if rank <= 10:
                r10 += 1
                
    elapsed = time.time() - t0
    print(f" done in {time.time() - t_probe:.1f}s.", flush=True)
    print(f"  -> Total time: {elapsed:.1f}s | Rank-1: {r1}/{total} ({r1/total*100:.1f}%) | Top-10: {r10}/{total} ({r10/total*100:.1f}%) | MeanSim: {np.mean(sims_target):.4f}")
    
    return {
        "engine": engine_name,
        "probes_count": total,
        "avg_gap": np.mean(age_gaps),
        "rank1_pct": (r1 / total) * 100,
        "top5_pct": (r5 / total) * 100,
        "top10_pct": (r10 / total) * 100,
        "mean_sim": np.mean(sims_target) if sims_target else 0.0,
        "sec_per_query": elapsed / total
    }


def print_summary(res_list):
    print("\n" + "=" * 80)
    print("  AGEDB BENCHMARK RESULTS (CROSS-AGE MATCHING: 30+ YEAR GAPS)")
    print("=" * 80)
    header = f"{'Engine':<20} | {'Rank-1':<10} | {'Top-5':<10} | {'Top-10':<10} | {'Mean Sim':<10} | {'Time/Query':<10}"
    print(header)
    print("-" * 80)
    for r in res_list:
        line = (f"{r['engine']:<20} | "
                f"{r['rank1_pct']:>5.1f}%     | "
                f"{r['top5_pct']:>5.1f}%     | "
                f"{r['top10_pct']:>5.1f}%     | "
                f"{r['mean_sim']:>8.4f}   | "
                f"{r['sec_per_query']:>6.2f}s")
        print(line)
    print("=" * 80)


if __name__ == "__main__":
    identities = parse_agedb(AGEDB_DIR)
    gallery, probes = build_protocol(identities, min_gap=30, max_probes=60, distractor_count=40, seed=42)
    avg_gap = np.mean([p[2] - p[3] for p in probes])
    
    print("=" * 80)
    print("  AI-MIDS BENCHMARK: AGEDB DATASET (30+ YEAR LONG-GAP PROGRESSION)")
    print("=" * 80)
    print(f"Identities in AgeDB: {len(identities)}")
    print(f"Protocol: {len(gallery)} gallery identities (60 targets + 40 distractors), {len(probes)} test probes.")
    print(f"Average age gap: {avg_gap:.1f} years (Range: {min([p[2]-p[3] for p in probes])} - {max([p[2]-p[3] for p in probes])} years).\n")
    
    results = []
    
    # 1. buffalo_l
    engine_b = FaceEngine(use_gpu=False)
    results.append(run_benchmark(engine_b, "buffalo_l (R-50)", gallery, probes))
    
    # 2. antelopev2
    engine_a = FaceEngineAntelopeV2(use_gpu=False, use_tta=False)
    results.append(run_benchmark(engine_a, "antelopev2 (R-100)", gallery, probes))
    
    # Summary
    print_summary(results)
