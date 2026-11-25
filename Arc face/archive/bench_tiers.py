"""
bench_tiers.py - are the 0.36 / 0.26 confidence tiers still correct for how the API really scores?

bench_verify.py measures a PAIR of photos: cos(probe, other_photo). The production search in
store.py scores a PROBE against a registered PERSON:

    score = max( best single photo of that person, cos(probe, centroid of that person's photos) )

so the tier boundaries were calibrated on a slightly different quantity than the one the API
emits. This tool re-runs the same AgeDB-30 protocol and measures all four scores side by side,
each with its own 10-fold accuracy, optimal threshold, TAR@FAR and tier distribution:

    single     cos(probe, partner photo)                        <- what bench_verify.py reports
    max_photo  max over the person's gallery photos             <- half of the production rule
    centroid   cos(probe, mean-normalised gallery vector)       <- the other half
    fused      max(max_photo, centroid)                         <- EXACTLY store.py's rule

Gallery photos are the person's g oldest photos (default 3, minus the probe), which mirrors how a
family registers a missing child: a few old photos, then a newer photo is searched.

    cd "D:\GRAD\Project\Main files\Arc face"
    python bench_tiers.py --engine antelopev2 --tta --limit 200 --gallery 3

Reuses bench_verify.py's embedding cache (.npz), so only the new gallery photos cost CPU time on
the first run. Writes a per-pair CSV plus a printed comparison table.
"""
import os
import re
import sys
import csv
import time
import argparse

import numpy as np

ARC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARC_DIR)
os.chdir(ARC_DIR)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from bench_verify import (
    load_engine,
    load_cache,
    load_quality_cache,
    save_cache,
    get_cache_path,
    embed_unique_images,
    compute_verification_metrics,
)
from bench_datasets import load_benchmark_dataset, parse_agedb_images
from face_engine_custom import compute_asnorm_scores, calibrate_asnorm_to_cosine
from calibration import CalibrationBundle, resolve_bundle_path, compute_cllr, pav_cllr_min, compute_brier, compute_ece

BASE_RULES = ("single", "max_photo", "centroid", "fused", "qa_centroid", "qa_fused")
ASNORM_RULES = ("asnorm_cal", "asnorm_z")
CALIB_RULES = ("cal_cos", "cal_llr")
NAME_PATTERN = re.compile(r"^(\d+)_(.+?)_(\d+)_([mf])\.jpg$", re.IGNORECASE)


def identity_of(path):
    """Identity name from an AgeDB filename, or None if the file is not an AgeDB image."""
    m = NAME_PATTERN.match(os.path.basename(str(path)))
    return m.group(2) if m else None


def age_of(path):
    """Age integer from an AgeDB filename, or None if not an AgeDB image."""
    m = NAME_PATTERN.match(os.path.basename(str(path)))
    return int(m.group(3)) if m else None


def build_gallery(person_photos, probe_path, gallery_size):
    """The person's `gallery_size` oldest photos, never including the probe itself."""
    photos = [p for _, p, _ in sorted(person_photos, key=lambda x: (x[0], x[1]))
              if os.path.normcase(str(p)) != os.path.normcase(str(probe_path))]
    return photos[:gallery_size]


def probe_side(path_a, path_b, age_a, age_b):
    """Search with the NEWER photo; the old ones are what a family would have registered."""
    if age_b > age_a:
        return path_b, path_a
    return path_a, path_b


def compute_qa_centroid(gallery_embs, gallery_ages, gallery_quals):
    """Quality + inverse age-density weighted L2-normalized centroid (mirrors FaceStore._update_centroid)."""
    weights = np.empty(len(gallery_embs), dtype=np.float32)
    for i, (age_i, qual_i) in enumerate(zip(gallery_ages, gallery_quals)):
        q_raw = 1.0 if qual_i is None or not np.isfinite(qual_i) else float(qual_i)
        q_clamped = min(max(q_raw, 0.25), 1.0)
        if age_i is None:
            w_age = 1.0
        else:
            age_f = float(age_i)
            nearby = sum(
                1 for j, age_j in enumerate(gallery_ages)
                if j != i and age_j is not None and abs(float(age_j) - age_f) <= 2.0
            )
            w_age = 1.0 / (1.0 + nearby)
        weights[i] = q_clamped * w_age

    vecs = np.stack(gallery_embs).astype(np.float32)
    weighted = np.sum(vecs * weights[:, None], axis=0)
    norm = float(np.linalg.norm(weighted))
    return (weighted / norm).astype(np.float32) if norm > 1e-8 else None


def score_pair(probe_emb, gallery_embs, partner_emb, gallery_ages=None, gallery_quals=None):
    """Candidate scores for one probe against one registered person."""
    sims = [float(np.dot(probe_emb, g)) for g in gallery_embs]
    best_photo = max(sims)
    mean_v = np.mean(gallery_embs, axis=0)
    norm = float(np.linalg.norm(mean_v))
    centroid_vec = (mean_v / norm).astype(np.float32) if norm > 1e-8 else None
    centroid_sim = float(np.dot(probe_emb, centroid_vec)) if centroid_vec is not None else -1.0

    if gallery_ages is None:
        gallery_ages = [None] * len(gallery_embs)
    if gallery_quals is None:
        gallery_quals = [1.0] * len(gallery_embs)
    qa_centroid_vec = compute_qa_centroid(gallery_embs, gallery_ages, gallery_quals)
    qa_centroid_sim = float(np.dot(probe_emb, qa_centroid_vec)) if qa_centroid_vec is not None else -1.0

    return {
        "single": float(np.dot(probe_emb, partner_emb)),
        "max_photo": best_photo,
        "centroid": centroid_sim,
        "fused": max(best_photo, centroid_sim),
        "qa_centroid": qa_centroid_sim,
        "qa_fused": max(best_photo, qa_centroid_sim),
        "_qa_centroid_vec": qa_centroid_vec if qa_centroid_vec is not None else centroid_vec,
    }


def collect_evaluations(raw_pairs, identities, cache, quality_cache, gallery_size):
    """Turns protocol pairs into (probe, multi-photo gallery) comparisons. Returns rows, skipped."""
    rows, skipped = [], 0
    for path_a, path_b, is_same, fold in raw_pairs:
        id_a, id_b = identity_of(path_a), identity_of(path_b)
        if id_a is None or id_b is None:
            skipped += 1
            continue

        if is_same == 1:
            age_a = next(a for a, p, _ in identities[id_a] if os.path.normcase(p) == os.path.normcase(path_a))
            age_b = next(a for a, p, _ in identities[id_b] if os.path.normcase(p) == os.path.normcase(path_b))
            probe, partner = probe_side(path_a, path_b, age_a, age_b)
            person, photos = id_a, identities[id_a]
        else:
            probe, partner = path_a, path_b
            person, photos = id_b, identities[id_b]

        gallery = build_gallery(photos, probe, gallery_size)
        if not any(os.path.normcase(str(p)) == os.path.normcase(str(partner)) for p in gallery):
            gallery = [partner] + gallery[:gallery_size - 1]      # the protocol partner is always registered
        gallery_pairs = [(p, cache.get(p)) for p in gallery]
        probe_emb, partner_emb = cache.get(probe), cache.get(partner)
        if probe_emb is None or partner_emb is None or not any(e is not None for _, e in gallery_pairs):
            skipped += 1                                # no detectable face on either side
            continue

        valid_gallery = [(p, e) for p, e in gallery_pairs if e is not None]
        gallery_embs = [e for _, e in valid_gallery]
        gallery_ages = [age_of(p) for p, _ in valid_gallery]
        gallery_quals = [quality_cache.get(p, 1.0) for p, _ in valid_gallery]
        scores = score_pair(probe_emb, gallery_embs, partner_emb, gallery_ages=gallery_ages, gallery_quals=gallery_quals)
        qa_vec = scores.pop("_qa_centroid_vec")
        rows.append({"probe": probe, "person": person, "fold": fold, "is_same": is_same,
                     "gallery_n": len(gallery_embs), "gallery": gallery,
                     "_probe_emb": probe_emb, "_qa_centroid_vec": qa_vec,
                     "_probe_qual": quality_cache.get(probe, 1.0),
                     "_gallery_embs": gallery_embs, "_gallery_quals": gallery_quals,
                     "_probe_age": age_of(probe), "_gallery_ages": gallery_ages,
                     **scores})
    return rows, skipped


def apply_tiers_fold_disjoint_asnorm(rows, identities, cache, quality_cache, num_folds=10, top_k=50):
    """Applies strictly fold-disjoint AS-Norm over `qa_fused` using other-fold identity centroids."""
    cohort_sizes = []
    for test_fold in range(num_folds):
        test_rows = [r for r in rows if r["fold"] == test_fold]
        if not test_rows:
            continue

        test_identities = set()
        for r in test_rows:
            pid_probe = identity_of(r["probe"])
            if pid_probe is not None:
                test_identities.add(pid_probe)
            test_identities.add(r["person"])

        other_identities = set()
        for r in rows:
            if r["fold"] == test_fold:
                continue
            pid_probe = identity_of(r["probe"])
            if pid_probe is not None and pid_probe not in test_identities:
                other_identities.add(pid_probe)
            if r["person"] not in test_identities:
                other_identities.add(r["person"])

        if other_identities & test_identities:
            raise AssertionError("Fold-disjoint AS-Norm cohort identity leakage detected")

        cohort_vecs = []
        for ident in sorted(other_identities):
            embs, ages, quals = [], [], []
            for age_i, p_i, _ in identities.get(ident, []):
                e = cache.get(p_i)
                if e is not None:
                    embs.append(e)
                    ages.append(age_i)
                    quals.append(quality_cache.get(p_i, 1.0))
            if embs:
                c_vec = compute_qa_centroid(embs, ages, quals)
                if c_vec is not None:
                    cohort_vecs.append(c_vec)

        if not cohort_vecs:
            for r in test_rows:
                r["asnorm_z"] = r["qa_fused"]
                r["asnorm_cal"] = r["qa_fused"]
            cohort_sizes.append(0)
            continue

        cohort_matrix = np.stack(cohort_vecs).astype(np.float32)
        cohort_sizes.append(cohort_matrix.shape[0])
        for r in test_rows:
            cand_vec = r["_qa_centroid_vec"]
            raw_sim = float(r["qa_fused"])
            if cand_vec is None:
                r["asnorm_z"] = raw_sim
                r["asnorm_cal"] = raw_sim
                continue
            raw_arr = np.array([raw_sim], dtype=np.float32)
            z_score = float(compute_asnorm_scores(
                r["_probe_emb"],
                cand_vec[None, :],
                cohort_matrix,
                candidate_raw_sims=raw_arr,
                top_k=top_k,
            )[0])
            cal_score = float(calibrate_asnorm_to_cosine(
                np.array([z_score], dtype=np.float32),
                raw_arr,
            )[0])
            r["asnorm_z"] = z_score
            r["asnorm_cal"] = float(np.clip(cal_score, -1.0, 1.0))

    if cohort_sizes:
        print(f">> Applied fold-disjoint AS-Norm (top_k={top_k}, zero test identity overlap): "
              f"cohort size min={min(cohort_sizes)}, max={max(cohort_sizes)} centroids/fold")


def apply_bundle_calibration(rows, engine_name, calib_path=None):
    """Applies opt-in 4-layer CalibrationBundle (External AS-Norm + QMF LLR) symmetrically to photos & centroids."""
    path = calib_path or resolve_bundle_path(engine_name)
    bundle = CalibrationBundle.load(path)
    for r in rows:
        q_emb = r["_probe_emb"]
        q_probe = float(r.get("_probe_qual", 1.0))
        p_stats = bundle.asnorm.probe_stats(q_emb)

        # Photo side
        g_embs = np.stack(r["_gallery_embs"]).astype(np.float32)
        g_quals = np.asarray(r.get("_gallery_quals", [1.0] * len(g_embs)), dtype=np.float32)
        raw_p = (g_embs @ q_emb).astype(np.float32)
        mu_p, sig_p = bundle.asnorm.enroll_stats(g_embs)
        z_p = bundle.asnorm.score_batch(q_emb, cand_mu=mu_p, cand_sigma=sig_p, raw_sims=raw_p, probe_stats=p_stats)
        q_min_p = np.minimum(q_probe, g_quals)
        llr_p = bundle.qmf.predict_llr(z_p, raw_p, q_min=q_min_p)
        best_llr_p = float(np.max(llr_p))

        # Centroid side
        c_vec = r["_qa_centroid_vec"]
        if c_vec is not None:
            raw_c = np.array([float(np.dot(q_emb, c_vec))], dtype=np.float32)
            mu_c, sig_c = bundle.asnorm.enroll_stats(c_vec[None, :])
            z_c = bundle.asnorm.score_batch(q_emb, cand_mu=mu_c, cand_sigma=sig_c, raw_sims=raw_c, probe_stats=p_stats)
            q_cent = float(np.max(g_quals)) if len(g_quals) > 0 else 1.0
            best_llr_c = float(bundle.qmf.predict_llr(float(z_c[0]), float(raw_c[0]), q_min=min(q_probe, q_cent)))
        else:
            best_llr_c = -1e9

        best_llr = max(best_llr_p, best_llr_c)
        r["cal_llr"] = best_llr
        r["cal_cos"] = float(bundle.qmf.llr_to_cosine(best_llr))
        r["cal_prob"] = float(bundle.qmf.predict_prob(best_llr))

    llrs = np.array([r["cal_llr"] for r in rows], dtype=np.float64)
    probs = np.array([r["cal_prob"] for r in rows], dtype=np.float64)
    labels = np.array([r["is_same"] for r in rows], dtype=np.int32)
    diag = {
        "cllr": compute_cllr(llrs, labels),
        "cllr_min": pav_cllr_min(llrs, labels),
        "brier": compute_brier(probs, labels),
        "ece": compute_ece(probs, labels),
    }
    print(f">> Applied CalibrationBundle ({os.path.basename(path)}): "
          f"C_llr={diag['cllr']:.4f} (PAV min={diag['cllr_min']:.4f}), "
          f"Brier={diag['brier']:.4f}, ECE={diag['ece']:.4f}")
    return diag


def metrics_per_rule(rows, rules, num_folds=10):
    """Runs the existing 10-fold metric code once per scoring rule."""
    out = {}
    for rule in rules:
        if rule in ("asnorm_z", "cal_llr"):
            pairs = [{"cosine": float(r[rule]), "is_same": r["is_same"], "fold": r["fold"]} for r in rows]
        else:
            pairs = [{"cosine": max(-1.0, min(1.0, r[rule])), "is_same": r["is_same"], "fold": r["fold"]}
                     for r in rows]
        out[rule] = compute_verification_metrics(pairs, num_folds=num_folds)
    return out


def print_table(all_metrics, rows, header, rules):
    print("\n" + "=" * 96)
    print("  AI-MIDS TIER CALIBRATION REPORT (pair score vs the score the API really uses)")
    print("=" * 96)
    for k, v in header.items():
        print(f"  {k:22}: {v}")

    print(f"\n  {'rule':<13}{'10-fold acc':>13}{'opt tau':>10}{'TAR@FAR1%':>11}{'tau@FAR1%':>11}"
          f"{'gen mean':>10}{'imp mean':>10}{'imp->POSS+':>11}")
    print("  " + "-" * 90)
    for rule in rules:
        m = all_metrics[rule]
        leakage = m["imp_strong"] + m["imp_possible"]
        print(f"  {rule:<13}{m['mean_acc']:>11.2f}%{m['mean_threshold']:>10.4f}{m['tar_far1']:>10.2f}%"
              f"{m['tau_far1']:>11.4f}{m['gen_mean']:>10.4f}{m['imp_mean']:>10.4f}{leakage:>10.1f}%")

    print("\n  Genuine / impostor share of each tier (tiers are the hardcoded 0.36 / 0.26 cut-offs):")
    print(f"  {'rule':<13}{'gen STRONG':>12}{'gen POSSIBLE':>14}{'gen WEAK':>10}"
          f"{'imp STRONG':>12}{'imp POSSIBLE':>14}{'imp WEAK':>10}")
    print("  " + "-" * 88)
    for rule in rules:
        m = all_metrics[rule]
        print(f"  {rule:<13}{m['gen_strong']:>11.1f}%{m['gen_possible']:>13.1f}%{m['gen_weak']:>9.1f}%"
              f"{m['imp_strong']:>11.1f}%{m['imp_possible']:>13.1f}%{m['imp_weak']:>9.1f}%")

    fused = all_metrics.get("qa_fused", all_metrics["fused"])
    single = all_metrics["single"]
    print(f"\n  Mean gallery size: {np.mean([r['gallery_n'] for r in rows]):.2f} photos per person")
    print(f"  qa_fused runs {fused['gen_mean'] - single['gen_mean']:+.4f} higher than single on genuine"
          f" and {fused['imp_mean'] - single['imp_mean']:+.4f} on impostor matches.")
    print(f"  Threshold that holds impostors to FAR=1% under qa_fused: {fused['tau_far1']:.4f}"
          f" (accepts {fused['tar_far1']:.1f}% of genuine).")
    if "asnorm_cal" in all_metrics:
        cal = all_metrics["asnorm_cal"]
        print(f"  Threshold that holds impostors to FAR=1% under asnorm_cal: {cal['tau_far1']:.4f}"
              f" (accepts {cal['tar_far1']:.1f}% of genuine, imp->POSS+: {cal['imp_strong'] + cal['imp_possible']:.1f}%).")
    print(f"  Current tiers (STRONG 0.36 / POSSIBLE 0.26) were calibrated on the pair score"
          f" (opt tau {single['mean_threshold']:.4f}).")


def save_rows_csv(csv_path, rows, include_asnorm=False, include_calibrated=False):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        cols = ["probe", "person", "fold", "is_same", "gallery_n",
                "cos_single", "cos_max_photo", "cos_centroid", "cos_fused",
                "cos_qa_centroid", "cos_qa_fused"]
        if include_asnorm:
            cols.extend(["asnorm_cal", "asnorm_z"])
        if include_calibrated:
            cols.extend(["cal_cos", "cal_llr", "cal_prob"])
        cols.append("gallery_photos")
        writer.writerow(cols)
        for r in rows:
            vals = [r["probe"], r["person"], r["fold"], r["is_same"], r["gallery_n"],
                    f"{r['single']:.5f}", f"{r['max_photo']:.5f}",
                    f"{r['centroid']:.5f}", f"{r['fused']:.5f}",
                    f"{r['qa_centroid']:.5f}", f"{r['qa_fused']:.5f}"]
            if include_asnorm:
                vals.extend([f"{r['asnorm_cal']:.5f}", f"{r['asnorm_z']:.5f}"])
            if include_calibrated:
                vals.extend([f"{r['cal_cos']:.5f}", f"{r['cal_llr']:.5f}", f"{r['cal_prob']:.5f}"])
            vals.append(" | ".join(str(p) for p in r["gallery"]))
            writer.writerow(vals)
    print(f">> Saved per-pair scores ({len(rows)} rows) to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="AI-MIDS tier calibration under the production scoring rule")
    parser.add_argument("--dataset", type=str, default="agedb30")
    parser.add_argument("--engine", type=str, default="antelopev2",
                        choices=["antelopev2", "buffalo_l", "aimids_ft", "aimids_crossage"],
                        help="Face recognition engine: antelopev2, buffalo_l, aimids_ft, or aimids_crossage (default: antelopev2)")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--asnorm", action="store_true",
                        help="Also evaluate strictly fold-disjoint AS-Norm (calibrated & raw z-score)")
    parser.add_argument("--asnorm-topk", type=int, default=50,
                        help="Top-k cohort members for AS-Norm (default: 50)")
    parser.add_argument("--calibrated", action="store_true",
                        help="Also evaluate opt-in 4-layer CalibrationBundle (External AS-Norm + QMF LLR)")
    parser.add_argument("--calib-path", type=str, default=None,
                        help="Optional explicit path to calib_<model>.npz")
    parser.add_argument("--limit", type=int, default=200, help="Protocol pairs (bench_verify default run used 200)")
    parser.add_argument("--gallery", type=int, default=3, help="Registered photos per person (production uses up to 3)")
    parser.add_argument("--root", type=str, default=None, help="Dataset root (default D:\\GRAD\\Assets\\AgeDB)")
    parser.add_argument("--cache-dir", type=str, default=ARC_DIR)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    results_dir = args.results_dir or os.path.join(ARC_DIR, "bench_results")

    raw_pairs, protocol_desc = load_benchmark_dataset(args.dataset, root=args.root, limit=args.limit, seed=args.seed)
    print(f">> Protocol: {protocol_desc} | {len(raw_pairs)} pairs")

    root = args.root or r"D:\GRAD\Assets\AgeDB"
    identities, _ = parse_agedb_images(os.path.join(root, "images"))

    cache_path = get_cache_path(args.cache_dir, args.dataset, args.engine, args.tta)
    cache = {} if args.no_cache else load_cache(cache_path)
    quality_cache = {} if args.no_cache else load_quality_cache(cache_path)

    # Every photo that could end up in a gallery, plus the protocol photos themselves
    needed = set()
    for path_a, path_b, _is_same, _fold in raw_pairs:
        for path in (path_a, path_b):
            pid = identity_of(path)
            if pid is not None:
                needed.add(path)
                for g in build_gallery(identities[pid], path, args.gallery):
                    needed.add(g)
    needed = sorted(needed)
    print(f">> Images to embed for galleries of up to {args.gallery} photos: {len(needed)}")

    missing = [p for p in needed if p not in cache]
    if missing:
        engine = load_engine(args.engine, use_tta=args.tta)
        cache = embed_unique_images(needed, engine, cache, cache_path, quality_dict=quality_cache)
        save_cache(cache_path, cache, quality_dict=quality_cache)
    else:
        print(">> Fast path: every gallery photo already in the embedding cache")

    rows, skipped = collect_evaluations(raw_pairs, identities, cache, quality_cache, args.gallery)
    gen = sum(1 for r in rows if r["is_same"] == 1)
    print(f">> Evaluated {len(rows)} comparisons ({gen} genuine / {len(rows) - gen} impostor), skipped {skipped}")

    rules = BASE_RULES
    if args.asnorm:
        apply_tiers_fold_disjoint_asnorm(rows, identities, cache, quality_cache, num_folds=10, top_k=args.asnorm_topk)
        rules = rules + ASNORM_RULES
    calib_diag = None
    if args.calibrated:
        calib_diag = apply_bundle_calibration(rows, args.engine, calib_path=args.calib_path)
        rules = rules + CALIB_RULES

    all_metrics = metrics_per_rule(rows, rules)
    header = {
        "Dataset": args.dataset.upper(),
        "Protocol": protocol_desc,
        "Engine": f"{args.engine} (Flip-TTA: {'ON' if args.tta else 'OFF'})",
        "AS-Norm": f"ON (fold-disjoint, top_k={args.asnorm_topk})" if args.asnorm else "OFF",
        "4-Layer Calib": (f"ON (C_llr={calib_diag['cllr']:.4f}, PAV={calib_diag['cllr_min']:.4f})"
                          if calib_diag is not None else "OFF"),
        "Gallery photos": f"up to {args.gallery} per person, probe excluded",
        "Comparisons": f"{len(rows)} ({gen} genuine / {len(rows) - gen} impostor)",
        "Skipped": skipped,
        "Runtime": f"{(time.time() - t0) / 60:.1f} min",
    }
    print_table(all_metrics, rows, header, rules)

    tta_suffix = "_tta" if args.tta else ""
    asnorm_suffix = "_asnorm" if args.asnorm else ""
    cal_suffix = "_calibrated" if args.calibrated else ""
    csv_path = os.path.join(
        results_dir,
        f"{args.dataset.lower()}_{args.engine.lower()}{tta_suffix}{asnorm_suffix}{cal_suffix}_tiers.csv",
    )
    save_rows_csv(csv_path, rows, include_asnorm=args.asnorm, include_calibrated=args.calibrated)
    return all_metrics


if __name__ == "__main__":
    main()
