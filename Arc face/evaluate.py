"""
evaluate.py - Unified GPU-Enabled Evaluation CLI for AI-MIDS.

Replaces bench_verify.py and bench_tiers.py with a single tool that evaluates
exact production scoring across all three project benchmarks:
  - agedb30: 10-fold AgeDB-30 verification with multi-photo QA-weighted galleries (--gallery 3)
  - calfw:   10-fold CALFW 1:1 pair verification
  - fgnet:   Child (<= 12y) -> Adult (> 12y) 1:N retrieval directly through FaceStore(":memory:")

Usage Examples:
    python evaluate.py
    python evaluate.py --dataset agedb30 --engine antelopev2 --limit 600
    python evaluate.py --dataset fgnet --engine aimids_ft --onnx pretrained\\aimids_scratch_ir50_crossage.onnx
    python evaluate.py --dataset all --limit 0
"""

import os

# Must be set before importing numpy / onnxruntime / faiss to avoid Windows virtual commit bloat.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import re
import sys
import time
import numpy as np

ARC_DIR = os.path.dirname(os.path.abspath(__file__))
if ARC_DIR not in sys.path:
    sys.path.insert(0, ARC_DIR)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bench_datasets import (
    load_benchmark_dataset,
    load_cache,
    load_caf_gallery,
    load_ecaf_gallery,
    load_fgnet_gallery,
    load_quality_cache,
    normalize_identity,
    parse_agedb_images,
    save_cache,
)
from calibration import CalibrationBundle, extract_face_quality, resolve_bundle_path
from face_engine import FaceEngine, ensure_cuda_dll_paths
from face_engine_adaface_ft import FaceEngineAdaFaceFT
from face_engine_antelopev2 import FaceEngineAntelopeV2
from face_engine_custom import FaceEngineCrossAge
from store import FaceStore

AGEDB_ROOT = r"D:\GRAD\Assets\AgeDB"
_AGEDB_FILE_RE = re.compile(r"^(\d+)_(.+?)_(\d+)_([mf])\.jpg$", re.IGNORECASE)


def _sanitize_onnx_stem(onnx_path):
    """Returns a filesystem-safe tag from an ONNX filename stem."""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_").lower()
    return clean or "custom"


def resolve_custom_onnx(onnx_arg):
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


def get_cache_path(cache_dir, dataset_name, engine_name, use_tta, onnx_path=None):
    """Builds the .npz embedding cache path, keeping custom --onnx runs isolated."""
    onnx_tag = f"_{_sanitize_onnx_stem(onnx_path)}" if onnx_path else ""
    tta_tag = "_tta" if use_tta else ""
    fname = f"bench_cache_{dataset_name.lower()}_{engine_name.lower()}{onnx_tag}{tta_tag}.npz"
    return os.path.join(cache_dir, fname)


def load_engine(engine_name, use_gpu=True, use_tta=True, onnx_path=None):
    """Initializes the requested recognition engine matching production settings."""
    name = engine_name.lower()
    if onnx_path and name not in ("aimids_ft", "aimids_crossage"):
        raise ValueError(f"--onnx is only supported with --engine aimids_ft or aimids_crossage, got '{engine_name}'")

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


def resolve_calibration_bundle(engine_name, onnx_path=None):
    """Loads CalibrationBundle for standard calibrated engines; returns None for custom ONNX or uncalibrated engines."""
    if onnx_path is not None:
        return None
    bundle_path = resolve_bundle_path(engine_name)
    if os.path.isfile(bundle_path):
        return CalibrationBundle.load(bundle_path)
    return None


def ensure_embeddings(image_paths, cache_path, get_engine_fn):
    """Loads disk embeddings/qualities and embeds only uncached images."""
    unique_paths = sorted(set(image_paths))
    cache_dict = load_cache(cache_path)
    quality_dict = load_quality_cache(cache_path)

    missing = [p for p in unique_paths if p not in cache_dict]
    if not missing:
        print(f">> Fast path: all {len(unique_paths)} unique images found in cache.")
        return cache_dict, quality_dict

    engine = get_engine_fn()
    device_label = "GPU" if getattr(engine, "use_gpu", False) else "CPU"
    print(f">> Embedding {len(missing)} / {len(unique_paths)} uncached images on {device_label}...")
    t0 = time.time()

    try:
        for idx, p in enumerate(missing, 1):
            try:
                img = engine._read(p)
                emb_res = engine.embed(img)
                if emb_res is not None and "embedding" in emb_res:
                    cache_dict[p] = np.asarray(emb_res["embedding"], dtype=np.float32).reshape(-1)
                    quality_dict[p] = float(extract_face_quality(engine, img, emb_res))
                else:
                    cache_dict[p] = None
            except Exception:
                cache_dict[p] = None

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


def _clamp_quality(q_val):
    if q_val is None or not np.isfinite(q_val):
        return 1.0
    return float(min(max(float(q_val), 0.25), 1.0))


def compute_qa_centroid(gallery_embs, gallery_ages, gallery_quals):
    """Quality + inverse age-density weighted L2-normalized centroid (mirrors FaceStore._update_centroid)."""
    weights = np.empty(len(gallery_embs), dtype=np.float32)
    for i, (age_i, qual_i) in enumerate(zip(gallery_ages, gallery_quals)):
        q_clamped = _clamp_quality(qual_i)
        if age_i is None:
            w_age = 1.0
        else:
            age_f = float(age_i)
            nearby = sum(
                1
                for j, age_j in enumerate(gallery_ages)
                if j != i and age_j is not None and abs(float(age_j) - age_f) <= 2.0
            )
            w_age = 1.0 / (1.0 + nearby)
        weights[i] = q_clamped * w_age

    vecs = np.stack(gallery_embs).astype(np.float32)
    weighted = np.sum(vecs * weights[:, None], axis=0)
    norm = float(np.linalg.norm(weighted))
    return (weighted / norm).astype(np.float32) if norm > 1e-8 else None


def compute_verification_metrics(evaluated_pairs, num_folds=10):
    """Computes 10-fold accuracy, TAR@FAR=1%, Genuine STRONG %, and Impostor >=0.26 %."""
    if not evaluated_pairs:
        raise ValueError("No valid pairs to evaluate!")

    all_scores = np.array([p["score"] for p in evaluated_pairs], dtype=np.float32)
    s_min, s_max = float(np.min(all_scores)), float(np.max(all_scores))
    if s_min < -0.20 or s_max > 0.80:
        threshold_grid = np.linspace(s_min - 0.05, s_max + 0.05, 2001)
    else:
        threshold_grid = np.linspace(-0.20, 0.80, 1001)

    fold_accuracies = []
    for test_fold in range(num_folds):
        train = [p for p in evaluated_pairs if p["fold"] != test_fold]
        test = [p for p in evaluated_pairs if p["fold"] == test_fold]
        if not train or not test:
            continue

        train_sims = np.array([p["score"] for p in train], dtype=np.float32)
        train_labels = np.array([p["is_same"] for p in train], dtype=np.int32)
        test_sims = np.array([p["score"] for p in test], dtype=np.float32)
        test_labels = np.array([p["is_same"] for p in test], dtype=np.int32)

        preds = train_sims[None, :] >= threshold_grid[:, None]
        accs = np.mean(preds == train_labels[None, :], axis=1)
        best_tau = float(threshold_grid[int(np.argmax(accs))])
        fold_accuracies.append(float(np.mean((test_sims >= best_tau) == test_labels)))

    gen_sims = np.array([p["score"] for p in evaluated_pairs if p["is_same"] == 1], dtype=np.float32)
    imp_sims = np.array([p["score"] for p in evaluated_pairs if p["is_same"] == 0], dtype=np.float32)

    if len(imp_sims) > 0 and len(gen_sims) > 0:
        sorted_imp = np.sort(imp_sims)
        cutoff_idx = min(int(np.floor(0.99 * len(sorted_imp))), len(sorted_imp) - 1)
        tau_far1 = float(sorted_imp[cutoff_idx])
        tar_far1 = float(np.mean(gen_sims >= tau_far1)) * 100.0
    else:
        tar_far1 = 0.0

    gen_strong = float(np.mean(gen_sims >= 0.36)) * 100.0 if len(gen_sims) else 0.0
    imp_poss_plus = float(np.mean(imp_sims >= 0.26)) * 100.0 if len(imp_sims) else 0.0

    return {
        "mean_acc": float(np.mean(fold_accuracies)) * 100.0 if fold_accuracies else 0.0,
        "std_acc": float(np.std(fold_accuracies)) * 100.0 if fold_accuracies else 0.0,
        "tar_far1": tar_far1,
        "gen_strong": gen_strong,
        "imp_poss_plus": imp_poss_plus,
        "evaluated": len(evaluated_pairs),
    }


def _agedb_identity(path):
    m = _AGEDB_FILE_RE.match(os.path.basename(str(path)))
    return m.group(2) if m else None


def _agedb_age(path):
    m = _AGEDB_FILE_RE.match(os.path.basename(str(path)))
    return int(m.group(3)) if m else None


def _build_agedb_gallery(person_photos, probe_path, gallery_size):
    """Selects up to `gallery_size` oldest photos of a person excluding the probe."""
    photos = [
        p
        for _, p, _ in sorted(person_photos, key=lambda x: (x[0], x[1]))
        if os.path.normcase(str(p)) != os.path.normcase(str(probe_path))
    ]
    return photos[:gallery_size]


def evaluate_agedb30(engine_name, use_tta, pair_limit, gallery_size, seed, onnx_path, bundle, get_engine_fn):
    """Evaluates AgeDB-30 using multi-photo QA-weighted galleries matching production search."""
    print("\n>> [AgeDB-30] Preparing multi-photo gallery evaluation...")
    raw_pairs, _ = load_benchmark_dataset("agedb30", root=AGEDB_ROOT, limit=pair_limit, seed=seed)
    identities, _ = parse_agedb_images(os.path.join(AGEDB_ROOT, "images"))

    needed = set()
    for path_a, path_b, _is_same, _fold in raw_pairs:
        for path in (path_a, path_b):
            pid = _agedb_identity(path)
            if pid is not None and pid in identities:
                needed.add(path)
                for g in _build_agedb_gallery(identities[pid], path, gallery_size):
                    needed.add(g)

    cache_path = get_cache_path(ARC_DIR, "agedb30", engine_name, use_tta, onnx_path=onnx_path)
    cache, quality_cache = ensure_embeddings(needed, cache_path, get_engine_fn)

    scored_pairs = []
    skipped = 0
    for path_a, path_b, is_same, fold_idx in raw_pairs:
        id_a, id_b = _agedb_identity(path_a), _agedb_identity(path_b)
        if id_a is None or id_b is None or id_a not in identities or id_b not in identities:
            skipped += 1
            continue

        if is_same == 1:
            age_a = next(a for a, p, _ in identities[id_a] if os.path.normcase(p) == os.path.normcase(path_a))
            age_b = next(a for a, p, _ in identities[id_b] if os.path.normcase(p) == os.path.normcase(path_b))
            probe, partner = (path_b, path_a) if age_b > age_a else (path_a, path_b)
            photos = identities[id_a]
        else:
            probe, partner = path_a, path_b
            photos = identities[id_b]

        gallery = _build_agedb_gallery(photos, probe, gallery_size)
        if not any(os.path.normcase(str(p)) == os.path.normcase(str(partner)) for p in gallery):
            gallery = [partner] + gallery[: max(0, gallery_size - 1)]

        probe_emb = cache.get(probe)
        partner_emb = cache.get(partner)
        valid_gallery = [(p, cache.get(p)) for p in gallery if cache.get(p) is not None]
        if probe_emb is None or partner_emb is None or not valid_gallery:
            skipped += 1
            continue

        gallery_embs = [e for _, e in valid_gallery]
        gallery_ages = [_agedb_age(p) for p, _ in valid_gallery]
        gallery_quals = [_clamp_quality(quality_cache.get(p, 1.0)) for p, _ in valid_gallery]
        qa_centroid_vec = compute_qa_centroid(gallery_embs, gallery_ages, gallery_quals)

        if bundle is not None:
            q_probe = _clamp_quality(quality_cache.get(probe, 1.0))
            p_stats = bundle.asnorm.probe_stats(probe_emb)

            g_mat = np.stack(gallery_embs).astype(np.float32)
            g_quals = np.asarray(gallery_quals, dtype=np.float32)
            raw_p = (g_mat @ probe_emb).astype(np.float32)
            mu_p, sig_p = bundle.asnorm.enroll_stats(g_mat)
            z_p = bundle.asnorm.score_batch(
                probe_emb, cand_mu=mu_p, cand_sigma=sig_p, raw_sims=raw_p, probe_stats=p_stats
            )
            llr_p = bundle.qmf.predict_llr(z_p, raw_p, q_min=np.minimum(q_probe, g_quals))
            best_llr_p = float(np.max(llr_p))

            if qa_centroid_vec is not None:
                raw_c = np.array([float(np.dot(probe_emb, qa_centroid_vec))], dtype=np.float32)
                mu_c, sig_c = bundle.asnorm.enroll_stats(qa_centroid_vec[None, :])
                z_c = bundle.asnorm.score_batch(
                    probe_emb, cand_mu=mu_c, cand_sigma=sig_c, raw_sims=raw_c, probe_stats=p_stats
                )
                q_cent = float(np.max(g_quals)) if len(g_quals) > 0 else 1.0
                best_llr_c = float(bundle.qmf.predict_llr(float(z_c[0]), float(raw_c[0]), q_min=min(q_probe, q_cent)))
            else:
                best_llr_c = -1e9

            score = float(bundle.qmf.llr_to_cosine(max(best_llr_p, best_llr_c)))
        else:
            best_photo = max(float(np.clip(np.dot(probe_emb, g), -1.0, 1.0)) for g in gallery_embs)
            qa_cent_sim = (
                float(np.clip(np.dot(probe_emb, qa_centroid_vec), -1.0, 1.0))
                if qa_centroid_vec is not None
                else -1.0
            )
            score = max(best_photo, qa_cent_sim)

        scored_pairs.append({"score": score, "is_same": is_same, "fold": fold_idx})

    metrics = compute_verification_metrics(scored_pairs, num_folds=10)
    return {
        "dataset": f"AgeDB-30 (gallery={gallery_size})",
        "mode": "Calibrated" if bundle is not None else "Cosine Fallback",
        "skipped": skipped,
        **metrics,
    }


def evaluate_calfw(engine_name, use_tta, pair_limit, seed, onnx_path, bundle, get_engine_fn):
    """Evaluates CALFW 10-fold 1:1 pair verification."""
    print("\n>> [CALFW] Preparing 1:1 pair verification...")
    raw_pairs, _ = load_benchmark_dataset("calfw", limit=pair_limit, seed=seed)
    unique_imgs = [p for pair in raw_pairs for p in (pair[0], pair[1])]

    cache_path = get_cache_path(ARC_DIR, "calfw", engine_name, use_tta, onnx_path=onnx_path)
    cache, quality_cache = ensure_embeddings(unique_imgs, cache_path, get_engine_fn)

    probe_stats_cache = {}
    enroll_stats_cache = {}
    scored_pairs = []
    skipped = 0

    for path_a, path_b, is_same, fold_idx in raw_pairs:
        emb_a = cache.get(path_a)
        emb_b = cache.get(path_b)
        if emb_a is None or emb_b is None:
            skipped += 1
            continue

        raw_sim = float(np.clip(np.dot(emb_a, emb_b), -1.0, 1.0))
        if bundle is not None:
            p_stats = probe_stats_cache.get(path_a)
            if p_stats is None:
                p_stats = bundle.asnorm.probe_stats(emb_a)
                probe_stats_cache[path_a] = p_stats
            e_stats = enroll_stats_cache.get(path_b)
            if e_stats is None:
                e_stats = bundle.asnorm.enroll_stats(emb_b[None, :])
                enroll_stats_cache[path_b] = e_stats
            mu_b, sig_b = e_stats
            z = float(
                bundle.asnorm.score_batch(
                    emb_a,
                    cand_mu=mu_b,
                    cand_sigma=sig_b,
                    raw_sims=np.array([raw_sim], dtype=np.float32),
                    probe_stats=p_stats,
                )[0]
            )
            q_min = min(_clamp_quality(quality_cache.get(path_a, 1.0)), _clamp_quality(quality_cache.get(path_b, 1.0)))
            llr = float(bundle.qmf.predict_llr(z, raw_sim, q_min=q_min))
            score = float(bundle.qmf.llr_to_cosine(llr))
        else:
            score = raw_sim

        scored_pairs.append({"score": score, "is_same": is_same, "fold": fold_idx})

    metrics = compute_verification_metrics(scored_pairs, num_folds=10)
    return {
        "dataset": "CALFW (1 vs 1)",
        "mode": "Calibrated" if bundle is not None else "Cosine Fallback",
        "skipped": skipped,
        **metrics,
    }


def _evaluate_retrieval_gallery(
    dataset_label,
    cache_name,
    gallery_by_id,
    probes,
    engine_name,
    use_tta,
    onnx_path,
    bundle,
    get_engine_fn,
    cache_override=None,
    quality_override=None,
    person_prefix="",
):
    """Evaluates 1:N missing->found retrieval directly through FaceStore(":memory:")."""
    if cache_override is not None and quality_override is not None:
        cache, quality_cache = cache_override, quality_override
    else:
        all_paths = [path for items in gallery_by_id.values() for _, path in items] + [pr["path"] for pr in probes]
        cache_path = get_cache_path(ARC_DIR, cache_name, engine_name, use_tta, onnx_path=onnx_path)
        cache, quality_cache = ensure_embeddings(all_paths, cache_path, get_engine_fn)

    store_model_name = f"{engine_name}__{_sanitize_onnx_stem(onnx_path)}" if onnx_path else engine_name
    store = FaceStore(":memory:")
    try:
        if bundle is not None:
            store.set_calibration_bundle(store_model_name, bundle)

        pid_map = {}
        for subj_id in sorted(gallery_by_id.keys()):
            pid = store.add_person(f"{person_prefix}{subj_id}")
            added = 0
            for age, path in gallery_by_id[subj_id]:
                emb = cache.get(path)
                if emb is None:
                    continue
                q_val = _clamp_quality(quality_cache.get(path, 1.0))
                store.add_photo(
                    pid,
                    emb,
                    photo_path=path,
                    age=age,
                    model=store_model_name,
                    quality=q_val,
                )
                added += 1
            if added > 0:
                pid_map[subj_id] = pid
            else:
                store.delete_person(pid)

        mode_label = "Calibrated" if store.calibrated_ready(store_model_name) else "Cosine Fallback"
        evaluated = 0
        skipped = 0
        rank1 = 0
        top5 = 0
        top10 = 0
        strong_hits = 0
        possible_plus_hits = 0

        for pr in probes:
            emb_p = cache.get(pr["path"])
            target_pid = pid_map.get(pr["subject_id"])
            if emb_p is None or target_pid is None:
                skipped += 1
                continue

            evaluated += 1
            q_probe = _clamp_quality(quality_cache.get(pr["path"], 1.0))
            hits = store.search(
                emb_p,
                k=10,
                min_sim=-1.0,
                model=store_model_name,
                probe_quality=q_probe,
            )
            hit_pids = [h["person_id"] for h in hits]
            if hit_pids and hit_pids[0] == target_pid:
                rank1 += 1
            if target_pid in hit_pids[:5]:
                top5 += 1
            if target_pid in hit_pids[:10]:
                top10 += 1

            target_hit = next((h for h in hits if h["person_id"] == target_pid), None)
            if target_hit is not None:
                if target_hit["tier"] == "STRONG":
                    strong_hits += 1
                if target_hit["tier"] in ("STRONG", "possible"):
                    possible_plus_hits += 1
    finally:
        store.db.close()

    denom = max(1, evaluated)
    return {
        "dataset": dataset_label,
        "mode": mode_label,
        "gallery_ids": len(pid_map),
        "probes": evaluated,
        "skipped": skipped,
        "rank1": rank1 / denom * 100.0,
        "top5": top5 / denom * 100.0,
        "top10": top10 / denom * 100.0,
        "strong": strong_hits / denom * 100.0,
        "possible_plus": possible_plus_hits / denom * 100.0,
    }, cache, quality_cache


def evaluate_fgnet(engine_name, use_tta, onnx_path, bundle, get_engine_fn):
    """Evaluates FG-NET child (<=12y) -> adult (>12y) 1:N retrieval directly through FaceStore."""
    print("\n>> [FG-NET] Preparing child -> adult FaceStore retrieval...")
    gallery_by_id, probes = load_fgnet_gallery()
    row, _, _ = _evaluate_retrieval_gallery(
        "FG-NET (<=12y -> >12y)",
        "fgnet",
        gallery_by_id,
        probes,
        engine_name,
        use_tta,
        onnx_path,
        bundle,
        get_engine_fn,
        person_prefix="FGNET_",
    )
    return row


def evaluate_caf(engine_name, use_tta, onnx_path, bundle, get_engine_fn, exclude_known=False, split="child_to_adult"):
    """Evaluates CAF missing->found 1:N retrieval (cohort-disjoint by default)."""
    scope = "unseen" if exclude_known else "cohort-disjoint"
    print(f"\n>> [CAF] Preparing {split} ({scope}) FaceStore retrieval...")
    gallery_by_id, probes = load_caf_gallery(
        exclude_cohort=True,
        exclude_known=exclude_known,
        split=split,
    )
    tag = "<=12y->adult" if split == "child_to_adult" else "oldest-2"
    row, _, _ = _evaluate_retrieval_gallery(
        f"CAF ({tag}, {scope})",
        "caf",
        gallery_by_id,
        probes,
        engine_name,
        use_tta,
        onnx_path,
        bundle,
        get_engine_fn,
        person_prefix="CAF_",
    )
    return row


def evaluate_ecaf(
    engine_name,
    use_tta,
    onnx_path,
    bundle,
    get_engine_fn,
    exclude_known=False,
    official_test_only=False,
):
    """Evaluates ECAF child -> adult missing-person 1:N retrieval (cohort-disjoint by default)."""
    scope = "unseen" if exclude_known else "cohort-disjoint"
    split_tag = "official, " if official_test_only else ""
    print(f"\n>> [ECAF] Preparing child -> adult ({split_tag}{scope}) FaceStore retrieval...")
    gallery_by_id, probes = load_ecaf_gallery(
        exclude_cohort=True,
        exclude_known=exclude_known,
        official_test_only=official_test_only,
    )
    row, _, _ = _evaluate_retrieval_gallery(
        f"ECAF ({split_tag}{scope})",
        "ecaf",
        gallery_by_id,
        probes,
        engine_name,
        use_tta,
        onnx_path,
        bundle,
        get_engine_fn,
        person_prefix="ECAF_",
    )
    return row


def evaluate_missing_all(
    engine_name,
    use_tta,
    onnx_path,
    bundle,
    get_engine_fn,
    exclude_known=False,
    caf_split="child_to_adult",
    official_test_only=False,
):
    """
    Evaluates FG-NET, CAF, ECAF, and a combined cross-deduplicated missing->found 1:N pool.
    Cohort identities in lag_cohort_cache.npz are always excluded from CAF/ECAF.
    """
    rows = []
    merged_cache = {}
    merged_quality = {}
    combined_gallery = {}
    combined_probes = []

    if not exclude_known:
        fg_gal, fg_probes = load_fgnet_gallery()
        fg_row, fg_c, fg_q = _evaluate_retrieval_gallery(
            "FG-NET (<=12y -> >12y)",
            "fgnet",
            fg_gal,
            fg_probes,
            engine_name,
            use_tta,
            onnx_path,
            bundle,
            get_engine_fn,
            person_prefix="FGNET_",
        )
        rows.append(fg_row)
        merged_cache.update(fg_c)
        merged_quality.update(fg_q)
        for sid, items in fg_gal.items():
            key = f"fgnet:{sid}"
            combined_gallery[key] = items
        for pr in fg_probes:
            combined_probes.append({**pr, "subject_id": f"fgnet:{pr['subject_id']}"})

    scope = "unseen" if exclude_known else "cohort-disjoint"
    caf_gal, caf_probes = load_caf_gallery(
        exclude_cohort=True,
        exclude_known=exclude_known,
        split=caf_split,
    )
    caf_tag = "<=12y->adult" if caf_split == "child_to_adult" else "oldest-2"
    caf_row, caf_c, caf_q = _evaluate_retrieval_gallery(
        f"CAF ({caf_tag}, {scope})",
        "caf",
        caf_gal,
        caf_probes,
        engine_name,
        use_tta,
        onnx_path,
        bundle,
        get_engine_fn,
        person_prefix="CAF_",
    )
    rows.append(caf_row)
    merged_cache.update(caf_c)
    merged_quality.update(caf_q)
    for sid, items in caf_gal.items():
        combined_gallery[sid] = items
    combined_probes.extend(caf_probes)

    ecaf_gal, ecaf_probes = load_ecaf_gallery(
        exclude_cohort=True,
        exclude_known=exclude_known,
        official_test_only=official_test_only,
    )
    split_tag = "official, " if official_test_only else ""
    ecaf_row, ecaf_c, ecaf_q = _evaluate_retrieval_gallery(
        f"ECAF ({split_tag}{scope})",
        "ecaf",
        ecaf_gal,
        ecaf_probes,
        engine_name,
        use_tta,
        onnx_path,
        bundle,
        get_engine_fn,
        person_prefix="ECAF_",
    )
    rows.append(ecaf_row)
    merged_cache.update(ecaf_c)
    merged_quality.update(ecaf_q)

    # Deduplicate ECAF identities already enrolled from CAF in the combined pool
    caf_norm_set = set(caf_gal.keys())
    for sid, items in ecaf_gal.items():
        if sid not in caf_norm_set:
            combined_gallery[sid] = items
    for pr in ecaf_probes:
        if pr["subject_id"] not in caf_norm_set:
            combined_probes.append(pr)

    pool_label = f"Missing-All Dedup ({scope})"
    print(
        f"\n>> [Missing-All] Combined pool: {len(combined_gallery)} dedup IDs, "
        f"{len(combined_probes)} probes ({scope})..."
    )
    comb_row, _, _ = _evaluate_retrieval_gallery(
        pool_label,
        "missing_all",
        combined_gallery,
        combined_probes,
        engine_name,
        use_tta,
        onnx_path,
        bundle,
        get_engine_fn,
        cache_override=merged_cache,
        quality_override=merged_quality,
    )
    rows.append(comb_row)
    return rows


def print_results(engine_name, onnx_path, use_gpu, use_tta, bundle, verification_rows, retrieval_rows, elapsed_sec):
    """Prints clean plain-language summary tables."""
    mode_str = "Calibrated (4-Layer Bundle)" if bundle is not None else "Cosine Fallback"
    print("\n" + "=" * 106)
    print("  AI-MIDS UNIFIED EVALUATION SUMMARY")
    print("=" * 106)
    print(f"  Engine     : {engine_name} (Flip-TTA: {'ON' if use_tta else 'OFF'}, Device: {'GPU' if use_gpu else 'CPU'})")
    if onnx_path:
        print(f"  Custom ONNX: {onnx_path}")
    print(f"  Scoring    : {mode_str}")
    print(f"  Runtime    : {elapsed_sec:.1f}s ({elapsed_sec / 60:.2f} min)")

    if verification_rows:
        print("\n  [10-Fold Cross-Age Verification]")
        print(
            f"  {'Dataset':<22} | {'Mode':<15} | {'Pairs':>6} | {'10-Fold Acc':>16} | "
            f"{'TAR@FAR=1%':>10} | {'Gen STRONG':>10} | {'Imp >=0.26':>10}"
        )
        print("  " + "-" * 103)
        for r in verification_rows:
            acc_str = f"{r['mean_acc']:.2f}% ± {r['std_acc']:.2f}%"
            print(
                f"  {r['dataset']:<22} | {r['mode']:<15} | {r['evaluated']:>6} | {acc_str:>16} | "
                f"{r['tar_far1']:>9.2f}% | {r['gen_strong']:>9.1f}% | {r['imp_poss_plus']:>9.1f}%"
            )

    if retrieval_rows:
        print("\n  [Child -> Adult Missing-Person Retrieval (FaceStore)]")
        print(
            f"  {'Dataset':<36} | {'Mode':<15} | {'IDs':>5} | {'Probes':>6} | {'Rank-1':>8} | "
            f"{'Top-5':>8} | {'Top-10':>8} | {'STRONG':>8} | {'possible+':>9}"
        )
        print("  " + "-" * 118)
        for row in retrieval_rows:
            print(
                f"  {row['dataset']:<36} | {row['mode']:<15} | {row['gallery_ids']:>5} | {row['probes']:>6} | "
                f"{row['rank1']:>7.2f}% | {row['top5']:>7.2f}% | {row['top10']:>7.2f}% | "
                f"{row['strong']:>7.2f}% | {row['possible_plus']:>8.2f}%"
            )
            if row["skipped"]:
                print(f"    (Skipped {row['skipped']} probes with undetected faces for {row['dataset']})")

    print("=" * 106 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI-MIDS Unified Face Recognition Evaluation CLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["agedb30", "calfw", "fgnet", "caf", "ecaf", "missing_all", "all"],
        help="Benchmark dataset to evaluate (default: all = agedb30+calfw+fgnet)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="aimids_crossage",
        choices=["aimids_crossage", "antelopev2", "aimids_ft", "buffalo_l"],
        help="Face recognition engine (default: aimids_crossage)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Optional explicit .onnx path (for aimids_ft or aimids_crossage)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=600,
        help="Pair limit for AgeDB-30 / CALFW (0 means full protocol; default: 600)",
    )
    parser.add_argument(
        "--gallery",
        type=int,
        default=3,
        help="Registered photos per person on AgeDB-30 (default: 3)",
    )
    parser.add_argument(
        "--disjoint",
        action="store_true",
        help="Exclude AgeDB/CALFW/FG-NET/demo DB overlapping IDs (in addition to LAG cohort) for CAF/ECAF",
    )
    parser.add_argument(
        "--include-known",
        action="store_true",
        help="Keep non-cohort IDs that overlap AgeDB/CALFW/FG-NET (default unless --disjoint is passed)",
    )
    parser.add_argument(
        "--official-test",
        action="store_true",
        help="For ECAF, restrict to official test=True identities in name_list.json",
    )
    parser.add_argument(
        "--caf-split",
        type=str,
        default="child_to_adult",
        choices=["child_to_adult", "oldest"],
        help="CAF retrieval split: child_to_adult (<=12y -> >12y) or oldest (default: child_to_adult)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution instead of CUDA GPU",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic protocol seed (default: 0)",
    )
    args = parser.parse_args()

    if args.disjoint and args.include_known:
        parser.error("Cannot combine --disjoint and --include-known")

    t_start = time.time()
    use_gpu = not args.cpu
    use_tta = args.engine != "buffalo_l"
    pair_limit = None if args.limit <= 0 else int(args.limit)
    gallery_size = max(1, int(args.gallery))
    onnx_path = resolve_custom_onnx(args.onnx)

    if onnx_path and args.engine not in ("aimids_ft", "aimids_crossage"):
        parser.error(f"--onnx is only supported with --engine aimids_ft or aimids_crossage, got '{args.engine}'")

    if use_gpu:
        ensure_cuda_dll_paths()

    bundle = resolve_calibration_bundle(args.engine, onnx_path=onnx_path)

    engine_holder = {"engine": None}

    def get_engine():
        if engine_holder["engine"] is None:
            engine_holder["engine"] = load_engine(
                args.engine, use_gpu=use_gpu, use_tta=use_tta, onnx_path=onnx_path
            )
        return engine_holder["engine"]

    datasets = ["agedb30", "calfw", "fgnet"] if args.dataset == "all" else [args.dataset]
    verification_rows = []
    retrieval_rows = []

    if "agedb30" in datasets:
        verification_rows.append(
            evaluate_agedb30(
                args.engine, use_tta, pair_limit, gallery_size, args.seed, onnx_path, bundle, get_engine
            )
        )
    if "calfw" in datasets:
        verification_rows.append(
            evaluate_calfw(args.engine, use_tta, pair_limit, args.seed, onnx_path, bundle, get_engine)
        )
    if "fgnet" in datasets:
        retrieval_rows.append(evaluate_fgnet(args.engine, use_tta, onnx_path, bundle, get_engine))
    if "caf" in datasets:
        retrieval_rows.append(
            evaluate_caf(
                args.engine,
                use_tta,
                onnx_path,
                bundle,
                get_engine,
                exclude_known=args.disjoint,
                split=args.caf_split,
            )
        )
    if "ecaf" in datasets:
        retrieval_rows.append(
            evaluate_ecaf(
                args.engine,
                use_tta,
                onnx_path,
                bundle,
                get_engine,
                exclude_known=args.disjoint,
                official_test_only=args.official_test,
            )
        )
    if "missing_all" in datasets:
        retrieval_rows.extend(
            evaluate_missing_all(
                args.engine,
                use_tta,
                onnx_path,
                bundle,
                get_engine,
                exclude_known=args.disjoint,
                caf_split=args.caf_split,
                official_test_only=args.official_test,
            )
        )

    elapsed_sec = time.time() - t_start
    actual_gpu = (
        bool(getattr(engine_holder["engine"], "use_gpu", use_gpu))
        if engine_holder["engine"] is not None
        else use_gpu
    )
    print_results(args.engine, onnx_path, actual_gpu, use_tta, bundle, verification_rows, retrieval_rows, elapsed_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
