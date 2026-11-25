"""
test_strict_eval.py - Offline synthetic unittest suite for strict_eval.py.

Runs 100% offline with synthetic embeddings, temp files, fake engines, and
synthetic .mat/.tar/.zip fixtures (no GPU, ONNX, InsightFace, or real datasets).
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

import cv2
import numpy as np
import scipy.io

ARC_DIR = os.path.dirname(os.path.abspath(__file__))
if ARC_DIR not in sys.path:
    sys.path.insert(0, ARC_DIR)

import strict_eval
from calibration import CalibrationBundle, ExternalCohortASNorm, OpenSetCalibrator, QMFCalibrator
from strict_eval import (
    GuardError,
    build_arg_parser,
    build_pairwise_comparisons,
    cluster_bootstrap_ci,
    ensure_strict_embeddings,
    evaluate_retrieval_scope,
    evaluate_verification_pairs,
    load_agedb30_official,
    load_cacd_finetune_identities,
    mcnemar_exact,
    should_skip_caf,
    validate_headline_scope_size,
    verify_cache_integrity,
    wilson_ci,
)


def _unit_vec(arr: np.ndarray) -> np.ndarray:
    v = np.asarray(arr, dtype=np.float32).reshape(-1)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _write_dummy_jpg(path: str, seed_byte: int, size: int = 16) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = np.full((size, size, 3), int(seed_byte) % 256, dtype=np.uint8)
    img[0, 0, 0] = (int(seed_byte) * 37) % 256
    img[-1, -1, 2] = (int(seed_byte) * 73) % 256
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def _make_synthetic_bundle(dim: int = 64, seed: int = 7) -> CalibrationBundle:
    rng = np.random.default_rng(seed)
    cohort = rng.normal(size=(80, dim)).astype(np.float32)
    cohort /= np.linalg.norm(cohort, axis=1, keepdims=True)
    asnorm = ExternalCohortASNorm(cohort_vecs=cohort, top_k=20, std_floor=0.05)

    n_trials = 200
    z_gen = rng.normal(loc=5.5, scale=0.8, size=n_trials)
    z_imp = rng.normal(loc=0.0, scale=0.8, size=n_trials)
    r_gen = rng.uniform(0.55, 0.90, size=n_trials)
    r_imp = rng.uniform(-0.15, 0.25, size=n_trials)
    qmf = QMFCalibrator.fit(
        asnorm_z=np.concatenate([z_gen, z_imp]),
        raw_sim=np.concatenate([r_gen, r_imp]),
        labels=np.concatenate([np.ones(n_trials, dtype=np.int32), np.zeros(n_trials, dtype=np.int32)]),
    )

    n_grid = np.array([5, 25, 100], dtype=np.int64)
    base = rng.normal(loc=-2.5, scale=1.2, size=500).astype(np.float32)
    max_scores = np.stack([
        rng.normal(loc=-1.0, scale=0.8, size=200),
        rng.normal(loc=-0.3, scale=0.8, size=200),
        rng.normal(loc=0.5, scale=0.8, size=200),
    ]).astype(np.float32)
    openset = OpenSetCalibrator.fit(n_grid=n_grid, max_scores_by_n=max_scores, base_nonmate_scores=base)
    return CalibrationBundle(model_name="synthetic", dim=dim, asnorm=asnorm, qmf=qmf, openset=openset)


def _make_separable_verification_data(n_per_fold: int = 100, dim: int = 512, seed: int = 42):
    """Creates 10-fold perfectly separable verification pairs (genuine cos ~ 0.95, imp cos ~ 0.0)."""
    rng = np.random.default_rng(seed)
    pairs = []
    cache = {}
    half = n_per_fold // 2
    for f in range(10):
        for i in range(half):
            base = _unit_vec(rng.normal(size=dim))
            noise = _unit_vec(rng.normal(size=dim))
            # Make genuine partner ~0.95 cosine with base
            partner = _unit_vec(0.95 * base + 0.30 * noise)
            ka = f"g_{f}_{i}_a"
            kb = f"g_{f}_{i}_b"
            cache[ka] = base
            cache[kb] = partner
            pairs.append((ka, kb, 1, f))

        for i in range(half):
            va = _unit_vec(rng.normal(size=dim))
            vb = _unit_vec(rng.normal(size=dim))
            # Orthogonalize vb against va so impostor similarity is near 0.0
            vb = _unit_vec(vb - float(np.dot(va, vb)) * va)
            ka = f"i_{f}_{i}_a"
            kb = f"i_{f}_{i}_b"
            cache[ka] = va
            cache[kb] = vb
            pairs.append((ka, kb, 0, f))
    return pairs, cache


def _build_synthetic_retrieval_files(
    tmp_dir: str,
    n_subjects: int = 25,
    dim: int = 512,
    probes_per_subj: int = 2,
    gal_per_subj: int = 2,
    random_mode: bool = False,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    gallery_by_id = {}
    probes = []
    cache = {}
    quality = {}
    file_idx = 0

    for s in range(n_subjects):
        sid = f"subj_{s:03d}"
        if not random_mode and s < dim:
            # Orthonormal basis vectors + mild perturbation (< 0.97 near-dup threshold!)
            base = np.zeros(dim, dtype=np.float32)
            base[s] = 1.0
        else:
            base = _unit_vec(rng.normal(size=dim))

        gal_items = []
        for g in range(gal_per_subj):
            p = os.path.join(tmp_dir, f"{sid}_gal_{g}.jpg")
            _write_dummy_jpg(p, file_idx)
            file_idx += 1
            if random_mode:
                vec = _unit_vec(rng.normal(size=dim))
            else:
                ortho = _unit_vec(rng.normal(size=dim))
                ortho = _unit_vec(ortho - float(np.dot(ortho, base)) * base)
                vec = _unit_vec(0.92 * base + 0.39 * ortho)
            cache[p] = vec
            quality[p] = 0.9
            gal_items.append((5 + g, p))
        gallery_by_id[sid] = gal_items

        for q in range(probes_per_subj):
            p = os.path.join(tmp_dir, f"{sid}_probe_{q}.jpg")
            _write_dummy_jpg(p, file_idx)
            file_idx += 1
            if random_mode:
                vec = _unit_vec(rng.normal(size=dim))
            else:
                ortho = _unit_vec(rng.normal(size=dim))
                ortho = _unit_vec(ortho - float(np.dot(ortho, base)) * base)
                # cos with gallery ~ 0.92^2 = 0.846 (< 0.97 near-dup threshold, > 0.70 STRONG)
                vec = _unit_vec(0.92 * base + 0.39 * ortho)
            cache[p] = vec
            quality[p] = 0.9
            probes.append({"subject_id": sid, "age": 25 + q, "path": p})

    return gallery_by_id, probes, cache, quality


class TestStrictEval(unittest.TestCase):
    def test_00_pure_import_and_cli_no_limit(self):
        self.assertNotIn("onnxruntime", sys.modules)
        self.assertNotIn("insightface", sys.modules)

        parser = build_arg_parser()
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--limit", "100"])

        self.assertTrue(should_skip_caf("pretrained/aimids_scratch_ir50_crossage.onnx"))
        self.assertTrue(should_skip_caf(r"C:\models\MY_SCRATCH_MODEL.ONNX"))
        self.assertFalse(should_skip_caf("pretrained/aimids_adaface_ir101_ft.onnx"))
        self.assertFalse(should_skip_caf(None))

    def test_01_random_unit_embeddings(self):
        rng = np.random.default_rng(101)
        pairs = []
        cache = {}
        for f in range(10):
            for i in range(50):
                ka, kb = f"rf_{f}_{i}_a", f"rf_{f}_{i}_b"
                cache[ka] = _unit_vec(rng.normal(size=512))
                cache[kb] = _unit_vec(rng.normal(size=512))
                pairs.append((ka, kb, 1, f))
            for i in range(50):
                ka, kb = f"ri_{f}_{i}_a", f"ri_{f}_{i}_b"
                cache[ka] = _unit_vec(rng.normal(size=512))
                cache[kb] = _unit_vec(rng.normal(size=512))
                pairs.append((ka, kb, 0, f))

        ver_res, oof = evaluate_verification_pairs(pairs, cache)
        self.assertEqual(len(oof), 1000)
        self.assertGreaterEqual(ver_res["mean_acc"], 42.0)
        self.assertLessEqual(ver_res["mean_acc"], 58.0)

        with tempfile.TemporaryDirectory() as td:
            gal, probes, r_cache, r_qual = _build_synthetic_retrieval_files(
                td, n_subjects=25, dim=512, random_mode=True, seed=202
            )
            ret_res, _ = evaluate_retrieval_scope(
                gal, probes, r_cache, r_qual, engine_name="test_eng", bootstrap_n=100, seed=0
            )
            self.assertLess(ret_res["modes"]["raw"]["clean"]["rank1"], 20.0)

    def test_02_perfect_separable_embeddings(self):
        pairs, cache = _make_separable_verification_data(n_per_fold=100, dim=512, seed=42)
        ver_res, oof = evaluate_verification_pairs(pairs, cache)
        self.assertEqual(ver_res["mean_acc"], 100.0)
        self.assertEqual(ver_res["pooled_acc"], 100.0)
        self.assertEqual(ver_res["mean_tar_far1"], 100.0)
        self.assertTrue(np.all(oof))

        with tempfile.TemporaryDirectory() as td:
            gal, probes, r_cache, r_qual = _build_synthetic_retrieval_files(
                td, n_subjects=25, dim=512, random_mode=False, seed=7
            )
            ret_res, _ = evaluate_retrieval_scope(
                gal, probes, r_cache, r_qual, engine_name="test_eng", bootstrap_n=200, seed=0
            )
            self.assertEqual(ret_res["modes"]["raw"]["clean"]["rank1"], 100.0)
            self.assertEqual(ret_res["modes"]["raw"]["clean"]["top10"], 100.0)

            # Also verify calibrated FaceStore path works cleanly
            bundle = _make_synthetic_bundle(dim=512, seed=9)
            cal_res, _ = evaluate_retrieval_scope(
                gal, probes, r_cache, r_qual, engine_name="test_eng", bundle=bundle, bootstrap_n=200, seed=0
            )
            self.assertFalse(cal_res["modes"]["calibrated"]["skipped"])
            self.assertEqual(cal_res["modes"]["calibrated"]["clean"]["rank1"], 100.0)

    def test_03_shuffled_labels_collapse(self):
        pairs, cache = _make_separable_verification_data(n_per_fold=100, dim=512, seed=42)
        rng = np.random.default_rng(99)
        labels = [p[2] for p in pairs]
        rng.shuffle(labels)
        shuffled_pairs = [(ka, kb, int(lbl), f) for (ka, kb, _, f), lbl in zip(pairs, labels)]

        ver_res, _ = evaluate_verification_pairs(shuffled_pairs, cache)
        self.assertGreaterEqual(ver_res["mean_acc"], 42.0)
        self.assertLessEqual(ver_res["mean_acc"], 58.0)

    def test_04_out_of_fold_threshold_isolation(self):
        rng = np.random.default_rng(77)
        pairs = []
        cache = {}
        for f in range(10):
            for i in range(60):
                is_same = 1 if i < 30 else 0
                base = _unit_vec(rng.normal(size=512))
                noise = _unit_vec(rng.normal(size=512))
                cos_t = 0.55 if is_same else 0.15
                partner = _unit_vec(cos_t * base + np.sqrt(1.0 - cos_t ** 2) * noise)
                ka, kb = f"oof_{f}_{i}_a", f"oof_{f}_{i}_b"
                cache[ka] = base
                cache[kb] = partner
                pairs.append((ka, kb, is_same, f))

        res_orig, _ = evaluate_verification_pairs(pairs, cache)
        flipped_pairs = [
            (ka, kb, 1 - lbl if f == 0 else lbl, f)
            for (ka, kb, lbl, f) in pairs
        ]
        res_flip, _ = evaluate_verification_pairs(flipped_pairs, cache)

        self.assertEqual(res_orig["tau_acc_per_fold"][0], res_flip["tau_acc_per_fold"][0])
        self.assertEqual(res_orig["tau_far1_per_fold"][0], res_flip["tau_far1_per_fold"][0])
        self.assertNotEqual(res_orig["fold_acc_per_fold"][0], res_flip["fold_acc_per_fold"][0])

    def test_05_failed_detections_stay_in_denominator(self):
        pairs, cache = _make_separable_verification_data(n_per_fold=100, dim=512, seed=42)
        cache_3pct = dict(cache)
        gen_keys = [ka for (ka, _kb, lbl, _f) in pairs if lbl == 1]
        for k in gen_keys[:30]:  # 30 / 1000 = 3.0% <= 5.0%
            cache_3pct[k] = None

        res_3pct, oof = evaluate_verification_pairs(pairs, cache_3pct)
        self.assertEqual(res_3pct["total_pairs"], 1000)
        self.assertEqual(res_3pct["n_failed_pairs"], 30)
        self.assertAlmostEqual(res_3pct["pooled_acc"], 97.0, places=5)
        self.assertEqual(int(np.sum(oof)), 970)

        cache_6pct = dict(cache)
        for k in gen_keys[:60]:  # 60 / 1000 = 6.0% > 5.0%
            cache_6pct[k] = None
        with self.assertRaises(GuardError):
            evaluate_verification_pairs(pairs, cache_6pct)

        with tempfile.TemporaryDirectory() as td:
            gal, probes, r_cache, r_qual = _build_synthetic_retrieval_files(
                td, n_subjects=25, dim=512, probes_per_subj=2, gal_per_subj=2, seed=11
            )
            # Fail 5 probe detections + fail ALL gallery photos of 1 subject (2 probes unenrolled)
            for pr in probes[:5]:
                r_cache[pr["path"]] = None
            for _, gpath in gal["subj_010"]:
                r_cache[gpath] = None

            ret_res, _ = evaluate_retrieval_scope(
                gal, probes, r_cache, r_qual, engine_name="test_eng", bootstrap_n=100, seed=0
            )
            clean = ret_res["modes"]["raw"]["clean"]
            self.assertEqual(clean["gallery_ids_total"], 25)
            self.assertEqual(clean["gallery_ids_enrolled"], 24)
            self.assertEqual(clean["probes"], 50)
            self.assertEqual(clean["n_failed_probe_detect"], 5)
            # 5 failed probes + 2 probes on unenrolled subj_010 = 7 guaranteed misses -> 43/50 = 86%
            self.assertAlmostEqual(clean["rank1"], 86.0, places=5)

    def test_06_path_and_sha256_duplicate_guards(self):
        with tempfile.TemporaryDirectory() as td:
            gal, probes, r_cache, r_qual = _build_synthetic_retrieval_files(
                td, n_subjects=10, dim=512, probes_per_subj=2, gal_per_subj=2, seed=17
            )
            # A) Probe path inside gallery -> GuardError
            bad_probes = list(probes) + [{"subject_id": "subj_000", "age": 30, "path": gal["subj_000"][0][1]}]
            with self.assertRaises(GuardError):
                evaluate_retrieval_scope(
                    gal, bad_probes, r_cache, r_qual, engine_name="test_eng", bootstrap_n=50
                )

            # B) Same-ID byte-identical probe at distinct path -> excluded (n_exact_dup_excluded == 1)
            dup_same_path = os.path.join(td, "subj_000_exact_dup.jpg")
            with open(gal["subj_000"][0][1], "rb") as src, open(dup_same_path, "wb") as dst:
                dst.write(src.read())
            r_cache[dup_same_path] = r_cache[gal["subj_000"][0][1]]
            r_qual[dup_same_path] = 0.9

            # C) Cross-ID byte-identical file -> recorded in cross_id_byte_collisions
            cross_path = os.path.join(td, "subj_001_cross_dup.jpg")
            with open(gal["subj_002"][0][1], "rb") as src, open(cross_path, "wb") as dst:
                dst.write(src.read())
            r_cache[cross_path] = r_cache[gal["subj_002"][0][1]]
            r_qual[cross_path] = 0.9

            aug_probes = list(probes) + [
                {"subject_id": "subj_000", "age": 31, "path": dup_same_path},
                {"subject_id": "subj_001", "age": 31, "path": cross_path},
            ]
            ret_res, probe_map = evaluate_retrieval_scope(
                gal, aug_probes, r_cache, r_qual, engine_name="test_eng", bootstrap_n=50
            )
            self.assertEqual(ret_res["n_exact_dup_excluded"], 1)
            self.assertEqual(ret_res["modes"]["raw"]["clean"]["n_exact_dup_excluded"], 1)
            self.assertEqual(ret_res["modes"]["raw"]["clean"]["probes"], len(probes) + 1)
            self.assertGreaterEqual(ret_res["cross_id_byte_collisions"]["count"], 1)
            collided_ids = set(ret_res["cross_id_byte_collisions"]["samples"][0]["subject_ids"])
            self.assertTrue({"subj_001", "subj_002"}.issubset(collided_ids))
            dup_key = strict_eval._make_probe_key("subj_000", dup_same_path)
            self.assertTrue(probe_map["raw"][dup_key]["is_exact_dup"])

    def test_07_near_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as td:
            gal, probes, r_cache, r_qual = _build_synthetic_retrieval_files(
                td, n_subjects=15, dim=512, probes_per_subj=2, gal_per_subj=2, seed=21
            )
            near_path = os.path.join(td, "subj_000_near_dup.jpg")
            _write_dummy_jpg(near_path, 250)  # distinct bytes -> not exact_dup
            gal_vec = r_cache[gal["subj_000"][0][1]]
            ortho = _unit_vec(np.random.default_rng(5).normal(size=512))
            ortho = _unit_vec(ortho - float(np.dot(ortho, gal_vec)) * gal_vec)
            # Cosine >= 0.995 with gallery vector -> triggers >= 0.97 near-dup guard
            near_vec = _unit_vec(0.996 * gal_vec + np.sqrt(1.0 - 0.996 ** 2) * ortho)
            r_cache[near_path] = near_vec
            r_qual[near_path] = 0.9

            aug_probes = list(probes) + [{"subject_id": "subj_000", "age": 29, "path": near_path}]
            ret_res, probe_map = evaluate_retrieval_scope(
                gal, aug_probes, r_cache, r_qual, engine_name="test_eng", bootstrap_n=50
            )
            self.assertEqual(ret_res["n_near_dup_excluded"], 1)
            clean_v = ret_res["modes"]["raw"]["clean"]
            near_v = ret_res["modes"]["raw"]["with_near_duplicates"]
            self.assertEqual(clean_v["n_near_dup_excluded"], 1)
            self.assertEqual(clean_v["probes"], len(probes))
            self.assertEqual(near_v["probes"], len(probes) + 1)
            near_key = strict_eval._make_probe_key("subj_000", near_path)
            self.assertTrue(probe_map["raw"][near_key]["is_near_dup"])

    def test_08_cache_integrity_guard(self):
        with tempfile.TemporaryDirectory() as td:
            paths = []
            cache = {}
            rng = np.random.default_rng(31)
            by_pixel = {}
            for i in range(10):
                p = os.path.join(td, f"img_{i}.jpg")
                _write_dummy_jpg(p, i * 19 + 3)
                paths.append(p)
                vec = _unit_vec(rng.normal(size=64))
                cache[p] = vec
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                by_pixel[img.tobytes()] = vec

            class _FakeEngine:
                use_gpu = False

                def embed(self, img_bgr):
                    v = by_pixel.get(img_bgr.tobytes())
                    return None if v is None else {"embedding": v.copy(), "det_score": 0.99}

            res_ok = verify_cache_integrity(paths, cache, lambda: _FakeEngine(), verify_sample=10, seed=0)
            self.assertEqual(res_ok["n_checked"], 10)
            self.assertGreaterEqual(res_ok["min_cos"], 0.999)
            self.assertEqual(res_ok["offending"], [])

            # Perturb one cached vector so cosine < 0.995
            bad_cache = dict(cache)
            orig = bad_cache[paths[3]]
            ortho = _unit_vec(rng.normal(size=64))
            ortho = _unit_vec(ortho - float(np.dot(ortho, orig)) * orig)
            bad_cache[paths[3]] = _unit_vec(0.90 * orig + np.sqrt(1.0 - 0.90 ** 2) * ortho)
            with self.assertRaises(GuardError):
                verify_cache_integrity(paths, bad_cache, lambda: _FakeEngine(), verify_sample=10, seed=0)

            # None live detection also raises GuardError
            class _NoneEngine:
                use_gpu = False

                def embed(self, img_bgr):
                    return None

            with self.assertRaises(GuardError):
                verify_cache_integrity(paths, cache, lambda: _NoneEngine(), verify_sample=5, seed=0)

    def test_09_cluster_bootstrap_ci(self):
        def _build_clusters(n_subj: int):
            rng = np.random.default_rng(555)
            sids, h1, h10 = [], [], []
            for s in range(n_subj):
                sid = f"id_{s}"
                subj_p = 0.72 if (s % 2 == 0) else 0.58
                for _ in range(3):
                    hit1 = bool(rng.random() < subj_p)
                    hit10 = hit1 or bool(rng.random() < 0.5)
                    sids.append(sid)
                    h1.append(hit1)
                    h10.append(hit10)
            return sids, h1, h10

        s20, h1_20, h10_20 = _build_clusters(20)
        s200, h1_200, h10_200 = _build_clusters(200)

        r1_ci_20, t10_ci_20 = cluster_bootstrap_ci(s20, h1_20, h10_20, n_boot=2000, seed=0)
        r1_ci_200, t10_ci_200 = cluster_bootstrap_ci(s200, h1_200, h10_200, n_boot=2000, seed=0)

        pt_r1_20 = float(np.mean(h1_20)) * 100.0
        pt_t10_20 = float(np.mean(h10_20)) * 100.0
        pt_r1_200 = float(np.mean(h1_200)) * 100.0
        pt_t10_200 = float(np.mean(h10_200)) * 100.0

        self.assertLessEqual(r1_ci_20[0], pt_r1_20)
        self.assertGreaterEqual(r1_ci_20[1], pt_r1_20)
        self.assertLessEqual(t10_ci_20[0], pt_t10_20)
        self.assertGreaterEqual(t10_ci_20[1], pt_t10_20)
        self.assertLessEqual(r1_ci_200[0], pt_r1_200)
        self.assertGreaterEqual(r1_ci_200[1], pt_r1_200)
        self.assertLessEqual(t10_ci_200[0], pt_t10_200)
        self.assertGreaterEqual(t10_ci_200[1], pt_t10_200)

        w20 = r1_ci_20[1] - r1_ci_20[0]
        w200 = r1_ci_200[1] - r1_ci_200[0]
        self.assertLess(w200, w20)

        # Verify Wilson CI bounds sanity as well
        lo, hi = wilson_ci(70, 100)
        self.assertTrue(0.0 < lo < 0.70 < hi < 1.0)

    def test_10_exact_mcnemar_and_alignment(self):
        same = [True, False, True, True, False] * 10
        res_same = mcnemar_exact(same, same)
        self.assertEqual(res_same["b"], 0)
        self.assertEqual(res_same["c"], 0)
        self.assertEqual(res_same["p_value"], 1.0)
        self.assertFalse(res_same["significant_p05"])

        # b = 25 (A only), c = 2 (B only), 73 agreed True
        a = [True] * 98 + [False] * 2
        b = [True] * 73 + [False] * 25 + [True] * 2
        res_diff = mcnemar_exact(a, b)
        self.assertEqual(res_diff["b"], 25)
        self.assertEqual(res_diff["c"], 2)
        self.assertLess(res_diff["p_value"], 0.01)
        self.assertTrue(res_diff["significant_p05"])
        self.assertGreater(res_diff["diff_pct"], 0.0)

        # Pairwise comparison excludes probes flagged near_dup by EITHER engine
        artifacts = {
            "eng_a": {
                "verification_oof": {"agedb30_official": np.array(a, dtype=bool)},
                "retrieval_probes": {
                    ("fgnet", "default"): {
                        "raw": {
                            "k1": {"hit_1": True, "is_exact_dup": False, "is_near_dup": False},
                            "k2": {"hit_1": True, "is_exact_dup": False, "is_near_dup": True},  # near-dup in A
                            "k3": {"hit_1": False, "is_exact_dup": False, "is_near_dup": False},
                        }
                    }
                },
            },
            "eng_b": {
                "verification_oof": {"agedb30_official": np.array(b, dtype=bool)},
                "retrieval_probes": {
                    ("fgnet", "default"): {
                        "raw": {
                            "k1": {"hit_1": True, "is_exact_dup": False, "is_near_dup": False},
                            "k2": {"hit_1": False, "is_exact_dup": False, "is_near_dup": False},
                            "k3": {"hit_1": True, "is_exact_dup": False, "is_near_dup": False},
                        }
                    }
                },
            },
        }
        cmp_out = build_pairwise_comparisons(["eng_a", "eng_b"], artifacts)
        fg_raw = cmp_out["eng_a_vs_eng_b"]["retrieval"]["fgnet:default"]["raw"]
        self.assertEqual(fg_raw["n_shared_total"], 3)
        self.assertEqual(fg_raw["n"], 2)
        self.assertEqual(fg_raw["n_excluded_dup_either"], 1)

    def test_11_synthetic_agedb_tar_cacd_zip_and_headline_guards(self):
        with tempfile.TemporaryDirectory() as td:
            # Build synthetic 6,000-pair .mat + .tar containing .jpg + decoy .pts
            tar_path = os.path.join(td, "03_Protocol_Images.tar")
            mat_path = os.path.join(td, "04_FINAL_protocol_30_years.mat")

            stems = [f"0_Person{i}_{i}_A_30" for i in range(40)]
            by_pixel = {}
            with tarfile.open(tar_path, "w:") as tf:
                for idx, st in enumerate(stems):
                    img_p = os.path.join(td, f"{st}.jpg")
                    _write_dummy_jpg(img_p, idx * 5 + 1)
                    img = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), cv2.IMREAD_COLOR)
                    by_pixel[img.tobytes()] = _unit_vec(np.random.default_rng(idx + 1).normal(size=32))
                    tf.add(img_p, arcname=f"03_Protocol_Images/{st}.jpg")

                    pts_p = os.path.join(td, f"{st}.pts")
                    with open(pts_p, "w", encoding="utf-8") as f:
                        f.write("version: 1\nn_points: 5\n")
                    tf.add(pts_p, arcname=f"03_Protocol_Images/{st}.pts")

            splits = np.empty((10, 1), dtype=object)
            for fold_idx in range(10):
                mat_pairs = np.empty((2, 600), dtype=object)
                annot = np.empty((1, 600), dtype=np.int16)
                for i in range(600):
                    s1 = stems[(fold_idx * 3 + i) % len(stems)]
                    s2 = stems[(fold_idx * 3 + i + 1) % len(stems)]
                    mat_pairs[0, i] = np.array([(s1,)], dtype=[("name", "O")])
                    mat_pairs[1, i] = np.array([(s2,)], dtype=[("name", "O")])
                    annot[0, i] = 1 if i < 300 else -1
                splits[fold_idx, 0] = np.array(
                    [(mat_pairs, annot)], dtype=[("pairs", "O"), ("annot", "O")]
                )
            scipy.io.savemat(mat_path, {"splits": splits})

            pairs, tar_map, meta = load_agedb30_official(mat_path=mat_path, tar_path=tar_path)
            self.assertEqual(len(pairs), 6000)
            self.assertEqual(meta["tar_image_count"], 40)
            self.assertEqual(meta["tar_non_image_ignored"], 40)

            # Embed and verify live against agedbtar://
            class _FakeTarEngine:
                use_gpu = False

                def embed(self, img_bgr):
                    v = by_pixel[img_bgr.tobytes()]
                    return {"embedding": v.copy(), "det_score": 0.95}

            cache_path = os.path.join(td, "strict_cache_agedb30_official_test.npz")
            needed = {k for a, b, _, _ in pairs for k in (a, b)}
            with tarfile.open(tar_path, "r:") as tf:
                cache, _ = ensure_strict_embeddings(
                    needed, cache_path, lambda: _FakeTarEngine(), tar_handle=tf, tar_map=tar_map
                )
                chk = verify_cache_integrity(
                    needed, cache, lambda: _FakeTarEngine(), verify_sample=16, seed=0, tar_handle=tf, tar_map=tar_map
                )
                self.assertEqual(chk["n_checked"], 16)
                self.assertGreaterEqual(chk["min_cos"], 0.999)

            # Corrupt .mat fold balance -> GuardError
            bad_mat = os.path.join(td, "bad.mat")
            splits_bad = splits.copy()
            bad_pairs = splits_bad[0, 0]["pairs"][0]
            splits_bad[0, 0] = np.array(
                [(bad_pairs, np.ones((1, 600), dtype=np.int16))], dtype=[("pairs", "O"), ("annot", "O")]
            )
            scipy.io.savemat(bad_mat, {"splits": splits_bad})
            with self.assertRaises(GuardError):
                load_agedb30_official(mat_path=bad_mat, tar_path=tar_path)

            # Missing stem from tar -> GuardError
            bad_tar = os.path.join(td, "missing_stem.tar")
            with tarfile.open(bad_tar, "w:") as tf:
                for st in stems[:-1]:
                    tf.add(os.path.join(td, f"{st}.jpg"), arcname=f"03_Protocol_Images/{st}.jpg")
            with self.assertRaises(GuardError):
                load_agedb30_official(mat_path=mat_path, tar_path=bad_tar)

            # Synthetic CACD manifest zip check + unexpected split guard
            cacd_zip = os.path.join(td, "cacd.zip")
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                tw = io.TextIOWrapper(gz, encoding="utf-8", newline="")
                writer = csv.DictWriter(tw, fieldnames=["dataset", "identity", "split"])
                writer.writeheader()
                writer.writerow({"dataset": "CACD", "identity": "Alice Smith", "split": "train"})
                writer.writerow({"dataset": "CACD", "identity": "Bob Jones", "split": "validation"})
                writer.writerow({"dataset": "HDA", "identity": "Ignored", "split": "test"})
                tw.flush()
            with zipfile.ZipFile(cacd_zip, "w") as zf:
                zf.writestr("custom_model/split_manifest_v2.csv.gz", buf.getvalue())

            cacd_ids, c_meta = load_cacd_finetune_identities(zip_path=cacd_zip)
            self.assertEqual(cacd_ids, {"alicesmith", "bobjones"})
            self.assertEqual(c_meta["splits_seen"], ["train", "validation"])

            bad_zip = os.path.join(td, "cacd_bad.zip")
            buf2 = io.BytesIO()
            with gzip.GzipFile(fileobj=buf2, mode="wb") as gz:
                tw = io.TextIOWrapper(gz, encoding="utf-8", newline="")
                writer = csv.DictWriter(tw, fieldnames=["dataset", "identity", "split"])
                writer.writeheader()
                writer.writerow({"dataset": "CACD", "identity": "Leak Person", "split": "test"})
                tw.flush()
            with zipfile.ZipFile(bad_zip, "w") as zf:
                zf.writestr("custom_model/split_manifest_v2.csv.gz", buf2.getvalue())
            with self.assertRaises(GuardError):
                load_cacd_finetune_identities(zip_path=bad_zip)

            # Headline minimum ID guard (< 50 -> GuardError)
            with self.assertRaises(GuardError):
                validate_headline_scope_size("caf:fully_unseen", {f"id{i}": [] for i in range(49)})

            # main() returns exit code 2 on GuardError
            with mock.patch.object(strict_eval, "load_agedb30_official", side_effect=GuardError("test guard")):
                rc = strict_eval.main(["--engines", "antelopev2", "--suites", "agedb30_official", "--out", td])
                self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
