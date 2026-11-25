"""
test_calibration.py - Comprehensive unit and integration tests for AI-MIDS 4-layer calibration.

Tests:
  1. ExternalCohortASNorm (exact O(MD+N) equivalence vs brute-force, N=1 support, top_k cap)
  2. QMFCalibrator (monotonicity, C_llr vs PAV C_llr_min, missing age_gap handling, Brier/ECE)
  3. OpenSetCalibrator (GPD bounded-tail clamp xi<0, |xi|<1e-6 exp limit, N-scaling monotonicity)
  4. CoveCalibrator & compute_bg_zscore (binom(m,2) conformal bound, duplicate immunity s_{3..50})
  5. CalibrationBundle save/load + hash/schema checks (+ disk bundles if built)
  6. FaceStore.search() automatic calibrated search vs cosine fallback for uncalibrated models
     (including N<80, single-photo hub suppression, add_photo/delete_person cache updates)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import tempfile
import unittest
import numpy as np

ARC_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(ARC_DIR)
if ARC_DIR not in sys.path:
    sys.path.insert(0, ARC_DIR)

from calibration import (
    SCHEMA_VERSION,
    CalibrationBundle,
    CoveCalibrator,
    ExternalCohortASNorm,
    OpenSetCalibrator,
    QMFCalibrator,
    compute_age_subspace_basis,
    compute_bg_zscore,
    compute_brier,
    compute_cllr,
    compute_ece,
    compute_proj_hash,
    fit_qmf,
    pav_cllr_min,
    resolve_bundle_path,
)
from face_engine_custom import compute_asnorm_scores
from store import FaceStore


def _make_unit_vecs(rng, n, d):
    arr = rng.normal(size=(n, d)).astype(np.float32)
    arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    return arr


def _make_synthetic_bundle(model_name="synthetic_512", dim=512, seed=7):
    rng = np.random.default_rng(seed)
    cohort = _make_unit_vecs(rng, 250, dim)
    asnorm = ExternalCohortASNorm(cohort, top_k=200, std_floor=0.05)

    # Synthetic LLR calibration trials
    z_gen = rng.normal(loc=5.5, scale=1.5, size=300)
    z_imp = rng.normal(loc=0.0, scale=1.0, size=300)
    r_gen = np.clip(0.45 + 0.05 * (z_gen - 5.5), -1.0, 1.0)
    r_imp = np.clip(0.05 + 0.05 * z_imp, -1.0, 1.0)
    z_all = np.concatenate([z_gen, z_imp])
    r_all = np.concatenate([r_gen, r_imp])
    y_all = np.array([1] * 300 + [0] * 300, dtype=np.int32)
    q_all = rng.uniform(0.5, 1.0, size=600)
    qmf = fit_qmf(z_all, r_all, y_all, q_min=q_all, l2_reg=1e-2)

    # Synthetic OpenSet max-of-N curves (N = 1, 20, 80, 300)
    base = rng.normal(loc=-3.0, scale=1.5, size=2000).astype(np.float32)
    n_grid = [1, 20, 80, 300]
    max_rows = []
    for n in n_grid:
        draws = rng.choice(base, size=(400, n), replace=True)
        max_rows.append(np.max(draws, axis=1))
    openset = OpenSetCalibrator.fit(n_grid, max_rows, base, tail_quantile=0.90)

    cove_vecs = _make_unit_vecs(rng, 30, dim)
    cove = CoveCalibrator.fit(cove_vecs, [f"id_{i}" for i in range(30)])
    return CalibrationBundle(model_name=model_name, dim=dim, asnorm=asnorm, qmf=qmf, openset=openset, cove=cove)


class TestCalibrationLayers(unittest.TestCase):
    def test_external_cohort_asnorm_equivalence_and_topk_cap(self):
        rng = np.random.default_rng(11)
        cohort = _make_unit_vecs(rng, 120, 64)  # < 200 -> top_k capped cleanly to 120
        asnorm = ExternalCohortASNorm(cohort, top_k=200, std_floor=0.05)
        self.assertEqual(asnorm.top_k, 120)

        cands = _make_unit_vecs(rng, 45, 64)
        q = _make_unit_vecs(rng, 1, 64)[0]

        mu_e, sig_e = asnorm.enroll_stats(cands, batch_size=16)
        p_stats = asnorm.probe_stats(q)
        raw_sims = cands @ q
        z_fast = asnorm.score_batch(q, cand_mu=mu_e, cand_sigma=sig_e, raw_sims=raw_sims, probe_stats=p_stats)

        z_ref = compute_asnorm_scores(
            q, cands, cohort, candidate_raw_sims=raw_sims, top_k=200, std_floor=0.05, normalized_inputs=True
        )
        np.testing.assert_allclose(z_fast, z_ref, atol=1e-5, rtol=1e-5)

        # Also works on N=1 directly
        z_one = asnorm.score_batch(q, cand_vecs=cands[:1])
        np.testing.assert_allclose(z_one, z_ref[:1], atol=1e-5)

    def test_qmf_calibrator_cllr_pav_and_missing_age(self):
        rng = np.random.default_rng(22)
        n = 500
        y = np.array([1] * 250 + [0] * 250, dtype=np.int32)
        z = np.concatenate([rng.normal(4.2, 1.3, 250), rng.normal(0.0, 1.0, 250)])
        r = np.clip(0.15 + 0.08 * z + rng.normal(0, 0.03, n), -1.0, 1.0)
        q = rng.uniform(0.4, 1.0, n)
        age_gap = np.concatenate([rng.uniform(5, 45, 300), np.full(200, np.nan)])

        qmf = fit_qmf(z, r, y, q_min=q, age_gap=age_gap)
        self.assertGreater(qmf.weights[0], 0.0)
        self.assertGreaterEqual(qmf.weights[1], 0.0)

        llrs = qmf.predict_llr(z, r, q_min=q, age_gap=age_gap)
        probs = qmf.predict_prob(llrs, pi=0.5)
        cllr = compute_cllr(llrs, y)
        cllr_min = pav_cllr_min(llrs, y)
        self.assertLess(cllr, 0.50)
        self.assertLessEqual(cllr_min, cllr + 1e-9)
        self.assertLess(compute_brier(probs, y), 0.15)
        self.assertLess(compute_ece(probs, y), 0.15)

        # Missing age_gap produces exact marginalized prediction (has_age_gap=0 -> 0 contribution)
        llr_none = qmf.predict_llr(3.0, 0.35, q_min=0.8, age_gap=None)
        llr_nan = qmf.predict_llr(3.0, 0.35, q_min=0.8, age_gap=np.nan)
        self.assertAlmostEqual(llr_none, llr_nan, places=7)

        # Monotonicity in score & cosine range
        c_lo = qmf.llr_to_cosine(-5.0)
        c_hi = qmf.llr_to_cosine(+8.0)
        self.assertTrue(-1.0 <= c_lo < c_hi <= 1.0)
        self.assertLess(qmf.predict_prob(2.0, pi=0.05), qmf.predict_prob(2.0, pi=0.5))

    def test_openset_calibrator_gpd_clamp_and_n_monotonicity(self):
        # 1. Bounded tail clamp (xi < 0) and exponential limit (|xi| < 1e-6)
        n_grid = np.array([50, 300, 1000], dtype=np.int64)
        base = np.linspace(-2.0, 1.0, 500, dtype=np.float32)
        ms = np.stack([base[-200:] + 0.1 * i for i in range(3)])
        os_neg_xi = OpenSetCalibrator(n_grid, ms, u=0.5, p_u=0.10, xi=-0.25, sigma_u=0.40, base_scores=base)
        # Upper endpoint for xi=-0.25, u=0.5, sigma_u=0.4 is 0.5 - 0.4/(-0.25) = 2.1
        p_above = os_neg_xi._s1_tail(5.0)
        self.assertEqual(p_above, 0.0)
        self.assertEqual(os_neg_xi.p_nonmate(5.0, 5000), 0.0)

        os_exp = OpenSetCalibrator(n_grid, ms, u=0.5, p_u=0.10, xi=1e-9, sigma_u=0.50, base_scores=base)
        self.assertAlmostEqual(os_exp._s1_tail(1.5), 0.10 * np.exp(-2.0), places=6)

        # 2. Monotonicity across N and score on real fitted bundle
        bundle = _make_synthetic_bundle()
        os_cal = bundle.openset
        prev_s = -1e9
        prev_p = -1e9
        for n_val in (1, 10, 50, 80, 300, 1000, 10000, 100000):
            ts, tp = os_cal.tier_thresholds(n_val, alpha_strong=0.01, alpha_possible=0.08)
            self.assertGreaterEqual(ts, tp)
            self.assertGreaterEqual(ts, prev_s - 1e-9)
            self.assertGreaterEqual(tp, prev_p - 1e-9)
            self.assertLessEqual(os_cal.p_nonmate(ts, n_val), 0.01 + 1e-6)
            self.assertEqual(os_cal.classify(ts + 0.1, n_val), "STRONG")
            self.assertEqual(os_cal.classify(tp - 0.5, n_val), "weak")
            prev_s, prev_p = ts, tp

        # Higher N -> higher p_nonmate; higher score -> lower p_nonmate
        self.assertGreaterEqual(os_cal.p_nonmate(1.5, 5000), os_cal.p_nonmate(1.5, 50))
        self.assertGreaterEqual(os_cal.p_nonmate(0.5, 300), os_cal.p_nonmate(4.0, 300))

    def test_cove_and_bg_zscore_duplicate_immunity(self):
        rng = np.random.default_rng(33)
        embs = _make_unit_vecs(rng, 40, 128)
        cove = CoveCalibrator.fit(embs, [f"person_{i}" for i in range(40)])
        self.assertEqual(cove.n_pairs, 40 * 39 // 2)
        tau = cove.threshold(alpha=0.01)
        self.assertLessEqual(cove.p_value(tau), 0.01)
        self.assertEqual(cove.verify(tau + 0.1, alpha_accept=0.01)["decision"], "ACCEPT")

        # Duplicate immunity: s1=10.0, s2=9.6 (duplicate!), s_{3..50} around 0.0 +/- 0.5
        bg = rng.normal(0.0, 0.5, size=48).tolist()
        scores_dup = [10.0, 9.6] + sorted(bg, reverse=True)
        res_dup = compute_bg_zscore(scores_dup)
        self.assertAlmostEqual(res_dup["margin_12"], 0.4, places=5)
        self.assertGreater(res_dup["z_bg"], 10.0)  # s_{3..50} preserves high z_bg!
        self.assertIn("standout", res_dup["advisory_flags"])
        self.assertIn("ambiguous_or_duplicate", res_dup["advisory_flags"])

        # Pure standout (no rank-2 duplicate)
        scores_clean = [10.0, 0.4] + sorted(bg, reverse=True)
        res_clean = compute_bg_zscore(scores_clean)
        self.assertIn("standout", res_clean["advisory_flags"])
        self.assertNotIn("ambiguous_or_duplicate", res_clean["advisory_flags"])

        # Fallback for < 4 items and <= 1 item
        res_2 = compute_bg_zscore([5.0, 1.0])
        self.assertIsNotNone(res_2["z_bg"])
        res_1 = compute_bg_zscore([5.0])
        self.assertIsNone(res_1["z_bg"])

    def test_bundle_roundtrip_and_disk_bundles(self):
        bundle = _make_synthetic_bundle()
        with tempfile.TemporaryDirectory() as td:
            npz_p = os.path.join(td, "calib_synthetic_512.npz")
            npz_out, json_out = bundle.save(npz_p)
            self.assertTrue(os.path.isfile(npz_out))
            self.assertTrue(os.path.isfile(json_out))
            loaded = CalibrationBundle.load(npz_out)
            self.assertEqual(loaded.model_name, bundle.model_name)
            self.assertEqual(loaded.dim, bundle.dim)
            self.assertEqual(loaded.cohort_hash, bundle.cohort_hash)
            np.testing.assert_allclose(loaded.qmf.weights, bundle.qmf.weights)
            self.assertEqual(
                loaded.openset.tier_thresholds(300),
                bundle.openset.tier_thresholds(300),
            )

        # Verify real saved bundles if calibrated
        for m in ("antelopev2", "aimids_crossage"):
            bp = resolve_bundle_path(m)
            if os.path.isfile(bp):
                b = CalibrationBundle.load(bp)
                self.assertEqual(b.model_name, m)
                t_lo = b.openset.tier_thresholds(80)
                t_hi = b.openset.tier_thresholds(3935)
                self.assertGreaterEqual(t_hi[0], t_lo[0])

    def test_store_calibrated_search(self):
        rng = np.random.default_rng(55)
        dim = 512
        bundle = _make_synthetic_bundle(model_name="antelopev2", dim=dim, seed=55)

        store = FaceStore(":memory:")
        store.set_calibration_bundle("antelopev2", bundle)

        # Build < 80 person gallery first (40 people)
        p_vecs = _make_unit_vecs(rng, 40, dim)
        pids = []
        for i in range(40):
            pid = store.add_person(f"Person_{i}")
            pids.append(pid)
            store.add_photo(pid, p_vecs[i], model="antelopev2", quality=0.85)

        q = p_vecs[0] + 0.2 * _make_unit_vecs(rng, 1, dim)[0]
        q /= np.linalg.norm(q)

        # Uncalibrated model name (no bundle set or on disk) runs cosine fallback cleanly
        self.assertFalse(store.calibrated_ready("uncalibrated_test_model"))
        uncal_pids = []
        for i in range(5):
            pid_u = store.add_person(f"Uncal_{i}")
            uncal_pids.append(pid_u)
            store.add_photo(pid_u, p_vecs[i], model="uncalibrated_test_model", quality=0.85)
        fb_hits = store.search(q, k=5, model="uncalibrated_test_model")
        self.assertEqual(fb_hits[0]["person_id"], uncal_pids[0])
        self.assertIn(fb_hits[0]["match_source"], ("centroid", "photo"))
        self.assertIn(fb_hits[0]["tier"], ("STRONG", "possible", "weak"))
        self.assertNotIn("calibrated_llr", fb_hits[0])
        self.assertNotIn("p_nonmate", fb_hits[0])

        # Calibrated mode runs automatically at N=40 (< 80) and emits full calibrated payload
        self.assertTrue(store.calibrated_ready("antelopev2"))
        cal_hits = store.search(q, k=5, model="antelopev2", probe_quality=0.9)
        self.assertEqual(cal_hits[0]["person_id"], pids[0])
        self.assertIn("calibrated_llr", cal_hits[0])
        self.assertIn("match_prob", cal_hits[0])
        self.assertIn("p_nonmate", cal_hits[0])
        self.assertIn("z_bg", cal_hits[0])
        self.assertIn("advisory_flags", cal_hits[0])
        self.assertIn(cal_hits[0]["tier"], ("STRONG", "possible", "weak"))
        self.assertIn("raw_centroid_similarity", cal_hits[0])
        self.assertIn("raw_best_photo_similarity", cal_hits[0])

        # Expand past >= 80 persons (total 90) and verify hub suppression & cache updates on add/delete
        # Create a clustered cohort hub direction (high similarity across all top-200 cohort members)
        # vs an orthogonal direction -> AS-Norm must rank clean > hub when raw cosine is equal!
        base_coh = _make_unit_vecs(rng, 1, dim)[0]
        clustered_coh = 0.75 * base_coh[None, :] + 0.35 * _make_unit_vecs(rng, 250, dim)
        clustered_coh /= np.linalg.norm(clustered_coh, axis=1, keepdims=True)
        hub_bundle = CalibrationBundle(
            model_name="antelopev2",
            dim=dim,
            asnorm=ExternalCohortASNorm(clustered_coh, top_k=200, std_floor=0.05),
            qmf=bundle.qmf,
            openset=bundle.openset,
            cove=bundle.cove,
        )
        store.set_calibration_bundle("antelopev2", hub_bundle)

        hub_dir = base_coh
        orth_dir = _make_unit_vecs(rng, 1, dim)[0]
        orth_dir = orth_dir - float(np.dot(orth_dir, hub_dir)) * hub_dir
        orth_dir /= np.linalg.norm(orth_dir)

        pid_hub = store.add_person("Hub_Single_Photo")
        store.add_photo(pid_hub, hub_dir, model="antelopev2", quality=0.9)
        pid_clean = store.add_person("Clean_Single_Photo")
        store.add_photo(pid_clean, orth_dir, model="antelopev2", quality=0.9)

        # Add 49 more fillers -> 91 total people for antelopev2
        for j in range(49):
            pid_f = store.add_person(f"Filler_{j}")
            store.add_photo(pid_f, _make_unit_vecs(rng, 1, dim)[0], model="antelopev2", quality=0.9)

        # Query equidistant in raw cosine from hub_dir and orth_dir
        q_eq = 0.5 * hub_dir + 0.5 * orth_dir
        q_eq /= np.linalg.norm(q_eq)

        cal_eq = store.search(q_eq, k=10, model="antelopev2", probe_quality=0.9)
        rank_clean = next(i for i, h in enumerate(cal_eq) if h["person_id"] == pid_clean)
        rank_hub = next((i for i, h in enumerate(cal_eq) if h["person_id"] == pid_hub), 999)
        # Symmetric external AS-Norm suppresses 1-photo hub below clean candidate (no best_photo_sim loophole!)
        self.assertLess(rank_clean, rank_hub)

        # Delete person cleans stats cleanly
        store.delete_person(pid_hub)
        cal_after_del = store.search(q_eq, k=10, model="antelopev2")
        self.assertFalse(any(h["person_id"] == pid_hub for h in cal_after_del))
        store.db.close()

    def test_aimids_ft_onnx_path_bundle_resolution(self):
        rng = np.random.default_rng(77)
        dim = 512
        bundle = _make_synthetic_bundle(model_name="aimids_ft", dim=dim, seed=77)
        prev_calib_dir = os.environ.get("AIMIDS_CALIB_DIR")
        prev_onnx_path = os.environ.get("AIMIDS_ONNX_PATH")

        with tempfile.TemporaryDirectory() as td:
            store = None
            try:
                os.environ["AIMIDS_CALIB_DIR"] = td
                os.environ.pop("AIMIDS_ONNX_PATH", None)

                bundle_path = os.path.join(td, "calib_aimids_scratch_ir50_crossage.npz")
                bundle.save(bundle_path)

                store = FaceStore(":memory:")
                p_vec = _make_unit_vecs(rng, 1, dim)[0]
                pid = store.add_person("Scratch_Test_Person")
                store.add_photo(pid, p_vec, model="aimids_ft", quality=0.9)

                # a) When AIMIDS_ONNX_PATH is unset -> default calib_aimids_ft.npz (not saved)
                default_path = resolve_bundle_path("aimids_ft")
                self.assertTrue(default_path.endswith("calib_aimids_ft.npz"))
                self.assertFalse(store.calibrated_ready("aimids_ft"))

                # b) When AIMIDS_ONNX_PATH points to custom ONNX -> resolves calib_<stem>.npz
                os.environ["AIMIDS_ONNX_PATH"] = os.path.join(td, "aimids_scratch_ir50_crossage.onnx")
                onnx_bundle_path = resolve_bundle_path("aimids_ft")
                self.assertTrue(onnx_bundle_path.endswith("calib_aimids_scratch_ir50_crossage.npz"))
                self.assertTrue(resolve_bundle_path("antelopev2").endswith("calib_antelopev2.npz"))
                self.assertTrue(store.calibrated_ready("aimids_ft"))

                q = p_vec + 0.15 * _make_unit_vecs(rng, 1, dim)[0]
                q /= np.linalg.norm(q)
                hits = store.search(q, k=5, model="aimids_ft", probe_quality=0.9)
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0]["person_id"], pid)
                self.assertIn("calibrated_llr", hits[0])
                self.assertIn("match_prob", hits[0])
                self.assertIn("p_nonmate", hits[0])
                self.assertIn("z_bg", hits[0])
                self.assertIn("advisory_flags", hits[0])
                self.assertIn(hits[0]["tier"], ("STRONG", "possible", "weak"))
            finally:
                if store is not None:
                    store.db.close()
                if prev_calib_dir is None:
                    os.environ.pop("AIMIDS_CALIB_DIR", None)
                else:
                    os.environ["AIMIDS_CALIB_DIR"] = prev_calib_dir
                if prev_onnx_path is None:
                    os.environ.pop("AIMIDS_ONNX_PATH", None)
                else:
                    os.environ["AIMIDS_ONNX_PATH"] = prev_onnx_path

    def test_v2_bundle_roundtrip_and_v1_backward_compat(self):
        rng = np.random.default_rng(101)
        dim = 512
        base = _make_synthetic_bundle(model_name="synthetic_v2", dim=dim, seed=101)

        # Build synthetic paired (young, adult) embeddings and compute top-5 age subspace
        idents = []
        is_young = []
        vecs = []
        age_dir = _make_unit_vecs(rng, 1, dim)[0]
        for i in range(25):
            id_vec = _make_unit_vecs(rng, 1, dim)[0]
            y_vec = id_vec + 0.5 * age_dir + 0.05 * _make_unit_vecs(rng, 1, dim)[0]
            a_vec = id_vec - 0.5 * age_dir + 0.05 * _make_unit_vecs(rng, 1, dim)[0]
            y_vec /= np.linalg.norm(y_vec)
            a_vec /= np.linalg.norm(a_vec)
            idents.extend([f"id_{i}", f"id_{i}"])
            is_young.extend([True, False])
            vecs.extend([y_vec, a_vec])
        basis = compute_age_subspace_basis(np.stack(vecs), idents, is_young, n_components=5)
        self.assertEqual(basis.shape, (5, dim))
        self.assertEqual(basis.dtype, np.float32)
        np.testing.assert_allclose(basis @ basis.T, np.eye(5, dtype=np.float32), atol=1e-5)

        self.assertEqual(compute_proj_hash(None), "none")
        self.assertNotEqual(compute_proj_hash(basis), "none")

        v2 = CalibrationBundle(
            model_name="synthetic_v2",
            dim=dim,
            asnorm=base.asnorm,
            qmf=base.qmf,
            openset=base.openset,
            cove=base.cove,
            proj_basis=basis,
            photo_hybrid_weight=0.35,
        )
        self.assertEqual(v2.schema_version, SCHEMA_VERSION)
        proj_vecs = v2.apply_subspace(np.stack(vecs))
        self.assertEqual(proj_vecs.shape, (50, dim))
        np.testing.assert_allclose(np.linalg.norm(proj_vecs, axis=1), 1.0, atol=1e-5)
        # Projected vectors are orthogonal to every basis vector
        np.testing.assert_allclose(proj_vecs @ basis.T, 0.0, atol=1e-5)

        with tempfile.TemporaryDirectory() as td:
            npz_p = os.path.join(td, "calib_v2.npz")
            _, json_p = v2.save(npz_p)
            loaded = CalibrationBundle.load(npz_p)
            self.assertEqual(loaded.schema_version, 2)
            self.assertIsNotNone(loaded.proj_basis)
            np.testing.assert_allclose(loaded.proj_basis, basis, atol=1e-6)
            self.assertEqual(loaded.proj_hash, v2.proj_hash)
            self.assertAlmostEqual(loaded.photo_hybrid_weight, 0.35, places=6)

            # Simulate a v1 .npz without proj_basis / proj_hash / photo_hybrid_weight
            v1_npz = os.path.join(td, "calib_v1.npz")
            with np.load(npz_p, allow_pickle=False) as d:
                v1_payload = {
                    k: d[k]
                    for k in d.files
                    if k not in ("proj_basis", "proj_hash", "photo_hybrid_weight")
                }
            v1_payload["schema_version"] = np.array([1], dtype=np.int64)
            np.savez_compressed(v1_npz, **v1_payload)

            loaded_v1 = CalibrationBundle.load(v1_npz)
            self.assertEqual(loaded_v1.schema_version, 1)
            self.assertIsNone(loaded_v1.proj_basis)
            self.assertEqual(loaded_v1.proj_hash, "none")
            self.assertAlmostEqual(loaded_v1.photo_hybrid_weight, 1.0, places=6)
            raw_normed = loaded_v1.apply_subspace(np.stack(vecs))
            np.testing.assert_allclose(raw_normed, np.stack(vecs), atol=1e-5)

    def test_store_projected_index_lifecycle_and_soft_hybrid(self):
        rng = np.random.default_rng(202)
        dim = 512
        cohort_raw = _make_unit_vecs(rng, 100, dim)
        basis = np.eye(5, dim, dtype=np.float32)  # first 5 dims are the "age subspace"
        cohort_proj = cohort_raw - (cohort_raw @ basis.T) @ basis
        cohort_proj /= np.maximum(np.linalg.norm(cohort_proj, axis=1, keepdims=True), 1e-12)

        asnorm = ExternalCohortASNorm(cohort_proj, top_k=50, std_floor=0.05)
        qmf = QMFCalibrator(weights=[1.5, 2.0, 0.0, 0.0, 0.0], bias=-1.0)
        base_os = rng.normal(loc=-2.5, scale=1.0, size=400).astype(np.float32)
        openset = OpenSetCalibrator.fit([1, 20], [base_os, base_os + 0.5], base_os)

        b_hybrid = CalibrationBundle(
            model_name="antelopev2",
            dim=dim,
            asnorm=asnorm,
            qmf=qmf,
            openset=openset,
            proj_basis=basis,
            photo_hybrid_weight=0.35,
        )
        b_max = CalibrationBundle(
            model_name="antelopev2",
            dim=dim,
            asnorm=asnorm,
            qmf=qmf,
            openset=openset,
            proj_basis=basis,
            photo_hybrid_weight=1.0,
        )

        store = FaceStore(":memory:")
        try:
            store.set_calibration_bundle("antelopev2", b_hybrid)

            # Person A has 2 photos: photo 0 orthogonal to query, photo 1 close to query in non-age subspace
            v_a0 = np.zeros(dim, dtype=np.float32)
            v_a0[10] = 1.0
            v_a1 = np.zeros(dim, dtype=np.float32)
            v_a1[11] = 1.0
            pid_a = store.add_person("Person_A")
            fid_a0 = store.add_photo(pid_a, v_a0, age=5, model="antelopev2", quality=0.9)
            fid_a1 = store.add_photo(pid_a, v_a1, age=25, model="antelopev2", quality=0.9)

            # Person B has 1 photo aligned along the age subspace (dim 0) with tiny dim-12 component
            v_b = np.zeros(dim, dtype=np.float32)
            v_b[0] = 0.99
            v_b[12] = float(np.sqrt(1.0 - 0.99 ** 2))
            pid_b = store.add_person("Person_B")
            store.add_photo(pid_b, v_b, age=10, model="antelopev2", quality=0.9)

            # Verify SQLite keeps raw unprojected vectors!
            blob_b = store.db.execute("SELECT embedding FROM faces WHERE person_id=?", (pid_b,)).fetchone()[0]
            stored_b = np.frombuffer(blob_b, dtype=np.float32)
            self.assertAlmostEqual(float(stored_b[0]), 0.99, places=5)

            # Query aligned with v_a1 (photo 1 of A), also sharing age-dim 0 with B
            q = np.zeros(dim, dtype=np.float32)
            q[0] = 0.8
            q[11] = 0.6

            hits_hyb = store.search(q, k=5, model="antelopev2", probe_quality=0.9)
            self.assertEqual(hits_hyb[0]["person_id"], pid_a)
            self.assertEqual(hits_hyb[0]["match_source"], "photo")
            # Projected FAISS index must suppress dim-0 correlation on Person_B
            hit_b = next(h for h in hits_hyb if h["person_id"] == pid_b)
            self.assertAlmostEqual(hit_b["raw_centroid_similarity"], 0.0, places=4)

            # Verify soft hybrid formula vs w_h=1.0 (pure max) on Person_A
            q_eff = b_hybrid.apply_subspace(q[None, :])[0]
            p_stats = asnorm.probe_stats(q_eff)
            cent_a = b_hybrid.apply_subspace(v_a0[None, :])[0] * 0.5 + b_hybrid.apply_subspace(v_a1[None, :])[0] * 0.5
            cent_a /= np.linalg.norm(cent_a)
            raw_c = float(np.dot(cent_a, q_eff))
            z_c = float(asnorm.score_batch(q_eff, cand_vecs=cent_a[None, :], raw_sims=np.array([raw_c]), probe_stats=p_stats)[0])
            llr_c = qmf.predict_llr(z_c, raw_c, q_min=0.9)
            raw_p = float(np.dot(b_hybrid.apply_subspace(v_a1[None, :])[0], q_eff))
            z_p = float(asnorm.score_batch(q_eff, cand_vecs=v_a1[None, :], raw_sims=np.array([raw_p]), probe_stats=p_stats)[0])
            llr_p = qmf.predict_llr(z_p, raw_p, q_min=0.9)
            expected_hyb_llr = llr_c + 0.35 * max(0.0, llr_p - llr_c)
            self.assertAlmostEqual(hits_hyb[0]["calibrated_llr"], expected_hyb_llr, places=3)

            store.set_calibration_bundle("antelopev2", b_max)
            hits_max = store.search(q, k=5, model="antelopev2", probe_quality=0.9)
            self.assertAlmostEqual(hits_max[0]["calibrated_llr"], max(llr_c, llr_p), places=3)
            self.assertGreater(hits_max[0]["calibrated_llr"], hits_hyb[0]["calibrated_llr"])

            # Verify incremental add_photo and delete_person keep projected index & centroids synchronized
            v_c = np.zeros(dim, dtype=np.float32)
            v_c[13] = 1.0
            pid_c = store.add_person("Person_C")
            store.add_photo(pid_c, v_c, age=30, model="antelopev2", quality=0.9)
            hits_after_add = store.search(v_c, k=5, model="antelopev2", probe_quality=0.9)
            self.assertEqual(hits_after_add[0]["person_id"], pid_c)

            store.delete_person(pid_c)
            hits_after_del = store.search(v_c, k=5, model="antelopev2", probe_quality=0.9)
            self.assertFalse(any(h["person_id"] == pid_c for h in hits_after_del))
            self.assertIn(fid_a0, store._calib_face_stats["antelopev2"]["fids"].tolist())
            self.assertIn(fid_a1, store._calib_face_stats["antelopev2"]["fids"].tolist())
        finally:
            store.db.close()

    def test_store_projected_delete_all_and_incremental_rebuild_parity(self):
        rng = np.random.default_rng(303)
        dim = 512
        cohort_raw = _make_unit_vecs(rng, 80, dim)
        basis = compute_age_subspace_basis(
            cohort_raw, [f"id_{i // 2}" for i in range(80)], [i % 2 == 0 for i in range(80)], n_components=5
        )
        asnorm = ExternalCohortASNorm(cohort_raw - (cohort_raw @ basis.T) @ basis, top_k=40, std_floor=0.05)
        qmf = QMFCalibrator(weights=[1.2, 1.5, 0.0, 0.0, 0.0], bias=-1.0)
        base_os = rng.normal(loc=-2.0, scale=1.0, size=300).astype(np.float32)
        openset = OpenSetCalibrator.fit([1, 20], [base_os, base_os + 0.4], base_os)
        b_proj = CalibrationBundle(
            model_name="antelopev2", dim=dim, asnorm=asnorm, qmf=qmf, openset=openset,
            proj_basis=basis, photo_hybrid_weight=0.35,
        )
        b_noproj = CalibrationBundle(
            model_name="antelopev2", dim=dim, asnorm=ExternalCohortASNorm(cohort_raw, top_k=40, std_floor=0.05),
            qmf=qmf, openset=openset, proj_basis=None, photo_hybrid_weight=0.35,
        )

        with tempfile.TemporaryDirectory() as td:
            # Verify tampered proj_basis triggers hash mismatch on load
            p_pr = os.path.join(td, "calib_proj.npz")
            b_proj.save(p_pr)
            with np.load(p_pr, allow_pickle=False) as d:
                bad = {k: d[k] for k in d.files}
            bad["proj_basis"] = _make_unit_vecs(rng, 5, dim)
            p_bad = os.path.join(td, "calib_bad.npz")
            np.savez_compressed(p_bad, **bad)
            with self.assertRaises(ValueError):
                CalibrationBundle.load(p_bad)

            db_path = os.path.join(td, "parity.db")
            s_inc = FaceStore(db_path)
            s_fresh = None
            try:
                s_inc.set_calibration_bundle("antelopev2", b_proj)
                p1 = s_inc.add_person("OnlyPerson")
                s_inc.add_photo(p1, _make_unit_vecs(rng, 1, dim)[0], age=5, model="antelopev2", quality=0.8)
                self.assertEqual(len(s_inc.search(_make_unit_vecs(rng, 1, dim)[0], k=5, model="antelopev2")), 1)
                # Delete ALL persons -> empty gallery -> add new persons -> exact parity with fresh FaceStore
                s_inc.delete_person(p1)
                self.assertEqual(s_inc.search(_make_unit_vecs(rng, 1, dim)[0], k=5, model="antelopev2"), [])

                pids = []
                for i in range(6):
                    pid = s_inc.add_person(f"Person_{i}")
                    pids.append(pid)
                    s_inc.add_photo(pid, _make_unit_vecs(rng, 1, dim)[0], age=5 + i, model="antelopev2", quality=0.65)
                    s_inc.add_photo(pid, _make_unit_vecs(rng, 1, dim)[0], age=6 + i, model="antelopev2", quality=0.90)
                    s_inc.add_photo(pid, _make_unit_vecs(rng, 1, dim)[0], age=25 + i, model="antelopev2", quality=0.75)
                _ = s_inc.search(_make_unit_vecs(rng, 1, dim)[0], k=5, model="antelopev2")
                s_inc.delete_person(pids[2])
                pid_extra = s_inc.add_person("Extra")
                s_inc.add_photo(pid_extra, _make_unit_vecs(rng, 1, dim)[0], age=14, model="antelopev2", quality=0.85)

                q = _make_unit_vecs(rng, 1, dim)[0]
                res_inc = s_inc.search(q, k=10, model="antelopev2", probe_quality=0.8)

                s_fresh = FaceStore(db_path)
                s_fresh.set_calibration_bundle("antelopev2", b_proj)
                res_fresh = s_fresh.search(q, k=10, model="antelopev2", probe_quality=0.8)

                self.assertEqual(len(res_inc), len(res_fresh))
                for a, b in zip(res_inc, res_fresh):
                    self.assertEqual(a["person_id"], b["person_id"])
                    self.assertEqual(a["match_source"], b["match_source"])
                    self.assertAlmostEqual(a["calibrated_llr"], b["calibrated_llr"], places=4)
                    self.assertAlmostEqual(a["similarity"], b["similarity"], places=4)

                # Switch bundle proj -> noproj -> proj
                s_inc.set_calibration_bundle("antelopev2", b_noproj)
                _ = s_inc.search(q, k=5, model="antelopev2", probe_quality=0.8)
                s_inc.set_calibration_bundle("antelopev2", b_proj)
                res_sw = s_inc.search(q, k=10, model="antelopev2", probe_quality=0.8)
                for a, b in zip(res_sw, res_fresh):
                    self.assertEqual(a["person_id"], b["person_id"])
                    self.assertAlmostEqual(a["calibrated_llr"], b["calibrated_llr"], places=4)
            finally:
                s_inc.db.close()
                if s_fresh is not None:
                    s_fresh.db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
