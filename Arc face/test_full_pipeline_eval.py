"""
test_full_pipeline_eval.py - Offline synthetic unittest suite for full_pipeline_eval.py.

Runs 100% offline using synthetic images, embeddings, and CalibrationBundles
(no GPU, no ONNX weights, no real datasets, no modifications to existing .db or .npz files).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import tempfile
import unittest

import cv2
import numpy as np

ARC_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(ARC_DIR)
for _p in (MAIN_DIR, ARC_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, MAIN_DIR)
sys.path.insert(0, ARC_DIR)

from bench_datasets import normalize_identity
from calibration import (
    CalibrationBundle,
    CoveCalibrator,
    ExternalCohortASNorm,
    OpenSetCalibrator,
    QMFCalibrator,
)
import full_pipeline_eval as fpe


def _unit_vec(arr: np.ndarray) -> np.ndarray:
    v = np.asarray(arr, dtype=np.float32).reshape(-1)
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)


def _write_dummy_jpg(path: str, seed_byte: int, size: int = 16) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = np.full((size, size, 3), int(seed_byte) % 256, dtype=np.uint8)
    img[0, 0, 0] = (int(seed_byte) * 41 + 7) % 256
    img[-1, -1, 2] = (int(seed_byte) * 89 + 13) % 256
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def _make_synthetic_bundle(dim: int = 512, seed: int = 17) -> CalibrationBundle:
    rng = np.random.default_rng(seed)
    cohort = rng.normal(size=(120, dim)).astype(np.float32)
    cohort /= np.maximum(np.linalg.norm(cohort, axis=1, keepdims=True), 1e-12)
    asnorm = ExternalCohortASNorm(cohort_vecs=cohort, top_k=40, std_floor=0.05)

    n_trials = 150
    z_gen = rng.normal(loc=5.5, scale=0.8, size=n_trials)
    z_imp = rng.normal(loc=0.0, scale=0.8, size=n_trials)
    r_gen = rng.uniform(0.55, 0.90, size=n_trials)
    r_imp = rng.uniform(-0.15, 0.25, size=n_trials)
    qmf = QMFCalibrator.fit(
        asnorm_z=np.concatenate([z_gen, z_imp]),
        raw_sim=np.concatenate([r_gen, r_imp]),
        labels=np.concatenate([np.ones(n_trials, dtype=np.int32), np.zeros(n_trials, dtype=np.int32)]),
        age_gap=None,
    )

    n_grid = np.array([1, 15, 50], dtype=np.int64)
    base = rng.normal(loc=-2.5, scale=1.1, size=600).astype(np.float32)
    max_scores = np.stack([
        rng.normal(loc=-1.2, scale=0.8, size=200),
        rng.normal(loc=-0.3, scale=0.8, size=200),
        rng.normal(loc=0.4, scale=0.8, size=200),
    ]).astype(np.float32)
    openset = OpenSetCalibrator.fit(n_grid=n_grid, max_scores_by_n=max_scores, base_nonmate_scores=base)
    cove_vecs = rng.normal(size=(20, dim)).astype(np.float32)
    cove_vecs /= np.maximum(np.linalg.norm(cove_vecs, axis=1, keepdims=True), 1e-12)
    cove = CoveCalibrator.fit(cove_vecs, [f"c_{i}" for i in range(20)])
    return CalibrationBundle(model_name="synthetic", dim=dim, asnorm=asnorm, qmf=qmf, openset=openset, cove=cove)


class TestFullPipelineEval(unittest.TestCase):
    def test_cross_dataset_dedup_and_disjointness(self):
        with tempfile.TemporaryDirectory() as td:
            cohort_ids = {"cohortperson", "Cohort Alias"}
            cacd_ids = {"cacdperson", "Cacd Alias"}

            # FG-NET 001 (child + adult)
            fgnet_gal = {"001": [(4, os.path.join(td, "fgnet_001_c.jpg"))]}
            fgnet_probes = [{"subject_id": "001", "age": 22, "path": os.path.join(td, "fgnet_001_a.jpg")}]

            # ECAF has John Lennon (highest celebrity priority) + ECAF Only + Cohort/CACD blocked entries
            ecaf_ids = {
                "johnlennon": {
                    "display_name": "John Lennon",
                    "child": [(0, os.path.join(td, "ecaf_jl_c.png"))],
                    "adult": [(0, os.path.join(td, "ecaf_jl_a.png"))],
                    "official_test": True,
                },
                "ecafonly": {
                    "display_name": "ECAF Only",
                    "child": [(0, os.path.join(td, "ecaf_only_c.png"))],
                    "adult": [(0, os.path.join(td, "ecaf_only_a.png"))],
                    "official_test": False,
                },
                "cohortperson": {
                    "display_name": "Cohort Person",
                    "child": [(0, os.path.join(td, "ecaf_coh_c.png"))],
                    "adult": [(0, os.path.join(td, "ecaf_coh_a.png"))],
                    "official_test": True,
                },
                "cacdperson": {
                    "display_name": "CACD Person",
                    "child": [(0, os.path.join(td, "ecaf_cacd_c.png"))],
                    "adult": [(0, os.path.join(td, "ecaf_cacd_a.png"))],
                    "official_test": True,
                },
            }

            # CAF has spelling alias "Jhon Lenon" (must dedup to ECAF!), "CAF Overlap AgeDB", and "CAF Only"
            caf_ids = {
                "Jhon Lenon": {
                    "display_name": "Jhon Lenon",
                    "photos": [(7, os.path.join(td, "caf_jl_c.jpg")), (25, os.path.join(td, "caf_jl_a.jpg"))],
                },
                "Overlap Star": {
                    "display_name": "Overlap Star",
                    "photos": [(8, os.path.join(td, "caf_ov_c.jpg")), (24, os.path.join(td, "caf_ov_a.jpg"))],
                },
                "CAF Only": {
                    "display_name": "CAF Only",
                    "photos": [(9, os.path.join(td, "caf_only_c.jpg")), (28, os.path.join(td, "caf_only_a.jpg"))],
                },
            }

            # AgeDB has AgeDB prefix alias for John Lennon, CAF overlap, mated >=15y, tight 1-probe >=15y, and <15y distractor
            agedb_ids = {
                "AgeDB John Lennon": [
                    (15, os.path.join(td, "adb_jl_1.jpg"), "m"),
                    (40, os.path.join(td, "adb_jl_2.jpg"), "m"),
                ],
                "Overlap Star": [
                    (12, os.path.join(td, "adb_ov_1.jpg"), "f"),
                    (35, os.path.join(td, "adb_ov_2.jpg"), "f"),
                ],
                "Mated Star": [
                    (10, os.path.join(td, "adb_m_10.jpg"), "m"),
                    (12, os.path.join(td, "adb_m_12.jpg"), "m"),
                    (30, os.path.join(td, "adb_m_30.jpg"), "m"),
                    (35, os.path.join(td, "adb_m_35.jpg"), "m"),
                ],
                "TightGap Star": [
                    (10, os.path.join(td, "adb_tg_10.jpg"), "f"),
                    (12, os.path.join(td, "adb_tg_12.jpg"), "f"),
                    (26, os.path.join(td, "adb_tg_26.jpg"), "f"),
                ],
                "ShortGap Star": [
                    (20, os.path.join(td, "adb_sg_20.jpg"), "m"),
                    (28, os.path.join(td, "adb_sg_28.jpg"), "m"),
                ],
            }

            # Synthetic CALFW pairs across folds 0..9 + overlapping / contaminated IDs
            calfw_pairs = []
            # Overlaps / exclusions that must NOT enter any CALFW split
            overlap_names = ["Jhon_Lenon", "Mated_Star", "Cohort_Person", "Patrick_Dennehy"]
            for idx, nm in enumerate(overlap_names):
                p1 = os.path.join(td, f"{nm}_0001.jpg")
                p2 = os.path.join(td, f"{nm}_0002.jpg")
                calfw_pairs.append((p1, p2, 1, idx % 10))

            def _add_fold_ids(prefix: str, count: int, fold_start: int, fold_span: int):
                ids_paths = []
                for i in range(count):
                    nm = f"{prefix}_{i:02d}"
                    p1 = os.path.join(td, f"{nm}_0001.jpg")
                    p2 = os.path.join(td, f"{nm}_0002.jpg")
                    f = fold_start + (i % fold_span)
                    ids_paths.append((nm, p1, p2, f))
                    calfw_pairs.append((p1, p2, 1, f))
                for i in range(0, len(ids_paths) - 1, 2):
                    calfw_pairs.append((ids_paths[i][1], ids_paths[i + 1][2], 0, ids_paths[i][3]))

            _add_fold_ids("QmfCal", 12, 0, 2)     # folds 0..1 -> qmf_calib_ids
            _add_fold_ids("NmCal", 10, 2, 2)      # folds 2..3 -> calib_nonmate_ids
            _add_fold_ids("GalCal", 20, 4, 6)     # folds 4..9 -> stranger + calfw gallery

            plan = fpe.build_unified_5dataset_plan(
                cohort_ids=cohort_ids,
                cacd_ids=cacd_ids,
                calfw_pairs=calfw_pairs,
                fgnet_gallery_by_id=fgnet_gal,
                fgnet_probes=fgnet_probes,
                ecaf_identities=ecaf_ids,
                caf_identities=caf_ids,
                agedb_identities=agedb_ids,
                n_calib_nonmates=8,
                n_strangers=5,
                agedb_min_gap=15,
            )

            self.assertTrue(plan["disjointness_verified"])
            gal_ids = set(plan["gallery_by_id"].keys())
            self.assertEqual(len(gal_ids), len({normalize_identity(x) for x in gal_ids}))

            # Verify strict priority order: fgnet > ecaf > caf > agedb > calfw
            self.assertEqual(plan["dataset_by_id"]["fgnet_001"], "fgnet")
            self.assertEqual(plan["dataset_by_id"]["johnlennon"], "ecaf")
            self.assertEqual(plan["dataset_by_id"]["overlapstar"], "caf")
            self.assertEqual(plan["dataset_by_id"]["matedstar"], "agedb")
            self.assertEqual(plan["dataset_by_id"]["shortgapstar"], "agedb")
            self.assertNotIn("cohortperson", gal_ids)
            self.assertNotIn("cacdperson", gal_ids)
            self.assertNotIn("patrickdennehy", gal_ids)

            # Verify >= 15y AgeDB gap invariant and gallery-only < 15y distractor
            for pr in plan["mated_probes"]:
                if pr["dataset"] == "agedb":
                    max_gal_age = max(a for a, _ in plan["gallery_by_id"][pr["subject_id"]] if a is not None)
                    self.assertGreaterEqual(pr["age"] - max_gal_age, 15)
            self.assertFalse(any(pr["subject_id"] == "shortgapstar" for pr in plan["mated_probes"]))

            # Verify tight 1-probe fallback kept 16y gap (10 -> 26)
            tg_probes = [pr for pr in plan["mated_probes"] if pr["subject_id"] == "tightgapstar"]
            self.assertEqual(len(tg_probes), 1)
            self.assertEqual(tg_probes[0]["age"], 26)
            self.assertEqual([a for a, _ in plan["gallery_by_id"]["tightgapstar"]], [10])

            # Verify all 6 sets are pairwise disjoint
            all_sets = [
                {normalize_identity(x) for x in gal_ids},
                {normalize_identity(x) for x in plan["qmf_calib_ids"]},
                {normalize_identity(x) for x in plan["calib_nonmate_ids"]},
                {normalize_identity(x) for x in plan["stranger_test_ids"]},
                {normalize_identity(x) for x in cohort_ids},
                {normalize_identity(x) for x in cacd_ids},
            ]
            for i in range(len(all_sets)):
                for j in range(i + 1, len(all_sets)):
                    self.assertEqual(all_sets[i] & all_sets[j], set())
            self.assertEqual(len(plan["calib_nonmate_ids"]), 8)
            self.assertEqual(len(plan["stranger_test_ids"]), 5)

    def test_caf_train_leakage_flag_and_fair_subset(self):
        self.assertTrue(
            fpe.should_skip_caf(
                fpe.ENGINE_SPECS["fromscratchv2"]["onnx_path"],
                engine_name=fpe.ENGINE_SPECS["fromscratchv2"]["engine_name"],
            )
        )
        self.assertFalse(
            fpe.should_skip_caf(
                fpe.ENGINE_SPECS["aimids_crossage"]["onnx_path"],
                engine_name=fpe.ENGINE_SPECS["aimids_crossage"]["engine_name"],
            )
        )

        with tempfile.TemporaryDirectory() as td:
            dim = 512
            rng = np.random.default_rng(77)
            bundle = _make_synthetic_bundle(dim=dim, seed=19)
            gallery_by_id = {}
            dataset_by_id = {}
            mated_probes = []
            emb_cache = {}
            qual_cache = {}
            f_idx = 0

            # 4 subjects per dataset (20 total); CAF probes match well, non-CAF probes are orthogonal misses
            for ds_idx, ds in enumerate(("fgnet", "ecaf", "caf", "agedb", "calfw")):
                for s in range(4):
                    sid = f"{ds}_id_{s}"
                    g_path = os.path.join(td, f"{sid}_gal.jpg")
                    p_path = os.path.join(td, f"{sid}_pr.jpg")
                    _write_dummy_jpg(g_path, f_idx); f_idx += 1
                    _write_dummy_jpg(p_path, f_idx); f_idx += 1

                    base = np.zeros(dim, dtype=np.float32)
                    base[ds_idx * 10 + s] = 1.0
                    emb_cache[g_path] = base
                    qual_cache[g_path] = 0.9

                    if ds == "caf":
                        ortho = np.zeros(dim, dtype=np.float32)
                        ortho[200 + s] = 1.0
                        # cos = 0.90 (< 0.97 near-dup threshold) -> guaranteed Rank-1 hit for CAF only
                        emb_cache[p_path] = _unit_vec(0.90 * base + float(np.sqrt(1.0 - 0.90 ** 2)) * ortho)
                    else:
                        # Non-CAF probes align strongly (cos = 0.85) with a DIFFERENT subject (s+1)%4 -> guaranteed miss
                        wrong = np.zeros(dim, dtype=np.float32)
                        wrong[ds_idx * 10 + ((s + 1) % 4)] = 1.0
                        ortho = np.zeros(dim, dtype=np.float32)
                        ortho[300 + ds_idx * 10 + s] = 1.0
                        emb_cache[p_path] = _unit_vec(0.85 * wrong + float(np.sqrt(1.0 - 0.85 ** 2)) * ortho)
                    qual_cache[p_path] = 0.9
                    gallery_by_id[sid] = [(None, g_path)]
                    dataset_by_id[sid] = ds
                    mated_probes.append({
                        "subject_id": sid,
                        "norm_id": sid,
                        "display_name": sid,
                        "dataset": ds,
                        "age": None,
                        "path": p_path,
                    })

            stranger_probes = []
            for s in range(5):
                sp = os.path.join(td, f"stranger_{s}.jpg")
                _write_dummy_jpg(sp, f_idx); f_idx += 1
                emb_cache[sp] = _unit_vec(rng.normal(size=dim))
                qual_cache[sp] = 0.9
                stranger_probes.append({"subject_id": f"str_{s}", "path": sp})

            plan = {
                "gallery_by_id": gallery_by_id,
                "dataset_by_id": dataset_by_id,
                "mated_probes": mated_probes,
                "stranger_probes": stranger_probes,
            }
            summary, _ = fpe.evaluate_unified_gallery(
                plan=plan,
                engine_key="fromscratchv2",
                spec=fpe.ENGINE_SPECS["fromscratchv2"],
                emb_cache=emb_cache,
                qual_cache=qual_cache,
                bundles_by_mode={"matched_calibrated": bundle},
                bootstrap_n=50,
                seed=0,
            )
            self.assertTrue(summary["caf_in_train_set"])
            m_cal = summary["modes"]["matched_calibrated"]
            self.assertTrue(m_cal["caf_in_train_set"])
            self.assertTrue(m_cal["all_5_datasets"]["caf_in_train_set"])
            self.assertEqual(m_cal["fair_4_datasets_no_caf"]["probes"], 16)
            self.assertEqual(m_cal["all_5_datasets"]["probes"], 20)
            self.assertAlmostEqual(m_cal["fair_4_datasets_no_caf"]["rank1"], 0.0, places=4)
            self.assertAlmostEqual(m_cal["all_5_datasets"]["rank1"], 20.0, places=4)

    def test_synthetic_unified_calibrated_chance_rank1(self):
        with tempfile.TemporaryDirectory() as td:
            dim = 512
            rng = np.random.default_rng(1234)
            bundle = _make_synthetic_bundle(dim=dim, seed=42)
            gallery_by_id = {}
            dataset_by_id = {}
            mated_probes = []
            stranger_probes = []
            emb_cache = {}
            qual_cache = {}
            file_idx = 0

            datasets = ("fgnet", "ecaf", "caf", "agedb", "calfw")
            for i in range(50):
                ds = datasets[i % len(datasets)]
                sid = f"rand_{i:03d}"
                gp = os.path.join(td, f"{sid}_g.jpg")
                pp = os.path.join(td, f"{sid}_p.jpg")
                _write_dummy_jpg(gp, file_idx); file_idx += 1
                _write_dummy_jpg(pp, file_idx); file_idx += 1
                emb_cache[gp] = _unit_vec(rng.normal(size=dim))
                emb_cache[pp] = _unit_vec(rng.normal(size=dim))
                qual_cache[gp] = 0.85
                qual_cache[pp] = 0.85
                gallery_by_id[sid] = [(6, gp)]
                dataset_by_id[sid] = ds
                mated_probes.append({
                    "subject_id": sid,
                    "norm_id": sid,
                    "display_name": sid,
                    "dataset": ds,
                    "age": 26,
                    "path": pp,
                })

            for i in range(20):
                sp = os.path.join(td, f"str_{i:02d}.jpg")
                _write_dummy_jpg(sp, file_idx); file_idx += 1
                emb_cache[sp] = _unit_vec(rng.normal(size=dim))
                qual_cache[sp] = 0.85
                stranger_probes.append({"subject_id": f"str_{i:02d}", "path": sp})

            plan = {
                "gallery_by_id": gallery_by_id,
                "dataset_by_id": dataset_by_id,
                "mated_probes": mated_probes,
                "stranger_probes": stranger_probes,
            }
            summary, _ = fpe.evaluate_unified_gallery(
                plan=plan,
                engine_key="antelopev2",
                spec=fpe.ENGINE_SPECS["antelopev2"],
                emb_cache=emb_cache,
                qual_cache=qual_cache,
                bundles_by_mode={"matched_calibrated": bundle, "production_bundle": bundle},
                bootstrap_n=50,
                seed=0,
            )
            for mode_name in ("matched_calibrated", "production_bundle"):
                m_res = summary["modes"][mode_name]
                self.assertTrue(m_res["calibrated_ready"])
                self.assertEqual(m_res["gallery_ids_enrolled"], 50)
                self.assertLess(m_res["fair_4_datasets_no_caf"]["rank1"], 15.0)
                self.assertLess(m_res["all_5_datasets"]["rank1"], 15.0)

    def test_mcnemar_and_report_schema(self):
        engines = ["aimids_crossage", "antelopev2", "fromscratchv2"]
        # 30 probes across fgnet (15), ecaf (10), caf (5), with 1 near-dup in fromscratchv2
        probe_specs = (
            [("fgnet", f"fg_{i}") for i in range(15)]
            + [("ecaf", f"ec_{i}") for i in range(10)]
            + [("caf", f"cf_{i}") for i in range(5)]
        )

        artifacts = {}
        for ek in engines:
            mode_dict = {}
            for mode in ("matched_calibrated", "production_bundle"):
                rec_map = {}
                for idx, (ds, pkey) in enumerate(probe_specs):
                    if ek == "aimids_crossage":
                        hit = idx < 28  # 28/30
                    elif ek == "antelopev2":
                        hit = idx < 18  # 18/30 (significantly lower than aimids_crossage)
                    else:
                        hit = (idx < 25) and (idx != 2)  # 24/30
                    rec_map[pkey] = {
                        "dataset": ds,
                        "subject_id": pkey,
                        "hit_1": hit,
                        "is_exact_dup": False,
                        "is_near_dup": (ek == "fromscratchv2" and idx == 0),
                    }
                mode_dict[mode] = rec_map
            artifacts[ek] = {"probe_records": mode_dict}

        cmp_out = fpe.build_full_compare(engines, artifacts)
        expected_pairs = [
            "aimids_crossage_vs_antelopev2",
            "aimids_crossage_vs_fromscratchv2",
            "antelopev2_vs_fromscratchv2",
        ]
        self.assertEqual(sorted(cmp_out.keys()), sorted(expected_pairs))

        for pair_key in expected_pairs:
            for mode in ("matched_calibrated", "production_bundle"):
                mode_cmp = cmp_out[pair_key]["modes"][mode]
                self.assertIn("fair_4_datasets_no_caf", mode_cmp)
                self.assertIn("all_5_datasets", mode_cmp)
                for sub_key in ("fair_4_datasets_no_caf", "all_5_datasets"):
                    entry = mode_cmp[sub_key]
                    for req_field in ("n", "diff_pct", "p_value", "significant_p05", "b", "c"):
                        self.assertIn(req_field, entry)

        # Cross-check exact counts on aimids_crossage vs antelopev2 (fair_4 excludes 5 caf -> n=25)
        fair_ca_ant = cmp_out["aimids_crossage_vs_antelopev2"]["modes"]["matched_calibrated"]["fair_4_datasets_no_caf"]
        self.assertEqual(fair_ca_ant["n"], 25)
        self.assertEqual(fair_ca_ant["b"], 7)
        self.assertEqual(fair_ca_ant["c"], 0)
        self.assertTrue(fair_ca_ant["significant_p05"])

        # Cross-check paired exclusion when fromscratchv2 flags idx=0 as near_dup
        all5_ca_fs = cmp_out["aimids_crossage_vs_fromscratchv2"]["modes"]["matched_calibrated"]["all_5_datasets"]
        self.assertEqual(all5_ca_fs["n_shared_total"], 30)
        self.assertEqual(all5_ca_fs["n_excluded_dup_either"], 1)
        self.assertEqual(all5_ca_fs["n"], 29)

    def test_matched_calibration_bundle_fitting(self):
        with tempfile.TemporaryDirectory() as td:
            rng = np.random.default_rng(2026)
            dim = 512
            engines = ["antelopev2", "fromscratchv2"]
            caches = {ek: ({}, {}) for ek in engines}
            cohorts = {}
            for ek_idx, ek in enumerate(engines):
                coh = rng.normal(size=(100, dim)).astype(np.float32)
                coh /= np.maximum(np.linalg.norm(coh, axis=1, keepdims=True), 1e-12)
                cohorts[ek] = coh

            fit_pairs = []
            for i in range(20):
                pa = os.path.join(td, f"FitGen_{i:02d}_0001.jpg")
                pb = os.path.join(td, f"FitGen_{i:02d}_0002.jpg")
                fit_pairs.append((pa, pb, 1, i % 2))
                for ek in engines:
                    base = _unit_vec(rng.normal(size=dim))
                    noise = _unit_vec(rng.normal(size=dim))
                    caches[ek][0][pa] = base
                    caches[ek][0][pb] = _unit_vec(0.85 * base + float(np.sqrt(1.0 - 0.85 ** 2)) * noise)
                    caches[ek][1][pa] = 0.9
                    caches[ek][1][pb] = 0.85

            for i in range(20):
                pa = os.path.join(td, f"FitImpA_{i:02d}_0001.jpg")
                pb = os.path.join(td, f"FitImpB_{i:02d}_0001.jpg")
                fit_pairs.append((pa, pb, 0, i % 2))
                for ek in engines:
                    caches[ek][0][pa] = _unit_vec(rng.normal(size=dim))
                    caches[ek][0][pb] = _unit_vec(rng.normal(size=dim))
                    caches[ek][1][pa] = 0.9
                    caches[ek][1][pb] = 0.9

            # Extra pair missing in fromscratchv2 -> must be excluded by shared intersection
            p_miss_a = os.path.join(td, "FitMiss_00_0001.jpg")
            p_miss_b = os.path.join(td, "FitMiss_00_0002.jpg")
            fit_pairs.append((p_miss_a, p_miss_b, 1, 0))
            caches["antelopev2"][0][p_miss_a] = _unit_vec(rng.normal(size=dim))
            caches["antelopev2"][0][p_miss_b] = _unit_vec(rng.normal(size=dim))
            caches["fromscratchv2"][0][p_miss_a] = None
            caches["fromscratchv2"][0][p_miss_b] = _unit_vec(rng.normal(size=dim))

            calib_nm = []
            for i in range(15):
                pn = os.path.join(td, f"Nm_{i:02d}_0001.jpg")
                calib_nm.append((f"nm{i:02d}", pn))
                for ek in engines:
                    caches[ek][0][pn] = _unit_vec(rng.normal(size=dim))
                    caches[ek][1][pn] = 0.9
            pn_miss = os.path.join(td, "NmMiss_00_0001.jpg")
            calib_nm.append(("nmmiss00", pn_miss))
            caches["antelopev2"][0][pn_miss] = _unit_vec(rng.normal(size=dim))
            caches["fromscratchv2"][0][pn_miss] = None

            gallery_by_id = {}
            for i in range(40):
                gp = os.path.join(td, f"gal_{i:02d}.jpg")
                gallery_by_id[f"gal_{i:02d}"] = [(10, gp)]
                for ek in engines:
                    caches[ek][0][gp] = _unit_vec(rng.normal(size=dim))
                    caches[ek][1][gp] = 0.9

            plan = {
                "fit_pairs": fit_pairs,
                "calib_nonmate_probes": calib_nm,
                "gallery_by_id": gallery_by_id,
            }
            out_npz = os.path.join(td, "calib_scratch_test.npz")
            bundles, diag = fpe.fit_matched_calibration_bundle(
                engines, plan, caches, cohorts, save_paths={"fromscratchv2": out_npz}
            )
            self.assertEqual(diag["antelopev2"]["fit_trials"], 40)
            self.assertEqual(diag["fromscratchv2"]["fit_trials"], 40)
            self.assertEqual(diag["antelopev2"]["nonmate_calib_probes"], 15)
            self.assertEqual(diag["fromscratchv2"]["nonmate_calib_probes"], 15)
            for ek in engines:
                self.assertEqual(bundles[ek].qmf.weights[3], 0.0)
                self.assertEqual(bundles[ek].qmf.weights[4], 0.0)
                self.assertIsNotNone(bundles[ek].cove)
            self.assertTrue(os.path.isfile(out_npz))
            reloaded = CalibrationBundle.load(out_npz)
            self.assertEqual(reloaded.metadata.get("onnx_stem"), "aimids_scratch_ir50_crossage")

    def test_nfkd_cp1252_and_14_alias_dedup(self):
        # Verify NFKD accent-folding, CP1252 mojibake recovery, and CAF 'íª' apostrophe fix
        self.assertEqual(normalize_identity("José Mourinho"), "josemourinho")
        self.assertEqual(normalize_identity("JosÃ© Mourinho"), "josemourinho")
        self.assertEqual(normalize_identity("Pelé"), "pele")
        self.assertEqual(normalize_identity("Zinédine Zidane"), "zinedinezidane")
        self.assertEqual(normalize_identity("Khloé Kardashian"), "khloekardashian")
        self.assertEqual(normalize_identity("Luka Modrić"), "lukamodric")
        self.assertEqual(normalize_identity("Carmen Dell\u00ed\u00aaOrefice"), "carmendellorefice")
        self.assertEqual(normalize_identity("Dixie D\u00ed\u00aaAmelio"), "dixiedamelio")

        expected_14 = {
            "Andrew Warhola": "andywarhol",
            "Audrey Kathleen Ruston": "audreyhepburn",
            "William James Murray": "billmurray",
            "Courteney Bass Cox": "courtneycox",
            "Diego Armando Maradona": "diegomaradona",
            "Elizabeth Jane Hurley": "elizabethhurley",
            "Jackie Chan Tribute": "jackiechan",
            "Norma Jeane Mortenson": "marilynmonroe",
            "Nicole Mary Kidman": "nicolekidman",
            "pel": "pele",
            "Ronaldo Luis Nazario de Lima": "ronaldoelfenomeno",
            "Vincent Lamar Carter": "vincecarter",
            "zindinezidane": "zinedinezidane",
            "josmourinho": "josemourinho",
        }
        for raw, canon in expected_14.items():
            self.assertEqual(normalize_identity(raw), canon, msg=f"Alias failed for {raw}")

        # Verify build_unified_5dataset_plan deduplicates cross-dataset accented/aliased IDs
        with tempfile.TemporaryDirectory() as td:
            ecaf_ids = {
                "pele": {
                    "display_name": "Pele",
                    "child": [(0, os.path.join(td, "ecaf_pele_c.png"))],
                    "adult": [(0, os.path.join(td, "ecaf_pele_a.png"))],
                    "official_test": True,
                }
            }
            caf_ids = {
                "Pelé": {
                    "display_name": "Pelé",
                    "photos": [(8, os.path.join(td, "caf_pele_c.jpg")), (30, os.path.join(td, "caf_pele_a.jpg"))],
                },
                "José Mourinho": {
                    "display_name": "José Mourinho",
                    "photos": [(9, os.path.join(td, "caf_jm_c.jpg")), (40, os.path.join(td, "caf_jm_a.jpg"))],
                },
            }
            agedb_ids = {
                "Pele": [
                    (15, os.path.join(td, "adb_pele_1.jpg"), "m"),
                    (40, os.path.join(td, "adb_pele_2.jpg"), "m"),
                ],
                "Ronaldo Luis Nazario de Lima": [
                    (18, os.path.join(td, "adb_r9_1.jpg"), "m"),
                    (36, os.path.join(td, "adb_r9_2.jpg"), "m"),
                ],
            }
            calfw_pairs = []
            # Add duplicate alias into CALFW + clean CALFW fold IDs
            p_mm1 = os.path.join(td, "Norma_Jeane_Mortenson_0001.jpg")
            p_mm2 = os.path.join(td, "Norma_Jeane_Mortenson_0002.jpg")
            calfw_pairs.append((p_mm1, p_mm2, 1, 0))
            agedb_ids["Marilyn Monroe"] = [
                (15, os.path.join(td, "adb_mm_1.jpg"), "f"),
                (35, os.path.join(td, "adb_mm_2.jpg"), "f"),
            ]
            for fold_prefix, fold_start, count in (("Q", 0, 6), ("N", 2, 6), ("G", 4, 6)):
                for i in range(count):
                    pa = os.path.join(td, f"{fold_prefix}{i:02d}_0001.jpg")
                    pb = os.path.join(td, f"{fold_prefix}{i:02d}_0002.jpg")
                    calfw_pairs.append((pa, pb, 1, fold_start + (i % 2)))

            plan = fpe.build_unified_5dataset_plan(
                cohort_ids=set(),
                cacd_ids=set(),
                calfw_pairs=calfw_pairs,
                fgnet_gallery_by_id={},
                fgnet_probes=[],
                ecaf_identities=ecaf_ids,
                caf_identities=caf_ids,
                agedb_identities=agedb_ids,
                n_calib_nonmates=4,
                n_strangers=2,
                agedb_min_gap=15,
            )
            self.assertTrue(plan["disjointness_verified"])
            # Pelé deduplicated across ECAF, CAF, AgeDB -> stays in ECAF only
            self.assertEqual(plan["dataset_by_id"]["pele"], "ecaf")
            self.assertEqual(plan["dataset_by_id"]["josemourinho"], "caf")
            self.assertEqual(plan["dataset_by_id"]["ronaldoelfenomeno"], "agedb")
            self.assertEqual(plan["dataset_by_id"]["marilynmonroe"], "agedb")
            # Marilyn Monroe alias in CALFW must NOT enter qmf_calib_ids
            self.assertNotIn("marilynmonroe", plan["qmf_calib_ids"])

    def test_v2_matched_calibration_bundle_fitting(self):
        with tempfile.TemporaryDirectory() as td:
            rng = np.random.default_rng(2027)
            dim = 512
            engines = ["antelopev2", "fromscratchv2"]
            caches = {ek: ({}, {}) for ek in engines}
            cohorts = {}
            cohort_idents = []
            cohort_young = []
            for i in range(50):
                cohort_idents.extend([f"coh_{i}", f"coh_{i}"])
                cohort_young.extend([True, False])

            for ek in engines:
                age_dir = _unit_vec(rng.normal(size=dim))
                rows = []
                for i in range(50):
                    base = _unit_vec(rng.normal(size=dim))
                    y_v = _unit_vec(base + 0.6 * age_dir + 0.05 * rng.normal(size=dim))
                    a_v = _unit_vec(base - 0.6 * age_dir + 0.05 * rng.normal(size=dim))
                    rows.extend([y_v, a_v])
                cohorts[ek] = np.stack(rows).astype(np.float32)

            fit_pairs = []
            for i in range(18):
                pa = os.path.join(td, f"V2Gen_{i:02d}_0001.jpg")
                pb = os.path.join(td, f"V2Gen_{i:02d}_0002.jpg")
                fit_pairs.append((pa, pb, 1, i % 2))
                for ek in engines:
                    base = _unit_vec(rng.normal(size=dim))
                    noise = _unit_vec(rng.normal(size=dim))
                    caches[ek][0][pa] = base
                    caches[ek][0][pb] = _unit_vec(0.85 * base + float(np.sqrt(1.0 - 0.85 ** 2)) * noise)
                    caches[ek][1][pa] = 0.9
                    caches[ek][1][pb] = 0.85

            for i in range(18):
                pa = os.path.join(td, f"V2ImpA_{i:02d}_0001.jpg")
                pb = os.path.join(td, f"V2ImpB_{i:02d}_0001.jpg")
                fit_pairs.append((pa, pb, 0, i % 2))
                for ek in engines:
                    caches[ek][0][pa] = _unit_vec(rng.normal(size=dim))
                    caches[ek][0][pb] = _unit_vec(rng.normal(size=dim))
                    caches[ek][1][pa] = 0.9
                    caches[ek][1][pb] = 0.9

            calib_nm = []
            for i in range(15):
                pn = os.path.join(td, f"V2Nm_{i:02d}_0001.jpg")
                calib_nm.append((f"v2nm{i:02d}", pn))
                for ek in engines:
                    caches[ek][0][pn] = _unit_vec(rng.normal(size=dim))
                    caches[ek][1][pn] = 0.9

            gallery_by_id = {}
            for i in range(30):
                gp = os.path.join(td, f"v2gal_{i:02d}.jpg")
                gallery_by_id[f"v2gal_{i:02d}"] = [(10, gp)]
                for ek in engines:
                    caches[ek][0][gp] = _unit_vec(rng.normal(size=dim))
                    caches[ek][1][gp] = 0.9

            plan = {
                "fit_pairs": fit_pairs,
                "calib_nonmate_probes": calib_nm,
                "gallery_by_id": gallery_by_id,
            }
            out_npz = os.path.join(td, "calib_scratch_v2_test.npz")
            bundles, diag = fpe.fit_matched_calibration_bundle(
                engines,
                plan,
                caches,
                cohorts,
                save_paths={"fromscratchv2": out_npz},
                cohort_identities=cohort_idents,
                cohort_is_young=cohort_young,
                proj_k=5,
                photo_hybrid_weight=0.35,
            )
            for ek in engines:
                b = bundles[ek]
                self.assertEqual(b.schema_version, 2)
                self.assertIsNotNone(b.proj_basis)
                self.assertEqual(b.proj_basis.shape, (5, dim))
                self.assertAlmostEqual(b.photo_hybrid_weight, 0.35, places=6)
                self.assertEqual(b.asnorm.m, 50)  # filtered to LAG_ADULT (~is_young)
                self.assertEqual(diag[ek]["proj_k"], 5)
                self.assertAlmostEqual(diag[ek]["photo_hybrid_weight"], 0.35, places=6)
                self.assertEqual(diag[ek]["cohort_size"], 50)
                # Cohort stored in asnorm must already be orthogonal to proj_basis
                np.testing.assert_allclose(b.asnorm.cohort_vecs @ b.proj_basis.T, 0.0, atol=1e-5)

            reloaded = CalibrationBundle.load(out_npz)
            self.assertEqual(reloaded.schema_version, 2)
            self.assertIsNotNone(reloaded.proj_basis)
            self.assertEqual(reloaded.proj_basis.shape, (5, dim))
            self.assertAlmostEqual(reloaded.photo_hybrid_weight, 0.35, places=6)


if __name__ == "__main__":
    unittest.main()
