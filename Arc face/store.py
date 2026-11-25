"""
store.py - where people and their face vectors live.

SQLite (one file) is the single source of truth. FAISS is rebuilt from it when the program
starts, so the two can never get out of sync. Fine for tens of thousands of photos.

    from store import FaceStore
    store = FaceStore("aimids.db")
    pid = store.add_person("Ali Hassan", notes="missing since 2015, last seen Cairo")
    store.add_photo(pid, embedding, photo_path="photos/ali_2012.jpg", age=9, model="antelopev2")
    hits = store.search(query_embedding, k=10, model="antelopev2")
    # -> [{"person_id", "name", "notes", "similarity", "tier", "photo_path", "age"}, ...] best first

Every embedding is tagged with the engine (model) that produced it, and each model gets its own
FAISS index and its own per-person centroid. Vectors from two different engines are never compared
and never averaged together, so switching AIMIDS_ENGINE cannot silently corrupt an existing gallery:
old-model photos simply stop being searched until they are re-registered. stats()["models"] shows
what is actually in the database.

Search returns ONE entry per person (their best photo), at most k people, only those at or
above min_sim. Multi-photo centroid fusion combines a person's same-model photos into one
identity vector weighted by quality and inverse age density. When a calibration bundle is
available (`calibrated_ready(model)`), search runs the 4-layer calibrated pipeline (external
AS-Norm -> QMF logistic LLR -> N-adaptive open-set conformal/GPD tiers -> background rank
z-score advisories). Otherwise, it falls back to cosine similarity `max(best single photo, centroid)`
with fixed tiers: "STRONG" (>= 0.36), "possible" (>= 0.26), or "weak" (< 0.26).
"""
import os
import sqlite3
import threading
import time
import numpy as np
import faiss

try:
    from calibration import CalibrationBundle, compute_bg_zscore, resolve_bundle_path
except ImportError:
    try:
        from .calibration import CalibrationBundle, compute_bg_zscore, resolve_bundle_path
    except ImportError:
        CalibrationBundle = None
        compute_bg_zscore = None
        resolve_bundle_path = None

DIM = 512
SUPPORTED_DIMS = (512, 1024)
STRONG, POSSIBLE = 0.36, 0.26
UNKNOWN_MODEL = "unknown"   # rows written before embeddings were tagged, or callers that pass no model


class FaceStore:
    def __init__(self, db_path="aimids.db"):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS persons(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created REAL
            );
            CREATE TABLE IF NOT EXISTS faces(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL REFERENCES persons(id),
                photo_path TEXT,
                age INTEGER,
                embedding BLOB NOT NULL,
                model TEXT,
                quality REAL DEFAULT 1.0
            );
            CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
        """)
        self._migrate()
        self.indexes = {}          # model -> faiss IndexIDMap
        self._model_dims = {}      # model -> embedding dimension (512 or 1024)
        self._centroids = {}       # (person_id, model) -> unit vector
        self._centroid_cache = {}  # model -> (person_ids array, centroid matrix)
        self._calib_bundles = {}   # model -> CalibrationBundle
        self._calib_face_stats = {}  # model -> precomputed cohort stats for faces
        self._calib_cent_stats = {}  # model -> precomputed cohort stats for centroids
        self._calib_indexes = {}     # model -> faiss IndexIDMap (aliased or projected)
        self._calib_centroids = {}   # model -> (person_ids array, centroid matrix)
        self._calib_proj_hashes = {} # model -> proj_hash built into _calib_indexes/_calib_centroids
        self._face_to_person = {}
        self._face_to_model = {}
        self._load()

    # ------------------------------------------------------------------ internals
    def _migrate(self):
        """Adds missing columns (model, quality) to a database created by an older version."""
        cols = [r[1] for r in self.db.execute("PRAGMA table_info(faces)").fetchall()]
        if "model" not in cols:
            self.db.execute("ALTER TABLE faces ADD COLUMN model TEXT")
        if "quality" not in cols:
            self.db.execute("ALTER TABLE faces ADD COLUMN quality REAL DEFAULT 1.0")
        self.db.execute("UPDATE faces SET model=? WHERE model IS NULL", (UNKNOWN_MODEL,))
        self.db.execute("UPDATE faces SET quality=1.0 WHERE quality IS NULL")
        self.db.commit()

    def _index(self, model, dim=None):
        idx = self.indexes.get(model)
        if idx is None:
            d = dim if dim is not None else self._model_dims.get(model)
            if d is None:
                raise ValueError(f"unknown embedding dimension for model '{model}'")
            d = int(d)
            if d not in SUPPORTED_DIMS:
                raise ValueError(f"unsupported embedding dimension {d} for model '{model}'")
            self._model_dims[model] = d
            idx = faiss.IndexIDMap(faiss.IndexFlatIP(d))
            self.indexes[model] = idx
        elif dim is not None and int(dim) != self._model_dims.get(model):
            raise ValueError(
                f"model '{model}' expects {self._model_dims.get(model)}-d embedding, got {int(dim)}-d"
            )
        return idx

    def _invalidate_centroid_cache(self):
        self._centroid_cache.clear()

    @staticmethod
    def _compute_weighted_centroid(vecs, ages, quals):
        if len(vecs) == 1:
            v = np.asarray(vecs[0], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            if abs(norm - 1.0) <= 1e-6:
                return v
            return (v / norm if norm > 0 else v).astype(np.float32)
        weights = np.empty(len(vecs), dtype=np.float32)
        for i, (age_i, qual_i) in enumerate(zip(ages, quals)):
            q_raw = 1.0 if qual_i is None or not np.isfinite(qual_i) else float(qual_i)
            q_clamped = min(max(q_raw, 0.25), 1.0)
            if age_i is None:
                w_age = 1.0
            else:
                age_f = float(age_i)
                nearby = sum(
                    1 for j, age_j in enumerate(ages)
                    if j != i and age_j is not None and abs(float(age_j) - age_f) <= 2.0
                )
                w_age = 1.0 / (1.0 + nearby)
            weights[i] = q_clamped * w_age
        weighted = np.sum(vecs * weights[:, None], axis=0)
        norm = float(np.linalg.norm(weighted))
        return (weighted / norm if norm > 0 else weighted).astype(np.float32)

    def _update_centroid(self, person_id, model):
        rows = self.db.execute(
            "SELECT embedding, age, quality FROM faces WHERE person_id=? AND model=?",
            (person_id, model),
        ).fetchall()
        if rows:
            vecs = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
            ages = [r[1] for r in rows]
            quals = [r[2] for r in rows]
            self._centroids[(person_id, model)] = self._compute_weighted_centroid(vecs, ages, quals)
        else:
            self._centroids.pop((person_id, model), None)
        self._invalidate_centroid_cache()

    def _model_centroids(self, model):
        """(person_ids, centroid matrix) for one model, built once and cached until the next write."""
        cached = self._centroid_cache.get(model)
        if cached is not None:
            return cached
        pids, vecs = [], []
        for (pid, m), v in self._centroids.items():
            if m == model:
                pids.append(pid)
                vecs.append(v)
        if pids:
            cached = (np.array(pids, dtype=np.int64), np.stack(vecs))
        else:
            d = self._model_dims.get(model, DIM)
            cached = (np.array([], dtype=np.int64), np.empty((0, d), dtype=np.float32))
        self._centroid_cache[model] = cached
        return cached

    def _rebuild_all(self):
        self.indexes.clear()
        self._model_dims.clear()
        self._centroids.clear()
        self._centroid_cache.clear()
        self._calib_face_stats.clear()
        self._calib_cent_stats.clear()
        self._calib_indexes.clear()
        self._calib_centroids.clear()
        self._calib_proj_hashes.clear()
        self._face_to_person.clear()
        self._face_to_model.clear()

        rows = self.db.execute("SELECT id, person_id, embedding, model FROM faces").fetchall()
        by_model = {}
        for fid, pid, blob, model in rows:
            by_model.setdefault(model or UNKNOWN_MODEL, []).append((fid, pid, blob))
        for model, items in by_model.items():
            vecs_list = [np.frombuffer(i[2], dtype=np.float32) for i in items]
            dim = int(vecs_list[0].shape[0])
            if dim not in SUPPORTED_DIMS:
                raise ValueError(f"unsupported stored embedding dimension {dim} for model '{model}'")
            for v in vecs_list:
                if int(v.shape[0]) != dim:
                    raise ValueError(
                        f"inconsistent stored dimensions for model '{model}': expected {dim}, got {int(v.shape[0])}"
                    )
            self._model_dims[model] = dim
            ids = np.array([i[0] for i in items], dtype=np.int64)
            vecs = np.stack(vecs_list)
            self._index(model, dim=dim).add_with_ids(vecs, ids)
            for fid, pid, _ in items:
                self._face_to_person[fid] = pid
                self._face_to_model[fid] = model
        for pid, model in {(i[1], m) for m, items in by_model.items() for i in items}:
            self._update_centroid(pid, model)
        self._invalidate_centroid_cache()

    def _load(self):
        self._rebuild_all()

    def _norm(self, v, model=None):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        dim = int(v.shape[0])
        if dim not in SUPPORTED_DIMS:
            raise ValueError(f"embedding must have 512 or 1024 numbers, got {dim}")
        if model is not None:
            expected = self._model_dims.get(model)
            if expected is not None and dim != expected:
                raise ValueError(f"model '{model}' expects {expected}-d embedding, got {dim}-d")
        norm = float(np.linalg.norm(v))
        if not np.all(np.isfinite(v)) or norm < 1e-8:
            raise ValueError("invalid face embedding (zero-length or non-finite vector)")
        return (v / norm).astype(np.float32)

    # ------------------------------------------------------------------ write
    def add_person(self, name, notes=""):
        with self.lock:
            cur = self.db.execute("INSERT INTO persons(name, notes, created) VALUES (?,?,?)",
                                  (name, notes, time.time()))
            self.db.commit()
            return cur.lastrowid

    def add_photo(self, person_id, embedding, photo_path=None, age=None, model=UNKNOWN_MODEL, quality=1.0):
        with self.lock:
            v = self._norm(embedding, model=model)
            dim = int(v.shape[0])
            if not self.db.execute("SELECT 1 FROM persons WHERE id=?", (person_id,)).fetchone():
                raise KeyError(f"no person with id {person_id}")
            q_val = 1.0 if quality is None else float(quality)
            if not np.isfinite(q_val):
                q_val = 1.0
            cur = self.db.execute(
                "INSERT INTO faces(person_id, photo_path, age, embedding, model, quality) VALUES (?,?,?,?,?,?)",
                (person_id, photo_path, age, v.tobytes(), model, q_val))
            self.db.commit()
            fid = cur.lastrowid
            self._model_dims.setdefault(model, dim)
            self._index(model, dim=dim).add_with_ids(v[None, :], np.array([fid], dtype=np.int64))
            self._face_to_person[fid] = person_id
            self._face_to_model[fid] = model
            self._update_centroid(person_id, model)
            if (
                model in self._calib_face_stats
                or model in self._calib_cent_stats
                or model in self._calib_indexes
                or model in self._calib_centroids
            ):
                self._update_calib_stats_on_add(fid, person_id, model, v, q_val)
            return fid

    def delete_person(self, person_id):
        with self.lock:
            fids = [r[0] for r in self.db.execute("SELECT id FROM faces WHERE person_id=?", (person_id,))]
            if fids:
                ids = np.array(fids, dtype=np.int64)
                for idx in self.indexes.values():
                    idx.remove_ids(ids)
                for model_k, c_idx in list(self._calib_indexes.items()):
                    if c_idx is not None and c_idx is not self.indexes.get(model_k):
                        c_idx.remove_ids(ids)
                for fid in fids:
                    self._face_to_person.pop(fid, None)
                    self._face_to_model.pop(fid, None)
            self.db.execute("DELETE FROM faces WHERE person_id=?", (person_id,))
            self.db.execute("DELETE FROM persons WHERE id=?", (person_id,))
            self.db.commit()
            for key in [k for k in self._centroids if k[0] == person_id]:
                self._centroids.pop(key, None)
            self._invalidate_centroid_cache()
            self._remove_person_calib_stats(person_id)

    # ------------------------------------------------------------------ calibration
    def set_calibration_bundle(self, model, bundle):
        """Registers an in-memory CalibrationBundle for `model` and resets its cohort stats cache."""
        with self.lock:
            expected = self._model_dims.get(model)
            if expected is not None and int(bundle.dim) != int(expected):
                raise ValueError(
                    f"Calibration bundle dim ({bundle.dim}) != stored dim ({expected}) for model '{model}'"
                )
            self._calib_bundles[model] = bundle
            self._calib_face_stats.pop(model, None)
            self._calib_cent_stats.pop(model, None)
            self._calib_indexes.pop(model, None)
            self._calib_centroids.pop(model, None)
            self._calib_proj_hashes.pop(model, None)

    def _get_calibration_bundle(self, model):
        if CalibrationBundle is None or resolve_bundle_path is None:
            raise RuntimeError("calibration module is unavailable")
        bundle = self._calib_bundles.get(model)
        if bundle is None:
            path = resolve_bundle_path(model)
            bundle = CalibrationBundle.load(path)
            self._calib_bundles[model] = bundle
        expected = self._model_dims.get(model)
        if expected is not None and int(bundle.dim) != int(expected):
            raise ValueError(
                f"Calibration bundle dim ({bundle.dim}) != stored dim ({expected}) for model '{model}'"
            )
        return bundle

    def calibrated_ready(self, model=UNKNOWN_MODEL) -> bool:
        """True if a calibration bundle is loaded or present on disk for `model`."""
        with self.lock:
            if model in self._calib_bundles:
                return True
            if resolve_bundle_path is None:
                return False
            return os.path.isfile(resolve_bundle_path(model))

    def _ensure_cohort_stats(self, model, bundle):
        f_stats = self._calib_face_stats.get(model)
        c_stats = self._calib_cent_stats.get(model)
        chash = bundle.cohort_hash
        phash = getattr(bundle, "proj_hash", "none")
        has_proj = getattr(bundle, "proj_basis", None) is not None

        if not has_proj:
            cent_pids, cent_vecs = self._model_centroids(model)
            self._calib_indexes[model] = self.indexes.get(model)
            self._calib_centroids[model] = (cent_pids, cent_vecs)
            self._calib_proj_hashes[model] = phash

            if f_stats is None or f_stats.get("cohort_hash") != chash or f_stats.get("proj_hash") != phash:
                rows = self.db.execute(
                    "SELECT id, person_id, embedding, quality FROM faces WHERE model=? ORDER BY id",
                    (model,),
                ).fetchall()
                if rows:
                    fids = np.array([r[0] for r in rows], dtype=np.int64)
                    pids = np.array([r[1] for r in rows], dtype=np.int64)
                    vecs = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows]).astype(np.float32)
                    quals = np.array(
                        [
                            min(max(1.0 if r[3] is None or not np.isfinite(r[3]) else float(r[3]), 0.25), 1.0)
                            for r in rows
                        ],
                        dtype=np.float32,
                    )
                    mu_f, sigma_f = bundle.asnorm.enroll_stats(vecs)
                    fid_to_idx = {int(fid): idx for idx, fid in enumerate(fids.tolist())}
                else:
                    fids = np.empty((0,), dtype=np.int64)
                    pids = np.empty((0,), dtype=np.int64)
                    quals = np.empty((0,), dtype=np.float32)
                    mu_f = np.empty((0,), dtype=np.float32)
                    sigma_f = np.empty((0,), dtype=np.float32)
                    fid_to_idx = {}

                f_stats = {
                    "cohort_hash": chash,
                    "proj_hash": phash,
                    "fids": fids,
                    "pids": pids,
                    "fid_to_idx": fid_to_idx,
                    "qualities": quals,
                    "mu": mu_f,
                    "sigma": sigma_f,
                }
                self._calib_face_stats[model] = f_stats

            if (
                c_stats is None
                or c_stats.get("cohort_hash") != chash
                or c_stats.get("proj_hash") != phash
                or not np.array_equal(c_stats["pids"], cent_pids)
            ):
                if len(cent_pids) > 0:
                    max_q_by_pid = {}
                    for pid_val, q_val in zip(f_stats["pids"].tolist(), f_stats["qualities"].tolist()):
                        prev = max_q_by_pid.get(pid_val)
                        if prev is None or q_val > prev:
                            max_q_by_pid[pid_val] = q_val
                    c_quals = np.array(
                        [max_q_by_pid.get(int(pid_val), 1.0) for pid_val in cent_pids.tolist()],
                        dtype=np.float32,
                    )
                    mu_c, sigma_c = bundle.asnorm.enroll_stats(cent_vecs)
                else:
                    c_quals = np.empty((0,), dtype=np.float32)
                    mu_c = np.empty((0,), dtype=np.float32)
                    sigma_c = np.empty((0,), dtype=np.float32)

                c_stats = {
                    "cohort_hash": chash,
                    "proj_hash": phash,
                    "pids": cent_pids,
                    "qualities": c_quals,
                    "mu": mu_c,
                    "sigma": sigma_c,
                }
                self._calib_cent_stats[model] = c_stats

            return f_stats, c_stats

        need_rebuild_proj = (
            self._calib_proj_hashes.get(model) != phash
            or model not in self._calib_indexes
            or model not in self._calib_centroids
            or f_stats is None
            or f_stats.get("cohort_hash") != chash
            or f_stats.get("proj_hash") != phash
        )
        if need_rebuild_proj:
            rows = self.db.execute(
                "SELECT id, person_id, age, embedding, quality FROM faces WHERE model=? ORDER BY id",
                (model,),
            ).fetchall()
            d = int(bundle.dim)
            c_idx = faiss.IndexIDMap(faiss.IndexFlatIP(d))
            if rows:
                fids = np.array([r[0] for r in rows], dtype=np.int64)
                pids = np.array([r[1] for r in rows], dtype=np.int64)
                ages = [r[2] for r in rows]
                raw_vecs = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows]).astype(np.float32)
                quals = np.array(
                    [
                        min(max(1.0 if r[4] is None or not np.isfinite(r[4]) else float(r[4]), 0.25), 1.0)
                        for r in rows
                    ],
                    dtype=np.float32,
                )
                proj_vecs = bundle.apply_subspace(raw_vecs)
                c_idx.add_with_ids(proj_vecs, fids)
                mu_f, sigma_f = bundle.asnorm.enroll_stats(proj_vecs)
                fid_to_idx = {int(fid): idx for idx, fid in enumerate(fids.tolist())}

                by_pid = {}
                for row_i, pid_val in enumerate(pids.tolist()):
                    by_pid.setdefault(pid_val, []).append(row_i)
                c_pid_list = sorted(by_pid.keys())
                c_vec_list = []
                c_qual_list = []
                for pid_val in c_pid_list:
                    idxs = by_pid[pid_val]
                    p_vecs = proj_vecs[idxs]
                    p_ages = [ages[i] for i in idxs]
                    p_quals = [float(quals[i]) for i in idxs]
                    c_vec_list.append(self._compute_weighted_centroid(p_vecs, p_ages, p_quals))
                    c_qual_list.append(max(p_quals))
                cent_pids = np.array(c_pid_list, dtype=np.int64)
                cent_vecs = np.stack(c_vec_list).astype(np.float32)
                c_quals = np.array(c_qual_list, dtype=np.float32)
            else:
                fids = np.empty((0,), dtype=np.int64)
                pids = np.empty((0,), dtype=np.int64)
                quals = np.empty((0,), dtype=np.float32)
                mu_f = np.empty((0,), dtype=np.float32)
                sigma_f = np.empty((0,), dtype=np.float32)
                fid_to_idx = {}
                cent_pids = np.empty((0,), dtype=np.int64)
                cent_vecs = np.empty((0, d), dtype=np.float32)
                c_quals = np.empty((0,), dtype=np.float32)

            self._calib_indexes[model] = c_idx
            self._calib_centroids[model] = (cent_pids, cent_vecs)
            self._calib_proj_hashes[model] = phash

            f_stats = {
                "cohort_hash": chash,
                "proj_hash": phash,
                "fids": fids,
                "pids": pids,
                "fid_to_idx": fid_to_idx,
                "qualities": quals,
                "mu": mu_f,
                "sigma": sigma_f,
            }
            self._calib_face_stats[model] = f_stats

            if len(cent_pids) > 0:
                mu_c, sigma_c = bundle.asnorm.enroll_stats(cent_vecs)
            else:
                mu_c = np.empty((0,), dtype=np.float32)
                sigma_c = np.empty((0,), dtype=np.float32)
            c_stats = {
                "cohort_hash": chash,
                "proj_hash": phash,
                "pids": cent_pids,
                "qualities": c_quals,
                "mu": mu_c,
                "sigma": sigma_c,
            }
            self._calib_cent_stats[model] = c_stats
        elif (
            c_stats is None
            or c_stats.get("cohort_hash") != chash
            or c_stats.get("proj_hash") != phash
        ):
            cent_pids, cent_vecs = self._calib_centroids[model]
            if len(cent_pids) > 0:
                max_q_by_pid = {}
                for pid_val, q_val in zip(f_stats["pids"].tolist(), f_stats["qualities"].tolist()):
                    prev = max_q_by_pid.get(pid_val)
                    if prev is None or q_val > prev:
                        max_q_by_pid[pid_val] = q_val
                c_quals = np.array(
                    [max_q_by_pid.get(int(pid_val), 1.0) for pid_val in cent_pids.tolist()],
                    dtype=np.float32,
                )
                mu_c, sigma_c = bundle.asnorm.enroll_stats(cent_vecs)
            else:
                c_quals = np.empty((0,), dtype=np.float32)
                mu_c = np.empty((0,), dtype=np.float32)
                sigma_c = np.empty((0,), dtype=np.float32)
            c_stats = {
                "cohort_hash": chash,
                "proj_hash": phash,
                "pids": cent_pids,
                "qualities": c_quals,
                "mu": mu_c,
                "sigma": sigma_c,
            }
            self._calib_cent_stats[model] = c_stats

        return f_stats, c_stats

    def _update_calib_stats_on_add(self, fid, person_id, model, vec, quality):
        bundle = self._calib_bundles.get(model)
        phash = getattr(bundle, "proj_hash", "none") if bundle is not None else None
        if bundle is None or self._calib_proj_hashes.get(model) != phash:
            self._calib_face_stats.pop(model, None)
            self._calib_cent_stats.pop(model, None)
            self._calib_indexes.pop(model, None)
            self._calib_centroids.pop(model, None)
            self._calib_proj_hashes.pop(model, None)
            return

        has_proj = getattr(bundle, "proj_basis", None) is not None
        vec_eff = bundle.apply_subspace(vec[None, :])[0] if has_proj else vec
        q_clamped = float(min(max(1.0 if quality is None or not np.isfinite(quality) else float(quality), 0.25), 1.0))

        if has_proj:
            c_idx = self._calib_indexes.get(model)
            if c_idx is not None:
                c_idx.add_with_ids(vec_eff[None, :], np.array([fid], dtype=np.int64))
            rows = self.db.execute(
                "SELECT embedding, age, quality FROM faces WHERE person_id=? AND model=?",
                (person_id, model),
            ).fetchall()
            if rows:
                raw_p = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows]).astype(np.float32)
                proj_p = bundle.apply_subspace(raw_p)
                cent_vec = self._compute_weighted_centroid(proj_p, [r[1] for r in rows], [r[2] for r in rows])
            else:
                cent_vec = None

            if model in self._calib_centroids and cent_vec is not None:
                pids_c, vecs_c = self._calib_centroids[model]
                matches_c = np.flatnonzero(pids_c == person_id)
                if matches_c.size > 0:
                    vecs_c[int(matches_c[0])] = cent_vec
                else:
                    pids_c = np.append(pids_c, np.int64(person_id))
                    vecs_c = np.vstack([vecs_c, cent_vec[None, :]]) if vecs_c.shape[0] > 0 else cent_vec[None, :]
                    self._calib_centroids[model] = (pids_c, vecs_c)
        else:
            cent_vec = self._centroids.get((person_id, model))
            self._calib_indexes[model] = self.indexes.get(model)
            self._calib_centroids[model] = self._model_centroids(model)

        f_stats = self._calib_face_stats.get(model)
        if f_stats is not None and f_stats.get("cohort_hash") == bundle.cohort_hash and f_stats.get("proj_hash") == phash:
            mu_1, sigma_1 = bundle.asnorm.enroll_stats(vec_eff[None, :])
            new_idx = len(f_stats["fids"])
            f_stats["fids"] = np.append(f_stats["fids"], np.int64(fid))
            f_stats["pids"] = np.append(f_stats["pids"], np.int64(person_id))
            f_stats["qualities"] = np.append(f_stats["qualities"], np.float32(q_clamped))
            f_stats["mu"] = np.append(f_stats["mu"], mu_1[0])
            f_stats["sigma"] = np.append(f_stats["sigma"], sigma_1[0])
            f_stats["fid_to_idx"][int(fid)] = new_idx
        else:
            self._calib_face_stats.pop(model, None)

        c_stats = self._calib_cent_stats.get(model)
        if (
            c_stats is not None
            and c_stats.get("cohort_hash") == bundle.cohort_hash
            and c_stats.get("proj_hash") == phash
            and cent_vec is not None
        ):
            mu_c1, sigma_c1 = bundle.asnorm.enroll_stats(cent_vec[None, :])
            matches = np.flatnonzero(c_stats["pids"] == person_id)
            if matches.size > 0:
                idx = int(matches[0])
                c_stats["qualities"][idx] = np.float32(max(float(c_stats["qualities"][idx]), q_clamped))
                c_stats["mu"][idx] = mu_c1[0]
                c_stats["sigma"][idx] = sigma_c1[0]
            else:
                c_stats["pids"] = np.append(c_stats["pids"], np.int64(person_id))
                c_stats["qualities"] = np.append(c_stats["qualities"], np.float32(q_clamped))
                c_stats["mu"] = np.append(c_stats["mu"], mu_c1[0])
                c_stats["sigma"] = np.append(c_stats["sigma"], sigma_c1[0])
        else:
            self._calib_cent_stats.pop(model, None)

    def _remove_person_calib_stats(self, person_id):
        for model, f_stats in list(self._calib_face_stats.items()):
            mask = f_stats["pids"] != person_id
            if not np.all(mask):
                new_fids = f_stats["fids"][mask]
                f_stats["fids"] = new_fids
                f_stats["pids"] = f_stats["pids"][mask]
                f_stats["qualities"] = f_stats["qualities"][mask]
                f_stats["mu"] = f_stats["mu"][mask]
                f_stats["sigma"] = f_stats["sigma"][mask]
                f_stats["fid_to_idx"] = {int(f): i for i, f in enumerate(new_fids.tolist())}
        for model, c_stats in list(self._calib_cent_stats.items()):
            mask = c_stats["pids"] != person_id
            if not np.all(mask):
                c_stats["pids"] = c_stats["pids"][mask]
                c_stats["qualities"] = c_stats["qualities"][mask]
                c_stats["mu"] = c_stats["mu"][mask]
                c_stats["sigma"] = c_stats["sigma"][mask]
        for model in list(self._calib_centroids.keys()):
            if self._calib_proj_hashes.get(model, "none") == "none":
                self._calib_centroids[model] = self._model_centroids(model)
            else:
                pids_c, vecs_c = self._calib_centroids[model]
                mask_c = pids_c != person_id
                if not np.all(mask_c):
                    self._calib_centroids[model] = (pids_c[mask_c], vecs_c[mask_c])

    # ------------------------------------------------------------------ read
    def _fetch_person_info(self, top_k, model):
        """Fetches representative face and person metadata for selected top-k candidates."""
        needed_fids = [c["faiss_fid"] for c in top_k if c["faiss_fid"] is not None]
        needed_pids_no_fid = [c["person_id"] for c in top_k if c["faiss_fid"] is None]

        info_by_pid = {}
        if needed_fids:
            marks = ",".join("?" * len(needed_fids))
            for r in self.db.execute(
                f"SELECT f.id, f.person_id, f.photo_path, f.age, p.name, p.notes "
                f"FROM faces f JOIN persons p ON p.id = f.person_id WHERE f.id IN ({marks})",
                needed_fids,
            ):
                info_by_pid[r[1]] = (r[0], r[2], r[3], r[4], r[5])

        if needed_pids_no_fid:
            marks = ",".join("?" * len(needed_pids_no_fid))
            for r in self.db.execute(
                f"SELECT f.id, f.person_id, f.photo_path, f.age, p.name, p.notes "
                f"FROM faces f JOIN persons p ON p.id = f.person_id "
                f"WHERE f.person_id IN ({marks}) AND f.model=? ORDER BY f.id DESC",
                [*needed_pids_no_fid, model],
            ):
                pid = r[1]
                if pid not in info_by_pid:
                    info_by_pid[pid] = (r[0], r[2], r[3], r[4], r[5])
        return info_by_pid

    def search(self, embedding, k=10, min_sim=-1.0, model=UNKNOWN_MODEL,
               probe_quality=1.0, age_gap=None):
        """Ranked people for one engine's vectors only. Pass model=engine.model_name."""
        with self.lock:
            q = self._norm(embedding, model=model)
            index = self.indexes.get(model)
            if k <= 0 or index is None or index.ntotal == 0:
                return []
            if self.calibrated_ready(model):
                return self._search_calibrated(
                    q, index, k=k, min_sim=min_sim, model=model,
                    probe_quality=probe_quality, age_gap=age_gap,
                )
            return self._search_cosine_fallback(
                q, index, k=k, min_sim=min_sim, model=model,
            )

    def _search_cosine_fallback(self, q, index, k=10, min_sim=-1.0, model=UNKNOWN_MODEL):
        """Cosine fallback search for uncalibrated engines: max(best_photo_sim, centroid_sim)."""
        # A person can own many photos, and only the photos inside this window get a per-photo
        # score, so the window has to stay deep compared to k. Brute force over 500 vectors is
        # still sub-millisecond, so depth costs nothing here.
        kk = min(index.ntotal, max(int(k) * 50, 500))
        sims, fids = index.search(q[None, :], kk)
        pairs = [(int(f), float(s)) for f, s in zip(fids[0], sims[0]) if f != -1]

        missing_fids = [f for f, _ in pairs if f not in self._face_to_person]
        if missing_fids:
            marks = ",".join("?" * len(missing_fids))
            for r in self.db.execute(f"SELECT id, person_id, model FROM faces WHERE id IN ({marks})", missing_fids):
                self._face_to_person[r[0]] = r[1]
                self._face_to_model[r[0]] = r[2] or UNKNOWN_MODEL

        faiss_person_best = {}
        for fid, sim in pairs:
            pid = self._face_to_person.get(fid)
            if pid is not None and pid not in faiss_person_best:
                faiss_person_best[pid] = (sim, fid)

        centroid_pids, centroid_matrix = self._model_centroids(model)
        centroid_sims = ({int(p): float(s) for p, s in zip(centroid_pids, np.dot(centroid_matrix, q))}
                         if len(centroid_pids) else {})

        # Candidate order: FAISS hits first, then everyone else with a centroid for this model
        candidate_pids = []
        seen_pids = set()
        for fid, _ in pairs:
            pid = self._face_to_person.get(fid)
            if pid is not None and pid not in seen_pids:
                seen_pids.add(pid)
                candidate_pids.append(pid)
        for pid in centroid_pids:
            pid = int(pid)
            if pid not in seen_pids:
                seen_pids.add(pid)
                candidate_pids.append(pid)

        candidates = []
        for pid in candidate_pids:
            best_photo_sim, best_fid = faiss_person_best.get(pid, (-1.0, None))
            has_centroid = pid in centroid_sims
            cent_sim = centroid_sims[pid] if has_centroid else -1.0
            if has_centroid and cent_sim >= best_photo_sim:
                sim = cent_sim
                match_source = "centroid"
            else:
                sim = best_photo_sim
                match_source = "photo"
            if sim >= min_sim:
                tier = "STRONG" if sim >= STRONG else ("possible" if sim >= POSSIBLE else "weak")
                candidates.append({
                    "person_id": pid,
                    "similarity": round(sim, 4),
                    "match_source": match_source,
                    "centroid_similarity": None if not has_centroid else round(cent_sim, 4),
                    "best_photo_similarity": None if best_fid is None else round(best_photo_sim, 4),
                    "tier": tier,
                    "faiss_fid": best_fid,
                })

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        top_k = candidates[:k]
        if not top_k:
            return []

        info_by_pid = self._fetch_person_info(top_k, model)
        result = []
        for c in top_k:
            pid = c["person_id"]
            if pid in info_by_pid:
                fid, path, age, name, notes = info_by_pid[pid]
                result.append({
                    "face_id": fid,
                    "person_id": pid,
                    "name": name,
                    "notes": notes,
                    "similarity": c["similarity"],
                    "match_source": c["match_source"],
                    "centroid_similarity": c["centroid_similarity"],
                    "best_photo_similarity": c["best_photo_similarity"],
                    "tier": c["tier"],
                    "photo_path": path,
                    "age": age,
                })
        return result

    def _search_calibrated(self, q, index, k=10, min_sim=-1.0, model=UNKNOWN_MODEL,
                           probe_quality=1.0, age_gap=None):
        """4-layer calibrated 1:N search: external AS-Norm -> QMF LLR -> GPD/Conformal tiers -> z_bg."""
        bundle = self._get_calibration_bundle(model)
        face_stats, cent_stats = self._ensure_cohort_stats(model, bundle)
        q_probe = float(
            np.clip(1.0 if probe_quality is None or not np.isfinite(probe_quality) else float(probe_quality), 0.25, 1.0)
        )
        q_eff = bundle.apply_subspace(q[None, :])[0] if getattr(bundle, "proj_basis", None) is not None else q
        probe_stats = bundle.asnorm.probe_stats(q_eff)

        c_index = self._calib_indexes.get(model)
        if c_index is None:
            c_index = index
        kk_photo = c_index.ntotal if c_index.ntotal <= 2000 else min(c_index.ntotal, max(int(k) * 200, 2000))
        sims, fids = c_index.search(q_eff[None, :], kk_photo)
        fid_to_idx = face_stats["fid_to_idx"]
        valid_pairs = [(int(f), float(s), fid_to_idx[int(f)])
                       for f, s in zip(fids[0], sims[0])
                       if f != -1 and int(f) in fid_to_idx]

        best_photo_by_pid = {}
        if valid_pairs:
            p_idxs = np.array([vp[2] for vp in valid_pairs], dtype=np.int64)
            raw_p_arr = np.array([vp[1] for vp in valid_pairs], dtype=np.float32)
            z_p = bundle.asnorm.score_batch(
                q_eff,
                cand_mu=face_stats["mu"][p_idxs],
                cand_sigma=face_stats["sigma"][p_idxs],
                raw_sims=raw_p_arr,
                probe_stats=probe_stats,
            )
            q_min_p = np.minimum(q_probe, face_stats["qualities"][p_idxs])
            llr_p = bundle.qmf.predict_llr(z_p, raw_p_arr, q_min=q_min_p, age_gap=age_gap)
            pids_p = face_stats["pids"][p_idxs].tolist()
            quals_p = face_stats["qualities"][p_idxs].tolist()
            for (fid_val, raw_val, _), pid_val, llr_val, q_val in zip(valid_pairs, pids_p, llr_p.tolist(), quals_p):
                prev = best_photo_by_pid.get(pid_val)
                if prev is None or llr_val > prev[0]:
                    best_photo_by_pid[pid_val] = (float(llr_val), float(raw_val), float(q_val), int(fid_val))

        if model in self._calib_centroids:
            cent_pids, cent_vecs = self._calib_centroids[model]
        else:
            cent_pids, cent_vecs = self._model_centroids(model)
        cent_by_pid = {}
        if len(cent_pids) > 0:
            raw_c_arr = (cent_vecs @ q_eff).astype(np.float32)
            z_c = bundle.asnorm.score_batch(
                q_eff,
                cand_mu=cent_stats["mu"],
                cand_sigma=cent_stats["sigma"],
                raw_sims=raw_c_arr,
                probe_stats=probe_stats,
            )
            q_min_c = np.minimum(q_probe, cent_stats["qualities"])
            llr_c = bundle.qmf.predict_llr(z_c, raw_c_arr, q_min=q_min_c, age_gap=age_gap)
            for pid_val, llr_val, raw_val, q_val in zip(
                cent_pids.tolist(), llr_c.tolist(), raw_c_arr.tolist(), cent_stats["qualities"].tolist()
            ):
                cent_by_pid[pid_val] = (float(llr_val), float(raw_val), float(q_val))

        all_pids = list(cent_pids.tolist())
        seen = set(all_pids)
        for pid_val in best_photo_by_pid:
            if pid_val not in seen:
                seen.add(pid_val)
                all_pids.append(pid_val)

        if not all_pids:
            return []

        w_h = float(np.clip(getattr(bundle, "photo_hybrid_weight", 1.0), 0.0, 1.0))
        raw_candidates = []
        for pid in all_pids:
            c_info = cent_by_pid.get(pid)
            p_info = best_photo_by_pid.get(pid)
            if c_info is not None and p_info is not None:
                c_llr = c_info[0]
                p_llr = p_info[0]
                best_llr = c_llr + w_h * max(0.0, p_llr - c_llr)
                if c_llr >= p_llr:
                    match_source = "centroid"
                    eff_q_min = min(q_probe, c_info[2])
                else:
                    match_source = "photo"
                    eff_q_min = min(q_probe, p_info[2])
            elif c_info is not None:
                best_llr = c_info[0]
                match_source = "centroid"
                eff_q_min = min(q_probe, c_info[2])
            else:
                best_llr = p_info[0]
                match_source = "photo"
                eff_q_min = min(q_probe, p_info[2])
            raw_candidates.append((best_llr, pid, match_source, eff_q_min, c_info, p_info))

        raw_candidates.sort(key=lambda x: x[0], reverse=True)
        bg_info = compute_bg_zscore([c[0] for c in raw_candidates[:50]])
        mu_bg = bg_info["mu_bg"]
        sigma_bg = bg_info["sigma_bg"]
        margin_12 = bg_info["margin_12"]
        top1_flags = list(bg_info["advisory_flags"])

        top_k = []
        for idx_cand, cand in enumerate(raw_candidates):
            best_llr, pid, match_source, eff_q_min, c_info, p_info = cand
            cal_sim = round(float(bundle.qmf.llr_to_cosine(best_llr)), 4)
            if cal_sim < min_sim:
                break
            cal_cent_cos = None if c_info is None else round(float(bundle.qmf.llr_to_cosine(c_info[0])), 4)
            cal_photo_cos = None if p_info is None else round(float(bundle.qmf.llr_to_cosine(p_info[0])), 4)
            raw_cent_sim = None if c_info is None else round(float(c_info[1]), 4)
            raw_photo_sim = None if p_info is None else round(float(p_info[1]), 4)
            top_k.append({
                "person_id": pid,
                "best_llr": best_llr,
                "similarity": cal_sim,
                "match_source": match_source,
                "centroid_similarity": cal_cent_cos,
                "best_photo_similarity": cal_photo_cos,
                "raw_centroid_similarity": raw_cent_sim,
                "raw_best_photo_similarity": raw_photo_sim,
                "eff_q_min": eff_q_min,
                "faiss_fid": None if p_info is None else p_info[3],
                "_is_rank0": idx_cand == 0,
            })
            if len(top_k) >= k:
                break

        if not top_k:
            return []

        n_gallery = max(1, len(cent_pids))
        info_by_pid = self._fetch_person_info(top_k, model)

        result = []
        for rank_idx, c in enumerate(top_k):
            pid = c["person_id"]
            if pid not in info_by_pid:
                continue
            fid, path, age, name, notes = info_by_pid[pid]
            llr_val = c["best_llr"]
            tier = bundle.openset.classify(llr_val, n_gallery)
            p_nonmate = bundle.openset.p_nonmate(llr_val, n_gallery)
            match_prob = bundle.qmf.predict_prob(llr_val)
            cand_z_bg = None if mu_bg is None else round(float((llr_val - mu_bg) / sigma_bg), 3)
            flags = list(top1_flags) if rank_idx == 0 and c["_is_rank0"] else []
            if c["eff_q_min"] < 0.55:
                flags.append("low_quality")

            result.append({
                "face_id": fid,
                "person_id": pid,
                "name": name,
                "notes": notes,
                "similarity": c["similarity"],
                "match_source": c["match_source"],
                "centroid_similarity": c["centroid_similarity"],
                "best_photo_similarity": c["best_photo_similarity"],
                "raw_centroid_similarity": c["raw_centroid_similarity"],
                "raw_best_photo_similarity": c["raw_best_photo_similarity"],
                "tier": tier,
                "calibrated_llr": round(float(llr_val), 4),
                "match_prob": round(float(match_prob), 4),
                "p_nonmate": round(float(p_nonmate), 6),
                "z_bg": cand_z_bg,
                "margin_12": None if margin_12 is None else round(float(margin_12), 4),
                "advisory_flags": flags,
                "photo_path": path,
                "age": age,
            })
        return result

    def stats(self):
        with self.lock:
            n_people = self.db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            photos = sum(idx.ntotal for idx in self.indexes.values())
            models = {}
            for r in self.db.execute("SELECT model, COUNT(*) FROM faces GROUP BY model"):
                models[r[0] or UNKNOWN_MODEL] = int(r[1])
            return {"people": int(n_people), "photos": int(photos), "models": models}
