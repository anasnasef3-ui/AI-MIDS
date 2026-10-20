"""
store.py - where people and their face vectors live.

SQLite (one file) is the single source of truth. FAISS is rebuilt from it when the program
starts, so the two can never get out of sync. Fine for tens of thousands of photos.

    from store import FaceStore
    store = FaceStore("aimids.db")
    pid = store.add_person("Ali Hassan", notes="missing since 2015, last seen Cairo")
    store.add_photo(pid, embedding, photo_path="photos/ali_2012.jpg", age=9)
    hits = store.search(query_embedding, k=10)
    # -> [{"person_id", "name", "notes", "similarity", "tier", "photo_path", "age"}, ...] best first

Search returns ONE entry per person (their best photo), at most k people, only those at or
above min_sim. Multi-photo centroid fusion computes fused identity representations across all
registered photos per person.
tier is "STRONG" (>= 0.36), "possible" (>= 0.26), or "weak" (< 0.26).
"""
import sqlite3
import threading
import time
import numpy as np
import faiss

DIM = 512
STRONG, POSSIBLE = 0.36, 0.26


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
                embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
        """)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
        self._centroids = {}
        self._load()

    # ------------------------------------------------------------------ internals
    def _update_centroid(self, person_id):
        rows = self.db.execute("SELECT embedding FROM faces WHERE person_id=?", (person_id,)).fetchall()
        if not rows:
            self._centroids.pop(person_id, None)
            return
        vecs = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
        mean_v = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_v)
        if norm > 0:
            self._centroids[person_id] = (mean_v / norm).astype(np.float32)
        else:
            self._centroids[person_id] = mean_v.astype(np.float32)

    def _load(self):
        rows = self.db.execute("SELECT id, embedding FROM faces").fetchall()
        if rows:
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            self.index.add_with_ids(vecs, ids)
        pids = [r[0] for r in self.db.execute("SELECT DISTINCT person_id FROM faces").fetchall()]
        for pid in pids:
            self._update_centroid(pid)

    @staticmethod
    def _norm(v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] != DIM:
            raise ValueError(f"embedding must have {DIM} numbers, got {v.shape[0]}")
        return v / np.linalg.norm(v)

    # ------------------------------------------------------------------ write
    def add_person(self, name, notes=""):
        with self.lock:
            cur = self.db.execute("INSERT INTO persons(name, notes, created) VALUES (?,?,?)",
                                  (name, notes, time.time()))
            self.db.commit()
            return cur.lastrowid

    def add_photo(self, person_id, embedding, photo_path=None, age=None):
        v = self._norm(embedding)
        with self.lock:
            if not self.db.execute("SELECT 1 FROM persons WHERE id=?", (person_id,)).fetchone():
                raise KeyError(f"no person with id {person_id}")
            cur = self.db.execute("INSERT INTO faces(person_id, photo_path, age, embedding) VALUES (?,?,?,?)",
                                  (person_id, photo_path, age, v.tobytes()))
            self.db.commit()
            fid = cur.lastrowid
            self.index.add_with_ids(v[None, :], np.array([fid], dtype=np.int64))
            self._update_centroid(person_id)
            return fid

    def delete_person(self, person_id):
        with self.lock:
            fids = [r[0] for r in self.db.execute("SELECT id FROM faces WHERE person_id=?", (person_id,))]
            if fids:
                self.index.remove_ids(np.array(fids, dtype=np.int64))
            self.db.execute("DELETE FROM faces WHERE person_id=?", (person_id,))
            self.db.execute("DELETE FROM persons WHERE id=?", (person_id,))
            self.db.commit()
            self._centroids.pop(person_id, None)

    # ------------------------------------------------------------------ read
    def search(self, embedding, k=10, min_sim=-1.0):
        q = self._norm(embedding)
        with self.lock:
            if self.index.ntotal == 0:
                return []
            kk = min(self.index.ntotal, max(k * 5, 50))  # several photos per person -> look deeper, then dedupe
            sims, fids = self.index.search(q[None, :], kk)
            pairs = [(int(f), float(s)) for f, s in zip(fids[0], sims[0]) if f != -1]
            if pairs:
                marks = ",".join("?" * len(pairs))
                info = {r[0]: r for r in self.db.execute(
                    f"SELECT f.id, f.person_id, f.photo_path, f.age, p.name, p.notes "
                    f"FROM faces f JOIN persons p ON p.id = f.person_id WHERE f.id IN ({marks})",
                    [f for f, _ in pairs])}
            else:
                info = {}

            person_best = {}
            for fid, sim in pairs:  # pairs are sorted best-first, so the first hit per person is their best
                if fid in info:
                    fid_, pid, path, age, name, notes = info[fid]
                    if pid not in person_best:
                        person_best[pid] = {
                            "face_id": fid,
                            "person_id": pid,
                            "name": name,
                            "notes": notes,
                            "photo_path": path,
                            "age": age,
                            "best_photo_sim": sim,
                        }

            # Evaluate centroid similarity for all persons
            centroid_sims = {pid: float(np.dot(q, c_vec)) for pid, c_vec in self._centroids.items()}

            # Any person whose centroid similarity >= min_sim not already in pairs
            missing_pids = [pid for pid, c_sim in centroid_sims.items() if pid not in person_best and c_sim >= min_sim]
            if missing_pids:
                m_marks = ",".join("?" * len(missing_pids))
                rows = self.db.execute(
                    f"SELECT f.id, f.person_id, f.photo_path, f.age, p.name, p.notes "
                    f"FROM faces f JOIN persons p ON p.id = f.person_id "
                    f"WHERE f.person_id IN ({m_marks}) ORDER BY f.id DESC", missing_pids
                ).fetchall()
                for r in rows:
                    fid_, pid, path, age, name, notes = r
                    if pid not in person_best:
                        person_best[pid] = {
                            "face_id": fid_,
                            "person_id": pid,
                            "name": name,
                            "notes": notes,
                            "photo_path": path,
                            "age": age,
                            "best_photo_sim": -1.0,
                        }

            candidates = []
            for pid, pdata in person_best.items():
                c_sim = centroid_sims.get(pid, -1.0)
                sim = max(pdata["best_photo_sim"], c_sim)
                if sim >= min_sim:
                    tier = "STRONG" if sim >= STRONG else ("possible" if sim >= POSSIBLE else "weak")
                    candidates.append({
                        "face_id": pdata["face_id"],
                        "person_id": pid,
                        "name": pdata["name"],
                        "notes": pdata["notes"],
                        "similarity": round(sim, 4),
                        "tier": tier,
                        "photo_path": pdata["photo_path"],
                        "age": pdata["age"],
                    })

            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            return candidates[:k]

    def stats(self):
        with self.lock:
            n_people = self.db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            return {"people": n_people, "photos": int(self.index.ntotal)}
