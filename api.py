"""
api.py - FastAPI Backend for AI-MIDS (Missing Persons Search & ID Verification).

Endpoints:
  GET  /                            - System status & database stats
  GET  /api/stats                   - Database statistics (people and photos count)
  GET  /api/persons                 - List all registered persons
  POST /api/persons                 - Register a new person (name, notes)
  GET  /api/persons/{person_id}     - Get details of a specific person
  DELETE /api/persons/{person_id}   - Delete a person and their embeddings
  POST /api/persons/{person_id}/photos - Register a photo for a person (quality checked)
  POST /api/search                  - Search missing persons using an uploaded photo (single or group)
  POST /api/verify-id               - Verify Egyptian ID card photo against a selfie with liveness
  POST /api/verify-id/check-framing - Lightweight SCRFD face framing check for live webcam guide
"""
import os
import sys

# Limit OpenMP/MKL threads before importing numpy/faiss to prevent ~1.1+ GB Windows virtual commit bloat.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import uuid
import time
import base64
import threading
from typing import Optional, List
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# Setup directory paths to import modules
HERE = os.path.dirname(os.path.abspath(__file__))
ARC_DIR = os.path.join(HERE, "Arc face")
OCR_DIR = os.path.join(HERE, "OCR_Egyptian_ID-main")

sys.path.insert(0, OCR_DIR)
sys.path.insert(0, ARC_DIR)

from face_engine import FaceEngine, ensure_cuda_dll_paths
from face_engine_antelopev2 import FaceEngineAntelopeV2
from face_engine_adaface_ft import FaceEngineAdaFaceFT
from face_engine_custom import FaceEngineCrossAge
from store import FaceStore
from calibration import extract_face_quality
from photo_quality import check_photo, draw_faces
import utils
from id_verify import verify_id
from liveness import LivenessChecker, framing, FACE_MAX

# Global instances initialized on startup
ENGINE_NAME = os.environ.get("AIMIDS_ENGINE", "aimids_crossage").lower()
if ENGINE_NAME == "adaface_ft":
    ENGINE_NAME = "aimids_ft"
USE_TTA = os.environ.get("AIMIDS_TTA", "1").lower() in ("1", "true", "yes")
USE_GPU = os.environ.get("AIMIDS_USE_GPU", "1").lower() in ("1", "true", "yes", "auto")

engine = None
store: Optional[FaceStore] = None
liveness: Optional[LivenessChecker] = None
verify_lock = threading.Lock()

DB_PATH = os.environ.get("AIMIDS_DB_PATH", os.path.join(ARC_DIR, "aimids.db"))
UPLOAD_DIR = os.path.join(HERE, "uploads")
PREVIEWS_DIR = os.path.join(UPLOAD_DIR, "previews")
DEMO_PROBES_DIR = os.path.join(HERE, "demo_probes")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREVIEWS_DIR, exist_ok=True)


def _cleanup_old_previews(max_age_seconds: int = 3600):
    """Deletes preview images older than max_age_seconds (default 1 hour)."""
    try:
        now = time.time()
        if os.path.isdir(PREVIEWS_DIR):
            for fname in os.listdir(PREVIEWS_DIR):
                if fname.startswith("."):
                    continue
                fpath = os.path.join(PREVIEWS_DIR, fname)
                if os.path.isfile(fpath):
                    if now - os.path.getmtime(fpath) > max_age_seconds:
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
    except Exception:
        pass


def _get_liveness() -> LivenessChecker:
    """Lazily initializes LivenessChecker on first /api/verify-id call (caller holds verify_lock)."""
    global liveness
    if liveness is None:
        liveness = LivenessChecker()
    return liveness


def _init_engine():
    """Initializes the configured face engine with automatic fallback to antelopev2 -> buffalo_l."""
    global ENGINE_NAME
    if USE_GPU:
        ensure_cuda_dll_paths()
    if ENGINE_NAME == "aimids_crossage":
        try:
            print(f">> [AI-MIDS API] Initializing FaceEngineCrossAge (TTA={USE_TTA}, GPU={USE_GPU})...")
            return FaceEngineCrossAge(use_gpu=USE_GPU, use_tta=USE_TTA)
        except Exception as e:
            print(f">> [AI-MIDS API] Warning: Failed to load FaceEngineCrossAge ({e}), falling back to antelopev2.")
            ENGINE_NAME = "antelopev2"
    if ENGINE_NAME == "aimids_ft":
        try:
            print(f">> [AI-MIDS API] Initializing FaceEngineAdaFaceFT (TTA={USE_TTA}, GPU={USE_GPU})...")
            return FaceEngineAdaFaceFT(use_gpu=USE_GPU, use_tta=USE_TTA)
        except Exception as e:
            print(f">> [AI-MIDS API] Warning: Failed to load FaceEngineAdaFaceFT ({e}), falling back to antelopev2.")
            ENGINE_NAME = "antelopev2"
    if ENGINE_NAME == "antelopev2":
        try:
            print(f">> [AI-MIDS API] Initializing FaceEngineAntelopeV2 (TTA={USE_TTA}, GPU={USE_GPU})...")
            return FaceEngineAntelopeV2(use_gpu=USE_GPU, use_tta=USE_TTA)
        except Exception as e:
            print(f">> [AI-MIDS API] Warning: Failed to load AntelopeV2 ({e}), falling back to buffalo_l.")
            ENGINE_NAME = "buffalo_l"
    ENGINE_NAME = "buffalo_l"
    print(f">> [AI-MIDS API] Initializing standard FaceEngine (buffalo_l, GPU={USE_GPU})...")
    return FaceEngine(use_gpu=USE_GPU)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, store, liveness, ENGINE_NAME
    print(">> [AI-MIDS API] Initializing models and storage...")
    engine = _init_engine()
    store = FaceStore(db_path=DB_PATH)
    stored = store.stats()["models"]
    if stored and stored.get(engine.model_name, 0) == 0:
        others = ", ".join(f"{m} ({n} photos)" for m, n in stored.items() if m != engine.model_name)
        print(f">> [AI-MIDS API] WARNING: the active engine is {engine.model_name}, but this database "
              f"only holds vectors from {others}. Those photos stay in the database and are not "
              f"searchable until re-registered, or set AIMIDS_ENGINE to match them.")
    gpu_active = bool(getattr(engine, "use_gpu", USE_GPU))
    cal_label = "calibrated" if store.calibrated_ready(engine.model_name) else "cosine_fallback"
    print(f">> [AI-MIDS API] Ready (Engine: {ENGINE_NAME}, GPU: {gpu_active}, Calibration: {cal_label}). Database loaded from: {DB_PATH}")
    yield
    print(">> [AI-MIDS API] Shutting down.")


app = FastAPI(
    title="AI-MIDS API",
    description="Face Matching and Egyptian ID Verification System",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration. Pinned to localhost on purpose: this API has no auth and
# holds photos of real people, so "*" here would let any website the operator visits read the
# database. Set AIMIDS_ALLOWED_ORIGINS to the real frontend origin(s) when the UI is built.
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "AIMIDS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def block_cross_site_writes(request, call_next):
    # CORS controls response access, but a cross-site form can still submit a simple POST without it.
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        same_origin = origin == str(request.base_url).rstrip("/") if origin else False
        cross_site_fetch = request.headers.get("sec-fetch-site") == "cross-site"
        if cross_site_fetch or (origin and not same_origin and origin not in CORS_ORIGINS):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Cross-site state-changing requests are not allowed."}
            )
    return await call_next(request)


app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


# --------------------------------------------------------------------------- Helper functions
MAX_UPLOAD_BYTES = 15 * 1024 * 1024      # a card photo or missing-person picture never needs more
MAX_IMAGE_PIXELS = 20_000_000           # stops a decompression bomb from allocating all the RAM


async def _read_upload(upload: UploadFile) -> np.ndarray:
    """Reads an uploaded image file into a BGR OpenCV numpy array without disk writes."""
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{upload.filename}' is bigger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{upload.filename}' is not a valid image format."
        )
    if img.shape[0] * img.shape[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image is too large ({img.shape[1]}x{img.shape[0]} px), max {MAX_IMAGE_PIXELS // 1_000_000} megapixels."
        )
    return img


# --------------------------------------------------------------------------- System & Stats
@app.get("/")
def root():
    cal_ready = bool(store and engine and store.calibrated_ready(engine.model_name))
    return {
        "app": "AI-MIDS API",
        "status": "online",
        "engine": ENGINE_NAME,
        "tta": USE_TTA,
        "gpu": bool(getattr(engine, "use_gpu", USE_GPU)) if engine is not None else USE_GPU,
        "calibration": "calibrated" if cal_ready else "cosine_fallback",
        "database": DB_PATH,
        "asnorm_active": cal_ready,
        "stats": store.stats() if store else {}
    }


@app.get("/demo")
def get_demo_page():
    """Serves the built-in demo page (search + add missing person)."""
    demo_path = os.path.join(HERE, "demo_page.html")
    if not os.path.isfile(demo_path):
        raise HTTPException(status_code=404, detail="demo_page.html not found.")
    return FileResponse(demo_path, media_type="text/html")


def _resolve_demo_probes_dir() -> tuple[str, bool]:
    """Resolves the demo probes base directory, honoring optional AIMIDS_DEMO_WORK_DIR."""
    raw = os.environ.get("AIMIDS_DEMO_WORK_DIR", "").strip()
    if not raw:
        return DEMO_PROBES_DIR, False
    candidate = raw if os.path.isabs(raw) else os.path.join(HERE, raw)
    if not os.path.exists(candidate) and not os.path.isabs(raw):
        candidate = os.path.abspath(raw)
    nested = os.path.join(candidate, "demo_probes")
    if os.path.isdir(nested):
        return nested, True
    return candidate, True


@app.get("/api/demo/probes/{file_path:path}")
def get_demo_probe_file(file_path: str):
    """Serves probe photos, manifest.json and summary.json from the demo_probes directory."""
    probes_dir, is_overridden = _resolve_demo_probes_dir()
    base = os.path.realpath(probes_dir)
    target = os.path.realpath(os.path.join(base, file_path))
    if not (target == base or target.startswith(base + os.sep)):
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not is_overridden and file_path == "summary.json":
        summary_candidates = []
        if ENGINE_NAME == "aimids_crossage":
            summary_candidates.extend([
                os.path.join(HERE, "demo_crossage_work", "demo_results_calibrated", "demo_summary.json"),
                os.path.join(HERE, "demo_crossage_work", "demo_results", "demo_summary.json"),
            ])
        elif ENGINE_NAME == "antelopev2":
            summary_candidates.append(
                os.path.join(HERE, "demo_results_calibrated", "demo_summary.json")
            )
        for cand in summary_candidates:
            resolved = os.path.realpath(cand)
            if os.path.isfile(resolved):
                return FileResponse(resolved, media_type="application/json")
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="Demo file not found.")
    media = (
        "application/json"
        if target.endswith(".json")
        else ("image/png" if target.lower().endswith(".png") else "image/jpeg")
    )
    return FileResponse(target, media_type=media)


@app.get("/api/stats")
def get_stats():
    return store.stats()


# --------------------------------------------------------------------------- Persons Management
@app.get("/api/persons")
def list_persons():
    """Lists all registered persons in the database."""
    with store.lock:
        rows = store.db.execute(
            "SELECT id, name, notes, created FROM persons ORDER BY id DESC"
        ).fetchall()
        result = []
        for r in rows:
            photo_count = store.db.execute(
                "SELECT COUNT(*) FROM faces WHERE person_id=?", (r[0],)
            ).fetchone()[0]
            result.append({
                "person_id": r[0],
                "name": r[1],
                "notes": r[2],
                "created": r[3],
                "photos_count": photo_count
            })
        return result


@app.post("/api/persons", status_code=status.HTTP_201_CREATED)
def create_person(name: str = Form(...), notes: str = Form("")):
    """Registers a new person profile."""
    name_clean = name.strip()
    if not name_clean:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    pid = store.add_person(name_clean, notes=notes.strip())
    return {"person_id": pid, "name": name_clean, "notes": notes.strip()}


@app.get("/api/persons/{person_id}")
def get_person(person_id: int):
    """Retrieves details and photo entries for a specific person without exposing disk paths."""
    with store.lock:
        p = store.db.execute("SELECT id, name, notes, created FROM persons WHERE id=?", (person_id,)).fetchone()
        if not p:
            raise HTTPException(status_code=404, detail=f"Person with ID {person_id} not found.")
        photos = store.db.execute(
            "SELECT id, age FROM faces WHERE person_id=?", (person_id,)
        ).fetchall()
        return {
            "person_id": p[0],
            "name": p[1],
            "notes": p[2],
            "created": p[3],
            "photos": [{"face_id": r[0], "photo_url": f"/api/photos/{r[0]}", "age": r[1]} for r in photos]
        }


@app.delete("/api/persons/{person_id}")
def delete_person(person_id: int):
    """Deletes a person, their photo files from disk, and all associated photo embeddings."""
    with store.lock:
        exists = store.db.execute("SELECT 1 FROM persons WHERE id=?", (person_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail=f"Person with ID {person_id} not found.")
        photo_rows = store.db.execute("SELECT photo_path FROM faces WHERE person_id=?", (person_id,)).fetchall()
        photo_paths = [r[0] for r in photo_rows if r[0]]
        store.delete_person(person_id)

    # Delete photo files from disk
    for path in photo_paths:
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass

    return {"deleted": True, "person_id": person_id}


# --------------------------------------------------------------------------- Photo Registration & Serving
@app.get("/api/photos/{face_id}")
def get_face_photo(face_id: int):
    """Serves the registered photo for a given face_id."""
    with store.lock:
        row = store.db.execute("SELECT photo_path FROM faces WHERE id=?", (face_id,)).fetchone()
        if not row or not row[0]:
            raise HTTPException(status_code=404, detail=f"Photo for face_id {face_id} not found.")
        photo_path = row[0]
        if not os.path.isfile(photo_path):
            raise HTTPException(status_code=404, detail="Photo file not found on disk.")
        return FileResponse(photo_path, media_type="image/jpeg")


@app.post("/api/persons/{person_id}/photos")
async def add_person_photo(
    person_id: int,
    file: UploadFile = File(...),
    age: Optional[int] = Form(None),
    face_index: Optional[int] = Form(None)
):
    """
    Validates photo quality and registers a face embedding for a person.
    If multiple faces exist, returns 'choose_face' status and lists the detected faces.
    """
    with store.lock:
        exists = store.db.execute("SELECT 1 FROM persons WHERE id=?", (person_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail=f"Person with ID {person_id} not found.")

    img = await _read_upload(file)

    # Check photo quality & face selection in threadpool
    try:
        q = await run_in_threadpool(check_photo, engine, img, face_index=face_index)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if q["status"] == "reject":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "reject", "reasons": q["reasons"], "faces": q["faces"]}
        )

    if q["status"] == "choose_face":
        # Group photo detected and no face_index specified:
        # Create a numbered visual preview in uploads/previews/ and clean up previews older than 1 hour
        _cleanup_old_previews(max_age_seconds=3600)
        preview_id = uuid.uuid4().hex[:8]
        preview_filename = f"preview_{preview_id}.jpg"
        preview_path = os.path.join(PREVIEWS_DIR, preview_filename)
        annotated_img = draw_faces(img, q["faces"])
        cv2.imencode(".jpg", annotated_img)[1].tofile(preview_path)

        _, buf = cv2.imencode(".jpg", annotated_img)
        b64_data = base64.b64encode(buf).decode("utf-8")

        return {
            "status": "choose_face",
            "message": "Multiple faces detected. Open the 'preview_url' in your browser to see the faces numbered (#0, #1, #2), then re-submit with 'face_index'.",
            "preview_url": f"/api/preview/{preview_filename}",
            "preview_base64": f"data:image/jpeg;base64,{b64_data}",
            "faces": q["faces"],
            "reasons": q["reasons"]
        }

    # Save photo to uploads folder always with .jpg extension (content is JPEG)
    saved_filename = f"p_{person_id}_{uuid.uuid4().hex[:8]}.jpg"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)
    cv2.imencode(".jpg", img)[1].tofile(saved_path)

    # Add embedding to FAISS + SQLite
    face_quality = extract_face_quality(engine, img, q["face"])
    try:
        fid = store.add_photo(
            person_id=person_id,
            embedding=q["face"]["embedding"],
            photo_path=saved_path,
            age=age,
            model=engine.model_name,
            quality=face_quality
        )
    except ValueError as e:
        try:
            os.remove(saved_path)  # don't leave an orphan photo on disk
        except OSError:
            pass
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return {
        "status": q["status"],
        "face_id": fid,
        "photo_url": f"/api/photos/{fid}",
        "person_id": person_id,
        "age": age,
        "metrics": q["metrics"],
        "reasons": q["reasons"]
    }


@app.get("/api/preview/{filename}")
def get_preview_image(filename: str):
    """Serves the visual preview image with numbered face boxes from previews directory."""
    safe_filename = os.path.basename(filename)
    preview_path = os.path.join(PREVIEWS_DIR, safe_filename)
    if not os.path.isfile(preview_path):
        raise HTTPException(status_code=404, detail="Preview image not found or expired.")
    return FileResponse(preview_path, media_type="image/jpeg")


# --------------------------------------------------------------------------- Missing Persons Search
@app.post("/api/search")
async def search_persons(
    file: UploadFile = File(...),
    k: int = Form(10),
    min_sim: float = Form(-1.0)
):
    """
    Searches the missing-persons database using an uploaded image.
    Supports single or group photos by detecting and querying every face.
    Always returns the top k (default 10, range 1..50) matches regardless of score when min_sim=-1.0.
    """
    if not (1 <= k <= 50):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="k must be between 1 and 50."
        )

    img = await _read_upload(file)
    faces = await run_in_threadpool(engine.embed_all, img)

    if not faces:
        return {
            "faces_detected": 0,
            "results": [],
            "message": "No faces found in the search image."
        }

    results = []
    for idx, f in enumerate(faces):
        try:
            q_probe = extract_face_quality(engine, img, f)
            hits = await run_in_threadpool(
                store.search,
                f["embedding"],
                k,
                min_sim,
                engine.model_name,
                q_probe,
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
        formatted_hits = []
        for h in hits:
            item = {
                "person_id": h["person_id"],
                "name": h["name"],
                "notes": h["notes"],
                "similarity": h["similarity"],
                "match_source": h.get("match_source"),
                "centroid_similarity": h.get("centroid_similarity"),
                "best_photo_similarity": h.get("best_photo_similarity"),
                "tier": h["tier"],
                "face_id": h.get("face_id"),
                "photo_url": f"/api/photos/{h['face_id']}" if h.get("face_id") else None,
                "age": h.get("age")
            }
            if "calibrated_llr" in h:
                item.update({
                    "calibrated_llr": h.get("calibrated_llr"),
                    "match_prob": h.get("match_prob"),
                    "p_nonmate": h.get("p_nonmate"),
                    "z_bg": h.get("z_bg"),
                    "margin_12": h.get("margin_12"),
                    "advisory_flags": h.get("advisory_flags", []),
                    "raw_centroid_similarity": h.get("raw_centroid_similarity"),
                    "raw_best_photo_similarity": h.get("raw_best_photo_similarity"),
                })
            formatted_hits.append(item)
        confident_hits = [h for h in formatted_hits if h["tier"] in ("STRONG", "possible")]
        confident_match_count = len(confident_hits)
        is_calibrated_result = (
            ("calibrated_llr" in formatted_hits[0])
            if formatted_hits
            else store.calibrated_ready(engine.model_name)
        )
        if formatted_hits and formatted_hits[0]["tier"] == "STRONG":
            verdict = "MATCH_FOUND"
            message = f"Strong match found: {formatted_hits[0]['name']} (similarity {formatted_hits[0]['similarity']})."
        elif formatted_hits and formatted_hits[0]["tier"] == "possible":
            verdict = "POSSIBLE_MATCH"
            message = f"Possible match found: {formatted_hits[0]['name']} (similarity {formatted_hits[0]['similarity']}). Manual verification recommended."
        else:
            verdict = "NO_CONFIDENT_MATCH"
            if is_calibrated_result:
                message = "No confident match in database (no candidate cleared open-set calibrated threshold). Weak candidates are shown below only for severe age-gap review."
            else:
                message = "No confident match in database (all scores < 0.26). Weak candidates are shown below only for severe age-gap review."

        results.append({
            "face_index": idx,
            "bbox": [round(v) for v in f["bbox"]],
            "det_score": round(f["det_score"], 3),
            "verdict": verdict,
            "message": message,
            "confident_match_count": confident_match_count,
            "match_count": len(formatted_hits),
            "matches": formatted_hits
        })

    return {
        "faces_detected": len(faces),
        "results": results
    }


# --------------------------------------------------------------------------- Egyptian ID Verification
def _check_selfie_framing(img: np.ndarray) -> dict:
    """Lightweight SCRFD-only face detection + 20-55% framing check (no embeddings or liveness)."""
    det, _ = engine.app.det_model.detect(img, max_num=0, metric="default")
    boxes = [d[:4] for d in det] if det is not None and len(det) > 0 else []

    if len(boxes) == 0:
        return {
            "faces_detected": 0,
            "ok": False,
            "status": "no_face",
            "face_pct": None,
            "message": "No face detected — look at the camera.",
        }

    if len(boxes) > 1:
        return {
            "faces_detected": len(boxes),
            "ok": False,
            "status": "multiple_faces",
            "face_pct": None,
            "message": f"{len(boxes)} faces detected — keep only yourself in frame.",
        }

    h, w = img.shape[:2]
    raw = boxes[0]
    bbox = [
        max(0.0, min(float(raw[0]), float(w))),
        max(0.0, min(float(raw[1]), float(h))),
        max(0.0, min(float(raw[2]), float(w))),
        max(0.0, min(float(raw[3]), float(h))),
    ]
    ok, _, pct = framing(bbox, img.shape)
    bx1, by1, bx2, by2 = bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h
    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    ph = (bbox[3] - bbox[1]) / h
    pw = (bbox[2] - bbox[0]) / w
    pct_round = round(float(pct) * 100.0, 1)

    if not ok:
        if ph > FACE_MAX or pw > FACE_MAX:
            res_status = "too_close"
            message = f"Too close ({pct_round:.0f}%) — move back a little"
        else:
            res_status = "too_far"
            message = f"Too far ({pct_round:.0f}%) — move closer"
    elif cx < 0.16 or cx > 0.84 or cy < 0.16 or cy > 0.84:
        ok = False
        res_status = "off_center"
        message = "Center your face inside the oval."
    else:
        res_status = "ready"
        message = f"Hold still — framing OK ({pct_round:.0f}%)"

    return {
        "faces_detected": 1,
        "ok": bool(ok),
        "status": res_status,
        "face_pct": pct_round,
        "message": message,
        "bbox": [round(bx1, 4), round(by1, 4), round(bx2, 4), round(by2, 4)],
    }


@app.post("/api/verify-id/check-framing")
async def check_verify_framing(file: UploadFile = File(...)):
    """Lightweight live webcam framing check using SCRFD detection only (20-55% face height)."""
    img = await _read_upload(file)
    return await run_in_threadpool(_check_selfie_framing, img)


@app.post("/api/verify-id")
async def verify_egyptian_id(
    id_card: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    """
    Verifies an Egyptian ID card photo against a selfie photo with anti-spoofing liveness check.

    Security & framing rules:
    - The selfie MUST be captured live by the client app (a normal photo file of a real face passes liveness).
    - Face height must be 20-55% of the frame or the result is 'uncertain' -> REVIEW.
    - Liveness check is mandatory and cannot be skipped.

    Decisions: ACCEPT, REVIEW, REJECT, RETRY_CARD, RETRY_SELFIE, REJECT_SPOOF.
    """
    card_img = await _read_upload(id_card)
    selfie_img = await _read_upload(selfie)

    def _sync_verify_id(c_img, s_img):
        with verify_lock:
            return verify_id(c_img, s_img, engine, liveness=_get_liveness())

    result = await run_in_threadpool(_sync_verify_id, card_img, selfie_img)

    return {
        "decision": result["decision"],
        "similarity": result["similarity"],
        "liveness": result["liveness"],
        "card_info": {
            "nid": result["nid"],
            "birth_date": result["birth_date"],
            "governorate": result["governorate"],
            "gender": result["gender"],
            "nid_valid": result["nid_valid"],
        },
        "problems": result["problems"]
    }
