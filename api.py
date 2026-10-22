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
"""
import os
import sys
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
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# Setup directory paths to import modules
HERE = os.path.dirname(os.path.abspath(__file__))
ARC_DIR = os.path.join(HERE, "Arc face")
OCR_DIR = os.path.join(HERE, "OCR_Egyptian_ID-main")

sys.path.insert(0, ARC_DIR)
sys.path.insert(0, OCR_DIR)

from face_engine import FaceEngine
from face_engine_antelopev2 import FaceEngineAntelopeV2
from store import FaceStore, STRONG, POSSIBLE
from photo_quality import check_photo, draw_faces
import utils
from id_verify import verify_id
from liveness import LivenessChecker

# Global instances initialized on startup
ENGINE_NAME = os.environ.get("AIMIDS_ENGINE", "antelopev2").lower()
USE_TTA = os.environ.get("AIMIDS_TTA", "1").lower() in ("1", "true", "yes")

engine = None
store: Optional[FaceStore] = None
liveness: Optional[LivenessChecker] = None
verify_lock = threading.Lock()

DB_PATH = os.environ.get("AIMIDS_DB_PATH", os.path.join(ARC_DIR, "aimids.db"))
UPLOAD_DIR = os.path.join(HERE, "uploads")
PREVIEWS_DIR = os.path.join(UPLOAD_DIR, "previews")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREVIEWS_DIR, exist_ok=True)


def _cleanup_old_previews(max_age_seconds: int = 3600):
    """Deletes preview images older than max_age_seconds (default 1 hour)."""
    try:
        now = time.time()
        if os.path.isdir(PREVIEWS_DIR):
            for fname in os.listdir(PREVIEWS_DIR):
                fpath = os.path.join(PREVIEWS_DIR, fname)
                if os.path.isfile(fpath):
                    if now - os.path.getmtime(fpath) > max_age_seconds:
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, store, liveness, ENGINE_NAME
    print(">> [AI-MIDS API] Initializing models and storage...")
    if ENGINE_NAME == "antelopev2":
        try:
            print(f">> [AI-MIDS API] Initializing FaceEngineAntelopeV2 (TTA={USE_TTA})...")
            engine = FaceEngineAntelopeV2(use_gpu=False, use_tta=USE_TTA)
        except Exception as e:
            print(f">> [AI-MIDS API] Warning: Failed to load AntelopeV2 ({e}), falling back to buffalo_l.")
            ENGINE_NAME = "buffalo_l"
            engine = FaceEngine(use_gpu=False)
    else:
        print(">> [AI-MIDS API] Initializing standard FaceEngine (buffalo_l)...")
        engine = FaceEngine(use_gpu=False)
    store = FaceStore(db_path=DB_PATH)
    liveness = LivenessChecker()
    print(f">> [AI-MIDS API] Ready (Engine: {ENGINE_NAME}). Database loaded from: {DB_PATH}")
    yield
    print(">> [AI-MIDS API] Shutting down.")


app = FastAPI(
    title="AI-MIDS API",
    description="Face Matching and Egyptian ID Verification System",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- Helper functions
async def _read_upload(upload: UploadFile) -> np.ndarray:
    """Reads an uploaded image file into a BGR OpenCV numpy array without disk writes."""
    try:
        data = await upload.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError()
        return img
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{upload.filename}' is not a valid image format."
        )


# --------------------------------------------------------------------------- System & Stats
@app.get("/")
def root():
    return {
        "app": "AI-MIDS API",
        "status": "online",
        "engine": ENGINE_NAME,
        "database": DB_PATH,
        "stats": store.stats() if store else {}
    }


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
    q = await run_in_threadpool(check_photo, engine, img, face_index=face_index)

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
    try:
        fid = store.add_photo(
            person_id=person_id,
            embedding=q["face"]["embedding"],
            photo_path=saved_path,
            age=age
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
            hits = await run_in_threadpool(store.search, f["embedding"], k, min_sim)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        formatted_hits = []
        for h in hits:
            formatted_hits.append({
                "person_id": h["person_id"],
                "name": h["name"],
                "notes": h["notes"],
                "similarity": h["similarity"],
                "tier": h["tier"],
                "face_id": h.get("face_id"),
                "photo_url": f"/api/photos/{h['face_id']}" if h.get("face_id") else None,
                "age": h.get("age")
            })
        confident_hits = [h for h in formatted_hits if h["tier"] in ("STRONG", "possible")]
        confident_match_count = len(confident_hits)
        if formatted_hits and formatted_hits[0]["tier"] == "STRONG":
            verdict = "MATCH_FOUND"
            message = f"Strong match found: {formatted_hits[0]['name']} (similarity {formatted_hits[0]['similarity']})."
        elif formatted_hits and formatted_hits[0]["tier"] == "possible":
            verdict = "POSSIBLE_MATCH"
            message = f"Possible match found: {formatted_hits[0]['name']} (similarity {formatted_hits[0]['similarity']}). Manual verification recommended."
        else:
            verdict = "NO_CONFIDENT_MATCH"
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
@app.post("/api/verify-id")
async def verify_egyptian_id(
    id_card: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    """
    Verifies an Egyptian ID card photo against a selfie photo with anti-spoofing liveness check.

    Security & framing rules:
    - The selfie MUST be captured live by the client app (a normal photo file of a real face passes liveness).
    - Face height must be 20-37% of the frame or the result is 'uncertain' -> REVIEW.
    - Liveness check is mandatory and cannot be skipped.

    Decisions: ACCEPT, REVIEW, REJECT, RETRY_CARD, RETRY_SELFIE, REJECT_SPOOF.
    """
    card_img = await _read_upload(id_card)
    selfie_img = await _read_upload(selfie)

    def _sync_verify_id(c_img, s_img):
        with verify_lock:
            return verify_id(c_img, s_img, engine, liveness=liveness)

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
