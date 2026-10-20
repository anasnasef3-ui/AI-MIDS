# AI-MIDS (AI Missing Identity Detection System)

**AI-MIDS** is an end-to-end AI system designed for two core missions:
1. **Missing-Persons Search**: Register missing individuals (children or adults) with one or more photos, and search the database with newer photos across long age gaps (e.g. 10 to 25+ years) or group photos.
2. **Egyptian National ID Verification**: Detect an Egyptian national ID card, locate fields, read and validate the 14-digit national ID, decode demographic data (birth date, governorate, gender), extract the card photo, and verify it against a live webcam selfie with anti-spoofing liveness detection.

> **Important Deployment Notice**: AI-MIDS is currently a **local demo system with no authentication**. Do **not** expose this server directly to a public network.

---

## Architecture Overview

* **Face Embedding Engine (`face_engine.py` / `face_engine_antelopev2.py`)**:
  * **InsightFace antelopev2** (Default: Glint360K ResNet-100 backbone, 512-dimensional L2-normalized vectors) with optional Flip Test-Time Augmentation (`AIMIDS_TTA=1`).
  * **InsightFace buffalo_l** (Fallback / alternative: ArcFace ResNet-50 backbone).
  * Configurable via environment variable: `AIMIDS_ENGINE=antelopev2` (default) or `AIMIDS_ENGINE=buffalo_l`.
* **Vector Index & Storage (`store.py`)**:
  * **SQLite (`aimids.db`)**: Single source of truth for persons, metadata, and embeddings.
  * **FAISS (`IndexIDMap(IndexFlatIP)`)**: In-memory vector search engine for sub-millisecond similarity matching.
  * **Multi-Photo Centroid Fusion**: Maintains normalized 512-d centroid embeddings per person across all registered photos. In searches, candidates are scored using $\max(\text{best\_individual\_photo\_sim}, \text{centroid\_sim})$ for maximum cross-age robustness.
  * **Search Philosophy**: Always returns a ranked top-10 list of candidate matches for a human investigator to inspect, rather than silently dropping true matches with rigid score cutoffs.
  * **Confidence Tiers**:
    * `STRONG`: Cosine similarity $\ge 0.36$ (high-confidence lead).
    * `possible`: Cosine similarity $\ge 0.26$ (candidate for review).
    * `weak`: Cosine similarity $< 0.26$ (typical for 10–25+ year age gaps; presented for manual verification).
* **ID Reader & Validator (`utils.py`, `id_verify.py`)**:
  * 3-stage YOLO pipeline running on CPU for card detection, field localization, digit OCR, and demographic decoding.
  * **25% Border Padding on ID Card Crop**: ID card photo crops receive 25% replicate padding prior to embedding to eliminate detection failures on tight card crops.
  * Standalone script supports `--antelopev2` CLI flag or `AIMIDS_ENGINE=antelopev2`.
* **Anti-Spoofing Liveness (`liveness.py`)**: MiniFASNet (V2 + V1SE ensemble).
  * **Client Framing Rule**: The selfie **MUST** be captured live by the client app (a static photo file of a real face will pass liveness if not checked live). Face height must occupy **20%–37%** of the frame height; outside this range the result is flagged as uncertain (`REVIEW`).
* **Photo Quality Gatekeeper (`photo_quality.py`)**: Checks blur, face resolution, and pose angle (`SIDE_ON = 0.15`), and handles multi-face group photo selection with visual numbered previews in `uploads/previews/` (auto-cleaned after 1 hour).
* **Backend API (`api.py`)**: Local demo FastAPI service exposing all functionality via asynchronous endpoints with background threadpool execution and concurrency locking. Root endpoint `GET /` reports the active engine (`antelopev2` or `buffalo_l`).

---

## Getting Started

### 1. Environment & Setup
All models and dependencies run in the project virtual environment on Python 3.10:
* Windows terminal with virtual environment active (`.venv`).
* Auto-activates in VS Code terminals via `dev_env.bat`.

### 2. Running the FastAPI Server
To launch the API service:
```cmd
cd "D:\GRAD\Project\Main files"
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Once running, access the interactive Swagger documentation in your browser:
👉 **`http://127.0.0.1:8000/docs`**

---

## API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Health check & database summary (non-blocking) |
| `/api/stats` | `GET` | Returns database statistics (people and photos count) |
| `/api/persons` | `GET` | Lists all registered persons |
| `/api/persons` | `POST` | Registers a new person profile (`name`, `notes`) |
| `/api/persons/{id}` | `GET` | Returns details and registered photos with relative `photo_url`s (no disk paths) |
| `/api/persons/{id}` | `DELETE` | Deletes a person, their photo files from disk, and all associated embeddings |
| `/api/persons/{id}/photos` | `POST` | Uploads and validates a photo. If multiple faces exist, returns `choose_face` with numbered `preview_url` |
| `/api/photos/{face_id}` | `GET` | Serves the registered JPEG photo for a specific face |
| `/api/preview/{filename}` | `GET` | Serves the visual preview image with numbered face boxes (**#0**, **#1**, **#2**) |
| `/api/search` | `POST` | Searches missing persons using an uploaded photo. Returns top $k$ (1..50, default 10) matches without score cutoffs, including top-level `verdict` (`MATCH_FOUND`, `POSSIBLE_MATCH`, `NO_CONFIDENT_MATCH`), plain-language `message`, and `confident_match_count` |
| `/api/verify-id` | `POST` | Verifies Egyptian ID card against live selfie with mandatory anti-spoofing liveness |

---

## Realistic Aging Search Benchmark

Missing person searches face severe age progression gaps (10 to 25+ years between childhood photos and adult search probes).
On the **FG-NET aging benchmark** (370 probes, gallery registered at age $\le 12$):
* **Top-10 Hit Rate (no cutoff)**: **83%** (the true person appears in the top-10 candidates).
* **Top-10 Hit Rate (with 0.26 cutoff)**: **14%** (drops to only **7.5%** for 20+ year gaps, because genuine long-gap matches typically score between 0.10 and 0.25).

Therefore, AI-MIDS defaults to `min_sim = -1.0`, returning the top-ranked candidates labeled with confidence tiers (`STRONG`, `possible`, `weak`) alongside an unambiguous `verdict` (`MATCH_FOUND`, `POSSIBLE_MATCH`, `NO_CONFIDENT_MATCH`) and `confident_match_count` for human reviewer evaluation rather than prematurely discarding genuine leads.

To run the benchmark:
```cmd
cd "D:\GRAD\Project\Main files\Arc face"
python test_realistic_search.py
```

---

## Running Verification & Tests

### Integration Test Suite (All API Endpoints)
Exercises all API checks on a temporary test database, validating validation rules, tier labeling, concurrency, and file cleanup:
```cmd
cd "D:\GRAD\Project\Main files"
python test_api_endpoints.py
```

### Standalone ID Verification & Live Selfie
Runs the standalone webcam-guided verification script (supports default Buffalo_L or AntelopeV2):
```cmd
cd "D:\GRAD\Project\Main files\OCR_Egyptian_ID-main"
python id_verify.py ID.jpg
# Or with AntelopeV2:
python id_verify.py ID.jpg --antelopev2
```

---

## Project Structure

```text
D:\GRAD\Project\
├── .venv/                         # Python 3.10 virtual environment
├── dev_env.bat                    # Environment activator
├── README.md                      # Project root documentation
├── Main files/
│   ├── README.md                  # Main documentation
│   ├── api.py                     # FastAPI backend server (local demo)
│   ├── test_api_endpoints.py      # Automated API integration test suite
│   ├── uploads/                   # Uploaded registration photos (.jpg)
│   │   └── previews/              # Temporary group preview images (1h TTL)
│   ├── Arc face/
│   │   ├── face_engine.py         # ArcFace feature extraction engine
│   │   ├── store.py               # SQLite + FAISS vector storage (3 tiers)
│   │   ├── photo_quality.py       # Quality gatekeeper & numbered box drawer
│   │   ├── test_realistic_search.py # Aging benchmark test script
│   │   └── my_photos/             # Reference photos
│   └── OCR_Egyptian_ID-main/
│       ├── utils.py               # Egyptian ID card reader & decoder
│       ├── id_verify.py           # Standalone ID vs Selfie verification
│       ├── liveness.py            # MiniFASNet anti-spoofing checker
│       ├── best.pt                # YOLO ID card detector
│       ├── detect_odjects.pt      # YOLO ID field locator
│       └── detect_id_new.pt       # YOLO 14-digit reader model
└── _archive_old_OCR/              # Archived legacy files
```
