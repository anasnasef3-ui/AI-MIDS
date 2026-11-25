# AI-MIDS (AI Missing Identity Detection System)

**AI-MIDS** is an end-to-end AI system designed for two core missions:
1. **Missing-Persons Search**: Register missing individuals (children or adults) with one or more photos, and search the database with newer photos across long age gaps (e.g. 10 to 25+ years) or group photos.
2. **Egyptian National ID Verification**: Detect an Egyptian national ID card, locate fields, read and structurally validate the 14-digit national ID (digit count, century digit, governorate code, and a real calendar date; the registry's 14th check digit has no publicly documented algorithm, so it is not verified), decode demographic data (birth date, governorate, gender), extract the card photo, and verify it against a live webcam selfie with anti-spoofing liveness detection.

> **Important Deployment Notice**: AI-MIDS is currently a **local demo system with no authentication**. Do **not** expose this server directly to a public network.

---

## Architecture Overview

* **Face Embedding Engine (`face_engine.py` / `face_engine_antelopev2.py` / `face_engine_adaface_ft.py` / `face_engine_custom.py`)**:
  * **Cross-Age Dual-Branch Engine (`face_engine_custom.py`) (Default)**: 1024-dimensional engine (`AIMIDS_ENGINE=aimids_crossage`) that runs one shared SCRFD-10G detector pass, computes Flip-TTA `antelopev2` (512-d, `w_a=0.55`) + `aimids_ft` (512-d, `w_b=0.45`) embeddings, concatenates and L2-normalizes to 1024-d, and scores per-face `quality` in `[0.25, 1.0]`. Automatically falls back to `antelopev2` then `buffalo_l` if loading fails.
  * **InsightFace antelopev2**: Glint360K ResNet-100 backbone, 512-dimensional L2-normalized vectors with optional Flip Test-Time Augmentation (`AIMIDS_TTA=1`).
  * **InsightFace buffalo_l**: Fallback / alternative ArcFace ResNet-50 backbone.
  * **Custom Fine-Tuned / From-Scratch ONNX (`face_engine_adaface_ft.py`)**: Custom ONNX backbone (`AIMIDS_ENGINE=aimids_ft`, 512-d embeddings) with automatic fallback to `antelopev2` if the ONNX model is missing on disk. Configurable via `AIMIDS_ONNX_PATH` (defaults to `Main files\Arc face\pretrained\aimids_adaface_ir101_ft.onnx`, `Main files\Arc face\aimids_adaface_ir101_ft.onnx`, or `checkpoints\aimids_adaface_ir101_ft.onnx`; also supports `aimids_scratch_ir50_crossage.onnx`).
  * Configurable via environment variable: `AIMIDS_ENGINE=aimids_crossage` (default), `AIMIDS_ENGINE=antelopev2`, `AIMIDS_ENGINE=buffalo_l`, or `AIMIDS_ENGINE=aimids_ft`.
* **Vector Index & Storage (`store.py`)**:
  * **SQLite (`aimids.db`)**: Single source of truth for persons, metadata, and embeddings. Every embedding row is tagged with the engine that produced it (`faces.model`), and each engine gets its own `512`-d or `1024`-d FAISS index and its own centroids, so vectors from two engines (or dimensions) are never compared or averaged together. After switching `AIMIDS_ENGINE`, previously registered photos stay in the database but are not searchable until re-registered; the server prints a startup warning and `stats.models` shows what is in the database.
  * **FAISS (`IndexIDMap(IndexFlatIP)`)**: In-memory vector search engine for sub-millisecond similarity matching (strict per-model `512`-d / `1024`-d dimension enforcement).
  * **Quality & Age-Weighted Multi-Photo Centroids + Unified Calibrated Routing**: Maintains normalized per-person centroids weighted by clamped photo quality (`[0.25, 1.0]`) and inverse age-density (`1.0 / (1.0 + nearby_photos_within_2y)` when ages are provided). `FaceStore.search()` auto-routes via `store.calibrated_ready(model)` with zero `< 80`-person discontinuity: when `Arc face\pretrained\calib_<model>.npz` exists (`aimids_crossage`, `antelopev2`), it runs the 4-layer calibrated pipeline (`store._search_calibrated()`); for uncalibrated/experimental models (`aimids_ft`, `buffalo_l`), it cleanly runs `store._search_cosine_fallback()` ($\max(\text{best\_photo\_sim}, \text{centroid\_sim})$). Each hit includes `match_source` (`"centroid"` or `"photo"`), `centroid_similarity`, and `best_photo_similarity` (displayed in `/api/search` and `demo_page.html` via `Matched by: Centroid` / `Matched by: Single photo` badges and breakdown text). Invalid embeddings (zero-length / NaN) or mismatched dimensions are rejected with HTTP 400.
  * **Search Philosophy**: Always returns a ranked top-10 list of candidate matches for a human investigator to inspect, rather than silently dropping true matches with rigid score cutoffs.
  * **Confidence Tiers**:
    * **Calibrated Mode (Default for `aimids_crossage` & `antelopev2`, `Arc face\calibration.py` + `Arc face\pretrained\calib_<model>.npz`)**: Uses an identity-disjoint 1,200-face external LAG cohort (`ExternalCohortASNorm`, $O(N)$ precomputed enrollment stats) across both single photos and centroids, quality- and age-aware QMF logistic LLR fusion (`QMFCalibrator`, `extract_face_quality()` in `[0.25, 1.0]`), gallery-size ($N$) adaptive Split Conformal + GPD POT open-set thresholds (`OpenSetCalibrator`: `STRONG` at empirical `FPIR <= 1%`, `possible` at `<= 8%`, plus `p_nonmate`), and duplicate-immune background rank standardization (`compute_bg_zscore` over $s_{(3..50)}$ emitting non-gating `advisory_flags`).
    * **Cosine Fallback Mode (Automatic for uncalibrated/experimental engines such as `aimids_ft` / `buffalo_l`)**:
      * `STRONG`: Cosine similarity $\ge 0.36$ (high-confidence lead).
      * `possible`: Cosine similarity $\ge 0.26$ (candidate for review).
      * `weak`: Cosine similarity $< 0.26$ (typical for 10–25+ year age gaps; presented for manual verification).
* **ID Reader & Validator (`utils.py`, `id_verify.py`)**:
  * 3-stage YOLO pipeline running on CPU for card detection, field localization, digit OCR, and demographic decoding.
  * **25% Border Padding on ID Card Crop**: ID card photo crops receive 25% replicate padding prior to embedding to eliminate detection failures on tight card crops.
  * Standalone script supports `--antelopev2`, `--aimids-ft`, or `AIMIDS_ENGINE`.
* **Anti-Spoofing Liveness (`liveness.py`)**: MiniFASNet (V2 + V1SE ensemble).
  * **Client Framing Rule**: The selfie **MUST** be captured live by the client app (a static photo file of a real face will pass liveness if not checked live). Face height must occupy **20%–55%** of the frame height; outside this range the result is flagged as uncertain (`REVIEW`).
* **Photo Quality Gatekeeper (`photo_quality.py`)**: Checks blur, face resolution, and pose angle (`SIDE_ON = 0.15`), and handles multi-face group photo selection with visual numbered previews in `uploads/previews/` (auto-cleaned after 1 hour).
* **Backend API (`api.py`)**: Local demo FastAPI service exposing all functionality via asynchronous endpoints with background threadpool execution and concurrency locking. At process startup, `api.py` sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before importing `numpy`/`faiss` (preventing Windows OpenMP/MKL virtual commit bloat) and lazily initializes `LivenessChecker` via `_get_liveness()` on the first `/api/verify-id` request. Root endpoint `GET /` reports the active engine (`aimids_crossage` default, `antelopev2`, `buffalo_l`, or `aimids_ft`), `"calibration"` (`"calibrated"` or `"cosine_fallback"`), `"tta"` (`true`/`false`), `"gpu"` (`true` when `CUDAExecutionProvider` is active, controlled by `AIMIDS_USE_GPU`), and `"asnorm_active"` (`true` when `store.calibrated_ready(engine.model_name)` is ready, else `false`).

---

## Getting Started

### 1. Environment & Setup
All models and dependencies run in the project virtual environment on Python 3.10:
* Windows terminal with virtual environment active (`.venv`).
* Auto-activates in VS Code terminals via `dev_env.bat`.
* **GPU Acceleration (CUDA 12 / cuDNN 9) & Windows Memory Bounds**: `.venv` uses `onnxruntime-gpu==1.20.2` (installed with `--no-deps` so `numpy<2` is preserved for `faiss-cpu`, `insightface`, and `torch`). At engine initialization, `ensure_cuda_dll_paths()` in `Arc face\face_engine.py` registers `.venv\Lib\site-packages\torch\lib` (which bundles the CUDA 12 / cuDNN 9 DLLs) on Windows so `CUDAExecutionProvider` loads cleanly without `LoadLibrary` errors, uses bounded CUDA EP options (`cudnn_conv_algo_search="HEURISTIC"`, `cudnn_conv_use_max_workspace="0"`, `arena_extend_strategy="kSameAsRequested"`), and falls back automatically to `CPUExecutionProvider` if CUDA fails. `api.py` also pins `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` and defers `LivenessChecker` loading to `_get_liveness()`.

### 2. Running the FastAPI Server
To launch the API service:
```cmd
cd "D:\GRAD\Project\Main files"
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Once running, access the interactive Swagger documentation in your browser:
👉 **`http://127.0.0.1:8000/docs`**

### 3. Environment Variables & Input Limits

| Variable | Default | Purpose |
|---|---|---|
| `AIMIDS_ENGINE` | `aimids_crossage` | Face engine (`aimids_crossage` default, `antelopev2`, `buffalo_l`, or `aimids_ft`); falls back automatically to `antelopev2` then `buffalo_l` if loading fails |
| `AIMIDS_CALIB_DIR` | *(empty)* | Optional directory override for `calib_<model>.npz` (defaults to `Arc face\pretrained\`) |
| `AIMIDS_TTA` | `1` | Flip Test-Time Augmentation for embeddings |
| `AIMIDS_USE_GPU` | `1` | Use CUDA GPU (`CUDAExecutionProvider`) for face engines when available (`1` default; set `0` to force CPU) |
| `AIMIDS_DB_PATH` | `Arc face\aimids.db` | SQLite database location (`demo_crossage.db` for the default doctor demo, or `demo_aimids.db` with `AIMIDS_ENGINE=antelopev2`) |
| `AIMIDS_DEMO_WORK_DIR` | *(empty)* | Optional base work/probes directory override for `/api/demo/probes/*` |
| `AIMIDS_ONNX_PATH` | *(empty)* | Optional explicit path to custom ONNX model (defaults to `Main files\Arc face\pretrained\aimids_adaface_ir101_ft.onnx`, `Main files\Arc face\aimids_adaface_ir101_ft.onnx`, or `checkpoints\aimids_adaface_ir101_ft.onnx`; when set with `AIMIDS_ENGINE=aimids_ft`, e.g. `aimids_scratch_ir50_crossage.onnx`, calibration resolves `Arc face\pretrained\calib_<onnx_stem>.npz` if present) |
| `AIMIDS_ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated browser origins allowed to call the API. Set this to your real frontend origin when the UI exists |

Hard limits in `api.py`: upload bytes are capped at **15 MB** before decoding, decoded images are capped at **20 megapixels**, `k` must be 1..50, and an out-of-range `face_index` returns HTTP 400 instead of a server error. Because the service has no authentication, keep uvicorn bound to `127.0.0.1` and do not widen CORS to `*`. Only `127.0.0.1` and `localhost` are accepted as `Host` headers. On `POST`, `PUT`, `PATCH`, and `DELETE`, a cross-site request is rejected with HTTP 403 unless its `Origin` is same-origin or listed in `AIMIDS_ALLOWED_ORIGINS`; this also blocks simple form posts that CORS alone cannot stop.

---

## API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Health check, active `engine`, `calibration` (`calibrated` / `cosine_fallback`), `tta`, `gpu`, `asnorm_active` & database summary (non-blocking) |
| `/api/stats` | `GET` | Database statistics: `people`, `photos`, and `models` (photo count per embedding engine) |
| `/api/persons` | `GET` | Lists all registered persons |
| `/api/persons` | `POST` | Registers a new person profile (`name`, `notes`) |
| `/api/persons/{id}` | `GET` | Returns details and registered photos with relative `photo_url`s (no disk paths) |
| `/api/persons/{id}` | `DELETE` | Deletes a person, their photo files from disk, and all associated embeddings |
| `/api/persons/{id}/photos` | `POST` | Uploads and validates a photo. If multiple faces exist, returns `choose_face` with numbered `preview_url` |
| `/api/photos/{face_id}` | `GET` | Serves the registered JPEG photo for a specific face |
| `/api/preview/{filename}` | `GET` | Serves the visual preview image with numbered face boxes (**#0**, **#1**, **#2**) |
| `/api/search` | `POST` | Searches missing persons using an uploaded photo. Returns top $k$ (1..50, default 10) matches without score cutoffs, including `match_source` (`"centroid"` / `"photo"`), `centroid_similarity`, `best_photo_similarity`, top-level `verdict` (`MATCH_FOUND`, `POSSIBLE_MATCH`, `NO_CONFIDENT_MATCH`), plain-language `message`, and `confident_match_count`. When calibrated (default for `aimids_crossage` / `antelopev2`), each hit also includes `calibrated_llr`, `match_prob`, `p_nonmate`, `z_bg`, `margin_12`, `advisory_flags`, `raw_centroid_similarity`, and `raw_best_photo_similarity` (omitted in `cosine_fallback` mode). |
| `/api/verify-id` | `POST` | Verifies Egyptian ID card photo against live selfie with mandatory anti-spoofing liveness |
| `/api/verify-id/check-framing` | `POST` | Lightweight SCRFD-only face detection & `20%–55%` height/centering check for live webcam selfie framing (`no_face`, `multiple_faces`, `too_close`, `too_far`, `off_center`, `ready`) |
| `/demo` | `GET` | Full-pipeline 3-tab demo UI (`demo_page.html`): live status strip (`Calibrated` / `Cosine fallback` pill), Missing-Persons Search & Register (`Matched by: Centroid` / `Matched by: Single photo` badges + centroid/photo breakdown + calibrated LLR/`match_prob`/`p_nonmate`/`z_bg`/`advisory_flags` metrics, dynamic `caf`/`ecaf` filter & cards when present), Egyptian ID & Liveness Verification (upload or webcam with live 50% portrait oval `20%–55%` framing guide, vertical size meter on `0..70%` scale, hold-still auto-capture, and retake UX), and Registered Persons Manager |
| `/api/demo/probes/{path}` | `GET` | Serves held-back demo search photos (`.jpg`/`.png`) plus `manifest.json`/`summary.json` (honors `AIMIDS_DEMO_WORK_DIR`; otherwise serves `demo_crossage_work\demo_results_calibrated\demo_summary.json` -> `demo_crossage_work\demo_results\demo_summary.json` for `aimids_crossage`, or `demo_results_calibrated\demo_summary.json` for `antelopev2`, when present; path traversal returns 400) |

---

## Realistic Aging Search Benchmark

Missing person searches face severe age progression gaps (10 to 25+ years between childhood photos and adult search probes).
On the **FG-NET aging benchmark** (370 probes, gallery registered at age $\le 12$):
* **Top-10 Hit Rate (no cutoff)**: **83%** (the true person appears in the top-10 candidates).
* **Top-10 Hit Rate (with 0.26 cutoff)**: **14%** (drops to only **7.5%** for 20+ year gaps, because genuine long-gap matches typically score between 0.10 and 0.25).

Therefore, AI-MIDS defaults to `min_sim = -1.0`, returning the top-ranked candidates labeled with confidence tiers (`STRONG`, `possible`, `weak`) alongside an unambiguous `verdict` (`MATCH_FOUND`, `POSSIBLE_MATCH`, `NO_CONFIDENT_MATCH`) and `confident_match_count` for human reviewer evaluation rather than prematurely discarding genuine leads.

To run the child-to-adult FG-NET retrieval benchmark through `FaceStore`:
```powershell
cd "D:\GRAD\Project\Main files\Arc face"
D:\GRAD\Project\.venv\Scripts\python.exe evaluate.py --dataset fgnet
# (Original historical 370-probe script is preserved in Arc face\archive\test_realistic_search.py)
```

---

## Doctor Demo (AgeDB + FG-NET + CALFW)

A prebuilt search gallery for the doctor demo: **4,670 people / 21,664+ face photos**, one person per identity in AgeDB, FG-NET and CALFW. For each person the **2 oldest photos are held back as search photos** and the rest are registered. AgeDB and FG-NET are split by their age labels; CALFW by the trailing file counter (CALFW has no age labels, so the counter is not an age). Held-back photos are copied to `demo_probes\` and listed in `demo_probes\manifest.json`.

Start the server against the default cross-age demo database (`demo_crossage.db`) and open the page:
```powershell
cd "D:\GRAD\Project\Main files"
$env:AIMIDS_DB_PATH = "demo_crossage.db"
D:\GRAD\Project\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
# Or run with antelopev2 against demo_aimids.db:
# $env:AIMIDS_ENGINE = "antelopev2"; $env:AIMIDS_DB_PATH = "demo_aimids.db"
# D:\GRAD\Project\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```
👉 **`http://127.0.0.1:8000/demo`** — pick a test photo from the gallery (or upload one) and search; "Add a missing person" registers someone new from the browser, and a group photo asks which face to use.

### Results on the demo gallery (4,670 benchmark people / 4,672 DB entries, 7,761 held-back `oldest-2` probes)

> **Audit Caveats (2026-11-24):** At least **137 shared celebrities** appear twice under separate `"AgeDB ..."` and `"CALFW ..."` names, making Rank-1 ill-defined (matching the duplicate twin steals Rank #1). In addition, holding out only the 2 oldest photos registers younger-to-middle-aged photos up to `gallery_age_max`, so actual AgeDB/FG-NET probe age gaps here are often **0–11 years** rather than 20–30+ years (and `age_gap_years` is blank in CSV because `gallery_age_max` was saved to `_setup_state.json`).

`antelopev2` (`demo_aimids.db`, 4,672 DB people / 21,667 gallery photos, `demo_results\demo_summary.json`, **11.2 min** eval on GTX 1660 SUPER) vs. `aimids_crossage` (`demo_crossage.db`, 4,672 DB people / 21,783 gallery photos, GPU + quality/age-weighted centroids + Centroid-Only AS-Norm, `demo_crossage_work\demo_results\demo_summary.json`, **15.6 min** eval on GTX 1660 SUPER):

| set | probes | `antelopev2` R1 / T5 / T10 (STRONG) | `aimids_crossage` R1 / T5 / T10 (STRONG) |
|---|---|---|---|
| overall | 7,761 | 91.10% / 95.48% / 95.64% (7,384) | **91.21% / 95.64% / 95.77%** (**7,385**) |
| AgeDB (`2 oldest photos held out`) | 1,132 | **96.47%** / 97.44% / 97.44% (1,101) | 96.29% / **97.53% / 97.53%** (1,101) |
| FG-NET (`2 oldest photos held out`) | 164 | 98.78% / 99.39% / 99.39% (163) | **99.39% / 99.39% / 99.39%** (163) |
| CALFW (`2 photos held out`) | 6,465 | 89.96% / 95.03% / 95.24% (6,120) | **90.12% / 95.22% / 95.38%** (**6,121**) |

Summary JSONs: `demo_results\demo_summary.json` (`antelopev2` legacy, per-photo rows in `demo_results\demo_eval.csv`) and `demo_crossage_work\demo_results\demo_summary.json` (`aimids_crossage` legacy).

#### Full Demo Gallery Results Under Calibrated Search (`demo_eval.py --results-dir`, 7,761 held-back probes)

| Config (`demo_eval.py --results-dir`) | Runtime | Overall R1 / T5 / T10 | AgeDB (`1,132`) R1 / T10 | FG-NET (`164`) R1 / T10 | CALFW (`6,465`) R1 / T10 | Tiers of Correct Hits (`STRONG` / `possible` / `weak`) |
|---|---|---|---|---|---|---|
| `antelopev2` (`legacy`) | 11.2 min | 91.10% / 95.48% / 95.64% | 96.47% / 97.44% | 98.78% / 99.39% | 89.96% / 95.24% | 7,384 / 31 / 8 |
| `antelopev2` (`calibrated`, `demo_results_calibrated\`) | 11.7 min | 91.06% / 95.44% / 95.54% | **97.26%** / **97.53%** | 98.17% / 99.39% | 89.79% / 95.10% | 6,618 / 268 / 529 |
| `aimids_crossage` (`legacy`) | 15.6 min | **91.21% / 95.64% / 95.77%** | 96.29% / **97.53%** | **99.39%** / 99.39% | **90.12% / 95.38%** | **7,385** / 33 / 15 |
| `aimids_crossage` (`calibrated`, `demo_crossage_work\demo_results_calibrated\`) | 15.2 min | 91.20% / 95.54% / 95.63% | **97.26%** / **97.53%** | 98.78% / 99.39% | 89.95% / 95.20% | **7,269** / 46 / 107 |

Calibrated mode preserves top-10 retrieval while boosting **AgeDB-30** Rank-1 to **97.26%** on both engines and capping open-set false alarms on strangers under **0.7%** (`STRONG`). Between the two engines under calibrated mode, **`aimids_crossage`** outperforms **`antelopev2`** on every retrieval metric and keeps **+651 more true matches in `STRONG`** (**7,269** vs **6,618**). **`aimids_crossage` + Calibrated Search** is now the default production pipeline in `api.py`.

### Rebuilding or resuming (and isolated `--work-dir` / `--results-dir` builds)
```powershell
python demo_setup.py          # ≈2.5 h cold on CPU; resumes from demo_probes\_setup_state.json
python demo_eval.py           # ≈11–15 min on GPU; rewrites demo_results\ and demo_probes\summary.json
```
Pass `--results-dir` to `demo_eval.py` (for example `--results-dir demo_results_calibrated`) to write CSV/JSON results separately without overwriting `demo_results\` or `demo_probes\summary.json`. To build or evaluate an isolated gallery (for example `demo_crossage.db` with `aimids_crossage`) without touching `demo_aimids.db`, `demo_probes\`, or `demo_results\`, pass `--work-dir` to both scripts:
```powershell
$env:AIMIDS_ENGINE="aimids_crossage"; $env:AIMIDS_DB_PATH="demo_crossage.db"
D:\GRAD\Project\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
# In a second terminal:
D:\GRAD\Project\.venv\Scripts\python.exe demo_setup.py --work-dir "D:\GRAD\Project\Main files\demo_crossage_work"
D:\GRAD\Project\.venv\Scripts\python.exe demo_eval.py --work-dir "D:\GRAD\Project\Main files\demo_crossage_work" --results-dir "D:\GRAD\Project\Main files\demo_crossage_work\demo_results_calibrated"
```
The setup script checkpoints every 25 people and, on startup, deletes people registered after the last checkpoint of an interrupted run — so re-running it is safe, but never run two builds at once. The real database (`Arc face\aimids.db`) is never touched. To include **CAF** and **ECAF** (`D:\GRAD\Assets\CAF`, `D:\GRAD\Assets\ECAF`) in a separate leak-free isolated demo build without overlapping `lag_cohort_cache.npz` or duplicating celebrity names across datasets, pass `--datasets agedb,fgnet,calfw,caf,ecaf --exclude-cohort --dedup-existing` alongside `--work-dir` and a separate `AIMIDS_DB_PATH`.

---

## Running Verification & Tests

### Integration Test Suite (All API Endpoints)
**71/71 strict checks** under default `aimids_crossage` (`71` or `70` under `antelopev2` depending on demo summary presence; exits `0` on full pass and `1` on any failure) on a temporary test database: registration, quality gate, genuinely distinct 2nd photo upload (`child2_anas.jpg`), bad input (`face_index`, oversized upload), Rank-1 search match requirement, tier labeling, `match_source` / `centroid_similarity` / `best_photo_similarity` breakdown, calibrated `LLR` / `match_prob` / `p_nonmate` / `advisory_flags` fields, embedding provenance, `tta`, `calibration`, & `asnorm_active` root flags, CORS, cross-site write and `Host`-header guards, the 3-tab `/demo` page (including 50% portrait oval `20%–55%` framing UI), `POST /api/verify-id` exact fixture NID/validity/liveness (`ACCEPT`, `30512101404256`, `nid_valid=True`, `real`), `POST /api/verify-id/check-framing` (single-face framing, blank `no_face`, group `multiple_faces`) + `liveness` `20%–55%` framing threshold sanity check, `AIMIDS_DEMO_WORK_DIR` / demo `summary.json` serving and path-traversal guard, concurrency, file cleanup, fine-tuned AdaFace ONNX engine verification and non-vacuous model isolation against `api.engine.model_name`, and section `[11]` (`FaceStore` 1024-d indexing + strict 512/1024-d isolation, quality + inverse age-density centroid weighting, small-gallery `N=5` calibrated search with zero `< 80` cliff, and uncalibrated cosine fallback routing). It needs the private reference photos in `Arc face\my_photos\` and `OCR_Egyptian_ID-main\ID.jpg` (both gitignored; `ID.jpg` sits locally and must never be pushed; a fresh clone must copy it from `_archive_old_OCR\OCR_Egyptian_ID-main\ID.jpg` or verify-id checks cannot run).
```powershell
cd "D:\GRAD\Project\Main files"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; D:\GRAD\Project\.venv\Scripts\python.exe test_api_endpoints.py
# Or test under 512-d antelopev2:
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:AIMIDS_ENGINE="antelopev2"; D:\GRAD\Project\.venv\Scripts\python.exe test_api_endpoints.py; Remove-Item Env:AIMIDS_ENGINE
```

### Strict Leakage-Audited Full-Protocol Evaluation (`Arc face\strict_eval.py` & `test_strict_eval.py`)
**Trusted benchmark suite (added 2026-11-24).** Enforces zero `--limit`, reads the official `AgeDB-30` `.mat` images directly from `03_Protocol_Images.tar` in memory without extracting to disk (ignoring 12,240 `.pts` files), scores 1:1 pairs with raw single-image cosine and strict out-of-fold threshold/`TAR@FAR=1%` selection, retains failed detections in the denominator, blocks path/SHA-256/`cos >= 0.97` duplicates, excludes `1,390` `CACD` train/val IDs from `split_manifest_v2.csv.gz` alongside `lag_cohort_cache.npz` and known `AgeDB`/`CALFW`/`FG-NET`/demo overlaps, verifies live cache parity (`cos >= 0.995` on 32 sampled images per suite), reports Wilson / cluster-bootstrap 95% CIs, and computes paired exact McNemar p-values (`bench_results\strict\strict_compare.json`):
```powershell
cd "D:\GRAD\Project\Main files\Arc face"
# Offline synthetic guard & metric self-tests (12/12 pass, no GPU/ONNX required):
D:\GRAD\Project\.venv\Scripts\python.exe test_strict_eval.py
# Full GPU strict evaluation (~53 min across aimids_crossage & antelopev2):
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; D:\GRAD\Project\.venv\Scripts\python.exe strict_eval.py
```

#### Trusted Strict Verification (`strict_eval.py`, 6,000 official pairs each, 0 failed detections)

| Suite | `aimids_crossage` Acc ± Std [Wilson 95% CI] | `aimids_crossage` TAR@1% (Ach. FAR) | `antelopev2` Acc ± Std [Wilson 95% CI] | `antelopev2` TAR@1% (Ach. FAR) | McNemar (`b / c`, `p`) |
|---|---|---|---|---|---|
| **Official AgeDB-30** (`03_Protocol_Images.tar`) | 98.22% ± 0.64% `[97.85%, 98.52%]` | 97.23% (`1.03%`) | **98.33% ± 0.65%** `[97.98%, 98.63%]` | **97.30%** (`1.03%`) | `-0.12%` (`2/9`, `p=0.0654`, NS) |
| **Official CALFW** (`pairs_CALFW.txt`) | 96.05% ± 1.09% `[95.53%, 96.51%]` | 92.67% (`1.03%`) | **96.07% ± 1.15%** `[95.54%, 96.53%]` | **92.80%** (`1.00%`) | `-0.02%` (`2/3`, `p=1.00`, NS) |

#### Trusted Strict Child-to-Adult Retrieval (`strict_eval.py`, `age_gap=None`, `0` failed/dups)

| Dataset : Scope (`IDs / Probes`) | Mode | `aimids_crossage` R1 [95% CI] / T10 | `antelopev2` R1 [95% CI] / T10 | McNemar R1 Diff (`p`) |
|---|---|---|---|---|
| **FG-NET** : `default` (`74 / 463`) | `raw` / `calibrated` | **96.76%** `[92.67, 99.76]` / **96.11%** (`100%` T10) | 96.33% / 95.90% (`100%` T10) | `+0.43%` / `+0.22%` (`p>=0.77`, NS) |
| **CAF** : `fully_unseen` (`239 / 1,047`) | `raw` | **88.16%** `[84.02, 91.82]` / **93.79%** | 86.34% `[82.22, 90.09]` / 92.84% | **`+1.81%`** (`p=0.00055`, **SIG**) |
| **CAF** : `fully_unseen` (`239 / 1,047`) | `calibrated` | **87.39%** `[83.43, 91.14]` / **93.98%** | 85.67% `[81.46, 89.66]` / 92.65% | **`+1.72%`** (`p=0.00294`, **SIG**) |
| **CAF** : `model_strict` (`364 / 1,866`) | `raw` / `calibrated` | **83.44%** / **83.07%** | 81.56% / 81.14% | **`+1.88%`** / **`+1.93%`** (`p<=2.2e-5`, **SIG**) |
| **ECAF Official** : `model_strict` (`605 / 3,658`*) | `raw` | **83.90%** `[81.36, 86.41]` / **93.25%** | 80.95% `[78.07, 83.65]` / 91.74% | **`+2.95%`** (`p=1.1e-16`, **SIG**) |
| **ECAF Official** : `model_strict` (`605 / 3,658`*) | `calibrated` | **83.41%** `[80.89, 85.88]` / **93.17%** | 79.91% `[76.99, 82.70]` / 91.36% | **`+3.50%`** (`p=1.8e-22`, **SIG**) |
| **ECAF All** : `model_strict` (`648 / 3,820`) | `raw` / `calibrated` | **83.74%** / **83.17%** | 80.73% / 79.69% | **`+3.01%`** / **`+3.48%`** (`p<=3.7e-18`, **SIG**) |
| **ECAF All** : `fully_unseen` (`60 / 291`) | `raw` / `calibrated` | **85.57%** / **84.88%** (`97.94%`/`97.59%` T10) | 83.85% / 83.51% | `+1.72%` / `+1.37%` (`p>=0.23`, NS) |

*\*Note: `ECAF Official` has `0` CACD overlap, though `581/605` IDs overlap `AgeDB`/`CALFW` (affecting QMF/OpenSet calibration fit, not `aimids_ft` training); `2` upstream label collisions (`Serena_Williams` == `Venus_Williams`) were flagged by SHA-256. FG-NET calibrated `>=20y` gap (`104` probes): `aimids_crossage` `86.54%` (raw `87.50%`) vs `antelopev2` `89.42%`.*

### Unified 3-Engine Full-Pipeline Evaluation (`full_pipeline_eval.py` & `Arc face\test_full_pipeline_eval.py`)
Evaluates all 3 engines (**`aimids_crossage`**, **`antelopev2`**, and **`fromscratchv2`** = `aimids_ft` + `aimids_scratch_ir50_crossage.onnx`) equally under **Calibration v2 (`SCHEMA_VERSION = 2`)** through the complete Missing-Persons app pipeline (skipping ID verification & liveness):
- **v2 Mixed-Domain Calibration Design**: Raw SQLite embeddings and 1:1 `/api/verify-id` stay untouched; for 1:N search, `FaceStore` lazily projects embeddings orthogonally to the top-5 young-vs-adult difference vectors (`proj_basis`, `k=5` computed strictly inside external `LAG`), scores against the **600 adult faces (`LAG_ADULT`)** of the identity-disjoint LAG cohort (`std_floor=0.05`), and blends centroid + photo LLRs via `c_llr + 0.35 * max(0, p_llr - c_llr)`.
- **Stage 1**: Isolated FastAPI Missing-Persons HTTP smoke test (`30/30` per engine).
- **Stage 2**: One unified, identity-deduplicated 5-dataset `FaceStore` (`2,806` people / `6,607` gallery photos after NFKD + 14-alias deduplication across `FG-NET` + `ECAF` + `CAF` + `AgeDB >=15y` + `CALFW`, `7,714` mated probes + `200` unseen strangers), evaluated with matched `LAG_ADULT` `600`-face cohort + `Proj5` + `photo_hybrid_weight=0.35` + `CALFW` folds `0..1` QMF (`age_gap=None`) + folds `2..3` `350` non-mates (`matched_calibrated`) as well as on-disk `production_bundle` (`calib_aimids_scratch_ir50_crossage.npz` saved for `fromscratchv2`).
- **Stage 3**: Live HTTP `POST /api/search` vs in-process `FaceStore` parity on a unified 2,806-person SQLite DB across 30 deterministic single-face probes.

```powershell
cd "D:\GRAD\Project\Main files"
# Offline synthetic unit tests (7/7 pass, no GPU/ONNX required):
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; D:\GRAD\Project\.venv\Scripts\python.exe "Arc face\test_full_pipeline_eval.py"
# Full 3-engine calibrated GPU run (Stage 1 + Stage 2 + Stage 3):
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; D:\GRAD\Project\.venv\Scripts\python.exe full_pipeline_eval.py --engines aimids_crossage,antelopev2,fromscratchv2
```

#### Unified 3-Engine Calibrated App Pipeline Results (v2 vs v1 Baseline, `full_pipeline_eval.py`, GPU + Flip-TTA, `0` failed detections, `24/24` cache parity `min_cos >= 0.9999998`)

| Engine (`matched_calibrated` / `Stage 1` / `Stage 3`) | Fair 4 Datasets (`no CAF`, `6,506–6,507` probes) v2 R1 / T10 (v1 R1) | All 5 Datasets (`7,713–7,714` probes) v2 R1 / T10 | `FG-NET` (`463`) R1 / T10 | `ECAF` (`3,752`) v2 R1 / T10 (v1 R1) | `CAF` (`1,207`) R1 / T10 | `AgeDB` (`924`) R1 / T10 | `CALFW` (`1,367–1,368`) R1 / T10 | Stranger FP (`STRONG` / `Poss+`) | Query / HTTP Latency |
|---|---|---|---|---|---|---|---|---|---|
| **`aimids_crossage`** (`30/30` S1, `30/30` S3 `0.0` diff) | **80.39%** / **91.59%** *(73.96%)* | **79.47%** / **90.42%** | **93.74%** / **98.27%** | **70.42%** / **87.90%** *(59.70%)* | **74.48%** / **84.26%** | **95.35%** / **97.29%** | **93.13%** / **94.81%** | **1.00%** / **5.50%** | `15.57` / `136.99` ms |
| **`antelopev2`** (`30/30` S1, `30/30` S3 `0.0` diff) | 79.99% / 90.84% *(73.68%)* | 79.02% / 89.56% | 92.66% / 96.33% | 70.07% / 86.78% *(59.22%)* | 73.82% / 82.85% | 94.70% / 96.86% | 92.98% / 94.66% | **1.00%** / 6.00% | `13.72` / `110.77` ms |
| **`fromscratchv2`** (`30/30` S1, `30/30` S3 `0.0` diff) | 66.20% / 81.99% *(57.27%)* | 65.34%* / 81.04%* | 85.96% / 94.82% | 48.56% / 71.75% *(33.34%)* | 60.73%* / 76.06%* | 89.18% / 95.35% | 92.39% / 94.66% | **1.00%** / 7.00% | `14.11` / `83.90` ms |

*\*Note: `CAF` was in `fromscratchv2` Stage 2 training (`caf_in_train_set=True`), so `fair_4_datasets_no_caf` is the primary fair 3-way metric (`[78.30, 82.51]` / `[77.91, 81.99]` / `[63.80, 68.59]` 95% cluster CIs). Paired exact McNemar (`matched_calibrated`): `aimids_crossage` vs `antelopev2` is `+0.40%` on Fair-4 R1 (`p=0.1223`, NS; `+0.44%` `all_5`, `p=0.0632`, NS; `+0.75%` Fair-4 T10), while both lead `fromscratchv2` by **`+14.19%` / `+13.79%`** (`p < 1e-208`, **SIG**). Using on-disk `production_bundle` gives **80.42%** / **80.00%** / **66.20%** Fair-4 R1 (`53.34%` / `38.26%` / `35.23%` `STRONG` recall; `antelopev2` `calib_antelopev2.npz` has `tau_strong=3.4032` at `N=2,806`).*

### Calibration Unit Tests & Offline Fitter (`calibration.py`, `calibrate_thresholds.py`)
Runs 10 offline unit and integration checks (`ExternalCohortASNorm`, `QMFCalibrator`, `OpenSetCalibrator` GPD + Split Conformal monotonicity, `CoveCalibrator` + trimmed `compute_bg_zscore`, v2 `LAG_ADULT` + `proj_basis` + `photo_hybrid_weight` `.npz`/`.json` bundle round-trips, `FaceStore` automatic calibrated search vs uncalibrated cosine fallback, raw DB preservation, and `AIMIDS_ONNX_PATH` bundle resolution for `aimids_ft`):
```powershell
cd "D:\GRAD\Project\Main files\Arc face"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; D:\GRAD\Project\.venv\Scripts\python.exe test_calibration.py
# Rebuild v2 calib_antelopev2.npz + calib_aimids_crossage.npz (LAG_ADULT + Proj5 + w_h=0.35):
D:\GRAD\Project\.venv\Scripts\python.exe calibrate_thresholds.py --cohort-source lag --skip-ab
```

### Unified Cross-Age & Missing-Found Evaluation (`Arc face\evaluate.py`)
Replaces the 7 superseded evaluation/scratch scripts (now preserved in `Arc face\archive\`: `bench_verify.py`, `bench_tiers.py`, `test_realistic_search.py`, `test_agedb_benchmark.py`, `test_my_photos.py`, `try_store.py`, `show_faces.py`). Runs on CUDA GPU by default (`ensure_cuda_dll_paths()`, pass `--cpu` to force CPU) with Flip-TTA enabled (except `buffalo_l`), automatically matching production `FaceStore` scoring (`Calibrated` when `calib_<engine>.npz` exists without `--onnx`; `Cosine Fallback` for `--onnx` custom models or uncalibrated engines):
- **`agedb30`**: 10-fold AgeDB-30-style self-built deterministic verification (unless `03_Protocol_Images` is extracted; for strict official `.mat` from tar without extraction, use `strict_eval.py`) using quality + inverse age-density weighted multi-photo galleries (`--gallery 3`, `max(best_photo, centroid)`).
- **`calfw`**: 10-fold CALFW 1:1 pair verification.
- **`fgnet`**: Child ($\le 12\text{y}$) $\to$ Adult ($> 12\text{y}$) 1:N retrieval directly through `FaceStore(":memory:")`.
- **`caf`**: **Cross-Age Face** (`D:\GRAD\Assets\CAF`, ages `0–94`, `519` deduplicated IDs / `4,000` imgs) child ($\le 12\text{y}$) $\to$ adult ($> 12\text{y}$) 1:N retrieval (or `--caf-split oldest`), excluding `lag_cohort_cache.npz` (`604` IDs) by default; pass `--disjoint` to also exclude `AgeDB`/`CALFW`/`FG-NET`/demo DB overlaps.
- **`ecaf`**: **ECAF** (`D:\GRAD\Assets\ECAF`, `9,963` `img_ffhq` PNGs / `1,293` paired `_child_` $\to$ `_adult_` IDs) missing-child gallery $\to$ adult probe 1:N retrieval, excluding `lag_cohort_cache.npz` by default; supports `--disjoint` and `--official-test` (`name_list.json` `test=True` split).
- **`missing_all`**: Evaluates `fgnet`, `caf`, `ecaf`, and a cross-deduplicated combined `FaceStore(":memory:")` pool (`CAF` + `ECAF` deduplicated by `normalize_identity()`).

```powershell
cd "D:\GRAD\Project\Main files\Arc face"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
# Default aimids_crossage calibrated run across all 3 standard datasets (--limit 600 on AgeDB-30/CALFW):
D:\GRAD\Project\.venv\Scripts\python.exe evaluate.py --dataset all --limit 600
# Full child->adult missing-found evaluation across FG-NET, CAF, ECAF, and deduplicated combined pool:
D:\GRAD\Project\.venv\Scripts\python.exe evaluate.py --dataset missing_all
# Strictly unseen (cohort + AgeDB/CALFW/FG-NET/demo-disjoint) missing-found evaluation:
D:\GRAD\Project\.venv\Scripts\python.exe evaluate.py --dataset missing_all --disjoint
# Official ECAF test split (cohort-disjoint):
D:\GRAD\Project\.venv\Scripts\python.exe evaluate.py --dataset ecaf --official-test
# Full AgeDB/CALFW/FG-NET protocol (--limit 0) on antelopev2:
D:\GRAD\Project\.venv\Scripts\python.exe evaluate.py --dataset all --engine antelopev2 --limit 0
# Evaluate custom from-scratch / fine-tuned ONNX checkpoint on FG-NET (cosine fallback, isolated cache):
D:\GRAD\Project\.venv\Scripts\python.exe evaluate.py --dataset fgnet --engine aimids_ft --onnx "pretrained\aimids_scratch_ir50_crossage.onnx"
```

#### Historical Child-to-Adult Missing-Found Runs (`evaluate.py`, see `strict_eval.py` above for CACD-excluded numbers)

> **Leakage Note:** Legacy `evaluate.py --disjoint` excluded `lag_cohort_cache.npz` and `AgeDB`/`CALFW`/`FG-NET`/demo IDs, but did **not** exclude `CACD-train` (`509 / 569` IDs = `89.5%` in legacy `ECAF --disjoint` were CACD celebrities). Use `strict_eval.py`'s `fully_unseen` / `model_strict` tables above for clean claims.

| Dataset / Split | Gallery IDs | Probes | Rank-1 | Top-5 | Top-10 | `STRONG` | `possible+` |
|---|---|---|---|---|---|---|---|
| **FG-NET** ($\le 12\text{y} \to > 12\text{y}$) | 74 | 463 | **96.11%** | **99.78%** | **100.00%** | 91.14% | 93.52% |
| **CAF** ($\le 12\text{y} \to \text{adult}$, cohort-disjoint) | 403 | 2,019 | **82.42%** | **87.72%** | **89.30%** | 52.90% | 66.62% |
| **CAF** ($\le 12\text{y} \to \text{adult}$, unseen `--disjoint`) | 278 | 1,200 | **85.67%** | **91.00%** | **93.00%** | 58.33% | 71.33% |
| **CAF** (`oldest-2` split, cohort-disjoint / unseen) | 443 / 305 | 886 / 610 | **97.97% / 97.54%** | **98.42% / 97.87%** | **98.53% / 97.87%** | 96.05% / 94.75% | 97.86% / 97.54% |
| **ECAF** (`_child_` $\to$ `_adult_`, cohort-disjoint) | 1,157 | 5,733 | **78.11%** | **86.80%** | **89.06%** | 26.39% | 43.64% |
| **ECAF** (`--official-test`, cohort-disjoint) | 605 | 3,658 | **83.41%** | **90.84%** | **93.17%** | 29.44% | 47.84% |
| **ECAF** (unseen `--disjoint`) | 569 | 2,204 | **76.36%** | **84.94%** | **87.02%** | 38.84% | 52.36% |
| **Missing-All Combined** (deduped, cohort-disjoint) | 1,504 | 7,623 | **77.57%** | **85.73%** | **88.04%** | 32.24% | 47.99% |
| **Missing-All Combined** (deduped, unseen `--disjoint`) | 803 | 3,237 | **77.60%** | **85.14%** | **87.15%** | 41.98% | 55.89% |

### Standalone ID Verification & Live Selfie
Runs the standalone webcam-guided verification script (supports default Buffalo_L, AntelopeV2, or Fine-Tuned AdaFace):
```cmd
cd "D:\GRAD\Project\Main files\OCR_Egyptian_ID-main"
python id_verify.py ID.jpg
# Or with AntelopeV2:
python id_verify.py ID.jpg --antelopev2
# Or with Fine-Tuned AdaFace:
python id_verify.py ID.jpg --aimids-ft
```

---

## Custom Model Data Preparation (Colab)

Use `D:\GRAD\Project\training\aimids_cacd_hda_colab.ipynb` to download and audit the proposed **CACD + HDA-SynChildFaces** training data without storing either dataset on the Windows device, and to run the **age-projection probe** that decides whether a small trained adapter is worth building at all.

1. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and upload the `.ipynb` file.
2. In your own Kaggle account, open **Settings → API**. Preferred: add Colab Secret `KAGGLE_API_TOKEN` containing the current Kaggle API token. Alternative: add both `KAGGLE_USERNAME` and `KAGGLE_KEY` from legacy `kaggle.json`. Enable **Notebook access** for whichever secret method you use. Never paste or share these values in the notebook, chat, or reports.
3. Save the notebook to Drive, leave `MOUNT_DRIVE = True`, authorize Drive when prompted, then choose **Runtime → Run all**. No GPU is needed for this data audit. Missing secrets stop the CACD cell with setup instructions. The final report cell is self-contained, so it can also be run alone in a runtime that still holds the extracted images.
4. CACD and HDA archives/images go only to temporary `/content/aimids_age_training/datasets`; a runtime reset deletes them and requires another download. Reports and audit checkpoints go to `Drive/MyDrive/AI-MIDS/custom_model`. Kaggle API download resume is **not guaranteed**, so an interrupted CACD download may need to restart from the beginning.

The previous Colab warning was caused by prohibited peer-to-peer functionality. This version uses only direct HTTPS: the official Kaggle Python API for CACD and `gdown` for HDA. CACD comes from the **filtered third-party mirror** `pdombrza/cacd-filtered-dataset` (current v3; published filtered ZIP size **3,609,677,946 bytes**), not the original authors' official release. The notebook now sets either `KAGGLE_API_TOKEN` or the legacy username/key pair in the runtime **before** importing `KaggleApi`, avoiding the authentication-on-import `NameError`. It then requests version 3 when supported, scans for the actual archive filename rather than assuming one, checks the exact published size, computes SHA-256 for reproducibility only, tests CRC, and safely extracts it. If multiple archives appear, the notebook stops rather than selecting one. HDA retains its official Drive IDs and exact-size/hash/CRC behavior. Duplicate detection, splitting and report writing share one self-contained cell, so **Runtime → Run all** cannot break when a definitions-only cell is skipped — that skip was the cause of `NameError: name 'exact_duplicate_exclusions' is not defined`.

**Both real layouts are now proven from a read-only Colab probe** — the mirror ZIP extracts to `cacd_split/cacd_split/<Identity>/<age>_<Name>_<NNNN>.jpg` (154,915 images, 2,000 identities) with **no `CACD2000` folder and no CACD-VS files**, plus `CACD_features_sex.csv` (8,798,742 bytes, 154,915 rows) whose `name` column holds each filename and whose `identity` column holds the numeric id 1..2000. CACD training candidates are therefore chosen by provenance: exactly one feature CSV is required, every filename must match a row there, a filename mapped to several identity ids stops the run, and images with no row are counted and excluded instead of relabeled. HDA extracts to `database/{accepted,age_group_0..4}/{probes,references}/images/<seed>_<age>_<sex>_<race>[_<counter>].png` — 31,388 images per folder, 188,328 in total — and its bundled README states that `accepted` holds the 20+ real adult source photos while `age_group_0..4` (16-13, 13-10, 10-7, 7-4, 4-1) hold the synthetic age transforms, so both facts travel with every manifest row as `provenance`. From the real labels, 599 of the 2,000 CACD identities match the embedded evaluation names, which keeps 52,568 images out of training and leaves 1,401 identities / 102,347 images as candidates before duplicate and corrupt-file exclusions.

The audit fully decodes every candidate image, reports corrupt files and exact duplicates, excludes CACD-VS and known AgeDB/CALFW/LAG/demo evaluation identities, and writes deterministic 90/10 train/validation manifests **by identity**. Three manifest rules were corrected after the real audit (see `AGENTS.md`): a duplicate group holding **any** evaluation-blocked copy now loses every copy instead of keeping an arbitrary one; an LFW-flagged CACD identity is dropped **at identity level**, so one flagged photo takes the whole person out of training; and HDA race spellings that differ only by a hyphen or a space collapse into a new `race_normalized` column, so balanced sampling sees six races rather than eight. All three rules live in **one shared definitions cell** that both the full run and the Drive-only rebuild execute, so a correction cannot drift from a future full run. HDA identities are synthetic seeds and the `age_group_0..4` labels are broad generated age bands, not real longitudinal child photography, while the `accepted` folder holds real adult source photos. CACD remains academic-research-only; HDA prohibits commercial use and redistribution and requires research acknowledgement.

### The end cells: CACD-VS probe, Drive-only rebuild, then the age-projection probe

5. **CACD-VS probe — the cell whose first line reads `# >>> CORRECTIONS-ONLY PASS STEP 8/9`.** The official CARC page publishes exactly one CACD-VS artifact over direct HTTPS: `CACD_VS.tar`, labelled **198M** (its `celebrity2000_meta.mat` metadata link now 404s), and the cell fetches it into disposable Colab storage with `gdown`, never into Drive. VS files are named `pairId_0.jpg` / `pairId_1.jpg` — 4,000 pairs (2,000 positive cross-age + 2,000 negative) in 10 folds of 400, first 200 of each fold positive — and **carry no celebrity names and no pair list**, so the cell rebuilds pair/fold/polarity from that published numbering and then recovers *who is in each pair* by matching each VS image's SHA-256 against the CACD audit checkpoint. An image that matches nothing stays unnamed; a digest that matches two CACD people is reported as ambiguous; and the recovered people are cross-checked against the published label, with any contradiction printed instead of quietly accepted. VS names are deliberately **not** blocked from training, because VS reuses the same celebrities as CACD — only byte-identical photos leave training, recorded in `cacd_vs_blocked_digests.json` (every VS digest, so a later rebuild cannot miss a collision). `cacd_vs_probe_report.json` also reports `resolved_pairs_by_exposure`: VS is a leak-free age-gap verification test **only** for pairs whose people sit entirely in the held-out split. **This cell has now run on the real data (2026-11-05)**: `CACD_VS.tar` measured **198,440,960 bytes** with SHA-256 `555a1247…43393`, and inside it were **8,000 `.jpg` files under one `CACD_VS/` folder and no other file at all** — which proves from the bytes, not from the website, that no name list or pair list ships with VS. All 8,000 images decoded (0 bad) and formed exactly **4,000 complete pairs: 2,000 published-positive, 2,000 published-negative, folds 0-9, 0 filenames that are not pair names**. **The report is now read (2026-11-05): good news and one warning.**

6. **Drive-only rebuild — the cell whose first line reads `# >>> CORRECTIONS-ONLY PASS STEP 9/9`.** Needs no dataset: it reads `split_manifest.csv.gz` plus `image_audit_checkpoint.jsonl`, re-derives the copies the first run excluded straight from the checkpoint paths, applies the three corrected rules as **removals only**, keeps the original split assignments so runs stay comparable, and writes `split_manifest_v2.csv.gz`, `manifest_corrections.csv.gz` (each removed row with the rule or rules that fired), `manifest_corrections_report.json` and `audit_summary_v2.json`. It refuses to run when the checkpoint cannot reproduce the blocked-identity count the first audit reported, rather than guessing that it is the same run. The 21 GB never comes back. **Stage its inputs into `/content` before running it.** Reading the 112 MB `image_audit_checkpoint.jsonl` (112,198,283 bytes, 343,243 lines) through Colab's mounted Drive crawled at about **50 KB/s — roughly 35 minutes of total silence that looks exactly like a hang** — while the same bytes came over the authenticated Drive API in **2 seconds**. `gdown` cannot fetch these files because they are private, and the markdown cell above this one now carries the working snippet, including the two traps that cost time here: a chunk size larger than the file returns `HttpError 416`, and inverting the download loop's condition silently truncates the copy (it produced a 20.97 MB file that reported itself as complete). The cell prefers a `/content` copy of any input, falls back to Drive, always writes its four outputs to Drive, and prints each filename before writing it so a slow mount is visible. **Both rebuild failures only appear at the real 288,723-row scale.** The first attempt read the manifest (288,723 rows) and the checkpoint (343,243 records in 2 s) and then died on `TypeError: '<' not supported between instances of 'int' and 'str'` in the final sort: pandas typed that 288,723-row CSV in 262,144-row chunks, and because the mani...

7. **Age-projection probe — the five cells whose first lines read `# >>> AGE-PROJECTION PROBE STEP 1/5` through `5/5`.** This exists to answer one question *before* spending any training budget: **if we take the frozen production vectors of a young photo and push them through a simple learned mathematical map toward "old", do they then match old photos of the same person better than plain comparison does?** A FAIL rejects this tested linear map and shifts the planned review toward backbone fine-tuning; it does not prove every nonlinear adapter would fail. A PASS supports the adapter direction but does not approve an architecture before the loss, sampler, freeze-schedule and evaluation walkthrough. It costs no training, no GPU days and **no HDA** — its only requirement is that CACD is extracted in disposable `/content` again, because the runtime reset deletes the images even though no *new* dataset is being studied.
   * `1/5` picks the people: 300 identities from the CACD training rows to fit the map and 100 completely different identities from the CACD validation rows to test it, taken from `split_manifest_v2.csv.gz`, chosen by a fixed seed so the same 400 people come back on every rerun, at most 15 photos each and only people whose photos actually span years. It stops the run if the two groups share even one person.
   * `2/5` turns those photos into vectors with the **untouched production model** (`antelopev2`, biggest face in the frame, vectors scaled to length 1) and appends each result to `probe_embeddings.jsonl`, so a restart resumes where it stopped instead of re-doing hundreds of photos. The allow-list is `allowed_modules=["detection", "recognition"]`, matching the actual SCRFD/glintr100 task names; `face_recognition` would silently exclude the recognizer. It prints which ONNX providers each loaded model actually uses and warns if an extractor model lacks CUDA; selecting a GPU runtime alone does not fix a CPU-only `onnxruntime` installation. Use a GPU runtime plus `onnxruntime-gpu`, restart, and check the printed providers before trusting the GPU label. Remaining-time estimates appear after 50 photos, every 250, and at completion; skipped photos cannot swallow those progress lines. It reports why any photo was skipped (unreadable, no face found, several faces so the largest was kept, unusable vector) and refuses to continue if under 70% of the chosen photos produced a vector.
   * `3/5` builds the comparisons. A "same person, different age" pair needs an age gap of at least **8 years**. The "different person" pairs are deliberately picked from the **same two age bands** as the pair they belong to — this matters, because without it the probe could look like a winner simply by noticing that two 20-year-olds resemble each other, which says nothing about removing age.
   * `4/5` fits the map in one closed-form step, with one knob that controls how hard it is pulled back toward "do nothing at all" (at the extreme, the map becomes the identity, which *is* the current baseline). The knob setting and whether to transform one photo or both are chosen by a 5-fold split **among the 300 fitting people only** — the 100 test people are never consulted here, and a check proves this cell cannot even see their vectors.
   * `5/5` scores the two rival ways on the 100 untouched people: how well same-person pairs separate from different-person pairs, how often the right person appears in the top 1/5/10 results, an uncertainty range built by resampling **people** rather than photo pairs (photos of one face are not independent evidence), and a breakdown by how many years apart the photos were. It prints **PASS** only if the uncertainty range is entirely above zero *and* top-1 retrieval does not get worse, and saves `age_projection_probe_report.json` plus the map itself. The report says out loud that 100 people is a **signal test, not a benchmark**.

**After the audit in Colab**: steps 1/9 through 9/9 are done — `split_manifest_v2.csv.gz` (284,581 rows) exists — so the data track is closed, and the probe pass above has now been run end to end and returned **FAIL**, so the remaining work is the module walkthrough (loss, sampler, freeze schedule, evaluation protocol) plus the escalation the pre-registered plan names for a FAIL. Colab displays no cell numbers, so find cells by their banner. On a fresh runtime, run **only `CORRECTIONS-ONLY PASS STEP 1/9` setup**, then the probe `1/5` → `5/5` in order and read `age_projection_probe_report.json`; the other corrections cells are not prerequisites because the corrected manifest already exists. If the datasets were wiped (they are, on every reset), CACD has to be downloaded again through the Kaggle secret before `2/5` can embed anything; HDA is not needed. **Do not press Runtime → Run all** — it re-downloads 21 GB and re-decodes 343,242 images. No Kaggle secret is needed for the corrections pass itself: CACD-VS comes from the official Drive link through `gdown`. The title cell of the notebook repeats this checklist.

**Status:** all 22 code cells (37 cells total, no stored outputs) parse under Python 3.10 and contain no prohibited peer-to-peer mechanism — `torrent`/`aria2`/`magnet:`/`udp://tracker` appear nowhere in the notebook. `training/validation/notebook_structure_check.py` (48 checks — including that exactly nine cells carry the `CORRECTIONS-ONLY PASS` banners and exactly five carry the `AGE-PROJECTION PROBE` banners, that each set numbers from 1 in cell order, that none of them downloads a dataset, that those fourteen cells are **name-closed** together (every variable they use is defined by another marked cell or a builtin, so a fresh runtime cannot hit the `NameError` that skipping a definitions cell used to cause), that both probe output files are actually written by a cell, and that the map-fitting cell never references the held-out people's vectors) and `training/validation/rules_regression.py` (72 checks / 0 failures, on synthetic fixtures only — no real dataset bytes were downloaded for these tests) execute the real cells against a temp tree shaped like the proven mirror plus a fabricated `CACD_VS.tar`: the feature-table join, reporting of one unlisted image and one table row without a file, zero-byte and truncated-image detection, a resume pass that redecoded nothing, byte-identical manifests on a rerun, filename-name to identity-id one-to-one proof, probes/references roles, multi-token races, out-of-range ages counted, real-adult versus synthetic provenance, an evaluation identity caught by name, and identity-disjoint 90/10 splits. The corrected rules are covered end to end: whole contaminated duplicate groups removed, plain duplicates reduced to one trainable copy, every photo of an LFW-flagged person dropped, `race_normalized` non-empty and merged across spellings, per-rule reasons recorded in `exact_duplicates.csv`, and the Drive-only rebuild proven on a hand-made old-rules manifest (4 of 14 rows removed, nothing resurrected, splits preserved, byte-identical on...

```bash
python ../training/validation/notebook_structure_check.py
python ../training/validation/rules_regression.py
python ../training/validation/probe_regression.py
```

The structure and probe suites also accept an optional notebook path for reverted-copy tests.

**Colab extractor recovery (2026-11-07):** InsightFace 2.1 placed the ONNX files under `/root/.insightface/models/antelopev2/antelopev2/`, one level below the expected model folder, so detector registration failed. The manual repair moves only those model files up one directory, refuses overwrites, and does not touch CACD or re-download anything. Direct loading confirmed `scrfd_10g_bnkps.onnx` has task `detection` and `glintr100.onnx` has task `recognition`, both listing CUDA first; STEP 2/5 must allow exactly those task names, not `face_recognition`. The test fake previously ignored the allow-list and missed the bug; it now honors it, with a new task-selection check. The repaired embedding branch passes 18 checks and reverting the task token fails three; the full probe suite passes **51 checks / 0 failures**, with synthetic metrics unchanged. The other suites remain **48 structure / 72 rules checks, zero failures**. This model-cache repair is not automatic in the notebook, so check the layout after a fresh model download. **Real STEP 2/5 completed:** 6,000 photos attempted in 115 seconds, **5,996 usable vectors (99.9%), 512 numbers per vector**, both loaded model sessions listing CUDA then CPU. Four photos had no detected face; the 124 multi-face photos were kept using their largest face rather than skipped. **Real STEP 3/5 completed:** fitting uses **2,199 same-person and 2,199 different-person pairs over 300 people**; evaluation uses **773 of each over 100 different people**. The different-person pairs match the age bands of their same-person counterparts. Both median gaps are **8 years**, so any later result is not proof of broad childhood-to-adult recognition. **Real STEP 4/5 completed:** selected map strength `lambda=0.0001`, applied to both search and gallery vectors (`both_sides`), saved to Drive as `age_projection_ridge_map.npz`. Pair-separation gain on unseen fitting people averaged **+0.0044** on the 0–1 scale, with worst fold **-0.0017**. Runners-up: `0.001` gives **+0.0040 / +0.0002** (mean / worst), `0.003` gives **+0.0027 / -0.0013**, both `both_sides`. The map was saved to Drive as `age_projection_ridge_map.npz`; deviation from identity **0.794**, in-sample gain **+0.1273**. The large fitted-data gain versus small cross-validated gain is an overfitting warning, not a held-out success. Freeze the selected settings before STEP 5/5; do not choose a different lambda after seeing evaluation results. The final 100-person verdict is still pending and no adapter training has started.

**STEP 5 correction (2026-11-07, after independent Team review):** the old loop drew identities with replacement and then applied `set()`, discarding their multiplicities. A tiny independent score fixture drawing `A,A,C` proved the error: correct gain **0.111111**, old code **0.125**. Corrected cell 36 concatenates each drawn person's genuine and matched-impostor indices, including repeats. Each new check fails independently when its fix is reverted. The binary gate is unchanged (gain interval above zero and rank-1 not worse); PASS/FAIL wording now describes only this tested map and does not approve an architecture before the loss, sampler, freeze-schedule and evaluation walkthrough. No post-hoc minimum gain, lambda change, refit or re-embedding was introduced. Resampling groups are genuine people plus their fixed age-band-matched controls; the negative row's identity is its anchor label, not either compared face. This is a paired-cluster pilot estimate, not a claim to model every shared-impostor dependency. Also, rank-1 here searches the whole mixed-age gallery with only the query photo masked, not an older-only gallery. The probe's 320/default-0.5/no-TTA configuration differs from production 640/0.4/TTA; any successful internal gate still needs production-setting validation and the module walkthrough before approval. The old uncertainty range has been **replaced** by rerunning only corrected STEP 5 with the same map and cached vectors: the honest interval is now **-0.0300 to +0.0027** (the pre-correction -0.0249 to +0.0011 was optimistic). The suites are green at **51 probe checks / 0 failures**, with structure and rules unchanged at 48 and 72, and reverting either fix produces exactly the 2 expected failures.

**Real held-out verdict (2026-11-07): FAIL — the map did not help.** Scored on the 100 people the fitting stage never saw (773 same-person cross-age pairs, 773 age-band-matched impostor pairs), separating real matches from impostors got *worse*, not better: **0.8535 → 0.8412** on the 0-to-1 scale, a change of **-0.0123** whose uncertainty range **-0.0300 to +0.0027** crosses zero, so the gate cannot claim any gain and the best guess is a small loss. Finding the right person first also slipped, **0.771 → 0.765** (rank-10 0.789 → 0.784), failing the gate's second half. In everyday words: average similarity for genuine pairs fell **0.4808 → 0.4757** while average similarity for different people rose **0.0079 → 0.0141** — the map pushed real matches apart and pulled strangers together. This is exactly what STEP 4/5 warned about (a large +0.1273 gain on the pairs it fitted versus +0.0044 on unseen fitting people, worst fold -0.0017): the map memorized its 300 people instead of learning age. Two limits on how far this result reaches: **every one of the 773 evaluation pairs is 8-12 years apart** (median gap 8 years), so nothing here speaks to childhood-photo-to-adult-search matching, and the probe used lighter detection settings than production. Artifacts: `age_projection_probe_report.json` and `age_projection_ridge_map.npz` in `Drive/MyDrive/AI-MIDS/custom_model`. **What FAIL means:** this specific linear map over frozen production vectors is rejected, and the pre-registered plan escalates the review toward a backbone fine-tune. It does *not* prove every nonlinear adapter would fail, it changes nothing in the deployed system, and it approves no architecture — the loss, sampler, freeze-schedule and evaluation walkthrough is still unwalked, and no setting was tuned after seeing this number. Rerunning the same probe with different knobs would be threshold-shopping; a rerun is only legitimate for a different question written down in advance (wider age gaps or production settings), with the new question and its cutoffs pre-committed.

   * **Why the backbone was decided from bytes and not from assumption:** the deployed model is `glintr100.onnx` (260,665,334 bytes) — 103 convolutions, only 51 batch-norm layers, and every convolution carrying a non-zero bias, while the PyTorch version of the same network has ~148 batch-norm layers and bias-free convolutions. That gap means the released file is a **batch-norm-folded export**, so its weights cannot simply be loaded into a PyTorch network to start from, and `iresnet100(pretrained=True)` fails outright because the Glint360K weights were never published in PyTorch form at all; the local MTLFace checkpoint is not usable either (it crashes this machine's PyTorch when opened). Of the three ways forward, the one chosen first was **a frozen extractor plus a self-trained adapter**: it is the only route that stays exportable the same way production is, ships as a second model name without touching the existing one, and yields a model to defend. Rewriting the ONNX numbers into a PyTorch network was held as the fallback for exactly the case where the probe shows no signal — **and on 2026-11-07 the probe showed no signal**, so that fallback (route A) is now the escalation under review. It does not count as done, or even as started, until the rebuilt PyTorch network reproduces the ONNX outputs to within 0.999 cosine similarity on real photos; until then it is a plan, not a model. Building a backbone from scratch was rejected — this much data cannot justify it and nothing would be comparable to the deployed baseline. No training has started and the architecture is not approved yet — the gate that was supposed to unlock the adapter route returned **FAIL**, so nothing is unlocked. The production antelopev2 model, every evaluation dataset and `/api/search` are all unchanged, and the tier-calibration obligation (`bench_tiers.py`) is moot until some trained model actually beats the baseline.

---

### Next step: fine-tune our own model (planned 2026-11-10)
Because the age-projection probe failed and the deployed `glintr100.onnx` cannot be loaded into PyTorch, the next step is to fine-tune a model whose **PyTorch weights are published** (AdaFace IR-101 first) on Colab. The order is: score the untouched model first (FG-NET child→adult, AgeDB-30, LAG, CALFW, optional YLFW child verification, and UTKFace false-match safety test), then fine-tune on CACD training people with large age gaps, using a distillation loss so the model keeps what it already knows. After that, evaluate on people it never saw, export to ONNX, and add it as a new engine (`AIMIDS_ENGINE=aimids_ft`). The tiers must be re-measured with `bench_tiers.py` before it serves search. Details are in `AGENTS.md` §5 item 0.

#### Step 1: Baseline Verification Notebook (`training/aimids_baseline_colab.ipynb`)
- **Colab GPU Notebook**: `training/aimids_baseline_colab.ipynb` measures the untouched base model (AdaFace IR-101 WebFace12M) vs. production `antelopev2` (with and without Flip-TTA) before any fine-tuning.
- **Datasets**:
  - **AgeDB-30**: 10-fold cross-validation (TAR@FAR=1.0%, accuracy).
  - **FG-NET**: Child -> adult search (Rank-1, Top-5, Top-10).
  - **LAG**: Large Age Gap (3,400 pairs, 10-fold cross-validation, CACD-decontaminated).
  - **CALFW**: 10-fold cross-validation (CACD-decontaminated).
  - **YLFW** (Optional): Young LFW child verification (direct mail.ru download, CACD-decontaminated).
  - **UTKFace** (Optional): Impostor false-match safety test (% sim >= 0.26 and >= 0.36 across age bands, gender, race).
- **Sanity Gate**: Antelopev2 + Flip-TTA on AgeDB-30 must clear the **96.70%** floor (`97.00% - 0.30%`, untrusted otherwise; observed **97.41%** on all 6,000 pairs).
- **Offline Suite**: `training/validation/baseline_structure_check.py` statically validates the 10 code cell banners, AST, zero path leaks, secrets safety via `userdata.get`, runs 3 behavioral preprocessing checks on Cell 7 (padded `norm_crop` image, AdaFace BGR→RGB in `[-1, 1]`, and `antelopev2` Flip-TTA padding fallback), and executes 25 mutation self-tests. Run via:
```bash
python ../training/validation/baseline_structure_check.py
```
- **Completed Baseline Results (2026-11-13, Colab GPU)**:

| Dataset | Pairs / Probes | `antelopev2` (no-TTA) | `antelopev2` (Flip-TTA) | `AdaFace IR-101` (no-TTA) | `AdaFace IR-101` (Flip-TTA) |
|---|---|---|---|---|---|
| **AgeDB-30** | 5,995 / 6,000 pairs | 97.33% ± 0.87% (TAR@1%: 95.47%) | **97.41% ± 0.85%** (TAR@1%: **95.80%**) | 96.81% ± 0.92% (TAR@1%: 94.80%) | 96.91% ± 0.99% (TAR@1%: 95.07%) |
| **FG-NET** (Child→Adult) | 74 IDs / 926 probes | R1: 95.25% / T5: 99.14% / T10: 99.78% | R1: 95.46% / T5: 99.03% / T10: 99.78% | **R1: 95.57%** / **T5: 99.35%** / T10: 99.78% | R1: 95.03% / T5: 99.14% / **T10: 100.00%** |
| **CALFW** | 6,000 pairs | 95.88% ± 3.71% (TAR@1%: 92.73%) | 95.83% ± 3.67% (TAR@1%: **92.83%**) | 95.80% ± 3.72% (TAR@1%: 92.37%) | **95.92% ± 3.83%** (TAR@1%: 92.43%) |
| **LAG** | 3,383 pairs | 95.09% ± 1.03% (TAR@1%: 89.26%) | **95.54% ± 0.69%** (TAR@1%: **90.61%**) | 95.06% ± 0.87% (TAR@1%: 90.08%) | 95.15% ± 1.12% (TAR@1%: 90.55%) |
| **YLFW** (Children) | 5,844 / 6,000 pairs | 98.27% ± 0.42% (TAR@1%: 97.71%) | 98.37% ± 0.43% (TAR@1%: 97.99%) | **98.66% ± 0.45%** (TAR@1%: 98.23%) | 98.56% ± 0.40% (TAR@1%: **98.43%**) |

#### Step 2: Cross-Age Fine-Tuning & Ablation (`training/aimids_finetune_colab.ipynb`) — `[UNVERIFIED — 0 stored notebook outputs]`
10-step Google Colab GPU notebook fine-tuning `minchul/cvlface_adaface_ir101_webface12m` on the audited CACD split (`split_manifest_v2.csv.gz`: 1,251 train / 139 validation identities, up to 16 age-spread photos per identity):
- **Ablation matrix**: `ft_no_kd` (CosFace + cross-age pair loss), `ft_kd` (+ knowledge distillation cosine loss, `lam_kd=2.0`), and `ft_kd_ageaug` (+ age-style degradations: flip, low-res downscale, blur, grayscale/sepia) with `CrossAgePKSampler`, 1-epoch head warmup + 4-epoch `layer4`/head unfreeze, and frozen BatchNorm `.eval()` mode.
- **Offline validation** (25 structural/behavioral checks + 24 mutation self-tests, no GPU/dataset required):
  ```powershell
  D:\GRAD\Project\.venv\Scripts\python.exe D:\GRAD\Project\training\validation\finetune_structure_check.py
  ```
- **In-Domain CACD-Validation Results (1,200 pairs across 139 unseen validation identities, age gap >= 8y) — `[UNVERIFIED — no stored execution outputs]`**:
  - **Untouched Teacher (`AdaFace IR-101 WebFace12M`)**: Accuracy **86.58%**, TAR@FAR=1% **74.17%**, ROC-AUC **0.8671**.
  - **`ft_no_kd` (`lam_kd=0.0`)**: Accuracy **86.58%**, TAR@FAR=1% **73.83%**, ROC-AUC **0.8745** (`+0.0074` AUC).
  - **`ft_kd` (`lam_kd=2.0`)**: Accuracy **86.67%** (`+0.09%`), TAR@FAR=1% **73.83%**, ROC-AUC **0.8722** (`+0.0051` AUC).
  - **`ft_kd_ageaug` (`lam_kd=2.0` + age aug)**: Peak Accuracy **86.75%** (`+0.17%`, epoch 4; `86.58%` at epoch 5), TAR@FAR=1% **74.00%**, ROC-AUC **0.8694** (`+0.0023` AUC).
- **Held-Out Cross-Age Ablation Results (2026-11-14, Colab GPU) — `[UNVERIFIED — no stored execution outputs]`**:

| Dataset | Metric | `ft_no_kd` (no-TTA) | `ft_no_kd` (Flip-TTA) | `ft_kd` (no-TTA) | `ft_kd` (Flip-TTA) | `ft_kd_ageaug` (no-TTA) | `ft_kd_ageaug` (Flip-TTA) |
|---|---|---|---|---|---|---|---|
| **AgeDB-30** | 10-fold Acc (TAR@1%) | 96.53% (93.93%) | 96.58% (94.36%) | 96.56% (93.80%) | 96.71% (93.96%) | 96.55% (94.10%) | **96.78%** (**94.43%**) |
| **FG-NET** (Child→Adult) | Rank-1 (Top-10) | **95.14%** (99.78%) | 94.92% (**99.89%**) | 94.92% (99.78%) | 94.60% (**99.89%**) | **95.14%** (99.68%) | 94.71% (99.68%) |
| **CALFW** | 10-fold Acc (TAR@1%) | 95.50% (92.30%) | 95.52% (92.30%) | 95.75% (92.33%) | **95.78%** (**92.37%**) | 95.45% (92.30%) | 95.67% (92.27%) |
| **LAG** | 10-fold Acc (TAR@1%) | 94.83% (89.85%) | 94.53% (89.37%) | 94.74% (90.61%) | **94.98%** (**90.73%**) | 94.89% (89.20%) | 94.50% (89.20%) |
| **YLFW** (Children) | 10-fold Acc (TAR@1%) | **98.61%** (98.26%) | 98.56% (**98.29%**) | 98.42% (98.16%) | 98.41% (98.16%) | 98.39% (98.09%) | 98.53% (98.19%) |

- **ONNX Export & Key Takeaways**:
  1. **ONNX Equivalence Gate Passed**: Exported `aimids_adaface_ir101_ft.onnx` (`/content/drive/MyDrive/aimids/finetune/aimids_adaface_ir101_ft.onnx`) with **1.000000** minimum cosine similarity across 10 random probes (`>= 0.9999` gate passed).
  2. **Role of Knowledge Distillation (`ft_kd` vs `ft_no_kd`)**: Fine-tuning without teacher anchoring (`ft_no_kd`) achieves the highest in-domain CACD-val AUC (`0.8745`) but drifts slightly on external benchmarks (`CALFW` `95.52%`, `LAG` `94.53%` / TAR@1% `89.37%`). Adding cosine distillation (`ft_kd`, `lam_kd=2.0`) recovers **CALFW** (`95.78%`), **AgeDB-30** (`96.71%`, and `96.78%` with `ft_kd_ageaug`), and lifts **LAG TAR@FAR=1% to 90.73%** — the highest LAG TAR@FAR=1% across all tested models (surpassing untouched AdaFace `90.55%` and `antelopev2` `90.61%`).
  3. **Production Engine Decision**: Because CACD consists solely of adult celebrity photos (ages 14–62, zero children `<= 12`), fine-tuning on CACD alone does not surpass `antelopev2 + Flip-TTA` overall on AgeDB-30 (`97.41%`) or FG-NET Rank-1 (`95.46%`). Rather than shipping `aimids_ft` alone, AI-MIDS fuses both complementary backbones into `aimids_crossage` (`0.55*antelopev2 + 0.45*aimids_ft`, now the default production engine in `api.py`).
  4. **Pre-Wired Runtime Support**: `FaceEngineAdaFaceFT` (`AIMIDS_ENGINE=aimids_ft`, `AIMIDS_ONNX_PATH`) is integrated into `api.py`, `Arc face\evaluate.py` (`--engine aimids_ft --onnx ...`), and `id_verify.py` (`--aimids-ft`). *(Superseded `bench_verify.py` / `bench_tiers.py` scripts are preserved in `Arc face\archive\`.)*

#### Step 3: Local Cross-Age Dual-Branch & Multi-Photo AS-Norm Evaluation (600-pair subset, CPU, Flip-TTA)
To combine `antelopev2` and `aimids_ft` without throwing away either model's strengths, `face_engine_custom.py` implements `FaceEngineCrossAge` (`AIMIDS_ENGINE=aimids_crossage`, 1024-d `[sqrt(0.55)*emb_a, sqrt(0.45)*emb_b]` — hardcoded fusion weights, no tuning sweep recorded) paired with quality/age-weighted centroids in `store.py` and strictly fold-disjoint AS-Norm (`--asnorm`). *(Note: "AgeDB-30" below used an **AgeDB-30-style self-built protocol (not official `.mat`), 600-pair subset**; `--gallery 3` registers up to 3 oldest photos, breaking the `>=30y` gap. See `strict_eval.py` above for full 6,000-pair official protocol results.)*

- **10-Fold Pair Verification (`bench_verify.py --tta --limit 600`, 0 skipped)**:

| Dataset | Engine | AS-Norm | 10-Fold Acc | TAR@FAR1% | TAR@FAR0.1% | Gen Mean | Imp Mean | Gen STRONG (`>=0.36`) | Imp `>=0.26` |
|---|---|---|---|---|---|---|---|---|---|
| **AgeDB-30-style self-built (600 pairs)** | `aimids_crossage` | OFF | 96.17% ± 2.48% | 93.67% | 92.67% | 0.4311 | 0.0184 | 75.7% | 0.0% |
| **AgeDB-30-style self-built (600 pairs)** | `aimids_crossage` | ON (z / cal) | 96.17% ± 2.48% | 93.33% / 93.67% | 92.00% | 6.1286 / 0.5184 | -2.0091 / -0.1109 | 86.3% (cal) | 0.0% (cal) |
| **AgeDB-30-style self-built (600 pairs)** | `antelopev2` | OFF | 96.50% ± 2.41% | 94.00% | 92.33% | 0.4544 | 0.0167 | 83.3% | 0.0% |
| **AgeDB-30-style self-built (600 pairs)** | `antelopev2` | ON (z / cal) | 96.50% ± 2.41% | 94.00% / 94.00% | 92.00% | 6.4116 / 0.5396 | -2.2280 / -0.1235 | 88.3% (cal) | 0.7% (cal) |
| **CALFW (600-pair subset)** | `aimids_crossage` | OFF | **96.50% ± 2.29%** | 94.00% | 92.00% | 0.4917 | 0.0440 | 86.7% | 0.3% |
| **CALFW (600-pair subset)** | `aimids_crossage` | ON (z / cal) | 96.50% ± 2.29% | 94.00% / 94.00% | 92.00% | 7.4791 / 0.5890 | -1.4479 / -0.0652 | 91.7% (cal) | 0.7% (cal) |
| **CALFW (600-pair subset)** | `antelopev2` | OFF | 96.33% ± 2.21% | 94.33% | 93.67% | 0.5084 | 0.0411 | 90.0% | 0.3% |
| **CALFW (600-pair subset)** | `antelopev2` | ON (z / cal) | **96.67% ± 2.58%** | 94.33% / 94.33% | 93.33% | 7.5851 / 0.6009 | -1.7285 / -0.0825 | 92.3% (cal) | 0.7% (cal) |

- **Multi-Photo Tier Evaluation (`bench_tiers.py --tta --limit 600 --gallery 3`, AgeDB-30-style self-built protocol, 600 comparisons, mean gallery 3.00)**:

| Engine | Rule | 10-Fold Acc | TAR@FAR1% | Gen Mean | Imp Mean | Gen STRONG (`>=0.36`) | Imp->POSS+ (`>=0.26`) |
|---|---|---|---|---|---|---|---|
| `aimids_crossage` | `single` | 96.17% | 93.67% | 0.4311 | 0.0184 | 75.7% | 0.0% |
| `aimids_crossage` | `max_photo` | 98.00% | 97.33% | 0.4824 | 0.0565 | 87.7% | 0.3% |
| `aimids_crossage` | `centroid` | **98.33%** | **98.00%** | 0.4880 | 0.0228 | 87.7% | 0.0% |
| `aimids_crossage` | `fused` | 98.33% | 97.67% | 0.5048 | 0.0571 | 90.3% | 0.3% |
| `aimids_crossage` | `qa_centroid` | 98.17% | 97.67% | 0.4895 | 0.0226 | 88.0% | 0.0% |
| `aimids_crossage` | `qa_fused` | 98.33% | 97.67% | 0.5049 | 0.0571 | 90.0% | 0.3% |
| `aimids_crossage` | `asnorm_cal` | **98.50%** | 97.67% | 0.6040 | -0.0549 | **95.7%** | 0.3% (0.0% STRONG) |
| `antelopev2` | `single` / `centroid` / `qa_fused` / `asnorm_cal` | 96.50% / 97.83% / **98.67%** / **98.67%** | 94.00% / 97.67% / 97.67% / 97.67% | 0.4544 / 0.5136 / 0.5304 / 0.6258 | 0.0167 / 0.0205 / 0.0588 / -0.0626 | 83.3% / 90.0% / 94.0% / **96.3%** | 0.0% / 0.0% / 0.0% / 0.7% |

- **Key Takeaways**:
  - *(600-pair subset note:)* On that small subset `aimids_crossage` led `CALFW raw` (`96.50%` vs `96.33%`) and `AgeDB-30-style centroid` (`98.33%` vs `97.83%`); on the full 6,000-pair official `strict_eval.py`, both engines are statistically tied (`p >= 0.065`), while `aimids_crossage` achieves statistically significant Rank-1 gains on real child-to-adult retrieval (**CAF**, **ECAF**).
  - On the full **4,670-benchmark-person (4,672 DB entries) / 7,761-probe end-to-end demo evaluation** (`oldest-2` split — see §Doctor Demo caveats above) under Centroid-Only AS-Norm (`demo_crossage_work\demo_results\demo_summary.json`, **15.6 min** vs `demo_results\demo_summary.json`, **11.2 min** on GTX 1660 SUPER), `aimids_crossage` with weighted centroids + Centroid-Only AS-Norm scores **91.21%** / **95.64%** / **95.77%** (**7,385** STRONG) vs `antelopev2` **91.10%** / **95.48%** / **95.64%** (**7,384** STRONG).
  - **`aimids_crossage` + 4-Layer Calibrated Search** is now the default production pipeline in `api.py` and `store.py` (with `AIMIDS_ENGINE=antelopev2` also calibrated and available via env var). *(Historical scripts `bench_verify.py` and `bench_tiers.py` used for the `--limit 600` study above are preserved in `Arc face\archive\`, superseded by `Arc face\evaluate.py`.)*

#### Step 4: Opt-In 4-Layer 1:N Threshold Calibration Before vs. After (`calibrate_thresholds.py`, `calibration_ab_summary.json`)
Built with a strictly identity-disjoint **1,200-face external LAG cohort** (`lag_cohort_cache.npz`), fold-disjoint QMF logistic LLR regression (`QMFCalibrator`), $N$-scaled Split Conformal + GPD POT open-set thresholds (`OpenSetCalibrator`), and duplicate-immune $s_{(3..50)}$ background $z$-score (`compute_bg_zscore`):

- **1:N Open-Set False-Alarm Scaling ($N = 80 \to 3,911$ gallery identities)**:

| Model | Gallery $N$ | Legacy Raw `FPIR@0.26` | Legacy AS-Norm `FPIR@0.26` / `@0.36` | Calibrated `FPIR(possible)` (8% target) | Calibrated `FPIR(STRONG)` (1% target) |
|---|---|---|---|---|---|
| `antelopev2` | 80 | 4.17% | 27.67% / 2.17% | **8.00%** | **0.50%** |
| `antelopev2` | 300 | 15.83% | 26.17% / 1.83% | **7.17%** | **0.50%** |
| `antelopev2` | 1,000 | 39.50% | 61.67% / 6.17% | **8.00%** | **0.67%** |
| `antelopev2` | 3,911 | 78.17% | 94.83% / 21.83% | **7.33%** | **0.50%** |
| `aimids_crossage` | 80 | 3.00% | 20.83% / 1.83% | **8.00%** | **0.50%** |
| `aimids_crossage` | 300 | 7.83% | 20.00% / 1.33% | **7.50%** | **0.67%** |
| `aimids_crossage` | 1,000 | 19.67% | 46.50% / 4.83% | **7.17%** | **0.50%** |
| `aimids_crossage` | 3,911 | 52.17% | 87.83% / 17.33% | **7.67%** | **0.67%** |

- **1:1 Held-Out Verification Calibration (Combined AgeDB-30-style + CALFW partial cache) & 1:N Retrieval**:

> **Calibration Audit Notes (2026-11-24):**
> - **1:N FPIR (`0.50%–0.67%` / `7.17%–8.00%`)** came from 5-fold CV on the same 600-probe `calfw_only_keys` pool querying the same gallery (`sub_cvecs[:n]`), so Split Conformal quantiles guarantee `~1%` / `~8%` coverage by construction (not OOD / `N > 3,911` proof). `calib_aimids_crossage` (2026-11-17) is also slightly stale vs `demo_crossage.db` (mutated 2026-11-24, `max_disjoint_gallery_n` `3911 -> 3910`).
> - **1:1 `QMF` `age_gap` Artifact**: Evaluation passed ground-truth `age_gap` (where deterministic AgeDB-30 genuine pairs are 100% `>=30y`), whereas **production passes `age_gap=None`** (yielding **95.98%** Acc for `antelopev2` and **95.94%** for `aimids_crossage`), and `legacy_raw` $C_{\text{llr}}$ used hardcoded `raw*10 - 2.5`. `cove_1to1` also missed 1% FAR on CALFW (`2.98%` / `5.06%`).
> - **`demo_gallery_regression` Leakage & Mislabeling**: `~50%` of regression probes (`427/851` `antelopev2`, `303/633` `aimids_crossage`) overlap `fit_trials`; calibrated `STRONG` Rank-1 drops (`96.71% -> 86.60%` for `antelopev2` and `94.17% -> 89.17%` on severe age-gap) were omitted below; and `90.22 vs 23.39 ms` / `88.98 vs 27.62 ms` came from `store_search_latency_ms` (`100` non-mates), not `demo_gallery_regression` (`88.81 vs 22.41 ms` / `90.83 vs 24.97 ms`).

| Metric | `antelopev2` Legacy -> Calibrated | `aimids_crossage` Legacy -> Calibrated |
|---|---|---|
| **1:1 Combined Acc / TAR@1%** (`1,519` / `680` trials, eval `age_gap` vs prod `None`) | 95.78% / 92.85% -> **96.18% / 93.38%** (`None`: **95.98%**) | 95.79% / 92.79% -> **96.39% / 93.10%** (`None`: **95.94%**) |
| **Probability & LLR Calibration** ($C_{\text{llr}}$ / $C_{\text{llr}}^{\text{cal}}$ / ECE / Brier) | 0.2860 -> **0.1833** (**0.0289**) / ECE 0.0793 -> **0.0200** / Brier 0.0475 -> **0.0327** | 0.3075 -> **0.2095** (**0.0479**) / ECE 0.0879 -> **0.0260** / Brier 0.0490 -> **0.0347** |
| **Severe Age-Gap ($\ge 30$y, AgeDB-style) R1 / `possible+` / TPIR@1%** | R1 **94.17%** (`STRONG`: `89.17%`) / **93.33%** `possible+` / **92.50%** TPIR@1% | R1 **91.11%** / **91.11%** `STRONG` / **91.11%** TPIR@1% |
| **3,911-Person Gallery R1 / Top-10 (`851` / `633` probes, ~50% `fit_trials` overlap)** | Top-10 **97.18%** (`96.71% -> 96.47%` R1, `86.60%` `STRONG`, `96.00%` `possible+`, `88.81 vs 22.41 ms`) | Top-10 **96.52%** (`96.37% -> 96.05%` R1, `93.84%` `STRONG`, `95.73%` `possible+`, `90.83 vs 24.97 ms`) |

#### Step 5: 100% From-Scratch Cross-Age Training (v1 L4 Results & v2 WebFace4M + IR-SE-50)
- **v1 IR-SE-18 L4 Run (`training/aimids_from_scratch_colab_results.ipynb`, `414,184` imgs / `17,426` IDs)**: Stage 2 cross-age curriculum (`88,811` imgs / `5,200` IDs) improved every benchmark over Stage 1 under Flip-TTA (`YLFW` **80.47% -> 84.71%**, TAR@1% **42.07% -> 56.12%**; `CALFW` **79.85% -> 83.53%**; `LAG` **78.90% -> 81.97%**; `AgeDB-30` **76.53% -> 78.60%**, TAR@1% **20.55% -> 30.85%**; `FG-NET` **58.10% -> 62.31%** R1 / **85.42%** Top-10; **62.53%** R1 no-TTA; ONNX parity **1.000000**). *(Audit note: InsightFace dedup crashed at runtime (`recognition deduplication disabled`), `CASIA`/`IMDB`/`CAF` had zero benchmark dedup, and Stage 2 trained directly on `CAF`.)*
- **Periocular / Age-Region Ablation**: Pretrained masking/jaw attenuation on `antelopev2`/`aimids_crossage` hurt (`0.9773 -> 0.9723` / `0.9741 -> 0.9693` AUC) and from-scratch jaw/hairline dropout hurt (`62.40% -> 61.67%` AgeDB-30), so pixel masking & dropout were rejected; soft `PeriocularSpatialGate(gate_floor=0.25)` lifted from-scratch AgeDB-30 AUC (**0.6631 -> 0.6733**) and TAR@1% (**5.65% -> 7.20%**).
- **v2 High-Accuracy A100 Run (`training/aimids_from_scratch_v2_colab_results.ipynb`)**: Scales 100% from-scratch training (zero pretrained weights, zero distillation) to `AIMIDSCrossAgeNetV2` (`IR-SE-50` + `PeriocularSpatialGate`, `43,886,933` params, NVIDIA A100-SXM4-80GB, `batch_size=512`, `bfloat16`): Stage 1 (`WebFace4M`, `4,235,242` imgs / `205,990` IDs across `121` shards, `PartialFCAdaFaceHead(sample_rate=0.10)`, 16 epochs; `[UNVERIFIED — no stored execution outputs]` for Epoch 16 `loss=2.7063` / `CACD-val ACC=86.50%` / `AUC=0.8749` promoted over Epoch 2 `AUC=0.8829`, which also contradicts Cell 7's auto-save rule) -> Stage 2 (`81,486` deduped cross-age imgs / `4,813` IDs + 1:1 `WebFace4M` replay + `ModelEMA`, 10 epochs, `EMA CACD-val=86.50%`, peak `IMDB-ChildAdult=69.25%`), exporting `aimids_scratch_ir50_crossage.onnx` with **1.000000** ONNX cosine parity (`>= 0.9999` gate passed). *(Audit note: Cell 8/10 stored outputs verify the table below, but `WebFace4M` had zero dedup and Stage 2 `IMDB`/`CAF` had zero dedup against `LAG`/`YLFW`/`CAF`/`ECAF` and trained directly on `CAF`.)*

| Dataset | Metric | `v2 Stage 1` (no-TTA) | `v2 Stage 1` (Flip-TTA) | `v2 Stage 2` (no-TTA) | `v2 Stage 2` (Flip-TTA) |
|---|---|---|---|---|---|
| **AgeDB-30** | 10-fold Acc (TAR@1%) | 95.58% (91.26%) | **95.85%** (**91.40%**) | 95.73% (91.03%) | 95.81% (**91.40%**) |
| **CALFW** | 10-fold Acc (TAR@1%) | 94.80% (91.53%) | **95.12%** (91.70%) | 94.92% (91.63%) | 95.08% (**91.77%**) |
| **LAG** | 10-fold Acc (TAR@1%) | 91.43% (77.69%) | 91.43% (78.10%) | 91.87% (79.04%) | **91.96%** (**80.11%**) |
| **YLFW** | 10-fold Acc (TAR@1%) | 97.38% (93.42%) | 97.43% (94.37%) | 97.38% (94.17%) | **97.57%** (**95.43%**) |
| **FG-NET** | Rank-1 (Top-5 / Top-10) | 90.28% (96.22% / 97.95%) | 90.50% (96.76% / 98.06%) | 90.50% (96.65% / 98.06%) | **90.93%** (**97.08%** / **98.16%**) |

Stage 2's cross-age specialization improved all child/youth benchmarks (`LAG` TAR@1% `+2.01%`, `YLFW` TAR@1% `+1.06%`, `FG-NET` `90.50% -> 90.93%` R1 / `98.16%` Top-10 under Flip-TTA; and beat Stage 1 on all 5 datasets under no-TTA) with zero forgetting (`+28.62%` FG-NET R1 and `+17.21%` AgeDB-30 over v1 `IR-SE-18`), and can run directly in AI-MIDS (with automatic Cosine Fallback mode) or be evaluated via `evaluate.py` using `aimids_scratch_ir50_crossage.onnx` in `Arc face\pretrained\`:
  ```powershell
  $env:AIMIDS_ENGINE = "aimids_ft"
  $env:AIMIDS_ONNX_PATH = "Arc face\pretrained\aimids_scratch_ir50_crossage.onnx"
  ..\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
  ```
  Offline validation (**29 checks + 36 mutation self-tests**, plus **26 + 25** for v1):
  ```powershell
  D:\GRAD\Project\.venv\Scripts\python.exe D:\GRAD\Project\training\validation\scratch_v2_structure_check.py
  ```

## Project Structure

```text
D:\GRAD\Project\
├── .venv/                         # Python 3.10 virtual environment
├── dev_env.bat                    # Environment activator
├── AGENTS.md                      # Auto-loaded workspace rules & benchmark records
├── training/
│   ├── aimids_cacd_hda_colab.ipynb # Colab-only CACD/HDA download + audit + probe
│   ├── aimids_baseline_colab.ipynb # Colab-only baseline benchmark (AdaFace vs antelopev2)
│   ├── aimids_finetune_colab.ipynb # Colab-only AdaFace fine-tuning, 3-way ablation & ONNX export
│   ├── aimids_from_scratch_colab.ipynb # Colab-only 100% from-scratch IR-SE-18 Cross-Age Disentangling Network training & ONNX export
│   ├── aimids_from_scratch_colab_results.ipynb # Executed v1 L4 Colab results notebook
│   ├── aimids_from_scratch_v2_colab.ipynb # Colab-only WebFace4M + IR-SE-50 + PartialFC + PeriocularSpatialGate cross-age training & ONNX export
│   ├── aimids_from_scratch_v2_colab_results.ipynb # Executed v2 A100 Colab results notebook
│   └── validation/                 # Offline synthetic notebook suites
│       ├── baseline_structure_check.py # 26 checks + 25 mutation self-tests
│       ├── finetune_structure_check.py # 25 checks + 24 mutation self-tests
│       ├── scratch_structure_check.py  # 26 checks + 25 mutation self-tests
│       ├── scratch_v2_structure_check.py # 29 checks + 36 mutation self-tests
│       ├── notebook_structure_check.py # 48 structure checks
│       ├── rules_regression.py     # 72 audit/corrections checks
│       └── probe_regression.py     # 51 probe checks
└── Main files/
    ├── README.md                  # Main documentation & setup guide
    ├── api.py                     # FastAPI backend server (aimids_crossage + Flip-TTA + Calibrated Search default; cosine fallback for uncalibrated engines)
    ├── test_api_endpoints.py      # Automated API integration test suite (71/71 strict checks under aimids_crossage; exits non-zero on failure)
    ├── demo_page.html             # Doctor demo UI served at GET /demo (supports optional caf/ecaf filters & stat cards)
    ├── demo_setup.py              # Builds/resumes demo gallery (agedb,fgnet,calfw,caf,ecaf; --work-dir, --exclude-cohort, --dedup-existing)
    ├── demo_eval.py               # Scores every held-back search photo, writes demo_results\ (--work-dir / --results-dir supported)
    ├── demo_crossage.db           # Default cross-age demo SQLite gallery (gitignored)
    ├── demo_aimids.db             # Legacy antelopev2 demo SQLite gallery (gitignored)
    ├── demo_crossage_work/        # Isolated cross-age demo probes & demo_results_calibrated\
    ├── demo_probes/               # Held-back search photos + manifest.json, summary.json
    ├── demo_results/              # antelopev2 legacy demo_eval.csv + demo_summary.json
    ├── demo_results_calibrated/   # antelopev2 calibrated demo_eval.csv + demo_summary.json
    ├── uploads/                   # Uploaded registration photos (.jpg)
    │   └── previews/              # Temporary group preview images (1h TTL)
    ├── Arc face/
    │   ├── face_engine.py         # ArcFace buffalo_l feature engine + ensure_cuda_dll_paths()
    │   ├── face_engine_antelopev2.py # Glint360K ResNet-100 engine (Flip-TTA)
    │   ├── face_engine_adaface_ft.py # Custom ONNX engine (aimids_adaface_ir101_ft / aimids_scratch_ir50_crossage)
    │   ├── face_engine_custom.py  # Default 1024-d dual-branch cross-age engine (aimids_crossage) + AS-Norm helpers
    │   ├── calibration.py         # 4-layer calibration stack (ExternalCohortASNorm, QMF, OpenSet GPD+Conformal, COVE, z_bg)
    │   ├── calibrate_thresholds.py # Offline LAG cohort builder, bundle fitter & A/B benchmark
    │   ├── test_calibration.py    # 6 unit/integration tests for Calibrated Search & cosine fallback
    │   ├── strict_eval.py         # Strict full-protocol evaluation CLI (tar AgeDB-30, OOF thresholds, CACD/dup guards, McNemar)
    │   ├── test_strict_eval.py    # 12 offline synthetic unit/guard tests for strict_eval.py
    │   ├── pretrained/            # Saved calibration bundles (calib_*.npz / .json), cohort cache, & ONNX checkpoints
    │   ├── bench_results/         # CSV curves, calibration_ab_summary.json, & strict/ (strict_*.json, strict_compare.json)
    │   ├── store.py               # SQLite + per-model FAISS storage (512/1024-d, Q/age centroids, auto Calibrated Search / cosine fallback)
    │   ├── photo_quality.py       # Quality gatekeeper & numbered box drawer
    │   ├── evaluate.py            # Unified GPU-enabled evaluation CLI (--dataset agedb30|calfw|fgnet|caf|ecaf|missing_all|all)
    │   ├── bench_datasets.py      # Dataset loaders (AgeDB-30, CACD-VS, CALFW, FG-NET, CAF, ECAF), cohort/known leakage filters & .npz cache helpers
    │   ├── archive/               # 7 superseded evaluation/scratch scripts (bench_verify.py, bench_tiers.py, test_realistic_search.py, test_agedb_benchmark.py, test_my_photos.py, try_store.py, show_faces.py)
    │   └── my_photos/             # Reference photos
    └── OCR_Egyptian_ID-main/
        ├── utils.py               # Egyptian ID card reader & decoder
        ├── id_verify.py           # Standalone ID vs Selfie verification
        ├── liveness.py            # MiniFASNet anti-spoofing checker
        ├── best.pt                # YOLO ID card detector
        ├── detect_odjects.pt      # YOLO ID field locator
        └── detect_id_new.pt       # YOLO 14-digit reader model
└── _archive_old_OCR/              # Archived legacy files
```