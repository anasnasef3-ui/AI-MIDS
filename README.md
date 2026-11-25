# AI-MIDS (AI Missing Identity Detection System)

**AI-MIDS** is an end-to-end AI backend and demo UI built for two core missions:
1. **Missing-Persons Cross-Age Search**: Register missing individuals (children or adults) with one or more photos, and search with newer photos across 10–25+ year age gaps or group photos.
2. **Egyptian National ID Verification**: Detect an Egyptian national ID card, locate fields, read and structurally validate the 14-digit national ID (digit count, century digit, governorate code, and real calendar birth date; the registry's 14th check digit has no publicly documented algorithm), decode demographics (birth date, governorate, gender), crop the ID portrait, and verify it against a live selfie with anti-spoofing liveness detection.

> **Deployment Notice**: AI-MIDS is currently a **local demo service with no authentication**. Keep `uvicorn` bound to `127.0.0.1` and do not expose it directly to a public network.

---

## Architecture Overview

* **Face Embedding Engines (`Arc face/`)**:
  * **`aimids_crossage` (Default, `face_engine_custom.py`, 1024-d)**: Runs one shared SCRFD-10G detector pass, extracts Flip-TTA embeddings from `antelopev2` (512-d, `0.55` weight) and cross-age fine-tuned AdaFace IR-101 `aimids_ft` (512-d, `0.45` weight), concatenates and L2-normalizes to 1024-d, and scores per-face quality in `[0.25, 1.0]`. Automatically falls back to `antelopev2` → `buffalo_l` if custom ONNX weights are missing.
  * **`antelopev2` (`face_engine_antelopev2.py`, 512-d)**: InsightFace Glint360K ResNet-100 with optional Flip-TTA (`AIMIDS_TTA=1`).
  * **`aimids_ft` (`face_engine_adaface_ft.py`, 512-d)**: Drop-in custom ONNX backbone configured via `AIMIDS_ONNX_PATH` (supports `aimids_adaface_ir101_ft.onnx` or our 100% from-scratch `aimids_scratch_ir50_crossage.onnx`).
  * **`buffalo_l` (`face_engine.py`, 512-d)**: Lightweight fallback ArcFace ResNet-50 engine.
* **Vector Storage & Search (`Arc face/store.py` + `calibration.py`)**:
  * **SQLite + Per-Model FAISS (`IndexIDMap(IndexFlatIP)`)**: Every embedding is tagged with `faces.model`, isolating `512`-d and `1024`-d indexes strictly per engine. Multiple photos per person are combined into a quality- and inverse-age-density-weighted centroid.
  * **No Score Cutoff**: Always returns a ranked Top-10 shortlist (`min_sim=-1.0`) labeled with confidence tiers (`STRONG`, `possible`, `weak`) so severe 15–25+ year child-to-adult matches are never silently hidden.
  * **Calibration v2 (`SCHEMA_VERSION = 2`, Default for `aimids_crossage` & `antelopev2`, plus `fromscratchv2` via `AIMIDS_ONNX_PATH`)**: Automatically activates when `Arc face/pretrained/calib_<model>.npz` exists:
    1. **Cross-Age Subspace Projection + Adult-Only Reference Cohort**: Keeps raw SQLite embeddings and 1:1 `/api/verify-id` untouched. For 1:N search, lazily projects vectors orthogonal to the top-5 young-vs-adult shift directions (`proj_basis`, learned strictly inside the external `LAG` cohort) and normalizes against **600 adult faces (`LAG_ADULT`)** (`std_floor=0.05`).
    2. **Quality-Aware LLR & Soft Photo Blending**: Combines normalized similarity, raw similarity, and minimum pair quality (`q_min`) into a calibrated `LLR` and `match_prob`, blending centroid + best single photo via `c_llr + 0.35 * max(0, p_llr - c_llr)`.
    3. **Gallery-Size Adaptive Open-Set Tiers**: Adapts `STRONG` (~1% stranger FP target) and `possible` (~8% target) thresholds to the current database size `N` using empirical stranger max-score distributions + GPD tail scaling (`p_nonmate`), plus background standout `z_bg` and non-gating `advisory_flags`.
  * **Automatic Cosine Fallback**: Uncalibrated engines (e.g. `buffalo_l` or a custom ONNX without a matching `calib_*.npz`) cleanly fall back to `max(best_photo_sim, centroid_sim)` with fixed `STRONG` ($\ge 0.36$), `possible` ($\ge 0.26$), and `weak` ($< 0.26$) tiers.
* **Egyptian ID Reader & Liveness (`OCR_Egyptian_ID-main/`)**:
  * 3-stage YOLO pipeline (`best.pt`, `detect_odjects.pt`, `detect_id_new.pt`) for card detection, field localization, 14-digit OCR/validation, demographic decoding, and 25% padded ID portrait cropping.
  * MiniFASNet (V2 + V1SE) anti-spoofing liveness checker (`liveness.py`, lazily loaded on first `/api/verify-id` call) enforcing live capture and **20%–55%** face framing height.
* **Photo Quality Gatekeeper (`Arc face/photo_quality.py`)**: Checks blur, resolution, and pose angle, and generates numbered group-photo previews in `uploads/previews/` (auto-cleaned after 1 hour).

---

## Getting Started

### 1. Setup & Installation (Windows PowerShell, Python 3.10)
```powershell
git clone https://github.com/anasnasef3-ui/AI-MIDS.git
cd AI-MIDS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
* **Optional CUDA GPU Acceleration**: Install `onnxruntime-gpu==1.20.2` with `--no-deps` (keeps `numpy<2` compatible with `faiss-cpu`, `insightface`, and `torch`). On Windows, `ensure_cuda_dll_paths()` automatically registers bundled CUDA 12 / cuDNN 9 DLLs from `torch\lib` and falls back to CPU if CUDA is unavailable.
* **Optional Custom ONNX Weights (>100 MB, excluded from GitHub)**:
  * To run the default `aimids_crossage` engine, place `aimids_adaface_ir101_ft.onnx` inside `Arc face\pretrained\`.
  * To run the from-scratch `IR-SE-50` model, place `aimids_scratch_ir50_crossage.onnx` inside `Arc face\pretrained\` and set `AIMIDS_ENGINE=aimids_ft` + `AIMIDS_ONNX_PATH`.
  * Without the custom `.onnx` file, `api.py` automatically falls back to **`antelopev2` + Calibration v2** (`calib_antelopev2.npz` is included in the repo).

### 2. Running the Server & Demo UI
```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```
* Interactive Swagger API docs: **`http://127.0.0.1:8000/docs`**
* Built-in 3-tab Demo UI (Search/Register, Egyptian ID + Live Webcam Framing/Liveness, Database Manager): **`http://127.0.0.1:8000/demo`**

To run against a prebuilt demo database and held-back test probes (for example the 5-dataset gallery):
```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
$env:AIMIDS_DB_PATH="demo_all5_crossage.db"
$env:AIMIDS_DEMO_WORK_DIR="demo_all5_work"
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### 3. Environment Variables & Limits

| Variable | Default | Purpose |
|---|---|---|
| `AIMIDS_ENGINE` | `aimids_crossage` | Active face engine (`aimids_crossage`, `antelopev2`, `buffalo_l`, `aimids_ft`); auto-falls back if missing |
| `AIMIDS_TTA` | `1` | Horizontal Flip Test-Time Augmentation (`1` = on, `0` = off) |
| `AIMIDS_USE_GPU` | `1` | Use CUDA GPU when available (`0` forces CPU) |
| `AIMIDS_DB_PATH` | `Arc face\aimids.db` | SQLite database file path |
| `AIMIDS_CALIB_DIR` | `Arc face\pretrained\` | Optional override directory for `calib_<model>.npz` bundles |
| `AIMIDS_DEMO_WORK_DIR` | *(empty)* | Optional base folder for `/api/demo/probes/*` (`manifest.json`, `summary.json`, probe images) |
| `AIMIDS_ONNX_PATH` | *(empty)* | Explicit path to custom `.onnx` backbone when `AIMIDS_ENGINE=aimids_ft` (auto-resolves `calib_<stem>.npz`) |
| `AIMIDS_ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed browser CORS/Origin list for write requests |

**Safety Limits (`api.py`)**: Max upload **15 MB**, max image **20 megapixels**, `k` in `1..50`, `Host` pinned to `127.0.0.1`/`localhost`, and cross-site write requests (`POST`/`PUT`/`PATCH`/`DELETE`) require an allowed `Origin`.

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Health check: active `engine`, `calibration` (`calibrated`/`cosine_fallback`), `tta`, `gpu`, `asnorm_active`, and DB stats |
| `/api/stats` | `GET` | Counts of `people`, `photos`, and indexed photos per `model` |
| `/api/persons` | `GET` / `POST` | List registered persons / create a new person (`name`, `notes`) |
| `/api/persons/{id}` | `GET` / `DELETE` | Fetch person + relative `photo_url`s / delete person from SQLite, FAISS, and disk |
| `/api/persons/{id}/photos` | `POST` | Register a photo (quality checked; multi-face photos return `choose_face` + numbered `preview_url`) |
| `/api/photos/{face_id}` | `GET` | Serve a registered face JPEG |
| `/api/preview/{filename}` | `GET` | Serve a numbered group-photo preview (`#0`, `#1`, ...) |
| `/api/search` | `POST` | Search missing persons (single or group photo). Returns ranked Top-$k$ hits with `verdict` (`MATCH_FOUND`, `POSSIBLE_MATCH`, `NO_CONFIDENT_MATCH`), `tier`, `match_source` (`centroid`/`photo`), `similarity`, plus calibrated `calibrated_llr`, `match_prob`, `p_nonmate`, `z_bg`, and `advisory_flags` |
| `/api/verify-id` | `POST` | Verify Egyptian ID card image against live selfie with mandatory liveness (`ACCEPT`, `REVIEW`, `REJECT`, `RETRY_CARD`, `RETRY_SELFIE`, `REJECT_SPOOF`) |
| `/api/verify-id/check-framing` | `POST` | Lightweight SCRFD framing check (`20%–55%` height + centering) for live webcam capture |
| `/demo` | `GET` | Serves the 3-tab demo web UI (`demo_page.html`) |
| `/api/demo/probes/{path}` | `GET` | Serves held-back demo probe images, `manifest.json`, and `summary.json` |

---

## Verified Benchmarks

### 1. Unified 3-Engine Full-Pipeline Evaluation (`full_pipeline_eval.py`, Calibration v2)
Evaluates **`aimids_crossage`**, **`antelopev2`**, and **`fromscratchv2`** (`IR-SE-50` trained 100% from random weights on `WebFace4M` + cross-age data) under **identical Calibration v2 settings** on a single deduplicated 5-dataset gallery (**2,806 people / 6,607 gallery photos / 7,714 mated probes + 200 unseen strangers**, excluding all `LAG`, `CACD`, and calibration-fit identities; GPU + Flip-TTA, `0` failed detections, `24/24` cache parity `min_cos >= 0.9999998`):

| Engine (`Stage 1` Smoke / `Stage 3` Live HTTP Parity) | Fair-4 Datasets (`no CAF`, `6,506–6,507`) v2 R1 / T10 *(v1 R1)* | All-5 Datasets (`7,713–7,714`) R1 / T10 | `FG-NET` (`463`) R1 / T10 | `ECAF` (`3,752`) v2 R1 / T10 *(v1 R1)* | `CAF` (`1,207`) R1 / T10 | `AgeDB` (`924`) R1 / T10 | `CALFW` (`1,368`) R1 / T10 | Stranger FP (`STRONG` / `Poss+`) | Query / HTTP Latency |
|---|---|---|---|---|---|---|---|---|---|
| **`aimids_crossage`** (`30/30` S1, `30/30` S3 `0.0` diff) | **80.39%** / **91.59%** *(73.96%)* | **79.47%** / **90.42%** | **93.74%** / **98.27%** | **70.42%** / **87.90%** *(59.70%)* | **74.48%** / **84.26%** | **95.35%** / **97.29%** | **93.13%** / **94.81%** | **1.00%** / **5.50%** | `15.57` / `136.99` ms |
| **`antelopev2`** (`30/30` S1, `30/30` S3 `0.0` diff) | 79.99% / 90.84% *(73.68%)* | 79.02% / 89.56% | 92.66% / 96.33% | 70.07% / 86.78% *(59.22%)* | 73.82% / 82.85% | 94.70% / 96.86% | 92.98% / 94.66% | **1.00%** / 6.00% | `13.72` / `110.77` ms |
| **`fromscratchv2`** (`30/30` S1, `30/30` S3 `0.0` diff) | 66.20% / 81.99% *(57.27%)* | 65.34%\* / 81.04%\* | 85.96% / 94.82% | 48.56% / 71.75% *(33.34%)* | 60.73%\* / 76.06%\* | 89.18% / 95.35% | 92.39% / 94.66% | **1.00%** / 7.00% | `14.11` / `83.90` ms |

*\*`CAF` was included in `fromscratchv2` Stage 2 training, so **Fair-4** (`FG-NET`, `ECAF`, `AgeDB`, `CALFW`) is the primary fair 3-way metric. Paired exact McNemar: `aimids_crossage` vs `antelopev2` is `+0.40%` on Fair-4 R1 (`p=0.1223`, NS; `+0.75%` Top-10), while both lead `fromscratchv2` by `+14.19%` / `+13.79%` (`p < 1e-208`).*

### 2. Strict Single-Dataset Verification & Child-to-Adult Retrieval (`Arc face/strict_eval.py`)
Full official protocols (no `--limit`, strict out-of-fold 1:1 thresholds, `CACD` + duplicate + `LAG` cohort exclusion, `0` failed detections):

| Benchmark | Metric | `aimids_crossage` | `antelopev2` | McNemar (`p`) |
|---|---|---|---|---|
| **Official AgeDB-30** (`6,000` pairs) | 1:1 Acc / TAR@FAR=1% | 98.22% / 97.23% | **98.33%** / **97.30%** | `-0.12%` (`p=0.065`, NS) |
| **Official CALFW** (`6,000` pairs) | 1:1 Acc / TAR@FAR=1% | 96.05% / 92.67% | **96.07%** / **92.80%** | `-0.02%` (`p=1.00`, NS) |
| **FG-NET** (`74` IDs / `463` probes) | Raw R1 / T10 | **96.76%** / **100.00%** | 96.33% / 100.00% | `+0.43%` (`p=0.77`, NS) |
| **CAF `fully_unseen`** (`239` IDs / `1,047` probes) | Raw R1 / T10 | **88.16%** / **93.79%** | 86.34% / 92.84% | **`+1.81%`** (`p=0.00055`, **SIG**) |
| **ECAF Official `model_strict`** (`605` IDs / `3,658` probes) | Raw R1 / T10 | **83.90%** / **93.25%** | 80.95% / 91.74% | **`+2.95%`** (`p=1.1e-16`, **SIG**) |

*(In standalone Colab benchmarks, `fromscratchv2` IR-SE-50 also achieved **95.81%** AgeDB-30, **95.08%** CALFW, **91.96%** LAG, **97.57%** YLFW, and **90.93%** FG-NET R1 / **98.16%** T10 with `1.000000` ONNX parity.)*

---

## Running Tests & Evaluations

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"

# 1. Calibration v2 unit & integration tests (10/10, offline)
python "Arc face\test_calibration.py"

# 2. Strict evaluation guard & metric tests (12/12, offline)
python "Arc face\test_strict_eval.py"

# 3. Full-pipeline evaluation unit tests (7/7, offline)
python "Arc face\test_full_pipeline_eval.py"

# 4. End-to-end FastAPI integration suite (71/71; requires local private test fixtures)
python test_api_endpoints.py

# 5. Full 3-engine calibrated GPU pipeline benchmark (requires local datasets)
python full_pipeline_eval.py --engines aimids_crossage,antelopev2,fromscratchv2

# 6. Rebuild v2 calibration bundles (LAG_ADULT + Proj5 + w_h=0.35)
python "Arc face\calibrate_thresholds.py" --cohort-source lag --skip-ab

# 7. Standalone Egyptian ID + live webcam verification
python "OCR_Egyptian_ID-main\id_verify.py" ID.jpg
```

---

## Repository Structure

```text
AI-MIDS/
├── README.md                          # Project documentation & setup guide
├── requirements.txt                   # Python dependencies (numpy<2)
├── api.py                             # FastAPI backend server
├── demo_page.html                     # 3-tab demo web UI served at GET /demo
├── demo_setup.py                      # Builds/resumes demo SQLite gallery & held-back probes
├── demo_eval.py                       # Evaluates held-back demo probes against running API
├── full_pipeline_eval.py              # Unified 3-engine, 5-dataset calibrated pipeline evaluator
├── test_api_endpoints.py              # Full FastAPI integration test suite (71 checks)
├── uploads/                           # Runtime uploaded photos & temporary group previews
├── Arc face/
│   ├── face_engine.py                 # buffalo_l engine + Windows CUDA DLL registration
│   ├── face_engine_antelopev2.py      # Glint360K ResNet-100 engine (Flip-TTA)
│   ├── face_engine_adaface_ft.py      # Custom ONNX engine (aimids_ft / fromscratchv2)
│   ├── face_engine_custom.py          # Default 1024-d dual-branch engine (aimids_crossage)
│   ├── store.py                       # SQLite + per-model FAISS + Calibration v2 routing
│   ├── calibration.py                 # Calibration v2 stack (LAG_ADULT, Proj5, QMF, OpenSet, COVE, z_bg)
│   ├── calibrate_thresholds.py        # Offline v2 calibration bundle fitter
│   ├── photo_quality.py               # Photo quality gatekeeper & multi-face preview drawer
│   ├── bench_datasets.py              # Dataset loaders, NFKD/alias dedup & leakage filters
│   ├── strict_eval.py                 # Strict full-protocol 1:1 & 1:N evaluator
│   ├── evaluate.py                    # Legacy single-dataset evaluation CLI
│   ├── test_calibration.py            # Calibration v2 unit tests (10 checks)
│   ├── test_strict_eval.py            # Strict evaluator unit tests (12 checks)
│   ├── test_full_pipeline_eval.py     # Full-pipeline unit tests (7 checks)
│   ├── pretrained/                    # v2 calib_*.npz/.json bundles (+ place custom .onnx here)
│   ├── bench_results/                 # Strict & full-pipeline JSON/CSV benchmark reports
│   └── archive/                       # Superseded historical scripts
└── OCR_Egyptian_ID-main/
    ├── utils.py                       # 3-stage YOLO Egyptian ID reader & 14-digit validator
    ├── id_verify.py                   # ID card vs. selfie verification pipeline
    ├── liveness.py                    # MiniFASNet anti-spoofing & 20%–55% framing checker
    ├── best.pt                        # YOLO ID card detector weights
    ├── detect_odjects.pt              # YOLO ID field detector weights
    └── detect_id_new.pt               # YOLO 14-digit OCR weights
```
