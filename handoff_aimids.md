# AI-MIDS Project Handoff

Paste this whole file as the first message into any new Antigravity session or AI coding assistant.

---

## 1. Project Overview & Student Context
* **Project Name**: AI-MIDS (AI Missing Identity Detection System)
* **Student Context**: Single student graduation project. Explain things in plain language without unnecessary jargon.
* **Quota-Saving Multi-Agent Workflow Rule (PERMANENT)**:
  1. **PLAN (Main Agent - Argon/Opus)**: Keep responses concise. Design the step-by-step plan without running heavy commands directly.
  2. **CODE & RUN TESTS (Gemini Flash Subagent)**: Use `invoke_subagent` with `Model="flash"` to write code, run terminal tests/benchmarks, fix syntax errors, and return a brief summary of changes and test results.
  3. **FINAL REVIEW (Main Agent - Argon/Opus)**: Inspect modified files to verify correctness. Keep terminal output clean: do NOT print long markdown diff blocks in the chat (the student inspects visual diffs directly in VS Code via Source Control `Ctrl+Shift+G`). Explain the changes to the student in 2-3 plain-language bullet points.
* **OS / Environment**: Windows. Always provide Windows PowerShell / cmd copy-paste commands.
* **Python Environment**: `D:\GRAD\Project\.venv\Scripts\python.exe` (Python 3.10).
* **Two Core Missions**:
  1. **Missing-Persons Search**: Register missing individuals (children or adults) with photos, and search using newer photos across severe age progression gaps (10 to 25+ years) or group photos.
  2. **Egyptian National ID Verification**: Detect an Egyptian national ID card, locate fields, read and validate the 14-digit national ID, decode demographics (birth date, governorate, gender), crop the card photo, and verify it against a live selfie with anti-spoofing liveness detection.

---

## 2. Project Directory Structure
All active, production files live under `D:\GRAD\Project\Main files\`. (Do not use files outside `Main files/` as older duplicates in root are archived).

```text
D:\GRAD\Project\
├── .venv\                               # Virtual environment (Python 3.10)
├── README.md                            # Synchronized project documentation
├── handoff_aimids.md                    # This handoff file
└── Main files\
    ├── README.md                        # Documentation & setup guide
    ├── handoff_aimids.md                # Copy of this handoff file
    ├── api.py                           # FastAPI backend server (antelopev2 + Flip-TTA default)
    ├── test_api_endpoints.py            # Automated integration test suite (35/35 passing)
    ├── uploads\                         # Uploaded registration photos (.jpg)
    │   └── previews\                    # Temporary group face previews (auto-cleaned after 1h)
    ├── Arc face\
    │   ├── face_engine.py               # InsightFace ArcFace buffalo_l (ResNet-50, 512-d embeddings)
    │   ├── face_engine_antelopev2.py    # InsightFace antelopev2 (Glint360K ResNet-100 + Flip-TTA support)
    │   ├── store.py                     # SQLite (aimids.db) + FAISS + Multi-Photo Centroid Fusion
    │   ├── photo_quality.py             # Blur, small face, and pose quality check
    │   ├── test_realistic_search.py     # End-to-end aging benchmark (FG-NET)
    │   ├── test_agedb_benchmark.py      # Long-gap aging benchmark (AgeDB 30+ year gaps)
    │   └── my_photos\                   # Reference test photos (Anas, Hoz, distractors)
    └── OCR_Egyptian_ID-main\
        ├── utils.py                     # 3-stage YOLO pipeline for ID card detection & OCR
        ├── id_verify.py                 # Standalone ID vs Selfie verification (25% card padding + webcam GUI)
        ├── liveness.py                  # MiniFASNet anti-spoofing (20%-37% face height framing)
        ├── best.pt                      # YOLO card detector
        ├── detect_odjects.pt            # YOLO field detector
        └── detect_id_new.pt             # YOLO 14-digit reader
```

---

## 3. Key Architectural Decisions & Latest Fixes

### A. Search & Ranking Philosophy (`store.py`, `api.py`)
* **No Score Cutoff by Default**: `/api/search` uses `min_sim = -1.0` by default. Severe age gaps (10–25+ years) cause true match cosine similarity to naturally drop to 0.10–0.25. Setting a rigid cutoff like 0.26 discards 85% of real matches.
* **3 Confidence Tiers**:
  * `"STRONG"`: $\ge 0.36$ (high-confidence match)
  * `"possible"`: $\ge 0.26$ (review candidate)
  * `"weak"`: $< 0.26$ (long-gap aging match; candidate for human reviewer)
* **Search Output & Verdicts**: Returns a ranked top-$k$ candidate list (default 10, range 1..50) for a human investigator to inspect. Each detected face includes:
  * `"verdict"`: `"MATCH_FOUND"` (if top candidate is `STRONG`), `"POSSIBLE_MATCH"` (if top candidate is `possible`), or `"NO_CONFIDENT_MATCH"` (if all candidate scores $< 0.26$).
  * `"message"`: Plain-language explanation for non-technical users.
  * `"confident_match_count"`: Count of candidates in `STRONG` or `possible` tiers ($\ge 0.26$).
  * `"match_count"`: Total candidates returned (including `weak` candidates).
* **Privacy / URLs**: Search results and person details never return absolute disk paths. They return `face_id` and relative `photo_url` (`/api/photos/{face_id}`), served by `GET /api/photos/{face_id}`.

### B. ID Verification & Liveness Rules (`api.py`, `liveness.py`, `id_verify.py`)
* **Mandatory Liveness**: Anti-spoofing liveness check always runs on `/api/verify-id` (no bypass parameter).
* **Live Capture Rule**: A static photo file of a real person passes anti-spoofing; therefore, client apps must capture the selfie live.
* **Face Framing (20%–37% Rule)**: The face height must be between **20% and 37%** of the frame height. MiniFASNet requires a $2.7\times$ context crop around the face; if the face is too large ($> 37\%$), the crop clips the image borders and causes false spoof alarms. If outside 20%–37%, the server returns `REVIEW`.
* **Client Implementation**: On phones or web apps, the frontend uses built-in face detection (Google ML Kit on Flutter/Android, Apple Vision on iOS, MediaPipe in browser) or a fixed 30% oval guide to auto-capture when the user is in range.

### C. Concurrency & Concurrency Safety (`api.py`)
* **Non-Blocking Server**: Heavy compute (`check_photo`, `engine.embed_all`, `store.search`, `verify_id`) runs via `await run_in_threadpool(...)` from `starlette.concurrency`. `GET /` responds in $\le 0.01$s even while heavy verification is running.
* **YOLO Thread-Safety**: All `verify_id` executions are wrapped with a `threading.Lock()` (`verify_lock`) to prevent concurrent YOLO model corruption.

### D. File Storage & Housekeeping (`api.py`)
* `DELETE /api/persons/{id}` deletes database records, FAISS vectors, AND the uploaded `.jpg` files from disk.
* Previews are saved in `uploads/previews/` and files older than 1 hour are automatically deleted whenever a new preview is generated.
* Registered photos are always saved as `.jpg`.

### E. Photo Quality Gatekeeper (`photo_quality.py`)
* `det_score < 0.60` warning was removed because benchmark tests proved it caused 24% false warnings with zero drop in matching accuracy.
* `SIDE_ON` pose angle was calibrated from `0.22` down to `0.15`.

---

## 4. Benchmark Results (FG-NET & AgeDB Aging Datasets)

### A. FG-NET Childhood-to-Adult Benchmark (60 probes, gallery age $\le 12$, avg gap **22.8 years**)
* **`buffalo_l` (ResNet-50 Baseline)**:
  * **Rank-1 Accuracy (no cutoff)**: **56.7%** (34/60) | **Top-5**: **78.3%** (47/60) | **Top-10**: **85.0%** (51/60)
  * **Top-10 Hit Rate (with 0.26 cutoff)**: **10.0%** (6/60 — shows why rigid cutoffs fail on aging data)
  * **Mean Target Similarity**: `0.1975` (Range: 0.081 to 0.382)
* **`buffalo_l` + Flip-TTA**: Rank-1 **58.3%** (35/60) | Top-5 **80.0%** | Mean Sim `0.2012`
* **`antelopev2` (Glint360K ResNet-100, `face_engine_antelopev2.py`)**:
  * **Without TTA**: Rank-1 **60.0%** (36/60) | Top-5 **80.0%** | Top-10 **86.7%** (52/60) | Mean Sim `0.2198`
  * **With Flip-TTA**: Rank-1 **61.7%** (37/60) | Top-10 **86.7%** (52/60) | Mean Sim `0.2244`
* **Multi-Photo Centroid Gallery (up to 3 childhood photos averaged per child)**:
  * **Rank-1 Accuracy**: **88.3%** (53/60) | **Top-5**: **98.3%** (59/60) | **Top-10**: **100.0%** (60/60) | Mean Sim `0.3376`

### B. AgeDB Long-Gap Benchmark (`test_agedb_benchmark.py`, 100 gallery identities, 60 probes, avg gap **49.5 years**)
* **`buffalo_l` (ResNet-50)**: Rank-1 **80.0%** (48/60) | Top-5 **88.3%** | Top-10 **91.7%** | Mean Sim `0.3404`
* **`antelopev2` (ResNet-100)**: Rank-1 **88.3%** (53/60) | Top-5 **90.0%** | Top-10 **91.7%** | Mean Sim `0.3986`

### C. Egyptian ID Card vs. Live Selfie Verification
* **`buffalo_l` (raw crop)**: `0.5085` $\rightarrow$ **with 25% border padding**: `0.5124`
* **`antelopev2` (with 25% border padding)**: **`0.5714`** (+11.5% higher match confidence)

---

## 5. How to Run the System

### Run the FastAPI Backend Server
```powershell
cd "D:\GRAD\Project\Main files"
& "D:\GRAD\Project\.venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Swagger UI documentation available at: `http://127.0.0.1:8000/docs`

### Run the API Integration Tests (31/31 Automated Checks)
```powershell
cd "D:\GRAD\Project\Main files"
& "D:\GRAD\Project\.venv\Scripts\python.exe" test_api_endpoints.py
```

### Run the FG-NET & AgeDB Aging Benchmarks
```powershell
cd "D:\GRAD\Project\Main files\Arc face"
& "D:\GRAD\Project\.venv\Scripts\python.exe" test_realistic_search.py
& "D:\GRAD\Project\.venv\Scripts\python.exe" test_agedb_benchmark.py
```

### Run the Standalone ID Verification with Live Webcam GUI
```powershell
cd "D:\GRAD\Project\Main files\OCR_Egyptian_ID-main"
& "D:\GRAD\Project\.venv\Scripts\python.exe" id_verify.py ID.jpg
```

---

## 6. Immediate Next Steps / Roadmap
1. **Student Code Study & Custom PyTorch Model**:
   * **Model A**: Pure PyTorch CNN + ArcFace loss trained from scratch on local `AgeDB`/`CACD` (`D:\GRAD\Assets`).
   * **Model B**: Pre-trained ArcFace (`buffalo_l` / `antelopev2`) used directly.
   * **Model C (Transfer Learning)**: Pre-trained ArcFace + custom PyTorch `AgeAdapter` residual network trained on `AgeDB` 30-year aging pairs.
   * Walk through existing files line-by-line so the student can comfortably edit and defend every module.
2. **Apply Accuracy Upgrades to Production Pipeline**:
   * Multi-photo centroid fusion in `store.py`, 25% ID photo padding in `id_verify.py`, and optional switch to `FaceEngineAntelopeV2` in `api.py`.
3. **Frontend Integration**: Build or connect a web/mobile UI to `/api/search` and `/api/verify-id` using the 20%–37% visual oval guide.
