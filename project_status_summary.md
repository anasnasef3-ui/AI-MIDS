# AI-MIDS: Project Status & Progress Summary

**Project Title:** AI Missing Identity Detection System (AI-MIDS)  
**Date:** October 2026  
**Status:** Backend & AI Models Operational (35/35 Integration Tests Passing)

---

## 1. What AI-MIDS Does (Two Core Missions)
1. **Missing-Persons Search Across Decades of Aging:**
   * Registers missing individuals (often children) using old childhood photos.
   * Matches newer adult photos (10 to 30+ year age gaps) or scans group photos to identify candidates without prematurely discarding matches.
2. **Egyptian National ID Verification:**
   * Automatically detects an Egyptian ID card from camera frames.
   * Reads and decodes the 14-digit National ID (extracting birth date, governorate, and gender).
   * Extracts the ID card photo and verifies it against a live selfie using anti-spoofing liveness detection.

---

## 2. Major Milestones & Benchmarks Achieved

### A. Face Recognition Upgrade (`buffalo_l` vs. `antelopev2`)
* Evaluated standard ArcFace (`buffalo_l` ResNet-50) against a deeper model (`antelopev2` ResNet-100 trained on Glint360K with 360,000+ identities).
* **Egyptian ID vs. Live Selfie:** Match similarity increased from `0.5124` to **`0.5714`** (+11.5% higher match confidence) by using `antelopev2` + 25% border padding on the extracted ID photo.
* Created a dedicated module [`face_engine_antelopev2.py`](file:///d:/GRAD/Project/Main%20files/Arc%20face/face_engine_antelopev2.py) with built-in **Flip Test-Time Augmentation (Flip-TTA)**.

### B. Aging Search Benchmarks on Standard Academic Datasets
We tested our search pipeline across severe real-world age gaps on two benchmark datasets:

1. **FG-NET Dataset (Childhood to Adulthood — Avg. 22.8-year age gap):**
   * Single Childhood Photo: **61.7% Rank-1**, **86.7% Top-10**.
   * **Multi-Photo Centroid Fusion:** Averaging 2–3 childhood photos per missing person boosted Rank-1 accuracy to **88.3%** and Top-10 hit rate to **100.0%** (all 60 test probes successfully found).
2. **AgeDB Dataset (30+ Year Long-Gap Adult Progression — Avg. 49.5-year age gap):**
   * `buffalo_l` (ResNet-50): 80.0% Rank-1 | Mean Similarity: 0.3404
   * **`antelopev2` (ResNet-100): 88.3% Rank-1** | Mean Similarity: **0.3986**

### C. Search Ranking & Confidence Tiers
* Rather than using a rigid score cutoff that discards 85% of real aging matches, search results return a top-10 ranked list categorized into 3 human-readable tiers:
  * **`STRONG`** ($\ge 0.36$): High-confidence match.
  * **`possible`** ($\ge 0.26$): Review candidate.
  * **`weak`** ($< 0.26$): Candidate across severe age gaps for human inspector review.

### D. Egyptian ID & Anti-Spoofing Rules
* **3-Stage YOLO OCR:** Custom models detect the card (`best.pt`), locate regions (`detect_odjects.pt`), and read digits (`detect_id_new.pt`).
* **Liveness Detection (MiniFASNet):** Enforced a **20%–37% face height framing rule** in the camera frame to eliminate false spoof rejections caused by model context clipping.

### E. Production-Ready FastAPI Backend
* Asynchronous, non-blocking backend ([`api.py`](file:///d:/GRAD/Project/Main%20files/api.py)) with threadpool execution and concurrency locks for GPU/YOLO models.
* Privacy-safe: Endpoints serve images via relative URLs (`/api/photos/{face_id}`) without exposing internal disk paths.
* Automated integration test suite ([`test_api_endpoints.py`](file:///d:/GRAD/Project/Main%20files/test_api_endpoints.py)): **35/35 checks passing**.

---

## 3. Current Project Structure
```text
Main files/
├── api.py                           # FastAPI server (search, verify-id, persons management)
├── test_api_endpoints.py            # Automated integration test suite (35/35 PASS)
├── Arc face/
│   ├── face_engine_antelopev2.py    # ResNet-100 ArcFace engine + Flip-TTA
│   ├── face_engine.py               # Buffalo_l baseline engine
│   ├── store.py                     # SQLite (aimids.db) + FAISS vector indexing
│   ├── photo_quality.py             # Blur, resolution, and pose checks
│   ├── test_realistic_search.py     # FG-NET aging benchmark script
│   └── test_agedb_benchmark.py      # AgeDB aging benchmark script
└── OCR_Egyptian_ID-main/
    ├── utils.py                     # YOLO card detection & 14-digit NID reader
    ├── id_verify.py                 # Standalone ID vs Selfie verification
    ├── liveness.py                  # Anti-spoofing checker (20%-37% rule)
    ├── best.pt, detect_odjects.pt, detect_id_new.pt
```

---

## 4. Next Steps for the Team
1. **Frontend UI (Web / Mobile):** Connect the Flutter or Web interface to the FastAPI endpoints (`/api/search` and `/api/verify-id`) with a camera guide for the 20%–37% face oval.
2. **Graduation Documentation & Slides:** Compile the FG-NET and AgeDB benchmark tables, model comparison metrics, and system architecture diagrams for the graduation thesis and presentation.
