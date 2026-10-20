# AI-MIDS (AI Missing Identity Detection System)
## Project Overview, Core Objectives & Current Milestones

**Prepared For:** Graduation Project Team & Supervisors  
**System Name:** AI-MIDS  
**Core Purpose:** An AI-powered identity platform combining long-term missing-persons recovery with automated Egyptian National ID verification.

---

## 1. Executive Summary & Vision
Every year, thousands of families lose track of loved ones—especially children who go missing and grow into unrecognizable adults over 10 to 25+ years. Standard facial recognition systems fail in these scenarios because human facial bone structure and features change drastically with age. At the same time, manual identity verification in government services, banking, and checkpoints is slow and vulnerable to fake photos or printed identity fraud.

**AI-MIDS solves both challenges in one unified system:**
1. A **Missing-Persons Search Engine** specialized in finding people across severe age progression gaps (even when searching with a single childhood photo taken 20 years earlier) or scanning crowded group photos.
2. An **Automated Egyptian National ID Verification System** that reads ID cards, extracts card demographics, and verifies the card holder against a live camera selfie while preventing spoofing (screens, paper printouts).

---

## 2. Core Project Objectives

### Objective 1: Solve the "Age Gap" Problem in Missing Persons
* **The Goal:** Enable police investigators, NGOs, and families to register a missing child with old childhood photos, and find them decades later when newer adult photos or sightings are uploaded.
* **The Human-Centric Design:** In real-world aging, match similarity naturally drops. Instead of hiding potential matches behind a rigid cutoff, the system ranks the top candidates and flags them with clear confidence tiers (`STRONG`, `possible`, `weak`) so human investigators never miss a real lead.
* **Multi-Photo Advantage:** Allow families to upload multiple childhood photos from different ages. The system automatically combines them to create a persistent facial identity profile that survives aging.

### Objective 2: Intelligent Group Photo Scanning
* **The Goal:** Search for missing individuals directly inside crowded group photos (family gatherings, street photos, surveillance stills).
* **The Capability:** Detects every face in a scene simultaneously and queries each one against the national missing database in seconds.

### Objective 3: Secure, Frictionless Egyptian ID Verification
* **The Goal:** Eliminate manual data entry and human error when verifying Egyptian National IDs.
* **The Capability:** Instantly reads the card, extracts the 14-digit National ID, validates its authenticity, decodes birth date, governorate of origin, and gender, and crops the official card photograph.

### Objective 4: Anti-Spoofing & Live Identity Protection
* **The Goal:** Ensure that the person standing in front of the camera is actually present and alive—not an imposter holding up someone's printed photo or an iPhone screen showing their picture.
* **The Capability:** Integrated anti-spoofing liveness checks that verify natural facial depth and texture before approving any match.

### Objective 5: Scalable, Privacy-Safe Backend API
* **The Goal:** Provide a central backend service that any mobile app (Flutter/Android/iOS) or web portal can connect to seamlessly.
* **The Capability:** Standardized REST endpoints with fast response times, concurrency protection, and zero exposure of raw server disk paths.

---

## 3. Milestones Achieved So Far

| Milestone | What Was Achieved | Current Status |
| :--- | :--- | :---: |
| **1. Missing Persons Search Engine** | Completed and benchmarked against standard academic aging datasets (FG-NET and AgeDB). Achieved up to **88.3% Rank-1 accuracy** and **100% Top-10 recovery** across 20–50 year age gaps. | ✅ **Complete** |
| **2. Egyptian ID OCR & Decoding** | Automated 3-stage visual pipeline that detects the card, reads the 14-digit National ID without confusing security background patterns, and decodes age, birth date, governorate, and gender. | ✅ **Complete** |
| **3. Live Selfie Verification** | Matches the official Egyptian ID card photo against a live webcam selfie with high matching confidence. | ✅ **Complete** |
| **4. Anti-Spoofing Liveness Gate** | Implemented live anti-spoofing with calibrated framing rules to block paper printouts and screen replays. | ✅ **Complete** |
| **5. Photo Quality Gatekeeper** | Automatically filters out unusable blur or extreme side-profile photos at registration time, while accommodating old family scans with gentle warnings. | ✅ **Complete** |
| **6. Central Backend API & Automated Tests** | Fully functional FastAPI server with all endpoints active, non-blocking asynchronous execution, and **35/35 automated integration tests passing**. | ✅ **Complete** |

---

## 4. Remaining Objectives & Roadmap (Next Steps)
1. **User Interface (Mobile / Web App):**
   * Build the front-facing app screens for families (Missing Persons portal) and officers/clerks (ID Verification screen with a live camera oval guide).
2. **System Cross-Check Feature:**
   * Automatically cross-reference scanned ID cards against the missing persons database (e.g. checking if an adult renewing an ID matches a child reported missing 15 years ago).
3. **Graduation Report & Presentation:**
   * Package the benchmark findings, system architecture, and use cases into our final graduation thesis documentation and slide deck.
