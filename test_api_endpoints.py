"""
test_api_endpoints.py - Automated integration test for AI-MIDS FastAPI backend.

Spins up the server on an ephemeral port (8008), exercises all endpoints,
verifies JSON responses, non-blocking concurrency, file deletion, and shuts down cleanly.
"""
import os
import sys
import time
import threading
import requests
import uvicorn
import inspect
import numpy as np
import torch
import torch.nn as nn
import cv2

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "Arc face"))

# Point test DB to a temporary test file so we don't alter the real db
TEST_DB_PATH = os.path.join(HERE, "test_api_aimids.db")
if os.path.exists(TEST_DB_PATH):
    try:
        os.remove(TEST_DB_PATH)
    except Exception:
        pass
os.environ["AIMIDS_DB_PATH"] = TEST_DB_PATH

import api
from api import app
from face_engine_adaface_ft import FaceEngineAdaFaceFT
from store import FaceStore

PORT = 8008
BASE_URL = f"http://127.0.0.1:{PORT}"


class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def run_tests():
    print("=" * 70)
    print("  AI-MIDS FASTAPI BACKEND - INTEGRATION TEST SUITE")
    print("=" * 70)

    # 1. Start server
    print("\n[1] Starting API server in background on port 8008...")
    th = ServerThread()
    th.start()

    # Wait for server ready
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/", timeout=1)
            if r.status_code == 200:
                print("    [OK] Server is up and responding!")
                break
        except Exception:
            time.sleep(0.5)
    else:
        sys.exit("Error: Server failed to start within 15 seconds.")

    passed = 0
    total = 0

    def check(name, condition, extra=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name} {extra}")
        else:
            print(f"  [FAIL] {name} {extra}")

    # 2. Test GET /
    print("\n[2] Testing Health & Root Endpoint (GET /)...")
    r = requests.get(f"{BASE_URL}/")
    data = r.json()
    check("Root status online", data.get("status") == "online")
    check(
        "Database stats returned",
        isinstance(data.get("stats"), dict)
        and "people" in data["stats"]
        and "photos" in data["stats"],
    )
    active_eng = data.get("engine")
    check("Active engine returned in root", active_eng in ("aimids_crossage", "antelopev2", "buffalo_l", "aimids_ft"), f"(engine={active_eng})")
    check("tta boolean returned in root", isinstance(data.get("tta"), bool), f"(tta={data.get('tta')})")
    check("asnorm_active boolean returned in root", isinstance(data.get("asnorm_active"), bool), f"(asnorm_active={data.get('asnorm_active')})")
    if active_eng in ("aimids_crossage", "antelopev2"):
        check("Calibrated mode reported in root", data.get("calibration") == "calibrated", f"(calibration={data.get('calibration')})")
        check("asnorm_active True for calibrated engine", data.get("asnorm_active") is True, f"(asnorm_active={data.get('asnorm_active')})")
    else:
        check("Cosine fallback reported in root", data.get("calibration") == "cosine_fallback", f"(calibration={data.get('calibration')})")
        check("asnorm_active False for uncalibrated engine", data.get("asnorm_active") is False, f"(asnorm_active={data.get('asnorm_active')})")

    # 3. Test POST /api/persons (Create two persons for search & weak tier test)
    print("\n[3] Testing Person Registration (POST /api/persons)...")
    r = requests.post(f"{BASE_URL}/api/persons", data={"name": "Anas Test", "notes": "Registered via API test"})
    check("Create Anas status 201", r.status_code == 201)
    test_pid = r.json().get("person_id")
    check("Valid Anas person_id returned", isinstance(test_pid, int), f"(id={test_pid})")

    r2 = requests.post(f"{BASE_URL}/api/persons", data={"name": "Hoz Test", "notes": "Distractor for weak tier test"})
    check("Create Hoz status 201", r2.status_code == 201)
    hoz_pid = r2.json().get("person_id")

    # Clean up any leftover p_{test_pid}_* files from prior runs before starting
    upload_dir = os.path.join(HERE, "uploads")
    if os.path.isdir(upload_dir):
        for f_old in os.listdir(upload_dir):
            if f_old.startswith(f"p_{test_pid}_"):
                try:
                    os.remove(os.path.join(upload_dir, f_old))
                except OSError:
                    pass

    # 4. Test POST /api/persons/{id}/photos
    print("\n[4] Testing Photo Upload & Quality Gate (POST /api/persons/{id}/photos)...")
    child_photo_path = os.path.join(HERE, "Arc face", "my_photos", "anas", "child1_anas.jpeg")
    with open(child_photo_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/persons/{test_pid}/photos",
            files={"file": ("childhood.jpeg", f, "image/jpeg")},
            data={"age": 5}
        )
    check("Upload photo status 200", r.status_code == 200)
    photo_res = r.json()
    check("Quality check status OK/WARN", photo_res.get("status") in ("ok", "warn"))
    test_face_id = photo_res.get("face_id")
    check("Face ID assigned", isinstance(test_face_id, int))
    check(
        "Photo URL returned",
        isinstance(photo_res.get("photo_url"), str)
        and photo_res["photo_url"].startswith("/api/photos/"),
    )

    # Upload 2nd photo for Anas to exercise multi-photo centroid fusion
    child_photo2_path = os.path.join(HERE, "Arc face", "my_photos", "anas", "child2_anas.jpg")
    with open(child_photo2_path, "rb") as f:
        r_photo2 = requests.post(
            f"{BASE_URL}/api/persons/{test_pid}/photos",
            files={"file": ("childhood_2.jpg", f, "image/jpeg")},
            data={"age": 6}
        )
    check("Upload 2nd photo for centroid fusion status 200", r_photo2.status_code == 200)
    photo2_res = r_photo2.json()
    check("2nd photo quality check status OK/WARN", photo2_res.get("status") in ("ok", "warn"))
    test_face_id2 = photo2_res.get("face_id")
    check("2nd Face ID assigned", isinstance(test_face_id2, int) and test_face_id2 != test_face_id)

    # Upload distractor photo for Hoz (using portsaid.png face_index=2 which is a different person with < 0.26 similarity)
    hoz_photo_path = os.path.join(HERE, "Arc face", "my_photos", "anas", "portsaid.png")
    with open(hoz_photo_path, "rb") as f:
        r_hoz = requests.post(
            f"{BASE_URL}/api/persons/{hoz_pid}/photos",
            files={"file": ("portsaid.png", f, "image/png")},
            data={"face_index": 2}
        )
    check("Upload Hoz photo status 200", r_hoz.status_code == 200 and r_hoz.json().get("status") in ("ok", "warn"))

    # Test GET /api/photos/{face_id}
    r_photo = requests.get(f"{BASE_URL}/api/photos/{test_face_id}")
    check("GET /api/photos/{face_id} status 200", r_photo.status_code == 200)
    check("Photo served as image/jpeg", "image/jpeg" in r_photo.headers.get("content-type", ""))

    # Regression: bad input must fail as 400, never as a server error
    with open(child_photo_path, "rb") as f:
        r_bad_idx = requests.post(
            f"{BASE_URL}/api/persons/{test_pid}/photos",
            files={"file": ("childhood.jpeg", f, "image/jpeg")},
            data={"face_index": 99}
        )
    check("Out-of-range face_index rejected with 400", r_bad_idx.status_code == 400, f"(got {r_bad_idx.status_code})")
    # Content-Length over the cap must be rejected before reading the body (no 16MB allocation needed)
    r_big = requests.post(
        f"{BASE_URL}/api/persons/{test_pid}/photos",
        files={"file": ("huge.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 1024, "image/jpeg")},
        headers={"Content-Length": str(16 * 1024 * 1024 + 100)}
    )
    check("Oversized upload rejected with 400", r_big.status_code == 400, f"(got {r_big.status_code})")

    # Regression: embeddings are tagged with the engine that produced them (no silent cross-model mixing)
    root_now = requests.get(f"{BASE_URL}/").json()
    active_engine = root_now.get("engine")
    models_tagged = root_now.get("stats", {}).get("models", {})
    check("Stored photos tagged with the active engine",
          models_tagged.get(active_engine) == 3, f"(models={models_tagged}, engine={active_engine})")

    # Regression: an unknown website gets no CORS permission to read this database
    r_evil = requests.get(f"{BASE_URL}/api/persons", headers={"Origin": "https://evil.example.com"})
    check("Unknown CORS origin gets no allow header",
          r_evil.headers.get("access-control-allow-origin") is None,
          f"(got {r_evil.headers.get('access-control-allow-origin')})")

    # Regression: CORS alone does not stop a cross-site form from mutating the database
    r_cross_write = requests.post(
        f"{BASE_URL}/api/persons",
        data={"name": "Cross-site write must not be created"},
        headers={"Origin": "https://evil.example.com"}
    )
    check("Cross-site write requests are rejected with 403",
          r_cross_write.status_code == 403, f"(got {r_cross_write.status_code})")

    # Regression: a local unauthenticated API must not answer DNS-rebinding Host headers
    r_bad_host = requests.get(f"{BASE_URL}/api/stats", headers={"Host": "evil.example.com"})
    check("Untrusted Host header is rejected with 400",
          r_bad_host.status_code == 400, f"(got {r_bad_host.status_code})")

    # 5. Test GET /demo (built-in demo page)
    r_demo = requests.get(f"{BASE_URL}/demo")
    check("GET /demo serves the demo page",
          r_demo.status_code == 200 and "AI-MIDS" in r_demo.text and "text/html" in r_demo.headers.get("content-type", ""),
          f"(status {r_demo.status_code}, type {r_demo.headers.get('content-type')})")
    check("GET /demo includes 3-tab UI, ID verification, and persons manager",
          'role="tablist"' in r_demo.text
          and 'role="tabpanel"' in r_demo.text
          and "beforematch" in r_demo.text
          and "/api/verify-id" in r_demo.text
          and "/api/verify-id/check-framing" in r_demo.text
          and "framing-oval" in r_demo.text
          and "20%\u201355%" in r_demo.text
          and "framing-oval-max" not in r_demo.text
          and "autoCaptureToggle" in r_demo.text
          and "/api/persons" in r_demo.text)
    r_trav = requests.get(f"{BASE_URL}/api/demo/probes/..%2F..%2Fapi.py")
    check("Demo probe route blocks path traversal",
          r_trav.status_code in (400, 404),
          f"(status {r_trav.status_code})")

    # Test AIMIDS_DEMO_WORK_DIR override on /api/demo/probes/summary.json
    import tempfile, shutil, json
    tmp_work = tempfile.mkdtemp(prefix="aimids_demo_work_")
    old_demo_work = os.environ.get("AIMIDS_DEMO_WORK_DIR")
    try:
        tmp_probes = os.path.join(tmp_work, "demo_probes")
        os.makedirs(tmp_probes, exist_ok=True)
        with open(os.path.join(tmp_probes, "summary.json"), "w", encoding="utf-8") as sf:
            json.dump({"engine": "override_test", "overall": {"probes": 1, "rank1": 1.0, "top5": 1.0, "top10": 1.0}}, sf)
        os.environ["AIMIDS_DEMO_WORK_DIR"] = tmp_work
        r_sum_ovr = requests.get(f"{BASE_URL}/api/demo/probes/summary.json")
        check("AIMIDS_DEMO_WORK_DIR overrides /api/demo/probes/summary.json",
              r_sum_ovr.status_code == 200 and r_sum_ovr.json().get("engine") == "override_test")
    finally:
        if old_demo_work is None:
            os.environ.pop("AIMIDS_DEMO_WORK_DIR", None)
        else:
            os.environ["AIMIDS_DEMO_WORK_DIR"] = old_demo_work
        shutil.rmtree(tmp_work, ignore_errors=True)

    # If running under aimids_crossage or antelopev2 with demo summary present, verify summary.json routing
    crossage_cal_sum = os.path.join(HERE, "demo_crossage_work", "demo_results_calibrated", "demo_summary.json")
    crossage_leg_sum = os.path.join(HERE, "demo_crossage_work", "demo_results", "demo_summary.json")
    antelope_cal_sum = os.path.join(HERE, "demo_results_calibrated", "demo_summary.json")
    if api.ENGINE_NAME == "aimids_crossage" and (os.path.isfile(crossage_cal_sum) or os.path.isfile(crossage_leg_sum)) and not os.environ.get("AIMIDS_DEMO_WORK_DIR"):
        r_ca_sum = requests.get(f"{BASE_URL}/api/demo/probes/summary.json")
        ca_data = r_ca_sum.json() if r_ca_sum.status_code == 200 else {}
        check("aimids_crossage serves cross-age summary.json",
              r_ca_sum.status_code == 200 and ca_data.get("engine") == "aimids_crossage"
              and (not os.path.isfile(crossage_cal_sum) or ca_data.get("calibration") == "calibrated"))
    elif api.ENGINE_NAME == "antelopev2" and os.path.isfile(antelope_cal_sum) and not os.environ.get("AIMIDS_DEMO_WORK_DIR"):
        r_ant_sum = requests.get(f"{BASE_URL}/api/demo/probes/summary.json")
        ant_data = r_ant_sum.json() if r_ant_sum.status_code == 200 else {}
        check("antelopev2 serves calibrated summary.json",
              r_ant_sum.status_code == 200 and ant_data.get("engine") == "antelopev2"
              and ant_data.get("calibration") == "calibrated")

    # 5. Test GET /api/persons & GET /api/persons/{id}
    print("\n[5] Testing Listing and Details (GET /api/persons)...")
    r = requests.get(f"{BASE_URL}/api/persons")
    p_list = r.json()
    check("Person list returned", any(p["person_id"] == test_pid for p in p_list))

    r_det = requests.get(f"{BASE_URL}/api/persons/{test_pid}")
    p_det = r_det.json()
    check("Person details show 2 registered photos for centroid fusion", len(p_det.get("photos", [])) == 2)
    check("Person details contain photo_url without disk path",
          all("photo_url" in ph and "photo_path" not in ph for ph in p_det.get("photos", [])))

    # 6. Test POST /api/search (Validation, No Cutoff default, and 'weak' tier)
    print("\n[6] Testing Missing-Persons Search (POST /api/search)...")
    now_photo_path = os.path.join(HERE, "Arc face", "my_photos", "anas", "now.png")

    # Validate k bounds (1..50)
    with open(now_photo_path, "rb") as f:
        r_k0 = requests.post(f"{BASE_URL}/api/search", files={"file": ("now.png", f, "image/png")}, data={"k": 0})
        check("k=0 rejected with 400", r_k0.status_code == 400)
    with open(now_photo_path, "rb") as f:
        r_k51 = requests.post(f"{BASE_URL}/api/search", files={"file": ("now.png", f, "image/png")}, data={"k": 51})
        check("k=51 rejected with 400", r_k51.status_code == 400)

    with open(now_photo_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/search",
            files={"file": ("now.png", f, "image/png")},
            data={"k": 10}
        )
    check("Search response 200", r.status_code == 200)
    search_data = r.json()
    faces_detected = search_data.get("faces_detected", 0)
    check("Face detected in search probe", faces_detected >= 1)

    matches = search_data["results"][0].get("matches", []) if search_data["results"] else []
    target_hit = bool(matches) and matches[0]["person_id"] == test_pid
    top_sim = matches[0]["similarity"] if matches else 0.0
    check("Target matched at Rank-1 in search results", target_hit, f"(Top similarity: {top_sim})")
    check("Search results contain photo_url and no photo_path",
          bool(matches) and all(isinstance(m.get("photo_url"), str) and m["photo_url"].startswith("/api/photos/") and "photo_path" not in m for m in matches))
    check("Search results include match_source and centroid/photo similarity breakdown",
          bool(matches)
          and all(
              m.get("match_source") in ("centroid", "photo")
              and isinstance(m.get("centroid_similarity"), (int, float))
              and isinstance(m.get("best_photo_similarity"), (int, float))
              for m in matches
          ),
          f"(sample={[(m.get('match_source'), m.get('centroid_similarity'), m.get('best_photo_similarity')) for m in matches[:2]]})")
    if root_now.get("calibration") == "calibrated":
        check(
            "Calibrated search results include LLR, match_prob, p_nonmate, and advisory fields",
            bool(matches)
            and all(
                isinstance(m.get("calibrated_llr"), (int, float))
                and isinstance(m.get("match_prob"), (int, float))
                and 0.0 <= m["match_prob"] <= 1.0
                and isinstance(m.get("p_nonmate"), (int, float))
                and 0.0 <= m["p_nonmate"] <= 1.0
                and isinstance(m.get("advisory_flags"), list)
                and isinstance(m.get("raw_best_photo_similarity"), (int, float))
                for m in matches
            ),
            f"(sample={[(m.get('calibrated_llr'), m.get('p_nonmate'), m.get('tier')) for m in matches[:2]]})",
        )

    # Verify 'weak' tier returned for non-matching distractor
    has_weak_tier = any(m["tier"] == "weak" for m in matches)
    weak_sim = [m["similarity"] for m in matches if m["tier"] == "weak"]
    all_sims = [(m["name"], m["similarity"], m["tier"]) for m in matches]
    check("Search returns tier 'weak' for distractor match", has_weak_tier, f"(weak similarity: {weak_sim}, all: {all_sims})")

    # Verify verdict and confident_match_count
    res0 = search_data["results"][0] if search_data["results"] else {}
    check("Search returns confident verdict (MATCH_FOUND or POSSIBLE_MATCH)", res0.get("verdict") in ("MATCH_FOUND", "POSSIBLE_MATCH"))
    check("Search returns confident_match_count >= 1", res0.get("confident_match_count", 0) >= 1)

    # Test searching with high min_sim yielding NO_CONFIDENT_MATCH
    with open(now_photo_path, "rb") as f:
        r_high_sim = requests.post(
            f"{BASE_URL}/api/search",
            files={"file": ("now.png", f, "image/png")},
            data={"k": 5, "min_sim": 0.99}
        )
    high_sim_data = r_high_sim.json()
    high_res0 = high_sim_data["results"][0] if high_sim_data.get("results") else {}
    check("Verdict is NO_CONFIDENT_MATCH when no match meets min_sim",
          high_res0.get("verdict") == "NO_CONFIDENT_MATCH" and high_res0.get("confident_match_count") == 0)

    # 7. Test POST /api/search (Group Photo - portsaid.png with 4 faces)
    print("\n[7] Testing Group Photo Search with embed_all (POST /api/search)...")
    group_photo_path = os.path.join(HERE, "Arc face", "my_photos", "anas", "portsaid.png")
    with open(group_photo_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/search",
            files={"file": ("portsaid.png", f, "image/png")},
            data={"k": 3}
        )
    group_res = r.json()
    group_faces_count = group_res.get("faces_detected", 0)
    check("All group faces detected", group_faces_count == 4, f"(found {group_faces_count} faces)")
    has_no_confident = any(f_res.get("verdict") == "NO_CONFIDENT_MATCH" for f_res in group_res.get("results", []))
    check("Group photo contains face with NO_CONFIDENT_MATCH verdict for weak candidates", has_no_confident)

    # 8. Test POST /api/verify-id & Server Non-Blocking Check
    print("\n[8] Testing Egyptian ID Verification (POST /api/verify-id) & Concurrency...")
    id_card_path = os.path.join(HERE, "OCR_Egyptian_ID-main", "ID.jpg")

    # Verify GET / responds immediately while verify-id runs in background thread
    verify_response = {}

    def _call_verify():
        with open(id_card_path, "rb") as f_id, open(now_photo_path, "rb") as f_selfie:
            r_v = requests.post(
                f"{BASE_URL}/api/verify-id",
                files={
                    "id_card": ("ID.jpg", f_id, "image/jpeg"),
                    "selfie": ("now.png", f_selfie, "image/png")
                }
            )
            verify_response["status"] = r_v.status_code
            verify_response["data"] = r_v.json()

    v_thread = threading.Thread(target=_call_verify)
    v_thread.start()

    # Sleep slightly so verify_id starts heavy compute
    time.sleep(0.3)
    t_start = time.time()
    r_quick = requests.get(f"{BASE_URL}/", timeout=2)
    t_elapsed = time.time() - t_start
    check("GET / answers immediately while verify-id runs",
          r_quick.status_code == 200 and t_elapsed < 0.8,
          f"({t_elapsed:.3f}s)")

    v_thread.join()
    check("Verify ID response 200", verify_response.get("status") == 200)
    id_res = verify_response.get("data", {})
    check("Decision ACCEPT returned", id_res.get("decision") == "ACCEPT", f"(decision={id_res.get('decision')})")
    check(
        "Egyptian NID extracted and valid",
        id_res.get("card_info", {}).get("nid") == "30512101404256"
        and id_res.get("card_info", {}).get("nid_valid") is True,
        f"(card_info={id_res.get('card_info')})",
    )
    check("Liveness check verdict is real", id_res.get("liveness", {}).get("verdict") == "real", f"(liveness={id_res.get('liveness')})")

    # 8b. Test POST /api/verify-id/check-framing (live selfie framing guide)
    with open(now_photo_path, "rb") as f_frm:
        r_frm_one = requests.post(
            f"{BASE_URL}/api/verify-id/check-framing",
            files={"file": ("now.png", f_frm, "image/png")}
        )
    frm_one = r_frm_one.json() if r_frm_one.status_code == 200 else {}
    check(
        "check-framing single face returns framing status & face_pct",
        r_frm_one.status_code == 200
        and frm_one.get("faces_detected") == 1
        and isinstance(frm_one.get("ok"), bool)
        and frm_one.get("status") in ("ready", "too_close", "too_far", "off_center")
        and (frm_one.get("status") == "ready") == (frm_one.get("ok") is True)
        and isinstance(frm_one.get("face_pct"), (int, float))
        and 0.0 < frm_one.get("face_pct", 0.0) < 100.0
        and isinstance(frm_one.get("message"), str) and bool(frm_one.get("message")),
        f"(resp={frm_one})"
    )

    _, blank_buf = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    r_frm_zero = requests.post(
        f"{BASE_URL}/api/verify-id/check-framing",
        files={"file": ("blank.jpg", blank_buf.tobytes(), "image/jpeg")}
    )
    frm_zero = r_frm_zero.json() if r_frm_zero.status_code == 200 else {}
    check(
        "check-framing blank image returns no_face",
        r_frm_zero.status_code == 200
        and frm_zero.get("faces_detected") == 0
        and frm_zero.get("ok") is False
        and frm_zero.get("status") == "no_face",
        f"(resp={frm_zero})"
    )

    with open(group_photo_path, "rb") as f_grp:
        r_frm_multi = requests.post(
            f"{BASE_URL}/api/verify-id/check-framing",
            files={"file": ("portsaid.png", f_grp, "image/png")}
        )
    frm_multi = r_frm_multi.json() if r_frm_multi.status_code == 200 else {}
    check(
        "check-framing group photo returns multiple_faces",
        r_frm_multi.status_code == 200
        and frm_multi.get("faces_detected") == 4
        and frm_multi.get("ok") is False
        and frm_multi.get("status") == "multiple_faces",
        f"(resp={frm_multi})"
    )

    import liveness as liveness_mod
    ok_43, _, _ = liveness_mod.framing([160, 100, 400, 309], (480, 640, 3))  # 209/480 = 0.435
    ok_60, hint_60, _ = liveness_mod.framing([160, 50, 448, 338], (480, 640, 3))  # 288/480 = 0.60
    ok_15, hint_15, _ = liveness_mod.framing([240, 180, 312, 252], (480, 640, 3))  # 72/480 = 0.15
    check(
        "liveness framing thresholds accept 20%-55% (e.g. 43.5%) and reject too_close/too_far",
        liveness_mod.FACE_MIN == 0.20
        and liveness_mod.FACE_MAX == 0.55
        and ok_43 is True
        and ok_60 is False and "close" in hint_60.lower()
        and ok_15 is False and "far" in hint_15.lower(),
        f"(FACE_MIN={liveness_mod.FACE_MIN}, FACE_MAX={liveness_mod.FACE_MAX}, ok_43={ok_43}, ok_60={ok_60}, ok_15={ok_15})"
    )

    # 9. Test DELETE /api/persons/{id} & Verify disk file deletion
    print("\n[9] Testing Person Deletion & Disk File Cleanup (DELETE /api/persons/{id})...")
    # Identify photo file belonging to test_pid in uploads folder
    upload_dir = os.path.join(HERE, "uploads")
    matching_files = [f for f in os.listdir(upload_dir) if f.startswith(f"p_{test_pid}_")]
    check("Person photo files exist on disk before delete", len(matching_files) >= 2, f"(found {len(matching_files)} files)")

    r_del = requests.delete(f"{BASE_URL}/api/persons/{test_pid}")
    check("Delete status 200", r_del.status_code == 200)

    # Check file removed from disk
    remaining_files = [f for f in os.listdir(upload_dir) if f.startswith(f"p_{test_pid}_")]
    check("Person photo file removed from uploads after delete", len(remaining_files) == 0)

    r_check = requests.get(f"{BASE_URL}/api/persons/{test_pid}")
    check("Deleted person no longer found (404)", r_check.status_code == 404)

    # 10. Test Custom Fine-Tuned ONNX Engine Support (aimids_ft)
    print("\n[10] Testing Fine-Tuned AdaFace Engine (aimids_ft) & Model Isolation...")
    tmp_onnx_path = os.path.join(HERE, "test_dummy_aimids_ft.onnx")
    try:
        # Check 1: chip_bgr_to_rgb_chw normalization
        test_bgr = np.full((112, 112, 3), [10, 100, 230], dtype=np.uint8)
        chw = FaceEngineAdaFaceFT.chip_bgr_to_rgb_chw(test_bgr)
        cond1 = (
            chw.shape == (3, 112, 112)
            and np.isclose(chw[0, 0, 0], (230 / 127.5) - 1.0, atol=1e-4)
            and np.isclose(chw[1, 0, 0], (100 / 127.5) - 1.0, atol=1e-4)
            and np.isclose(chw[2, 0, 0], (10 / 127.5) - 1.0, atol=1e-4)
        )
        check("AdaFaceFT chip BGR-to-RGB CHW normalization", cond1)

        # Check 2: Export lightweight toy backbone to temporary ONNX
        class ToyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(3, 512)
                )

            def forward(self, x):
                return nn.functional.normalize(self.net(x), p=2, dim=1)

        toy_model = ToyBackbone().eval()
        dummy_inp = torch.randn(1, 3, 112, 112)
        sig = inspect.signature(torch.onnx.export)
        export_kwargs = {"dynamo": False} if "dynamo" in sig.parameters else {}
        torch.onnx.export(
            toy_model,
            dummy_inp,
            tmp_onnx_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=14,
            **export_kwargs
        )
        cond2 = os.path.isfile(tmp_onnx_path)
        check("Export toy AdaFaceFT ONNX backbone", cond2)

        # Check 3: Initialize FaceEngineAdaFaceFT with shared detector
        ft_engine = FaceEngineAdaFaceFT(onnx_path=tmp_onnx_path, use_gpu=False, use_tta=True, det_app=api.engine.app)
        cond3 = ft_engine.model_name == "aimids_ft" and ft_engine.use_tta is True
        check("Initialize FaceEngineAdaFaceFT with shared detector", cond3)

        # Check 4: Embed child photo to 512-d unit vector
        ft_res = ft_engine.embed(child_photo_path)
        cond4 = (
            ft_res is not None
            and len(ft_res.get("embedding", [])) == 512
            and abs(np.linalg.norm(ft_res["embedding"]) - 1.0) < 1e-5
            and "bbox" in ft_res
            and "kps" in ft_res
            and "det_score" in ft_res
        )
        check("FaceEngineAdaFaceFT embeds child photo to unit 512-d vector", cond4)

        # Check 5: Verify FaceStore model isolation between antelopev2, active engine, and aimids_ft
        ft_fid = api.store.add_photo(hoz_pid, ft_res["embedding"], model=ft_engine.model_name)
        models_stat = api.store.stats().get("models", {})
        ft_hits = api.store.search(ft_res["embedding"], k=5, min_sim=0.0, model="aimids_ft")
        antelope_hits = api.store.search(ft_res["embedding"], k=5, min_sim=-1.0, model="antelopev2")
        active_probe = api.engine.embed(child_photo_path)
        active_hits = (
            api.store.search(active_probe["embedding"], k=5, min_sim=-1.0, model=api.engine.model_name)
            if active_probe is not None
            else []
        )

        cond5 = (
            models_stat.get("aimids_ft") == 1
            and models_stat.get(api.engine.model_name, 0) >= 1
            and len(ft_hits) >= 1
            and ft_hits[0]["person_id"] == hoz_pid
            and ft_hits[0]["face_id"] == ft_fid
            and all(api.store._face_to_model.get(h["face_id"]) == "antelopev2" for h in antelope_hits)
            and ft_fid not in [h["face_id"] for h in antelope_hits]
            and active_probe is not None
            and len(active_hits) >= 1
            and all(api.store._face_to_model.get(h["face_id"]) == api.engine.model_name for h in active_hits)
            and ft_fid not in [h["face_id"] for h in active_hits]
        )
        check(
            "FaceStore model isolation between antelopev2, active engine, and aimids_ft",
            cond5,
            f"(models_stat={models_stat}, active_hits={len(active_hits)}, ft_hits={len(ft_hits)})",
        )

    finally:
        if os.path.isfile(tmp_onnx_path):
            try:
                os.remove(tmp_onnx_path)
            except OSError:
                pass
        # Delete distractor as well
        requests.delete(f"{BASE_URL}/api/persons/{hoz_pid}")

    # 11. Test Cross-Age 1024-d Indexing, Model Isolation, Quality+Age Centroids, Calibrated Search & Cosine Fallback
    print("\n[11] Testing Cross-Age 1024-d Store, Centroid Weighting, Calibrated Search & Cosine Fallback...")
    crossage_db_path = os.path.join(HERE, "test_crossage_store.db")
    if os.path.exists(crossage_db_path):
        try:
            os.remove(crossage_db_path)
        except OSError:
            pass

    cs = None
    try:
        cs = FaceStore(db_path=crossage_db_path)

        # 11a: 1024-d indexing + strict 512-d vs 1024-d model isolation
        pid_iso = cs.add_person("Isolation Test")
        v1024 = np.zeros(1024, dtype=np.float32)
        v1024[0] = 1.0
        v512 = np.zeros(512, dtype=np.float32)
        v512[0] = 1.0

        fid_1024 = cs.add_photo(pid_iso, v1024, model="aimids_crossage", quality=0.9)
        fid_512 = cs.add_photo(pid_iso, v512, model="antelopev2", quality=0.9)

        dim_err_add_512 = False
        try:
            cs.add_photo(pid_iso, v512, model="aimids_crossage")
        except ValueError:
            dim_err_add_512 = True

        dim_err_add_1024 = False
        try:
            cs.add_photo(pid_iso, v1024, model="antelopev2")
        except ValueError:
            dim_err_add_1024 = True

        dim_err_search_512 = False
        try:
            cs.search(v512, k=5, model="aimids_crossage")
        except ValueError:
            dim_err_search_512 = True

        dim_err_search_1024 = False
        try:
            cs.search(v1024, k=5, model="antelopev2")
        except ValueError:
            dim_err_search_1024 = True

        hits_1024 = cs.search(v1024, k=5, model="aimids_crossage")
        hits_512 = cs.search(v512, k=5, model="antelopev2")
        check(
            "FaceStore 1024-d indexing & strict 512-d vs 1024-d isolation",
            cs._model_dims.get("aimids_crossage") == 1024
            and cs._model_dims.get("antelopev2") == 512
            and len(hits_1024) == 1 and hits_1024[0]["face_id"] == fid_1024
            and len(hits_512) == 1 and hits_512[0]["face_id"] == fid_512
            and dim_err_add_512 and dim_err_add_1024
            and dim_err_search_512 and dim_err_search_1024,
            f"(dims={cs._model_dims})"
        )

        # 11b: Quality + inverse age-density centroid weighting
        pid_qual = cs.add_person("Quality Weight Test")
        e0 = np.zeros(1024, dtype=np.float32); e0[0] = 1.0
        e1 = np.zeros(1024, dtype=np.float32); e1[1] = 1.0
        e2 = np.zeros(1024, dtype=np.float32); e2[2] = 1.0
        cs.add_photo(pid_qual, e0, model="aimids_crossage", age=None, quality=0.05)  # clamped to 0.25
        cs.add_photo(pid_qual, e1, model="aimids_crossage", age=None, quality=1.50)  # clamped to 1.00
        q_cent = cs._centroids[(pid_qual, "aimids_crossage")]
        q_ratio = float(q_cent[1] / max(float(q_cent[0]), 1e-8))
        check(
            "Quality centroid weighting (clamped [0.25, 1.0])",
            abs(q_ratio - 4.0) < 1e-4 and abs(float(np.linalg.norm(q_cent)) - 1.0) < 1e-5,
            f"(ratio={q_ratio:.4f})"
        )

        pid_age = cs.add_person("Age Density Weight Test")
        cs.add_photo(pid_age, e0, model="aimids_crossage", age=5, quality=1.0)   # nearby=1 -> w_age=0.5
        cs.add_photo(pid_age, e1, model="aimids_crossage", age=6, quality=1.0)   # nearby=1 -> w_age=0.5
        cs.add_photo(pid_age, e2, model="aimids_crossage", age=25, quality=1.0)  # nearby=0 -> w_age=1.0
        a_cent = cs._centroids[(pid_age, "aimids_crossage")]
        a_ratio = float(a_cent[2] / max(float(a_cent[0]), 1e-8))
        check(
            "Inverse age-density centroid weighting",
            abs(a_ratio - 2.0) < 1e-4 and abs(float(a_cent[0]) - float(a_cent[1])) < 1e-5,
            f"(isolated_vs_clustered_ratio={a_ratio:.4f})"
        )

        # 11c: Calibrated search at small gallery size (N=5, no < 80 cliff)
        e3 = np.zeros(1024, dtype=np.float32); e3[3] = 1.0
        e4 = np.zeros(1024, dtype=np.float32); e4[4] = 1.0
        pid_c4 = cs.add_person("Calibrated Small N=4")
        cs.add_photo(pid_c4, e3, model="aimids_crossage", quality=0.9)
        pid_c5 = cs.add_person("Calibrated Small N=5")
        cs.add_photo(pid_c5, e4, model="aimids_crossage", quality=0.9)

        cal_ready_ca = cs.calibrated_ready("aimids_crossage")
        cal_hits_5 = cs.search(e3, k=5, model="aimids_crossage", probe_quality=0.9)
        hit_top5 = cal_hits_5[0] if cal_hits_5 else {}
        check(
            "Calibrated search works at small gallery N=5 (no < 80 cliff)",
            cal_ready_ca is True
            and len(cal_hits_5) == 5
            and hit_top5.get("person_id") == pid_c4
            and hit_top5.get("match_source") in ("centroid", "photo")
            and isinstance(hit_top5.get("calibrated_llr"), (int, float))
            and isinstance(hit_top5.get("match_prob"), (int, float))
            and 0.0 <= hit_top5["match_prob"] <= 1.0
            and isinstance(hit_top5.get("p_nonmate"), (int, float))
            and 0.0 <= hit_top5["p_nonmate"] <= 1.0
            and isinstance(hit_top5.get("advisory_flags"), list)
            and isinstance(hit_top5.get("raw_best_photo_similarity"), (int, float)),
            f"(n={len(cal_hits_5)}, top={hit_top5})"
        )

        # 11d: Uncalibrated model name uses clean cosine fallback (max(best_photo_sim, centroid_sim))
        uncal_model = "uncalibrated_test_model"
        u0 = np.zeros(512, dtype=np.float32); u0[0] = 1.0
        u1 = np.zeros(512, dtype=np.float32); u1[1] = 1.0
        u2 = np.zeros(512, dtype=np.float32); u2[2] = 1.0
        u3 = np.zeros(512, dtype=np.float32); u3[3] = 1.0
        u4 = np.zeros(512, dtype=np.float32); u4[4] = 1.0
        u5 = np.zeros(512, dtype=np.float32); u5[5] = 1.0
        u6 = np.zeros(512, dtype=np.float32); u6[6] = 1.0

        pid_u_cent = cs.add_person("Uncal Centroid Match")
        cs.add_photo(pid_u_cent, u0, model=uncal_model, quality=1.0)
        cs.add_photo(pid_u_cent, u1, model=uncal_model, quality=1.0)
        q_cent = (u0 + u1) / float(np.sqrt(2.0))

        pid_u_poss = cs.add_person("Uncal Possible Match")
        v_poss = 0.30 * q_cent + float(np.sqrt(1.0 - 0.30 ** 2)) * u2
        cs.add_photo(pid_u_poss, v_poss, model=uncal_model, quality=1.0)

        pid_u_weak = cs.add_person("Uncal Weak Match")
        cs.add_photo(pid_u_weak, u3, model=uncal_model, quality=1.0)

        u_hits = cs.search(q_cent, k=5, model=uncal_model)
        by_pid_u = {h["person_id"]: h for h in u_hits}

        pid_u_photo = cs.add_person("Uncal Photo Match")
        cs.add_photo(pid_u_photo, u4, model=uncal_model, quality=1.0)
        cs.add_photo(pid_u_photo, u5, model=uncal_model, quality=1.0)
        q_photo = 0.95 * u4 + float(np.sqrt(1.0 - 0.95 ** 2)) * u6
        u_photo_hits = cs.search(q_photo, k=5, model=uncal_model)
        hit_u_photo = next((h for h in u_photo_hits if h["person_id"] == pid_u_photo), None)

        check(
            "Uncalibrated model uses cosine fallback (centroid/photo max & STRONG/possible/weak tiers)",
            (not cs.calibrated_ready(uncal_model))
            and len(u_hits) == 3
            and by_pid_u[pid_u_cent].get("match_source") == "centroid"
            and abs(by_pid_u[pid_u_cent]["similarity"] - 1.0) < 1e-4
            and abs(by_pid_u[pid_u_cent]["centroid_similarity"] - 1.0) < 1e-4
            and abs(by_pid_u[pid_u_cent]["best_photo_similarity"] - float(1.0 / np.sqrt(2.0))) < 1e-3
            and by_pid_u[pid_u_cent]["tier"] == "STRONG"
            and by_pid_u[pid_u_poss]["tier"] == "possible"
            and by_pid_u[pid_u_weak]["tier"] == "weak"
            and all("calibrated_llr" not in h and "p_nonmate" not in h for h in u_hits)
            and hit_u_photo is not None
            and hit_u_photo.get("match_source") == "photo"
            and abs(hit_u_photo["similarity"] - 0.95) < 1e-3,
            f"(cent={by_pid_u.get(pid_u_cent)}, poss={by_pid_u.get(pid_u_poss)}, photo={hit_u_photo})"
        )
    finally:
        if cs is not None:
            try:
                cs.db.close()
            except Exception:
                pass
        if os.path.exists(crossage_db_path):
            try:
                os.remove(crossage_db_path)
            except OSError:
                pass

    # Clean up test DB
    th.stop()
    print("\n" + "=" * 70)
    print(f"  INTEGRATION TEST SUMMARY: {passed}/{total} CHECKS PASSED")
    print("=" * 70)

    test_db = os.path.join(HERE, "test_api_aimids.db")
    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except Exception:
            pass

    return passed == total and total > 0


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
