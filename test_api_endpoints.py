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

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Point test DB to a temporary test file so we don't alter the real db
TEST_DB_PATH = os.path.join(HERE, "test_api_aimids.db")
if os.path.exists(TEST_DB_PATH):
    try:
        os.remove(TEST_DB_PATH)
    except Exception:
        pass
os.environ["AIMIDS_DB_PATH"] = TEST_DB_PATH

from api import app

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
    check("Database stats returned", "stats" in data)
    check("Active engine returned in root", data.get("engine") in ("antelopev2", "buffalo_l"), f"(engine={data.get('engine')})")

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
    check("Photo URL returned", "photo_url" in photo_res)

    # Upload 2nd photo for Anas to exercise multi-photo centroid fusion
    with open(child_photo_path, "rb") as f:
        r_photo2 = requests.post(
            f"{BASE_URL}/api/persons/{test_pid}/photos",
            files={"file": ("childhood_2.jpeg", f, "image/jpeg")},
            data={"age": 6}
        )
    check("Upload 2nd photo for centroid fusion status 200", r_photo2.status_code == 200)
    photo2_res = r_photo2.json()
    test_face_id2 = photo2_res.get("face_id")
    check("2nd Face ID assigned", isinstance(test_face_id2, int))

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
    target_hit = any(m["person_id"] == test_pid for m in matches)
    top_sim = matches[0]["similarity"] if matches else 0.0
    check("Target matched in search results", target_hit, f"(Top similarity: {top_sim})")
    check("Search results contain photo_url and no photo_path",
          all("photo_url" in m and "photo_path" not in m for m in matches))

    # Verify 'weak' tier returned for low similarity match (< 0.26)
    has_weak_tier = any(m["tier"] == "weak" for m in matches)
    weak_sim = [m["similarity"] for m in matches if m["tier"] == "weak"]
    all_sims = [(m["name"], m["similarity"], m["tier"]) for m in matches]
    check("Search returns tier 'weak' for match < 0.26", has_weak_tier, f"(weak similarity: {weak_sim}, all: {all_sims})")

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
    check("Decision returned", id_res.get("decision") in ("ACCEPT", "REVIEW", "REJECT", "REJECT_SPOOF"))
    check("Egyptian NID extracted", len(id_res.get("card_info", {}).get("nid", "")) == 14)
    check("Liveness check ran (mandatory)", "verdict" in id_res.get("liveness", {}))

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

    # Delete distractor as well
    requests.delete(f"{BASE_URL}/api/persons/{hoz_pid}")

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


if __name__ == "__main__":
    run_tests()
