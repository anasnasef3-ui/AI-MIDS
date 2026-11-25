"""
test_realistic_search.py - End-to-end realistic test of the AI-MIDS search pipeline.

Simulates the exact real-world scenario:
1. Missing persons database is populated with childhood photos (ages 0-10).
2. Probes (searches) are performed using adult/recent photos (10-25+ year age gaps).
3. Tests both personal photos (Anas, brother Hoz, group photo Portsaid) and FG-NET benchmark.
"""
import os
import re
import sys
import time
import numpy as np
import cv2

# Ensure we import from local directory
ARC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARC)
os.chdir(ARC)
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from face_engine import FaceEngine
from store import FaceStore, STRONG, POSSIBLE
from photo_quality import check_photo, register_photo

print("=" * 70)
print("  AI-MIDS REALISTIC SEARCH TEST (CHILDHOOD -> ADULT)")
print("=" * 70)

# 1. Initialize Engine and In-Memory Store
print("\n[1] Initializing FaceEngine and Clean FaceStore...")
engine = FaceEngine(use_gpu=False)
store = FaceStore(":memory:")

# Helper: search every face in a photo
def search_photo(engine, store, photo_path, k=10, min_sim=POSSIBLE):
    faces = engine.embed_all(photo_path)
    if not faces:
        return []
    results = []
    for i, face in enumerate(faces):
        hits = store.search(face["embedding"], k=k, min_sim=min_sim)
        results.append({
            "face_index": i,
            "bbox": [round(v) for v in face["bbox"]],
            "det_score": round(face["det_score"], 3),
            "hits": hits
        })
    return results

# 2. Register Personal Profiles
print("\n[2] Registering Missing Persons into the Database...")

# Register Anas with childhood photo
p_anas = store.add_person("Anas (Missing Child)", notes="Registered with childhood photo")
fid_anas, q_anas = register_photo(engine, store, p_anas, r"my_photos\anas\childhood.jpeg", age=5)
print(f"  + Registered Anas: ID={p_anas}, FaceID={fid_anas}, Status={q_anas['status']}")

# Register Brother Hoz with photo #1
p_hoz = store.add_person("Hoz (Brother)", notes="Registered with baseline photo")
fid_hoz, q_hoz = register_photo(engine, store, p_hoz, r"my_photos\hoz\hoz (1).jpg")
print(f"  + Registered Hoz:  ID={p_hoz}, FaceID={fid_hoz}, Status={q_hoz['status']}")

# Register Distractors (strangers) from my_photos
distractors_count = 0
for stranger_dir in ["stranger1", "stranger2", "stranger3"]:
    s_path = os.path.join("my_photos", stranger_dir)
    if os.path.isdir(s_path):
        for f in os.listdir(s_path):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                full_p = os.path.join(s_path, f)
                pid = store.add_person(f"Distractor_{stranger_dir}")
                fid, q = register_photo(engine, store, pid, full_p)
                if fid:
                    distractors_count += 1
print(f"  + Registered {distractors_count} distractor photos from my_photos.")

# 3. Register FG-NET Identities (Childhood Photos)
fg_dir = r"D:\GRAD\Assets\FGNET\images"
fg_people = {}
if os.path.isdir(fg_dir):
    # Filename format: e.g. 001A02.JPG -> id=001, age=2
    pattern = re.compile(r"^(\d{3})A(\d{2})", re.IGNORECASE)
    for f in sorted(os.listdir(fg_dir)):
        m = pattern.match(f)
        if m:
            pid_str = m.group(1)
            age = int(m.group(2))
            full_p = os.path.join(fg_dir, f)
            if pid_str not in fg_people:
                fg_people[pid_str] = []
            fg_people[pid_str].append((age, full_p))

fg_registered = 0
fg_probes = []  # will hold (expected_person_id, person_name, probe_age, gallery_age, path)

for pid_str, photos in sorted(fg_people.items()):
    photos = sorted(photos, key=lambda x: x[0])  # sort by age
    youngest_age, youngest_path = photos[0]
    
    # We want identities that have a child photo (<= 12 years old)
    if youngest_age <= 12 and len(photos) >= 2:
        # Check if there is an adult/older probe (age >= 18 or gap >= 10)
        older_photos = [p for p in photos if p[0] >= 16 and (p[0] - youngest_age) >= 8]
        if older_photos:
            name = f"FGNET_{pid_str}"
            db_pid = store.add_person(name, notes=f"Childhood photo at age {youngest_age}")
            fid, q = register_photo(engine, store, db_pid, youngest_path, age=youngest_age)
            if fid:
                fg_registered += 1
                for o_age, o_path in older_photos:
                    fg_probes.append((db_pid, name, o_age, youngest_age, o_path))

print(f"  + Registered {fg_registered} FG-NET identities (youngest childhood photos).")
stats = store.stats()
print(f"\n>> Total in Database: {stats['people']} people, {stats['photos']} gallery photos.")

# =========================================================================
# 4. PERSONAL PROBE TESTS
# =========================================================================
print("\n" + "=" * 70)
print("  PART 1: PERSONAL SEARCH TESTS")
print("=" * 70)

# Test 4.1: Search Anas Adult Photo (now.png) -> 15+ year age gap
print("\n--- Test 4.1: Search with Anas Recent Photo (now.png) ---")
emb_now = engine.embed(r"my_photos\anas\now.png")
if emb_now:
    hits = store.search(emb_now["embedding"], k=5, min_sim=0.20)
    print(f"Searched with now.png (Age gap ~15+ years). Top matches:")
    for rank, h in enumerate(hits, 1):
        is_target = (h["person_id"] == p_anas)
        mark = "[TARGET HIT]" if is_target else "            "
        print(f"  Rank {rank}: {h['name']:<25} | Sim: {h['similarity']:.4f} ({h['tier']:<8}) {mark}")
else:
    print("Failed to detect face in now.png")

# Test 4.2: Search Anas Teen Photo (teen.jpeg) -> ~8-10 year age gap
print("\n--- Test 4.2: Search with Anas Teen Photo (teen.jpeg) ---")
emb_teen = engine.embed(r"my_photos\anas\teen.jpeg")
if emb_teen:
    hits = store.search(emb_teen["embedding"], k=5, min_sim=0.20)
    print(f"Searched with teen.jpeg (Age gap ~8-10 years). Top matches:")
    for rank, h in enumerate(hits, 1):
        is_target = (h["person_id"] == p_anas)
        mark = "[TARGET HIT]" if is_target else "            "
        print(f"  Rank {rank}: {h['name']:<25} | Sim: {h['similarity']:.4f} ({h['tier']:<8}) {mark}")

# Test 4.3: Brother Separation Test (search hoz (2).jpg)
print("\n--- Test 4.3: Brother Separation Test (search hoz (2).jpg) ---")
emb_hoz2 = engine.embed(r"my_photos\hoz\hoz (2).jpg")
if emb_hoz2:
    hits = store.search(emb_hoz2["embedding"], k=5, min_sim=0.20)
    print(f"Searched with hoz (2).jpg. Top matches:")
    for rank, h in enumerate(hits, 1):
        is_hoz = (h["person_id"] == p_hoz)
        is_anas = (h["person_id"] == p_anas)
        tag = "[BROTHER HOZ]" if is_hoz else ("[ANAS GENETIC]" if is_anas else "Distractor")
        print(f"  Rank {rank}: {h['name']:<25} | Sim: {h['similarity']:.4f} ({h['tier']:<8}) -> {tag}")

# Test 4.4: Group Photo Test (portsaid.png with 4 faces)
print("\n--- Test 4.4: Group Photo Search (portsaid.png - 4 faces) ---")
group_results = search_photo(engine, store, r"my_photos\anas\portsaid.png", k=3, min_sim=0.25)
print(f"Found {len(group_results)} faces in portsaid.png:")
for res in group_results:
    idx = res["face_index"]
    hits = res["hits"]
    top_str = "No match >= 0.25"
    if hits:
        top_str = f"Rank 1: {hits[0]['name']} (sim={hits[0]['similarity']}, {hits[0]['tier']})"
    print(f"  Face #{idx} (det_score={res['det_score']}): {top_str}")

# =========================================================================
# 5. FG-NET BENCHMARK (REALISTIC AGING TEST)
# =========================================================================
print("\n" + "=" * 70)
print(f"  PART 2: BENCHMARK WITH FG-NET ({len(fg_probes)} probes across {fg_registered} identities)")
print("=" * 70)

# Limit to 50 diverse probes for fast execution while maintaining statistical significance
np.random.seed(42)
if len(fg_probes) > 60:
    selected_indices = np.random.choice(len(fg_probes), size=60, replace=False)
    test_probes = [fg_probes[i] for i in selected_indices]
else:
    test_probes = fg_probes

rank1_hits = 0
top5_hits = 0
top10_hits = 0
top10_hits_cutoff26 = 0
total_probes = len(test_probes)
age_gaps = []
sims_correct = []

print(f"Running search for {total_probes} probes (evaluating Rank-1, Top-5, Top-10 with and without 0.26 cutoff)...")
t0 = time.time()

for db_pid, name, probe_age, gal_age, p_path in test_probes:
    gap = probe_age - gal_age
    age_gaps.append(gap)
    emb = engine.embed(p_path)
    if not emb:
        continue
    
    # Search top 10 with no cutoff to check true rank
    hits = store.search(emb["embedding"], k=10, min_sim=-1.0)
    hit_ids = [h["person_id"] for h in hits]
    
    if db_pid in hit_ids:
        rank = hit_ids.index(db_pid) + 1
        if rank == 1:
            rank1_hits += 1
        if rank <= 5:
            top5_hits += 1
        if rank <= 10:
            top10_hits += 1
            target_hit = [h for h in hits if h["person_id"] == db_pid][0]
            if target_hit["similarity"] >= 0.26:
                top10_hits_cutoff26 += 1
        
        # record similarity
        target_hit = [h for h in hits if h["person_id"] == db_pid][0]
        sims_correct.append(target_hit["similarity"])

elapsed = time.time() - t0
print(f"Completed in {elapsed:.1f}s ({elapsed/total_probes:.2f}s per search query).\n")

# Summary Table
print("=== FG-NET BENCHMARK RESULTS ===")
print(f"Probes Tested                : {total_probes}")
print(f"Average Age Gap              : {np.mean(age_gaps):.1f} years (Range: {min(age_gaps)} - {max(age_gaps)} years)")
print(f"Top-10 Hit Rate (no cutoff)  : {top10_hits / total_probes * 100:.1f}% ({top10_hits}/{total_probes})")
print(f"Top-10 Hit Rate (cutoff 0.26): {top10_hits_cutoff26 / total_probes * 100:.1f}% ({top10_hits_cutoff26}/{total_probes})")
print(f"Top-5 Hit Rate (no cutoff)   : {top5_hits / total_probes * 100:.1f}% ({top5_hits}/{total_probes})")
print(f"Rank-1 Accuracy (no cutoff)  : {rank1_hits / total_probes * 100:.1f}% ({rank1_hits}/{total_probes})")
if sims_correct:
    print(f"Mean Target Sim              : {np.mean(sims_correct):.3f} (Min: {min(sims_correct):.3f}, Max: {max(sims_correct):.3f})")

print("\n" + "=" * 70)
print("  EVALUATION COMPLETE")
print("=" * 70)
