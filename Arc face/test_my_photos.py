"""
test_my_photos.py - try the pipeline on your own photos.

Folder layout (one sub-folder per person, any file names):
    my_photos/
        ali/        childhood.jpg  teen.jpg  now.jpg
        sara/       a.jpg  b.jpg
        stranger1/  x.jpg            <- single photos are fine, they act as "distractors"

Run (same folder as face_engine.py, main venv):
    python test_my_photos.py                 # uses ./my_photos
    python test_my_photos.py "D:\\some\\other\\folder"
"""
import os
import sys
import numpy as np
from face_engine import FaceEngine

ROOT = sys.argv[1] if len(sys.argv) > 1 else "my_photos"
STRONG, POSSIBLE = 0.36, 0.26
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def tier(s):
    return "STRONG" if s >= STRONG else ("possible" if s >= POSSIBLE else "-")


engine = FaceEngine()
names, paths, embs = [], [], []
for person in sorted(os.listdir(ROOT)):
    folder = os.path.join(ROOT, person)
    if not os.path.isdir(folder):
        continue
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(EXT):
            continue
        p = os.path.join(folder, f)
        r = engine.embed(p)
        if r is None:
            print("NO FACE FOUND (skipped):", os.path.join(person, f))
            continue
        if r["n_faces"] > 1:
            print(f"note: {r['n_faces']} faces in {os.path.join(person, f)}, using the largest one")
        names.append(person)
        paths.append(os.path.join(person, f))
        embs.append(r["embedding"])

if len(embs) < 3:
    sys.exit("Need at least 3 usable photos from at least 2 people.")

names = np.array(names)
M = np.stack(embs)
S = M @ M.T
counts = {n: int((names == n).sum()) for n in set(names)}

print(f"\n{len(M)} photos, {len(counts)} people\n")
print("For every photo: its 3 best matches among ALL OTHER photos")
print("(OK = same person, X = different person)\n")
hits, tried = 0, 0
for i in range(len(M)):
    sims = S[i].copy()
    sims[i] = -2
    top = np.argsort(-sims)[:4]
    print(paths[i])
    for rank, j in enumerate(top, 1):
        mark = "OK" if names[j] == names[i] else "X "
        print(f"   {rank}. {sims[j]:.3f}  {tier(sims[j]):<8} {mark}  {paths[j]}")
    if counts[names[i]] >= 2:
        tried += 1
        hits += int(names[top[0]] == names[i])

same = np.array([S[i, j] for i in range(len(M)) for j in range(i + 1, len(M)) if names[i] == names[j]])
diff = np.array([S[i, j] for i in range(len(M)) for j in range(i + 1, len(M)) if names[i] != names[j]])
print("\n=== SUMMARY ===")
if tried:
    print(f"Best match is the right person for {hits} of {tried} photos (people with 2+ photos)")
if len(same):
    print(f"Same person:       lowest {same.min():.3f}   average {same.mean():.3f}   ({len(same)} pairs)")
if len(diff):
    print(f"Different people:  highest {diff.max():.3f}   average {diff.mean():.3f}   ({len(diff)} pairs)")
print(f"Cutoffs in use: STRONG >= {STRONG}, possible >= {POSSIBLE}")