"""
show_faces.py - draws a numbered box around every face in a photo and opens the result.
Green box = matched someone in aimids.db (name + score shown), red box = no match.

    python show_faces.py "my_photos\\anas\\portsaid.png"

The marked copy is saved next to the original as <name>_faces.jpg (the original is not changed).
"""
import os
import sys
import cv2
import numpy as np
from face_engine import FaceEngine
from store import FaceStore

if len(sys.argv) < 2:
    sys.exit('Usage: python show_faces.py "path\\to\\photo.jpg"')
path = sys.argv[1]

engine = FaceEngine(use_gpu=False)
store = FaceStore("aimids.db") if os.path.exists("aimids.db") else None
img = engine._read(path)
faces = engine.embed_all(img)

h, w = img.shape[:2]
thick = max(2, int(min(h, w) / 200))
scale = max(0.6, min(h, w) / 600)
out = img.copy()

print(f"\n{len(faces)} face(s) found in {path}")
for n, f in enumerate(faces, 1):
    x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
    hits = store.search(f["embedding"], k=3) if store else []
    color = (0, 200, 0) if hits else (0, 0, 255)  # BGR: green / red
    label = str(n)
    if hits:
        label += " " + hits[0]["name"].encode("ascii", "replace").decode() + f" {hits[0]['similarity']:.2f}"
        print(f"  face {n}: " + ", ".join(f"{x['name']} {x['similarity']:.3f} ({x['tier']})" for x in hits))
    else:
        print(f"  face {n}: no match")
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    cv2.putText(out, label, (max(x1, 0), max(y1 - 8, int(25 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

out_path = os.path.splitext(path)[0] + "_faces.jpg"
cv2.imencode(".jpg", out)[1].tofile(out_path)
print("\nSaved:", out_path)
try:
    os.startfile(out_path)  # opens the image on Windows
except Exception:
    pass
