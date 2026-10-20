"""
photo_quality.py - is this photo good enough to register a person with?

    from face_engine import FaceEngine
    from photo_quality import check_photo, register_photo
    engine = FaceEngine(use_gpu=False)

    q = check_photo(engine, "photo.jpg")            # or check_photo(engine, img, face_index=2)
    q["status"]   # "ok"          -> fine
                  # "warn"        -> usable, but tell the user why it may match badly (q["reasons"])
                  # "reject"      -> don't store it (q["reasons"] says why)
                  # "choose_face" -> several faces: show q["faces"] (numbered, largest first) and call
                  #                  again with face_index=<the number the user picked>
    q["face"]     # the chosen face (embedding, bbox, kps, det_score) when status is ok / warn

    fid = register_photo(engine, store, person_id, "photo.jpg", age=9)   # check + store in one step

Run directly to try a photo (saves <photo>_faces.png with numbered boxes):
    python photo_quality.py photo.jpg            # check
    python photo_quality.py group.jpg 2          # check face number 2

Design choice: registration photos of missing people are often old, scanned, small or blurry, and
they may be the ONLY photo a family has. So only clearly unusable photos are rejected; weaker ones are
accepted with a warning. All limits below are PROVISIONAL (see the numbers next to them).
"""
import sys
import numpy as np
import cv2
from insightface.utils import face_align

MIN_FACE_PX = 40      # face box height below this -> reject (too few pixels to recognise anyone)
SMALL_FACE_PX = 80    # below this -> warn
BLUR_REJECT = 15      # sharpness below this -> reject (almost no detail left)
BLUR_WARN = 60        # below this -> warn
SIDE_ON = 0.15        # nose this close to one eye (0.5 = straight on) -> warn: face turned sideways (calibrated on FG-NET)


def sharpness(img, kps):
    """Detail in the face: straighten it to the standard 112x112 crop, then measure edge strength
    (variance of the Laplacian). Measured on the same-size crop so big and small photos compare fairly."""
    crop = face_align.norm_crop(img, np.asarray(kps, dtype=np.float32), image_size=112)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frontalness(kps):
    """Where the nose sits between the eyes: 0.5 = looking straight at the camera, near 0 or 1 = side-on."""
    le, re, nose = (np.asarray(p, dtype=np.float32) for p in kps[:3])
    eye = re - le
    t = float(np.dot(nose - le, eye) / max(float(np.dot(eye, eye)), 1e-6))
    return min(max(t, 0.0), 1.0)


def _summary(i, f):
    x1, y1, x2, y2 = f["bbox"]
    return {"face_index": i, "bbox": [round(v) for v in f["bbox"]], "height_px": round(y2 - y1),
            "det_score": round(f["det_score"], 3)}


def check_photo(engine, src, face_index=None):
    img = engine._read(src)
    found = engine.embed_all(img)                                      # largest first
    usable = [f for f in found if f["bbox"][3] - f["bbox"][1] >= MIN_FACE_PX]
    out = {"status": None, "reasons": [], "face": None, "metrics": {},
           "faces": [_summary(i, f) for i, f in enumerate(usable)]}

    if not found:
        out["status"] = "reject"
        out["reasons"].append("No face found. Use a photo where the face is clearly visible.")
        return out
    if not usable:
        out["status"] = "reject"
        h = found[0]["bbox"][3] - found[0]["bbox"][1]
        out["reasons"].append(f"The face is too small ({h:.0f} px tall, need at least {MIN_FACE_PX}). "
                              "Use a bigger or closer photo, or crop around the face.")
        return out
    if face_index is None:
        if len(usable) > 1:
            out["status"] = "choose_face"
            out["reasons"].append(f"{len(usable)} faces found. Choose which one is the person.")
            return out
        face_index = 0
    if not 0 <= face_index < len(usable):
        raise ValueError(f"face_index must be 0..{len(usable) - 1}")

    f = usable[face_index]
    h = f["bbox"][3] - f["bbox"][1]
    sharp = sharpness(img, f["kps"])
    front = frontalness(f["kps"])
    out["face"] = f
    out["metrics"] = {"height_px": round(h), "sharpness": round(sharp, 1),
                      "frontal": round(front, 2), "det_score": round(f["det_score"], 3)}

    rejects, warns = [], []
    if sharp < BLUR_REJECT:
        rejects.append(f"The face is too blurry (sharpness {sharp:.0f}, need at least {BLUR_REJECT}).")
    elif sharp < BLUR_WARN:
        warns.append(f"The face is somewhat blurry (sharpness {sharp:.0f}); matches may be weaker.")
    if h < SMALL_FACE_PX:
        warns.append(f"The face is small ({h:.0f} px tall); a larger photo would match better.")
    if front < SIDE_ON or front > 1 - SIDE_ON:
        warns.append("The face is turned to the side; a front-facing photo would match better.")

    out["reasons"] = rejects + warns
    out["status"] = "reject" if rejects else ("warn" if warns else "ok")
    return out


def register_photo(engine, store, person_id, src, age=None, face_index=None, photo_path=None):
    """Check the photo and, if it is ok or warn, store its face. Returns (face_id or None, check result)."""
    q = check_photo(engine, src, face_index)
    if q["status"] not in ("ok", "warn"):
        return None, q
    path = photo_path if photo_path is not None else (src if isinstance(src, str) else None)
    return store.add_photo(person_id, q["face"]["embedding"], photo_path=path, age=age), q


def draw_faces(img, faces, chosen=None):
    """Numbered boxes, the chosen face in green, the rest in orange."""
    out = img.copy()
    for s in faces:
        x1, y1, x2, y2 = s["bbox"]
        color = (60, 200, 60) if s["face_index"] == chosen else (0, 165, 255)
        t = max(2, out.shape[0] // 300)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        cv2.putText(out, str(s["face_index"]), (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, t * 0.6, color, t)
    return out


if __name__ == "__main__":
    import os
    from face_engine import FaceEngine
    if len(sys.argv) < 2:
        sys.exit("Usage: python photo_quality.py photo.jpg [face_number]")
    path = sys.argv[1]
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    engine = FaceEngine(use_gpu=False)
    q = check_photo(engine, path, idx)
    print("\n=== PHOTO CHECK ===")
    print("status :", q["status"].upper())
    for r in q["reasons"]:
        print("  -", r)
    if q["metrics"]:
        print("metrics:", q["metrics"])
    debug_faces_path = os.path.splitext(path)[0] + "_faces.png"
    if q["faces"]:
        print("faces  :", ", ".join(f"#{s['face_index']} ({s['height_px']} px)" for s in q["faces"]))
        if q["status"] == "choose_face":
            # Multiple faces and user hasn't picked one yet: create debug image to help them pick
            img = engine._read(path)
            cv2.imencode(".png", draw_faces(img, q["faces"]))[1].tofile(debug_faces_path)
            print("boxes  :", debug_faces_path, "(check this image to choose face number)")
        else:
            # Face resolved or single face: clean up any leftover debug image
            if os.path.exists(debug_faces_path):
                try:
                    os.remove(debug_faces_path)
                except OSError:
                    pass
