"""
id_verify.py - Egyptian ID card + selfie check, using the SAME ArcFace engine as the rest of AI-MIDS
(no DeepFace / TensorFlow needed).

    python id_verify.py ID.jpg                # takes the selfie from the webcam (SPACE = capture, ESC = cancel)
    python id_verify.py ID.jpg selfie.jpg     # uses an existing selfie file

Put this file next to utils.py (+ the .pt model files) and copy face_engine.py there too.

Decisions
    RETRY_CARD    card or its photo could not be read -> take a new picture of the card
    RETRY_SELFIE  no face, or more than one face, in the selfie -> take a new selfie
    REJECT_SPOOF  the selfie looks like a photo or a screen, not a live face (see liveness.py)
    ACCEPT        similarity >= ACCEPT and the national ID decoded cleanly
    REVIEW        in between, or the ID number looks misread -> a person should check
    REJECT        similarity < REVIEW

!! ACCEPT / REVIEW are PROVISIONAL starting points. They have NOT been calibrated on real
!! ID-card-vs-selfie pairs. Run this on a few real people and look at the printed similarity.
"""
import os
import os
import sys
import cv2
import numpy as np
from face_engine import FaceEngine
import time
import utils

ACCEPT, REVIEW = 0.40, 0.28
from liveness import FACE_MIN, FACE_MAX, framing  # allowed face size in the selfie frame (liveness.py has the reasons)

HOLD_SECONDS = 0.6          # stay in the green zone this long -> photo is taken automatically
GREEN, RED, YELLOW = (60, 200, 60), (40, 40, 230), (0, 220, 255)


def _boxes(engine, frame):
    """Fast face boxes only (no recognition) for the live preview."""
    det, _ = engine.app.det_model.detect(frame, max_num=0, metric="default")
    return [d[:4] for d in det]


def _text(img, s, y, color, scale=0.8):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (10, y - th - 8), (20 + tw, y + 8), (0, 0, 0), -1)
    cv2.putText(img, s, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def _draw_meter(img, pct):
    """Vertical bar on the right: green band = allowed face size, white line = your face now."""
    h, w = img.shape[:2]
    x1, x2, top, bot = w - 40, w - 20, 60, h - 60
    lo, hi = 0.0, 0.7                                       # meter shows 0%..70% of frame height
    y = lambda v: int(bot - (min(max(v, lo), hi) - lo) / (hi - lo) * (bot - top))
    cv2.rectangle(img, (x1, top), (x2, bot), (80, 80, 80), -1)
    cv2.rectangle(img, (x1, y(FACE_MAX)), (x2, y(FACE_MIN)), GREEN, -1)
    if pct is not None:
        cv2.line(img, (x1 - 8, y(pct)), (x2 + 8, y(pct)), (255, 255, 255), 3)
        cv2.putText(img, f"{pct * 100:.0f}%", (x1 - 55, y(pct) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def capture_selfie(engine, camera=0):
    """Live guide: box turns green when your face is the right size; hold still and the photo
    is taken automatically. SPACE = take it now (if in range), ESC = cancel."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open the webcam")
    result, since, warn = None, None, ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        boxes = _boxes(engine, frame)
        good, pct, color = False, None, RED
        if not boxes:
            msg = "No face found - look at the camera"
        elif len(boxes) > 1:
            msg = f"{len(boxes)} faces - only you in the frame"
        else:
            good, hint, pct = framing(boxes[0], frame.shape)
            msg, color = ("Good - hold still", GREEN) if good else (hint, RED)

        since = (since or time.time()) if good else None
        held = (time.time() - since) if since else 0.0

        shown = cv2.flip(frame, 1)                           # mirror view, like a phone selfie camera
        for b in boxes:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(shown, (W - x2, y1), (W - x1, y2), color, 3)
        if good:                                              # progress bar for the auto-capture
            cv2.rectangle(shown, (10, H - 25), (10 + int((W - 70) * min(held / HOLD_SECONDS, 1)), H - 10), GREEN, -1)
        _draw_meter(shown, pct)
        _text(shown, msg, 40, color)
        _text(shown, warn or "Auto-capture when green   SPACE = now   ESC = cancel", 80, YELLOW, 0.6)
        cv2.imshow("Selfie", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if good and (held >= HOLD_SECONDS or key == 32):
            faces = engine.embed_all(frame)                   # final check with the full engine
            if len(faces) == 1 and framing(faces[0]["bbox"], frame.shape)[0]:
                result = frame
                break
            since, warn = None, "Moved during capture - hold still and try again"
    cap.release()
    cv2.destroyAllWindows()
    return result


def verify_id(id_image, selfie, engine, liveness=None):
    """id_image / selfie: file path or BGR image. liveness: optional LivenessChecker.
    Returns a dict; look at ['decision']."""
    card = utils.read_id_card(id_image)
    out = {"nid": card["nid"], "birth_date": card["birth_date"], "governorate": card["governorate"],
           "gender": card["gender"], "nid_valid": card["valid"], "problems": list(card["problems"]),
           "similarity": None, "liveness": None, "decision": None}

    if card["photo"] is None or card["photo"].size == 0:
        out["decision"] = "RETRY_CARD"
        return out
    pad = int(0.25 * max(card["photo"].shape[:2]))
    padded_photo = cv2.copyMakeBorder(card["photo"], pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    card_face = engine.embed(padded_photo)
    if card_face is None:
        card_face = engine.embed(card["photo"])
    if card_face is None:
        out["problems"].append("no face found in the photo on the card")
        out["decision"] = "RETRY_CARD"
        return out

    faces = engine.embed_all(selfie)
    if len(faces) != 1:
        out["problems"].append("no face in the selfie" if not faces else f"{len(faces)} faces in the selfie, need exactly 1")
        out["decision"] = "RETRY_SELFIE"
        return out

    sim = engine.similarity(card_face["embedding"], faces[0]["embedding"])
    out["similarity"] = round(sim, 4)

    suspicious = False
    if liveness is not None:  # is the selfie a live face, not a photo or screen?
        out["liveness"] = liveness.check(engine._read(selfie), faces[0]["bbox"])
        lv = out["liveness"]
        if lv["verdict"] != "real":
            why = lv["note"] or f"chance real {lv['real_prob']:.2f}"
            out["problems"].append(f"liveness: {lv['verdict']} ({why})")
            if lv["verdict"] == "fake":
                out["decision"] = "REJECT_SPOOF"
                return out
            suspicious = True  # uncertain: a person should look, but don't block a real user

    if sim >= ACCEPT:
        out["decision"] = "ACCEPT" if (card["valid"] and not suspicious) else "REVIEW"
    elif sim >= REVIEW:
        out["decision"] = "REVIEW"
    else:
        out["decision"] = "REJECT"
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit('Usage: python id_verify.py "ID.jpg" ["selfie.jpg"] [--no-liveness] [--antelopev2]')
    use_antelope = "--antelopev2" in sys.argv or os.environ.get("AIMIDS_ENGINE", "").lower() == "antelopev2"
    if use_antelope:
        arc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Arc face"))
        if arc_dir not in sys.path:
            sys.path.insert(0, arc_dir)
        from face_engine_antelopev2 import FaceEngineAntelopeV2
        engine = FaceEngineAntelopeV2(use_gpu=False)
    else:
        engine = FaceEngine(use_gpu=False)
    live = None
    if "--no-liveness" not in sys.argv:
        from liveness import LivenessChecker  # needs: pip install uniface
        live = LivenessChecker()
    selfie = args[1] if len(args) > 1 else capture_selfie(engine)
    if selfie is None:
        sys.exit("No selfie captured.")
    r = verify_id(args[0], selfie, engine, live)
    print("\n=== RESULT ===")
    for k in ("decision", "similarity", "liveness", "nid", "birth_date", "governorate", "gender", "nid_valid"):
        print(f"{k:12}: {r[k]}")
    if r["problems"]:
        print("problems    :", "; ".join(r["problems"]))
