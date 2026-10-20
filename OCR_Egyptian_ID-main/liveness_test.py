"""
liveness_test.py - live webcam view of the liveness scores, so we can pick settings from YOUR data.

    python liveness_test.py

Numbers shown are the chance the face is REAL (1.00 = surely real, 0.00 = surely fake):
    V2 / V1SE        the two models as the library runs them
    V2p / V1SEp      same, but with the picture padded first so the model gets the full context crop
    ens              average of V2p and V1SEp
"face" = how tall your face is compared with the frame. The models want it under about 35%.

Keys:   r = log this reading as REAL      f = log this reading as FAKE      ESC = finish
Do this:
   1. Real face, arm's length away: press r about 10 times (move your head a little between presses).
   2. Real face, close to the screen (like when you took the earlier selfies): press r about 10 times.
   3. Dim light, or glasses: press r about 5 times.
   4. Hold a phone showing your photo, then a printed photo, in front of the camera: press f about 10 times each.
When you press ESC it prints a summary and saves liveness_log.csv - paste the summary to Claude.
"""
import csv
import time
import cv2
import numpy as np


def p_real(model, img, bbox):
    r = model.predict(img, [float(v) for v in bbox])
    return r.confidence if r.is_real else 1.0 - r.confidence


def padded(img, bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(1.5 * max(w, h))
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return big, [bbox[0] + pad, bbox[1] + pad, bbox[2] + pad, bbox[3] + pad]


def measure(m_v2, m_v1, frame, bbox):
    big, bb = padded(frame, bbox)
    v2p, v1p = p_real(m_v2, big, bb), p_real(m_v1, big, bb)
    return {
        "face_pct": 100.0 * (bbox[3] - bbox[1]) / frame.shape[0],
        "V2": p_real(m_v2, frame, bbox), "V1SE": p_real(m_v1, frame, bbox),
        "V2p": v2p, "V1SEp": v1p, "ens": (v2p + v1p) / 2,
    }


KEYS = ["V2", "V1SE", "V2p", "V1SEp", "ens"]


def main():
    from face_engine import FaceEngine
    from uniface.spoofing import MiniFASNet
    from uniface.constants import MiniFASNetWeights

    engine = FaceEngine(use_gpu=False)
    m_v2 = MiniFASNet(model_name=MiniFASNetWeights.V2)
    m_v1 = MiniFASNet(model_name=MiniFASNetWeights.V1SE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open the webcam")
    last, last_t, log = None, 0.0, []
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if time.time() - last_t > 0.5:
            last_t = time.time()
            faces = engine.embed_all(frame)
            last = measure(m_v2, m_v1, frame, faces[0]["bbox"]) if faces else None
        shown = frame.copy()
        lines = ["no face"] if last is None else [f"face {last['face_pct']:.0f}% of frame"] + \
                [f"{k:6} {last[k]:.2f}" for k in KEYS]
        for i, t in enumerate(lines):
            cv2.putText(shown, t, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(shown, f"logged: {len(log)}   r=REAL  f=FAKE  ESC=done", (15, shown.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("liveness test", shown)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k in (ord("r"), ord("f")) and last is not None:
            log.append(("REAL" if k == ord("r") else "FAKE", dict(last)))
            print(f"logged {log[-1][0]}  face {last['face_pct']:.0f}%  " + "  ".join(f"{x} {last[x]:.2f}" for x in KEYS))
    cap.release()
    cv2.destroyAllWindows()

    if not log:
        return
    with open("liveness_log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "face_pct"] + KEYS)
        for lab, m in log:
            w.writerow([lab, round(m["face_pct"], 1)] + [round(m[x], 3) for x in KEYS])
    print("\n=== SUMMARY (chance the face is real; REAL should be high, FAKE low) ===")
    for lab in ("REAL", "FAKE"):
        rows = [m for l, m in log if l == lab]
        if not rows:
            continue
        print(f"{lab}: {len(rows)} readings, face size {np.mean([r['face_pct'] for r in rows]):.0f}% of frame on average")
        for x in KEYS:
            v = [r[x] for r in rows]
            print(f"   {x:6} mean {np.mean(v):.2f}   min {np.min(v):.2f}   max {np.max(v):.2f}")
    print("Saved liveness_log.csv")


if __name__ == "__main__":
    main()
