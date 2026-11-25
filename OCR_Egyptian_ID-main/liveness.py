"""
liveness.py - "is this a live face, or a photo / screen held up to the camera?"

Uses MiniFASNet (Silent-Face-Anti-Spoofing, Apache-2.0) through the `uniface` package.
    pip install uniface
The models (about 2 MB each) download themselves the first time.

    from liveness import LivenessChecker
    live = LivenessChecker()
    r = live.check(frame_bgr, face_bbox)     # bbox = [x1, y1, x2, y2] from FaceEngine.embed_all
    r["verdict"]    # "real", "uncertain" or "fake"
    r["real_prob"]  # 0..1 (None when the framing is wrong)
    r["note"]       # what to fix when the framing is wrong

FRAMING MATTERS. The models look at a crop 2.7x (V2) / 4x (V1SE) the size of the face. Below about 55%
of the frame height (FACE_MAX = 0.55, comfortable arm's-length webcam selfie range), MiniFASNet
still preserves enough head/neck/shoulder context and real faces score strongly real (0.79-1.00);
only above ~65% does the shrunk context crop degrade reliability and start scoring real faces as
fake. So this checker only scores faces between FACE_MIN and FACE_MAX of the frame height, and
otherwise answers "uncertain" and says to move.

PASS / HARD_FAKE / FACE_MIN / FACE_MAX are empirical defaults for webcam selfies. Re-check them
with more people, cameras, distances and lighting.

Limits: single-image check. It catches many printed photos and phone screens but NOT good video
replays, masks or deepfakes, and it can flag real people in bad lighting. "uncertain" means
"a person should look", not "fraud".
"""

FACE_MIN, FACE_MAX = 0.20, 0.55   # face height as a fraction of the frame height (arm's-length selfie upper bound)
PASS = 0.70                       # at or above: real
HARD_FAKE = 0.15                  # below: fake (hard reject); in between: uncertain


def framing(bbox, shape):
    """Is the face the right size for the liveness models? -> (ok, hint, face_pct).
    Used by both the live webcam guide (id_verify.py) and check(), so they always agree.
    Width is checked too, for portrait/phone cameras where the frame is narrow."""
    h, w = shape[:2]
    ph = (bbox[3] - bbox[1]) / h
    pw = (bbox[2] - bbox[0]) / w
    if ph > FACE_MAX or pw > FACE_MAX:
        return False, "Too close - move back a little", ph
    if ph < FACE_MIN:
        return False, "Too far - move closer", ph
    return True, "", ph


class LivenessChecker:
    def __init__(self):
        from uniface.spoofing import MiniFASNet
        from uniface.constants import MiniFASNetWeights
        cpu = ["CPUExecutionProvider"]  # onnxruntime's GPU provider fails to load here (error 126); CPU is fast enough
        self.v2 = MiniFASNet(model_name=MiniFASNetWeights.V2, providers=cpu)
        self.v1 = MiniFASNet(model_name=MiniFASNetWeights.V1SE, providers=cpu)

    @staticmethod
    def _p_real(model, img, bbox):
        r = model.predict(img, [float(v) for v in bbox])
        return r.confidence if r.is_real else 1.0 - r.confidence

    def check(self, image_bgr, bbox):
        ok, hint, pct = framing(bbox, image_bgr.shape)
        if not ok:
            return {"verdict": "uncertain", "real_prob": None, "face_pct": round(pct, 3), "note": hint}
        p = (self._p_real(self.v2, image_bgr, bbox) + self._p_real(self.v1, image_bgr, bbox)) / 2
        verdict = "real" if p >= PASS else ("uncertain" if p >= HARD_FAKE else "fake")
        return {"verdict": verdict, "real_prob": round(float(p), 4), "face_pct": round(pct, 3), "note": ""}
