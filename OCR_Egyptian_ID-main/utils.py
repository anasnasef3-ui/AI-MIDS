"""
utils.py - Egyptian ID card reader (card detection -> field detection -> digits -> decode -> photo).

Drop-in replacement for the old utils.py: the old function names and return values still work
(detect_and_process_id_card, process_image, decode_egyptian_id, detect_national_id, expand_bbox_height).

What changed:
  * models load on first use and are found next to this file, so it works from any folder
  * when several boxes are found (card / photo / nid) the most confident one is used, not the last one
  * duplicate/overlapping digit boxes are removed, and extra boxes beyond 14 are dropped
  * the 13-digit "missing digit" recovery only inserts one digit, and no longer blocks other fixes
  * decoding never crashes on a misread number: it reports problems instead
  * no files are written unless you run this file directly
  * new read_id_card() returns everything in one dict, including a valid flag and a problems list
"""
import os
from datetime import date

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_models = {}
# CPU on purpose: the models are tiny (~6 MB) so a card takes well under a second, and running them on
# the GPU crashed with "CUDA error: unknown error" after the webcam loop in id_verify.py.
DEVICE = "cpu"

GOVERNORATES = {
    '01': 'Cairo', '02': 'Alexandria', '03': 'Port Said', '04': 'Suez',
    '11': 'Damietta', '12': 'Dakahlia', '13': 'Ash Sharqia', '14': 'Kaliobeya',
    '15': 'Kafr El-Sheikh', '16': 'Gharbia', '17': 'Monoufia', '18': 'El Beheira',
    '19': 'Ismailia', '21': 'Giza', '22': 'Beni Suef', '23': 'Fayoum',
    '24': 'El Menia', '25': 'Assiut', '26': 'Sohag', '27': 'Qena',
    '28': 'Aswan', '29': 'Luxor', '31': 'Red Sea', '32': 'New Valley',
    '33': 'Matrouh', '34': 'North Sinai', '35': 'South Sinai', '88': 'Foreign'
}


def _get_models():
    if not _models:
        from ultralytics import YOLO
        _models["card"] = YOLO(os.path.join(HERE, 'best.pt'))                # your own card detector
        _models["objects"] = YOLO(os.path.join(HERE, 'detect_odjects.pt'))   # field locator
        _models["digit"] = YOLO(os.path.join(HERE, 'detect_id_new.pt'))      # your retrained digit model
    return _models


def _imread(src):
    """Path (any characters) or an already-loaded image."""
    if isinstance(src, np.ndarray):
        return src
    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src)
    return img


def _clamp(bbox, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    return [max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h)]


# --------------------------------------------------------------------------- digits
def _dedupe_digits(dets):
    """dets: list of (digit, x1, x2, conf). Drops boxes that overlap a more confident one."""
    kept = []
    for d in sorted(dets, key=lambda d: -d[3]):
        w = max(d[2] - d[1], 1)
        if all(min(d[2], k[2]) - max(d[1], k[1]) <= 0.5 * min(w, max(k[2] - k[1], 1)) for k in kept):
            kept.append(d)
    return sorted(kept, key=lambda d: d[1])


def detect_national_id(cropped_image):
    digit_model = _get_models()["digit"]
    results = digit_model(cropped_image, conf=0.1, iou=0.3, verbose=False, device=DEVICE)

    dets = []
    for result in results:
        for box in result.boxes:
            x1, _, x2, _ = map(int, box.xyxy[0])
            dets.append((int(box.cls), x1, x2, float(box.conf)))

    dets = _dedupe_digits(dets)
    if len(dets) > 14:  # extra boxes are noise: keep the 14 most confident
        dets = sorted(sorted(dets, key=lambda d: -d[3])[:14], key=lambda d: d[1])

    # exactly one digit missing? look for an unusually wide gap and re-try just that slice
    if len(dets) == 13:
        gaps = [dets[i + 1][1] - dets[i][2] for i in range(12)]
        median_gap = sorted(gaps)[len(gaps) // 2]
        i = int(np.argmax(gaps))
        if gaps[i] > max(median_gap, 1) * 2.5:
            gx1, gx2 = dets[i][2], dets[i + 1][1]
            if gx2 - gx1 >= 3:
                retry = digit_model(cropped_image[:, gx1:gx2], conf=0.01, verbose=False, device=DEVICE)
                cands = [(int(b.cls), gx1, gx2, float(b.conf)) for r in retry for b in r.boxes]
                if cands:
                    best = max(cands, key=lambda c: c[3])
                    dets.insert(i + 1, best)
                    print(f"Recovered missing digit: {best[0]} (conf={best[3]:.2f})")

    return ''.join(str(d[0]) for d in dets)


# --------------------------------------------------------------------------- decoding
def decode_egyptian_id(id_number):
    """Never raises. Invalid parts come back as None, with an explanation in 'Problems'.

    Validation is structural: 14 digits, century digit, a governorate code that exists, and a real
    calendar date that is not in the future. The 14th digit is the registry's check digit, but its
    algorithm is not publicly documented (Wikipedia calls it optional, and the open-source
    validators that claim a rule contradict each other), so it is NOT verified here.
    """
    out = {'Birth Date': None, 'Governorate': None, 'Gender': None, 'Problems': []}
    if not (id_number.isdigit() and len(id_number) == 14):
        out['Problems'].append(f"national ID is '{id_number}' - expected exactly 14 digits")
        return out

    century = int(id_number[0])
    year, month, day = int(id_number[1:3]), int(id_number[3:5]), int(id_number[5:7])
    out['Gender'] = "Male" if int(id_number[12]) % 2 else "Female"

    gov = GOVERNORATES.get(id_number[7:9])
    if gov:
        out['Governorate'] = gov
    else:
        out['Problems'].append(f"unknown governorate code {id_number[7:9]}")

    if century not in (2, 3):
        out['Problems'].append(f"first digit {century} is not 2 or 3")
        return out
    try:
        born = date(1900 + (century - 2) * 100 + year, month, day)
        if born > date.today():
            out['Problems'].append("birth date is in the future")
        else:
            out['Birth Date'] = born.isoformat()
    except ValueError:
        out['Problems'].append(f"impossible birth date (month {month}, day {day})")
    return out


def expand_bbox_height(bbox, scale=1.2, image_shape=None):
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    center_y = y1 + height // 2
    new_height = int(height * scale)
    bottom = center_y + new_height // 2
    if image_shape is not None:
        bottom = min(bottom, image_shape[0])
    return [x1, max(center_y - new_height // 2, 0), x2, bottom]


# --------------------------------------------------------------------------- reading a card
def _read_fields(card_img):
    objects_model = _get_models()["objects"]
    best = {}  # class name -> (conf, bbox)
    for result in objects_model(card_img, verbose=False, device=DEVICE):
        for box in result.boxes:
            name = result.names[int(box.cls[0].item())].replace('invalid_', '')
            conf = float(box.conf[0].item())
            if name in ('photo', 'nid') and (name not in best or conf > best[name][0]):
                best[name] = (conf, [int(c) for c in box.xyxy[0].tolist()])

    photo, nid_crop, nid = None, None, ''
    if 'photo' in best:
        x1, y1, x2, y2 = _clamp(best['photo'][1], card_img.shape)
        photo = card_img[y1:y2, x1:x2]
    if 'nid' in best:
        ex = expand_bbox_height(best['nid'][1], scale=1.5, image_shape=card_img.shape)
        x1, y1, x2, y2 = _clamp(ex, card_img.shape)
        nid_crop = card_img[y1:y2, x1:x2]
        nid = detect_national_id(nid_crop)

    info = decode_egyptian_id(nid)
    problems = list(info['Problems'])
    if photo is None:
        problems.append("no photo region found on the card")
    if 'nid' not in best:
        problems.append("no national-ID region found on the card")
    return {"nid": nid, "photo": photo, "nid_crop": nid_crop,
            "birth_date": info['Birth Date'], "governorate": info['Governorate'], "gender": info['Gender'],
            "valid": not problems, "problems": problems}


def _find_card(image):
    img = _imread(image)
    best = None
    for result in _get_models()["card"](img, verbose=False, device=DEVICE):
        for box in result.boxes:
            conf = float(box.conf[0].item())
            if best is None or conf > best[0]:
                best = (conf, [int(v) for v in box.xyxy[0].tolist()])
    if best is None:
        return None
    x1, y1, x2, y2 = _clamp(best[1], img.shape)
    return img[y1:y2, x1:x2]


def read_id_card(image):
    """image: file path or BGR array. Always returns a dict; check ['valid'] and ['problems']."""
    card = _find_card(image)
    if card is None or card.size == 0:
        return {"nid": "", "photo": None, "nid_crop": None, "birth_date": None, "governorate": None,
                "gender": None, "valid": False, "problems": ["no ID card detected in the image"]}
    return _read_fields(card)


# --------------------------------------------------------------------------- old-style API
def process_image(cropped_image):
    r = _read_fields(cropped_image)
    print(f"National ID: {r['nid']}")
    print(f"Photo extracted: {'yes' if r['photo'] is not None else 'no'}")
    for p in r["problems"]:
        print("WARNING:", p)
    return r["nid"], r["photo"], r["birth_date"], r["governorate"], r["gender"]


def detect_and_process_id_card(image_path):
    card = _find_card(image_path)
    if card is None or card.size == 0:
        raise RuntimeError("No ID card detected in the image.")
    return process_image(card)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'ID.jpg'
    r = read_id_card(path)
    print("\n=== RESULT ===")
    print("National ID:", r["nid"])
    print("Birth Date:", r["birth_date"])
    print("Governorate:", r["governorate"])
    print("Gender:", r["gender"])
    print("Valid:", r["valid"], r["problems"] or "")
    if r["photo"] is not None:
        cv2.imencode(".png", r["photo"])[1].tofile("extracted_photo.png")
        print("Photo saved to extracted_photo.png")
    if r["nid_crop"] is not None:
        cv2.imencode(".png", r["nid_crop"])[1].tofile("debug_nid_crop.png")
