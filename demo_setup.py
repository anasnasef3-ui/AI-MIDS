"""
Builds the AI-MIDS doctor-demo database from AgeDB, FG-NET and CALFW.

For every person in the three datasets:
  - their young/child photos are registered through the real API (POST /api/persons
    + POST /api/persons/{id}/photos) as missing-person records,
  - the 2 oldest photos (by age for AgeDB/FG-NET, by file counter for CALFW)
    are held back as search probes and copied to demo_probes/<dataset>/.

Requires the API to be already running, e.g.:
    set AIMIDS_DB_PATH=demo_aimids.db
    uvicorn api:app --port 8000

Usage:
    python demo_setup.py                       # all three datasets, full size
    python demo_setup.py --datasets fgnet --limit 10   # smoke test
Resume: safe to re-run after interruption; completed persons are skipped.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ARC_DIR = os.path.join(HERE, "Arc face")
sys.path.insert(0, ARC_DIR)

from bench_datasets import (  # noqa: E402
    load_lag_cohort_identities,
    normalize_identity,
    parse_agedb_images,
    parse_caf_images,
    parse_ecaf_images,
)

ASSETS = r"D:\GRAD\Assets"
AGEDB_DIR = os.path.join(ASSETS, "AgeDB", "images")
FGNET_DIR = os.path.join(ASSETS, "FGNET", "images")
CALFW_DIR = os.path.join(ASSETS, "calfw", "aligned images")
PROBES_DIR = os.path.join(HERE, "demo_probes")
STATE_PATH = os.path.join(PROBES_DIR, "_setup_state.json")
MANIFEST_PATH = os.path.join(PROBES_DIR, "manifest.json")

FGNET_RE = re.compile(r"^(\d+)A(\d+)\.JPG$", re.IGNORECASE)
CALFW_RE = re.compile(r"^(.+?)_(\d{4})\.jpg$", re.IGNORECASE)


def parse_fgnet():
    """Returns {person_id_str: [(age, path)]} from filenames like 001A02.JPG."""
    people = {}
    for fname in sorted(os.listdir(FGNET_DIR)):
        m = FGNET_RE.match(fname)
        if not m:
            continue
        pid, age = m.group(1), int(m.group(2))
        people.setdefault(pid, []).append((age, os.path.join(FGNET_DIR, fname)))
    return people


def parse_calfw():
    """Returns {display_name: [(counter, path)]}; CALFW has no age labels."""
    people = {}
    for fname in sorted(os.listdir(CALFW_DIR)):
        m = CALFW_RE.match(fname)
        if not m:
            continue
        name, counter = m.group(1), int(m.group(2))
        display = " ".join(part.capitalize() for part in name.split("_"))
        people.setdefault(display, []).append((counter, os.path.join(CALFW_DIR, fname)))
    return people


def parse_dataset(dataset):
    """Yields (person_key, display_name, sorted_photos, has_age) tuples."""
    if dataset == "agedb":
        identities, _ = parse_agedb_images(AGEDB_DIR)
        for name, photos in sorted(identities.items()):
            photos.sort(key=lambda x: x[0])
            yield name, f"AgeDB {name}", [(p[0], p[1]) for p in photos], True
    elif dataset == "fgnet":
        for pid, photos in sorted(parse_fgnet().items()):
            photos.sort(key=lambda x: x[0])
            yield pid, f"FG-NET {pid}", photos, True
    elif dataset == "calfw":
        for name, photos in sorted(parse_calfw().items()):
            photos.sort(key=lambda x: x[0])
            yield name, f"CALFW {name}", photos, False
    elif dataset == "caf":
        for norm_id, info in sorted(parse_caf_images().items()):
            yield norm_id, f"CAF {info['display_name']}", info["photos"], True
    elif dataset == "ecaf":
        for norm_id, info in sorted(parse_ecaf_images().items()):
            tagged = [(0, p) for _, p in info["child"]] + [(1, p) for _, p in info["adult"]]
            yield norm_id, f"ECAF {info['display_name']}", tagged, False
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def resolve_paths(work_dir=None):
    base = os.path.abspath(work_dir) if work_dir else HERE
    probes_dir = os.path.join(base, "demo_probes")
    state_path = os.path.join(probes_dir, "_setup_state.json")
    manifest_path = os.path.join(probes_dir, "manifest.json")
    return probes_dir, state_path, manifest_path


def load_state(state_path=STATE_PATH):
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    return {"done": {}, "manifest": []}


def save_state(state, state_path=STATE_PATH, manifest_path=MANIFEST_PATH):
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, state_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"probes": state["manifest"]}, f, indent=1)


def probe_filename(dataset, path):
    fname = os.path.basename(path)
    base, ext = os.path.splitext(fname)
    out_ext = ".png" if ext.lower() == ".png" else ".jpg"
    return f"{dataset}/{base}{out_ext}"


def copy_probe(dataset, path, probes_dir=PROBES_DIR):
    rel = probe_filename(dataset, path)
    dst = os.path.join(probes_dir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.isfile(dst):
        shutil.copyfile(path, dst)
    return rel


def _post_photo(client, api, person_id, path, data, face_index=None):
    body = dict(data)
    if face_index is not None:
        body["face_index"] = str(face_index)
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as fh:
        files = {"file": (os.path.basename(path), fh, mime)}
        return client.post(f"{api}/api/persons/{person_id}/photos", data=body,
                           files=files, timeout=180)


def _largest_face_index(faces):
    """Index of the biggest face box - in a single-subject dataset photo that is the subject."""
    def area(f):
        x1, y1, x2, y2 = f["bbox"]
        return (x2 - x1) * (y2 - y1)
    return max(range(len(faces)), key=lambda i: area(faces[i]))


def register_person(client, api, person_key, dataset, display_name, gallery, has_age):
    """Registers one person via the API. Returns (person_id, n_photos, failures)."""
    if has_age:
        notes = f"Demo gallery ({dataset}): ages {gallery[0][0]}-{gallery[-1][0]}"
    else:
        notes = f"Demo gallery ({dataset})"
    r = client.post(f"{api}/api/persons", data={"name": display_name, "notes": notes}, timeout=60)
    r.raise_for_status()
    person_id = r.json()["person_id"]

    n_ok, failures = 0, []
    for age, path in gallery:
        data = {"age": str(age)} if has_age else {}
        rp = _post_photo(client, api, person_id, path, data)
        try:
            body = rp.json()
        except ValueError:
            body = {}
        if rp.status_code == 200 and body.get("status") == "choose_face":
            faces = body.get("faces") or []
            if faces:
                rp = _post_photo(client, api, person_id, path, data,
                                 face_index=_largest_face_index(faces))
                try:
                    body = rp.json()
                except ValueError:
                    body = {}
        stored = rp.status_code == 200 and "face_id" in body
        if stored:
            n_ok += 1
        else:
            failures.append(os.path.basename(path))

    if n_ok == 0:
        client.delete(f"{api}/api/persons/{person_id}", timeout=60)
        return None, 0, failures
    return person_id, n_ok, failures


def reconcile_orphans(client, api, state):
    """Deletes demo persons registered after the last checkpoint of an interrupted run.

    Such persons are in the database but not in the state file, so the resumed run
    would register them a second time. Must only run while no other build is active.
    """
    kept = {e["person_id"] for e in state["done"].values() if "person_id" in e}
    r = client.get(f"{api}/api/persons", timeout=120)
    r.raise_for_status()
    orphans = [p["person_id"] for p in r.json()
               if (p.get("notes") or "").startswith("Demo gallery (") and p["person_id"] not in kept]
    for pid in orphans:
        client.delete(f"{api}/api/persons/{pid}", timeout=60).raise_for_status()
    return len(orphans)


def _canonical_demo_key(dataset, person_key, display_name):
    if dataset == "fgnet":
        return normalize_identity(f"fgnet_{person_key}")
    return normalize_identity(display_name)


def _split_gallery_probes(dataset, photos, max_probes):
    """Splits sorted photos into (gallery, probes)."""
    if len(photos) < 2:
        return [], []
    if dataset == "ecaf":
        child_items = [(None, path) for is_adult, path in photos if not is_adult]
        adult_items = [(None, path) for is_adult, path in photos if is_adult]
        if not child_items or not adult_items:
            return [], []
        k_probe = min(max(1, int(max_probes)), len(adult_items))
        return child_items, adult_items[:k_probe]
    n_probes = min(max(1, int(max_probes)), len(photos) - 1)
    return photos[:-n_probes], photos[-n_probes:]


def main():
    parser = argparse.ArgumentParser(description="AI-MIDS demo database builder")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--datasets", default="agedb,fgnet,calfw",
                        help="comma-separated subset of agedb,fgnet,calfw,caf,ecaf")
    parser.add_argument("--limit", type=int, default=None,
                        help="max persons per dataset (smoke test)")
    parser.add_argument("--probes", type=int, default=2,
                        help="oldest photos held back as search probes per person")
    parser.add_argument("--work-dir", default=None,
                        help="optional isolated directory for demo_probes, _setup_state.json, and manifest.json")
    parser.add_argument("--exclude-cohort", action="store_true",
                        help="skip identities present in lag_cohort_cache.npz (prevents AS-Norm cohort leakage)")
    parser.add_argument("--dedup-existing", action="store_true",
                        help="skip identities already present in the target DB or earlier datasets by normalized name")
    args = parser.parse_args()

    probes_dir, state_path, manifest_path = resolve_paths(args.work_dir)
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    os.makedirs(probes_dir, exist_ok=True)
    state = load_state(state_path)
    cohort_ids = load_lag_cohort_identities() if args.exclude_cohort else set()

    client = requests.Session()
    r = client.get(f"{args.api}/", timeout=30)
    r.raise_for_status()
    info = r.json()
    print(f">> Connected to {info['app']} | engine={info['engine']} | db={info['database']}")
    print(f">> Datasets: {datasets} | probes per person: {args.probes}"
          f" | exclude_cohort={args.exclude_cohort} | dedup_existing={args.dedup_existing}")
    removed = reconcile_orphans(client, args.api, state)
    if removed:
        print(f">> Removed {removed} leftover persons from an interrupted run")

    seen_norm_ids = set()
    if args.dedup_existing:
        existing_persons = client.get(f"{args.api}/api/persons", timeout=120)
        existing_persons.raise_for_status()
        for p in existing_persons.json():
            pname = p.get("name", "")
            # Normalize FG-NET prefix cleanly while normalizing all celebrity names
            if pname.upper().startswith("FG-NET "):
                seen_norm_ids.add(normalize_identity("fgnet_" + pname[7:].strip()))
            else:
                seen_norm_ids.add(normalize_identity(pname))
        for done_entry in state["done"].values():
            if isinstance(done_entry, dict) and not done_entry.get("skipped") and done_entry.get("expected_name"):
                ds_prev = done_entry.get("dataset", "")
                exp_name = done_entry["expected_name"]
                if ds_prev == "fgnet" and exp_name.upper().startswith("FG-NET "):
                    seen_norm_ids.add(normalize_identity("fgnet_" + exp_name[7:].strip()))
                else:
                    seen_norm_ids.add(normalize_identity(exp_name))

    t0 = time.time()
    totals = {"persons": 0, "photos": 0, "skipped_persons": 0, "failed_photos": 0}
    for dataset in datasets:
        ds_persons = ds_photos = ds_failed = 0
        count = 0
        for person_key, display_name, photos, has_age in parse_dataset(dataset):
            key = f"{dataset}:{person_key}"
            if key in state["done"]:
                continue

            canon_key = _canonical_demo_key(dataset, person_key, display_name)
            if args.exclude_cohort and canon_key in cohort_ids:
                totals["skipped_persons"] += 1
                state["done"][key] = {"skipped": True, "reason": "cohort excluded"}
                continue
            if args.dedup_existing and canon_key in seen_norm_ids:
                totals["skipped_persons"] += 1
                state["done"][key] = {"skipped": True, "reason": "duplicate identity"}
                continue

            if args.limit is not None and count >= args.limit:
                break
            count += 1

            gallery, probes = _split_gallery_probes(dataset, photos, args.probes)
            if not gallery or not probes:
                totals["skipped_persons"] += 1
                state["done"][key] = {"skipped": True, "reason": "insufficient gallery/probe photos"}
                continue

            try:
                person_id, n_ok, failures = register_person(
                    client, args.api, person_key, dataset, display_name, gallery, has_age)
            except requests.RequestException as e:
                print(f"!! API error on {key}: {e} — stopping; re-run to resume.")
                return 1

            if person_id is None:
                totals["skipped_persons"] += 1
                state["done"][key] = {"skipped": True, "reason": "no face detected in gallery"}
                continue

            entry = {
                "person_id": person_id,
                "dataset": dataset,
                "expected_name": display_name,
            }
            if has_age:
                entry["probe_age"] = probes[-1][0]
                entry["gallery_age_min"] = gallery[0][0]
                entry["gallery_age_max"] = gallery[-1][0]
                entry["age_gap_years"] = probes[-1][0] - gallery[-1][0]
            for age, path in probes:
                rel = copy_probe(dataset, path, probes_dir=probes_dir)
                state["manifest"].append({
                    "file": rel,
                    "dataset": dataset,
                    "expected_name": display_name,
                    **({"probe_age": age} if has_age else {}),
                })
            state["done"][key] = entry
            if args.dedup_existing:
                seen_norm_ids.add(canon_key)

            ds_persons += 1
            ds_photos += n_ok
            ds_failed += len(failures)
            totals["persons"] += 1
            totals["photos"] += n_ok
            totals["failed_photos"] += len(failures)

            if ds_persons % 25 == 0:
                save_state(state, state_path=state_path, manifest_path=manifest_path)
                rate = totals["photos"] / max(time.time() - t0, 1e-6)
                print(f">> [{dataset}] {ds_persons} persons | {ds_photos} photos "
                      f"({rate:.1f} photos/s) | failed: {ds_failed}", flush=True)

        save_state(state, state_path=state_path, manifest_path=manifest_path)
        print(f">> [{dataset}] DONE: {ds_persons} persons, {ds_photos} photos, "
              f"{ds_failed} failed photos")

    save_state(state, state_path=state_path, manifest_path=manifest_path)
    elapsed = time.time() - t0
    print("\n================ DEMO SETUP SUMMARY ================")
    print(f"  Persons registered : {totals['persons']}")
    print(f"  Photos registered  : {totals['photos']}")
    print(f"  Failed photos      : {totals['failed_photos']}")
    print(f"  Persons skipped    : {totals['skipped_persons']} (no usable gallery)")
    print(f"  Probes written     : {len(state['manifest'])} -> {probes_dir}")
    print(f"  Manifest           : {manifest_path}")
    print(f"  Runtime            : {elapsed/60:.1f} min")
    next_cmd = "python demo_eval.py --api " + args.api
    if args.work_dir:
        next_cmd += f' --work-dir "{args.work_dir}"'
    print("  Next: " + next_cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
