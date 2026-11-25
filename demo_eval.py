"""
Evaluates the AI-MIDS demo database: searches every probe photo through the
real API (POST /api/search, k=10) and reports whether the correct person
comes back, and at which rank.

Usage (API must be running on the demo database):
    python demo_eval.py --api http://127.0.0.1:8000
    python demo_eval.py --limit 20      # smoke test

Outputs:
    demo_results/demo_eval.csv     one row per probe
    demo_results/demo_summary.json full summary (also copied to
                                   demo_probes/summary.json for the demo page)
"""

import argparse
import csv
import json
import os
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES_DIR = os.path.join(HERE, "demo_probes")
MANIFEST_PATH = os.path.join(PROBES_DIR, "manifest.json")
RESULTS_DIR = os.path.join(HERE, "demo_results")

DATASET_LABELS = {
    "agedb": "AgeDB (2 oldest photos held out)",
    "fgnet": "FG-NET (2 oldest photos held out)",
    "calfw": "CALFW (2 photos held out)",
    "caf": "CAF (child -> adult, ages 0-94)",
    "ecaf": "ECAF (long-missing child -> adult)",
}


def resolve_paths(work_dir=None):
    base = os.path.abspath(work_dir) if work_dir else HERE
    probes_dir = os.path.join(base, "demo_probes")
    manifest_path = os.path.join(probes_dir, "manifest.json")
    results_dir = os.path.join(base, "demo_results")
    return probes_dir, manifest_path, results_dir


def search_probe(client, api, probe_path):
    mime = "image/png" if probe_path.lower().endswith(".png") else "image/jpeg"
    with open(probe_path, "rb") as fh:
        files = {"file": (os.path.basename(probe_path), fh, mime)}
        r = client.post(f"{api}/api/search", data={"k": "10", "min_sim": "-1.0"},
                        files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def evaluate(client, api, probes, limit, probes_dir=PROBES_DIR):
    rows = []
    for i, p in enumerate(probes):
        if limit is not None and i >= limit:
            break
        probe_path = os.path.join(probes_dir, p["file"])
        rec = {
            "file": p["file"],
            "dataset": p["dataset"],
            "expected_name": p["expected_name"],
            "probe_age": p.get("probe_age", ""),
            "rank": "",
            "found": 0,
            "similarity": "",
            "tier": "",
        }
        try:
            resp = search_probe(client, api, probe_path)
        except requests.RequestException as e:
            rec["note"] = f"API error: {e}"
            rows.append(rec)
            continue

        if resp.get("faces_detected", 0) == 0:
            rec["note"] = "no face detected in probe"
            rows.append(rec)
            continue

        matches = resp["results"][0].get("matches", [])
        for rank, m in enumerate(matches, start=1):
            if m["name"] == p["expected_name"]:
                rec.update(rank=rank, found=1,
                           similarity=round(m["similarity"], 4), tier=m["tier"])
                break
        if not rec["found"]:
            rec["note"] = "not in top 10"
        rows.append(rec)

        if (i + 1) % 100 == 0:
            found = sum(r["found"] for r in rows)
            print(f">> {i + 1}/{len(probes)} probes | correct in top-10 so far: "
                  f"{found} ({found / len(rows) * 100:.1f}%)", flush=True)
    return rows


def summarize(rows):
    def pct(n, d):
        return round(n / d, 4) if d else None

    def block(sub):
        n = len(sub)
        rank1 = sum(1 for r in sub if r["rank"] == 1)
        top5 = sum(1 for r in sub if isinstance(r["rank"], int) and r["rank"] <= 5)
        top10 = sum(1 for r in sub if r["found"])
        out = {
            "probes": n,
            "rank1": pct(rank1, n),
            "top5": pct(top5, n),
            "top10": pct(top10, n),
        }
        ages = [r["probe_age"] for r in sub if r["probe_age"] != ""]
        if ages:
            out["mean_probe_age"] = round(sum(ages) / len(ages), 1)
        tiers = {}
        for r in sub:
            if r["found"]:
                tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        out["tiers_of_correct_hits"] = tiers
        return out

    datasets = sorted({r["dataset"] for r in rows})
    return {
        "overall": block(rows),
        "datasets": {d: block([r for r in rows if r["dataset"] == d]) for d in datasets},
    }


def enrich_with_gaps(rows, manifest_path=MANIFEST_PATH):
    """AgeDB/FG-NET probes carry probe_age; gap vs gallery is stored in the manifest."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    age_map = {p["file"]: p for p in manifest["probes"]}
    for r in rows:
        p = age_map.get(r["file"], {})
        if "probe_age" in p and "gallery_age_max" in p:
            r["age_gap_years"] = p["probe_age"] - p["gallery_age_max"]
        else:
            r["age_gap_years"] = ""


def main():
    parser = argparse.ArgumentParser(description="AI-MIDS demo evaluator")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--work-dir", default=None,
                        help="optional isolated directory containing demo_probes and demo_results")
    parser.add_argument("--results-dir", default=None,
                        help="optional directory for demo_eval.csv and demo_summary.json (does not overwrite demo_probes/summary.json when set)")
    args = parser.parse_args()

    probes_dir, manifest_path, results_dir = resolve_paths(args.work_dir)
    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    with open(manifest_path, encoding="utf-8") as f:
        probes = json.load(f)["probes"]
    print(f">> Loaded {len(probes)} probes from manifest.")

    client = requests.Session()
    r = client.get(f"{args.api}/", timeout=30)
    r.raise_for_status()
    info = r.json()
    cal_mode = info.get("calibration", "legacy")
    print(f">> Connected to {info['app']} | engine={info['engine']} | calibration={cal_mode} | db={info['database']}")
    stats = client.get(f"{args.api}/api/stats", timeout=30).json()
    print(f">> Database: {stats.get('people')} people, {stats.get('photos')} photos")

    t0 = time.time()
    rows = evaluate(client, args.api, probes, args.limit, probes_dir=probes_dir)
    enrich_with_gaps(rows, manifest_path=manifest_path)
    summary = summarize(rows)
    summary["database"] = {"people": stats.get("people"), "photos": stats.get("photos")}
    summary["engine"] = info["engine"]
    summary["calibration"] = cal_mode
    elapsed = time.time() - t0

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "demo_eval.csv")
    fields = ["file", "dataset", "expected_name", "probe_age", "age_gap_years",
              "found", "rank", "similarity", "tier", "note"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    targets = [os.path.join(results_dir, "demo_summary.json")]
    if not args.results_dir:
        os.makedirs(probes_dir, exist_ok=True)
        targets.append(os.path.join(probes_dir, "summary.json"))
    for target in targets:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)

    print("\n================ DEMO EVAL SUMMARY ================")
    print(f"  Overall: {summary['overall']['probes']} probes | "
          f"#1 hit {summary['overall']['rank1']*100:.1f}% | "
          f"top-5 {summary['overall']['top5']*100:.1f}% | "
          f"top-10 {summary['overall']['top10']*100:.1f}%")
    for d, b in summary["datasets"].items():
        print(f"  {DATASET_LABELS.get(d, d)}: {b['probes']} probes | "
              f"#1 {b['rank1']*100:.1f}% | top-5 {b['top5']*100:.1f}% | "
              f"top-10 {b['top10']*100:.1f}%")
    print(f"  Runtime: {elapsed/60:.1f} min")
    print(f"  CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
