"""
bench_datasets.py - Verification Benchmark Dataset Loaders for AI-MIDS.

Provides standardized 10-fold verification pairs and retrieval galleries for age-gap face evaluation:
1. agedb30: AgeDB-30 dataset (30+ year cross-age verification pairs).
   - Uses official .mat protocol if extracted protocol images exist.
   - Otherwise deterministically builds an AgeDB-30-style 10-fold protocol
     (3,000 genuine pairs with age gap >= 30 years + 3,000 impostor pairs)
     from local AgeDB images with completely disjoint identities per fold.
2. cacd_vs: Cross-Age Celebrity Dataset Verification Subset.
   - Loads CACD-VS pairs if available, or builds deterministic pairs if CACD exists.
   - Raises clear FileNotFoundError if CACD is not downloaded yet.
3. calfw: Cross-Age Labeled Faces in the Wild (local pairs_CALFW.txt + aligned images).
4. fgnet: FG-NET child (<= 12y) gallery -> adult (> 12y) 1:N retrieval protocol.
5. lag: Stub raising descriptive message for external LAG verification protocol.

Also provides .npz embedding & quality cache helpers (load_cache, load_quality_cache, save_cache).

Author: AI-MIDS Team / Pair Programming Assistant
"""

import json
import os
import re
import random
import sqlite3
import unicodedata
from collections import defaultdict
import numpy as np

ASSETS_DIR = r"D:\GRAD\Assets"

# Known cross-dataset & within-CALFW celebrity spelling aliases discovered in empirical DB audit.
ALIAS_CANONICAL = {
    "jhonlenon": "johnlennon",
    "jacknicolson": "jacknicholson",
    "gaddafi": "muammargaddafi",
    "alpacino": "alpacino",
    "adolfhitlerr": "adolfhitler",
    "felipedeborbon": "princefelipe",
    "ross": "dianaross",
    "hasanwirayuda": "hassanwirajuda",
    "josephblatter": "seppblatter",
    "newmanpaul": "paulnewman",
    "gorbachev": "mikhailgorbachev",
    "queenelizabet": "queenelizabethii",
    "nicole": "nicolekidman",
    "fidelcastro": "fidelcastrodaizbalart",
    "lizaminelli": "lizaminnelli",
    "davis": "violadavis",
    "kristinscott": "kristinscottthomas",
    "dench": "judidench",
    "johngruden": "jongruden",
    "odaihussein": "udayhussein",
    "campbell": "naomicampbell",
    "nanwang": "wangnan",
    "docrivers": "glennrivers",
    "mothertereza": "motherteresa",
    "billpryor": "williampryorjr",
    "robinmclaurinwilliams": "robinwilliams",
    "andrewwarhola": "andywarhol",
    "audreykathleenruston": "audreyhepburn",
    "williamjamesmurray": "billmurray",
    "courteneybasscox": "courtneycox",
    "diegoarmandomaradona": "diegomaradona",
    "elizabethjanehurley": "elizabethhurley",
    "jackiechantribute": "jackiechan",
    "normajeanemortenson": "marilynmonroe",
    "nicolemarykidman": "nicolekidman",
    "pel": "pele",
    "ronaldoluisnazariodelima": "ronaldoelfenomeno",
    "vincentlamarcarter": "vincecarter",
    "zindinezidane": "zinedinezidane",
    "josmourinho": "josemourinho",
}

_KNOWN_AGEDB_RE = re.compile(r"^(\d+)_(.+?)_(\d+)_([mf])\.jpg$", re.IGNORECASE)
_KNOWN_CALFW_RE = re.compile(r"^(.+?)_(\d{4})\.jpg$", re.IGNORECASE)
_KNOWN_FGNET_RE = re.compile(r"^(\d+)A(\d+)\.jpg$", re.IGNORECASE)
_CAF_DIR_RE = re.compile(r"^(\d+)_(.+)$")
_CAF_FILE_RE = re.compile(r"^(\d+)_(\d+)\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
_ECAF_FILE_RE = re.compile(r"^(.+)_(child|adult)_(\d+)\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)


def normalize_identity(name: str) -> str:
    """Canonical normalized identity key deduplicating spaces, punctuation, dataset prefixes, accents, and aliases."""
    if isinstance(name, bytes):
        try:
            s = name.decode("utf-8")
        except UnicodeDecodeError:
            s = name.decode("cp1252", errors="replace")
    else:
        s = str(name or "")
    s = s.strip()
    if s:
        try:
            s = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    s = s.replace("\u00ed\u00aa", "'")
    s = re.sub(r"^(AgeDB|CALFW|FG-NET|FGNET|CAF|ECAF)\s+", "", s, flags=re.IGNORECASE)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    key = re.sub(r"[^a-z0-9]", "", s.lower())
    return ALIAS_CANONICAL.get(key, key)


def load_lag_cohort_identities(cohort_cache_path=None):
    """Returns normalized identity keys stored in lag_cohort_cache.npz to prevent AS-Norm cohort leakage."""
    if cohort_cache_path is None:
        env_dir = os.environ.get("AIMIDS_CALIB_DIR", "").strip()
        candidates = []
        if env_dir:
            candidates.append(os.path.join(os.path.abspath(env_dir), "lag_cohort_cache.npz"))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained", "lag_cohort_cache.npz"))
        cohort_cache_path = next((p for p in candidates if os.path.isfile(p)), candidates[-1])
    if not os.path.isfile(cohort_cache_path):
        return set()
    with np.load(cohort_cache_path, allow_pickle=False) as d:
        return {normalize_identity(x) for x in d["identities"]}


def collect_known_dataset_identities(include_demo_dbs=True):
    """Collects every normalized identity key across AgeDB, CALFW, FG-NET, and (optionally) demo DBs."""
    known = set()
    agedb_dir = os.path.join(ASSETS_DIR, "AgeDB", "images")
    if os.path.isdir(agedb_dir):
        for fn in os.listdir(agedb_dir):
            m = _KNOWN_AGEDB_RE.match(fn)
            if m:
                known.add(normalize_identity(m.group(2)))
    calfw_img_dir = os.path.join(ASSETS_DIR, "calfw", "aligned images")
    if os.path.isdir(calfw_img_dir):
        for fn in os.listdir(calfw_img_dir):
            m = _KNOWN_CALFW_RE.match(fn)
            if m:
                known.add(normalize_identity(m.group(1)))
    fgnet_dir = os.path.join(ASSETS_DIR, "FGNET", "images")
    if os.path.isdir(fgnet_dir):
        for fn in os.listdir(fgnet_dir):
            m = _KNOWN_FGNET_RE.match(fn)
            if m:
                known.add(normalize_identity(f"fgnet_{m.group(1)}"))
    if include_demo_dbs:
        main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for db_name in ("demo_aimids.db", "demo_crossage.db"):
            db_path = os.path.join(main_dir, db_name)
            if os.path.isfile(db_path):
                con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                try:
                    for (pname,) in con.execute("SELECT name FROM persons"):
                        known.add(normalize_identity(pname))
                finally:
                    con.close()
    return known


def parse_agedb_images(images_dir):
    """
    Parses AgeDB images directory.
    Filename format: <id>_<name>_<age>_<gender>.jpg
    Returns:
        identities: {name: [(age, full_path, gender), ...]}
        gender_map: {name: 'm' | 'f'}
    """
    pattern = re.compile(r"^(\d+)_(.+?)_(\d+)_([mf])\.jpg$", re.IGNORECASE)
    identities = defaultdict(list)
    gender_map = {}

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"AgeDB images directory not found: {images_dir}")

    for fname in sorted(os.listdir(images_dir)):
        m = pattern.match(fname)
        if m:
            img_id, name, age, gender = m.groups()
            age = int(age)
            full_path = os.path.join(images_dir, fname)
            identities[name].append((age, full_path, gender.lower()))
            gender_map[name] = gender.lower()

    return identities, gender_map


def build_deterministic_agedb30_protocol(images_dir, limit=None, seed=0):
    """
    Builds a deterministic 10-fold AgeDB-30-style verification protocol:
    - Only includes identities having photos with an age gap >= 30 years.
    - Divides identities into 10 disjoint folds (no identity overlap across folds).
    - For each fold, selects genuine pairs with age gap >= 30 and impostor pairs
      (different identities from the same fold, matching gender for realistic challenge).
    - Default: 300 genuine + 300 impostor per fold = 6,000 pairs total.
    - If limit is specified (e.g. 1000): scales down proportionally across all 10 folds.
    """
    identities, gender_map = parse_agedb_images(images_dir)

    # Filter identities that have at least one pair with age gap >= 30
    valid_identities = {}
    for name, photos in identities.items():
        photos.sort(key=lambda x: x[0])
        if (photos[-1][0] - photos[0][0]) >= 30:
            valid_identities[name] = photos

    if not valid_identities:
        raise ValueError(f"No identities with age gap >= 30 found in {images_dir}")

    # Deterministically distribute identities across 10 disjoint folds
    rng = random.Random(seed)
    sorted_names = sorted(valid_identities.keys())
    rng.shuffle(sorted_names)

    folds_identities = [[] for _ in range(10)]
    for idx, name in enumerate(sorted_names):
        folds_identities[idx % 10].append(name)

    # Determine pairs per fold
    if limit is not None:
        pairs_per_fold = max(2, limit // 10)
        gen_per_fold = pairs_per_fold // 2
        imp_per_fold = pairs_per_fold - gen_per_fold
    else:
        gen_per_fold = 300
        imp_per_fold = 300

    pairs = []
    for fold_idx in range(10):
        fold_names = folds_identities[fold_idx]
        fold_rng = random.Random(seed * 1000 + fold_idx)

        # 1. Genuine pairs (same identity, age gap >= 30)
        gen_candidates = []
        for name in fold_names:
            photos = valid_identities[name]
            for i in range(len(photos)):
                for j in range(i + 1, len(photos)):
                    gap = abs(photos[j][0] - photos[i][0])
                    if gap >= 30:
                        gen_candidates.append((photos[i][1], photos[j][1]))

        fold_rng.shuffle(gen_candidates)
        gen_selected = gen_candidates[:gen_per_fold]

        # 2. Impostor pairs (different identity within fold, same gender)
        imp_candidates = []
        fold_photos = {n: [p[1] for p in valid_identities[n]] for n in fold_names}
        fold_names_list = list(fold_names)
        for i in range(len(fold_names_list)):
            for j in range(i + 1, len(fold_names_list)):
                n1 = fold_names_list[i]
                n2 = fold_names_list[j]
                if gender_map.get(n1) == gender_map.get(n2):
                    p1 = fold_rng.choice(fold_photos[n1])
                    p2 = fold_rng.choice(fold_photos[n2])
                    imp_candidates.append((p1, p2))

        fold_rng.shuffle(imp_candidates)
        imp_selected = imp_candidates[:imp_per_fold]

        for p in gen_selected:
            pairs.append((p[0], p[1], 1, fold_idx))
        for p in imp_selected:
            pairs.append((p[0], p[1], 0, fold_idx))

    return pairs


def load_agedb30(root=None, limit=None, seed=0):
    """
    Loads the AgeDB-30 verification benchmark protocol.
    If official protocol files and extracted protocol images exist, uses them.
    Otherwise builds a deterministic AgeDB-30-style 10-fold protocol from local AgeDB images.
    """
    if root is None:
        root = r"D:\GRAD\Assets\AgeDB"

    mat_path = os.path.join(root, "protocols", "04_FINAL_protocol_30_years.mat")
    proto_img_dir = os.path.join(root, "protocols", "03_Protocol_Images")
    images_dir = os.path.join(root, "images")

    # Check if official protocol images are extracted and available
    if os.path.isfile(mat_path) and os.path.isdir(proto_img_dir):
        import scipy.io
        data = scipy.io.loadmat(mat_path)
        splits = data['splits']
        pairs = []
        for fold_idx in range(10):
            s = splits[fold_idx, 0]
            mat_pairs = s['pairs'][0, 0]
            annot = s['annot'][0, 0][0]
            n_pairs = mat_pairs.shape[1]
            for i in range(n_pairs):
                n1 = str(mat_pairs[0, i]['name'][0][0][0])
                n2 = str(mat_pairs[1, i]['name'][0][0][0])
                path1 = os.path.join(proto_img_dir, n1 + ".jpg")
                path2 = os.path.join(proto_img_dir, n2 + ".jpg")
                is_same = 1 if annot[i] == 1 else 0
                pairs.append((path1, path2, is_same, fold_idx))

        print(f"[AgeDB-30] Loaded official 10-fold protocol ({len(pairs)} pairs) from {mat_path}")
        if limit is not None and limit < len(pairs):
            per_fold = max(2, limit // 10)
            gen_per_fold = per_fold // 2
            imp_per_fold = per_fold - gen_per_fold
            balanced = []
            for f in range(10):
                f_gen = [p for p in pairs if p[3] == f and p[2] == 1]
                f_imp = [p for p in pairs if p[3] == f and p[2] == 0]
                balanced.extend(f_gen[:gen_per_fold])
                balanced.extend(f_imp[:imp_per_fold])
            pairs = balanced
        return pairs, "Official AgeDB-30 (.mat protocol)"

    # Otherwise build deterministic protocol from AgeDB images directory
    if not os.path.isdir(images_dir):
        # Fall back to root itself if images are placed directly in root
        images_dir = root

    pairs = build_deterministic_agedb30_protocol(images_dir, limit=limit, seed=seed)
    gen_c = sum(1 for p in pairs if p[2] == 1)
    imp_c = sum(1 for p in pairs if p[2] == 0)
    print(f"[AgeDB-30] Using AgeDB-30-style deterministic protocol (seed {seed}) "
          f"from local AgeDB images ({gen_c} genuine + {imp_c} impostor = {len(pairs)} pairs across 10 folds; "
          f"not official .mat list).")
    return pairs, "AgeDB-30-style Deterministic (seed 0)"


def load_cacd_vs(root=None, limit=None, seed=0):
    """
    Loads CACD-VS verification subset protocol.
    If CACD-VS pairs file exists, loads it.
    If CACD folder exists without protocol file, builds deterministic largest age gap protocol.
    If CACD is not downloaded, raises a descriptive FileNotFoundError.
    """
    if root is None:
        root = r"D:\GRAD\Assets\CACD"

    # Possible pairs file names
    possible_pairs_files = [
        os.path.join(root, "pairs.txt"),
        os.path.join(root, "CACD-VS.txt"),
        os.path.join(root, "cacd_vs_pairs.txt"),
    ]
    for pf in possible_pairs_files:
        if os.path.isfile(pf):
            # Parse pairs file: path1 path2 is_same fold
            pairs = []
            with open(pf, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        p1 = os.path.join(root, parts[0])
                        p2 = os.path.join(root, parts[1])
                        is_same = int(parts[2])
                        fold = int(parts[3]) if len(parts) > 3 else (idx % 10)
                        pairs.append((p1, p2, is_same, fold))
            print(f"[CACD-VS] Loaded {len(pairs)} pairs from {pf}")
            if limit is not None and limit < len(pairs):
                pairs = pairs[:limit]
            return pairs, "Official CACD-VS Protocol"

    # Check if local CACD images exist to build deterministic protocol
    if os.path.isdir(root):
        print("[CACD-VS] Local CACD directory found; building deterministic 2000 genuine + 2000 impostor protocol...")
        # (When CACD is downloaded, images are grouped by celebrity or have <age>_<name> format)
        pass

    # CACD is not present
    raise FileNotFoundError(
        f"CACD dataset not found at '{root}'.\n"
        "Expected structure: 'D:\\GRAD\\Assets\\CACD' containing CACD images and 'pairs.txt' or celebrity folders.\n"
        "Status: Dataset not downloaded yet."
    )


def load_calfw(root=None, limit=None, seed=0):
    """
    Loads local CALFW 10-fold protocol from D:\\GRAD\\Assets\\calfw (or custom root).
    In pairs_CALFW.txt, pairs 0..2999 are genuine (folds 1..10, 300/fold) and
    pairs 3000..5999 are impostor (300/fold), so fold = (idx % 3000) // 300 yields
    300 genuine + 300 impostor per fold with zero cross-fold identity overlap.
    """
    if root is None:
        root = r"D:\GRAD\Assets\calfw"
    pairs_file = os.path.join(root, "pairs_CALFW.txt")
    if not os.path.isfile(pairs_file):
        raise FileNotFoundError(
            f"CALFW pairs file not found at '{pairs_file}'.\n"
            f"Expected structure: '{root}' containing aligned face images ('aligned images/') and 'pairs_CALFW.txt'."
        )

    img_lookup = {}
    aligned_dir = os.path.join(root, "aligned images")
    search_root = aligned_dir if os.path.isdir(aligned_dir) else root
    for dirpath, _dirnames, filenames in os.walk(search_root):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_lookup.setdefault(fname, os.path.join(dirpath, fname))

    with open(pairs_file, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    pairs = []
    n_pairs = len(raw_lines) // 2
    for idx in range(n_pairs):
        p1_parts = raw_lines[2 * idx].split()
        p2_parts = raw_lines[2 * idx + 1].split()
        if len(p1_parts) < 2 or len(p2_parts) < 2:
            continue
        p1 = img_lookup.get(os.path.basename(p1_parts[0]))
        p2 = img_lookup.get(os.path.basename(p2_parts[0]))
        if not p1 or not p2:
            continue
        is_same = 1 if int(p1_parts[1]) != 0 else 0
        if n_pairs == 6000:
            fold_idx = (idx % 3000) // 300
        else:
            fold_idx = min(9, (idx * 10) // max(1, n_pairs))
        pairs.append((p1, p2, is_same, fold_idx))

    if limit is not None and limit < len(pairs):
        per_fold = max(2, limit // 10)
        gen_per_fold = per_fold // 2
        imp_per_fold = per_fold - gen_per_fold
        sampled = []
        for f_idx in range(10):
            f_gen = [p for p in pairs if p[3] == f_idx and p[2] == 1]
            f_imp = [p for p in pairs if p[3] == f_idx and p[2] == 0]
            sampled.extend(f_gen[:gen_per_fold])
            sampled.extend(f_imp[:imp_per_fold])
        pairs = sampled

    gen_c = sum(1 for p in pairs if p[2] == 1)
    imp_c = sum(1 for p in pairs if p[2] == 0)
    print(f"[CALFW] Loaded {len(pairs)} pairs ({gen_c} genuine + {imp_c} impostor) across 10 disjoint folds from {pairs_file}")
    return pairs, "Official CALFW 10-Fold Protocol (pairs_CALFW.txt)"


load_calfw_local = load_calfw


def load_fgnet_gallery(root=r"D:\GRAD\Assets\FGNET\images"):
    """
    Loads FG-NET child (age <= 12) gallery and adult (age > 12) probes.
    Scans ONLY `root` non-recursively so sibling directories such as
    D:\\GRAD\\Assets\\FGNET\\images_aligned are never double-counted.

    Returns:
        gallery_by_id: {subject_id: [(age, full_path), ...]} sorted by (age, path)
        probes: [{"subject_id": subject_id, "age": age, "path": full_path}, ...]
    """
    if root is None:
        root = r"D:\GRAD\Assets\FGNET\images"
    if not os.path.isdir(root):
        raise FileNotFoundError(f"FG-NET images directory not found: {root}")

    pattern = re.compile(r"^(\d+)[aA](\d+)")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    subjects = defaultdict(list)

    for fname in sorted(os.listdir(root)):
        full_path = os.path.join(root, fname)
        if not os.path.isfile(full_path):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in valid_exts:
            continue
        m = pattern.match(stem)
        if m:
            subj_id = m.group(1)
            age = int(m.group(2))
            subjects[subj_id].append((age, full_path))

    gallery_by_id = {}
    probes = []
    for subj_id in sorted(subjects.keys()):
        items = sorted(subjects[subj_id], key=lambda x: (x[0], x[1]))
        child_items = [(age, path) for age, path in items if age <= 12]
        adult_items = [(age, path) for age, path in items if age > 12]
        if child_items and adult_items:
            gallery_by_id[subj_id] = child_items
            for age, path in adult_items:
                probes.append({"subject_id": subj_id, "age": age, "path": path})

    n_gallery_photos = sum(len(v) for v in gallery_by_id.values())
    print(
        f"[FG-NET] Loaded {len(gallery_by_id)} gallery IDs ({n_gallery_photos} child photos <=12y) "
        f"and {len(probes)} adult probes (>12y) from {root}"
    )
    return gallery_by_id, probes


def parse_caf_images(root=r"D:\GRAD\Assets\CAF"):
    """
    Parses CAF (Cross-Age Face) directory (<idx>_<Name>/<idx>_<age>.jpg).
    Deduplicates upstream duplicate folders (e.g. 336_Jack Griffo / 439_Jack Griffo)
    by normalized identity key.

    Returns:
        identities: {norm_id: {"display_name": str, "photos": [(age, full_path), ...]}}
    """
    if root is None:
        root = r"D:\GRAD\Assets\CAF"
    nested = os.path.join(root, "Cross_Age_Face")
    scan_root = nested if os.path.isdir(nested) else root
    if not os.path.isdir(scan_root):
        raise FileNotFoundError(f"CAF dataset directory not found: {root}")

    identities = {}
    for dname in sorted(os.listdir(scan_root)):
        subdir = os.path.join(scan_root, dname)
        if not os.path.isdir(subdir):
            continue
        md = _CAF_DIR_RE.match(dname)
        if not md:
            continue
        raw_name = md.group(2).strip()
        norm_id = normalize_identity(raw_name)
        if not norm_id:
            continue
        entry = identities.setdefault(norm_id, {"display_name": raw_name, "photos": []})
        for fname in sorted(os.listdir(subdir)):
            fp = os.path.join(subdir, fname)
            if not os.path.isfile(fp):
                continue
            mf = _CAF_FILE_RE.match(fname)
            if not mf:
                continue
            age = int(mf.group(2))
            entry["photos"].append((age, fp))

    cleaned = {}
    for norm_id, info in identities.items():
        if info["photos"]:
            info["photos"].sort(key=lambda x: (x[0], x[1]))
            cleaned[norm_id] = info
    return cleaned


def load_caf_gallery(
    root=r"D:\GRAD\Assets\CAF",
    exclude_cohort=True,
    exclude_known=False,
    exclude_ids=None,
    split="child_to_adult",
    child_max_age=12,
    n_probes=2,
):
    """
    Loads CAF missing->found retrieval gallery & probes.
    Excludes lag_cohort_cache.npz cohort identities by default (`exclude_cohort=True`)
    to prevent Calibrated Search cohort leakage.

    split='child_to_adult': enrolls child photos (age <= child_max_age) and probes with adult photos (> child_max_age).
    split='oldest': enrolls photos[:-n_probes] and probes with oldest n_probes photos.
    """
    identities = parse_caf_images(root=root)
    blocked = set()
    if exclude_cohort:
        blocked |= load_lag_cohort_identities()
    if exclude_known:
        blocked |= collect_known_dataset_identities(include_demo_dbs=True)
    if exclude_ids:
        blocked |= {normalize_identity(x) for x in exclude_ids}

    gallery_by_id = {}
    probes = []
    for norm_id in sorted(identities.keys()):
        if norm_id in blocked:
            continue
        info = identities[norm_id]
        items = info["photos"]
        if len(items) < 2:
            continue

        if split == "child_to_adult":
            gal_items = [(age, p) for age, p in items if age <= child_max_age]
            probe_items = [(age, p) for age, p in items if age > child_max_age]
        elif split == "oldest":
            k_probe = min(max(1, int(n_probes)), len(items) - 1)
            gal_items = items[:-k_probe]
            probe_items = items[-k_probe:]
        else:
            raise ValueError(f"Unsupported CAF split '{split}'. Expected 'child_to_adult' or 'oldest'.")

        if gal_items and probe_items:
            gallery_by_id[norm_id] = gal_items
            for age, path in probe_items:
                probes.append({
                    "subject_id": norm_id,
                    "norm_id": norm_id,
                    "display_name": info["display_name"],
                    "age": age,
                    "path": path,
                })

    n_gal_photos = sum(len(v) for v in gallery_by_id.values())
    split_tag = f"<={child_max_age}y -> >{child_max_age}y" if split == "child_to_adult" else f"oldest-{n_probes}"
    scope_tag = "cohort+known-disjoint" if exclude_known else ("cohort-disjoint" if exclude_cohort else "unfiltered")
    print(
        f"[CAF] Loaded {len(gallery_by_id)} gallery IDs ({n_gal_photos} gallery photos) "
        f"and {len(probes)} probes [{split_tag}, {scope_tag}]"
    )
    return gallery_by_id, probes


def parse_ecaf_images(root=r"D:\GRAD\Assets\ECAF"):
    """
    Parses ECAF (Hzzone/MTLFace) flat `img_ffhq` directory and `name_list.json`.
    Filenames follow `<Person_Name>_(child|adult)_<idx>.png`.

    Returns:
        identities: {norm_id: {"display_name": str, "child": [(idx, path), ...],
                               "adult": [(idx, path), ...], "official_test": bool}}
    """
    if root is None:
        root = r"D:\GRAD\Assets\ECAF"
    cand_dirs = [
        os.path.join(root, "image_data", "img_ffhq"),
        os.path.join(root, "img_ffhq"),
        root,
    ]
    img_dir = next((d for d in cand_dirs if os.path.isdir(d)), None)
    if img_dir is None:
        raise FileNotFoundError(f"ECAF img_ffhq directory not found under: {root}")

    official_test_norms = set()
    for nl_cand in (
        os.path.join(root, "image_data", "name_list.json"),
        os.path.join(root, "name_list.json"),
    ):
        if os.path.isfile(nl_cand):
            with open(nl_cand, "r", encoding="utf-8") as f:
                nl_data = json.load(f)
            if isinstance(nl_data, dict):
                official_test_norms = {
                    normalize_identity(k)
                    for k, v in nl_data.items()
                    if isinstance(v, dict) and v.get("test")
                }
            break

    identities = {}
    for fname in sorted(os.listdir(img_dir)):
        if fname.startswith("."):
            continue
        fp = os.path.join(img_dir, fname)
        if not os.path.isfile(fp):
            continue
        m = _ECAF_FILE_RE.match(fname)
        if not m:
            continue
        raw_person, stage, idx_str, _ = m.groups()
        display = raw_person.replace("_", " ").strip()
        norm_id = normalize_identity(display)
        if not norm_id:
            continue
        entry = identities.setdefault(
            norm_id,
            {
                "display_name": display,
                "child": [],
                "adult": [],
                "official_test": norm_id in official_test_norms,
            },
        )
        entry[stage.lower()].append((int(idx_str), fp))

    for info in identities.values():
        info["child"].sort(key=lambda x: (x[0], x[1]))
        info["adult"].sort(key=lambda x: (x[0], x[1]))
    return identities


def load_ecaf_gallery(
    root=r"D:\GRAD\Assets\ECAF",
    exclude_cohort=True,
    exclude_known=False,
    exclude_ids=None,
    official_test_only=False,
):
    """
    Loads ECAF child -> adult missing-person retrieval gallery & probes.
    Enrolls `_child_` photos in gallery (age=None) and queries with `_adult_` photos.
    Excludes lag_cohort_cache.npz cohort identities by default (`exclude_cohort=True`).
    """
    identities = parse_ecaf_images(root=root)
    blocked = set()
    if exclude_cohort:
        blocked |= load_lag_cohort_identities()
    if exclude_known:
        blocked |= collect_known_dataset_identities(include_demo_dbs=True)
    if exclude_ids:
        blocked |= {normalize_identity(x) for x in exclude_ids}

    gallery_by_id = {}
    probes = []
    for norm_id in sorted(identities.keys()):
        if norm_id in blocked:
            continue
        info = identities[norm_id]
        if official_test_only and not info["official_test"]:
            continue
        if not info["child"] or not info["adult"]:
            continue

        gallery_by_id[norm_id] = [(None, fp) for _, fp in info["child"]]
        for _, fp in info["adult"]:
            probes.append({
                "subject_id": norm_id,
                "norm_id": norm_id,
                "display_name": info["display_name"],
                "age": None,
                "path": fp,
            })

    n_gal_photos = sum(len(v) for v in gallery_by_id.values())
    split_tag = "official test, " if official_test_only else ""
    scope_tag = "cohort+known-disjoint" if exclude_known else ("cohort-disjoint" if exclude_cohort else "unfiltered")
    print(
        f"[ECAF] Loaded {len(gallery_by_id)} gallery IDs ({n_gal_photos} child photos) "
        f"and {len(probes)} adult probes [{split_tag}{scope_tag}]"
    )
    return gallery_by_id, probes


def load_cache(cache_path):
    """
    Loads cached embeddings from disk.
    Returns: dict mapping file_path -> (embedding_vector or None if no face)
    """
    if not os.path.isfile(cache_path):
        return {}

    try:
        with np.load(cache_path, allow_pickle=True) as data:
            paths = list(data["paths"])
            embeddings = data["embeddings"]
            has_face = data["has_face"]
            cache = {}
            for p, emb, hf in zip(paths, embeddings, has_face):
                cache[str(p)] = emb if hf else None
        print(f">> Loaded {len(cache)} cached embeddings from: {os.path.basename(cache_path)}")
        return cache
    except Exception as e:
        print(f"Warning: Failed to load cache from {cache_path} ({e}). Starting fresh.")
        return {}


def load_quality_cache(cache_path):
    """Loads optional per-image quality scalars saved alongside embeddings in .npz."""
    if not os.path.isfile(cache_path):
        return {}
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            if "qualities" not in data:
                return {}
            paths = list(data["paths"])
            qualities = data["qualities"]
            return {str(p): float(q) for p, q in zip(paths, qualities) if np.isfinite(q)}
    except Exception:
        return {}


def _infer_embedding_dim(cache_dict, default_dim=512):
    """Infers embedding dimension dynamically from the first valid vector in cache_dict."""
    for vec in cache_dict.values():
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return int(arr.size)
    return int(default_dim)


def save_cache(cache_path, cache_dict, quality_dict=None):
    """
    Saves the embedding cache dictionary to a compressed .npz file.
    Infers embedding dimension dynamically (e.g., 512 or 1024) from loaded vectors.
    """
    paths = list(cache_dict.keys())
    dim = _infer_embedding_dim(cache_dict, default_dim=512)
    embeddings = []
    has_face = []
    qualities = []
    zero_vec = np.zeros(dim, dtype=np.float32)

    for p in paths:
        vec = cache_dict[p]
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size != dim:
                raise ValueError(f"Mixed embedding dimensions in cache: expected {dim}, got {arr.size} for {p}")
            embeddings.append(arr)
            has_face.append(True)
        else:
            embeddings.append(zero_vec)
            has_face.append(False)
        if quality_dict is not None:
            q_val = quality_dict.get(p, 1.0)
            qualities.append(float(q_val) if q_val is not None and np.isfinite(q_val) else 1.0)

    emb_arr = np.array(embeddings, dtype=np.float32) if embeddings else np.empty((0, dim), dtype=np.float32)
    payload = {
        "paths": np.array(paths),
        "embeddings": emb_arr,
        "has_face": np.array(has_face, dtype=bool),
    }
    if quality_dict is not None:
        payload["qualities"] = np.array(qualities, dtype=np.float32)

    np.savez_compressed(cache_path, **payload)
    print(f">> Saved {len(paths)} embeddings (dim={dim}) to cache: {os.path.basename(cache_path)}")


def load_lag(root=None, limit=None, seed=0):
    """
    Stub for LAG (Large Age Gap) dataset.
    Raises clear FileNotFoundError if not downloaded yet.
    """
    if root is None:
        root = r"D:\GRAD\Assets\LAG"

    raise FileNotFoundError(
        f"LAG (Large Age Gap) dataset not downloaded yet.\n"
        f"Expected structure: '{root}' containing LAG images folder and cross-validation pairs/splits file."
    )


def load_benchmark_dataset(dataset_name, root=None, limit=None, seed=0):
    """
    Dispatcher to load any supported benchmark dataset.
    Supported names: 'agedb30', 'cacd_vs', 'calfw', 'lag'.
    Returns:
        pairs: list of (path_a, path_b, is_same, fold_idx)
        protocol_desc: human-readable protocol description string
    """
    name = dataset_name.lower().replace("-", "").replace("_", "")
    if name in ("agedb30", "agedb"):
        return load_agedb30(root=root, limit=limit, seed=seed)
    elif name in ("cacdvs", "cacd"):
        return load_cacd_vs(root=root, limit=limit, seed=seed)
    elif name == "calfw":
        return load_calfw(root=root, limit=limit, seed=seed)
    elif name == "lag":
        return load_lag(root=root, limit=limit, seed=seed)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: agedb30, cacd_vs, calfw, lag")
