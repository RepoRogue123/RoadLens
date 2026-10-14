"""
The potholes the team rates for Methods 2B and 3B, with the measurements the models learn from.

Both learned methods need the same thing: human judgements on potholes from as many kinds of
photo as possible, and for each pothole the seven measurements the pipeline computes for any
photo (hazard.FEATURES). This script draws the pool once, across every dataset with photos,
and runs the full pipeline (scripts/cvpr_severity_framework.py) on each to store them:

    pothrgbd   held-out test sessions only (the depth network trained on the rest)
    kaggle     data1 photos
    pothole600 its test split
    rdd        RDD2022 held-out box frames, spread across countries
    mendeley   UK phone photos (water-filled and dry)

At most two potholes per photo; photos are drawn at random per source until its quota is met.
The outline is frozen as a polygon so the packs show exactly the pothole measured here.

Output: data/hazard_pool.csv (one row per pothole: pool_id, image, pothole_idx, source,
polygon, familiar (photo like the training photos), the seven features). scripts/build_annotation_pack.py --task consequence|hazard_pairs
reads it; scripts/fit_priority_model.py and scripts/fit_hazard_pairs.py join labels to it.

Usage:
    python scripts/build_hazard_pool.py [--per-source 50]
"""
import argparse
import glob
import json
import os
import random
import sys

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import hazard                                                                    # noqa: E402
from scripts.annotate_pack import polygon_from_mask                              # noqa: E402
from scripts.cvpr_severity_framework import extract_pothole_features_cvpr        # noqa: E402
from scripts.label_pothole_water import MENDELEY                                 # noqa: E402
from scripts.pothrgbd_metric_labels import DATA_DIR, timestamp_key               # noqa: E402
from scripts.train_relief_depth import split_keys                                # noqa: E402

POOL_CSV = os.path.join(PROJECT_DIR, "data", "hazard_pool.csv")
SEED = 20261013
PER_PHOTO = 2


def source_paths():
    """Candidate photos per source, each list in a fixed shuffled order."""
    test = set(split_keys("test_pothrgbd"))
    rdd = pd.read_csv(os.path.join(PROJECT_DIR, "archive", "corpus_v3", "manifest.csv"))
    rdd = rdd[(rdd.source == "rdd") & (rdd.role == "box_test") & (rdd.n_pothole > 0)]
    rng = random.Random(SEED)
    # RDD: round-robin across countries so no country dominates
    by_country = {c: g.path.tolist() for c, g in rdd.groupby("country")}
    for v in by_country.values():
        rng.shuffle(v)
    rdd_order = [p for group in zip(*by_country.values()) for p in group]
    out = {
        "pothrgbd": [p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg")) if timestamp_key(p) in test],
        "kaggle": glob.glob(os.path.join(PROJECT_DIR, "data1", "*", "images", "*.jpg")),
        "pothole600": glob.glob(os.path.join(PROJECT_DIR, "pothole600", "testing", "rgb", "*.png")),
        "mendeley": glob.glob(os.path.join(MENDELEY, "*.jpg")),
    }
    for v in out.values():
        v.sort()
        rng.shuffle(v)
    out["rdd"] = [os.path.join(PROJECT_DIR, p) for p in rdd_order]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=50)
    args = ap.parse_args()
    done = pd.read_csv(POOL_CSV) if os.path.isfile(POOL_CSV) else pd.DataFrame()
    rows = done.to_dict("records")
    seen = {(r["image"], int(r["pothole_idx"])) for r in rows}
    for source, paths in source_paths().items():
        n = sum(r["source"] == source for r in rows)
        for p in paths:
            if n >= args.per_source:
                break
            rel = os.path.relpath(p, PROJECT_DIR).replace("\\", "/")
            if any(s[0] == rel for s in seen):
                continue
            data = extract_pothole_features_cvpr(p)
            for k, rec in enumerate((data or {}).get("potholes", [])[:PER_PHOTO]):
                poly = polygon_from_mask(rec["mask"])
                if not rec.get("measured") or poly is None or n >= args.per_source:
                    continue
                rows.append({"pool_id": f"h{len(rows) + 1:04d}", "image": rel, "pothole_idx": k, "source": source,
                             "polygon": json.dumps(poly, separators=(",", ":")), "familiar": rec["familiar"],
                             **{f: rec["features"][f] for f in hazard.FEATURES}})
                seen.add((rel, k))
                n += 1
            pd.DataFrame(rows).to_csv(POOL_CSV, index=False)          # resumable after every photo
        print(f"  {source}: {n} potholes", flush=True)
    print(f"wrote {len(rows)} potholes to {os.path.relpath(POOL_CSV, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
