"""
One index of the photos that segmenter v3 may learn from, with splits fixed before training.

Segmenter v2 learned from the three sources that carry true outlines (Kaggle, Pothole-600,
PothRGBD: archive/yolo_seg_v2). Two more sources carry potholes marked only by a BOX, and
one of them is by far the largest and most varied thing on disk:

    rdd       RDD2022 as downloaded (rdd_temp/): 38,385 annotated dashcam and motorbike
              frames from six countries plus a drone set. Class D40 is the pothole class.
    mendeley  713 UK phone photos, 1,156 pothole boxes.

This script decides, once, which of those frames are used and how:

    box_train    frames with D40 / pothole boxes, to be turned into outlines by SAM 2
                 (scripts/boxes_to_polygons.py) and added to training
    box_test     the same, for ONE RDD country held out entirely
    negative     RDD frames with no pothole box: half with other damage (cracks), half clean.
                 They teach the segmenter what is not a pothole.
    neg_test     negatives from the held-out country
    dropped      a perceptual-hash near-duplicate of a photo already in the v2 sets

Rules fixed here, before any result:
  - the held-out country is the one with the second-most pothole frames among the dashcam
    countries: large enough to measure on, and the largest stays in training;
  - the drone set is left out (a view the product never sees);
  - a box frame or negative within Hamming distance 6 of ANY v2 image (train, val or test)
    is dropped: either it is already in, or it would leak into an evaluation set.

Output: archive/corpus_v3/manifest.csv, rdd_annotations.csv (cache of the parsed XML).

Usage:
    python scripts/build_corpus_manifest.py [--negatives 3000]
"""
import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from scripts.build_provenance_index import hamming_matrix, phash          # noqa: E402

OUT = os.path.join(PROJECT_DIR, "archive", "corpus_v3")
V2 = os.path.join(PROJECT_DIR, "archive", "yolo_seg_v2")
RDD = os.path.join(PROJECT_DIR, "rdd_temp")
MENDELEY = os.path.join(PROJECT_DIR, "mendeley_water_filled_dataset",
                        "An Annotated Water-Filled, and Dry Potholes Dataset for Deep Learning Applications")
DASHCAM = ("Japan", "India", "Czech", "Norway", "United_States", "China_MotorBike")
POTHOLE_CLASSES = {"D40", "pothole"}
NEAR_DUP = 6
SEED = 20261003


def parse_voc(xml_path):
    """(width, height, pothole boxes, number of other objects) from a Pascal VOC file."""
    r = ET.parse(xml_path).getroot()
    s = r.find("size")
    boxes, other = [], 0
    for o in r.findall("object"):
        if o.find("name").text in POTHOLE_CLASSES:
            b = o.find("bndbox")
            boxes.append([round(float(b.find(k).text), 1) for k in ("xmin", "ymin", "xmax", "ymax")])
        else:
            other += 1
    return int(float(s.find("width").text)), int(float(s.find("height").text)), boxes, other


def box_rows():
    """Every RDD and Mendeley frame, from the annotation files (cached: 39,000 small files)."""
    cache = os.path.join(OUT, "rdd_annotations.csv")
    if os.path.isfile(cache):
        return pd.read_csv(cache)
    rows = []
    for c in DASHCAM:
        for x in sorted(glob.glob(os.path.join(RDD, c, c, "train", "annotations", "xmls", "*.xml"))):
            w, h, boxes, other = parse_voc(x)
            name = os.path.splitext(os.path.basename(x))[0]
            rows.append({"source": "rdd", "country": c, "name": name, "w": w, "h": h, "boxes": json.dumps(boxes),
                         "n_pothole": len(boxes), "n_other": other,
                         "path": os.path.relpath(os.path.join(RDD, c, c, "train", "images", name + ".jpg"), PROJECT_DIR).replace("\\", "/")})
        print(f"  parsed rdd {c}", flush=True)
    for x in sorted(glob.glob(os.path.join(MENDELEY, "XML", "*.xml"))):
        w, h, boxes, other = parse_voc(x)
        name = os.path.splitext(os.path.basename(x))[0]
        rows.append({"source": "mendeley", "country": "United_Kingdom", "name": name, "w": w, "h": h, "boxes": json.dumps(boxes),
                     "n_pothole": len(boxes), "n_other": other,
                     "path": os.path.relpath(os.path.join(MENDELEY, "IMG", name + ".jpg"), PROJECT_DIR).replace("\\", "/")})
    d = pd.DataFrame(rows)
    d.to_csv(cache, index=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", type=int, default=3000, help="RDD frames without a pothole box, for training")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = box_rows()
    d = d[[os.path.isfile(os.path.join(PROJECT_DIR, p)) for p in d.path]].reset_index(drop=True)

    rdd = d[d.source == "rdd"]
    per = rdd[rdd.n_pothole > 0].groupby("country").size().sort_values(ascending=False)
    held = per.index[1]
    print("\nRDD frames with a pothole box, by country:", per.to_dict())
    print(f"held-out country (second-most pothole frames): {held}")

    d["role"] = ""
    d.loc[(d.n_pothole > 0), "role"] = "box_train"
    d.loc[(d.n_pothole > 0) & (d.country == held), "role"] = "box_test"
    # Negatives: evenly across the training countries, half crack-only, half clean.
    train_c = [c for c in DASHCAM if c != held]
    share = args.negatives // (2 * len(train_c))
    for c in train_c:
        for kind in (d.n_other > 0, d.n_other == 0):
            pool = d.index[(d.source == "rdd") & (d.country == c) & (d.n_pothole == 0) & kind]
            d.loc[rng.choice(pool, min(share, len(pool)), replace=False), "role"] = "negative"
    pool = d.index[(d.source == "rdd") & (d.country == held) & (d.n_pothole == 0)]
    d.loc[rng.choice(pool, min(600, len(pool)), replace=False), "role"] = "neg_test"

    use = d[d.role != ""].copy()
    print(f"\nhashing {len(use)} selected frames and the v2 sets ...", flush=True)
    use["phash"] = [phash(os.path.join(PROJECT_DIR, p)) for p in use.path]
    v2 = []
    for split in ("train", "val", "test_kaggle", "test_p600", "test_pothrgbd"):
        for p in sorted(glob.glob(os.path.join(V2, split, "images", "*"))):
            v2.append({"split": split, "path": os.path.relpath(p, PROJECT_DIR).replace("\\", "/"), "phash": phash(p)})
    v2 = pd.DataFrame(v2)
    use = use[use.phash >= 0]
    dist = np.vstack([hamming_matrix(use.phash.values[i:i + 400].astype(np.uint64), v2.phash.values.astype(np.uint64))
                      for i in range(0, len(use), 400)])
    near = dist.min(1) <= NEAR_DUP
    use["dup_of"] = np.where(near, v2.path.values[dist.argmin(1)], "")
    use["dup_split"] = np.where(near, v2.split.values[dist.argmin(1)], "")
    use.loc[near, "role"] = "dropped"

    use.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print("\nframes by source and role:")
    print(use.groupby(["source", "role"]).size().unstack(fill_value=0).to_string())
    print("\nby country:")
    print(use[use.source == "rdd"].groupby(["country", "role"]).size().unstack(fill_value=0).to_string())
    print(f"\ndropped as near-duplicates of a v2 image: {int(near.sum())}  ({use[near].groupby(['source', 'dup_split']).size().to_dict()})")
    print(f"pothole boxes to convert: train {int(use[use.role == 'box_train'].n_pothole.sum())}, held-out {int(use[use.role == 'box_test'].n_pothole.sum())}")
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'manifest.csv'), PROJECT_DIR)}")


if __name__ == "__main__":
    main()
