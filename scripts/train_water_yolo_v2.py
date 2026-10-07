"""
Retrain the water segmenter with water IN POTHOLES, and find out honestly whether it helps.

The served water segmenter (scripts/train_water_yolo.py) learned from HanYang: puddles on
Korean city streets. On the 142 potholes the team labelled wet or dry it misses 11 of 53
wet ones and raises 15 false alarms, several on dry brown dirt. Those labels can now teach
it, but they are also the only test there is, so the test is cross-fitted: the labelled
photos are cut into three folds, a model is trained with two folds added to HanYang and
scored on the third, and every pothole is scored by a model that never saw its photo.

What a labelled photo adds to training (the team labelled wet / dry, not outlines):

    every labelled pothole dry     the photo with an empty label: "nothing here is water"
    a wet pothole the served       the served model's own water regions in that photo
    model already covers >= 30%
    a wet pothole it covers less   the pothole's outline stands in for the water outline.
                                   Crude (water rarely fills the whole pothole) but it is
                                   the only signal available for exactly the cases the
                                   served model gets wrong.

Added photos are listed four times: they are 5% of HanYang otherwise.

Rule, fixed beforehand: adopt the retrained model only if, cross-fitted at the served
threshold (water covers > 10% of the pothole), it misses fewer wet potholes than the served
model without raising more false alarms. Only then is the final model (all labels added) trained
and written to yolo-segmentation/model/water_best_v2.pt; installing it is a separate step.

Usage:
    python scripts/train_water_yolo_v2.py [--epochs 100] [--final]
"""
import argparse
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import water_config                                              # noqa: E402
from scripts.annotate_pack import mask_from_polygon              # noqa: E402
from scripts.train_water_yolo import RUNS, SEED, WORK, write_splits   # noqa: E402
from segmentation import _extract_binary_masks                   # noqa: E402

LABELS_CSV = os.path.join(PROJECT_DIR, "data", "pothole_water_labels.csv")
V2 = os.path.join(PROJECT_DIR, "archive", "water_seg_v2")
OUT_JSON = os.path.join(PROJECT_DIR, "ml_results", "pothole_water_retrain.json")
OUT_WEIGHTS = os.path.join(PROJECT_DIR, "yolo-segmentation", "model", "water_best_v2.pt")
FOLDS, REPEAT, CONFIDENT = 3, 4, 0.30
THRESHOLD = water_config.WATER_COVERAGE_THRESHOLD


def water_map(model, bgr):
    out = np.zeros(bgr.shape[:2], bool)
    for m in _extract_binary_masks(bgr, model, conf_threshold=water_config.WATER_SEGMENTER_CONF, min_area=50):
        out |= m > 0
    return out


def polygons(mask):
    h, w = mask.shape
    lines = []
    for c in cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) >= 50:
            pts = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2) / [w, h]
            if len(pts) >= 3:
                lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in np.clip(pts, 0, 1)))
    return lines


def photo_label(bgr, rows, served):
    """YOLO label lines for one labelled photo (possibly none: an all-dry photo)."""
    if all(r["label"] == "dry" for r in rows):
        return []
    seen = water_map(served, bgr)
    keep = np.zeros(bgr.shape[:2], bool)
    for r in rows:
        if r["label"] != "water":
            continue
        m = mask_from_polygon(r["polygon"], *bgr.shape[:2]) > 0
        keep |= seen if (seen & m).sum() / max(m.sum(), 1) >= CONFIDENT else m
    return polygons(keep)


def train(name, extra, epochs, batch, base_train):
    from ultralytics import YOLO
    d = os.path.join(V2, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(d, "extra", sub))
    listed = []
    for i, (bgr, lines) in enumerate(extra):
        p = os.path.join(d, "extra", "images", f"lab_{i:04d}.jpg")
        cv2.imwrite(p, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(d, "extra", "labels", f"lab_{i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        listed.append(p.replace("\\", "/"))
    with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(base_train + listed * REPEAT) + "\n")
    yaml = os.path.join(d, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"train: {d.replace(chr(92), '/')}/train.txt\nval: {WORK.replace(chr(92), '/')}/val.txt\nnames:\n  0: puddle\n")
    YOLO("yolov8n-seg.pt").train(data=yaml, epochs=epochs, patience=20, imgsz=640, batch=batch, seed=SEED,
                                 deterministic=True, workers=4, project=RUNS, name=f"water_v2_{name}", exist_ok=True, plots=False)
    return os.path.join(RUNS, f"water_v2_{name}", "weights", "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--final", action="store_true", help="also train on all labels if the rule passes")
    args = ap.parse_args()
    from ultralytics import YOLO

    with open(LABELS_CSV, encoding="utf-8") as f:
        labels = [r for r in csv.DictReader(f) if r["label"] in ("water", "dry") and r.get("polygon")]
    photos = {}
    for r in labels:
        photos.setdefault(r["image"], []).append(r)
    names = sorted(photos)
    fold_of = {n: i % FOLDS for i, n in enumerate(np.random.default_rng(SEED).permutation(names))}
    write_splits()
    with open(os.path.join(WORK, "train.txt"), encoding="utf-8") as f:
        base_train = [ln.strip() for ln in f if ln.strip()]

    served = YOLO(water_config.WATER_SEGMENTER_WEIGHTS)
    images = {n: cv2.imread(os.path.join(PROJECT_DIR, n)) for n in names}
    added = {n: photo_label(images[n], photos[n], served) for n in names}
    print(f"{len(labels)} labelled potholes in {len(names)} photos: {sum(1 for v in added.values() if not v)} all-dry photos, "
          f"{sum(1 for v in added.values() if v)} with water outlines", flush=True)

    rows = []
    for k in range(FOLDS):
        best = train(f"fold{k}", [(images[n], added[n]) for n in names if fold_of[n] != k], args.epochs, args.batch, base_train)
        model = YOLO(best)
        for n in names:
            if fold_of[n] != k:
                continue
            new, old = water_map(model, images[n]), water_map(served, images[n])
            for r in photos[n]:
                m = mask_from_polygon(r["polygon"], *images[n].shape[:2]) > 0
                rows.append({"image": n, "source": r["source"], "water": r["label"] == "water", "fold": k,
                             "served": float((old & m).sum() / max(m.sum(), 1)), "retrained": float((new & m).sum() / max(m.sum(), 1))})
        print(f"fold {k} scored", flush=True)

    y = np.array([r["water"] for r in rows])
    res = {}
    print(f"\n{len(rows)} potholes, {int(y.sum())} wet; water if covering > {THRESHOLD:.0%} of the pothole")
    print(f"{'model':28s} {'missed water':>13s} {'false alarms':>13s}")
    for name in ("served", "retrained"):
        pred = np.array([r[name] for r in rows]) > THRESHOLD
        res[name] = {"missed_water": int((~pred & y).sum()), "false_alarms": int((pred & ~y).sum())}
        print(f"{name + (' (cross-fitted)' if name == 'retrained' else ''):28s} {res[name]['missed_water']:13d} {res[name]['false_alarms']:13d}")
    # Compared with the served model as scored HERE, on the same potholes by the same code
    # (scripts/eval_pothole_water.py reported 11 and 15 through the full water pipeline).
    r, s = res["retrained"], res["served"]
    ok = r["missed_water"] < s["missed_water"] and r["false_alarms"] <= s["false_alarms"]
    res["adopt"] = bool(ok)
    res["rule"] = f"missed < {s['missed_water']} and false alarms <= {s['false_alarms']}"
    print(f"rule ({res['rule']}): {'PASSED' if ok else 'not met'}")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({**res, "potholes": rows}, f, indent=2)
    if ok and args.final:
        best = train("final", [(images[n], added[n]) for n in names], args.epochs, args.batch, base_train)
        shutil.copy2(best, OUT_WEIGHTS)
        print(f"final model -> {os.path.relpath(OUT_WEIGHTS, PROJECT_DIR)} (not installed: point water_config at it to serve)")
    print(f"wrote {os.path.relpath(OUT_JSON, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
