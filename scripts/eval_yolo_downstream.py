"""
Does a better segmenter give better DEPTH answers? Old vs new YOLO, end to end.

On the PothRGBD capture sessions held out from segmenter training
(archive/yolo_seg_v2/test_pothrgbd_keys.txt), for every RealSense-measured pothole:

    found      the segmenter produced a mask overlapping it (IoU >= 0.1, the rule the
               metric trainset uses) — an unfound pothole gets no estimate at all
    MAE        depth error of the metric model on the found ones, in mm
    correct    measured band reproduced by the served cut-offs, counted over ALL measured
               potholes (a missed pothole counts as not correct)
    under      measured band reported milder, over all measured potholes (missed ones
               count as under-reported: the user is told nothing)

The depth regressor is refitted on every row EXCEPT these sessions (same config as the
served model), so nothing here is scored by a model that saw its frame.

Each argument is a weights path, optionally '@conf' (default 0.25).

Usage:
    python scripts/eval_yolo_downstream.py yolo-segmentation/model/best.pt yolo-segmentation/model/best_v2.pt@0.35
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import metric_severity                                                          # noqa: E402
from metric_features import extract_metric_features, shape_features             # noqa: E402
from scripts.build_metric_trainset import iou, moge_for                         # noqa: E402
from scripts.pothrgbd_metric_labels import DATA_DIR, load_polygons, timestamp_key  # noqa: E402
from scripts.train_metric_regressor import models                               # noqa: E402
from segmentation import _extract_binary_masks                                  # noqa: E402

KEYS = os.path.join(PROJECT_DIR, "archive", "yolo_seg_v2", "test_pothrgbd_keys.txt")
TRAINSET = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_metric.csv")
BANDS = {"Shallow": 0, "Moderate": 1, "Deep": 2}


def main():
    weights = sys.argv[1:]
    test_keys = set(open(KEYS, encoding="utf-8").read().split())
    d = pd.read_csv(TRAINSET)
    _m, meta = metric_severity._load()
    cols = meta["features"]
    config = meta["config"]                      # e.g. metric_shape/HGB_L2/train=gt+yolo+sam2
    mname = config.split("/")[1]
    train_on = config.split("=")[1].split("+")
    tr = d[~d.key.isin(test_keys) & d.mask_source.isin(train_on)]
    reg = models()[mname]()
    reg.fit(tr[cols].values, tr.gt_mm.values)
    gt_rows = d[d.key.isin(test_keys) & (d.mask_source == "gt")]
    print(f"{len(gt_rows)} measured potholes in {gt_rows.key.nunique()} held-out frames; "
          f"depth model {mname} refitted on {tr.key.nunique()} other frames")

    imgs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}
    import moge_backend
    results = {}
    for arg in weights:
        wpath, _, conf = arg.partition("@")
        conf = float(conf) if conf else 0.25
        yolo = YOLO(wpath)
        rows = []
        for key, grp in gt_rows.groupby("key"):
            bgr = cv2.imread(imgs[key])
            h, w = np.load(glob.glob(os.path.join(DATA_DIR, "depths", f"{key}*.npy"))[0]).shape[:2]
            if bgr.shape[:2] != (h, w):
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            polys = load_polygons(labs[key], h, w)
            preds = _extract_binary_masks(bgr, yolo, conf_threshold=conf)
            g = moge_for(key, bgr, moge_backend)
            for _, r in grp.iterrows():
                gm = polys[int(r.pothole) - 1]
                ious = [iou(p, gm) for p in preds]
                best = int(np.argmax(ious)) if ious else -1
                rec = {"key": key, "pothole": r.pothole, "gt_mm": r.gt_mm, "gt_band": BANDS[r.gt_severity],
                       "iou": ious[best] if ious else 0.0, "pred_mm": np.nan}
                if ious and ious[best] >= 0.1:
                    mf = extract_metric_features(preds[best], g["depth"], g["intrinsics"],
                                                 normal=g["normal"], valid=g["mask"])
                    sf = shape_features(preds[best])
                    if mf and sf:
                        f = {**mf, **sf}
                        rec["pred_mm"] = float(np.clip(reg.predict(np.array([[f[c] for c in cols]]))[0], 0, 250))
                rows.append(rec)
        o = pd.DataFrame(rows)
        found = o.pred_mm.notna()
        band = np.full(len(o), -1)
        band[found.values] = [BANDS[metric_severity.decide(v, meta)] for v in o.pred_mm[found]]
        res = {
            "found": float(found.mean()),
            "mean_iou_found": float(o.iou[found].mean()),
            "mae_found": float(np.abs(o.pred_mm[found] - o.gt_mm[found]).mean()),
            "correct_of_all": float((band == o.gt_band.values).mean()),
            "under_of_all": int(((band < o.gt_band.values)).sum()),
            "deep_found_correct": int(((band == 2) & (o.gt_band.values == 2)).sum()),
            "n_deep": int((o.gt_band == 2).sum()),
            "n": int(len(o)),
        }
        results[f"{os.path.basename(wpath)}@{conf:g}"] = res

    print(f"\n{'weights':18s} {'found':>6s} {'IoU':>6s} {'MAE mm':>7s} {'correct band (of all)':>22s} "
          f"{'under/missed':>13s} {'Deep called Deep':>17s}")
    for name, r in results.items():
        print(f"{name:18s} {r['found']:6.3f} {r['mean_iou_found']:6.3f} {r['mae_found']:7.2f} "
              f"{r['correct_of_all']:22.3f} {r['under_of_all']:13d} {r['deep_found_correct']:>11d}/{r['n_deep']}")
    out = os.path.join(PROJECT_DIR, "ml_results", "yolo_seg_v2_downstream.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.relpath(out, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
