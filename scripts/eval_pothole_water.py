"""
Which water detector should decide for potholes? Scored on hand labels.

Reads data/pothole_water_labels.csv (scripts/label_pothole_water.py) and scores,
on exactly the labelled pothole outlines:

    ensemble     the served cue ensemble (6-cue logistic with CLIPSeg), > 0.5
    segmenter    the learned water segmenter's share of the pothole, at several
                 coverage thresholds
    clipseg      the CLIPSeg cue alone, > 0.5

The coverage threshold is chosen by cross-fitting over labelled PHOTOS (chosen on
half, scored on the other half, swapped), so the reported number was not tuned on
itself. Missed water is the safety error and is reported first.

Decision rule written down before any labels exist: switch
water_config.WATER_SEGMENTER_DECIDES on only if the segmenter's cross-fitted
missed-water count is lower than the ensemble's without more than doubling its
false alarms.

Usage:
    python scripts/eval_pothole_water.py [labels.csv]
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import semantic_water                                   # noqa: E402
import water_segmenter                                  # noqa: E402
from scripts.annotate_pack import mask_from_polygon     # noqa: E402
from segmentation import get_all_masks                  # noqa: E402
from water_detection import detect_water                # noqa: E402

LABELS_CSV = os.path.join(PROJECT_DIR, "data", "pothole_water_labels.csv")
OUT = os.path.join(PROJECT_DIR, "ml_results", "pothole_water_eval.json")
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def scores(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    return {"missed_water": fn, "false_alarms": fp,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan")}


def main():
    with open(sys.argv[1] if len(sys.argv) > 1 else LABELS_CSV, encoding="utf-8") as f:
        labels = [r for r in csv.DictReader(f) if r["label"] in ("water", "dry")]
    rows = []
    cache = {}
    for r in labels:
        path = os.path.join(PROJECT_DIR, r["image"])
        if path not in cache:
            bgr = cv2.imread(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cache = {path: (bgr, rgb, get_all_masks(path), semantic_water.water_prior_map(rgb),
                            water_segmenter.water_map(bgr))}
        bgr, rgb, masks, prior, seg = cache[path]
        # The frozen outline the labellers saw (merge_annotations.py); older files fall back.
        m = (mask_from_polygon(r["polygon"], *bgr.shape[:2]) if r.get("polygon")
             else masks[int(r["pothole_idx"])])
        w = detect_water(rgb, m, water_prior_map=prior, water_segmentation_map=seg)
        rows.append({"image": r["image"], "source": r["source"], "water": r["label"] == "water",
                     "ensemble": w["ensemble_probability"], "coverage": w["segmenter_coverage"],
                     "clipseg": w["semantic_score"]})

    y = np.array([r["water"] for r in rows])
    print(f"{len(rows)} labelled potholes: {int(y.sum())} water, {int((~y).sum())} dry")
    res = {"n": len(rows), "n_water": int(y.sum()),
           "ensemble": scores(y, [r["ensemble"] > 0.5 for r in rows]),
           "clipseg": scores(y, [(r["clipseg"] or 0) > 0.5 for r in rows]),
           "segmenter_by_threshold": {str(t): scores(y, [r["coverage"] > t for r in rows]) for t in THRESHOLDS}}

    # Cross-fitted threshold: minimise the project's policy cost, a missed water
    # pothole counting three times a false alarm.
    photos = sorted({r["image"] for r in rows})
    half = set(np.random.default_rng(20260929).permutation(photos)[: len(photos) // 2])
    in_a = np.array([r["image"] in half for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    pred = np.zeros(len(rows), bool)
    chosen = []
    for fit, score in ((in_a, ~in_a), (~in_a, in_a)):
        best = min(THRESHOLDS, key=lambda t: (3 * ((cov[fit] <= t) & y[fit]).sum() + ((cov[fit] > t) & ~y[fit]).sum()))
        chosen.append(best)
        pred[score] = cov[score] > best
    res["segmenter_crossfit"] = {**scores(y, pred), "thresholds_chosen": chosen}

    e, s = res["ensemble"], res["segmenter_crossfit"]
    switch = s["missed_water"] < e["missed_water"] and s["false_alarms"] <= 2 * max(e["false_alarms"], 1)
    res["decision"] = "segmenter decides" if switch else "ensemble keeps deciding"

    print(f"\n{'detector':34s} {'missed water':>12s} {'false alarms':>12s} {'precision':>9s} {'recall':>7s}")
    for name, r in [("cue ensemble (served)", e), ("CLIPSeg alone", res["clipseg"])] + \
            [(f"segmenter, coverage > {t}", res["segmenter_by_threshold"][str(t)]) for t in THRESHOLDS] + \
            [(f"segmenter, cross-fitted {chosen}", s)]:
        print(f"{name:34s} {r['missed_water']:12d} {r['false_alarms']:12d} {r['precision']:9.3f} {r['recall']:7.3f}")
    # The threshold to SERVE is fitted on every label with the same cost. Its own scores
    # are in-sample; the cross-fitted row above is the honest estimate of how it will do.
    served = min(THRESHOLDS, key=lambda t: (3 * ((cov <= t) & y).sum() + ((cov > t) & ~y).sum()))
    res["served_threshold"] = served
    src = np.array([r["source"] for r in rows])
    ens = np.array([r["ensemble"] > 0.5 for r in rows])
    res["by_source"] = {}
    print(f"\nBy source, ensemble vs segmenter at the served threshold ({served}):  missed water / false alarms")
    for name in sorted(set(src)):
        k = src == name
        a, b = scores(y[k], ens[k]), scores(y[k], cov[k] > served)
        res["by_source"][name] = {"n": int(k.sum()), "n_water": int(y[k].sum()), "ensemble": a, "segmenter": b}
        print(f"  {name:10s} {int(k.sum()):3d} potholes, {int(y[k].sum()):2d} water   "
              f"ensemble {a['missed_water']:2d} / {a['false_alarms']:2d}   "
              f"segmenter {b['missed_water']:2d} / {b['false_alarms']:2d}")

    print(f"\nDecision (rule fixed in advance): {res['decision']}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {os.path.relpath(OUT, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
