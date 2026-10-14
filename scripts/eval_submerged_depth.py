"""
How much depth does water hide, and how much of it can the dry wall give back?

1. Controlled submersion on the held-out PothRGBD test potholes. Water is level, so filling a
   pothole up to a level L below the road replaces every deeper point by L. We fill each
   measured pothole to s = 25 / 50 / 75 % of its depth and read it three ways:
     surface    the label protocol on the water surface (what any depth map shows)
     recovered  submerged_depth.recover: a bowl fitted to the dry wall and rim, extended under water
   once with RealSense walls (is the geometry recoverable at all?) and once with the relief
   network's walls (is the network's wall accurate enough?). For the network the water
   surface is modelled as level at the network's own reading along the waterline.
2. The real water-filled potholes: the team-labelled PothRGBD potholes holding water (RealSense
   depth exists), and the labelled Kaggle / Mendeley ones (no depth) for the qualitative figure.

Usage:
    python scripts/eval_submerged_depth.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import relief_depth as R                                                     # noqa: E402
import submerged_depth as S                                                  # noqa: E402
import water_segmenter                                                       # noqa: E402
from metric_features import road_plane_deviation as plane_dev                # noqa: E402
from scripts.pothrgbd_metric_labels import (                                 # noqa: E402
    DATA_DIR, DEPTH_VALID_MIN, load_polygons, road_plane_deviation, timestamp_key,
)
from scripts.train_relief_depth import split_keys                            # noqa: E402

TRAINSET = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_metric.csv")
LABELS = os.path.join(PROJECT_DIR, "data", "pothole_water_labels.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "water_depth")
FILLS = (0.25, 0.5, 0.75)


def bands(mm, t1, t2):
    mm = np.asarray(mm, float)
    return np.where(mm >= t2, 2, np.where(mm >= t1, 1, 0))


def net_relief(bgr, mask):
    """Relief network's map with the road plane re-fitted on the ring, as served."""
    rel = R.predict_relief(bgr)
    res = plane_dev(rel, mask.astype(np.uint8), np.ones(mask.shape, bool))
    return None if res is None else res[0]


def submerge(dev, mask, valid, level):
    """Level water at `level` mm below the road: deeper points are replaced by the surface."""
    w = mask & valid & (dev > level)
    obs = dev.copy()
    obs[w] = level
    return obs, w


def controlled():
    meta = R.meta()
    t1 = meta["decision_thresholds_mm"]["moderate_from"]
    t2 = meta["decision_thresholds_mm"]["deep_from"]
    d = pd.read_csv(TRAINSET)
    test = set(split_keys("test_pothrgbd"))
    gt = d[d.key.isin(test) & (d.mask_source == "gt")]
    imgs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}
    rows = []
    for key, grp in gt.groupby("key"):
        depth = np.load(glob.glob(os.path.join(DATA_DIR, "depths", f"{key}*.npy"))[0])
        h, w = depth.shape[:2]
        bgr = cv2.imread(imgs[key])
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        polys = load_polygons(labs[key], h, w)
        valid = depth >= DEPTH_VALID_MIN
        for _, r in grp.iterrows():
            m = polys[int(r.pothole) - 1] > 0
            res = road_plane_deviation(depth, m.astype(np.uint8))
            pdv = net_relief(bgr, m)
            if res is None or pdv is None:
                continue
            dev = res[0]
            g = float(np.percentile(dev[m & valid], 90))
            rec = {"key": key, "pothole": int(r.pothole), "gt_mm": g,
                   "net_dry_mm": float(np.percentile(pdv[m], 90))}
            for s in FILLS:
                level = (1 - s) * g
                obs, wmask = submerge(dev, m, valid, level)
                o = S.recover(obs, m, wmask, valid | wmask)
                rec[f"rs_surface_{s}"] = o["surface_mm"]
                rec[f"rs_recovered_{s}"] = o["recovered_mm"]
                rec[f"rs_abstain_{s}"] = o["abstained"]
                # the network sees a level water surface, at its own reading along the waterline
                ring = (cv2.dilate(wmask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~wmask & m
                pobs = pdv.copy()
                if wmask.any():
                    pobs[wmask] = float(np.median(pdv[ring])) if ring.any() else float(np.median(pdv[wmask]))
                o = S.recover(pobs, m, wmask)
                rec[f"net_surface_{s}"] = o["surface_mm"]
                rec[f"net_recovered_{s}"] = o["recovered_mm"]
                rec[f"net_abstain_{s}"] = o["abstained"]
            rows.append(rec)
    t = pd.DataFrame(rows)
    yb = bands(t.gt_mm, 25, 50)
    summary = {"n": int(len(t)), "n_deep": int((yb == 2).sum()), "cutoffs": [t1, t2]}

    def score(p):
        p = np.asarray(p, float)
        b = bands(p, t1, t2)
        return {"mae": float(np.mean(np.abs(p - t.gt_mm))), "bias": float(np.mean(p - t.gt_mm)),
                "under": int((b < yb).sum()), "deep_found": int(((b == 2) & (yb == 2)).sum()),
                "band_acc": float((b == yb).mean())}

    summary["dry"] = {"realsense": score(t.gt_mm), "network": score(t.net_dry_mm)}
    for s in FILLS:
        summary[str(s)] = {
            "realsense_surface": score(t[f"rs_surface_{s}"]),
            "realsense_recovered": score(t[f"rs_recovered_{s}"]),
            "realsense_abstained": int(t[f"rs_abstain_{s}"].isin(["too little visible wall", "visible wall is not bowl-shaped"]).sum()),
            "network_surface": score(t[f"net_surface_{s}"]),
            "network_recovered": score(t[f"net_recovered_{s}"]),
            "network_abstained": int(t[f"net_abstain_{s}"].isin(["too little visible wall", "visible wall is not bowl-shaped"]).sum()),
        }
    return t, summary


def polygon_mask(poly, h, w):
    pts = np.array(json.loads(poly), float) * [w, h]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    return m > 0


def real_water():
    """Team-labelled water-filled potholes: readings, and maps saved for the figure."""
    lab = pd.read_csv(LABELS)
    lab = lab[lab.label == "water"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = []
    for i, r in lab.reset_index(drop=True).iterrows():
        path = r.image if os.path.isabs(r.image) else os.path.join(PROJECT_DIR, r.image)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if max(h, w) > 1280:                       # same working size for every source
            f = 1280 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        m = polygon_mask(r.polygon, h, w)
        if m.sum() < 150:
            continue
        wm = water_segmenter.water_map(bgr)
        water = (wm > 0) & m if wm is not None else np.zeros_like(m)
        pdv = net_relief(bgr, m)
        if pdv is None:
            continue
        o = S.recover(pdv, m, water)
        rec = {"image": r.image, "source": r.source, "pothole_idx": int(r.pothole_idx),
               "water_share": o["water_share"], "surface_mm": o["surface_mm"],
               "recovered_mm": o["recovered_mm"], "abstained": o["abstained"]}
        if r.source == "pothrgbd":
            key = timestamp_key(os.path.basename(r.image))
            dp = glob.glob(os.path.join(DATA_DIR, "depths", f"{key}*.npy"))
            if dp:
                depth = np.load(dp[0])
                dm = cv2.resize(m.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
                dw = cv2.resize(water.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
                res = road_plane_deviation(depth, dm.astype(np.uint8))
                valid = depth >= DEPTH_VALID_MIN
                rec["realsense_valid_in_water"] = float(valid[dw].mean()) if dw.any() else None
                if res is not None and (dm & valid).any():
                    rec["realsense_mm"] = float(np.percentile(res[0][dm & valid], 90))
        np.savez_compressed(os.path.join(OUT_DIR, f"water_{i:02d}.npz"), mask=m, water=water, relief=pdv.astype(np.float32),
                            filled=(o["filled"] if o["filled"] is not None else pdv).astype(np.float32))
        rec["npz"] = f"water_{i:02d}.npz"
        out.append(rec)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t, summary = controlled()
    t.to_csv(os.path.join(OUT_DIR, "controlled_submersion.csv"), index=False)
    print(f"controlled submersion: {summary['n']} test potholes ({summary['n_deep']} Deep); cut-offs {summary['cutoffs']}")
    print(f"  dry network: {summary['dry']['network']}")
    for s in FILLS:
        x = summary[str(s)]
        print(f"  fill {s:.0%}:")
        for k in ("realsense_surface", "realsense_recovered", "network_surface", "network_recovered"):
            v = x[k]
            print(f"    {k:22s} MAE {v['mae']:6.2f}  bias {v['bias']:+6.2f}  under {v['under']:3d}  "
                  f"Deep {v['deep_found']:2d}/{summary['n_deep']}  band acc {v['band_acc']:.1%}")
        print(f"    abstained: RealSense walls {x['realsense_abstained']}, network walls {x['network_abstained']}")
    real = real_water()
    summary["real_water"] = real
    for r in real:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "image"}, r["image"][-40:])
    with open(os.path.join(OUT_DIR, "submerged_depth_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote ml_results/water_depth/submerged_depth_eval.json")


if __name__ == "__main__":
    main()
