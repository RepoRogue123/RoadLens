"""
Is the cascade bowl fit (submerged_depth.recover) reliable beyond the 207-pothole benchmark?

A. More measured potholes: the PothRGBD *train* split (745 potholes, RealSense ground truth), filled
   with level water to 25 / 50 / 75 % in simulation, as in scripts/eval_submerged_depth.py.
   RealSense walls are a clean test (the fit has no learned parts; its settings came from val).
   Network walls are optimistic here, because the relief network was trained on these photos.
B. Potholes with no measured depth (Kaggle and Pothole-600 test sets, human outlines). The water
   segmenter separates them:
     dry   the network's own dry reading is the reference; the pothole is filled in simulation
           and read back from its walls. This measures agreement with our depth pipeline, not
           accuracy: how often the recovered reading lands in the same severity band as the dry
           one, and how far from it.
           Also: with no water, recover() must return the dry reading unchanged.
     wet   real water: no reference exists. Reported: how often a bowl is fitted and by how much
           it raises the reading.

    python scripts/eval_bowl_reliability.py cache
    python scripts/eval_bowl_reliability.py run
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2                                                                    # noqa: E402
import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import submerged_depth as S                                                   # noqa: E402

OUT = os.path.join(PROJECT_DIR, "ml_results", "water_depth", "reliability")
CACHE = os.path.join(OUT, "cache")
SPLIT_DIR = os.path.join(PROJECT_DIR, "archive", "yolo_seg_v2")
FILLS = (0.25, 0.5, 0.75)
DRY_MAX, WET_MIN = 0.05, 0.20      # water share of the outline: dry below, wet above
cv2.setNumThreads(1)


def cache():
    import eval_ransac_bowl as E
    E.SPLITS = {"train": "train"}
    E.CACHE = os.path.join(CACHE, "pothrgbd")
    os.makedirs(E.CACHE, exist_ok=True)
    E.cache()                                               # A: train split, same format as before
    import relief_depth as R
    import water_segmenter
    from metric_features import road_plane_deviation as plane_dev
    from scripts.pothrgbd_metric_labels import load_polygons
    for name in ("test_kaggle", "test_p600"):              # B: human outlines, no measured depth
        os.makedirs(os.path.join(CACHE, name), exist_ok=True)
        n = 0
        for img in sorted(glob.glob(os.path.join(SPLIT_DIR, name, "images", "*"))):
            lab = os.path.join(SPLIT_DIR, name, "labels", os.path.splitext(os.path.basename(img))[0] + ".txt")
            bgr = cv2.imread(img)
            if bgr is None or not os.path.exists(lab):
                continue
            h, w = bgr.shape[:2]
            rel = R.predict_relief(bgr)
            wm = water_segmenter.water_map(bgr)
            wm = (wm > 0) if wm is not None else np.zeros((h, w), bool)
            for j, m in enumerate(load_polygons(lab, h, w)):
                m = m > 0
                if m.sum() < 400:
                    continue
                res = plane_dev(rel, m.astype(np.uint8), np.ones(m.shape, bool))
                if res is None:
                    continue
                np.savez_compressed(os.path.join(CACHE, name, f"{os.path.basename(img)}_{j}.npz"),
                                    pdv=res[0].astype(np.float32), mask=m, water=wm & m)
                n += 1
        print(f"cached {n} potholes for {name}")


def submerge(dev, m, valid, level):
    w = m & valid & (dev > level)
    obs = dev.copy()
    obs[w] = level
    return obs, w


def _part_a(args):
    i, path = args
    z = np.load(path)
    dev, pdv, m, valid, gt = z["dev"].astype(np.float64), z["pdv"].astype(np.float64), z["mask"], z["valid"], float(z["gt"])
    rows = []
    for s in FILLS:
        obs, wm = submerge(dev, m, valid, (1 - s) * gt)
        ring = (cv2.dilate(wm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~wm & m
        pobs = pdv.copy()
        if wm.any():
            pobs[wm] = float(np.median(pdv[ring])) if ring.any() else float(np.median(pdv[wm]))
        for src, rel, val in (("realsense", obs, valid | wm), ("network", pobs, None)):
            r = {"i": i, "fill": s, "walls": src, "gt": gt}
            for meth in ("ransac", "quadric"):
                o = S.recover(rel, m, wm, val, method=meth)
                r[f"{meth}"], r[f"{meth}_abst"], r["surface"] = o["recovered_mm"], o["abstained"], o["surface_mm"]
                if meth == "ransac":
                    r["bowl"] = o["bowl"]
            rows.append(r)
    return rows


def _part_b(args):
    i, path, ds = args
    z = np.load(path)
    pdv, m, water = z["pdv"].astype(np.float64), z["mask"], z["water"]
    share = float(water.sum() / max(1, m.sum()))
    dry = float(np.percentile(pdv[m], 90))
    rows = []
    if share < DRY_MAX:
        o = S.recover(pdv, m, np.zeros_like(m))                  # no water: must be unchanged
        rows.append({"i": i, "ds": ds, "kind": "dry, no water", "ref": dry, "ransac": o["recovered_mm"],
                     "quadric": o["recovered_mm"], "surface": o["surface_mm"], "bowl": o["bowl"], "share": share})
        for s in FILLS:
            level = (1 - s) * dry
            obs, wm = submerge(pdv, m, np.ones_like(m), level)
            r = {"i": i, "ds": ds, "kind": f"dry, filled {s:.0%}", "fill": s, "ref": dry, "share": share}
            for meth in ("ransac", "quadric"):
                o = S.recover(obs, m, wm, method=meth)
                r[meth], r[f"{meth}_abst"], r["surface"] = o["recovered_mm"], o["abstained"], o["surface_mm"]
                if meth == "ransac":
                    r["bowl"] = o["bowl"]
            rows.append(r)
    elif share >= WET_MIN:
        r = {"i": i, "ds": ds, "kind": "wet (real water)", "share": share, "ref": np.nan}
        for meth in ("ransac", "quadric"):
            o = S.recover(pdv, m, water, method=meth)
            r[meth], r[f"{meth}_abst"], r["surface"] = o["recovered_mm"], o["abstained"], o["surface_mm"]
            if meth == "ransac":
                r["bowl"] = o["bowl"]
        rows.append(r)
    return rows


def pool(fn, jobs):
    from concurrent.futures import ProcessPoolExecutor
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as ex:
        for k, r in enumerate(ex.map(fn, jobs, chunksize=4)):
            out.extend(r)
            if k % 100 == 0:
                print(f"  {k + 1}/{len(jobs)}  {time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(out)


def bands(x, a, b):
    x = np.asarray(x, float)
    return np.where(x >= b, 2, np.where(x >= a, 1, 0))


def run():
    meta = json.load(open(os.path.join(CACHE, "pothrgbd", "meta.json")))
    t1, t2 = meta["t1"], meta["t2"]
    summary = {}
    # A: measured train potholes
    a = pool(_part_a, list(enumerate(sorted(glob.glob(os.path.join(CACHE, "pothrgbd", "train", "*.npz"))))))
    a.to_csv(os.path.join(OUT, "train_pothrgbd.csv"), index=False)
    rows = []
    for (s, src), g in a.groupby(["fill", "walls"]):
        yb = bands(g["gt"], 25, 50)
        for meth in ("surface", "quadric", "ransac"):
            p = g[meth].to_numpy(float)
            rows.append({"fill": s, "walls": src, "method": meth, "n": len(g), "mae": np.mean(np.abs(p - g["gt"])),
                         "bias": np.mean(p - g["gt"]), "deep_found": int(((bands(p, t1, t2) == 2) & (yb == 2)).sum()),
                         "n_deep": int((yb == 2).sum()), "under": int((bands(p, t1, t2) < yb).sum()),
                         "fallback": int(g[f"{meth}_abst"].notna().sum()) if meth != "surface" else None})
    ta = pd.DataFrame(rows)
    ta.to_csv(os.path.join(OUT, "train_pothrgbd_scores.csv"), index=False)
    summary["A_train_pothrgbd"] = ta.round(3).to_dict("records")
    # B: Kaggle and Pothole-600, reference = our own dry reading
    jobs = []
    for ds in ("test_kaggle", "test_p600"):
        jobs += [(len(jobs) + k, p, ds) for k, p in enumerate(sorted(glob.glob(os.path.join(CACHE, ds, "*.npz"))))]
    b = pool(_part_b, jobs)
    b.to_csv(os.path.join(OUT, "kaggle_p600.csv"), index=False)
    rows = []
    for kind, g in b.groupby("kind"):
        r = {"kind": kind, "n": len(g)}
        if kind.startswith("dry"):
            rb = bands(g["ref"], t1, t2)
            for meth in ("surface", "quadric", "ransac"):
                p = g[meth].to_numpy(float)
                r[f"{meth}_mae"] = float(np.mean(np.abs(p - g["ref"])))
                r[f"{meth}_within5mm"] = float(np.mean(np.abs(p - g["ref"]) <= 5))
                r[f"{meth}_same_band"] = float(np.mean(bands(p, t1, t2) == rb))
            if kind == "dry, no water":
                r["changed_by_recover"] = int((np.abs(g["ransac"] - g["ref"]) > 1e-6).sum())
            else:
                r["ransac_fallback"] = int(g["ransac_abst"].notna().sum())
                r["quadric_fallback"] = int(g["quadric_abst"].notna().sum())
        else:
            r["ransac_bowl_fitted"] = int(g["ransac_abst"].isna().sum())
            r["quadric_bowl_fitted"] = int(g["quadric_abst"].isna().sum())
            r["median_raise_ransac_mm"] = float(np.median(g["ransac"] - g["surface"]))
            r["median_raise_quadric_mm"] = float(np.median(g["quadric"] - g["surface"]))
        r["bowls_used"] = g["bowl"].fillna("none").value_counts().to_dict()
        rows.append(r)
    summary["B_kaggle_p600"] = rows
    summary["B_skipped_middle_share"] = int(len(jobs) - b.drop_duplicates("i").shape[0])
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(ta.round(2).to_string(index=False))
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    {"cache": cache, "run": run}[sys.argv[1]]()
