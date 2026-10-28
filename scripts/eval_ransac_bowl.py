"""
Does a robust bowl fit (RANSAC and relatives, any polynomial degree) recover the depth hidden under
water better than the least-squares quadric in submerged_depth.recover?

Benchmark: the controlled submersion of scripts/eval_submerged_depth.py. Each measured pothole is
filled with level water to 25 / 50 / 75 % of its depth and read from its dry walls, once with
RealSense walls and once with the relief network's walls.

Protocol, fixed before any result was looked at:
  * every setting (inlier tolerance, Huber delta) is chosen on the `val` split (98 potholes);
  * the variant ranked best on `val` (lowest mean of the six fill x wall MAEs) is then scored once
    on the test split (208 potholes, the paper's benchmark);
  * it is adopted only if, on test, it lowers the 50 %-fill network-wall MAE (current 11.03 mm)
    with a paired-bootstrap 95 % interval of the improvement above zero, does not worsen the 25 %
    and 75 % network MAE by more than 0.2 mm, and does not raise the 50 % under-reported count.
All variants are reported on test as well, as secondary results.

    python scripts/eval_ransac_bowl.py cache     # network + plane once per pothole (GPU)
    python scripts/eval_ransac_bowl.py run       # all variants, val then test
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
import numpy as np
cv2.setNumThreads(1)
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import submerged_depth as S                                                  # noqa: E402

OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "water_depth", "ransac")
CACHE = os.path.join(OUT_DIR, "cache")
FILLS = (0.25, 0.5, 0.75)
SPLITS = {"val": "val", "test": "test_pothrgbd"}
N_SCORE = 2000          # points used to score a RANSAC hypothesis
N_HYP = 300             # hypotheses per fit
TOP_K = 10              # best hypotheses refitted and bowl-checked
N_REFIT = 6000          # support points used when a hypothesis is refitted on its inliers


# ── cache: RealSense relief, network relief, mask, ground truth ──────────────────────────────
def cache():
    import relief_depth as R
    from scripts.eval_submerged_depth import net_relief
    from scripts.pothrgbd_metric_labels import (DATA_DIR, DEPTH_VALID_MIN, load_polygons,
                                                road_plane_deviation, timestamp_key)
    from scripts.train_relief_depth import split_keys
    d = pd.read_csv(os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_metric.csv"))
    imgs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}
    for name, split in SPLITS.items():
        os.makedirs(os.path.join(CACHE, name), exist_ok=True)
        keys = set(split_keys(split))
        gt = d[d.key.isin(keys) & (d.mask_source == "gt")]
        n = 0
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
                np.savez_compressed(os.path.join(CACHE, name, f"{key}_{int(r.pothole)}.npz"),
                                    dev=dev.astype(np.float32), pdv=pdv.astype(np.float32), mask=m, valid=valid,
                                    gt=float(np.percentile(dev[m & valid], 90)))
                n += 1
        print(f"cached {n} potholes for {name}")
    meta = R.meta()["decision_thresholds_mm"]
    with open(os.path.join(CACHE, "meta.json"), "w") as f:
        json.dump({"t1": meta["moderate_from"], "t2": meta["deep_from"]}, f)


# ── surface models ───────────────────────────────────────────────────────────────────────────
def design(x, y, deg):
    """All monomials x^i y^j with i + j <= deg; for deg 2 the same order as submerged_depth._design."""
    if deg == 2:
        return np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    cols = [x ** i * y ** (k - i) for k in range(deg, -1, -1) for i in range(k, -1, -1)]
    return np.column_stack(cols)


def quad_is_bowl(c):
    hxx, hyy, hxy = 2 * c[..., 0], 2 * c[..., 1], c[..., 2]
    return (hxx < 0) & (hyy < 0) & (hxx * hyy - hxy * hxy > 0)


class Ctx:
    """Everything one fit needs: the support points, the water region and the normalisation."""

    def __init__(self, relief, mask, water, ok):
        self.relief, self.m, self.w, self.ok = relief, mask, water, ok
        k = np.ones((2 * S.ANCHOR_PX + 1, 2 * S.ANCHOR_PX + 1), np.uint8)
        anchor = (cv2.dilate(mask.astype(np.uint8), k) > 0) & ~mask
        self.walls = mask & ~water
        pts = (self.walls | anchor) & ok
        self.n_wall = int((self.walls & ok).sum())
        ys, xs = np.nonzero(pts)
        self.cy, self.cx = float(np.mean(np.nonzero(mask)[0])), float(np.mean(np.nonzero(mask)[1]))
        self.s = float(max(np.ptp(xs), np.ptp(ys), 1)) if xs.size else 1.0
        self.px, self.py = (xs - self.cx) / self.s, (ys - self.cy) / self.s
        self.z = relief[ys, xs].astype(np.float64)
        self.pys, self.pxs = ys, xs
        wy, wx = np.nonzero(water)
        self.wx, self.wy = (wx - self.cx) / self.s, (wy - self.cy) / self.s
        self.wpix = (wy, wx)
        # wall pixels next to the waterline: held out when a degree is chosen from the walls
        near = cv2.dilate(water.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        self.band = (near & self.walls & ok)[ys, xs]
        self.edge = water & ~(cv2.erode(water.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        self.level = float(np.median(relief[water])) if water.any() else 0.0
        # RANSAC refits use at most N_REFIT support points (a fixed random subset)
        self.sub = np.random.default_rng(0).permutation(xs.size)[:N_REFIT] if xs.size > N_REFIT else None

    def bowl_ok(self, f, deg, coef=None):
        """deg 2: negative-definite Hessian (as submerged_depth). Higher degrees: the fitted
        surface must peak strictly inside the water (not on its edge) and below the waterline."""
        if deg == 2:
            return bool(quad_is_bowl(coef))
        if self.wx.size < 5:
            return False
        v = f(self.wx, self.wy)
        i = int(np.argmax(v))
        wy, wx = self.wpix
        if self.edge[wy[i], wx[i]] or v[i] <= self.level + 0.5:
            return False
        e = 2.0 / self.s                                     # numeric Hessian, 2 px steps
        x0, y0 = self.wx[i], self.wy[i]
        fxx = (f(x0 + e, y0) - 2 * f(x0, y0) + f(x0 - e, y0)) / e ** 2
        fyy = (f(x0, y0 + e) - 2 * f(x0, y0) + f(x0, y0 - e)) / e ** 2
        fxy = (f(x0 + e, y0 + e) - f(x0 + e, y0 - e) - f(x0 - e, y0 + e) + f(x0 - e, y0 - e)) / (4 * e * e)
        return bool(fxx < 0 and fyy < 0 and fxx * fyy - fxy * fxy > 0)


def as_fn(coef, deg):
    return lambda x, y: design(np.atleast_1d(x), np.atleast_1d(y), deg) @ coef


def fit_ls(ctx, deg, sel=None, weights=None):
    x, y, z = ctx.px, ctx.py, ctx.z
    if sel is not None:
        x, y, z = x[sel], y[sel], z[sel]
    a = design(x, y, deg)
    if weights is not None:
        sw = np.sqrt(weights[sel] if sel is not None else weights)
        a, z = a * sw[:, None], z * sw
    if a.shape[0] < a.shape[1] * 3:
        return None
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    return coef


def ls(ctx, deg):
    coef = fit_ls(ctx, deg)
    if coef is None:
        return None
    f = as_fn(coef, deg)
    return f if ctx.bowl_ok(f, deg, coef) else None


def huber(ctx, deg, delta, iters=20):
    wts = np.ones_like(ctx.z)
    coef = None
    for _ in range(iters):
        coef = fit_ls(ctx, deg, weights=wts)
        if coef is None:
            return None
        r = np.abs(design(ctx.px, ctx.py, deg) @ coef - ctx.z)
        wts = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))
    f = as_fn(coef, deg)
    return f if ctx.bowl_ok(f, deg, coef) else None


def ransac(ctx, deg, tol, msac=False, sel=None, seed=None):
    """RANSAC (or MSAC: truncated squared residual) with bowl screening. Hypotheses from minimal
    samples are scored on up to N_SCORE points; the TOP_K best are refitted on their inliers over
    all points, and the best one that is a bowl wins."""
    rng = np.random.default_rng(int(os.environ.get("RANSAC_SEED", 0)) if seed is None else seed)
    x, y, z = ctx.px, ctx.py, ctx.z
    keep = np.ones(z.size, bool) if sel is None else sel.copy()
    if ctx.sub is not None:
        cap = np.zeros(z.size, bool)
        cap[ctx.sub] = True
        keep &= cap
    x, y, z = x[keep], y[keep], z[keep]
    a = design(x, y, deg)
    p = a.shape[1]
    if a.shape[0] < p * 3:
        return None
    sc = rng.choice(a.shape[0], min(N_SCORE, a.shape[0]), replace=False)
    idx = np.stack([rng.choice(a.shape[0], p, replace=False) for _ in range(N_HYP)])
    A = a[idx]                                                   # (H, p, p)
    A = A + 1e-9 * np.eye(p)
    try:
        C = np.linalg.solve(A, z[idx][..., None])[..., 0]        # (H, p)
    except np.linalg.LinAlgError:
        C = np.stack([np.linalg.lstsq(A[i], z[idx[i]], rcond=None)[0] for i in range(N_HYP)])
    if deg == 2:                                                 # cheap exact screen for quadrics
        C = C[quad_is_bowl(C)]
        if not len(C):
            return None
    r = np.abs(C @ a[sc].T - z[sc])                              # (H, n)
    score = (np.minimum(r, tol) ** 2).sum(1) if msac else -(r < tol).sum(1)
    best = None
    for h in np.argsort(score)[:TOP_K]:
        inl = np.abs(a @ C[h] - z) < tol
        if inl.sum() < p * 3:
            continue
        coef, *_ = np.linalg.lstsq(a[inl], z[inl], rcond=None)
        inl = np.abs(a @ coef - z) < tol                         # one re-estimation of the inliers
        if inl.sum() < p * 3:
            continue
        coef, *_ = np.linalg.lstsq(a[inl], z[inl], rcond=None)
        f = as_fn(coef, deg)
        if not ctx.bowl_ok(f, deg, coef):
            continue
        cost = (np.minimum(np.abs(a @ coef - z), tol) ** 2).sum() if msac else -inl.sum()
        if best is None or cost < best[0]:
            best = (cost, f)
    return None if best is None else best[1]


def by_band(ctx, degs, fitter):
    """Choose the degree from the walls alone: hold out the wall strip next to the waterline,
    fit on the rest, keep the degree that predicts the strip best; refit on all points."""
    if ctx.band.sum() < 20 or (~ctx.band).sum() < 60:
        return fitter(ctx, 2, None)
    best = None
    for d in degs:
        f = fitter(ctx, d, ~ctx.band)
        if f is None:
            continue
        err = float(np.median(np.abs(f(ctx.px[ctx.band], ctx.py[ctx.band]) - ctx.z[ctx.band])))
        if best is None or err < best[0]:
            best = (err, d)
    return None if best is None else fitter(ctx, best[1], None)


def recover(relief, mask, water, valid, method):
    """submerged_depth.recover with the bowl fit swapped for `method(ctx) -> f or None`."""
    m = mask
    w = water & m
    ok = np.isfinite(relief) if valid is None else valid & np.isfinite(relief)
    observed = float(np.percentile(relief[m & ok], 90))
    if not w.any():
        return observed, "no water"
    ctx = Ctx(relief, m, w, ok)
    if ctx.n_wall < S.MIN_WALL_PX:
        return observed, "too little visible wall"
    f = method(ctx)
    if f is None:
        return observed, "not a bowl"
    wy, wx = ctx.wpix
    filled = relief.astype(np.float64).copy()
    filled[wy, wx] = np.clip(np.maximum(filled[wy, wx], f(ctx.wx, ctx.wy)), 0, S.MAX_DEPTH_MM)
    return float(np.percentile(filled[m & (ok | w)], 90)), None


def ransac_sel(ctx, d, sel, tol, msac=False):
    return ransac(ctx, d, tol, msac=msac, sel=sel)


def ls_sel(ctx, d, sel):
    if sel is None:
        return ls(ctx, d)
    coef = fit_ls(ctx, d, sel)
    if coef is None:
        return None
    f = as_fn(coef, d)
    return f if ctx.bowl_ok(f, d, coef) else None


TOLS = (1.0, 2.0, 4.0, 8.0)


def variants():
    v = {"V0 least squares, quadric (current)": lambda c: ls(c, 2)}
    for t in TOLS:
        v[f"V1 RANSAC quadric, tol {t:g}"] = lambda c, t=t: ransac(c, 2, t)
        v[f"V2 MSAC quadric, tol {t:g}"] = lambda c, t=t: ransac(c, 2, t, msac=True)
        v[f"V3 Huber quadric, delta {t:g}"] = lambda c, t=t: huber(c, 2, t)
    for d in (3, 4):
        v[f"V4 least squares, degree {d}"] = lambda c, d=d: ls(c, d)
        for t in TOLS:
            v[f"V5 RANSAC degree {d}, tol {t:g}"] = lambda c, d=d, t=t: ransac(c, d, t)
    v["V6a least squares, degree by band"] = lambda c: by_band(c, (2, 3, 4), ls_sel)
    for t in TOLS:
        v[f"V6 RANSAC, degree by band, tol {t:g}"] = lambda c, t=t: by_band(c, (2, 3, 4),
                                                                              lambda cc, d, s: ransac_sel(cc, d, s, t))
    return v


# ── evaluation ───────────────────────────────────────────────────────────────────────────────
def load(split):
    out = []
    for p in sorted(glob.glob(os.path.join(CACHE, split, "*.npz"))):
        z = np.load(p)
        out.append({k: z[k] for k in z.files})
    return out


def submerge(dev, m, valid, level):
    w = m & valid & (dev > level)
    obs = dev.copy()
    obs[w] = level
    return obs, w


def _one(args):
    i, c, names = args
    methods = variants()
    return one_pothole(i, c, {k: methods[k] for k in names})


def run_split(split, methods, workers=None):
    """All variants on every pothole of a split, potholes spread over worker processes."""
    from concurrent.futures import ProcessPoolExecutor
    data = load(split)
    t0 = time.time()
    rows = []
    jobs = [(i, c, list(methods)) for i, c in enumerate(data)]
    with ProcessPoolExecutor(workers or max(1, os.cpu_count() - 2)) as ex:
        for k, r in enumerate(ex.map(_one, jobs)):
            rows.extend(r)
            if k % 20 == 0:
                print(f"  {split}: {k + 1}/{len(data)}  {time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows)


def one_pothole(i, c, methods):
    rows = []
    if True:
        dev, pdv, m, valid, gt = c["dev"].astype(np.float64), c["pdv"].astype(np.float64), c["mask"], c["valid"], float(c["gt"])
        for s in FILLS:
            level = (1 - s) * gt
            obs, wmask = submerge(dev, m, valid, level)
            ring = (cv2.dilate(wmask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~wmask & m
            pobs = pdv.copy()
            if wmask.any():
                pobs[wmask] = float(np.median(pdv[ring])) if ring.any() else float(np.median(pdv[wmask]))
            for name, meth in methods.items():
                rs, ra = recover(obs, m, wmask, valid | wmask, meth)
                nt, na = recover(pobs, m, wmask, None, meth)
                rows.append({"i": i, "fill": s, "method": name, "gt": gt, "realsense": rs, "network": nt,
                             "rs_abstain": ra, "net_abstain": na})
    return rows


def score(t, t1, t2):
    def bands(x, a=t1, b=t2):
        x = np.asarray(x, float)
        return np.where(x >= b, 2, np.where(x >= a, 1, 0))
    out = []
    for (meth, fill), g in t.groupby(["method", "fill"]):
        yb = bands(g["gt"], 25, 50)
        for src in ("realsense", "network"):
            p = g[src].to_numpy()
            b = bands(p)
            out.append({"method": meth, "fill": fill, "walls": src, "mae": float(np.mean(np.abs(p - g["gt"]))),
                        "bias": float(np.mean(p - g["gt"])), "under": int((b < yb).sum()),
                        "deep_found": int(((b == 2) & (yb == 2)).sum()), "n_deep": int((yb == 2).sum()),
                        "abstained": int(g[("rs" if src == "realsense" else "net") + "_abstain"]
                                         .isin(["too little visible wall", "not a bowl"]).sum())})
    return pd.DataFrame(out)


def boot_diff(t, a, b, fill, src="network", n=5000, seed=0):
    """Paired bootstrap of MAE(a) - MAE(b) over potholes."""
    ga = t[(t.method == a) & (t.fill == fill)].sort_values("i")
    gb = t[(t.method == b) & (t.fill == fill)].sort_values("i")
    ea, eb = np.abs(ga[src].to_numpy() - ga["gt"].to_numpy()), np.abs(gb[src].to_numpy() - gb["gt"].to_numpy())
    d = ea - eb
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run():
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    methods = variants()
    base = "V0 least squares, quadric (current)"
    results = {}
    for split in ("val", "test"):
        print(f"{split}: {len(methods)} variants")
        t = run_split(split, methods)
        t.to_csv(os.path.join(OUT_DIR, f"per_pothole_{split}.csv"), index=False)
        sc = score(t, meta["t1"], meta["t2"])
        sc.to_csv(os.path.join(OUT_DIR, f"scores_{split}.csv"), index=False)
        results[split] = (t, sc)
    # selection on val only: lowest mean MAE over the six fill x wall cells
    sv = results["val"][1]
    rank = sv.groupby("method").mae.mean().sort_values()
    chosen = rank.index[0]
    t, st = results["test"]
    cell = lambda s, m, f, src: s[(s.method == m) & (s.fill == f) & (s.walls == src)].iloc[0]  # noqa: E731
    d, lo, hi = boot_diff(t, chosen, base, 0.5)
    verdict = {
        "chosen_on_val": chosen, "val_rank": rank.round(3).to_dict(),
        "test_50_network": {"base": cell(st, base, 0.5, "network").mae, "chosen": cell(st, chosen, 0.5, "network").mae,
                            "diff": d, "ci95": [lo, hi]},
        "test_25_network_change": float(cell(st, chosen, 0.25, "network").mae - cell(st, base, 0.25, "network").mae),
        "test_75_network_change": float(cell(st, chosen, 0.75, "network").mae - cell(st, base, 0.75, "network").mae),
        "test_50_under": {"base": int(cell(st, base, 0.5, "network").under), "chosen": int(cell(st, chosen, 0.5, "network").under)},
    }
    v = verdict
    v["adopt"] = bool(hi < 0 and v["test_25_network_change"] <= 0.2 and v["test_75_network_change"] <= 0.2
                      and v["test_50_under"]["chosen"] <= v["test_50_under"]["base"])
    with open(os.path.join(OUT_DIR, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))


def check():
    """V0 must reproduce the published benchmark (duplicate row removed); time a few variants."""
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    v = variants()
    base = "V0 least squares, quadric (current)"
    t = run_split("test", {base: v[base]})
    print(score(t, meta["t1"], meta["t2"])[["fill", "walls", "mae", "under", "deep_found", "abstained"]].round(2).to_string())
    names = ["V1 RANSAC quadric, tol 2", "V5 RANSAC degree 4, tol 2", "V6 RANSAC, degree by band, tol 2"]
    t0 = time.time()
    run_split("val", {k: v[k] for k in names})
    print(f"{len(names)} RANSAC variants on val: {time.time() - t0:.0f} s")


def seeds():
    """The chosen variant against the current fit under other RANSAC seeds (test, network + RealSense)."""
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    chosen = json.load(open(os.path.join(OUT_DIR, "verdict.json")))["chosen_on_val"]
    for sd in (1, 2, 3):
        os.environ["RANSAC_SEED"] = str(sd)
        sc = score(run_split("test", {chosen: variants()[chosen]}), meta["t1"], meta["t2"])
        print(f"seed {sd}: " + "  ".join(f"{r.walls[:3]} {r.fill:.0%} {r.mae:.2f}" for r in sc.itertuples()), flush=True)


if __name__ == "__main__":
    {"cache": cache, "run": run, "check": check, "seeds": seeds}[sys.argv[1]]()
