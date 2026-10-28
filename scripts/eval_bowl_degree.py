"""
Follow-up to scripts/eval_ransac_bowl.py: can the RANSAC bowl (a) abstain less and (b) pick its
polynomial degree per pothole?

Per pothole, fill and wall source, RANSAC bowls of degree 2..5 are fitted once (inlier tolerance
8 mm, as chosen on val before) and every rule below is read off the same fits:

  cubic        degree 3 only (the adopted baseline, submerged_depth.recover)
  cascade      degree 3, else 4, else 2 (RANSAC), else the least-squares quadric, else surface
  band         degree with the smallest error predicting the held-out wall strip at the waterline
  bic          degree with the lowest BIC on its inliers: n log(RSS / n) + k log(n)
  cv           degree with the lowest 5-fold cross-validated error on its inliers
  median       median of the readings of every degree that passes the bowl check
  oracle       the degree whose reading is closest to the truth (upper bound, not usable)
Each selection rule falls back down the cascade when none of its degrees passes the bowl check.

Protocol, fixed before results: the rule ranked best on val (mean MAE over the six fill x wall
cells, oracle excluded) is scored once on test and adopted over `cubic` only if the 50 %-fill
network-wall MAE drops with a paired-bootstrap 95 % interval above zero, the 25 % and 75 % network
MAEs do not worsen by more than 0.2 mm, and 50 % under-reporting does not rise.

    python scripts/eval_bowl_degree.py
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2                                                                    # noqa: E402
import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ransac_bowl as E                                                  # noqa: E402

OUT = os.path.join(E.OUT_DIR, "degree")
DEGS = (2, 3, 4, 5)
TOL = 8.0
RULES = ("cubic", "cascade", "band", "bic", "cv", "median", "oracle")
cv2.setNumThreads(1)


def reading(ctx, relief, f):
    """Recovered p90 reading with bowl f filled under the water (as submerged_depth.recover)."""
    m, w, ok = ctx.m, ctx.w, ctx.ok
    filled = relief.astype(np.float64).copy()
    wy, wx = ctx.wpix
    filled[wy, wx] = np.clip(np.maximum(filled[wy, wx], f(ctx.wx, ctx.wy)), 0, E.S.MAX_DEPTH_MM)
    return float(np.percentile(filled[m & (ok | w)], 90))


def inlier_stats(ctx, f, deg):
    r = np.abs(f(ctx.px, ctx.py) - ctx.z)
    inl = r < TOL
    return inl, float((r[inl] ** 2).sum()), int(inl.sum()), E.design(ctx.px[:1], ctx.py[:1], deg).shape[1]


def cv_error(ctx, inl, deg, folds=5):
    """5-fold cross-validation of a least-squares refit on the RANSAC inliers."""
    idx = np.nonzero(inl)[0]
    if idx.size < 10 * E.design(ctx.px[:1], ctx.py[:1], deg).shape[1]:
        return np.inf
    part = np.random.default_rng(0).permutation(idx.size) % folds
    err = []
    a = E.design(ctx.px[idx], ctx.py[idx], deg)
    z = ctx.z[idx]
    for k in range(folds):
        tr, te = part != k, part == k
        coef, *_ = np.linalg.lstsq(a[tr], z[tr], rcond=None)
        err.append(np.abs(a[te] @ coef - z[te]))
    return float(np.median(np.concatenate(err)))


def all_rules(relief, mask, water, valid, gt):
    m, w = mask, water & mask
    ok = np.isfinite(relief) if valid is None else valid & np.isfinite(relief)
    surface = float(np.percentile(relief[m & ok], 90))
    out = {r: surface for r in RULES}
    out.update({f"abst_{r}": True for r in RULES})
    if not w.any():
        return out
    ctx = E.Ctx(relief, m, w, ok)
    if ctx.n_wall < E.S.MIN_WALL_PX:
        return out
    fits = {d: E.ransac(ctx, d, TOL) for d in DEGS}
    read = {d: reading(ctx, relief, f) for d, f in fits.items() if f is not None}
    lsq = E.ls(ctx, 2)
    order = [3, 4, 2]
    casc = next((read[d] for d in order if d in read), reading(ctx, relief, lsq) if lsq else None)

    def put(rule, val):
        if val is not None:
            out[rule], out[f"abst_{rule}"] = val, False
        elif casc is not None:
            out[rule], out[f"abst_{rule}"] = casc, False

    put("cubic", read.get(3))
    out["cubic"] = read.get(3, surface)
    out["abst_cubic"] = 3 not in read
    put("cascade", casc)
    if not read:
        for r in ("band", "bic", "cv", "median", "oracle"):
            put(r, None)
        return out
    # band: hold out the wall strip next to the waterline, refit each degree without it
    if ctx.band.sum() >= 20 and (~ctx.band).sum() >= 60:
        band_err = {}
        for d in read:
            f = E.ransac(ctx, d, TOL, sel=~ctx.band)
            if f is not None:
                band_err[d] = float(np.median(np.abs(f(ctx.px[ctx.band], ctx.py[ctx.band]) - ctx.z[ctx.band])))
        put("band", read[min(band_err, key=band_err.get)] if band_err else None)
    else:
        put("band", None)
    bic, cv = {}, {}
    for d, f in fits.items():
        if f is None:
            continue
        inl, rss, n, k = inlier_stats(ctx, f, d)
        if n > k + 2 and rss > 0:
            bic[d] = n * np.log(rss / n) + k * np.log(n)
        cv[d] = cv_error(ctx, inl, d)
    put("bic", read[min(bic, key=bic.get)] if bic else None)
    put("cv", read[min(cv, key=cv.get)] if cv and np.isfinite(min(cv.values())) else None)
    put("median", float(np.median(list(read.values()))))
    put("oracle", min(read.values(), key=lambda v: abs(v - gt)))
    for d in DEGS:
        out[f"deg{d}"] = read.get(d)
    return out


def _one(args):
    i, c = args
    dev, pdv, m, valid, gt = c["dev"].astype(np.float64), c["pdv"].astype(np.float64), c["mask"], c["valid"], float(c["gt"])
    rows = []
    for s in E.FILLS:
        level = (1 - s) * gt
        obs, wmask = E.submerge(dev, m, valid, level)
        ring = (cv2.dilate(wmask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~wmask & m
        pobs = pdv.copy()
        if wmask.any():
            pobs[wmask] = float(np.median(pdv[ring])) if ring.any() else float(np.median(pdv[wmask]))
        for src, rel, val in (("realsense", obs, valid | wmask), ("network", pobs, None)):
            o = all_rules(rel, m, wmask, val, gt)
            rows.append({"i": i, "fill": s, "walls": src, "gt": gt, **o})
    return rows


def run_split(split):
    from concurrent.futures import ProcessPoolExecutor
    data = E.load(split)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as ex:
        for k, r in enumerate(ex.map(_one, list(enumerate(data)))):
            rows.extend(r)
            if k % 25 == 0:
                print(f"  {split}: {k + 1}/{len(data)}  {time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows)


def score(t, t1, t2):
    def bands(x, a, b):
        x = np.asarray(x, float)
        return np.where(x >= b, 2, np.where(x >= a, 1, 0))
    out = []
    for (fill, src), g in t.groupby(["fill", "walls"]):
        yb = bands(g["gt"], 25, 50)
        for r in RULES:
            p = g[r].to_numpy(float)
            b = bands(p, t1, t2)
            out.append({"rule": r, "fill": fill, "walls": src, "mae": float(np.mean(np.abs(p - g["gt"]))),
                        "bias": float(np.mean(p - g["gt"])), "under": int((b < yb).sum()),
                        "deep_found": int(((b == 2) & (yb == 2)).sum()), "abstained": int(g[f"abst_{r}"].sum())})
    return pd.DataFrame(out)


def boot(t, a, b, fill=0.5, src="network", n=5000):
    g = t[(t.fill == fill) & (t.walls == src)].sort_values("i")
    d = np.abs(g[a] - g["gt"]).to_numpy() - np.abs(g[b] - g["gt"]).to_numpy()
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(E.CACHE, "meta.json")))
    res = {}
    for split in ("val", "test"):
        t = run_split(split)
        t.to_csv(os.path.join(OUT, f"per_pothole_{split}.csv"), index=False)
        sc = score(t, meta["t1"], meta["t2"])
        sc.to_csv(os.path.join(OUT, f"scores_{split}.csv"), index=False)
        res[split] = (t, sc)
    sv = res["val"][1]
    rank = sv[sv.rule != "oracle"].groupby("rule").mae.mean().sort_values()
    chosen = rank.index[0]
    t, st = res["test"]
    cell = lambda r, f: st[(st.rule == r) & (st.fill == f) & (st.walls == "network")].iloc[0]  # noqa: E731
    v = {"val_rank": rank.round(3).to_dict(), "chosen_on_val": chosen}
    if chosen != "cubic":
        d, lo, hi = boot(t, chosen, "cubic")
        v.update({"test_50_network": {"cubic": cell("cubic", 0.5).mae, "chosen": cell(chosen, 0.5).mae,
                                      "diff": d, "ci95": [lo, hi]},
                  "test_25_change": cell(chosen, 0.25).mae - cell("cubic", 0.25).mae,
                  "test_75_change": cell(chosen, 0.75).mae - cell("cubic", 0.75).mae,
                  "test_50_under": [int(cell("cubic", 0.5).under), int(cell(chosen, 0.5).under)]})
        v["adopt"] = bool(hi < 0 and v["test_25_change"] <= 0.2 and v["test_75_change"] <= 0.2
                          and v["test_50_under"][1] <= v["test_50_under"][0])
    else:
        v["adopt"] = False
    with open(os.path.join(OUT, "verdict.json"), "w") as f:
        json.dump(v, f, indent=2, default=float)
    print(json.dumps(v, indent=2, default=float))


if __name__ == "__main__":
    main()
