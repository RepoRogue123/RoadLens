"""
Per-pothole bowl degree, learned. Picking the degree from the walls alone (held-out strip, BIC,
cross-validation; scripts/eval_bowl_degree.py) lost to a fixed cubic, yet a truth-picked degree
would cut the half-fill error from 9.04 to 6.26 mm. Here the choice is learned from features
that do not need the truth.

Per pothole, fill and wall source: RANSAC bowls of degree 2..5 (8 mm tolerance), their readings,
inlier shares and residuals, plus the outline, the visible wall and the water share. Learners:
  clf   gradient-boosted classifier: which degree lands closest to the truth
  err   one gradient-boosted regressor per degree: how far its reading will be from the truth;
        the degree with the smallest predicted error is used
Missing degrees (no bowl) are skipped; with none left the cascade's answer is used.

Protocol, fixed before results: learners are trained on the PothRGBD train split (745 potholes),
one per wall source; the learner with the lower mean network-wall MAE over the three fills on val
is scored once on test and adopted over the cascade (submerged_depth.recover) only if the 50 %-fill
network MAE drops with a paired-bootstrap 95 % interval above zero, the 25 % and 75 % network
MAEs do not worsen by more than 0.2 mm, and 50 % under-reporting does not rise.

    python scripts/eval_bowl_degree_learned.py features
    python scripts/eval_bowl_degree_learned.py run
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ransac_bowl as E                                                  # noqa: E402
from eval_bowl_degree import reading                                         # noqa: E402

PROJECT_DIR = E.PROJECT_DIR
OUT = os.path.join(E.OUT_DIR, "degree_learned")
SOURCES = {"train": os.path.join(PROJECT_DIR, "ml_results", "water_depth", "reliability", "cache", "pothrgbd", "train"),
           "val": os.path.join(E.CACHE, "val"), "test": os.path.join(E.CACHE, "test")}
DEGS = (2, 3, 4, 5)
TOL = 8.0
cv2.setNumThreads(1)


def context_row(relief, m, w, valid, gt):
    import submerged_depth as S
    ok = np.isfinite(relief) if valid is None else valid & np.isfinite(relief)
    surface = float(np.percentile(relief[m & ok], 90))
    cas = S.recover(relief, m, w, valid)["recovered_mm"]
    r = {"gt": gt, "surface": surface, "cascade": cas, "f_area": float(np.log(m.sum())),
         "f_water": float(w.sum() / max(1, m.sum()))}
    if not w.any():
        return r
    ctx = E.Ctx(relief, m, w, ok)
    r["f_wall"] = float(ctx.n_wall / max(1, m.sum()))
    if ctx.n_wall < E.S.MIN_WALL_PX:
        return r
    wz = relief[ctx.walls & ok]
    r.update({"f_wall_span": float(np.ptp(wz)), "f_wall_max": float(wz.max()), "f_level": ctx.level,
              "f_surface": surface})
    reads = []
    for d in DEGS:
        f = E.ransac(ctx, d, TOL)
        if f is None:
            continue
        res = np.abs(f(ctx.px, ctx.py) - ctx.z)
        inl = res < TOL
        val = reading(ctx, relief, f)
        reads.append(val)
        r.update({f"deg{d}": val, f"f_d{d}_raise": val - surface, f"f_d{d}_inl": float(inl.mean()),
                  f"f_d{d}_rms": float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else np.nan})
    if reads:
        r["f_spread"] = float(np.std(reads))
        r["f_cascade_raise"] = cas - surface
    return r


def _one(args):
    split, i, path = args
    z = np.load(path)
    dev, pdv, m, valid, gt = z["dev"].astype(np.float64), z["pdv"].astype(np.float64), z["mask"], z["valid"], float(z["gt"])
    rows = []
    for s in E.FILLS:
        obs, wm = E.submerge(dev, m, valid, (1 - s) * gt)
        ring = (cv2.dilate(wm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~wm & m
        pobs = pdv.copy()
        if wm.any():
            pobs[wm] = float(np.median(pdv[ring])) if ring.any() else float(np.median(pdv[wm]))
        for src, rel, val in (("realsense", obs, valid | wm), ("network", pobs, None)):
            rows.append({"split": split, "i": i, "fill": s, "walls": src, **context_row(rel, m, wm, val, gt)})
    return rows


def features():
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(OUT, exist_ok=True)
    jobs = [(sp, i, p) for sp, d in SOURCES.items() for i, p in enumerate(sorted(glob.glob(os.path.join(d, "*.npz"))))]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=2)):
            rows.extend(r)
            if k % 100 == 0:
                print(f"  {k + 1}/{len(jobs)}  {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "features.csv"), index=False)
    print("wrote", len(rows), "rows")


def feat_cols(t):
    return [c for c in t.columns if c.startswith("f_")]


def choose(t, scores, higher_is_worse=True):
    """Per row: the reading of the available degree with the best score, else the cascade."""
    out = t["cascade"].to_numpy(float).copy()
    reads = np.column_stack([t[f"deg{d}"].to_numpy(float) if f"deg{d}" in t else np.full(len(t), np.nan) for d in DEGS])
    s = np.where(np.isnan(reads), np.inf if higher_is_worse else -np.inf, scores)
    pick = np.argmin(s, 1) if higher_is_worse else np.argmax(s, 1)
    has = ~np.all(np.isnan(reads), 1)
    out[has] = reads[has, pick[has]]
    return out


def fit_predict(tr, te):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    cols = feat_cols(tr)
    reads_tr = np.column_stack([tr[f"deg{d}"].to_numpy(float) for d in DEGS])
    err_tr = np.abs(reads_tr - tr["gt"].to_numpy()[:, None])
    have = ~np.all(np.isnan(reads_tr), 1)
    best = np.nanargmin(np.where(np.isnan(err_tr), np.inf, err_tr)[have], 1)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    clf.fit(tr.loc[have, cols], best)
    proba = np.zeros((len(te), len(DEGS)))
    proba[:, clf.classes_] = clf.predict_proba(te[cols])
    out = {"clf": choose(te, proba, higher_is_worse=False)}
    pred_err = np.full((len(te), len(DEGS)), np.inf)
    for k, d in enumerate(DEGS):
        ok = ~np.isnan(err_tr[:, k])
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
        reg.fit(tr.loc[ok, cols], err_tr[ok, k])
        pred_err[:, k] = reg.predict(te[cols])
    out["err"] = choose(te, pred_err)
    reads_te = np.column_stack([te[f"deg{d}"].to_numpy(float) for d in DEGS])
    e = np.abs(reads_te - te["gt"].to_numpy()[:, None])
    out["oracle"] = choose(te, np.where(np.isnan(e), np.inf, e))
    out["cubic"] = np.where(np.isnan(te["deg3"]), te["surface"], te["deg3"])
    out["cascade"] = te["cascade"].to_numpy(float)
    return out


def scores(te, preds, t1, t2):
    def bands(x, a, b):
        x = np.asarray(x, float)
        return np.where(x >= b, 2, np.where(x >= a, 1, 0))
    rows = []
    for fill in E.FILLS:
        g = te["fill"].to_numpy() == fill
        gt = te["gt"].to_numpy()[g]
        yb = bands(gt, 25, 50)
        for name, p in preds.items():
            q = p[g]
            b = bands(q, t1, t2)
            rows.append({"rule": name, "fill": fill, "mae": float(np.mean(np.abs(q - gt))), "bias": float(np.mean(q - gt)),
                         "under": int((b < yb).sum()), "deep_found": int(((b == 2) & (yb == 2)).sum())})
    return pd.DataFrame(rows)


def run():
    meta = json.load(open(os.path.join(E.CACHE, "meta.json")))
    t = pd.read_csv(os.path.join(OUT, "features.csv"))
    for d in DEGS:
        if f"deg{d}" not in t:
            t[f"deg{d}"] = np.nan
    res = {}
    for src in ("network", "realsense"):
        tr = t[(t.split == "train") & (t.walls == src)].reset_index(drop=True)
        for split in ("val", "test"):
            te = t[(t.split == split) & (t.walls == src)].reset_index(drop=True)
            p = fit_predict(tr, te)
            sc = scores(te, p, meta["t1"], meta["t2"])
            sc.to_csv(os.path.join(OUT, f"scores_{split}_{src}.csv"), index=False)
            res[(split, src)] = (te, p, sc)
            print(f"{split} {src}\n" + sc.pivot_table(index="rule", columns="fill", values="mae").round(2).to_string())
    sv = res[("val", "network")][2]
    rank = sv[sv.rule.isin(["clf", "err"])].groupby("rule").mae.mean().sort_values()
    chosen = rank.index[0]
    te, p, st = res[("test", "network")]
    g = te["fill"].to_numpy() == 0.5
    d = np.abs(p[chosen][g] - te["gt"].to_numpy()[g]) - np.abs(p["cascade"][g] - te["gt"].to_numpy()[g])
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(5000)])
    cell = lambda r, f: st[(st.rule == r) & (st.fill == f)].iloc[0]  # noqa: E731
    v = {"val_rank": rank.round(3).to_dict(), "chosen_on_val": chosen,
         "test_50": {"cascade": cell("cascade", 0.5).mae, "chosen": cell(chosen, 0.5).mae, "diff": float(d.mean()),
                     "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]},
         "test_25_change": float(cell(chosen, 0.25).mae - cell("cascade", 0.25).mae),
         "test_75_change": float(cell(chosen, 0.75).mae - cell("cascade", 0.75).mae),
         "test_50_under": [int(cell("cascade", 0.5).under), int(cell(chosen, 0.5).under)]}
    v["adopt"] = bool(v["test_50"]["ci95"][1] < 0 and v["test_25_change"] <= 0.2 and v["test_75_change"] <= 0.2
                      and v["test_50_under"][1] <= v["test_50_under"][0])
    with open(os.path.join(OUT, "verdict.json"), "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(v, indent=2))


def valtrain():
    """The same learners trained on `val`, whose network walls the network never trained on (the
    train split's walls are cleaner than any unseen photo's). The learner is chosen by 5-fold
    cross-validation inside val, grouped by pothole; then it is fitted on all of val and scored once
    on test with the same adoption rule. Network walls only (the served case)."""
    meta = json.load(open(os.path.join(E.CACHE, "meta.json")))
    t = pd.read_csv(os.path.join(OUT, "features.csv"))
    for d in DEGS:
        if f"deg{d}" not in t:
            t[f"deg{d}"] = np.nan
    va = t[(t.split == "val") & (t.walls == "network")].reset_index(drop=True)
    te = t[(t.split == "test") & (t.walls == "network")].reset_index(drop=True)
    folds = va["i"].to_numpy() % 5
    cvp = {k: np.zeros(len(va)) for k in ("clf", "err", "cascade")}
    for k in range(5):
        tr, ho = va[folds != k].reset_index(drop=True), va[folds == k].reset_index(drop=True)
        p = fit_predict(tr, ho)
        for name in cvp:
            cvp[name][folds == k] = p[name]
    cvs = scores(va, cvp, meta["t1"], meta["t2"])
    print("val, 5-fold CV:\n" + cvs.pivot_table(index="rule", columns="fill", values="mae").round(2).to_string())
    rank = cvs[cvs.rule.isin(["clf", "err"])].groupby("rule").mae.mean().sort_values()
    chosen = rank.index[0]
    p = fit_predict(va, te)
    st = scores(te, p, meta["t1"], meta["t2"])
    print("test:\n" + st.pivot_table(index="rule", columns="fill", values="mae").round(2).to_string())
    g = te["fill"].to_numpy() == 0.5
    d = np.abs(p[chosen][g] - te["gt"].to_numpy()[g]) - np.abs(p["cascade"][g] - te["gt"].to_numpy()[g])
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(5000)])
    cell = lambda r, f: st[(st.rule == r) & (st.fill == f)].iloc[0]  # noqa: E731
    v = {"val_cv_rank": rank.round(3).to_dict(), "chosen": chosen,
         "test_50": {"cascade": cell("cascade", 0.5).mae, "chosen": cell(chosen, 0.5).mae, "diff": float(d.mean()),
                     "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]},
         "test_25_change": float(cell(chosen, 0.25).mae - cell("cascade", 0.25).mae),
         "test_75_change": float(cell(chosen, 0.75).mae - cell("cascade", 0.75).mae),
         "test_50_under": [int(cell("cascade", 0.5).under), int(cell(chosen, 0.5).under)]}
    v["adopt"] = bool(v["test_50"]["ci95"][1] < 0 and v["test_25_change"] <= 0.2 and v["test_75_change"] <= 0.2
                      and v["test_50_under"][1] <= v["test_50_under"][0])
    with open(os.path.join(OUT, "verdict_valtrain.json"), "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    {"features": features, "run": run, "valtrain": valtrain}[sys.argv[1]]()
