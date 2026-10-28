"""
Regularised polynomial bowls. A high-degree polynomial fits the visible wall well and swings under
the water; regularisation keeps the flexibility and damps the swing. Every variant fits the RANSAC
inliers of the cubic bowl (so bad points are already out), checks the result is a bowl, and
otherwise answers as the cascade in submerged_depth.recover does.

  lsq      plain least squares on the inliers, degree 3..6 (lambda = 0)
  ridge    L2 on every coefficient except the constant (columns standardised)
  toquad   L2 on the terms of order >= 3 only: large lambda gives back the quadric, small lambda
           the full polynomial, so it slides from "stiff bowl" to "free shape"
  lasso    L1 on every coefficient except the constant (keeps the terms that matter)
  gcv      lambda per pothole by generalised cross-validation (ridge and toquad)
  band     lambda per pothole by predicting the held-out wall strip at the waterline (toquad)

Protocol, fixed before results: the variant with the lowest mean network-wall MAE over the three
fills on val is scored once on test, and adopted over the cascade only if the 50 %-fill
network MAE drops with a paired-bootstrap 95 % interval above zero, the 25 % and 75 % network MAEs
do not worsen by more than 0.2 mm, and 50 % under-reporting does not rise.

    python scripts/eval_bowl_regularized.py
"""
import json
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2                                                                    # noqa: E402
import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ransac_bowl as E                                                  # noqa: E402
from eval_bowl_degree import reading                                         # noqa: E402

OUT = os.path.join(E.OUT_DIR, "regularized")
TOL = 8.0
DEGS = (3, 4, 5, 6)
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LASSO = (0.03, 0.1, 0.3, 1.0)
cv2.setNumThreads(1)


def poly_terms(deg):
    """Total order of every column of E.design(..., deg)."""
    if deg == 2:
        return np.array([2, 2, 2, 1, 1, 0])
    return np.array([k for k in range(deg, -1, -1) for _ in range(k, -1, -1)])


class Fitter:
    """Standardised design on the inliers; closed-form penalised fits."""

    def __init__(self, ctx, inl, deg):
        self.deg = deg
        a = E.design(ctx.px[inl], ctx.py[inl], deg)
        order = poly_terms(deg)
        self.const = order == 0
        sc = np.sqrt(np.mean(a ** 2, 0))
        sc[self.const | (sc == 0)] = 1.0
        self.sc, self.order = sc, order
        self.a, self.z = a / sc, ctx.z[inl]
        self.ata, self.atz = self.a.T @ self.a, self.a.T @ self.z
        self.n = self.a.shape[0]

    def pen(self, kind):
        return (~self.const).astype(float) if kind == "ridge" else (self.order >= 3).astype(float)

    def solve(self, lam, kind, rows=None):
        a, z = (self.a, self.z) if rows is None else (self.a[rows], self.z[rows])
        m = a.T @ a + lam * a.shape[0] * np.diag(self.pen(kind))
        c = np.linalg.solve(m + 1e-10 * np.eye(m.shape[0]), a.T @ z)
        return c / self.sc

    def gcv(self, lam, kind):
        m = self.ata + lam * self.n * np.diag(self.pen(kind))
        minv_ata = np.linalg.solve(m + 1e-10 * np.eye(m.shape[0]), self.ata)
        c = np.linalg.solve(m + 1e-10 * np.eye(m.shape[0]), self.atz)
        rss = float(((self.a @ c - self.z) ** 2).sum())
        tr = float(np.trace(minv_ata))
        return self.n * rss / max(self.n - tr, 1.0) ** 2

    def lasso(self, alpha):
        from sklearn.linear_model import Lasso
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Lasso(alpha=alpha, max_iter=3000, tol=1e-3).fit(self.a[:, ~self.const], self.z)
        c = np.zeros(self.a.shape[1])
        c[~self.const] = m.coef_
        c[self.const] = m.intercept_
        return c / self.sc


def as_fn(coef, deg):
    return lambda x, y: E.design(np.atleast_1d(x), np.atleast_1d(y), deg) @ coef


def variants_for(ctx, relief, cascade):
    """All variant readings for one context. A non-bowl answer is replaced by the cascade's."""
    out = {}
    f3 = E.ransac(ctx, 3, TOL)
    if f3 is None:                                 # no robust cubic: inliers = every point
        inl = np.ones(ctx.z.size, bool)
    else:
        inl = np.abs(f3(ctx.px, ctx.py) - ctx.z) < TOL

    def put(name, coef, deg):
        f = as_fn(coef, deg)
        out[name] = reading(ctx, relief, f) if ctx.bowl_ok(f, deg, coef) else cascade

    for d in DEGS:
        if inl.sum() < 3 * E.design(ctx.px[:1], ctx.py[:1], d).shape[1]:
            continue
        fz = Fitter(ctx, inl, d)
        put(f"lsq d{d}", fz.solve(0.0, "ridge"), d)
        for kind in ("ridge", "toquad"):
            for lam in LAMS:
                put(f"{kind} d{d} lam {lam:g}", fz.solve(lam, kind), d)
            g = {lam: fz.gcv(lam, kind) for lam in LAMS}
            put(f"{kind} d{d} gcv", fz.solve(min(g, key=g.get), kind), d)
        band = ctx.band[inl]
        if band.sum() >= 20 and (~band).sum() >= 60:
            err = {}
            for lam in LAMS:
                c = fz.solve(lam, "toquad", rows=~band)
                err[lam] = float(np.median(np.abs(fz.a[band] @ (c * fz.sc) - fz.z[band])))
            put(f"toquad d{d} band", fz.solve(min(err, key=err.get), "toquad"), d)
        if d in (4, 6):
            for al in LASSO:
                put(f"lasso d{d} a {al:g}", fz.lasso(al), d)
    return out


def _one(args):
    import submerged_depth as S
    i, c = args
    dev, pdv, m, valid, gt = c["dev"].astype(np.float64), c["pdv"].astype(np.float64), c["mask"], c["valid"], float(c["gt"])
    rows = []
    for s in E.FILLS:
        obs, wm = E.submerge(dev, m, valid, (1 - s) * gt)
        ring = (cv2.dilate(wm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~wm & m
        pobs = pdv.copy()
        if wm.any():
            pobs[wm] = float(np.median(pdv[ring])) if ring.any() else float(np.median(pdv[wm]))
        for src, rel, val in (("realsense", obs, valid | wm), ("network", pobs, None)):
            cas = S.recover(rel, m, wm, val)["recovered_mm"]
            r = {"i": i, "fill": s, "walls": src, "gt": gt, "cascade": cas}
            ok = np.isfinite(rel) if val is None else val & np.isfinite(rel)
            if wm.any():
                ctx = E.Ctx(rel, m, wm & m, ok)
                if ctx.n_wall >= E.S.MIN_WALL_PX:
                    r.update(variants_for(ctx, rel, cas))
            rows.append(r)
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
    t = pd.DataFrame(rows)
    names = [c for c in t.columns if c not in ("i", "fill", "walls", "gt")]
    t[names] = t[names].apply(lambda col: col.fillna(t["cascade"]))   # no water / no wall: the cascade's answer
    return t, names


def score(t, names, t1, t2):
    def bands(x, a, b):
        x = np.asarray(x, float)
        return np.where(x >= b, 2, np.where(x >= a, 1, 0))
    rows = []
    for (fill, src), g in t.groupby(["fill", "walls"]):
        yb = bands(g["gt"], 25, 50)
        for n in names:
            p = g[n].to_numpy(float)
            rows.append({"variant": n, "fill": fill, "walls": src, "mae": float(np.mean(np.abs(p - g["gt"]))),
                         "under": int((bands(p, t1, t2) < yb).sum()),
                         "deep_found": int(((bands(p, t1, t2) == 2) & (yb == 2)).sum())})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(E.CACHE, "meta.json")))
    res = {}
    for split in ("val", "test"):
        t, names = run_split(split)
        t.to_csv(os.path.join(OUT, f"per_pothole_{split}.csv"), index=False)
        sc = score(t, names, meta["t1"], meta["t2"])
        sc.to_csv(os.path.join(OUT, f"scores_{split}.csv"), index=False)
        res[split] = (t, sc)
    sv = res["val"][1]
    rank = sv[sv.walls == "network"].groupby("variant").mae.mean().sort_values()
    chosen = rank.index[0]
    t, st = res["test"]
    cell = lambda v, f: st[(st.variant == v) & (st.fill == f) & (st.walls == "network")].iloc[0]  # noqa: E731
    g = t[(t.fill == 0.5) & (t.walls == "network")]
    d = (np.abs(g[chosen] - g["gt"]) - np.abs(g["cascade"] - g["gt"])).to_numpy()
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(5000)])
    v = {"val_top10": rank.head(10).round(3).to_dict(), "val_cascade": float(rank["cascade"]), "chosen_on_val": chosen,
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


if __name__ == "__main__":
    main()
