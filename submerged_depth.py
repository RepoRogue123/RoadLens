"""
Depth of a pothole whose floor is hidden under water.

Water is level, so a depth map of a water-filled pothole shows the water surface: the reading
stops at the waterline and the pothole looks shallower than it is. What stays visible is the
dry wall above the waterline and the road around the rim. This module fits a bowl to those
and extrapolates it under the water (the metric form of tier 1 in api.compute_depth_slice,
fitted over the whole visible wall instead of one cross-section).

The bowl is fitted by RANSAC (scripts/eval_ransac_bowl.py): a few wall points are wrong (glare
at the waterline, gravel, depth-sensor holes) and pull a least-squares fit off, and a quadric is
too stiff for most walls. A cascade tries a cubic bowl, then degree 4, then degree 2, then the
least-squares quadric, and only falls back to the water-surface reading when none of them is a
bowl (scripts/eval_bowl_degree.py). On the controlled-submersion benchmark (207 test potholes,
half full, network walls) the cubic alone lowers the error from 11.07 to 9.04 mm and the cascade
to 8.89 mm, with no fall-backs instead of 27. method="quadric" keeps the earlier fit.

All maps are relief in mm: depth below the road plane, positive downwards.
"""
from typing import Dict, Optional

import cv2
import numpy as np

ANCHOR_PX = 5          # road kept around the rim as anchor points (as in compute_depth_slice)
MIN_WALL_PX = 30       # fewer visible wall pixels than this: no fit
MAX_DEPTH_MM = 300.0   # same limit the training target uses for a plausible relief

# RANSAC bowl, settings chosen on the validation split (scripts/eval_ransac_bowl.py)
CASCADE = (3, 4, 2)    # RANSAC bowl degrees, tried in this order, then the least-squares quadric
INLIER_MM = 8.0        # a wall point agrees with a bowl within this many mm
N_HYP = 300            # random minimal samples
TOP_K = 10             # best hypotheses refitted on their inliers and checked for a bowl
N_SCORE = 2000         # points used to score a hypothesis
N_REFIT = 6000         # support points used for the refits
SEED = 0               # fixed, so a photo always gets the same answer


def _design(xs, ys, cx, cy, s):
    x, y = (xs - cx) / s, (ys - cy) / s
    return np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])


def _poly(x, y, deg):
    """All monomials x^i y^j with i + j <= deg (for deg 2 in the order of _design)."""
    if deg == 2:
        return np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    return np.column_stack([x ** i * y ** (k - i) for k in range(deg, -1, -1) for i in range(k, -1, -1)])


def _quad_bowl(c):
    """A quadric is a bowl when its quadratic part is negative definite (relief grows downwards)."""
    hxx, hyy, hxy = 2 * c[..., 0], 2 * c[..., 1], c[..., 2]
    return (hxx < 0) & (hyy < 0) & (hxx * hyy - hxy * hxy > 0)


def _is_bowl(f, wx, wy, edge, wpix, level, s):
    """The fitted surface must peak strictly inside the water (not on its edge), below the
    waterline, with a negative-definite Hessian there: a bowl, not a slope, saddle or ridge."""
    if wx.size < 5:
        return False
    v = f(wx, wy)
    i = int(np.argmax(v))
    if edge[wpix[0][i], wpix[1][i]] or v[i] <= level + 0.5:
        return False
    e = 2.0 / s                                              # numeric Hessian, 2-pixel steps
    x0, y0 = wx[i], wy[i]
    fxx = (f(x0 + e, y0) - 2 * f(x0, y0) + f(x0 - e, y0)) / e ** 2
    fyy = (f(x0, y0 + e) - 2 * f(x0, y0) + f(x0, y0 - e)) / e ** 2
    fxy = (f(x0 + e, y0 + e) - f(x0 + e, y0 - e) - f(x0 - e, y0 + e) + f(x0 - e, y0 - e)) / (4 * e * e)
    return bool(fxx < 0 and fyy < 0 and fxx * fyy - fxy * fxy > 0)


def _ransac_bowl(px, py, z, wx, wy, edge, wpix, level, s, deg):
    """Bowl of degree `deg` by RANSAC; returns f(x, y) in normalised coordinates, or None."""
    rng = np.random.default_rng(SEED)
    if z.size > N_REFIT:
        keep = np.sort(np.random.default_rng(0).permutation(z.size)[:N_REFIT])
        px, py, z = px[keep], py[keep], z[keep]
    a = _poly(px, py, deg)
    p = a.shape[1]
    if a.shape[0] < p * 3:
        return None
    sc = rng.choice(a.shape[0], min(N_SCORE, a.shape[0]), replace=False)
    idx = np.stack([rng.choice(a.shape[0], p, replace=False) for _ in range(N_HYP)])
    A = a[idx] + 1e-9 * np.eye(p)
    try:
        C = np.linalg.solve(A, z[idx][..., None])[..., 0]
    except np.linalg.LinAlgError:
        C = np.stack([np.linalg.lstsq(A[i], z[idx[i]], rcond=None)[0] for i in range(N_HYP)])
    if deg == 2:                                         # cheap exact screen for quadrics
        C = C[_quad_bowl(C)]
        if not len(C):
            return None
    order = np.argsort(-(np.abs(C @ a[sc].T - z[sc]) < INLIER_MM).sum(1))
    best = None
    for h in order[:TOP_K]:
        inl = np.abs(a @ C[h] - z) < INLIER_MM
        if inl.sum() < p * 3:
            continue
        coef, *_ = np.linalg.lstsq(a[inl], z[inl], rcond=None)
        inl = np.abs(a @ coef - z) < INLIER_MM                # one re-estimation of the inliers
        if inl.sum() < p * 3:
            continue
        coef, *_ = np.linalg.lstsq(a[inl], z[inl], rcond=None)
        f = (lambda c: lambda x, y: _poly(np.atleast_1d(x), np.atleast_1d(y), deg) @ c)(coef)
        if not (_quad_bowl(coef) if deg == 2 else _is_bowl(f, wx, wy, edge, wpix, level, s)):
            continue
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), f)
    return None if best is None else best[1]


def recover(relief: np.ndarray, mask: np.ndarray, water: np.ndarray,
            valid: Optional[np.ndarray] = None, method: str = "ransac") -> Dict[str, object]:
    """
    Extrapolate the hidden floor of a pothole from its dry walls.

    relief: HxW relief map (mm, positive below the road); mask: pothole outline;
    water: pixels whose surface is water. Returns the water-surface reading (a lower bound),
    the recovered reading and the filled map, or `abstained` with the reason when the
    visible wall cannot support a bowl; `bowl` names the fit used. method: "ransac" (the
    cascade) or "quadric" (least squares only, the earlier fit).
    """
    m = np.asarray(mask) > 0
    w = (np.asarray(water) > 0) & m
    ok = np.isfinite(relief) if valid is None else (np.asarray(valid) > 0) & np.isfinite(relief)
    observed = float(np.percentile(relief[m & ok], 90)) if (m & ok).any() else None
    out = {"surface_mm": observed, "recovered_mm": observed, "abstained": None,
           "water_share": float(w.sum() / max(1, m.sum())), "filled": None, "bowl": None}
    if not w.any():
        out["abstained"] = "no water"
        return out

    k = np.ones((2 * ANCHOR_PX + 1, 2 * ANCHOR_PX + 1), np.uint8)
    anchor = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~m
    walls = m & ~w
    pts = (walls | anchor) & ok
    if int((walls & ok).sum()) < MIN_WALL_PX:
        out["abstained"] = "too little visible wall"
        return out

    ys, xs = np.nonzero(pts)
    cy, cx = float(np.mean(np.nonzero(m)[0])), float(np.mean(np.nonzero(m)[1]))
    s = float(max(np.ptp(xs), np.ptp(ys), 1))
    wy, wx = np.nonzero(w)
    fit = None
    if method != "quadric":
        edge = w & ~(cv2.erode(w.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        nwx, nwy = (wx - cx) / s, (wy - cy) / s
        args = ((xs - cx) / s, (ys - cy) / s, relief[ys, xs].astype(np.float64), nwx, nwy, edge,
                (wy, wx), float(np.median(relief[w])), s)
        for deg in CASCADE:
            f = _ransac_bowl(*args, deg)
            if f is not None:
                fit, out["bowl"] = f(nwx, nwy), f"ransac degree {deg}"
                break
    if fit is None:
        a = _design(xs.astype(float), ys.astype(float), cx, cy, s)
        coef, *_ = np.linalg.lstsq(a, relief[ys, xs].astype(float), rcond=None)
        # relief grows downwards, so a bowl has its maximum inside: the quadratic part must be
        # negative definite. Anything else (flat, saddle, dome) is not a bowl and is not extended.
        if not _quad_bowl(coef):
            out["abstained"] = "visible wall is not bowl-shaped"
            return out
        fit, out["bowl"] = _design(wx.astype(float), wy.astype(float), cx, cy, s) @ coef, "least-squares quadric"

    filled = relief.astype(np.float64).copy()
    filled[wy, wx] = np.clip(np.maximum(filled[wy, wx], fit), 0, MAX_DEPTH_MM)
    out["filled"] = filled
    out["recovered_mm"] = float(np.percentile(filled[m & (ok | w)], 90))
    return out
