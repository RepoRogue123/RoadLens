"""
Depth of a pothole whose floor is hidden under water.

Water is level, so a depth map of a water-filled pothole shows the water surface: the reading
stops at the waterline and the pothole looks shallower than it is. What stays visible is the
dry wall above the waterline and the road around the rim. This module fits a bowl to those
and extrapolates it under the water (the metric form of tier 1 in api.compute_depth_slice,
fitted over the whole visible wall instead of one cross-section).

All maps are relief in mm: depth below the road plane, positive downwards.
"""
from typing import Dict, Optional

import cv2
import numpy as np

ANCHOR_PX = 5          # road kept around the rim as anchor points (as in compute_depth_slice)
MIN_WALL_PX = 30       # fewer visible wall pixels than this: no fit
MAX_DEPTH_MM = 300.0   # same limit the training target uses for a plausible relief


def _design(xs, ys, cx, cy, s):
    x, y = (xs - cx) / s, (ys - cy) / s
    return np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])


def recover(relief: np.ndarray, mask: np.ndarray, water: np.ndarray,
            valid: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Extrapolate the hidden floor of a pothole from its dry walls.

    relief: HxW relief map (mm, positive below the road); mask: pothole outline;
    water: pixels whose surface is water. Returns the water-surface reading (a lower bound),
    the recovered reading and the filled map, or `abstained` with the reason when the
    visible wall cannot support a bowl.
    """
    m = np.asarray(mask) > 0
    w = (np.asarray(water) > 0) & m
    ok = np.isfinite(relief) if valid is None else (np.asarray(valid) > 0) & np.isfinite(relief)
    observed = float(np.percentile(relief[m & ok], 90)) if (m & ok).any() else None
    out = {"surface_mm": observed, "recovered_mm": observed, "abstained": None,
           "water_share": float(w.sum() / max(1, m.sum())), "filled": None}
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
    a = _design(xs.astype(float), ys.astype(float), cx, cy, s)
    coef, *_ = np.linalg.lstsq(a, relief[ys, xs].astype(float), rcond=None)
    # relief grows downwards, so a bowl has its maximum inside: the quadratic part must be
    # negative definite. Anything else (flat, saddle, dome) is not a bowl and is not extended.
    hxx, hyy, hxy = 2 * coef[0], 2 * coef[1], coef[2]
    if not (hxx < 0 and hyy < 0 and hxx * hyy - hxy * hxy > 0):
        out["abstained"] = "visible wall is not bowl-shaped"
        return out

    wy, wx = np.nonzero(w)
    fit = _design(wx.astype(float), wy.astype(float), cx, cy, s) @ coef
    filled = relief.astype(np.float64).copy()
    filled[wy, wx] = np.clip(np.maximum(filled[wy, wx], fit), 0, MAX_DEPTH_MM)
    out["filled"] = filled
    out["recovered_mm"] = float(np.percentile(filled[m & (ok | w)], 90))
    return out
