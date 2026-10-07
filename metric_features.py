"""
Metric-preserving pothole features from MoGe geometry.

Why this module exists
----------------------
`features.py` min-max normalises every depth map to [0, 1] before measuring
anything. That is harmless for Depth-Anything-V2, whose output has no units
anyway, but it throws away the one thing MoGe offers: millimetres. Feeding MoGe
through the old extractors produced a "MoGe" feature set that scored within
0.3 mm of the Depth-Anything one (MASTER_DOC Part 12), despite MoGe correlating
0.545 with measured bowl depth against Depth-Anything's 0.019.

Everything here keeps physical units:

* bowl depth below a least-squares road plane, fitted to a trimmed ring of road
  around the mask — the same protocol that turns RealSense depth into the
  PothRGBD labels, applied to the estimated depth instead;
* metric area and equivalent diameter, from 3-D points rebuilt out of depth and
  MoGe's normalised intrinsics (points = depth x inverse(K) x pixel);
* camera distance, and scale-free ratios (bowl depth over camera distance,
  bowl depth over diameter). A monocular model's global scale can be off by tens
  of percent on a single frame; ratios cancel that error, raw millimetres do not.

The same function is used to build the training set and at inference, so the
served model sees exactly the features it was trained on.
"""
from typing import Dict, Optional

import cv2
import numpy as np

# Road-plane protocol — mirrors scripts/pothrgbd_metric_labels.py so that an
# estimated bowl depth means the same thing as a measured one.
RING_KERNEL = 25
MIN_RING_PX = 400
MIN_INSIDE_PX = 150

FEATURE_COLS = [
    "mf_p50_mm", "mf_p75_mm", "mf_p90_mm", "mf_p95_mm", "mf_max_mm", "mf_mean_mm",
    "mf_frac_gt10", "mf_frac_gt25",
    "mf_plane_rms_mm", "mf_cam_dist_mm",
    "mf_area_cm2", "mf_log_area_cm2", "mf_diam_mm",
    "mf_p90_over_dist", "mf_p90_over_diam",
    "mf_normal_dev_mean", "mf_normal_dev_p90",
    "mf_valid_frac",
]

SHAPE_COLS = ["sh_log_area_px", "sh_solidity", "sh_compactness", "sh_elongation",
              "cv_mean_curvature", "cv_p90_curvature", "cv_curvature_sign_changes"]

# PothRGBD frames are 640 px on their long side. The road ring and every pixel
# count are defined at that scale, so inference must measure at the same scale.
TRAIN_LONG_SIDE = 640


def to_train_scale(image: np.ndarray, masks):
    """Resize an image and its masks so the long side matches the training frames."""
    h, w = image.shape[:2]
    s = TRAIN_LONG_SIDE / max(h, w)
    if abs(s - 1.0) < 1e-3:
        return image, [(np.asarray(m) > 0).astype(np.uint8) for m in masks]
    size = (max(1, round(w * s)), max(1, round(h * s)))
    img = cv2.resize(image, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    ms = [(cv2.resize((np.asarray(m) > 0).astype(np.uint8), size,
                      interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) for m in masks]
    return img, ms


def shape_features(mask: np.ndarray) -> Optional[Dict[str, float]]:
    """Depth-free shape columns, computed identically at train and serve time."""
    from features import extract_curvature_features
    m = (np.asarray(mask) > 0).astype(np.uint8)
    area = float(m.sum())
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    per = cv2.arcLength(c, True)
    _x, _y, w, h = cv2.boundingRect(c)
    cv = extract_curvature_features(m) or {}
    return {
        "sh_log_area_px": float(np.log(max(area, 1.0))),
        "sh_solidity": area / hull if hull > 0 else 0.0,
        "sh_compactness": 4 * np.pi * area / (per ** 2) if per > 0 else 0.0,
        "sh_elongation": max(w, h) / max(min(w, h), 1),
        "cv_mean_curvature": float(cv.get("mean_curvature", 0.0)),
        "cv_p90_curvature": float(cv.get("p90_curvature", 0.0)),
        "cv_curvature_sign_changes": float(cv.get("curvature_sign_changes", 0.0)),
    }


def points_from_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """HxWx3 camera-space points from z-depth and MoGe's normalised intrinsics."""
    h, w = depth.shape
    k = np.asarray(intrinsics, dtype=np.float64)
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    uu, vv = np.meshgrid(u, v)
    x = depth * (uu - k[0, 2]) / k[0, 0]
    y = depth * (vv - k[1, 2]) / k[1, 1]
    return np.stack([x, y, depth], axis=-1)


def _pixel_areas(points: np.ndarray) -> np.ndarray:
    """Surface area each pixel covers, from the cross product of its neighbour steps."""
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, :-1] = points[:, 1:] - points[:, :-1]
    dy[:-1, :] = points[1:, :] - points[:-1, :]
    return np.linalg.norm(np.cross(dx, dy), axis=-1)


def road_plane_deviation(depth_mm: np.ndarray, mask: np.ndarray, valid: np.ndarray):
    """
    Per-pixel deviation from a plane fitted to the trimmed road ring.

    Positive means deeper than the road. Returns (deviation, ring_mask, rms) or
    None when there is not enough road to fit.
    """
    h, w = mask.shape
    ring = cv2.dilate(mask, np.ones((RING_KERNEL, RING_KERNEL), np.uint8), iterations=2)
    ring = (ring > 0) & (mask == 0) & valid
    if int(ring.sum()) < MIN_RING_PX:
        return None
    ys, xs = np.nonzero(ring)
    zs = depth_mm[ys, xs].astype(np.float64)
    lo, hi = np.percentile(zs, [5, 95])
    keep = (zs >= lo) & (zs <= hi)
    if keep.sum() < MIN_RING_PX // 2:
        return None
    ys, xs, zs = ys[keep], xs[keep], zs[keep]
    a = np.column_stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
    try:
        coef, *_ = np.linalg.lstsq(a, zs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    rms = float(np.sqrt(np.mean((zs - a @ coef) ** 2)))
    gy, gx = np.mgrid[0:h, 0:w]
    plane = coef[0] * gx + coef[1] * gy + coef[2]
    return depth_mm.astype(np.float64) - plane, ring, rms


def extract_metric_features(mask: np.ndarray, depth_m: np.ndarray,
                            intrinsics: np.ndarray,
                            normal: Optional[np.ndarray] = None,
                            valid: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
    """
    Metric features for one pothole mask from one MoGe prediction.

    Args:
        mask: HxW {0,1} pothole mask at the depth map's resolution.
        depth_m: HxW z-depth in metres (MoGe `depth`).
        intrinsics: 3x3 normalised intrinsics (MoGe `intrinsics`).
        normal: optional HxWx3 unit normals (MoGe `normal`).
        valid: optional HxW validity mask (MoGe `mask`).

    Returns a dict keyed by FEATURE_COLS, or None when the road plane cannot be
    fitted or too little of the pothole has valid depth.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    ok = np.isfinite(depth_m) & (depth_m > 0)
    if valid is not None:
        ok &= np.asarray(valid) > 0.5
    depth_mm = depth_m * 1000.0

    res = road_plane_deviation(depth_mm, mask, ok)
    if res is None:
        return None
    dev, ring, rms = res
    inside_px = (mask > 0) & ok
    n_mask = int(mask.sum())
    if int(inside_px.sum()) < MIN_INSIDE_PX:
        return None
    d = dev[inside_px]

    pts = points_from_depth(np.where(ok, depth_m, 0.0), intrinsics)
    areas = _pixel_areas(pts)
    area_m2 = float(areas[inside_px].sum()) * (n_mask / inside_px.sum())
    area_cm2 = area_m2 * 1e4
    diam_mm = 2.0 * np.sqrt(area_m2 / np.pi) * 1000.0
    cam_dist_mm = float(np.median(depth_mm[ring]))

    p50, p75, p90, p95, pmax = np.percentile(d, [50, 75, 90, 95, 99])
    out = {
        "mf_p50_mm": p50, "mf_p75_mm": p75, "mf_p90_mm": p90, "mf_p95_mm": p95,
        "mf_max_mm": pmax, "mf_mean_mm": float(d.mean()),
        "mf_frac_gt10": float((d > 10).mean()), "mf_frac_gt25": float((d > 25).mean()),
        "mf_plane_rms_mm": rms, "mf_cam_dist_mm": cam_dist_mm,
        "mf_area_cm2": area_cm2, "mf_log_area_cm2": float(np.log(max(area_cm2, 1e-3))),
        "mf_diam_mm": diam_mm,
        "mf_p90_over_dist": p90 / max(cam_dist_mm, 1.0),
        "mf_p90_over_diam": p90 / max(diam_mm, 1.0),
        "mf_normal_dev_mean": 0.0, "mf_normal_dev_p90": 0.0,
        "mf_valid_frac": float(inside_px.sum() / max(n_mask, 1)),
    }

    if normal is not None:
        n = np.asarray(normal, dtype=np.float64)
        ref = n[ring].mean(axis=0)
        norm = np.linalg.norm(ref)
        if norm > 1e-9:
            ref /= norm
            cos = np.clip(n[inside_px] @ ref, -1.0, 1.0)
            ang = np.degrees(np.arccos(cos))
            out["mf_normal_dev_mean"] = float(ang.mean())
            out["mf_normal_dev_p90"] = float(np.percentile(ang, 90))

    return {k: float(v) for k, v in out.items()}
