"""
SAM 2 mask refinement — better boundaries for everything downstream.

Why this module exists
----------------------
YOLOv8-seg produces masks from low-resolution prototypes that are then upsampled,
so the boundary carries interpolation staircase. Four separate things consume
that boundary:

  1. features.extract_curvature_features — curvature is a SECOND derivative, so
     boundary noise is amplified, not attenuated
  2. foundation_features — the DINOv2 interior/ring split thresholds a 14x14
     downsample of the mask
  3. water_detection._get_road_surround_mask — the road ring is a dilation of it
  4. every road-plane fit in the PothRGBD work

One improvement upstream therefore improves four modules at once. This is also
the first time the improvement can be VERIFIED rather than assumed: PothRGBD
gives real millimetre depth, so a better mask should measurably improve the
road-plane fit and the correlation of geometry features with true depth.

Licence: Apache 2.0 for both code and weights — the cleanest licence in the
project, unlike Intrinsic (academic only) or AGSENet (no licence file).

Prompting
---------
Box plus centroid point. The coarse mask is NOT passed as `mask_input`: SAM 2
expects that in 256x256 logit space rather than as a binary mask, and the
box-plus-point prompt already identifies the object unambiguously here — there
is exactly one pothole, and we know where it is. Passing a wrongly-scaled
mask_input would degrade the result silently, which is worse than not using it.

Safety gate
-----------
SAM 2 is perfectly capable of returning "the whole road" when prompted at a
pothole, and a silent swap of a pothole mask for a road mask would corrupt every
downstream feature invisibly. `refine_mask` therefore REJECTS its own output and
returns the original whenever the refinement disagrees too much with the coarse
mask — see MIN_IOU and MAX_AREA_RATIO.

Usage:
    from mask_refine import HAS_SAM2, refine_mask
    better = refine_mask(image_rgb, coarse_mask)   # falls back to coarse_mask
"""
import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _HAS_DEPS = True
except Exception:
    _HAS_DEPS = False

HAS_SAM2 = _HAS_DEPS  # public flag for callers

# hiera_small: 46M parameters, ~85 FPS. Large is 224M for roughly +1 J&F, which
# is not worth 2x the latency on a per-pothole call.
MODEL_NAME = os.environ.get("SAM2_MODEL", "facebook/sam2-hiera-small")

# ── Safety gate ──────────────────────────────────────────────────────────
# A refinement that agrees with the coarse mask on less than MIN_IOU of their
# union is not a refinement, it is a different object. Likewise an area change
# beyond MAX_AREA_RATIO means SAM latched onto the road or a shadow.
MIN_IOU = 0.5
MAX_AREA_RATIO = 2.5
MIN_AREA_RATIO = 0.4
MIN_MASK_PX = 100

_PREDICTOR = None
_STATS = {"calls": 0, "refined": 0, "rejected": 0, "failed": 0}


def _load_sam2() -> Optional[Any]:
    """Lazy-load SAM 2. Returns None if unavailable, never raises."""
    global _PREDICTOR
    if not HAS_SAM2:
        return None
    if _PREDICTOR is not None:
        return _PREDICTOR
    try:
        _PREDICTOR = SAM2ImagePredictor.from_pretrained(MODEL_NAME)
        return _PREDICTOR
    except Exception as e:
        print(f"  !! SAM 2 loading failed: {type(e).__name__}: {str(e)[:120]}")
        return None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _prompt_from_mask(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Bounding box and an interior centroid point for the coarse mask."""
    ys, xs = np.nonzero(mask > 0)
    if ys.size < MIN_MASK_PX:
        return None
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    # Use the distance-transform peak rather than the raw centroid: for a
    # crescent or ring-shaped mask the centroid can land OUTSIDE the region,
    # which would prompt SAM 2 with a background point.
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(int(np.argmax(dt)), dt.shape)
    point = np.array([[px, py]], dtype=np.float32)
    return box, point


def refine_mask(image_rgb: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
    """
    Sharpen a coarse YOLO mask with SAM 2.

    Returns a binary uint8 mask of the same shape. Falls back to `coarse_mask`
    whenever SAM 2 is unavailable, inference fails, or the safety gate rejects
    the result — so a caller can use this unconditionally.
    """
    _STATS["calls"] += 1
    predictor = _load_sam2()
    if predictor is None or coarse_mask is None:
        return coarse_mask

    prompt = _prompt_from_mask(coarse_mask)
    if prompt is None:
        return coarse_mask
    box, point = prompt

    try:
        with torch.inference_mode():
            predictor.set_image(image_rgb)
            masks, scores, _ = predictor.predict(
                point_coords=point,
                point_labels=np.array([1], dtype=np.int32),
                box=box[None, :],
                multimask_output=False,
            )
    except Exception as e:
        _STATS["failed"] += 1
        print(f"  !! SAM 2 inference failed: {type(e).__name__}: {str(e)[:100]}")
        return coarse_mask

    m = np.asarray(masks)
    if m.ndim == 4:
        m = m[0]
    if m.ndim == 3:
        m = m[int(np.argmax(np.asarray(scores).ravel()))] if m.shape[0] > 1 else m[0]
    refined = (m > 0).astype(np.uint8)

    if refined.shape != coarse_mask.shape[:2]:
        refined = cv2.resize(refined, (coarse_mask.shape[1], coarse_mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    # ── safety gate ──
    a_coarse = int((coarse_mask > 0).sum())
    a_refined = int(refined.sum())
    if a_refined < MIN_MASK_PX or a_coarse == 0:
        _STATS["rejected"] += 1
        return coarse_mask
    ratio = a_refined / a_coarse
    if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
        _STATS["rejected"] += 1
        return coarse_mask
    if _iou(refined, coarse_mask) < MIN_IOU:
        _STATS["rejected"] += 1
        return coarse_mask

    _STATS["refined"] += 1
    return refined


def stats() -> dict:
    """Refinement counters — how often the safety gate actually fires."""
    return dict(_STATS)


def describe() -> str:
    """One-line status string, safe to log or surface in the API."""
    if not HAS_SAM2:
        return "SAM 2 unavailable (sam2/torch not importable)"
    state = "loaded" if _PREDICTOR is not None else "not yet loaded"
    return (f"SAM 2 {MODEL_NAME} ({state}); box+point prompt, "
            f"gate IoU>={MIN_IOU} area in [{MIN_AREA_RATIO}, {MAX_AREA_RATIO}]")


if __name__ == "__main__":
    import sys

    print(describe())
    if len(sys.argv) > 1:
        from segmentation import get_all_masks

        path = sys.argv[1]
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for i, cm in enumerate(get_all_masks(path), start=1):
            rm = refine_mask(rgb, cm)
            print(f"  p{i}: area {int((cm>0).sum()):>7} -> {int((rm>0).sum()):>7}  "
                  f"IoU {_iou(rm, cm):.3f}  "
                  f"{'refined' if rm is not cm else 'kept coarse'}")
        print(f"  stats: {stats()}")
