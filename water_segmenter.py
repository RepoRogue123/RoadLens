"""
Learned water segmenter: one YOLOv8n-seg forward pass per image.

Reported beside the cue ensemble; it decides only when
water_config.WATER_SEGMENTER_DECIDES (see water_config for the evidence and why not yet).
Follows the project's gate pattern: `HAS_WATER_SEGMENTER` is False when the
weights or ultralytics are missing, and `water_map` returns None instead of
raising, so detect_water falls back to the cue ensemble and says so.
"""
import os
from typing import Optional

import numpy as np

import water_config

_MODEL = None

try:
    from ultralytics import YOLO
    HAS_WATER_SEGMENTER = (water_config.ENABLE_WATER_SEGMENTER
                           and os.path.isfile(water_config.WATER_SEGMENTER_WEIGHTS))
except Exception:  # pragma: no cover - import guard
    HAS_WATER_SEGMENTER = False


def water_map(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """HxW bool map of pixels the segmenter marks as water, or None if unavailable."""
    global _MODEL
    if not HAS_WATER_SEGMENTER:
        return None
    try:
        from segmentation import _extract_binary_masks
        if _MODEL is None:
            _MODEL = YOLO(water_config.WATER_SEGMENTER_WEIGHTS)
        out = np.zeros(image_bgr.shape[:2], dtype=bool)
        for m in _extract_binary_masks(image_bgr, _MODEL,
                                       conf_threshold=water_config.WATER_SEGMENTER_CONF, min_area=50):
            out |= m > 0
        return out
    except Exception as e:
        print(f"water_segmenter: failed: {e}")
        return None


def describe() -> str:
    if not HAS_WATER_SEGMENTER:
        return "learned water segmenter OFF (weights missing or disabled)"
    return (f"learned water segmenter {os.path.basename(water_config.WATER_SEGMENTER_WEIGHTS)} "
            f"(HanYang held out: P 0.998 / R 0.925); "
            + ("decides: water if it covers >" f"{water_config.WATER_COVERAGE_THRESHOLD:.0%} of the pothole"
               if water_config.WATER_SEGMENTER_DECIDES else "reported only, the cue ensemble decides"))
