import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Production segmenter, first found wins, each with the confidence threshold it was
# evaluated at. best_v2.pt adds Pothole-600 and PothRGBD outlines to the Kaggle set
# (scripts/build_yolo_seg_v2.py, 25 Sep 2026): on held-out PothRGBD sessions it finds
# 97% of measured potholes against 73%. Its threshold, 0.35, was chosen on the
# validation split to hold false masks near the old model's rate.
# best_v3.pt (scripts/build_yolo_seg_v3.py, 3 Oct 2026) adds RDD dashcam and Mendeley phone
# photos outlined by SAM 2 from their boxes, plus 2,400 RDD frames with no pothole. Same
# threshold; better on every test set, and on Japan's dashcam frames (never seen) it finds
# 22% of pothole boxes against 13% and marks 7% of pothole-free frames against 41%.
# ROADLENS_SEG_WEIGHTS=<path> overrides (threshold 0.25 unless it is a listed file).
MODEL_CANDIDATES = [
    (os.path.join(SCRIPT_DIR, "yolo-segmentation", "model", "best_v3.pt"), 0.35),
    (os.path.join(SCRIPT_DIR, "yolo-segmentation", "model", "best_v2.pt"), 0.35),
    (os.path.join(SCRIPT_DIR, "yolo-segmentation", "model", "best.pt"), 0.25),
    (os.path.join(SCRIPT_DIR, "model", "best.pt"), 0.25),
]
_override = os.getenv("ROADLENS_SEG_WEIGHTS")
if _override:
    _known = {os.path.abspath(p): c for p, c in MODEL_CANDIDATES}
    MODEL_CANDIDATES = [(_override, _known.get(os.path.abspath(_override), 0.25))]

MODEL_PATH, CONF_THRESHOLD = next(
    ((path, conf) for path, conf in MODEL_CANDIDATES if os.path.isfile(path)), (None, None))
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find segmentation weights. Expected one of: "
        + ", ".join(p for p, _c in MODEL_CANDIDATES)
    )

# Load YOLOv8 segmentation model once and reuse it for inference calls.
MODEL = YOLO(MODEL_PATH)


def _extract_binary_masks(
    image: np.ndarray,
    model: YOLO,
    conf_threshold: Optional[float] = None,
    min_area: int = 100,
) -> List[np.ndarray]:
    """Return all pothole masks sorted by descending area.

    conf_threshold=None uses the production threshold, which only makes sense for the
    production MODEL; pass an explicit value when evaluating other weights.
    """
    if conf_threshold is None:
        conf_threshold = CONF_THRESHOLD
    height, width = image.shape[:2]
    results = model.predict(source=image, imgsz=640, conf=conf_threshold, verbose=False)
    result = results[0]

    if result.masks is None or len(result.masks.data) == 0:
        return []

    masks = result.masks.data.detach().cpu().numpy()
    binary_masks = (masks > 0.5).astype(np.uint8)

    filtered_masks = []
    for m in binary_masks:
        if m.shape != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
            m = (m > 0).astype(np.uint8)

        area = int(m.sum())
        if area >= min_area:
            filtered_masks.append((area, m))

    filtered_masks.sort(key=lambda x: x[0], reverse=True)
    return [mask for _, mask in filtered_masks]


def get_pothole_mask(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run YOLOv8 segmentation on an image and return the largest pothole mask.

    Args:
        image_path: Path to input image.

    Returns:
        A tuple of (binary_mask, original_image) where:
        - binary_mask is an HxW np.uint8 array containing values {0, 1}
        - original_image is the loaded image in OpenCV BGR format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    all_masks = _extract_binary_masks(image=image, model=MODEL)
    if not all_masks:
        return np.zeros((height, width), dtype=np.uint8), image

    largest_mask = all_masks[0]

    return largest_mask, image

def get_largest_mask(img_path):
    mask, _ = get_pothole_mask(img_path)
    return mask

def get_all_masks(
    image_path,
    model_path=None,
    conf_threshold=None,
    min_area=100,
):
    """All pothole masks for an image, largest first.

    Defaults to the production model at its own threshold. A different `model_path`
    builds a fresh YOLO on every call (no cache) and uses 0.25 unless told otherwise.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    model = MODEL
    if (model_path and os.path.exists(model_path)
            and os.path.abspath(model_path) != os.path.abspath(MODEL_PATH)):
        model = YOLO(model_path)
        if conf_threshold is None:
            conf_threshold = 0.25

    return _extract_binary_masks(
        image=image,
        model=model,
        conf_threshold=conf_threshold,
        min_area=min_area,
    )

def get_mask_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour
