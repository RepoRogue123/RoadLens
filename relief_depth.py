"""
Relief network: per-pixel depth below the road surface, in millimetres.

Depth-Anything-V2 (ViT-S) fine-tuned on PothRGBD's RealSense depth maps
(scripts/train_relief_depth.py). Instead of scene depth it predicts RELIEF: how far
each pixel lies below the plane of the surrounding road, positive = deeper. That is
the quantity the severity labels are made of, and it removes the scale ambiguity that
makes raw monocular depth useless for potholes: the network never has to know how far
away the road is, only how the surface departs from flat.

Why this exists: the MoGe-feature regressor (metric_severity.py) learns from one
number per pothole, 1,051 in all, computed from MoGe's zero-shot depth. PothRGBD also
carries ~300,000 measured pixels per frame; this is the model that learns from them.
On capture sessions held out from it and from the segmenter, with the segmenter's own
outlines: bowl-depth error 8.70 mm against the regressor's 11.25 mm, 22 of 29 Deep
potholes found against 8 (ml_results/pothrgbd/relief_eval_ens23.json).

Every checkpoint matching ml_models/metric/relief_vits*.pth is loaded and their
predictions averaged (the served model is an average of two training runs).
Gate pattern as elsewhere: HAS_RELIEF is False without weights, and calls return None
rather than raise. Needs only torch and the Depth-Anything-V2 code — not MoGe.
"""
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "ml_models", "metric")
META_PATH = os.path.join(MODEL_DIR, "relief_meta.json")
BASE_CKPT = os.path.join(PROJECT_DIR, "Depth-Anything-V2", "checkpoints", "depth_anything_v2_vits.pth")

LONG_SIDE = 518                  # 37 patches of 14 px; a 640x480 PothRGBD frame becomes 518x392
IN_W, IN_H = 518, 392            # the fixed training size (PothRGBD's 4:3)
RELIEF_UNIT_MM = 50.0            # the network works in units of 50 mm
ZOOM_BELOW = 0.05                # a pothole under this share of the frame is read from a close-up
ZOOM_TARGET = 0.20               # ... in which it fills this share: the median of the measured photos
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def installed_weights() -> List[str]:
    return sorted(glob.glob(os.path.join(MODEL_DIR, "relief_vits*.pth")))


HAS_RELIEF = bool(installed_weights()) and os.path.isfile(META_PATH)
_MODELS = None
_META: Optional[Dict[str, Any]] = None


def meta() -> Dict[str, Any]:
    """Cut-offs, error bar and evaluation record written by scripts/eval_relief_depth.py --install."""
    global _META
    if _META is None:
        with open(META_PATH, encoding="utf-8") as f:
            _META = json.load(f)
    return _META


def build_model(weights: Optional[str] = None):
    """DA-V2 ViT-S with a signed linear output. `weights=None` starts from the DA-V2 checkpoint."""
    import torch
    import torch.nn as nn
    root = os.path.join(PROJECT_DIR, "Depth-Anything-V2")
    if root not in sys.path:
        sys.path.append(root)
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    if weights is None:
        model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu"))
    # Relief is signed (bumps are negative), so the final ReLU goes.
    model.depth_head.scratch.output_conv2[3] = nn.Identity()
    if weights is None:
        # Start from "everything is flat road": the pretrained last layer outputs
        # disparity-like values in the hundreds, far from the relief scale.
        last = model.depth_head.scratch.output_conv2[2]
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)
    else:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model


def forward_relief(model, x):
    """Network output in units of RELIEF_UNIT_MM, shape (B, H, W). Skips DA-V2's own ReLU."""
    ph, pw = x.shape[-2] // 14, x.shape[-1] // 14
    feats = model.pretrained.get_intermediate_layers(
        x, model.intermediate_layer_idx[model.encoder], return_class_token=True)
    return model.depth_head(feats, ph, pw).squeeze(1)


def input_size(h: int, w: int):
    """(width, height) fed to the network: long side 518, both multiples of 14, aspect kept."""
    s = LONG_SIDE / max(h, w)
    return max(14, int(round(w * s / 14)) * 14), max(14, int(round(h * s / 14)) * 14)


def to_tensor(image_bgr: np.ndarray, size=(IN_W, IN_H)) -> np.ndarray:
    """BGR image -> normalised CHW float32 at `size` (width, height)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def _served_models():
    global _MODELS
    if _MODELS is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODELS = [build_model(w).to(device).eval() for w in installed_weights()]
    return _MODELS


def predict_relief(image_bgr: np.ndarray, model=None, tta: bool = True) -> Optional[np.ndarray]:
    """
    HxW float32 relief map in mm at the image's own resolution, or None if unavailable.

    `model` may be one network or a list to average; None uses the installed ones.
    `tta` averages each prediction with that of the mirrored image.
    """
    import torch
    if model is None:
        if not HAS_RELIEF:
            return None
        model = _served_models()
    nets = model if isinstance(model, (list, tuple)) else [model]
    device = next(nets[0].parameters()).device
    h, w = image_bgr.shape[:2]
    x = torch.from_numpy(to_tensor(image_bgr, input_size(h, w)))[None].to(device)
    if tta:
        x = torch.cat([x, x.flip(-1)])
    outs = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for net in nets:
            o = forward_relief(net, x).float()
            outs.append((o[0] + o[1].flip(-1)) / 2 if tta else o[0])
    out = torch.stack(outs).mean(0).cpu().numpy()
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR) * RELIEF_UNIT_MM


def bowl_depth_mm(relief_mm: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """
    The label protocol applied to a predicted relief map: re-fit the road plane on the
    ring around the mask (removing any tilt the network left in), then the 90th
    percentile of what lies below it inside the mask.
    """
    from metric_features import road_plane_deviation
    m = (np.asarray(mask) > 0).astype(np.uint8)
    res = road_plane_deviation(relief_mm, m, np.ones(m.shape, bool))
    if res is None or int(m.sum()) < 150:
        return None
    dev, _ring, _rms = res
    return float(np.percentile(dev[m > 0], 90))


def zoom_window(mask: np.ndarray, target: float = ZOOM_TARGET):
    """
    4:3 window (x0, y0, x1, y1), as fractions of the frame, in which the pothole fills
    about `target` of the picture. None if the pothole is already that prominent or the
    window would be the whole frame.
    """
    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or ys.size / (h * w) >= ZOOM_BELOW:
        return None
    ww = float(np.sqrt(ys.size / target * IN_W / IN_H))
    # never tighter than the outline plus the ring of road the label protocol fits its plane on
    ww = max(ww, (xs.max() - xs.min()) * 1.6, (ys.max() - ys.min()) * 1.6 * IN_W / IN_H)
    wh = ww * IN_H / IN_W
    if ww >= w or wh >= h:
        return None
    x0 = float(np.clip((xs.min() + xs.max()) / 2 - ww / 2, 0, w - ww))
    y0 = float(np.clip((ys.min() + ys.max()) / 2 - wh / 2, 0, h - wh))
    return x0 / w, y0 / h, (x0 + ww) / w, (y0 + wh) / h


def bowl_depth_zoomed(image_bgr: np.ndarray, mask: np.ndarray, model=None) -> Optional[float]:
    """
    Bowl depth read from a close-up cut around the pothole, or None when no zoom applies.

    Why: in the measured photos the pothole fills a fifth of the frame (median; 3% at the
    5th percentile), and the network learned that a pothole small in the picture is shallow.
    On a second camera it read a 45 mm hole covering 1% of the frame as 5 mm. Depth below the
    road does not change when the photo is cropped, so the pothole is shown to the network
    the way the training photos showed theirs. `image_bgr` may be larger than `mask`
    (the original photo): the cut is taken from it, keeping its detail.
    """
    win = zoom_window(mask)
    if win is None:
        return None
    H, W = image_bgr.shape[:2]
    h, w = mask.shape[:2]
    crop = image_bgr[int(win[1] * H):int(win[3] * H), int(win[0] * W):int(win[2] * W)]
    m = mask[int(win[1] * h):int(win[3] * h), int(win[0] * w):int(win[2] * w)]
    size = (640, 480)
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if crop.shape[1] > size[0] else cv2.INTER_CUBIC)
    m = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    relief = predict_relief(crop, model)
    return None if relief is None else bowl_depth_mm(relief, m)


_JET_BGR = None


def jet_to_scalar(colour_bgr: np.ndarray):
    """
    Undo a JET colour rendering: (HxW uint8 index 0-255, mean colour distance to the scale).

    Pothole-600 ships its transformed disparity only as JET pictures. The colours sit on
    the scale almost exactly (mean distance ~1.3 of a possible 441), so the nearest scale
    entry recovers the number that was rendered. What comes back is an 8-bit SHAPE with an
    unknown per-image scale: usable for rank and shape comparisons, never for millimetres.
    """
    global _JET_BGR
    if _JET_BGR is None:
        _JET_BGR = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                                     cv2.COLORMAP_JET).reshape(256, 3).astype(np.int32)
    h, w = colour_bgr.shape[:2]
    px = colour_bgr.reshape(-1, 3)
    colours, inverse = np.unique(px, axis=0, return_inverse=True)
    d = ((colours[:, None, :].astype(np.int32) - _JET_BGR[None, :, :]) ** 2).sum(-1)
    index = d.argmin(1).astype(np.uint8)[inverse.ravel()].reshape(h, w)
    dist = float(np.sqrt(d.min(1))[inverse.ravel()].mean())
    return index, dist


def describe() -> str:
    if not HAS_RELIEF:
        return "relief network OFF (no weights in ml_models/metric/)"
    m = meta()
    t = m["decision_thresholds_mm"]
    return (f"fine-tuned relief network ({len(installed_weights())} averaged, fitted {m.get('fitted')}); "
            f"held-out MAE {m['test_yolo']['mae']:.1f} mm on the segmenter's outlines, "
            f"80% within {m['abs_error_q80_mm']:.0f} mm; "
            f"Moderate from {t['moderate_from']:g} mm, Deep from {t['deep_from']:g} mm")
