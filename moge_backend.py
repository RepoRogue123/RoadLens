"""
MoGe monocular GEOMETRY backend — metric depth, normals, intrinsics, point map.

Why this module exists
----------------------
Depth-Anything-V2, our current backbone, produces RELATIVE depth normalised per
image. It therefore has no scale at all: a 2 cm depression and a 20 cm crater can
produce identical maps, and the same pothole photographed at 2 m and 10 m gives
different values. That is why the severity thresholds in classifier.py are 0.3
and 0.6 in arbitrary units rather than millimetres, and why
scripts/convert_pothole600.py had to be corrected to CALIBRATION_AVAILABLE=False.

MoGe returns METRIC geometry from a single forward pass:

    points      (H,W,3)  metric point map, OpenCV camera coords
    depth       (H,W)    metric depth map        <- fixes the scale problem
    intrinsics  (3,3)    estimated from the image <- fixes the calibration gap
    mask        (H,W)    valid pixels             <- an abstention signal
    normal      (H,W,3)  surface normals          <- replaces Sobel-differentiated
                                                     relative depth in features.py

Adoption discipline
-------------------
Nothing here is wired into the live pipeline. `inference.get_depth_map` is
untouched, because all eight trained models and the 39-feature scaler are fitted
on Depth-Anything relative depth — swapping the units underneath them silently
invalidates every model and the whole ablation table. This module exists so the
swap can be EVALUATED first, against real RealSense measurements from PothRGBD.

Model selection
---------------
MoGe-3 (v3) is the default and requires an explicit checkpoint. It needs
FlexGEMM/Triton; that installs cleanly on this machine (triton-windows 3.8.0),
but MoGe-2 is kept as a fallback because it needs no such toolchain and already
provides metric scale and normals.

    moge-3-vitl   370M   metric + normal   <- default
    moge-3-vitg  1.25B   metric + normal   too large for 8 GB
    moge-2-vitl-normal  331M               fallback, no Triton needed
    moge-2-vitb-normal  104M               low-VRAM fallback

Licence: MIT, except the vendored DINOv2 (Apache 2.0). Note the pleasing
coincidence that MoGe uses DINOv2 as its backbone — the same representation our
Phase 3 semantic witness reads, decoded for a different question.

Usage:
    from moge_backend import HAS_MOGE, get_geometry, get_depth_map_metric
    g = get_geometry(image_bgr)      # dict, or None
    d = get_depth_map_metric(image_bgr)   # HxW float32 metres, or None
"""
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Version is resolved lazily: v3 needs FlexGEMM at import time on some builds,
# so a failure here must degrade to v2 rather than take the module down.
MODEL_SPECS = [
    ("v3", "Ruicheng/moge-3-vitl"),
    ("v2", "Ruicheng/moge-2-vitl-normal"),
    ("v2", "Ruicheng/moge-2-vitb-normal"),
]

# Environment overrides, so an evaluation can pin a specific model without edits.
ENV_VERSION = os.environ.get("MOGE_VERSION")      # "v2" / "v3"
ENV_CHECKPOINT = os.environ.get("MOGE_CHECKPOINT")

REFINE_STEPS = int(os.environ.get("MOGE_REFINE_STEPS", "3"))   # v3 only
USE_FP16 = os.environ.get("MOGE_FP16", "1") != "0"


def _probe() -> bool:
    """True if any MoGe model class can be imported at all."""
    if not _HAS_TORCH:
        return False
    for ver, _ in MODEL_SPECS:
        try:
            __import__(f"moge.model.{ver}", fromlist=["MoGeModel"])
            return True
        except Exception:
            continue
    return False


HAS_MOGE = _probe()

_MODEL = None
_LOADED: Optional[str] = None       # human-readable "v3 Ruicheng/moge-3-vitl"


def _candidates():
    if ENV_VERSION and ENV_CHECKPOINT:
        return [(ENV_VERSION, ENV_CHECKPOINT)]
    if ENV_VERSION:
        return [(v, c) for v, c in MODEL_SPECS if v == ENV_VERSION] or MODEL_SPECS
    return MODEL_SPECS


def _load_moge():
    """
    Lazy-load the first MoGe model that actually works, largest first.

    Tries each candidate in turn rather than failing on the first — a
    checkpoint can be unavailable, or too large for the GPU, and neither should
    take the whole backend offline when a smaller one would run.
    """
    global _MODEL, _LOADED
    if not HAS_MOGE:
        return None
    if _MODEL is not None:
        return _MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ver, ckpt in _candidates():
        try:
            mod = __import__(f"moge.model.{ver}", fromlist=["MoGeModel"])
            model = mod.MoGeModel.from_pretrained(ckpt).to(device).eval()
            # NOTE: do NOT call model.half(). MoGe mixes precisions internally
            # and a hard cast produces "mat1 and mat2 must have the same dtype,
            # but got Float and Half" inside the first Linear. fp16 is applied
            # via autocast at inference instead, which handles the mixing.
            _MODEL, _LOADED = model, f"{ver} {ckpt}"
            print(f"  MoGe loaded: {_LOADED} on {device}"
                  f"{' (autocast fp16)' if USE_FP16 and device == 'cuda' else ''}")
            return _MODEL
        except Exception as e:
            print(f"  !! MoGe {ver} {ckpt} unavailable: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            continue

    print("  !! No MoGe checkpoint could be loaded")
    return None


def get_geometry(image_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Run MoGe on a BGR image.

    Returns a dict with 'depth' (HxW float32, METRES), 'points', 'normal',
    'intrinsics', 'mask' — every map resized back to the input resolution — or
    None if MoGe is unavailable or inference fails. Never raises.
    """
    model = _load_moge()
    if model is None:
        return None

    try:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        device = next(model.parameters()).device
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device)
        t = t.permute(2, 0, 1)

        kwargs = {}
        if _LOADED and _LOADED.startswith("v3"):
            kwargs["refine_steps"] = REFINE_STEPS

        use_amp = USE_FP16 and device.type == "cuda"
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model.infer(t, **kwargs)
            else:
                out = model.infer(t, **kwargs)

        def to_np(v):
            if v is None:
                return None
            a = v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            return a

        res: Dict[str, Any] = {}
        for key in ("depth", "points", "normal", "mask", "intrinsics"):
            a = to_np(out.get(key)) if isinstance(out, dict) else None
            if a is None:
                continue
            # Intrinsics are a 3x3, never an image — leave alone.
            if key != "intrinsics" and a.ndim >= 2 and a.shape[:2] != (h, w):
                interp = cv2.INTER_NEAREST if key == "mask" else cv2.INTER_LINEAR
                a = cv2.resize(a.astype(np.float32), (w, h), interpolation=interp)
            res[key] = a.astype(np.float32) if key != "intrinsics" else a

        return res or None
    except Exception as e:
        print(f"  !! MoGe inference failed: {type(e).__name__}: {str(e)[:150]}")
        return None


def get_depth_map_metric(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Metric depth only, HxW float32 in METRES.

    Deliberately NOT named `get_depth_map`: inference.get_depth_map returns
    relative depth and is consumed by five separate call sites in api.py. Giving
    these the same name would invite an accidental drop-in swap, which is the
    single change most likely to silently invalidate every trained model.
    """
    g = get_geometry(image_bgr)
    return None if g is None else g.get("depth")


def describe() -> str:
    """One-line status string, safe to log or surface in the API."""
    if not HAS_MOGE:
        return "MoGe unavailable (moge/torch not importable)"
    if _MODEL is None:
        return f"MoGe available, not yet loaded (will try {_candidates()[0][1]})"
    return f"MoGe {_LOADED}, refine_steps={REFINE_STEPS}, metric depth in metres"


if __name__ == "__main__":
    import sys

    print(describe())
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        g = get_geometry(img)
        if g is None:
            print("no geometry produced")
        else:
            for k, v in g.items():
                a = np.asarray(v)
                if k == "intrinsics":
                    print(f"  {k:10s} {a.shape}\n{a}")
                else:
                    print(f"  {k:10s} {a.shape} min={a.min():.4f} "
                          f"max={a.max():.4f} mean={a.mean():.4f}")
