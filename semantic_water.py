"""
CLIPSeg open-vocabulary water prior — Phase 4 cue 7.

The six existing cues in water_detection.py all measure low-level optics: edge
density, gradient magnitude, specular clustering, colour, saturation, and a
depth/geometry disagreement. Every one of them responds to *contrast*, and a
hard shadow on dry asphalt produces contrast. That shared blind spot is why the
module needed hand-tuned penalties to stop firing on pic-65 (dry gravel in
sunlight).

CLIPSeg answers a different question — "does this region read as water to a
model trained on image/text pairs" — so its failure mode is not the same one.
That independence is the whole reason to add it.

Two design points worth stating:

1. The score is a CONTRAST between water prompts and dry prompts, never a raw
   water response. CLIPSeg's absolute activation is strongly prompt-dependent:
   on a genuinely wet test image, "dry asphalt road surface" returns a higher
   mean response (0.55) than "a puddle of water on the road" (0.30) simply
   because most of the frame is road. Only the difference carries signal.

2. One of the dry prompts names the exact confuser — a dark stain on tarmac —
   because that is the case Phase 3's texture-illusion rule already struggles
   with, and it is the case where a naive water prompt would fire hardest.

Runs once per IMAGE, not once per pothole. The caller computes the prior map and
passes it into detect_water for each region, so the ViT forward pass is paid
once regardless of how many potholes were segmented.

Usage:
    from semantic_water import HAS_CLIPSEG, water_prior_map
    prior = water_prior_map(image_rgb)          # HxW float32 in [0, 1], or None
    detect_water(image_rgb, mask, water_prior_map=prior)
"""
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    _HAS_DEPS = True
except Exception:
    _HAS_DEPS = False

HAS_CLIPSEG = _HAS_DEPS  # public flag for callers

MODEL_NAME = "CIDAS/clipseg-rd64-refined"

# Averaged within each group, then contrasted against the other group.
WATER_PROMPTS = (
    "a puddle of water on the road",
    "standing water on the road surface",
    "a reflective wet patch on asphalt",
)
DRY_PROMPTS = (
    "dry asphalt road surface",
    "dry gravel and broken stones",
    "a dark stain on dry tarmac",   # the confuser, named deliberately
)

_MODEL = None
_PROCESSOR = None


def _load_clipseg() -> Optional[Tuple[Any, Any]]:
    """Lazy-load CLIPSeg. Returns None if unavailable, never raises."""
    global _MODEL, _PROCESSOR

    if not HAS_CLIPSEG:
        return None
    if _MODEL is not None:
        return (_MODEL, _PROCESSOR)

    try:
        _PROCESSOR = CLIPSegProcessor.from_pretrained(MODEL_NAME)
        _MODEL = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = _MODEL.to(device).eval()
        return (_MODEL, _PROCESSOR)
    except Exception as e:
        print(f"  ⚠  CLIPSeg loading failed: {e}")
        return None


def water_prior_map(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Per-pixel probability that a region reads as water rather than dry road.

    Args:
        image_rgb: RGB image (HxWx3, uint8).

    Returns:
        HxW float32 in [0, 1] at the input resolution, or None if CLIPSeg is
        unavailable or inference fails. 0.5 means the two prompt groups agree,
        i.e. no evidence either way.
    """
    loaded = _load_clipseg()
    if loaded is None:
        return None
    model, processor = loaded

    try:
        prompts = list(WATER_PROMPTS) + list(DRY_PROMPTS)
        n_water = len(WATER_PROMPTS)

        device = next(model.parameters()).device
        inputs = processor(
            text=prompts,
            images=[image_rgb] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits          # (n_prompts, 352, 352)

        if logits.ndim == 2:                          # single-prompt edge case
            logits = logits.unsqueeze(0)

        water_logit = logits[:n_water].mean(dim=0)
        dry_logit = logits[n_water:].mean(dim=0)

        # Softmax over the two groups: a proper contrast, bounded in [0, 1],
        # and invariant to CLIPSeg's overall activation level on this image.
        stacked = torch.stack([water_logit, dry_logit], dim=0)
        prior = torch.softmax(stacked, dim=0)[0]

        prior_np = prior.detach().cpu().numpy().astype(np.float32)

        h, w = image_rgb.shape[:2]
        if prior_np.shape != (h, w):
            prior_np = cv2.resize(prior_np, (w, h), interpolation=cv2.INTER_LINEAR)

        return np.clip(prior_np, 0.0, 1.0)
    except Exception as e:
        print(f"  ⚠  CLIPSeg inference failed: {e}")
        return None


def describe() -> str:
    """One-line status string, safe to log or surface in the API."""
    if not HAS_CLIPSEG:
        return "CLIPSeg unavailable (transformers/torch not importable)"
    state = "loaded" if _MODEL is not None else "not yet loaded"
    return (
        f"CLIPSeg {MODEL_NAME} ({state}); "
        f"{len(WATER_PROMPTS)} water vs {len(DRY_PROMPTS)} dry prompts, "
        "softmax contrast"
    )


if __name__ == "__main__":
    import sys

    print(describe())
    if len(sys.argv) > 1:
        img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
        m = water_prior_map(img)
        if m is None:
            print("no map produced")
        else:
            print(f"prior map {m.shape} min={m.min():.4f} "
                  f"mean={m.mean():.4f} max={m.max():.4f}")
