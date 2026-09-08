"""
Intrinsic image decomposition — Phase 4 cue 8, and a repaired input for Phase 2.

Why this module exists
----------------------
The Phase 4 argument for water rests on Fresnel reflectance. At a dielectric
interface the reflected fraction is approximately

    R(theta) = R0 + (1 - R0) * (1 - cos theta)^5,    R0 ~ 0.02 for water

Road photography is inherently grazing-angle: a phone or dashcam looks ALONG the
surface, so theta is typically 80-88 degrees, giving R between roughly 0.40 and
0.84. Most of what the sensor receives from a puddle is reflected sky, not the
pothole.

That argument has so far been rhetorical. Intrinsic decomposition makes it
measurable. The pipeline separates an image into albedo, diffuse shading, and a
RESIDUAL that carries the non-diffuse content — which is exactly the specular
Fresnel term. Cue 8 measures residual energy inside the pothole against the
surrounding road ring.

Two further uses, beyond water detection
----------------------------------------
1. Phase 2's shape-from-shading inverts a Lambertian model, I = rho * (n . l),
   in which radiance does not depend on view direction. Specular reflection is
   view-dependent by definition, so on any wet surface SfS is not approximately
   wrong, it is structurally wrong. Running SfS on the DIFFUSE SHADING channel
   removes the specular term before inversion. The recorded baseline to beat is
   mean correlation +0.062, with 2/15 strong agreement and 7/15 anti-correlated
   (ml_results/sfs_vs_depth_anything/).

2. Phase 3's texture-illusion problem is an albedo/geometry confusion: a tar
   seal is a dark ALBEDO patch with flat shading, whereas a real crater is
   SHADING structure. Comparing the two channels is a more principled test than
   DINOv2 patch variance, which cannot separate the two by construction.

Licence
-------
compphoto/Intrinsic is released for ACADEMIC USE ONLY. Acceptable for coursework;
it forecloses commercial deployment. Recorded here rather than discovered later.
Papers: "Intrinsic Image Decomposition via Ordinal Shading" (TOG 2023) and
"Colorful Diffuse Intrinsic Image Decomposition in the Wild" (TOG 2024).

Runs once per IMAGE, not once per pothole.
"""
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    from intrinsic.pipeline import load_models, run_pipeline
    _HAS_DEPS = True
except Exception:
    _HAS_DEPS = False

HAS_INTRINSIC = _HAS_DEPS  # public flag for callers

MODEL_VERSION = "v2"

# The pipeline is a stack of networks at high resolution. Region statistics do
# not need full resolution, and capping the long side keeps a batch run
# tractable. None disables the cap.
DEFAULT_MAX_SIDE = 1024

_MODELS = None


def _load_intrinsic() -> Optional[Any]:
    """Lazy-load the Intrinsic pipeline. Returns None if unavailable."""
    global _MODELS

    if not HAS_INTRINSIC:
        return None
    if _MODELS is not None:
        return _MODELS

    try:
        _MODELS = load_models(MODEL_VERSION)
        return _MODELS
    except Exception as e:
        print(f"  ⚠  Intrinsic loading failed: {e}")
        return None


def _to_scalar(arr: np.ndarray) -> np.ndarray:
    """Collapse a possibly-3-channel component to a per-pixel magnitude."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] > 1:
        return np.linalg.norm(a, axis=2) / np.sqrt(a.shape[2])
    return a.squeeze()


def decompose(
    image_rgb: np.ndarray,
    max_side: Optional[int] = DEFAULT_MAX_SIDE,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Decompose an image into intrinsic components.

    Args:
        image_rgb: RGB image (HxWx3, uint8).
        max_side:  Cap the longer side before inference, for speed. None to
                   disable. Output is always resized back to the input shape.

    Returns:
        Dict of HxW float32 maps at the input resolution, or None on failure:

            albedo           reflectance      (hr_alb)
            shading          diffuse shading  (dif_shd) — the repaired SfS input
            residual_signed  signed residual  (residual)
            residual_pos     positive residual (pos_res) — the specular term
            residual_neg     negative residual (neg_res) — shadowing
            residual         alias for residual_pos, what cue 8 consumes

        Three-channel components are collapsed to a per-pixel RMS magnitude.

        Values are NOT rescaled to [0, 1]: the comparison that matters is
        interior versus surrounding road, and squashing here would destroy the
        ratio the cue depends on.
    """
    models = _load_intrinsic()
    if models is None:
        return None

    try:
        h, w = image_rgb.shape[:2]

        work = image_rgb
        if max_side is not None and max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            work = cv2.resize(
                image_rgb,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        # The pipeline expects a float array in [0, 1].
        img = np.asarray(work, dtype=np.float32) / 255.0

        results = run_pipeline(models, img)

        # Key names follow the repo's documented pipeline output. Fall back
        # across plausible aliases rather than raising, since a rename upstream
        # should degrade this cue, not break the whole water module.
        def pick(*names):
            for n in names:
                if n in results:
                    return results[n]
            return None

        albedo = pick("hr_alb", "albedo", "alb")
        shading = pick("dif_shd", "shading", "shd")
        residual = pick("residual", "res", "nd_shd")

        # The pipeline separates the residual by sign, and the distinction is
        # physical rather than cosmetic. Specular reflection ADDS radiance the
        # diffuse model cannot account for, so it lands in the positive
        # residual. The negative side carries the opposite — regions darker
        # than the diffuse model predicts, i.e. shadowing and inter-reflection.
        # Verified on a real road image: `residual` is signed, spanning
        # -0.338 to 0.880. Using the signed channel for a water cue would let
        # shadow cancel specularity, which is precisely backwards.
        residual_pos = pick("pos_res")
        residual_neg = pick("neg_res")

        if residual is None or shading is None:
            print(f"  ⚠  Intrinsic returned unexpected keys: {sorted(results.keys())}")
            return None

        out = {}
        for name, arr in (
            ("albedo", albedo),
            ("shading", shading),
            ("residual_signed", residual),
            ("residual_pos", residual_pos),
            ("residual_neg", residual_neg),
        ):
            if arr is None:
                continue
            a = _to_scalar(arr)
            if a.shape != (h, w):
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            out[name] = a.astype(np.float32)

        # The specular term the water cue consumes. Prefer the positive
        # residual; fall back to the magnitude of the signed one if this build
        # of the pipeline does not expose it.
        if "residual_pos" in out:
            out["residual"] = out["residual_pos"]
        else:
            out["residual"] = np.abs(out["residual_signed"])

        return out
    except Exception as e:
        print(f"  ⚠  Intrinsic inference failed: {e}")
        return None


def residual_energy_map(
    image_rgb: np.ndarray,
    max_side: Optional[int] = DEFAULT_MAX_SIDE,
) -> Optional[np.ndarray]:
    """
    Convenience wrapper returning only the non-diffuse residual magnitude.

    This is the channel that carries the Fresnel specular term, and the one
    water_detection consumes as cue 8.
    """
    d = decompose(image_rgb, max_side=max_side)
    return None if d is None else d["residual"]


def describe() -> str:
    """One-line status string, safe to log or surface in the API."""
    if not HAS_INTRINSIC:
        return "Intrinsic unavailable (compphoto/Intrinsic not importable)"
    state = "loaded" if _MODELS is not None else "not yet loaded"
    return (
        f"Intrinsic {MODEL_VERSION} ({state}); albedo/shading/residual, "
        f"max_side={DEFAULT_MAX_SIDE}; ACADEMIC USE ONLY licence"
    )


if __name__ == "__main__":
    import sys

    print(describe())
    if len(sys.argv) > 1:
        img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
        d = decompose(img)
        if d is None:
            print("no decomposition produced")
        else:
            for k, v in d.items():
                print(f"  {k:13s} {v.shape} min={v.min():.4f} "
                      f"mean={v.mean():.4f} max={v.max():.4f}")
