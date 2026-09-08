"""
DINOv2 Foundation Feature Extractor
====================================
Extracts semantic features from Meta's DINOv2 vision foundation model
for pothole severity analysis.  DINOv2 provides rich, pre-trained
patch-level features that capture material, texture, and structural
differences between pothole interior and surrounding road.

Three features are computed:
    dinov2_dissimilarity:    cosine distance between pothole interior
                             and surrounding road patch embeddings
    dinov2_inside_variance:  feature variance inside the pothole mask
                             (higher = more heterogeneous damage)
    dinov2_outside_variance: feature variance in the surrounding road

Graceful Degradation:
    If `transformers` is not installed, all functions return None and
    the rest of the pipeline silently skips these features.

Usage:
    from foundation_features import extract_foundation_features
    feats = extract_foundation_features(image_rgb, mask)
    # feats = {'dinov2_dissimilarity': 0.42, ...} or None
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Availability Check
# ═══════════════════════════════════════════════════════════════════════════

_HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoModel, AutoImageProcessor
    _HAS_TRANSFORMERS = True
except ImportError:
    pass

HAS_DINOV2 = _HAS_TRANSFORMERS  # public flag for callers


# ═══════════════════════════════════════════════════════════════════════════
#  Model Loading (Lazy Singleton)
# ═══════════════════════════════════════════════════════════════════════════

_DINOV2_MODEL = None
_DINOV2_PROCESSOR = None


def _load_dinov2_model() -> Optional[Tuple[Any, Any]]:
    """
    Lazy-load the DINOv2-base model from HuggingFace.

    Returns (model, processor) tuple on success, None on failure.
    Model is loaded once and cached as a module-level singleton.
    """
    global _DINOV2_MODEL, _DINOV2_PROCESSOR

    if not _HAS_TRANSFORMERS:
        return None

    if _DINOV2_MODEL is not None:
        return (_DINOV2_MODEL, _DINOV2_PROCESSOR)

    try:
        model_name = "facebook/dinov2-base"
        _DINOV2_PROCESSOR = AutoImageProcessor.from_pretrained(model_name)
        _DINOV2_MODEL = AutoModel.from_pretrained(model_name)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DINOV2_MODEL = _DINOV2_MODEL.to(device).eval()

        return (_DINOV2_MODEL, _DINOV2_PROCESSOR)
    except Exception as e:
        print(f"  ⚠  DINOv2 loading failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

#  Variance-comparison helpers (A1 — measurement correction)
# ═══════════════════════════════════════════════════════════════════════════

#  Seeded so calibration runs are reproducible.
_RNG = np.random.default_rng(20260815)

#  How many patch-rings out from the mask to treat as "local road".
_RING_DILATION = 2
#  Repeat count for size-matched subsampling.
_MATCH_DRAWS = 20


def _ring_patches(inside_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Boolean grid selecting road patches in a narrow ring around the pothole.

    Comparing the interior against the immediately surrounding road — rather
    than against the entire rest of the scene — controls for scene composition.
    This mirrors the road-surround idea already used in
    water_detection._get_road_surround_mask, applied at patch resolution.

    Returns None if the mask is empty.
    """
    inside_u8 = inside_mask.astype(np.uint8)
    if inside_u8.sum() == 0:
        return None
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(inside_u8, kernel, iterations=_RING_DILATION)
    return (dilated > 0) & (~inside_mask)


def _size_matched_variance(features, k: int) -> float:
    """
    Mean feature variance of `features`, computed over random subsets of size k.

    Variance estimated over many patches is not comparable with variance over a
    handful, so the road sample is drawn down to the same patch count as the
    pothole interior. The median across repeated draws is used for stability.
    """
    n = int(features.shape[0])
    if n == 0:
        return 0.0
    if n <= k:
        return float(features.var(dim=0).mean().item())

    vals = []
    for _ in range(_MATCH_DRAWS):
        idx = _RNG.choice(n, size=k, replace=False)
        subset = features[torch.as_tensor(idx, device=features.device)]
        vals.append(float(subset.var(dim=0).mean().item()))
    return float(np.median(vals))


def extract_foundation_features(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> Optional[Dict[str, float]]:
    """
    Extract DINOv2 foundation features comparing pothole interior
    to surrounding road.

    The key insight: DINOv2 patch features encode high-level material
    and structural properties.  A large cosine distance between pothole
    patches and road patches indicates the pothole has significantly
    different visual properties (damaged, waterlogged, or severely
    degraded), correlating with severity.

    Args:
        image_rgb: RGB image (HxW×3, uint8).
        mask: Binary pothole mask (HxW, {0,1}).

    Returns:
        Dictionary with:
            dinov2_dissimilarity:    cosine distance (0-2 range)
            dinov2_inside_variance:  mean feature variance inside
            dinov2_outside_variance: mean feature variance outside
        Returns None if DINOv2 is unavailable or extraction fails.
    """
    result = _load_dinov2_model()
    if result is None:
        return None

    model, processor = result

    try:
        # ── Prepare input ──
        # DINOv2-base uses 224×224 input with 14×14 patch grid (16px patches)
        inputs = processor(images=image_rgb, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ── Forward pass ──
        with torch.no_grad():
            outputs = model(**inputs)

        # last_hidden_state: (1, num_patches+1, 768)
        # Skip [CLS] token (index 0), keep patch tokens
        patch_features = outputs.last_hidden_state[0, 1:, :]  # (N_patches, 768)

        # ── Map patches to spatial grid ──
        # DINOv2-base: 14×14 = 196 patch tokens for 224×224 input
        num_patches = patch_features.shape[0]
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            # Non-square patch grid — fall back
            return None

        # Reshape to spatial grid: (grid_size, grid_size, 768)
        patch_grid = patch_features.reshape(grid_size, grid_size, -1)

        # ── Resize mask to patch resolution ──
        mask_resized = cv2.resize(
            mask.astype(np.float32),
            (grid_size, grid_size),
            interpolation=cv2.INTER_AREA,
        )
        inside_mask = mask_resized > 0.5   # patches mostly inside pothole
        outside_mask = mask_resized <= 0.5  # patches mostly outside

        n_inside = int(inside_mask.sum())
        n_outside = int(outside_mask.sum())

        if n_inside < 2 or n_outside < 2:
            return None

        # ── Extract feature vectors ──
        inside_features = patch_grid[inside_mask]   # (n_inside, 768)
        outside_features = patch_grid[outside_mask]  # (n_outside, 768)

        # ── Compute dissimilarity (cosine distance) ──
        mean_inside = inside_features.mean(dim=0)    # (768,)
        mean_outside = outside_features.mean(dim=0)  # (768,)

        cos_sim = torch.nn.functional.cosine_similarity(
            mean_inside.unsqueeze(0),
            mean_outside.unsqueeze(0),
        ).item()
        dissimilarity = 1.0 - cos_sim  # cosine distance

        # ── Compute variances ──
        inside_variance = float(inside_features.var(dim=0).mean().item())
        outside_variance = float(outside_features.var(dim=0).mean().item())

        # ── Corrected comparison (see _ring_patches / _size_matched_variance) ──
        # The raw outside_variance above is confounded: it is computed over the
        # whole scene (lane markings, vehicles, kerbs, sky) against a small,
        # necessarily uniform pothole interior, so the ratio largely measures
        # region size and scene composition rather than material heterogeneity.
        # The corrected statistic compares against a local ring of road patches,
        # sampled to the same patch count as the interior.
        ring_mask = _ring_patches(inside_mask)
        if ring_mask is not None and int(ring_mask.sum()) >= 2:
            ring_features = patch_grid[ring_mask]
            ring_source = "ring"
        else:
            # Degenerate ring (pothole fills the frame) — fall back to all-outside
            ring_features = outside_features
            ring_source = "scene"

        ring_variance = _size_matched_variance(ring_features, n_inside)
        ratio_corrected = (
            inside_variance / ring_variance if ring_variance > 1e-9 else float("nan")
        )

        return {
            "dinov2_dissimilarity": round(dissimilarity, 6),
            "dinov2_inside_variance": round(inside_variance, 6),
            "dinov2_outside_variance": round(outside_variance, 6),
            # Corrected measures (A1) — used by the calibrated override
            "dinov2_ring_variance": round(ring_variance, 6),
            "dinov2_ratio_corrected": round(ratio_corrected, 6),
            "dinov2_patch_count": n_inside,
            "dinov2_ring_source": ring_source,
        }

    except Exception as e:
        print(f"  ⚠  DINOv2 feature extraction failed: {e}")
        return None
