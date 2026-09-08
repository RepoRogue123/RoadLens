"""
Water Detection Module for Pothole Analysis — v3 (Edge-Aware)
=============================================================
Fixes from v2 → v3 (based on per-image ground-truth feedback):
  - pic-52 was falsely classified DRY (16.9%) but has subtle moisture
  - pic-65 was falsely classified WATER (47.1%) but is dry gravel in sunlight

Root cause: v2 had no texture roughness analysis, so:
  - Bright sunlight on rough gravel triggered specular + saturation cues (false +)
  - Subtle moisture on dark surfaces was missed (false -)

v3 changes:
  1. NEW: Edge density cue — Canny edge fraction inside vs surroundings.
     Water surfaces have very few edges (<5%); gravel has many (>30%).
     This is the strongest single anti-false-positive signal.
  2. NEW: Gradient smoothness cue — Sobel gradient magnitude ratio.
     Water has gradient ratio <0.7; dry rough surfaces >1.2.
  3. FIXED: Specular cue now checks spatial clustering of bright pixels.
     Water = clustered bright spots; sunlight = diffuse uniform brightness.
  4. FIXED: Smoothness cue now uses the SURROUNDING ROAD ONLY (excludes
     grass/curbs via hue filtering) for more accurate comparison.
  5. Adjusted ensemble weights to prioritize edge density and gradient.
  6. Added multiplicative roughness penalty: if edge density is very high,
     the final probability is scaled down regardless of other cues.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np

import water_config


def _get_road_surround_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Build a surrounding region mask that focuses on ROAD surface only,
    excluding grass, curbs, sky, and other non-road areas.

    Uses hue/saturation filtering to exclude vegetation (green) and
    brightness filtering to exclude sky.
    """
    h, w = mask.shape[:2]
    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=3)
    surround = np.clip(dilated - mask.astype(np.uint8), 0, 1)

    # Filter out vegetation (high green saturation, hue 35-85)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) &
                  (hsv[:, :, 1] > 40))
    surround[green_mask] = 0

    # Filter out sky (very bright, low saturation, high in image)
    sky_mask = ((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30))
    # Only apply sky filter to top half of image
    sky_filter = np.zeros((h, w), dtype=bool)
    sky_filter[:h // 2, :] = True
    surround[sky_mask & sky_filter] = 0

    # Ensure we have enough surrounding pixels
    if np.sum(surround) < 100:
        # Fall back to full dilation without filtering
        surround = np.clip(dilated - mask.astype(np.uint8), 0, 1)

    return surround


def _edge_density_score(image_gray: np.ndarray, mask: np.ndarray,
                        surround_mask: np.ndarray) -> float:
    """
    Cue 1: Edge Density (strongest anti-false-positive signal).

    Water surfaces are physically smooth — they have almost NO internal
    edges. Rough surfaces (gravel, broken asphalt) have many edges.

    Measured as: 1 - (edge_density_inside / max_expected_density)
    Also considers ratio to surrounding road.

    Returns score in [0, 1] where high = few edges (water-like).
    """
    edges = cv2.Canny(image_gray, 50, 150)

    inside_pixels = edges[mask > 0]
    if len(inside_pixels) < 10:
        return 0.0

    edge_frac_inside = float(np.mean(inside_pixels > 0))

    # Absolute score: water has <5% edges, gravel has >25%
    # Map: 0% → 1.0, 5% → 0.8, 15% → 0.3, 25%+ → 0.0
    abs_score = 1.0 - (edge_frac_inside / 0.20)
    abs_score = float(np.clip(abs_score, 0.0, 1.0))

    # Relative score: compare to surroundings
    surr_pixels = edges[surround_mask > 0]
    if len(surr_pixels) > 10:
        edge_frac_surr = float(np.mean(surr_pixels > 0))
        if edge_frac_surr > 0.001:
            ratio = edge_frac_inside / edge_frac_surr
            # Water: ratio < 0.7 (fewer edges than road)
            # Gravel: ratio > 1.0 (more edges than surroundings)
            rel_score = 1.0 - (ratio - 0.3) / (1.2 - 0.3)
            rel_score = float(np.clip(rel_score, 0.0, 1.0))
        else:
            rel_score = abs_score
    else:
        rel_score = abs_score

    return 0.5 * abs_score + 0.5 * rel_score


def _gradient_smoothness_score(image_gray: np.ndarray, mask: np.ndarray,
                               surround_mask: np.ndarray) -> float:
    """
    Cue 2: Gradient Smoothness.

    Computes Sobel gradient magnitude inside vs surrounding road.
    Water has very low gradient (smooth surface reflecting light uniformly).
    Gravel/broken asphalt has high gradient (sharp texture transitions).

    Returns score in [0, 1] where high = smooth gradients (water-like).
    """
    sx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sx ** 2 + sy ** 2)

    grad_inside = grad[mask > 0]
    if len(grad_inside) < 10:
        return 0.0

    mean_grad_inside = float(np.mean(grad_inside))

    # Absolute score: water has gradient <40, gravel >100
    abs_score = 1.0 - (mean_grad_inside - 15) / (120 - 15)
    abs_score = float(np.clip(abs_score, 0.0, 1.0))

    # Relative score
    grad_surr = grad[surround_mask > 0]
    if len(grad_surr) > 10:
        mean_grad_surr = float(np.mean(grad_surr))
        if mean_grad_surr > 0.1:
            ratio = mean_grad_inside / mean_grad_surr
            # Water: ratio < 0.8 (smoother than road)
            rel_score = 1.0 - (ratio - 0.4) / (1.5 - 0.4)
            rel_score = float(np.clip(rel_score, 0.0, 1.0))
        else:
            rel_score = abs_score
    else:
        rel_score = abs_score

    return 0.5 * abs_score + 0.5 * rel_score


def _specular_reflection_score(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """
    Cue 3: Specular Reflection (v3 — spatially-aware).

    v3 improvement: distinguishes CLUSTERED bright spots (water reflection)
    from DIFFUSE uniform brightness (sunlight on rough surface).

    Water reflections create localized bright regions with high local contrast.
    Sunlight on gravel creates uniform brightness with low local contrast.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    pixels = value[mask > 0]

    if len(pixels) < 10:
        return 0.0

    # Bright pixel fraction at moderate threshold
    bright_frac = float(np.mean(pixels > 180))

    if bright_frac < 0.02:
        return 0.0  # Too few bright pixels

    # Key v3 check: spatial clustering of bright pixels
    # Create bright pixel mask within pothole
    bright_mask = np.zeros_like(mask, dtype=np.uint8)
    bright_mask[(value > 180) & (mask > 0)] = 255

    # Count connected components of bright regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright_mask, connectivity=8
    )

    if num_labels <= 1:
        return 0.0

    # Analyze bright region properties
    bright_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    total_bright = sum(bright_areas)
    pothole_area = int(np.sum(mask))

    if pothole_area < 1:
        return 0.0

    bright_coverage = total_bright / pothole_area

    # Water reflection pattern: few large bright clusters (< 10 components)
    # Sunlight on gravel: many small bright spots (> 20 components)
    # OR uniform brightness (1 huge component covering most of the area)
    n_components = num_labels - 1

    if bright_coverage > 0.6 and n_components <= 3:
        # Uniform brightness over most of pothole = likely sunlight, not water
        # Water reflections rarely cover >60% of pothole uniformly
        clustering_score = 0.2
    elif n_components > 20:
        # Many tiny bright spots = textured surface in sun, not water
        clustering_score = 0.1
    elif n_components <= 8 and bright_coverage < 0.5:
        # Few clustered bright regions covering partial area = water reflection
        clustering_score = 0.9
    else:
        clustering_score = 0.5

    # Combine bright fraction and clustering quality
    bright_score = min(1.0, bright_frac * 3.0)  # Scale up
    score = 0.4 * bright_score + 0.6 * clustering_score

    return float(np.clip(score, 0.0, 1.0))


def _color_distribution_score(image_bgr: np.ndarray, mask: np.ndarray,
                              surround_mask: np.ndarray) -> float:
    """
    Cue 4: Color Distribution.

    Water reflects sky → bluish tint. Also checks brightness relative
    to surroundings (water is often brighter due to reflection).
    """
    pixels = image_bgr[mask > 0].astype(np.float32)
    if len(pixels) < 10:
        return 0.0

    blue, green, red = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    total = blue + green + red + 1e-8
    blue_ratio = float(np.mean(blue / total))

    # Blue ratio score: water has ratio > 0.33 (blue-shifted)
    blue_score = (blue_ratio - 0.30) / (0.40 - 0.30)
    blue_score = float(np.clip(blue_score, 0.0, 1.0))

    # Brightness comparison to surrounding road
    surr_pixels = image_bgr[surround_mask > 0].astype(np.float32)
    brightness_score = 0.0
    if len(surr_pixels) > 10:
        mean_in = float(np.mean(pixels))
        mean_surr = float(np.mean(surr_pixels))
        if mean_surr > 0:
            ratio = mean_in / mean_surr
            # Water is often brighter (reflecting sky)
            brightness_score = float(np.clip((ratio - 1.0) / 0.4, 0.0, 1.0))

    return float(np.clip(max(blue_score, brightness_score) * 0.6 +
                         min(blue_score, brightness_score) * 0.4, 0.0, 1.0))


def _saturation_uniformity_score(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """
    Cue 5: Saturation Uniformity.

    Water is achromatic → low saturation. But v3 also checks that
    the VALUE channel is relatively high (wet surfaces reflect light).
    Pure dark surfaces can also have low saturation but aren't water.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    inside_sat = sat[mask > 0]
    inside_val = val[mask > 0]
    if len(inside_sat) < 10:
        return 0.0

    mean_sat = float(np.mean(inside_sat))
    std_sat = float(np.std(inside_sat))
    mean_val = float(np.mean(inside_val))

    # Low saturation score
    low_sat = 1.0 - (mean_sat / 80.0)
    low_sat = float(np.clip(low_sat, 0.0, 1.0))

    # Uniform saturation score
    uniform = 1.0 - (std_sat / 50.0)
    uniform = float(np.clip(uniform, 0.0, 1.0))

    # Brightness gate: dark surfaces (val < 100) shouldn't score high
    # on saturation alone — they're dark, not wet
    brightness_gate = float(np.clip((mean_val - 80) / 100.0, 0.3, 1.0))

    score = (0.5 * low_sat + 0.5 * uniform) * brightness_gate
    return float(np.clip(score, 0.0, 1.0))


def _depth_appearance_inconsistency_score(
    curvature_features: Dict[str, Any],
    depth_map: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Cue 6: Depth-Appearance Inconsistency.

    If curvature says deep boundary but depth says flat interior → water
    is filling the depression and masking its true depth.
    """
    max_curv = float(curvature_features.get('max_curvature', 0.0))
    mean_curv = float(curvature_features.get('mean_curvature', 0.0))
    high_frac = float(curvature_features.get('high_curvature_fraction', 0.0))

    curv_severity = min(1.0, (max_curv * 0.4 + mean_curv * 5.0 + high_frac) / 3.0)

    d = depth_map.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    else:
        return 0.0

    depth_inside = d[mask > 0]
    if len(depth_inside) == 0:
        return 0.0

    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    boundary_ring = dilated - mask.astype(np.uint8)
    depth_outside = d[boundary_ring > 0]

    if len(depth_outside) == 0:
        return 0.0

    depth_contrast = abs(float(np.mean(depth_inside)) - float(np.mean(depth_outside)))
    depth_severity = min(1.0, depth_contrast / 0.2)

    inconsistency = max(0.0, curv_severity - depth_severity)
    return float(np.clip(inconsistency, 0.0, 1.0))


def _semantic_prior_score(water_prior: np.ndarray, mask: np.ndarray,
                          surround_mask: np.ndarray) -> float:
    """
    Cue 7: Open-vocabulary semantic prior (CLIPSeg).

    Cues 1-6 all respond to CONTRAST, and a hard shadow on dry asphalt produces
    contrast. That shared blind spot is why this module needed hand-tuned
    penalties to stop firing on pic-65 (dry gravel in sunlight). CLIPSeg answers
    a different question — does this region READ as water — so its errors are
    not correlated with theirs, which is the entire reason to add it.

    `water_prior` is already a softmax contrast between water prompts and dry
    prompts (see semantic_water.py), so 0.5 means "no evidence either way"
    rather than "slightly wet".

    Returns score in [0, 1] where high = reads as water.
    """
    inside = water_prior[mask > 0]
    if inside.size < 10:
        return 0.0

    mean_inside = float(np.mean(inside))

    # Absolute: the prior taken at face value.
    abs_score = float(np.clip(mean_inside, 0.0, 1.0))

    # Relative: is the interior more water-like than the road around it?
    # Centred on 0.5 so parity between the two contributes nothing either way.
    # This keeps the cue alive on a uniformly wet scene, where the relative
    # term is uninformative but the absolute one is not.
    surr = water_prior[surround_mask > 0]
    if surr.size > 10:
        delta = mean_inside - float(np.mean(surr))
        rel_score = float(np.clip(0.5 + delta, 0.0, 1.0))
    else:
        rel_score = abs_score

    return 0.5 * abs_score + 0.5 * rel_score


def _specular_residual_score(residual_map: np.ndarray, mask: np.ndarray,
                             surround_mask: np.ndarray) -> float:
    """
    Cue 8: Non-diffuse residual energy (intrinsic decomposition).

    Fresnel reflectance rises steeply toward grazing incidence:

        R(theta) = R0 + (1 - R0) * (1 - cos theta)^5,   R0 ~ 0.02 for water

    Road photography looks ALONG the surface, so theta is typically 80-88
    degrees and R lands between roughly 0.40 and 0.84 — most of the radiance
    from a puddle is reflected, not diffuse. Intrinsic decomposition isolates
    that non-diffuse term in its residual channel, which turns the Fresnel
    argument from rhetoric into a measurement.

    Relative only, deliberately. The residual carries no absolute calibration —
    its scale depends on exposure and on the decomposition — so an absolute
    threshold would not transfer between images.

    IMPORTANT — measured, not assumed. Over 13 real regions the positive-residual
    interior/ring ratio ran min 0.193, median 0.712, max 1.181, with only 23%
    above 1.0. Potholes are recessed and shadowed, so they receive LESS direct
    light than the sunlit road around them and are therefore less specular in
    general. An earlier 1.0 -> 2.5 mapping clipped 77% of regions to exactly
    zero, and since the cue carries weight in the ensemble that made it a
    constant dilutant: the one image labelled wet dropped from 0.234 to 0.210
    when the cue was switched on.

    The range below is taken from that observed distribution. It is a scale
    derived from data, not a fitted decision boundary — the boundary is what
    scripts/eval_water_puddle1000.py exists to establish.

    Returns score in [0, 1] where high = strongly non-diffuse (water-like).
    """
    inside = residual_map[mask > 0]
    surr = residual_map[surround_mask > 0]
    if inside.size < 10 or surr.size < 10:
        return 0.0

    mean_surr = float(np.mean(surr))
    if mean_surr <= 1e-6:
        return 0.0

    ratio = float(np.mean(inside)) / mean_surr

    lo = water_config.RESIDUAL_RATIO_LOW
    hi = water_config.RESIDUAL_RATIO_HIGH
    score = (ratio - lo) / max(hi - lo, 1e-8)
    return float(np.clip(score, 0.0, 1.0))


def detect_water(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    depth_map: np.ndarray = None,
    curvature_features: Dict[str, Any] = None,
    water_prior_map: Optional[np.ndarray] = None,
    residual_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Detect water inside a pothole using a 6-cue edge-aware ensemble.

    v3 key improvements over v2:
    - Edge density cue: strongest anti-false-positive (gravel has 35% edges,
      water has 3%). This alone fixes pic-65 false positive.
    - Gradient smoothness: water has gradient ratio <0.7, gravel >1.9
    - Specular cue checks spatial clustering (clustered = water, diffuse = sun)
    - Surrounding region filters out grass/sky for accurate comparison
    - Roughness penalty: high edge density directly reduces final probability

    Args:
        image_rgb: Original image in RGB format (HxWx3, uint8).
        mask: Binary pothole mask (HxW, uint8, values {0, 1}).
        depth_map: Depth map (HxW, float). Optional — enables cue 6.
        curvature_features: Dict from extract_curvature_features(). Optional —
            required alongside depth_map for cue 6.
        water_prior_map: HxW float in [0, 1] from semantic_water.water_prior_map().
            Optional — enables cue 7. Computed once per IMAGE by the caller, not
            once per pothole, so the ViT forward pass is paid once regardless of
            how many regions were segmented.
        residual_map: HxW float from intrinsic_cues.residual_energy_map().
            Optional — enables cue 8. Same once-per-image contract.

    Returns:
        Dictionary with detection results and per-cue breakdown. Cues whose
        input was not supplied report None rather than 0.0, so an absent witness
        is distinguishable from one that voted against.
    """
    if mask is None or np.sum(mask) == 0:
        return {
            'is_water': False, 'water_probability': 0.0,
            'confidence_level': 'none',
            'edge_density_score': 0.0, 'gradient_score': 0.0,
            'specular_score': 0.0, 'color_score': 0.0,
            'saturation_score': 0.0, 'inconsistency_score': None,
            'semantic_score': None, 'residual_score': None,
            'cue_weights': {}, 'corrections_applied': False,
        }

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Build road-only surrounding mask (excludes grass, sky)
    surround_mask = _get_road_surround_mask(image_bgr, mask)

    # ── Compute all cue scores ──
    edge_score = _edge_density_score(image_gray, mask, surround_mask)
    gradient_score = _gradient_smoothness_score(image_gray, mask, surround_mask)
    specular_score = _specular_reflection_score(image_bgr, mask)
    color_score = _color_distribution_score(image_bgr, mask, surround_mask)
    saturation_score = _saturation_uniformity_score(image_bgr, mask)

    inconsistency_score = None
    if depth_map is not None and curvature_features is not None:
        inconsistency_score = _depth_appearance_inconsistency_score(
            curvature_features, depth_map, mask
        )

    semantic_score = None
    if water_prior_map is not None:
        semantic_score = _semantic_prior_score(water_prior_map, mask, surround_mask)

    residual_score = None
    if residual_map is not None:
        residual_score = _specular_residual_score(residual_map, mask, surround_mask)

    # ── Weighted ensemble ──
    # Weights now live in water_config so they can be overridden by a fit
    # against a labelled set. The defaults are the original hand-set values, so
    # behaviour with only the base cues available is unchanged.
    W = water_config.CUE_WEIGHTS
    weights = {k: W[k] for k in water_config.BASE_CUES}
    scores = {
        'edge_density': edge_score,
        'gradient': gradient_score,
        'specular': specular_score,
        'color': color_score,
        'saturation': saturation_score,
    }

    # Optional cues join the ensemble only when their input was supplied. The
    # division by total_weight below renormalises over whatever is present, so
    # an absent cue abstains rather than voting zero.
    for name, value in (
        ('inconsistency', inconsistency_score),
        ('semantic', semantic_score),
        ('residual', residual_score),
    ):
        if value is not None:
            weights[name] = W[name]
            scores[name] = value

    if water_config.USE_LOGISTIC:
        # Signed combination. Admits cues that argue AGAINST water, which the
        # normalised sum below cannot express — see water_config.USE_LOGISTIC.
        # Only cues present in BOTH the fitted coefficients and this call
        # contribute, so an absent optional cue abstains rather than being
        # treated as zero evidence.
        z = water_config.LOGISTIC_INTERCEPT
        for k, s in scores.items():
            c = water_config.LOGISTIC_COEF.get(k)
            if c is not None:
                z += c * s
        water_prob = 1.0 / (1.0 + float(np.exp(-np.clip(z, -30.0, 30.0))))
    else:
        total_weight = sum(weights.values())
        water_prob = (sum(weights[k] * scores[k] for k in weights)
                      / max(total_weight, 1e-8))

    # ── Non-linear corrections ──
    # These are the most overfitted part of the module — each was introduced to
    # fix one named image, and the constants are recorded in water_config so the
    # evaluation harness can disable them wholesale and measure how much of the
    # behaviour they account for.
    C = water_config.CORRECTIONS
    corrections_applied = bool(water_config.APPLY_CORRECTIONS)

    if corrections_applied:
        # Boost: the two strongest cues agreeing compresses toward the high end.
        if (edge_score > C['agreement_boost_threshold']
                and gradient_score > C['agreement_boost_threshold']):
            water_prob = (water_prob * C['agreement_boost_scale']
                          + C['agreement_boost_offset'])

        # Penalty: a rough interior hard-caps the probability. Introduced for
        # pic-65, where dry gravel in sunlight scored 47.1%.
        if edge_score < C['rough_cap_hard_edge']:
            water_prob = min(water_prob, C['rough_cap_hard_value'])
        elif edge_score < C['rough_cap_soft_edge']:
            water_prob = min(water_prob, C['rough_cap_soft_value'])

        # Penalty: high interior gradient scales the result down.
        if gradient_score < C['gradient_penalty_edge']:
            water_prob *= C['gradient_penalty_scale']

    water_prob = float(np.clip(water_prob, 0.0, 1.0))

    # ── Decision ──
    bands = water_config.CONFIDENCE_BANDS
    is_water = water_prob > water_config.DECISION_THRESHOLD

    if water_prob > bands['high']:
        confidence_level = 'high'
    elif water_prob > bands['medium']:
        confidence_level = 'medium'
    elif water_prob > bands['low']:
        confidence_level = 'low'
    else:
        confidence_level = 'none'

    return {
        'is_water': bool(is_water),
        'water_probability': round(water_prob, 4),
        'confidence_level': confidence_level,
        'edge_density_score': round(edge_score, 4),
        'gradient_score': round(gradient_score, 4),
        'specular_score': round(specular_score, 4),
        'color_score': round(color_score, 4),
        'saturation_score': round(saturation_score, 4),
        'inconsistency_score': round(inconsistency_score, 4) if inconsistency_score is not None else None,
        'semantic_score': round(semantic_score, 4) if semantic_score is not None else None,
        'residual_score': round(residual_score, 4) if residual_score is not None else None,
        'cue_weights': {k: round(v, 3) for k, v in weights.items()},
        'corrections_applied': corrections_applied,
    }
