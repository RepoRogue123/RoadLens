from typing import Any, Dict, Optional
import cv2
import numpy as np
import scipy.stats
import scipy.signal


def extract_depth_features(
    mask: np.ndarray, depth_map: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Extract depth statistics from pixels where mask == 1.

    Args:
        mask: Binary mask array with foreground as 1.
        depth_map: Depth map array with same shape as mask.

    Returns:
        Dictionary containing:
        mean_depth, max_depth, min_depth, depth_std, depth_range,
        area, and p90_depth.
        Returns None when no pothole pixels are present in mask.
    """
    mask_array = np.asarray(mask)
    depth_array = np.asarray(depth_map, dtype=np.float32)

    if mask_array.shape != depth_array.shape:
        raise ValueError("mask and depth_map must have the same shape")

    # Normalize depth map to [0, 1] before computing features.
    depth_min_all = float(np.min(depth_array))
    depth_max_all = float(np.max(depth_array))
    if depth_max_all > depth_min_all:
        depth_array = (depth_array - depth_min_all) / (depth_max_all - depth_min_all)
    else:
        depth_array = np.zeros_like(depth_array, dtype=np.float32)

    depth_values = depth_array[mask_array == 1]
    area = int(depth_values.size)

    if area == 0:
        return None

    max_depth = float(np.max(depth_values))
    min_depth = float(np.min(depth_values))
    
    # Calculate local depth contrast (difference between hole and surrounding road context)
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(mask_array.astype(np.uint8), kernel, iterations=2)
    boundary_ring = dilated - mask_array.astype(np.uint8)
    depth_boundary = depth_array[boundary_ring > 0]
    
    mean_val = float(np.mean(depth_values))
    if len(depth_boundary) > 0 and len(depth_values) > 0:
        local_depth_contrast = float(abs(mean_val - np.mean(depth_boundary)))
    else:
        local_depth_contrast = 0.0

    return {
        "mean_depth": mean_val,
        "max_depth": max_depth,
        "min_depth": min_depth,
        "depth_std": float(np.std(depth_values)),
        "depth_range": float(max_depth - min_depth),
        "area": area,
        "p90_depth": float(np.percentile(depth_values, 90)),
        "local_depth_contrast": local_depth_contrast
    }

def extract_features(mask: np.ndarray, depth_map: np.ndarray) -> Optional[Dict[str, Any]]:
    h, w = mask.shape[:2]
    pothole_area = int(np.sum(mask))
    if pothole_area == 0:
        return None
        
    ys, xs = np.where(mask > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    p_width = max(1, x_max - x_min)
    p_height = max(1, y_max - y_min)
    box_area = p_width * p_height
    nonpothole_area = max(0, box_area - pothole_area)
    
    d = depth_map.astype(np.float32)
    d_min_all = float(np.min(d))
    d_max_all = float(np.max(d))
    if d_max_all > d_min_all:
        d = (d - d_min_all) / (d_max_all - d_min_all)
    else:
        d = np.zeros_like(d)
        
    depth_values = d[mask > 0]
    if len(depth_values) == 0:
        return None
        
    max_depth = float(np.max(depth_values))
    min_depth = float(np.min(depth_values))
    mean_depth = float(np.mean(depth_values))
    depth_std = float(np.std(depth_values))
    depth_range = max_depth - min_depth
    p90_depth = float(np.percentile(depth_values, 90))
    
    # Calculate local depth contrast (difference between hole and surrounding road context)
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    boundary_ring = dilated - mask.astype(np.uint8)
    depth_boundary = d[boundary_ring > 0]
    
    if len(depth_boundary) > 0 and len(depth_values) > 0:
        # Since disparity means closer=higher, we take absolute difference or look at the drop.
        # For a hole, depth disparity usually changes abruptly compared to the immediate flat ring.
        local_depth_contrast = float(abs(mean_depth - depth_boundary.mean()))
    else:
        local_depth_contrast = 0.0
    
    return {
        'height': p_height,
        'width': p_width,
        'box_area': box_area,
        'pothole_area': pothole_area,
        'nonpothole_area': nonpothole_area,
        'mean_depth': mean_depth,
        'max_depth': max_depth,
        'min_depth': min_depth,
        'depth_std': depth_std,
        'depth_range': depth_range,
        'p90_depth': p90_depth,
        'local_depth_contrast': local_depth_contrast
    }

def polygon_surface_area(mask, pixel_size_cm=0.5):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    n = len(contour)
    area_px = 0
    for i in range(n):
        j = (i + 1) % n
        xi, yi = contour[i][0]
        xj, yj = contour[j][0]
        area_px += xi * yj
        area_px -= xj * yi
    area_px = abs(area_px) / 2.0
    area_cm2 = area_px * (pixel_size_cm ** 2)
    return {
        'surface_area_px2': area_px,
        'surface_area_cm2': area_cm2,
        'pixel_size_cm_assumption': pixel_size_cm
    }

def extract_features_extended(mask, depth_map):
    orig_features = extract_features(mask, depth_map)
    if orig_features is None:
        return None
        
    height = orig_features['height']
    width = orig_features['width']
    pothole_area = orig_features['pothole_area']
    
    aspect_ratio = width / (height + 1e-8)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = pothole_area / (hull_area + 1e-8)
        
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (4 * np.pi * pothole_area) / (perimeter**2 + 1e-8)
    else:
        solidity = 0.0
        compactness = 0.0
        
    # Re-normalize depth for consistent feature extraction as done in extract_features
    d = depth_map.astype(np.float32)
    d_min_all = float(np.min(d))
    d_max_all = float(np.max(d))
    if d_max_all > d_min_all:
        d = (d - d_min_all) / (d_max_all - d_min_all)
    else:
        d = np.zeros_like(d)
        
    depth_values_inside_mask = d[mask > 0]
    
    depth_skewness = scipy.stats.skew(depth_values_inside_mask)
    if np.isnan(depth_skewness):
        depth_skewness = 0.0
    depth_kurtosis = scipy.stats.kurtosis(depth_values_inside_mask)
    if np.isnan(depth_kurtosis):
        depth_kurtosis = 0.0
        
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    boundary_ring = dilated - mask.astype(np.uint8)
    depth_boundary = d[boundary_ring > 0]
    
    if len(depth_boundary) > 0 and len(depth_values_inside_mask) > 0:
        boundary_gradient = orig_features['mean_depth'] - depth_boundary.mean()
    else:
        boundary_gradient = 0.0
        
    ext_features = orig_features.copy()
    
    ys, xs = np.where(mask > 0)
    cy, cx = ys.mean(), xs.mean()
    distances = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    weights = distances / (distances.sum() + 1e-8)
    weighted_mean_depth = (depth_values_inside_mask * weights).sum()
    
    poly_area = polygon_surface_area(mask)
    if poly_area is not None:
        surface_area_px2 = poly_area['surface_area_px2']
        surface_area_cm2 = poly_area['surface_area_cm2']
    else:
        surface_area_px2 = float(pothole_area)
        surface_area_cm2 = 0.0

    ext_features.update({
        'aspect_ratio': float(aspect_ratio),
        'solidity': float(solidity),
        'compactness': float(compactness),
        'depth_skewness': float(depth_skewness),
        'depth_kurtosis': float(depth_kurtosis),
        'boundary_gradient': float(boundary_gradient),
        'weighted_mean_depth': float(weighted_mean_depth),
        'surface_area_px2': float(surface_area_px2),
        'surface_area_cm2': float(surface_area_cm2)
    })
    
    return ext_features
        
    orig_features.update({
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'compactness': compactness,
        'depth_skewness': depth_skewness,
        'depth_kurtosis': depth_kurtosis,
        'boundary_gradient': boundary_gradient,
        'weighted_mean_depth': weighted_mean_depth,
        'surface_area_px2': surface_area_px2,
        'surface_area_cm2': surface_area_cm2
    })
    
    return orig_features

# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY-BASED SEVERITY FEATURES (Novel Research Extension)
# These features estimate pothole severity from mask boundary geometry and
# depth surface analysis, enabling severity classification that is robust to
# water-filled, mud-covered, or shadow-obscured potholes where monocular
# depth models fail.
# ──────────────────────────────────────────────────────────────────────────────


# ── Curvature smoothing window ───────────────────────────────────────────
# "fractional" scales the Savitzky-Golay window with contour length so the
# measurement scale is the same for a small pothole and a large one.
# "constant" restores the original min(n-1, 31) behaviour, kept so the two can
# be compared directly rather than assumed equivalent.
CURVATURE_WINDOW_MODE = "fractional"

# Fraction of the contour used as the smoothing window. 0.08 keeps roughly the
# same absolute window as the old constant at n≈390 (the middle of our observed
# range), so typical potholes are treated about as before while small and large
# ones stop being treated differently from each other.
CURVATURE_WINDOW_FRACTION = 0.08
CURVATURE_WINDOW_MIN = 5
CURVATURE_WINDOW_MAX = 51


def _curvature_window(n_points: int) -> int:
    """Odd Savitzky-Golay window length for a contour of `n_points`."""
    if CURVATURE_WINDOW_MODE == "constant":
        win = min(n_points - 1, 31)
    else:
        win = int(round(n_points * CURVATURE_WINDOW_FRACTION))
        win = min(win, n_points - 1, CURVATURE_WINDOW_MAX)

    if win % 2 == 0:
        win -= 1
    # polyorder is 3, so the window must exceed it.
    return max(win, CURVATURE_WINDOW_MIN)


def extract_curvature_features(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Extract curvature-based features from the pothole mask boundary contour.

    The boundary curvature of a pothole encodes severity information that is
    independent of what fills the pothole interior. Deep potholes have steep
    boundary walls producing high curvature at edges, regardless of whether
    the interior is dry, wet, or water-filled.

    Args:
        mask: Binary mask (HxW, uint8, values {0, 1}).

    Returns:
        Dictionary with 10 curvature features, or None if mask is empty or
        contour is too small for analysis.
    """
    if mask is None or np.sum(mask) == 0:
        return None

    # Extract full contour with all boundary pixels (no approximation)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour.squeeze()

    # Need at least enough points for the Savitzky-Golay filter
    if contour_pts.ndim != 2 or len(contour_pts) < 15:
        return None

    x = contour_pts[:, 0].astype(np.float64)
    y = contour_pts[:, 1].astype(np.float64)

    # Smooth contour coordinates to reduce pixel-level noise while preserving shape.
    #
    # The window MUST scale with contour length. The original code used a
    # constant `min(len(x) - 1, 31)`, which smooths a 40-point contour almost
    # out of existence (window covers ~78% of it) while barely touching an
    # 800-point one (~4%). Curvature is a second derivative, so the effective
    # measurement scale differs wildly between small and large potholes and
    # `max_curvature` is not comparable across them.
    #
    # That defect was not theoretical. Measured against real RealSense depth,
    # mean_curvature correlated -0.254 with true depth — but controlling for
    # area collapsed it to -0.041. The apparent signal was the window artefact:
    # bigger potholes are deeper AND less smoothed, so curvature fell as depth
    # rose. A fraction of contour length removes that confound.
    win_len = _curvature_window(len(x))

    x_smooth = scipy.signal.savgol_filter(x, window_length=win_len, polyorder=3, mode='wrap')
    y_smooth = scipy.signal.savgol_filter(y, window_length=win_len, polyorder=3, mode='wrap')

    # Compute first and second derivatives
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Discrete signed curvature: kappa = (dx*ddy - dy*ddx) / (dx^2 + dy^2)^(3/2)
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom[denom < 1e-10] = 1e-10  # avoid division by zero
    signed_curvature = (dx * ddy - dy * ddx) / denom
    abs_curvature = np.abs(signed_curvature)

    # --- Absolute curvature profile features ---
    max_curvature = float(np.max(abs_curvature))
    mean_curvature = float(np.mean(abs_curvature))
    std_curvature = float(np.std(abs_curvature))
    p90_curvature = float(np.percentile(abs_curvature, 90))

    # Fraction of boundary points with curvature > mean + 1*std
    threshold = mean_curvature + std_curvature
    high_curvature_fraction = float(np.mean(abs_curvature > threshold))

    # Curvature entropy from normalized distribution
    hist, _ = np.histogram(abs_curvature, bins=30, density=True)
    hist = hist[hist > 0]
    if len(hist) > 0:
        hist_norm = hist / hist.sum()
        curvature_entropy = float(-np.sum(hist_norm * np.log(hist_norm + 1e-12)))
    else:
        curvature_entropy = 0.0

    # --- Signed curvature profile features ---
    concave_fraction = float(np.mean(signed_curvature < 0))
    curvature_sign_changes = int(np.sum(np.diff(np.sign(signed_curvature)) != 0))

    # --- Contour geometry ---
    contour_length = float(cv2.arcLength(contour, True))
    rect = cv2.boundingRect(contour)
    contour_elongation = float(rect[2] / max(rect[3], 1))  # width / height

    return {
        'max_curvature': max_curvature,
        'mean_curvature': mean_curvature,
        'std_curvature': std_curvature,
        'p90_curvature': p90_curvature,
        'high_curvature_fraction': high_curvature_fraction,
        'curvature_entropy': curvature_entropy,
        'concave_fraction': concave_fraction,
        'curvature_sign_changes': curvature_sign_changes,
        'contour_length': contour_length,
        'contour_elongation': contour_elongation,
    }


def extract_depth_profile_features(
    mask: np.ndarray, depth_map: np.ndarray, n_slices: int = 8
) -> Optional[Dict[str, Any]]:
    """
    Extract cross-sectional depth profile features by fitting quadratic
    surfaces to the road region outside the pothole and extrapolating the
    expected road depth at the pothole center.

    This approach is novel because it estimates pothole bowl depth relative
    to the surrounding undamaged road surface rather than using raw depth
    values directly. Even if depth inside a water-filled pothole is wrong,
    the road surface extrapolation from outside the mask is still correct.

    Args:
        mask: Binary mask (HxW, uint8).
        depth_map: Depth map (HxW, float).
        n_slices: Number of angular cross-sections to sample.

    Returns:
        Dictionary with 5 depth profile features, or None on failure.
    """
    if mask is None or depth_map is None or np.sum(mask) == 0:
        return None

    h, w = mask.shape[:2]

    # Normalize depth to [0, 1]
    d = depth_map.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = np.zeros_like(d)

    # Find pothole centroid
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    cy, cx = float(ys.mean()), float(xs.mean())
    center_depth = float(d[int(cy), int(cx)])

    # Get bounding box with margin for road sampling
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    p_height = max(1, y_max - y_min)
    p_width = max(1, x_max - x_min)
    margin = max(p_height, p_width)  # sample road at 1x pothole size away

    bowl_depths = []
    road_curvatures = []
    road_slopes = []

    for i in range(n_slices):
        angle = np.pi * i / n_slices
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Sample points along this slice direction, extending beyond the pothole
        max_dist = int(margin * 1.5)
        sample_positions = np.arange(-max_dist, max_dist + 1)

        outside_positions = []
        outside_depths = []

        for t in sample_positions:
            px = int(cx + t * cos_a)
            py = int(cy + t * sin_a)
            if 0 <= px < w and 0 <= py < h:
                if mask[py, px] == 0:  # outside pothole
                    outside_positions.append(t)
                    outside_depths.append(float(d[py, px]))

        if len(outside_positions) < 6:
            continue

        positions_arr = np.array(outside_positions, dtype=np.float64)
        depths_arr = np.array(outside_depths, dtype=np.float64)

        # Fit quadratic to road surface outside the pothole
        try:
            coeffs = np.polyfit(positions_arr, depths_arr, 2)
            # Extrapolate road depth at center (t=0)
            extrapolated_road_depth = float(coeffs[2])  # c in at² + bt + c
            bowl_depth = abs(extrapolated_road_depth - center_depth)
            bowl_depths.append(bowl_depth)
            road_curvatures.append(abs(float(coeffs[0])))  # curvature = 2a
            road_slopes.append(abs(float(coeffs[1])))
        except (np.linalg.LinAlgError, ValueError):
            continue

    if not bowl_depths:
        return None

    return {
        'mean_bowl_depth': float(np.mean(bowl_depths)),
        'max_bowl_depth': float(np.max(bowl_depths)),
        'std_bowl_depth': float(np.std(bowl_depths)),
        'mean_road_curvature': float(np.mean(road_curvatures)),
        'slope_variance': float(np.var(road_slopes)),
    }


def extract_surface_normal_features(
    depth_map: np.ndarray, mask: np.ndarray, neighborhood_size: int = 15
) -> Optional[Dict[str, Any]]:
    """
    Extract surface normal deviation features at the pothole boundary.

    Steep pothole walls produce large angular deviations between the boundary
    surface normals and the reference road normal. This is directly related
    to severity — deeper potholes have steeper walls.

    Args:
        depth_map: Depth map (HxW, float).
        mask: Binary mask (HxW, uint8).
        neighborhood_size: Kernel size for gradient computation.

    Returns:
        Dictionary with 4 surface normal features, or None on failure.
    """
    if mask is None or depth_map is None or np.sum(mask) == 0:
        return None

    h, w = mask.shape[:2]

    # Normalize depth
    d = depth_map.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = np.zeros_like(d)

    # Compute depth gradients (surface slopes)
    ksize = min(neighborhood_size, 31)
    if ksize % 2 == 0:
        ksize -= 1
    ksize = max(ksize, 3)

    # Smooth depth before gradient computation
    d_smooth = cv2.GaussianBlur(d, (ksize, ksize), 0)
    dzdx = cv2.Sobel(d_smooth, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(d_smooth, cv2.CV_32F, 0, 1, ksize=3)

    # Surface normals: n = (-dzdx, -dzdy, 1) normalized
    norm_factor = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)
    nx = -dzdx / norm_factor
    ny = -dzdy / norm_factor
    nz = 1.0 / norm_factor

    # Compute reference normal from road region (outside mask)
    road_mask = (mask == 0).astype(np.uint8)
    road_pixels = np.sum(road_mask)
    if road_pixels < 10:
        return None

    ref_nx = float(np.mean(nx[road_mask > 0]))
    ref_ny = float(np.mean(ny[road_mask > 0]))
    ref_nz = float(np.mean(nz[road_mask > 0]))
    ref_norm = np.sqrt(ref_nx ** 2 + ref_ny ** 2 + ref_nz ** 2)
    if ref_norm < 1e-8:
        return None
    ref_nx /= ref_norm
    ref_ny /= ref_norm
    ref_nz /= ref_norm

    # Extract boundary ring (dilated mask minus original mask)
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    boundary_ring = dilated - mask.astype(np.uint8)
    boundary_ring = np.clip(boundary_ring, 0, 1)

    boundary_pixels = np.where(boundary_ring > 0)
    if len(boundary_pixels[0]) < 5:
        return None

    # Compute angular deviation at each boundary pixel
    b_nx = nx[boundary_pixels]
    b_ny = ny[boundary_pixels]
    b_nz = nz[boundary_pixels]

    # Dot product with reference normal
    dot_product = b_nx * ref_nx + b_ny * ref_ny + b_nz * ref_nz
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angular_deviation = np.arccos(dot_product) * (180.0 / np.pi)  # degrees

    return {
        'mean_normal_deviation': float(np.mean(angular_deviation)),
        'max_normal_deviation': float(np.max(angular_deviation)),
        'std_normal_deviation': float(np.std(angular_deviation)),
        'p90_normal_deviation': float(np.percentile(angular_deviation, 90)),
    }


def extract_all_geometry_features(
    mask: np.ndarray,
    depth_map: np.ndarray = None,
    image_rgb: np.ndarray = None,
) -> Optional[Dict[str, Any]]:
    """
    Combined geometry feature extractor — primary entry point for the
    geometry-based severity pipeline.

    Calls all geometry sub-extractors and merges results into a single dict.
    If any sub-function returns None, the combined result still returns what
    is available with missing features set to zero.

    Args:
        mask: Binary mask (HxW, uint8).
        depth_map: Depth map (HxW, float). Optional for curvature-only mode.
        image_rgb: Original image in RGB. Reserved for future extensions.

    Returns:
        Dictionary with all available geometry features, or None if mask is
        empty and no features can be extracted.
    """
    if mask is None or np.sum(mask) == 0:
        return None

    combined = {}

    # 1. Curvature features (mask only — no depth required)
    curvature = extract_curvature_features(mask)
    if curvature is not None:
        combined.update(curvature)
    else:
        # Set defaults for missing curvature features
        for key in ['max_curvature', 'mean_curvature', 'std_curvature',
                     'p90_curvature', 'high_curvature_fraction',
                     'curvature_entropy', 'concave_fraction',
                     'curvature_sign_changes', 'contour_length',
                     'contour_elongation']:
            combined[key] = 0.0

    # 2. Depth profile features (requires depth_map)
    if depth_map is not None:
        profiles = extract_depth_profile_features(mask, depth_map)
        if profiles is not None:
            combined.update(profiles)
        else:
            for key in ['mean_bowl_depth', 'max_bowl_depth', 'std_bowl_depth',
                         'mean_road_curvature', 'slope_variance']:
                combined[key] = 0.0

        # 3. Surface normal features (requires depth_map)
        normals = extract_surface_normal_features(depth_map, mask)
        if normals is not None:
            combined.update(normals)
        else:
            for key in ['mean_normal_deviation', 'max_normal_deviation',
                         'std_normal_deviation', 'p90_normal_deviation']:
                combined[key] = 0.0

    # 4. DINOv2 foundation features (optional — requires transformers)
    if image_rgb is not None:
        try:
            from foundation_features import HAS_DINOV2, extract_foundation_features
            if HAS_DINOV2:
                dino_feats = extract_foundation_features(image_rgb, mask)
                if dino_feats is not None:
                    combined.update(dino_feats)
        except ImportError:
            pass  # foundation_features.py not available — silently skip

    return combined if combined else None

