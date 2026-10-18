"""
Optical Quality Mask & False-Alarm Detection Service (ISRO PS 26167 §2.2.3)
=============================================================================
Provides automated cloud, haze, and cast-shadow detection for optical satellite imagery
(Sentinel-2 MSI and RGB rasters). Eliminates confounding atmospheric artifacts before
and during change detection inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logger = logging.getLogger(__name__)

# Quality Mask Thresholds (Calibrated for Sentinel-2 Normalized Reflectance [0, 1])
CLOUD_BRIGHTNESS_MIN = 0.32       # Overall mean reflectance threshold
CLOUD_BLUE_MIN = 0.22             # Blue band threshold (clouds scatter heavily in blue)
CLOUD_WHITENESS_MAX = 0.16        # Max normalized color variance for neutral white clouds
SHADOW_NIR_MAX = 0.09             # Low NIR threshold for dark cast shadows
SHADOW_BRIGHTNESS_MAX = 0.14      # Max brightness threshold for shadows
MIN_CONFOUNDED_OVERLAP = 0.25     # Overlap fraction above which candidate is confounded


def _normalize_band_data(arr: np.ndarray) -> np.ndarray:
    """Safely normalizes any input raster array to [0.0, 1.0] float32."""
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    
    max_val = float(np.max(arr)) if arr.size > 0 else 1.0
    if max_val > 255.0:
        # Sentinel-2 Level-2A surface reflectance scaled by 10,000 (DN)
        return np.clip(arr / 10000.0, 0.0, 1.0)
    elif max_val > 1.0:
        # Standard 8-bit RGB [0, 255]
        return np.clip(arr / 255.0, 0.0, 1.0)
    else:
        # Already normalized [0, 1]
        return np.clip(arr, 0.0, 1.0)


def detect_optical_quality_mask(
    image_input: Union[str, Path, np.ndarray],
    transform: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Detects opaque clouds, semitransparent haze, and cloud shadows in an optical scene.
    
    Returns a dictionary containing:
      - cloud_mask: (H, W) boolean ndarray (True = cloud)
      - shadow_mask: (H, W) boolean ndarray (True = shadow)
      - quality_mask: (H, W) boolean ndarray (True = confounded pixel)
      - cloud_fraction: float (0.0 to 1.0)
      - shadow_fraction: float (0.0 to 1.0)
      - is_severely_occluded: bool (cloud_fraction > 0.30)
      - shape: Tuple[int, int] (H, W)
      - transform: raster transform if available
    """
    r_band = None
    g_band = None
    b_band = None
    nir_band = None
    affine_transform = transform

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Input scene raster not found: {p}")

        if HAS_RASTERIO and p.suffix.lower() in (".tif", ".tiff"):
            try:
                with rasterio.open(str(p)) as src:
                    affine_transform = src.transform
                    count = src.count
                    if count >= 4:
                        # Standard 4+ band raster: 1=B, 2=G, 3=R, 4=NIR (or similar)
                        b_band = _normalize_band_data(src.read(1))
                        g_band = _normalize_band_data(src.read(2))
                        r_band = _normalize_band_data(src.read(3))
                        nir_band = _normalize_band_data(src.read(4))
                    elif count == 3:
                        r_band = _normalize_band_data(src.read(1))
                        g_band = _normalize_band_data(src.read(2))
                        b_band = _normalize_band_data(src.read(3))
                    elif count == 1:
                        gray = _normalize_band_data(src.read(1))
                        r_band = g_band = b_band = gray
            except Exception as e:
                logger.warning(f"rasterio failed on {p}: {e}, falling back to PIL")

        if r_band is None:
            # Fallback using PIL
            img = Image.open(str(p)).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            r_band = arr[:, :, 0]
            g_band = arr[:, :, 1]
            b_band = arr[:, :, 2]

    elif isinstance(image_input, np.ndarray):
        arr = _normalize_band_data(image_input)
        if arr.ndim == 3:
            if arr.shape[0] in (3, 4):  # CHW format
                r_band = arr[0]
                g_band = arr[1]
                b_band = arr[2]
                if arr.shape[0] >= 4:
                    nir_band = arr[3]
            elif arr.shape[2] in (3, 4):  # HWC format
                r_band = arr[:, :, 0]
                g_band = arr[:, :, 1]
                b_band = arr[:, :, 2]
                if arr.shape[2] >= 4:
                    nir_band = arr[:, :, 3]
        elif arr.ndim == 2:
            r_band = g_band = b_band = arr

    if r_band is None or g_band is None or b_band is None:
        raise ValueError("Could not parse image input into RGB spectral bands.")

    h, w = r_band.shape

    # 1. Compute Mean Brightness & Whiteness (Color Flatness)
    brightness = (r_band + g_band + b_band) / 3.0
    safe_brightness = np.where(brightness < 1e-4, 1e-4, brightness)
    
    # Whiteness: sum of absolute deviations from mean, divided by mean
    whiteness = (
        np.abs(r_band - brightness)
        + np.abs(g_band - brightness)
        + np.abs(b_band - brightness)
    ) / safe_brightness

    # 2. Cloud Mask Heuristic
    # Clouds are bright, scatter blue, and exhibit low color saturation (white/neutral)
    cloud_mask = (
        (brightness >= CLOUD_BRIGHTNESS_MIN)
        & (b_band >= CLOUD_BLUE_MIN)
        & (whiteness <= CLOUD_WHITENESS_MAX)
    )

    # If NIR is available, refine cloud mask (clouds maintain high NIR reflectance)
    if nir_band is not None:
        # Exclude bright sandy/barren ground that has high red/SWIR but low blue
        cloud_mask = cloud_mask & (nir_band >= 0.20)

    # 3. Cloud Shadow Mask Heuristic
    # Shadows are dark overall, particularly in NIR (water/land absorb NIR in shadow)
    if nir_band is not None:
        shadow_mask = (nir_band <= SHADOW_NIR_MAX) & (brightness <= SHADOW_BRIGHTNESS_MAX)
    else:
        shadow_mask = (brightness <= SHADOW_BRIGHTNESS_MAX) & (b_band <= 0.16)

    # 4. Morphological Fringe Dilation
    # Atmospheric clouds have semi-transparent wispy boundaries; dilate by 3x3 to cover fringes
    struct = ndimage.generate_binary_structure(2, 2)
    cloud_dilated = ndimage.binary_dilation(cloud_mask, structure=struct, iterations=2)
    shadow_dilated = ndimage.binary_dilation(shadow_mask, structure=struct, iterations=2)

    combined_quality_mask = cloud_dilated | shadow_dilated

    total_pixels = float(h * w) if (h * w) > 0 else 1.0
    cloud_frac = float(np.sum(cloud_dilated)) / total_pixels
    shadow_frac = float(np.sum(shadow_dilated)) / total_pixels

    return {
        "cloud_mask": cloud_dilated,
        "shadow_mask": shadow_dilated,
        "quality_mask": combined_quality_mask,
        "cloud_fraction": round(cloud_frac, 4),
        "shadow_fraction": round(shadow_frac, 4),
        "is_severely_occluded": cloud_frac > 0.30,
        "shape": (h, w),
        "transform": affine_transform,
    }


def compute_polygon_quality_metrics(
    polygon_geom: Dict[str, Any],
    quality_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes the spatial overlap fraction between a change candidate polygon and
    the atmospheric quality mask (clouds and shadows).
    """
    q_mask = quality_result.get("quality_mask")
    c_mask = quality_result.get("cloud_mask")
    s_mask = quality_result.get("shadow_mask")
    transform = quality_result.get("transform")
    shape = quality_result.get("shape", (256, 256))

    if q_mask is None or not isinstance(polygon_geom, dict):
        return {
            "cloud_overlap": 0.0,
            "shadow_overlap": 0.0,
            "is_confounded": False,
            "confounding_factor": "CLEAR",
        }

    h, w = shape
    poly_raster = None

    if HAS_RASTERIO and transform is not None:
        try:
            poly_raster = rasterize(
                [(polygon_geom, 1)],
                out_shape=(h, w),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
        except Exception:
            poly_raster = None

    if poly_raster is None:
        # Fallback: Approximate bounding box in pixel coordinates
        coords = polygon_geom.get("coordinates")
        if coords:
            pts = []
            def _collect(o):
                if isinstance(o, (list, tuple)) and len(o) == 2 and isinstance(o[0], (int, float)):
                    pts.append(o)
                elif isinstance(o, (list, tuple)):
                    for s in o: _collect(s)
            _collect(coords)
            
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                poly_raster = np.zeros((h, w), dtype=np.uint8)
                if max_x <= 1.0 and max_y <= 1.0:
                    px1 = max(0, min(w - 1, int(min_x * w)))
                    px2 = max(0, min(w, int(max_x * w) + 1))
                    py1 = max(0, min(h - 1, int(min_y * h)))
                    py2 = max(0, min(h, int(max_y * h) + 1))
                    poly_raster[py1:py2, px1:px2] = 1
                else:
                    poly_raster.fill(1)

    if poly_raster is None or np.sum(poly_raster) == 0:
        return {
            "cloud_overlap": 0.0,
            "shadow_overlap": 0.0,
            "is_confounded": False,
            "confounding_factor": "CLEAR",
        }

    poly_pixels = float(np.sum(poly_raster))
    cloud_intersect = float(np.sum(poly_raster & c_mask)) if c_mask is not None else 0.0
    shadow_intersect = float(np.sum(poly_raster & s_mask)) if s_mask is not None else 0.0

    cloud_overlap = round(cloud_intersect / poly_pixels, 3)
    shadow_overlap = round(shadow_intersect / poly_pixels, 3)

    is_cloud = cloud_overlap >= MIN_CONFOUNDED_OVERLAP
    is_shadow = shadow_overlap >= MIN_CONFOUNDED_OVERLAP
    is_confounded = is_cloud or is_shadow

    factor = "CLEAR"
    if is_cloud and is_shadow:
        factor = "CLOUD_AND_SHADOW"
    elif is_cloud:
        factor = "CLOUD"
    elif is_shadow:
        factor = "SHADOW"

    return {
        "cloud_overlap": cloud_overlap,
        "shadow_overlap": shadow_overlap,
        "is_confounded": is_confounded,
        "confounding_factor": factor,
    }
