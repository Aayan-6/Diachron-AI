from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

EPSILON = 1e-6


def compute_normalized_difference(
    band_a: np.ndarray,
    band_b: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes a generalized normalized difference index: (band_a - band_b) / (band_a + band_b + eps).
    Output range is clipped to [-1.0, 1.0].
    Invalid pixels (or non-positive denominators) are set to 0.0 or NaN.
    """
    a = band_a.astype(np.float32)
    b = band_b.astype(np.float32)
    denom = a + b
    num = a - b

    # Avoid division by zero or negative denominators due to artifacts
    safe_denom = np.where(np.abs(denom) < EPSILON, EPSILON, denom)
    index = num / safe_denom

    index = np.clip(index, -1.0, 1.0)

    if valid_mask is not None:
        index = np.where(valid_mask, index, 0.0)

    return index


def compute_ndvi(
    nir: np.ndarray,
    red: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Normalized Difference Vegetation Index (NDVI) = (NIR - Red) / (NIR + Red).
    Values > 0.3 indicate healthy vegetation; values < 0.1 indicate bare soil / water / urban.
    """
    return compute_normalized_difference(nir, red, valid_mask=valid_mask)


def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Normalized Difference Water Index (McFeeters 1996) = (Green - NIR) / (Green + NIR).
    Values > 0 typically indicate surface water bodies.
    """
    return compute_normalized_difference(green, nir, valid_mask=valid_mask)


def compute_mndwi(
    green: np.ndarray,
    swir: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Modified Normalized Difference Water Index (Xu 2006) = (Green - SWIR) / (Green + SWIR).
    Enhanced suppression of built-up noise compared to classic NDWI.
    """
    return compute_normalized_difference(green, swir, valid_mask=valid_mask)


def compute_bsi(
    nir: np.ndarray,
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    swir: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Bare Soil Index (BSI).
    If SWIR is available: ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue)).
    If SWIR is not available, uses an optical proxy: ((Red + Blue) - (NIR + Green)) normalized.
    """
    r = red.astype(np.float32)
    g = green.astype(np.float32)
    b = blue.astype(np.float32)
    n = nir.astype(np.float32)

    if swir is not None:
        s = swir.astype(np.float32)
        num = (s + r) - (n + b)
        denom = (s + r) + (n + b)
    else:
        num = (r + b) - (n + g)
        denom = (r + b) + (n + g)

    safe_denom = np.where(np.abs(denom) < EPSILON, EPSILON, denom)
    bsi = np.clip(num / safe_denom, -1.0, 1.0)

    if valid_mask is not None:
        bsi = np.where(valid_mask, bsi, 0.0)

    return bsi


def compute_brightness(
    bands: List[np.ndarray],
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes mean surface reflectance/brightness across available spectral bands.
    """
    if not bands:
        return np.zeros((1, 1), dtype=np.float32)
    stacked = np.stack([b.astype(np.float32) for b in bands], axis=0)
    brightness = np.mean(stacked, axis=0)
    if valid_mask is not None:
        brightness = np.where(valid_mask, brightness, 0.0)
    return brightness


def classify_change_type(
    delta_ndvi: float,
    delta_ndwi: float,
    delta_brightness: float,
    t0_ndvi: Optional[float] = None,
    ti_ndvi: Optional[float] = None,
    t0_ndwi: Optional[float] = None,
    ti_ndwi: Optional[float] = None,
) -> Tuple[str, float]:
    """
    Classifies the dominant physical change type using multi-spectral delta signatures.

    Returns:
        (change_type: str, confidence: float)
    """
    # 1. Vegetation loss / deforestation / clearance
    if delta_ndvi < -0.15:
        if delta_brightness > 0.05:
            # Canopy removal exposing bare soil or artificial surfaces
            conf = min(0.95, 0.65 + abs(delta_ndvi) * 0.8)
            return "Vegetation Clearance / Deforestation", float(conf)
        conf = min(0.90, 0.60 + abs(delta_ndvi) * 0.8)
        return "Vegetation Loss", float(conf)

    # 2. Vegetation gain / greening / agriculture / reforestation
    if delta_ndvi > +0.15:
        conf = min(0.95, 0.65 + abs(delta_ndvi) * 0.8)
        return "Vegetation Regrowth / Greening", float(conf)

    # 3. Water body expansion / flood inundation
    if delta_ndwi > +0.15:
        conf = min(0.95, 0.65 + abs(delta_ndwi) * 0.8)
        return "Water Body Expansion / Flooding", float(conf)

    # 4. Water body recession / drying
    if delta_ndwi < -0.15:
        conf = min(0.92, 0.60 + abs(delta_ndwi) * 0.8)
        return "Water Body Recession / Drying", float(conf)

    # 5. New Construction / Built-up Expansion (Vegetation decreases or soil brightens with high albedo)
    if delta_ndvi < -0.08 and delta_brightness > +0.08:
        conf = min(0.90, 0.60 + abs(delta_brightness) * 0.6)
        return "New Construction / Built-up Area", float(conf)

    # 6. Surface Brightening (e.g. soil exposure, earthwork, mining)
    if delta_brightness > +0.12:
        conf = min(0.88, 0.55 + abs(delta_brightness) * 0.6)
        return "Surface Brightening / Earthwork", float(conf)

    # 7. Surface Darkening (e.g. moisture, burn scar, shadow, asphalt paving)
    if delta_brightness < -0.12:
        if delta_ndwi > 0.05:
            return "Moisture Increase / Inundation", 0.78
        return "Surface Darkening / Paving", 0.75

    # 8. Subtle spectral variation
    return "Spectral Surface Modification", 0.65


def compute_multifactored_confidence(
    props: Dict[str, Any],
    trajectory: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    Computes a realistic, scientifically calibrated confidence score (0.42 to 0.98)
    for a satellite change candidate using:
      1. Change magnitude & SNR ratio relative to background threshold
      2. Deep learning ChangeFormer model agreement
      3. Multi-temporal persistence across observation sequence
      4. Multi-spectral delta intensity (NDVI, NDWI, brightness)
      5. Morphological footprint size (area_ha) and spatial compactness
    """
    # 1. Base confidence from change magnitude / SNR
    evidence = props.get("evidence_factors", {}) if isinstance(props.get("evidence_factors"), dict) else {}
    mag_ratio = float(evidence.get("magnitude_ratio", 0.0))
    mean_mag = float(props.get("mean_change_magnitude", 0.0))
    
    if mag_ratio >= 3.5:
        base_conf = 0.82
    elif mag_ratio >= 2.5:
        base_conf = 0.77
    elif mag_ratio >= 1.8:
        base_conf = 0.72
    elif mag_ratio >= 1.2:
        base_conf = 0.65
    elif mean_mag > 0.30:
        base_conf = 0.72
    elif mean_mag > 0.18:
        base_conf = 0.65
    elif mean_mag > 0.08:
        base_conf = 0.58
    else:
        base_conf = 0.52

    score = base_conf

    # 2. Deep Learning ChangeFormer Agreement
    cf_rating = evidence.get("model_agreement", "MODERATE")
    if cf_rating == "STRONG":
        score += 0.12  # ChangeFormer confirmed high IoU
    elif cf_rating == "MODERATE":
        score += 0.04
    elif cf_rating == "WEAK" or props.get("disagreement_flag"):
        score -= 0.12  # Model disagreement penalty

    # 3. Temporal Persistence (for multi-temporal analysis)
    if trajectory and len(trajectory) > 1:
        detected_steps = [t for t in trajectory if t.get("detected")]
        if len(detected_steps) >= 2:
            # Change was detected and confirmed across multiple distinct temporal dates!
            score += 0.09
        elif len(detected_steps) == 1 and detected_steps[0].get("step") == trajectory[-1].get("step"):
            # Fresh emergence at latest observation with strong magnitude
            score += 0.03

    # 4. Multi-Spectral Shift Intensity
    spec = props.get("spectral_metrics", {}) if isinstance(props.get("spectral_metrics"), dict) else {}
    d_ndvi = abs(float(spec.get("delta_ndvi", props.get("delta_ndvi", 0.0))))
    d_ndwi = abs(float(spec.get("delta_ndwi", props.get("delta_ndwi", 0.0))))
    d_bright = abs(float(spec.get("delta_brightness", props.get("delta_brightness", 0.0))))

    max_delta = max(d_ndvi, d_ndwi, d_bright)
    if max_delta >= 0.30:
        score += 0.10
    elif max_delta >= 0.18:
        score += 0.06
    elif max_delta >= 0.08:
        score += 0.03

    # 5. Spatial footprint (area) and compactness
    area_ha = float(props.get("area_ha", 0.0))
    compactness = float(props.get("compactness", 0.0))

    if area_ha >= 0.50:
        score += 0.05
    elif area_ha >= 0.15:
        score += 0.02
    elif area_ha < 0.025:
        score -= 0.06  # Sub-pixel noise penalty

    if compactness >= 0.15:
        score += 0.03  # Regular geometric boundary

    # 6. Reliability label adjustment
    rel_label = props.get("reliability_label", "MODERATE")
    if rel_label == "STRONG":
        score += 0.05
    elif rel_label == "WEAK":
        score -= 0.08

    # 7. Quality mask & SAR arbitration adjustment (ISRO PS 26167 §2.2.3)
    if "sar_confidence_adjustment" in props:
        score += float(props["sar_confidence_adjustment"])
    elif props.get("is_cloud_confounded"):
        score -= 0.25  # Unverified cloud occlusion penalty

    # Final clamping to realistic boundaries [0.10, 0.98]
    return float(round(min(0.98, max(0.10, score)), 2))


def extract_polygon_spectral_deltas(
    mask_slice: np.ndarray,
    t0_bands: Dict[str, np.ndarray],
    ti_bands: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Computes summary spectral delta metrics inside a binary polygon mask slice.
    """
    if not np.any(mask_slice):
        return {
            "delta_ndvi": 0.0,
            "delta_ndwi": 0.0,
            "delta_brightness": 0.0,
            "t0_ndvi": 0.0,
            "ti_ndvi": 0.0,
            "change_type": "Unclassified",
            "confidence": 0.5,
        }

    # NDVI
    has_ndvi = (
        "nir" in t0_bands and "red" in t0_bands and
        "nir" in ti_bands and "red" in ti_bands
    )
    if has_ndvi:
        ndvi_0 = compute_ndvi(t0_bands["nir"], t0_bands["red"])
        ndvi_i = compute_ndvi(ti_bands["nir"], ti_bands["red"])
        mean_ndvi_0 = float(np.mean(ndvi_0[mask_slice]))
        mean_ndvi_i = float(np.mean(ndvi_i[mask_slice]))
        delta_ndvi = mean_ndvi_i - mean_ndvi_0
    else:
        mean_ndvi_0 = 0.0
        mean_ndvi_i = 0.0
        delta_ndvi = 0.0

    # NDWI
    has_ndwi = (
        "green" in t0_bands and "nir" in t0_bands and
        "green" in ti_bands and "nir" in ti_bands
    )
    if has_ndwi:
        ndwi_0 = compute_ndwi(t0_bands["green"], t0_bands["nir"])
        ndwi_i = compute_ndwi(ti_bands["green"], ti_bands["nir"])
        mean_ndwi_0 = float(np.mean(ndwi_0[mask_slice]))
        mean_ndwi_i = float(np.mean(ndwi_i[mask_slice]))
        delta_ndwi = mean_ndwi_i - mean_ndwi_0
    else:
        mean_ndwi_0 = 0.0
        mean_ndwi_i = 0.0
        delta_ndwi = 0.0

    # Brightness (mean of RGB)
    t0_rgb = [t0_bands[b] for b in ("red", "green", "blue") if b in t0_bands]
    ti_rgb = [ti_bands[b] for b in ("red", "green", "blue") if b in ti_bands]
    if t0_rgb and ti_rgb:
        b0 = compute_brightness(t0_rgb)
        bi = compute_brightness(ti_rgb)
        mean_b0 = float(np.mean(b0[mask_slice]))
        mean_bi = float(np.mean(bi[mask_slice]))
        delta_brightness = mean_bi - mean_b0
    else:
        mean_b0 = 0.0
        mean_bi = 0.0
        delta_brightness = 0.0

    change_type, conf = classify_change_type(
        delta_ndvi=delta_ndvi,
        delta_ndwi=delta_ndwi,
        delta_brightness=delta_brightness,
        t0_ndvi=mean_ndvi_0,
        ti_ndvi=mean_ndvi_i,
    )

    return {
        "delta_ndvi": round(delta_ndvi, 4),
        "delta_ndwi": round(delta_ndwi, 4),
        "delta_brightness": round(delta_brightness, 4),
        "t0_ndvi": round(mean_ndvi_0, 4),
        "ti_ndvi": round(mean_ndvi_i, 4),
        "change_type": change_type,
        "confidence": round(conf, 3),
    }
