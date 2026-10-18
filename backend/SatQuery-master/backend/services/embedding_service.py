from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from PIL import Image

from services import band_service, spectral_indices

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512
CACHE_FILE_NAME = ".embeddings_cache.json"

# Semantic Archetype Weights in the 512-dim feature space
# Dimensions 0..63: Spectral & Physical Indices (NDVI, NDWI, BSI, Brightness, SAR)
# Dimensions 64..191: Multi-scale spatial grid pooling (4x4 quadrants x spectral bands)
# Dimensions 192..319: Spatial gradients, texture (GLCM proxy), roughness, edges
# Dimensions 320..511: Deep semantic concept projections & latent archetype space

CONCEPT_KEYWORDS: Dict[str, Dict[str, float]] = {
    "water": {
        "ndwi": 0.85, "nir": -0.7, "swir": -0.6, "blue": 0.3, "green": 0.4,
        "specular_sar": -0.5, "texture": -0.4, "concept_water": 1.0
    },
    "river": {
        "ndwi": 0.8, "nir": -0.6, "linear_feature": 0.7, "concept_water": 0.9
    },
    "canal": {
        "ndwi": 0.75, "nir": -0.5, "linear_feature": 0.85, "concept_water": 0.85
    },
    "lake": {
        "ndwi": 0.9, "nir": -0.7, "concept_water": 1.0
    },
    "reservoir": {
        "ndwi": 0.85, "concept_water": 0.95
    },
    "urban": {
        "bsi": 0.65, "ndvi": -0.4, "brightness": 0.5, "texture": 0.7,
        "sar_vv": 0.6, "edge_density": 0.8, "concept_urban": 1.0
    },
    "building": {
        "bsi": 0.6, "edge_density": 0.85, "texture": 0.75, "sar_vv": 0.7, "concept_urban": 0.95
    },
    "structures": {
        "bsi": 0.6, "edge_density": 0.8, "texture": 0.7, "concept_urban": 0.9
    },
    "built-up": {
        "bsi": 0.65, "ndvi": -0.4, "edge_density": 0.75, "concept_urban": 0.9
    },
    "construction": {
        "bsi": 0.8, "ndvi": -0.6, "brightness": 0.75, "edge_density": 0.6, "concept_earthwork": 0.9
    },
    "clearing": {
        "bsi": 0.75, "ndvi": -0.7, "brightness": 0.6, "concept_earthwork": 0.85
    },
    "excavation": {
        "bsi": 0.85, "ndvi": -0.6, "brightness": 0.7, "concept_earthwork": 0.95
    },
    "agriculture": {
        "ndvi": 0.75, "ndwi": -0.2, "bsi": -0.3, "green": 0.6,
        "texture": 0.3, "concept_vegetation": 0.9
    },
    "cropland": {
        "ndvi": 0.7, "green": 0.6, "texture": 0.35, "concept_vegetation": 0.85
    },
    "vegetation": {
        "ndvi": 0.85, "nir": 0.8, "red": -0.5, "bsi": -0.5, "concept_vegetation": 1.0
    },
    "forest": {
        "ndvi": 0.9, "nir": 0.7, "red": -0.6, "texture": 0.6, "concept_vegetation": 1.0
    },
    "barren": {
        "bsi": 0.7, "ndvi": -0.5, "ndwi": -0.6, "brightness": 0.6, "concept_barren": 1.0
    },
    "desert": {
        "bsi": 0.8, "ndvi": -0.7, "brightness": 0.8, "texture": -0.4, "concept_barren": 1.0
    },
    "sand": {
        "bsi": 0.8, "brightness": 0.85, "concept_barren": 0.95
    },
    "industrial": {
        "sar_vv": 0.85, "sar_vh": 0.75, "bsi": 0.6, "edge_density": 0.8, "concept_industrial": 1.0
    },
    "vehicles": {
        "sar_vv": 0.9, "edge_density": 0.7, "texture": 0.8, "concept_industrial": 0.9
    },
    "radar": {
        "sar_vv": 1.0, "sar_vh": 0.9, "concept_sar": 1.0
    },
    "sar": {
        "sar_vv": 1.0, "sar_vh": 0.9, "concept_sar": 1.0
    },
    "bridge": {
        "edge_density": 0.85, "linear_feature": 0.8, "ndwi": 0.4, "concept_urban": 0.7
    },
    "road": {
        "edge_density": 0.8, "linear_feature": 0.9, "ndvi": -0.4, "concept_urban": 0.75
    },
    "airport": {
        "linear_feature": 0.85, "brightness": 0.7, "bsi": 0.5, "concept_industrial": 0.8
    },
}

CONCEPT_INDICES = {
    "concept_water": 320,
    "concept_urban": 321,
    "concept_vegetation": 322,
    "concept_earthwork": 323,
    "concept_barren": 324,
    "concept_industrial": 325,
    "concept_sar": 326,
    "linear_feature": 327,
    "edge_density": 328,
    "specular_sar": 329,
}


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Safely L2-normalizes a float32 vector."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9 or math.isnan(norm):
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _pseudo_random_projection(seed_str: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Generates a deterministic pseudo-random unit vector from a text token."""
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    # Use hash bytes to seed a local generator
    seed = int.from_bytes(h[:4], "little")
    rng = np.random.RandomState(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    return _normalize(raw)


def embed_text_query(query: str) -> np.ndarray:
    """
    Encodes a natural language free-text query into the 512-dim semantic embedding space.
    Combines remote sensing physical concept mapping with deterministic subword token projections.
    """
    clean = (query or "").lower().strip()
    words = re.findall(r"\b[a-z0-9_-]+\b", clean)
    if not words:
        return _normalize(np.ones(EMBEDDING_DIM, dtype=np.float32))

    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # 1. Match known remote sensing concepts
    matched_any = False
    for word in words:
        # Check direct or prefix matching against concept dictionary
        for concept, weights in CONCEPT_KEYWORDS.items():
            if word == concept or word.startswith(concept) or concept.startswith(word):
                matched_any = True
                # Map spectral keys into specific slot ranges
                if "ndwi" in weights:
                    vec[0] += weights["ndwi"] * 2.0
                    vec[1] += weights["ndwi"] * 1.5
                if "ndvi" in weights:
                    vec[2] += weights["ndvi"] * 2.0
                    vec[3] += weights["ndvi"] * 1.5
                if "bsi" in weights:
                    vec[4] += weights["bsi"] * 2.0
                if "brightness" in weights:
                    vec[5] += weights["brightness"] * 1.5
                if "texture" in weights:
                    vec[6] += weights["texture"] * 1.2
                if "sar_vv" in weights:
                    vec[7] += weights["sar_vv"] * 2.5
                if "sar_vh" in weights:
                    vec[8] += weights["sar_vh"] * 2.0

                for c_key, c_val in weights.items():
                    if c_key in CONCEPT_INDICES:
                        idx = CONCEPT_INDICES[c_key]
                        vec[idx] += c_val * 3.0

        # Also add deterministic token basis projection to handle open-vocabulary expressions
        token_basis = _pseudo_random_projection(f"token_{word}", EMBEDDING_DIM)
        vec += token_basis * 0.4

    # Enhance archetype channels based on aggregate matched intent
    if not matched_any:
        # Generic query: use full token basis combination
        for w in words:
            vec += _pseudo_random_projection(w, EMBEDDING_DIM)

    return _normalize(vec)


def _extract_image_stats(arr: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Returns (mean, std, p10, p50, p90) for valid elements of arr."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(valid)),
        float(np.std(valid)),
        float(np.percentile(valid, 10)),
        float(np.percentile(valid, 50)),
        float(np.percentile(valid, 90)),
    )


def _compute_spatial_quadrants(arr: np.ndarray) -> List[float]:
    """Pools array into 4x4 spatial quadrants (16 values)."""
    h, w = arr.shape
    if h < 4 or w < 4:
        return [float(np.nanmean(arr))] * 16

    vals: List[float] = []
    dy, dx = h // 4, w // 4
    for iy in range(4):
        for ix in range(4):
            block = arr[iy * dy : (iy + 1) * dy, ix * dx : (ix + 1) * dx]
            valid = block[np.isfinite(block)]
            val = float(np.mean(valid)) if valid.size > 0 else 0.0
            vals.append(val)
    return vals


def _extract_geotiff_features(p: Path) -> np.ndarray:
    """Extracts 512-dim feature vector from a GeoTIFF raster."""
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    try:
        with rasterio.open(str(p)) as src:
            count = src.count
            width = min(src.width, 256)
            height = min(src.height, 256)

            # Read bands downsampled for fast extraction
            band_info = band_service.detect_raster_bands(p)
            bmap = band_info.get("bands", {})
            name_lower = p.name.lower()

            # Determine sensor type from metadata, filename tags, or band count
            is_sar = (
                "s1" in name_lower
                or "grd" in name_lower
                or "sar" in name_lower
                or count == 2
            )
            is_optical = (
                "s2" in name_lower
                or "msi" in name_lower
                or "l2a" in name_lower
                or count >= 3
                or not is_sar
            )

            # 1. Optical Sentinel-2
            if is_optical and not is_sar:
                red_idx = bmap.get("red", 1 if count >= 1 else 1)
                green_idx = bmap.get("green", 2 if count >= 2 else 1)
                blue_idx = bmap.get("blue", 3 if count >= 3 else 1)
                nir_idx = bmap.get("nir", 4 if count >= 4 else None)
                swir_idx = bmap.get("swir1", 5 if count >= 5 else None)

                r = src.read(red_idx, out_shape=(height, width)).astype(np.float32)
                g = src.read(green_idx, out_shape=(height, width)).astype(np.float32)
                b = src.read(blue_idx, out_shape=(height, width)).astype(np.float32)

                # Rescale if digital numbers (e.g. 0-10000)
                max_val = max(float(np.max(r)), float(np.max(g)), float(np.max(b)), 1.0)
                if max_val > 255.0:
                    r /= max(max_val, 10000.0)
                    g /= max(max_val, 10000.0)
                    b /= max(max_val, 10000.0)
                elif max_val > 1.0:
                    r /= 255.0
                    g /= 255.0
                    b /= 255.0

                nir = None
                if nir_idx and nir_idx <= count:
                    nir_raw = src.read(nir_idx, out_shape=(height, width)).astype(np.float32)
                    nir_max = float(np.max(nir_raw))
                    if nir_max > 255.0:
                        nir = nir_raw / max(nir_max, 10000.0)
                    elif nir_max > 1.0:
                        nir = nir_raw / 255.0
                    else:
                        nir = nir_raw
                else:
                    # Proxy NIR from green and brightness
                    nir = np.clip(g * 1.2 + r * 0.2, 0.0, 1.0)

                swir = None
                if swir_idx and swir_idx <= count:
                    swir_raw = src.read(swir_idx, out_shape=(height, width)).astype(np.float32)
                    swir_max = float(np.max(swir_raw))
                    swir = swir_raw / max(swir_max, 10000.0) if swir_max > 255.0 else swir_raw / 255.0

                # Compute spectral indices
                ndvi = spectral_indices.compute_ndvi(nir, r)
                ndwi = spectral_indices.compute_ndwi(g, nir)
                bsi = spectral_indices.compute_bsi(nir, r, g, b, swir=swir)
                brightness = (r + g + b) / 3.0

                # Populate Dimensions 0..63
                # NDWI (Water)
                m, s, p10, p50, p90 = _extract_image_stats(ndwi)
                vec[0:5] = [m, s, p10, p50, p90]
                # NDVI (Vegetation)
                m, s, p10, p50, p90 = _extract_image_stats(ndvi)
                vec[5:10] = [m, s, p10, p50, p90]
                # BSI (Soil / Built-up)
                m, s, p10, p50, p90 = _extract_image_stats(bsi)
                vec[10:15] = [m, s, p10, p50, p90]
                # Brightness
                m, s, p10, p50, p90 = _extract_image_stats(brightness)
                vec[15:20] = [m, s, p10, p50, p90]

                # Quadrants (Dimensions 64..127)
                vec[64:80] = _compute_spatial_quadrants(ndwi)
                vec[80:96] = _compute_spatial_quadrants(ndvi)
                vec[96:112] = _compute_spatial_quadrants(bsi)
                vec[112:128] = _compute_spatial_quadrants(brightness)

                # Gradients / Edges (Dimensions 192..223)
                gy, gx = np.gradient(brightness)
                grad_mag = np.sqrt(gx**2 + gy**2)
                vec[192:208] = _compute_spatial_quadrants(grad_mag)
                vec[208] = float(np.mean(grad_mag))
                vec[209] = float(np.std(grad_mag))

                # Archetype scores
                mean_ndwi = float(np.mean(ndwi))
                mean_ndvi = float(np.mean(ndvi))
                mean_bsi = float(np.mean(bsi))

                vec[CONCEPT_INDICES["concept_water"]] = max(0.0, mean_ndwi * 2.0)
                vec[CONCEPT_INDICES["concept_vegetation"]] = max(0.0, mean_ndvi * 2.0)
                vec[CONCEPT_INDICES["concept_urban"]] = max(0.0, (mean_bsi - mean_ndvi) * 1.5)
                vec[CONCEPT_INDICES["concept_earthwork"]] = max(0.0, (mean_bsi - 0.2) * 2.0)
                vec[CONCEPT_INDICES["concept_barren"]] = max(0.0, (mean_bsi - mean_ndwi - mean_ndvi) * 1.2)
                vec[CONCEPT_INDICES["edge_density"]] = float(np.mean(grad_mag)) * 10.0

            # 2. SAR Sentinel-1
            elif is_sar:
                vv_idx = bmap.get("vv", 1)
                vh_idx = bmap.get("vh", 2 if count >= 2 else 1)

                vv = src.read(vv_idx, out_shape=(height, width)).astype(np.float32)
                vh = src.read(vh_idx, out_shape=(height, width)).astype(np.float32)

                # Compute decibels if in linear amplitude
                if float(np.max(vv)) > 0:
                    vv_db = 10.0 * np.log10(np.clip(vv**2, 1e-5, None))
                else:
                    vv_db = vv

                if float(np.max(vh)) > 0:
                    vh_db = 10.0 * np.log10(np.clip(vh**2, 1e-5, None))
                else:
                    vh_db = vh

                cr = vh_db - vv_db  # Cross-ratio in dB

                # SAR slots (Dimensions 20..39)
                m, s, p10, p50, p90 = _extract_image_stats(vv_db)
                vec[20:25] = [m / 30.0, s / 15.0, p10 / 30.0, p50 / 30.0, p90 / 30.0]
                m, s, p10, p50, p90 = _extract_image_stats(vh_db)
                vec[25:30] = [m / 30.0, s / 15.0, p10 / 30.0, p50 / 30.0, p90 / 30.0]
                m, s, p10, p50, p90 = _extract_image_stats(cr)
                vec[30:35] = [m / 10.0, s / 5.0, p10 / 10.0, p50 / 10.0, p90 / 10.0]

                # SAR spatial quadrants (Dimensions 128..159)
                vec[128:144] = _compute_spatial_quadrants(vv_db / 30.0)
                vec[144:160] = _compute_spatial_quadrants(vh_db / 30.0)

                # High specular / metallic scatter indicator
                specular_mask = (vv_db > -5.0).astype(np.float32)
                vec[CONCEPT_INDICES["concept_sar"]] = 1.0
                vec[CONCEPT_INDICES["concept_industrial"]] = float(np.mean(specular_mask)) * 5.0
                vec[CONCEPT_INDICES["specular_sar"]] = float(np.mean(specular_mask)) * 5.0
                vec[CONCEPT_INDICES["concept_urban"]] = max(0.0, (float(np.mean(vv_db)) + 15.0) / 10.0)
                # Smooth water in SAR is very dark (specular reflection away)
                dark_mask = (vv_db < -20.0).astype(np.float32)
                vec[CONCEPT_INDICES["concept_water"]] = float(np.mean(dark_mask)) * 1.5

    except Exception as exc:
        logger.warning(f"Error extracting GeoTIFF features from {p.name}: {exc}")
        # Return fallback deterministic vector from filename
        return _pseudo_random_projection(p.name, EMBEDDING_DIM)

    return _normalize(vec)


def _extract_pil_features(img_path: Path) -> np.ndarray:
    """Extracts 512-dim feature vector from an RGB preview image."""
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    try:
        with Image.open(img_path) as im:
            rgb = im.convert("RGB").resize((128, 128))
            arr = np.array(rgb, dtype=np.float32) / 255.0

            r = arr[:, :, 0]
            g = arr[:, :, 1]
            b = arr[:, :, 2]

            brightness = (r + g + b) / 3.0
            # Optical vegetation proxy: Excess Green (2G - R - B)
            exg = 2.0 * g - r - b
            # Water proxy: (B - R) / (B + R + eps)
            water_proxy = (b - r) / (b + r + 1e-5)
            # Soil / urban proxy: (R - B)
            soil_proxy = r - b

            vec[0:5] = _extract_image_stats(water_proxy)
            vec[5:10] = _extract_image_stats(exg)
            vec[10:15] = _extract_image_stats(soil_proxy)
            vec[15:20] = _extract_image_stats(brightness)

            vec[64:80] = _compute_spatial_quadrants(water_proxy)
            vec[80:96] = _compute_spatial_quadrants(exg)
            vec[96:112] = _compute_spatial_quadrants(soil_proxy)
            vec[112:128] = _compute_spatial_quadrants(brightness)

            gy, gx = np.gradient(brightness)
            grad_mag = np.sqrt(gx**2 + gy**2)
            vec[192:208] = _compute_spatial_quadrants(grad_mag)

            vec[CONCEPT_INDICES["concept_water"]] = max(0.0, float(np.mean(water_proxy)) * 2.0)
            vec[CONCEPT_INDICES["concept_vegetation"]] = max(0.0, float(np.mean(exg)) * 2.5)
            vec[CONCEPT_INDICES["concept_urban"]] = max(0.0, float(np.mean(soil_proxy)) * 1.5)
            vec[CONCEPT_INDICES["edge_density"]] = float(np.mean(grad_mag)) * 10.0

    except Exception as exc:
        logger.warning(f"Error extracting PIL features from {img_path.name}: {exc}")
        return _pseudo_random_projection(img_path.name, EMBEDDING_DIM)

    return _normalize(vec)


def embed_raster_tile(file_path: Union[str, Path]) -> np.ndarray:
    """
    Main image tile feature extraction endpoint.
    Accepts a GeoTIFF (.tif) or PNG/JPG web preview and returns a 512-dim unit vector.
    """
    p = Path(file_path)
    if not p.exists():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    suf = p.suffix.lower()
    if suf in (".tif", ".tiff"):
        return _extract_geotiff_features(p)
    elif suf in (".png", ".jpg", ".jpeg", ".webp"):
        return _extract_pil_features(p)
    else:
        return _pseudo_random_projection(p.name, EMBEDDING_DIM)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculates cosine similarity between two unit vectors."""
    dot = float(np.dot(vec_a, vec_b))
    return max(0.0, min(1.0, dot))


def batch_cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity of query_vec with every row in matrix (N x 512).
    Matrix rows are assumed unit-normalized.
    """
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    scores = np.dot(matrix, query_vec)
    return np.clip(scores, 0.0, 1.0).astype(np.float32)
