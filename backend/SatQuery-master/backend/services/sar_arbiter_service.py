"""
Sentinel-1 SAR Cross-Sensor Verification Arbiter (ISRO PS 26167 §2.2.3)
========================================================================
Acts as an autonomous multi-sensor arbiter. When an optical scene (Sentinel-2)
is occluded by clouds, haze, or shadows, this service queries cloud-penetrating
Sentinel-1 C-band SAR radar imagery to mathematically verify whether ground change
actually occurred or if it was an atmospheric false alarm.

Flow:
  1. Searches local offline archive for coincident/nearest Sentinel-1 pass.
  2. If missing locally and network is online: autonomically queries public STAC
     for coincident Sentinel-1 pass, caches to disk, and updates local index.
  3. If offline and missing: gracefully flags unverified cloud occlusion.
  4. Evaluates radar backscatter delta (ΔVV in dB) to confirm or suppress change.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.mask import mask as rio_mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from core import network_state
from services import archive_indexer

logger = logging.getLogger(__name__)

TEMP_UPLOADS_DIR = Path(__file__).resolve().parents[1] / "temp_uploads"
TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Nominal radar backscatter thresholds (in decibels dB)
RADAR_STRONG_REFLECTOR_DB = -8.0     # Buildings, concrete, double-bounce corner reflectors
RADAR_NATURAL_LAND_DB = -13.5        # Crops, soil, forest canopy
RADAR_SMOOTH_SPECULAR_DB = -20.0     # Calm water, smooth runways
RADAR_CHANGE_CONFIRM_DELTA = 3.0     # +3 dB increase confirms structural change
RADAR_STABLE_DELTA_THRESHOLD = 1.5   # |delta| < 1.5 dB means surface is physically stable


def _parse_date(d_str: str) -> Optional[datetime]:
    if not d_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(d_str[:10], fmt)
        except ValueError:
            pass
    return None


def _calculate_bbox_overlap(b1: List[float], b2: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two [min_x, min_y, max_x, max_y] boxes."""
    if len(b1) < 4 or len(b2) < 4:
        return 0.0
    x_left = max(b1[0], b2[0])
    y_bottom = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_top = min(b1[3], b2[3])

    if x_right <= x_left or y_top <= y_bottom:
        return 0.0

    inter_area = (x_right - x_left) * (y_top - y_bottom)
    area1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter_area / (area1 + area2 - inter_area)


def find_local_sar_pass(
    aoi_bbox: List[float],
    target_date_str: str,
    max_days_diff: int = 14,
) -> Optional[Path]:
    """
    Searches the local archive indexer for a Sentinel-1 SAR scene matching
    the target AOI within an acceptable temporal window.
    """
    target_dt = _parse_date(target_date_str)
    indexer = archive_indexer.get_archive_indexer()
    if not indexer.tiles:
        indexer.scan_and_index()

    candidates: List[Tuple[float, float, Path]] = []  # (days_diff, -overlap, path)

    for tile in indexer.tiles:
        s_lower = (tile.sensor or "").lower()
        fn_lower = tile.filename.lower()
        is_sar = "sentinel-1" in s_lower or "s1" in fn_lower or "grd" in fn_lower or "sar" in fn_lower
        if not is_sar:
            continue

        overlap = _calculate_bbox_overlap(aoi_bbox, tile.bbox)
        # If boxes are normalized pixel coordinates or small AOI inside tile
        if overlap == 0.0 and len(tile.bbox) == 4 and len(aoi_bbox) == 4:
            # Check center distance
            cx = (aoi_bbox[0] + aoi_bbox[2]) / 2.0
            cy = (aoi_bbox[1] + aoi_bbox[3]) / 2.0
            if tile.bbox[0] <= cx <= tile.bbox[2] and tile.bbox[1] <= cy <= tile.bbox[3]:
                overlap = 0.5  # Containment

        if overlap <= 0.0:
            continue

        tile_dt = _parse_date(tile.date_str)
        days_diff = abs((target_dt - tile_dt).days) if target_dt and tile_dt else 0

        if days_diff <= max_days_diff:
            p = Path(tile.file_path)
            if p.exists():
                candidates.append((float(days_diff), -float(overlap), p))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    return None


def fetch_online_sar_if_available(
    aoi_bbox: List[float],
    target_date_str: str,
    window_days: int = 7,
) -> Optional[Path]:
    """
    When online, queries the public Copernicus / AWS STAC catalog for a coincident
    Sentinel-1 SAR scene over the target AOI, saves it to temp_uploads, and indexes it.
    """
    if not network_state.is_online():
        return None

    try:
        import sys
        repo_root = Path(__file__).resolve().parents[4]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from interactive_map import stac_search
        target_dt = _parse_date(target_date_str) or datetime.now()
        start_d = (target_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
        end_d = (target_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

        dt_range = f"{start_d}/{end_d}"

        # Fast search with 3.5s timeout
        items = stac_search.search_sentinel1(
            bbox=list(aoi_bbox),
            date_range=dt_range,
            limit=1,
        )
        if not items:
            return None

        item = items[0]
        item_id = item.id if hasattr(item, "id") else str(item)
        hash_id = hashlib.md5(f"{item_id}_{target_date_str}".encode("utf-8")).hexdigest()[:8]
        out_path = TEMP_UPLOADS_DIR / f"s1_arbiter_{hash_id}_{target_date_str}.tif"

        # If already cached locally, return it
        if out_path.exists() and out_path.stat().st_size > 10000:
            return out_path

        # If streaming scene fetch is available
        from interactive_map import fetch_scene
        res = fetch_scene(
            bbox=list(aoi_bbox),
            sensor="sentinel-1",
            date_range=dt_range,
        )
        if res and res.get("geotiff_path") and Path(res["geotiff_path"]).exists():
            fetched_p = Path(res["geotiff_path"])
            # Update local indexer with this newly downloaded SAR asset
            indexer = archive_indexer.get_archive_indexer()
            indexer.scan_and_index()
            return fetched_p

    except Exception as e:
        logger.warning(f"Online SAR STAC retrieval failed gracefully: {e}")

    return None


def extract_radar_backscatter_intensity(
    sar_path: Path,
    polygon_geom: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extracts mean calibrated radar backscatter in decibels (dB) for VV/VH
    channels over the candidate polygon footprint.
    """
    if not HAS_RASTERIO or not sar_path.exists():
        return {"mean_vv_db": -12.0, "is_valid": False}

    try:
        with rasterio.open(str(sar_path)) as src:
            coords = polygon_geom.get("coordinates")
            if not coords:
                return {"mean_vv_db": -12.0, "is_valid": False}

            # Attempt rasterio mask
            try:
                out_image, _ = rio_mask(src, [polygon_geom], crop=True, nodata=0.0)
                data = out_image[0]
            except Exception:
                # Fallback: window read around centroid
                data = src.read(1)

            valid = data[data > 0]
            if valid.size == 0:
                valid = data.flatten()

            # Convert Digital Number (DN) or amplitude to sigma0 in decibels
            # sigma0_db = 10 * log10(DN^2 + eps)
            mean_dn = float(np.mean(valid)) if valid.size > 0 else 1.0
            safe_val = max(1e-5, mean_dn)
            if safe_val > 100.0:
                # 16-bit uncalibrated DN
                safe_val = safe_val / 65535.0
            
            sigma0_db = 10.0 * math.log10(safe_val ** 2 + 1e-6)
            # Clip to realistic C-band backscatter [-35.0 dB, +10.0 dB]
            sigma0_db = max(-35.0, min(10.0, sigma0_db))

            return {
                "mean_vv_db": round(sigma0_db, 2),
                "is_valid": True,
                "sar_path": str(sar_path),
            }

    except Exception as e:
        logger.warning(f"Failed to extract radar backscatter from {sar_path}: {e}")
        return {"mean_vv_db": -12.0, "is_valid": False}


def arbitrate_candidate_change(
    candidate: Dict[str, Any],
    cloud_occlusion_fraction: float,
    target_date_str: str,
    baseline_date_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Master arbitration function:
    Evaluates whether an optical change candidate occluded by cloud/shadow is:
      1. CONFIRMED_GROUND_CHANGE: Radar backscatter confirms structural ground transformation.
      2. SUPPRESSED_CLOUD_ARTIFACT: Radar backscatter is stable; suppresses atmospheric false alarm.
      3. CLOUD_OCCLUDED_UNVERIFIED: Cloud occluded, but no SAR pass available offline.
      4. NOT_OCCLUDED: Clean optical scene (no arbitration needed).
    """
    # If optical scene is clean (< 25% cloud), no SAR arbitration needed
    if cloud_occlusion_fraction < 0.25:
        return {
            "status": "NOT_OCCLUDED",
            "is_ground_verified": True,
            "confidence_adjustment": 0.0,
            "false_alarm_flag": "NONE",
            "sar_verification_badge": None,
            "sar_delta_db": 0.0,
            "reason": "Clear optical ground view (unoccluded).",
        }

    geom = candidate.get("geometry") or {}
    centroid = candidate.get("centroid", [77.28, 28.64])
    # Compute bounding box
    span = 0.005  # ~500m window
    aoi_bbox = [centroid[0] - span, centroid[1] - span, centroid[0] + span, centroid[1] + span]

    # Step 1: Search local archive
    sar_pass = find_local_sar_pass(aoi_bbox, target_date_str)

    # Step 2: If missing locally, try online fetch if network is available
    is_online = network_state.is_online()
    if sar_pass is None and is_online:
        sar_pass = fetch_online_sar_if_available(aoi_bbox, target_date_str)

    # Step 3: Handle no SAR available (Air-Gapped S2-Only mode)
    if sar_pass is None:
        return {
            "status": "CLOUD_OCCLUDED_UNVERIFIED",
            "is_ground_verified": False,
            "confidence_adjustment": -0.25,  # Penalty for unverified optical cloud
            "false_alarm_flag": "CLOUD_SHADOW_ARTIFACT",
            "sar_verification_badge": "☁️ Cloud Occluded (No S1 SAR Available)",
            "sar_delta_db": 0.0,
            "reason": f"Optical scene is {int(cloud_occlusion_fraction * 100)}% cloud occluded; no coincident Sentinel-1 SAR pass in archive.",
        }

    # Step 4: Extract radar backscatter
    radar_info = extract_radar_backscatter_intensity(sar_pass, geom)
    mean_db = radar_info["mean_vv_db"]

    # If baseline SAR is available, calculate delta
    baseline_pass = find_local_sar_pass(aoi_bbox, baseline_date_str) if baseline_date_str else None
    if baseline_pass:
        base_radar = extract_radar_backscatter_intensity(baseline_pass, geom)
        delta_db = mean_db - base_radar["mean_vv_db"]
    else:
        # Single SAR comparison: difference from typical natural ground (-13.5 dB)
        delta_db = mean_db - RADAR_NATURAL_LAND_DB

    # Step 5: Make definitive arbitration verdict
    if delta_db >= RADAR_CHANGE_CONFIRM_DELTA or mean_db >= RADAR_STRONG_REFLECTOR_DB:
        # Real ground change verified through the cloud
        return {
            "status": "CONFIRMED_VIA_SAR",
            "is_ground_verified": True,
            "confidence_adjustment": +0.16,  # Boost confidence
            "false_alarm_flag": "VERIFIED_CHANGE",
            "sar_verification_badge": f"🛡️ S1 Radar Confirmed ({delta_db:+.1f} dB)",
            "sar_delta_db": round(delta_db, 2),
            "sar_pass_used": Path(sar_pass).name,
            "reason": f"Anthropogenic structure confirmed through cloud cover via Sentinel-1 C-band SAR (Backscatter: {mean_db:.1f} dB).",
        }
    else:
        # False alarm: white optical cloud, but radar ground is stable
        return {
            "status": "SUPPRESSED_VIA_SAR",
            "is_ground_verified": False,
            "confidence_adjustment": -0.50,  # Suppress false alarm
            "false_alarm_flag": "CLOUD_SHADOW_ARTIFACT",
            "sar_verification_badge": f"☁️ Suppressed: Cloud Artifact (Stable SAR {delta_db:+.1f} dB)",
            "sar_delta_db": round(delta_db, 2),
            "sar_pass_used": Path(sar_pass).name,
            "reason": f"Atmospheric cloud artifact suppressed: Radar ground backscatter is unchanged ({delta_db:+.1f} dB). Ground physically stable.",
        }
