"""
Interactive Map STAC + COG API Router.
Provides endpoints for querying AWS Open Data STAC catalogs, streaming windowed
Cloud Optimized GeoTIFFs, and serving Web Mercator overlays to MapLibre GL.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Dynamically locate workspace root containing interactive_map
current_file = Path(__file__).resolve()
BACKEND_DIR = current_file.parent.parent
WORKSPACE_ROOT = None
for parent in current_file.parents:
    if (parent / "interactive_map").exists():
        WORKSPACE_ROOT = parent
        break

if WORKSPACE_ROOT and str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from interactive_map import fetch_scene
    from interactive_map.config import STAC_API_URL, MAX_AOI_AREA_SQ_KM
    from interactive_map.cog_reader import compute_bbox_area_km2
    MODULE_AVAILABLE = True
except Exception as e:
    MODULE_AVAILABLE = False
    MODULE_ERROR = str(e)

router = APIRouter()

TEMP_DIR = Path(BACKEND_DIR / "temp_uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class FetchAOIRequest(BaseModel):
    bbox: List[float] = Field(
        ...,
        description="[min_lon, min_lat, max_lon, max_lat] in WGS84 decimal degrees",
        min_items=4,
        max_items=4
    )
    sensor: str = Field(
        "sentinel-2",
        description="'sentinel-2' (optical) or 'sentinel-1' (SAR radar)"
    )
    max_cloud_cover: int = Field(
        20,
        description="Maximum acceptable cloud coverage (0-100) for optical scenes",
        ge=0,
        le=100
    )
    date_range: Optional[str] = Field(
        None,
        description="Optional ISO 8601 date interval, e.g. '2026-06-01T00:00:00Z/2026-09-13T00:00:00Z'"
    )


class FetchBiTemporalAOIRequest(BaseModel):
    bbox: List[float] = Field(
        ...,
        description="[min_lon, min_lat, max_lon, max_lat] in WGS84 decimal degrees",
        min_items=4,
        max_items=4
    )
    sensor: str = Field(
        "sentinel-2",
        description="'sentinel-2' (optical) or 'sentinel-1' (SAR radar)"
    )
    max_cloud_cover: int = Field(
        20,
        description="Maximum acceptable cloud coverage (0-100) for optical scenes",
        ge=0,
        le=100
    )
    date_t1: str = Field(
        ...,
        description="Target date or date range for T1 (earlier acquisition), e.g. '2024-01-15'"
    )
    date_t2: str = Field(
        ...,
        description="Target date or date range for T2 (later acquisition), e.g. '2025-01-15'"
    )


@router.get("/health")
def interactive_map_health():
    """Reports status of the STAC + COG satellite fetch engine."""
    return {
        "status": "online" if MODULE_AVAILABLE else "degraded",
        "module_loaded": MODULE_AVAILABLE,
        "error": MODULE_ERROR if not MODULE_AVAILABLE else None,
        "stac_endpoint": STAC_API_URL if MODULE_AVAILABLE else None,
        "max_aoi_cap_sq_km": MAX_AOI_AREA_SQ_KM if MODULE_AVAILABLE else 2500.0,
        "supported_sensors": ["sentinel-2", "sentinel-1"]
    }


@router.post("/fetch")
async def fetch_satellite_aoi(req: FetchAOIRequest):
    """
    Streams genuine Sentinel-1 or Sentinel-2 satellite imagery for any bounding box.
    Returns metadata formatted for the SatQuery frontend's SceneOverlay and SceneDataset.
    """
    if not MODULE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Interactive map streaming engine unavailable: {MODULE_ERROR}"
        )

    min_lon, min_lat, max_lon, max_lat = req.bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bounding box: min coordinates must be strictly less than max. Got {req.bbox}"
        )

    area_sq_km = compute_bbox_area_km2(req.bbox)
    if area_sq_km > MAX_AOI_AREA_SQ_KM:
        raise HTTPException(
            status_code=400,
            detail=f"Requested area ({area_sq_km:.1f} km²) exceeds maximum allowed cap of {MAX_AOI_AREA_SQ_KM} km²."
        )

    # Call Phase 1 engine with output directory set to backend/temp_uploads/
    try:
        res = await asyncio.to_thread(
            fetch_scene,
            bbox=req.bbox,
            sensor=req.sensor,
            max_cloud_cover=req.max_cloud_cover,
            date_range=req.date_range,
            preprocess=True,
            generate_png=True,
            output_dir=TEMP_DIR
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Remote STAC/COG streaming failure: {str(e)}"
        )

    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Imagery fetch error"))
    if res.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=res.get("message", "No matching satellite scenes found"))

    raw_path = Path(res["raw_geotiff_path"])
    t1_filename = raw_path.name
    
    # Use Web Mercator overlay for map if available, otherwise standard preview
    web_overlay_name = res.get("web_overlay_name")
    if not web_overlay_name and res.get("preview_png_path"):
        web_overlay_name = Path(res["preview_png_path"]).name

    wgs84_bounds = res.get("wgs84_bounds") or req.bbox
    center_lon = round((wgs84_bounds[0] + wgs84_bounds[2]) / 2.0, 6)
    center_lat = round((wgs84_bounds[1] + wgs84_bounds[3]) / 2.0, 6)

    is_optical = "sentinel-2" in req.sensor.lower()
    sensor_display = "Optical (Sentinel-2)" if is_optical else "SAR (Sentinel-1)"

    suggested_queries = [
        "What is the predominant land use and vegetative health in this scene?",
        "Identify urban infrastructure, roadways, and built-up structures.",
        "Analyze water bodies, agricultural fields, and natural canopy coverage."
    ] if is_optical else [
        "Identify open water bodies, rivers, and moisture contours in this SAR scene.",
        "Analyze radar backscatter roughness across built-up urban zones.",
        "Detect terrain changes, coastal structures, and surface characteristics."
    ]

    return {
        "status": "success",
        "dataset_id": f"aoi_{res.get('scene_id', 'scene')}",
        "name": t1_filename,
        "scene_id": res.get("scene_id"),
        "sensor": sensor_display,
        "mode": "single",
        "georeferenced": True,
        "wgs84_bounds": wgs84_bounds,
        "center": [center_lon, center_lat],
        "crs": res.get("crs", "EPSG:4326"),
        "resolution": "10.0m GSD",
        "area_sq_km": res.get("area_sq_km", round(area_sq_km, 2)),
        "t1_image_url": f"/static/{web_overlay_name}",
        "t1_filename": t1_filename,
        "acquisition_date": res.get("acquisition_date"),
        "cloud_cover": res.get("cloud_cover"),
        "file_size_mb": res.get("file_size_mb", 0.0),
        "bands": res.get("bands", []),
        "timings": res.get("timings", {}),
        "suggested_queries": suggested_queries
    }


def _to_stac_interval(date_input: str) -> Tuple[datetime, str]:
    """
    Parses date_input (e.g. '2024-01-15' or ISO interval) into an RFC 3339 interval for STAC search.
    If a single date is provided, generates a +/- 15-day window around it so STAC can find
    the closest valid orbital pass.
    """
    date_input = date_input.strip()
    if "/" in date_input:
        parts = date_input.split("/")
        d_str = parts[0].strip()
        if "T" in d_str:
            d1 = datetime.fromisoformat(d_str.replace("Z", "+00:00"))
        else:
            d1 = datetime.strptime(d_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return d1, date_input

    clean = date_input[:10]
    try:
        target_dt = datetime.strptime(clean, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            target_dt = datetime.fromisoformat(date_input.replace("Z", "+00:00"))
        except Exception:
            raise ValueError(f"Invalid date format: '{date_input}'. Expected YYYY-MM-DD or ISO 8601.")

    start_dt = target_dt - timedelta(days=15)
    end_dt = target_dt + timedelta(days=15)
    interval_str = f"{start_dt.strftime('%Y-%m-%d')}T00:00:00Z/{end_dt.strftime('%Y-%m-%d')}T23:59:59Z"
    return target_dt, interval_str


@router.post("/fetch-bitemporal")
async def fetch_bitemporal_satellite_aoi(req: FetchBiTemporalAOIRequest):
    """
    Streams a bi-temporal pair of genuine Sentinel-1 or Sentinel-2 satellite imagery
    covering the identical Area of Interest (AOI) bounding box across two dates (T1 and T2).
    Registers the pair for immediate ChangeFormer V6 execution via /api/v1/satquery.
    """
    if not MODULE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Interactive map streaming engine unavailable: {MODULE_ERROR}"
        )

    min_lon, min_lat, max_lon, max_lat = req.bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bounding box: min coordinates must be strictly less than max. Got {req.bbox}"
        )

    area_sq_km = compute_bbox_area_km2(req.bbox)
    if area_sq_km > MAX_AOI_AREA_SQ_KM:
        raise HTTPException(
            status_code=400,
            detail=f"Requested area ({area_sq_km:.1f} km²) exceeds maximum allowed cap of {MAX_AOI_AREA_SQ_KM} km²."
        )

    # 1. Validate date ordering
    try:
        t1_dt, interval_t1 = _to_stac_interval(req.date_t1)
        t2_dt, interval_t2 = _to_stac_interval(req.date_t2)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    if t1_dt >= t2_dt:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date ordering: The second date (T2: {req.date_t2}) must be strictly later than the first date (T1: {req.date_t1})."
        )

    # 2. Fetch T1 scene (exact same AOI)
    try:
        res_t1 = await asyncio.to_thread(
            fetch_scene,
            bbox=req.bbox,
            sensor=req.sensor,
            max_cloud_cover=req.max_cloud_cover,
            date_range=interval_t1,
            preprocess=True,
            generate_png=True,
            output_dir=TEMP_DIR
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"T1 Remote STAC/COG streaming failure: {str(e)}"
        )

    if res_t1.get("status") == "error":
        raise HTTPException(status_code=400, detail=f"T1 Error: {res_t1.get('message', 'Imagery fetch error')}")
    if res_t1.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"No matching satellite scene found for T1 ({req.date_t1}) in this AOI. Try adjusting cloud coverage or selecting an earlier date.")

    # 3. Fetch T2 scene (exact same AOI)
    try:
        res_t2 = await asyncio.to_thread(
            fetch_scene,
            bbox=req.bbox,
            sensor=req.sensor,
            max_cloud_cover=req.max_cloud_cover,
            date_range=interval_t2,
            preprocess=True,
            generate_png=True,
            output_dir=TEMP_DIR
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"T2 Remote STAC/COG streaming failure: {str(e)}"
        )

    if res_t2.get("status") == "error":
        raise HTTPException(status_code=400, detail=f"T2 Error: {res_t2.get('message', 'Imagery fetch error')}")
    if res_t2.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"No matching satellite scene found for T2 ({req.date_t2}) in this AOI. Try adjusting cloud coverage or selecting a different date.")

    raw_path_t1 = Path(res_t1["raw_geotiff_path"])
    raw_path_t2 = Path(res_t2["raw_geotiff_path"])
    t1_filename = raw_path_t1.name
    t2_filename = raw_path_t2.name

    if res_t1.get("scene_id") and res_t1.get("scene_id") == res_t2.get("scene_id"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Both requested dates resolved to the same satellite acquisition scene ({res_t1.get('scene_id')}). "
                "Please select dates further apart for meaningful bi-temporal change detection."
            )
        )

    if raw_path_t1 == raw_path_t2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Both requested dates resolved to the same underlying satellite imagery file. "
                "Please select dates further apart for meaningful bi-temporal change detection."
            )
        )

    web_overlay_t1 = res_t1.get("web_overlay_name") or (Path(res_t1["preview_png_path"]).name if res_t1.get("preview_png_path") else t1_filename)
    web_overlay_t2 = res_t2.get("web_overlay_name") or (Path(res_t2["preview_png_path"]).name if res_t2.get("preview_png_path") else t2_filename)

    wgs84_bounds = res_t1.get("wgs84_bounds") or req.bbox
    center_lon = round((wgs84_bounds[0] + wgs84_bounds[2]) / 2.0, 6)
    center_lat = round((wgs84_bounds[1] + wgs84_bounds[3]) / 2.0, 6)

    is_optical = "sentinel-2" in req.sensor.lower()
    sensor_display = "Optical (Sentinel-2 Bi-Temporal)" if is_optical else "SAR (Sentinel-1 Bi-Temporal)"

    # 4. Build authoritative band capability contract
    band_contract = None
    try:
        from services.band_service import build_band_capability_contract
        band_contract = build_band_capability_contract(str(raw_path_t1), str(raw_path_t2))
    except Exception as e:
        print(f"[Warning] Band capability evaluation skipped: {e}")

    # 5. Calculate actual temporal interval between acquired scenes
    acq_t1 = res_t1.get("acquisition_date")
    acq_t2 = res_t2.get("acquisition_date")
    interval_days = 0
    try:
        d1 = datetime.fromisoformat(acq_t1.replace("Z", "+00:00")) if acq_t1 else t1_dt
        d2 = datetime.fromisoformat(acq_t2.replace("Z", "+00:00")) if acq_t2 else t2_dt
        interval_days = max(1, abs((d2 - d1).days))
    except Exception:
        interval_days = max(1, abs((t2_dt - t1_dt).days))

    # 6. Compatibility validation & warnings
    compatibility_warnings = []
    if band_contract:
        t1_bands = band_contract.get("t1", {})
        t2_bands = band_contract.get("t2", {})
        t1_has_nir = t1_bands.get("has_nir", False)
        t2_has_nir = t2_bands.get("has_nir", False)
        if is_optical:
            if t1_has_nir and not t2_has_nir:
                compatibility_warnings.append("Asymmetric NIR availability: T1 contains NIR (Band 8), but T2 lacks NIR. False-color comparison will be restricted.")
            elif not t1_has_nir and t2_has_nir:
                compatibility_warnings.append("Asymmetric NIR availability: T2 contains NIR (Band 8), but T1 lacks NIR. False-color comparison will be restricted.")
            elif not t1_has_nir and not t2_has_nir:
                compatibility_warnings.append("NIR band absent in both scenes. Analysis will run in True-Color RGB mode.")

    # 7. Register with existing DATASET_REGISTRY so /api/v1/satquery immediately recognizes it
    ds_id = f"aoi_pair_{res_t1.get('scene_id', 't1')}_{res_t2.get('scene_id', 't2')}"
    comp_name = f"{t1_filename}:::{t2_filename}"
    try:
        from api.routes import DATASET_REGISTRY
        DATASET_REGISTRY[ds_id] = [str(raw_path_t1), str(raw_path_t2)]
        DATASET_REGISTRY[comp_name] = [str(raw_path_t1), str(raw_path_t2)]
        DATASET_REGISTRY[t1_filename] = [str(raw_path_t1), str(raw_path_t2)]
        print(f"[Interactive Map API] Bi-temporal AOI pair registered in DATASET_REGISTRY as '{ds_id}' and '{comp_name}'")
    except Exception as e:
        print(f"[Interactive Map API] Could not register in DATASET_REGISTRY: {e}")

    return {
        "status": "success",
        "dataset_id": ds_id,
        "name": comp_name,
        "sensor": sensor_display,
        "mode": "bi-temporal",
        "georeferenced": True,
        "wgs84_bounds": wgs84_bounds,
        "center": [center_lon, center_lat],
        "crs": res_t1.get("crs", "EPSG:4326"),
        "resolution": "10.0m GSD",
        "area_sq_km": res_t1.get("area_sq_km", round(area_sq_km, 2)),
        "interval_days": interval_days,
        "t1_image_url": f"/static/{web_overlay_t1}",
        "t2_image_url": f"/static/{web_overlay_t2}",
        "t1_filename": t1_filename,
        "t2_filename": t2_filename,
        "t1": {
            "requested_date": req.date_t1,
            "acquisition_date": acq_t1,
            "scene_id": res_t1.get("scene_id"),
            "cloud_cover": res_t1.get("cloud_cover"),
            "bands": res_t1.get("bands", []),
            "file_size_mb": res_t1.get("file_size_mb", 0.0),
            "filename": t1_filename,
            "image_url": f"/static/{web_overlay_t1}",
        },
        "t2": {
            "requested_date": req.date_t2,
            "acquisition_date": acq_t2,
            "scene_id": res_t2.get("scene_id"),
            "cloud_cover": res_t2.get("cloud_cover"),
            "bands": res_t2.get("bands", []),
            "file_size_mb": res_t2.get("file_size_mb", 0.0),
            "filename": t2_filename,
            "image_url": f"/static/{web_overlay_t2}",
        },
        "band_contract": band_contract,
        "compatibility": {
            "compatible": True,
            "same_aoi": True,
            "same_sensor": True,
            "interval_days": interval_days,
            "warnings": compatibility_warnings,
        },
        "suggested_queries": [
            "Detect and outline all bi-temporal changes between T1 and T2 using ChangeFormer.",
            "Analyze urban development and infrastructure expansion across these dates.",
            "Inspect vegetation loss and canopy reduction in the selected area."
        ] if is_optical else [
            "Detect surface change and flood water inundation between T1 and T2.",
            "Analyze SAR radar backscatter variations across these temporal passes."
        ]
    }
