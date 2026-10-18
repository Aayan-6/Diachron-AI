"""
Cloud Optimized GeoTIFF (COG) Windowed Reader.
Streams only the requested Area of Interest (AOI) pixels via HTTP Range Requests,
avoiding full multi-gigabyte satellite scene downloads.
"""

import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from rasterio.transform import from_gcps, Affine
from pystac import Item

from .config import (
    S2_BANDS,
    S1_POLARIZATIONS,
    MAX_AOI_AREA_SQ_KM,
    FETCHED_DIR
)


def compute_bbox_area_km2(bbox: List[float]) -> float:
    """Computes the approximate surface area in square kilometers for a WGS84 bounding box."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    lat_dist = abs(max_lat - min_lat) * 111.32
    lon_dist = abs(max_lon - min_lon) * 111.32 * math.cos(math.radians(mid_lat))
    return float(lat_dist * lon_dist)


def validate_aoi(bbox: List[float]) -> None:
    """Validates that the bounding box is physically valid and within safety area limits."""
    if len(bbox) != 4:
        raise ValueError(f"BBox must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {bbox}")
    min_lon, min_lat, max_lon, max_lat = bbox
    
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValueError(f"Longitude values must be within [-180, 180], got [{min_lon}, {max_lon}]")
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError(f"Latitude values must be within [-90, 90], got [{min_lat}, {max_lat}]")
    if min_lon >= max_lon:
        raise ValueError(f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})")
    if min_lat >= max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})")

    area_sq_km = compute_bbox_area_km2(bbox)
    if area_sq_km > MAX_AOI_AREA_SQ_KM:
        raise ValueError(
            f"Requested AOI area ({area_sq_km:.1f} km²) exceeds maximum allowed cap of {MAX_AOI_AREA_SQ_KM} km²."
        )


def _resolve_url(href: str) -> str:
    """Converts S3 URI to public HTTPS URL if needed."""
    if href.startswith("s3://sentinel-s1-l1c/"):
        return href.replace("s3://sentinel-s1-l1c/", "https://sentinel-s1-l1c.s3.amazonaws.com/")
    return href


def fetch_aoi_sentinel2(
    item: Item,
    bbox: List[float],
    output_dir: Optional[Path] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Streams Sentinel-2 bands (Red, Green, Blue, NIR) covering the AOI.
    
    Returns:
        Tuple of (saved_geotiff_path, metadata_dict)
    """
    out_dir = output_dir or FETCHED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    clean_id = item.id.replace("/", "_")
    min_lon, min_lat, max_lon, max_lat = bbox
    filename = f"raw_S2_{clean_id[:25]}_{min_lon:.2f}_{min_lat:.2f}.tif"
    out_path = out_dir / filename

    # Identify band assets: prioritize B04, B03, B02, B08
    required_bands = ["red", "green", "blue", "nir"]
    band_urls = {}
    for b in required_bands:
        if b in item.assets:
            band_urls[b] = item.assets[b].href
            
    if "red" not in band_urls or "green" not in band_urls or "blue" not in band_urls:
        raise ValueError(f"STAC Item {item.id} is missing essential RGB band assets.")

    # Open first band to compute window and projection transform
    primary_band = "red"
    primary_url = _resolve_url(band_urls[primary_band])

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        "AWS_NO_SIGN_REQUEST": "YES"
    }

    with rasterio.Env(**env_kwargs):
        print(f"  Streaming primary optical band ({primary_band.upper()})...", flush=True)
        with rasterio.open(primary_url) as src:
            src_crs = src.crs
            src_transform = src.transform
            
            # Reproject WGS84 bbox to COG CRS
            bounds_proj = transform_bounds("EPSG:4326", src_crs, min_lon, min_lat, max_lon, max_lat)
            win = from_bounds(*bounds_proj, transform=src_transform)
            
            col_off = int(round(win.col_off))
            row_off = int(round(win.row_off))
            w = max(1, int(round(win.width)))
            h = max(1, int(round(win.height)))
            target_win = Window(col_off, row_off, w, h)
            
            win_transform = src.window_transform(target_win)
            dtype = src.dtypes[0]
            primary_arr = src.read(1, window=target_win, boundless=True, fill_value=0)

        # Read remaining bands windowed
        band_arrays = [primary_arr]
        band_names_loaded = [primary_band]
        for b_name in required_bands:
            if b_name == primary_band or b_name not in band_urls:
                continue
            b_url = _resolve_url(band_urls[b_name])
            print(f"  Streaming band {b_name.upper()}...", flush=True)
            with rasterio.open(b_url) as b_src:
                arr = b_src.read(1, window=target_win, boundless=True, fill_value=0)
                band_arrays.append(arr)
                band_names_loaded.append(b_name)

        stacked = np.stack(band_arrays, axis=0)  # Shape: (bands, h, w)

        # Write output GeoTIFF
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": len(band_arrays),
            "dtype": dtype,
            "crs": src_crs,
            "transform": win_transform,
            "compress": "deflate",
            "nodata": 0
        }

        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(stacked)
            for idx, b_name in enumerate(band_names_loaded, start=1):
                dst.set_band_description(idx, b_name.upper())

    meta = {
        "scene_id": item.id,
        "acquisition_date": item.datetime.isoformat() if item.datetime else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "sensor": "Sentinel-2",
        "crs": str(src_crs),
        "bands": band_names_loaded,
        "pixel_dimensions": [w, h],
        "file_size_bytes": out_path.stat().st_size,
        "file_size_mb": round(out_path.stat().st_size / (1024 * 1024), 2)
    }
    return str(out_path.resolve()), meta


def fetch_aoi_sentinel1(
    item: Item,
    bbox: List[float],
    output_dir: Optional[Path] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Streams Sentinel-1 SAR bands (VV, VH) covering the AOI.
    
    Returns:
        Tuple of (saved_geotiff_path, metadata_dict)
    """
    out_dir = output_dir or FETCHED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_id = item.id.replace("/", "_")
    min_lon, min_lat, max_lon, max_lat = bbox
    filename = f"raw_S1_{clean_id[:25]}_{min_lon:.2f}_{min_lat:.2f}.tif"
    out_path = out_dir / filename

    # Identify available SAR polarization assets (vv, vh)
    pol_urls = {}
    for pol in S1_POLARIZATIONS:
        if pol in item.assets:
            pol_urls[pol] = _resolve_url(item.assets[pol].href)

    if not pol_urls:
        raise ValueError(f"STAC Item {item.id} has neither VV nor VH polarization assets.")

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        "AWS_NO_SIGN_REQUEST": "YES"
    }

    primary_pol = "vv" if "vv" in pol_urls else list(pol_urls.keys())[0]
    primary_url = pol_urls[primary_pol]

    with rasterio.Env(**env_kwargs):
        print(f"  Streaming primary SAR band ({primary_pol.upper()})...", flush=True)
        with rasterio.open(primary_url) as src:
            dtype = src.dtypes[0]
            gcps, gcp_crs = src.gcps

            if gcps and len(gcps) >= 4:
                poly_transform = from_gcps(gcps)
                inv_transform = ~poly_transform
                # Calculate pixel box corners from WGS84 bbox
                corners = [
                    inv_transform * (min_lon, min_lat),
                    inv_transform * (min_lon, max_lat),
                    inv_transform * (max_lon, min_lat),
                    inv_transform * (max_lon, max_lat)
                ]
                cols = [c[0] for c in corners]
                rows = [c[1] for c in corners]
                col_min, col_max = min(cols), max(cols)
                row_min, row_max = min(rows), max(rows)

                col_off = int(round(col_min))
                row_off = int(round(row_min))
                w = max(1, int(round(col_max - col_min)))
                h = max(1, int(round(row_max - row_min)))
                target_win = Window(col_off, row_off, w, h)
                out_crs = gcp_crs or "EPSG:4326"
                out_transform = poly_transform * Affine.translation(col_off, row_off)
            else:
                # Fallback to projected coordinates if already georeferenced
                out_crs = src.crs or "EPSG:4326"
                bounds_proj = transform_bounds("EPSG:4326", out_crs, min_lon, min_lat, max_lon, max_lat)
                win = from_bounds(*bounds_proj, transform=src.transform)
                col_off = int(round(win.col_off))
                row_off = int(round(win.row_off))
                w = max(1, int(round(win.width)))
                h = max(1, int(round(win.height)))
                target_win = Window(col_off, row_off, w, h)
                out_transform = src.window_transform(target_win)

            primary_arr = src.read(1, window=target_win, boundless=True, fill_value=0)

        # Read remaining polarizations
        band_arrays = [primary_arr]
        pols_loaded = [primary_pol]
        for pol, p_url in pol_urls.items():
            if pol == primary_pol:
                continue
            print(f"  Streaming SAR band ({pol.upper()})...", flush=True)
            with rasterio.open(p_url) as p_src:
                arr = p_src.read(1, window=target_win, boundless=True, fill_value=0)
                band_arrays.append(arr)
                pols_loaded.append(pol)

        stacked = np.stack(band_arrays, axis=0)

        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": len(band_arrays),
            "dtype": dtype,
            "crs": out_crs,
            "transform": out_transform,
            "compress": "deflate",
            "nodata": 0
        }

        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(stacked)
            for idx, pol in enumerate(pols_loaded, start=1):
                dst.set_band_description(idx, pol.upper())

    meta = {
        "scene_id": item.id,
        "acquisition_date": item.datetime.isoformat() if item.datetime else None,
        "sensor": "Sentinel-1",
        "crs": str(out_crs),
        "bands": pols_loaded,
        "pixel_dimensions": [w, h],
        "file_size_bytes": out_path.stat().st_size,
        "file_size_mb": round(out_path.stat().st_size / (1024 * 1024), 2)
    }
    return str(out_path.resolve()), meta


def fetch_aoi_raster(
    item: Item,
    bbox: List[float],
    sensor: str = "sentinel-2",
    output_dir: Optional[Path] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Main entry point for AOI raster extraction.
    Validates the AOI and dispatches to the sensor-specific fetcher.
    """
    validate_aoi(bbox)
    
    if sensor == "sentinel-1":
        return fetch_aoi_sentinel1(item, bbox, output_dir)
    return fetch_aoi_sentinel2(item, bbox, output_dir)
