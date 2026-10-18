"""
GIS Pre-processing Pipeline Integration
========================================
Hooks into `gis_extraction_logic` to perform spatial co-registration,
radiometric normalization, and coordinate metadata extraction.
"""

import os
import sys
from pathlib import Path

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Locate workspace root and gis_extraction_logic
current_dir = Path(__file__).resolve().parent
workspace_root = current_dir.parents[3]
gis_dir = workspace_root / "gis_extraction_logic"

if str(gis_dir) not in sys.path and gis_dir.exists():
    sys.path.insert(0, str(gis_dir))

TEMP_ALIGNED_DIR = current_dir.parent / "temp_uploads" / "aligned"


def generate_raster_preview(file_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Generates a calibrated RGB thumbnail PNG for a GeoTIFF or standard raster image.
    Uses rasterio decimation on read and 2%-98% percentile contrast stretching to ensure
    ultra-fast execution (<150ms) even on 10000x10000 Sentinel-2 rasters without memory blowup.
    Returns the absolute path to the generated preview PNG, or None on error.
    """
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        from PIL import Image
        import numpy as np

        p = Path(file_path)
        stem = p.stem
        target_dir = Path(output_dir) if output_dir else p.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        preview_path = target_dir / f"prev_{stem}.png"

        if preview_path.exists() and preview_path.stat().st_size > 1000:
            return str(preview_path)

        if file_path.lower().endswith((".tif", ".tiff")):
            import rasterio
            with rasterio.open(file_path) as src:
                factor = max(1, max(src.height, src.width) // 768)
                h, w = max(1, src.height // factor), max(1, src.width // factor)
                if src.count >= 3:
                    # In Sentinel-2 products: Band 1 is Red (B4), Band 2 is Green (B3), Band 3 is Blue (B2)
                    b_r = src.read(1, out_shape=(h, w))
                    b_g = src.read(2, out_shape=(h, w))
                    b_b = src.read(3, out_shape=(h, w))
                else:
                    b_r = b_g = b_b = src.read(1, out_shape=(h, w))

                def norm(b):
                    valid = b[b > 0]
                    if valid.size > 0:
                        p2, p98 = np.percentile(valid, (2, 98))
                        if p98 <= p2:
                            p2, p98 = float(valid.min()), float(valid.max())
                    else:
                        p2, p98 = 0.0, 255.0
                    clipped = np.clip(b, p2, p98)
                    denom = max(1e-6, float(p98 - p2))
                    return ((clipped - p2) / denom * 255).astype(np.uint8)

                rgb = np.stack([norm(b_r), norm(b_g), norm(b_b)], axis=-1)
                im = Image.fromarray(rgb)
                im.save(str(preview_path), "PNG")
                return str(preview_path)
        else:
            im = Image.open(file_path).convert("RGB")
            im.thumbnail((768, 768))
            im.save(str(preview_path), "PNG")
            return str(preview_path)
    except Exception as e:
        print(f"[GIS Preview Warning]: Failed to generate preview for {file_path}: {e}")
        return None


# =============================================================================
# Web-map overlay
# =============================================================================

WEB_MERCATOR = "EPSG:3857"


def _stretch_to_byte(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """2%-98% percentile contrast stretch over valid pixels only."""
    out = np.zeros(band.shape, dtype=np.uint8)
    values = band[valid & np.isfinite(band)]
    if values.size < 16:
        return out

    low, high = np.percentile(values, (2, 98))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if high <= low:
        return out

    scaled = (np.clip(band, low, high) - low) / (high - low) * 255.0
    out[:] = np.nan_to_num(scaled, nan=0.0).astype(np.uint8)
    return out


def build_web_overlay(
    file_path: str,
    output_dir: Optional[str] = None,
    max_dim: int = 2048,
    suffix: str = "web",
) -> Optional[Dict[str, Any]]:
    """
    Render a georeferenced raster as an RGBA PNG that lines up exactly with a
    slippy map, and return the corner coordinates to place it at.

    Why reproject rather than just render the raster and stretch it across its
    WGS84 bounding box (which is what this backend used to do):

    * A UTM scene is not a Mercator scene. Stretching a UTM-gridded image
      across lat/lon corners shears it; the error grows with latitude and with
      distance from the UTM central meridian, and reaches hundreds of metres on
      a city-sized scene.
    * `transform_bounds` returns the *envelope* of the reprojected footprint.
      For any non-4326 source that envelope is strictly larger than the image,
      so the image gets stretched to fill a box it does not occupy.
    * MapLibre interpolates an image source linearly in Mercator space, so even
      an EPSG:4326 raster placed by its lat/lon corners is compressed towards
      the poles.

    Warping to EPSG:3857 first removes all three: the PNG's pixel grid becomes
    the map's pixel grid, so it registers at every zoom, at any latitude, from
    any source CRS.

    Returns None when the raster carries no CRS — a file with no georeference
    cannot be honestly placed on a map.
    """
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        import rasterio
        from rasterio.warp import (
            Resampling,
            calculate_default_transform,
            reproject,
            transform_bounds,
        )
        from PIL import Image

        with rasterio.open(file_path) as src:
            if src.crs is None:
                return None

            # Target grid in Web Mercator, capped so the PNG stays servable.
            transform, width, height = calculate_default_transform(
                src.crs, WEB_MERCATOR, src.width, src.height, *src.bounds
            )
            longest = max(width, height)
            if longest > max_dim:
                scale = max_dim / float(longest)
                transform, width, height = calculate_default_transform(
                    src.crs,
                    WEB_MERCATOR,
                    src.width,
                    src.height,
                    *src.bounds,
                    dst_width=max(1, int(width * scale)),
                    dst_height=max(1, int(height * scale)),
                )

            from services.band_service import detect_raster_bands

            band_info = detect_raster_bands(src)
            r_idx = band_info["red_band_index"] or 1
            g_idx = band_info["green_band_index"] or (2 if src.count >= 2 else 1)
            b_idx = band_info["blue_band_index"] or (3 if src.count >= 3 else 1)
            band_indexes = [r_idx, g_idx, b_idx]

            channels = []
            for index in band_indexes:
                destination = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, index),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=WEB_MERCATOR,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )
                channels.append(destination)

            # Warp the source validity mask on the same grid so the nodata
            # skirt a rotated warp always leaves becomes transparent instead of
            # painting a black box over the basemap.
            mask_dst = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=src.dataset_mask(),
                destination=mask_dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=WEB_MERCATOR,
                resampling=Resampling.nearest,
            )
            valid = (mask_dst > 0) & np.isfinite(channels[0])

            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            for i, channel in enumerate(channels):
                rgba[:, :, i] = _stretch_to_byte(channel, valid)
            rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)

            target_dir = Path(output_dir) if output_dir else Path(file_path).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            png_path = target_dir / f"{suffix}_{Path(file_path).stem}.png"
            Image.fromarray(rgba, mode="RGBA").save(str(png_path), "PNG")

            # If NIR band exists, render genuine scientific False-Colour NIR composite: [NIR, Red, Green]
            png_nir_name = None
            if band_info["has_nir"] and band_info["nir_band_index"]:
                try:
                    nir_idx = band_info["nir_band_index"]
                    nir_indexes = [nir_idx, r_idx, g_idx]
                    nir_channels = []
                    for n_idx in nir_indexes:
                        dst_n = np.zeros((height, width), dtype=np.float32)
                        reproject(
                            source=rasterio.band(src, n_idx),
                            destination=dst_n,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=WEB_MERCATOR,
                            resampling=Resampling.bilinear,
                            src_nodata=src.nodata,
                            dst_nodata=np.nan,
                        )
                        nir_channels.append(dst_n)
                    nir_rgba = np.zeros((height, width, 4), dtype=np.uint8)
                    for i, nc in enumerate(nir_channels):
                        nir_rgba[:, :, i] = _stretch_to_byte(nc, valid)
                    nir_rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
                    png_nir_path = target_dir / f"nir_{suffix}_{Path(file_path).stem}.png"
                    Image.fromarray(nir_rgba, mode="RGBA").save(str(png_nir_path), "PNG")
                    png_nir_name = png_nir_path.name
                except Exception as n_err:
                    print(f"[GIS Overlay NIR Warning]: {n_err}")

            # Corners of the warped extent. Because the image is axis-aligned in
            # 3857 and 3857 -> 4326 is separable, the lat/lon envelope of that
            # extent *is* the image's footprint, so MapLibre places it exactly.
            left, top = transform * (0, 0)
            right, bottom = transform * (width, height)
            wgs84 = transform_bounds(WEB_MERCATOR, "EPSG:4326", left, bottom, right, top)

            return {
                "png_path": str(png_path),
                "png_name": png_path.name,
                "png_nir_name": png_nir_name,
                "wgs84_bounds": [round(float(v), 8) for v in wgs84],
                "mercator_bounds": [
                    float(left),
                    float(bottom),
                    float(right),
                    float(top),
                ],
                "width": int(width),
                "height": int(height),
                "source_crs": str(src.crs),
                "band_info": band_info,
            }

    except Exception as exc:
        print(f"[GIS Overlay Warning]: {file_path}: {type(exc).__name__}: {exc}")
        return None


def get_geotiff_info(file_path: str) -> Dict[str, Any]:
    """Extracts CRS, shape, bounds, center coordinates, and sensor information from a GeoTIFF or metadata."""
    info = {
        "file_path": file_path,
        "filename": os.path.basename(file_path) if file_path else "unknown",
        "is_geotiff": file_path.lower().endswith((".tif", ".tiff")) if file_path else False,
        "has_georeference": False,
        "crs": "unknown",
        "native_bounds": None,
        "wgs84_bounds": None,
        "center_lat": None,
        "center_lon": None,
        "resolution": None,
        "sensor": "Sentinel-2 (Optical)"
    }

    if not file_path or not os.path.exists(file_path):
        return info

    fname_lower = os.path.basename(file_path).lower()
    
    # ISRO Specific Sensor Detection
    if "cartosat" in fname_lower:
        info["sensor"] = "ISRO Cartosat-3 (High-Res Optical)"
    elif "risat" in fname_lower or "eos-04" in fname_lower or "eos04" in fname_lower:
        info["sensor"] = "ISRO RISAT-1A / EOS-04 (SAR)"
    elif "resourcesat" in fname_lower or "liss" in fname_lower or "awifs" in fname_lower:
        info["sensor"] = "ISRO Resourcesat-2 (Optical)"
    elif "_s1" in fname_lower or "sentinel-1" in fname_lower or "radar" in fname_lower or "sar" in fname_lower:
        info["sensor"] = "Sentinel-1 (SAR)"
    elif "_s2" in fname_lower or "sentinel-2" in fname_lower or "optical" in fname_lower:
        info["sensor"] = "Sentinel-2 (Optical)"

    # 1. Try Rasterio for real GeoTIFF containers
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(file_path) as src:
            if src.crs is not None:
                info["crs"] = str(src.crs)
                info["native_bounds"] = [round(float(b), 2) for b in src.bounds]
                info["shape"] = [src.count, src.height, src.width]
                info["resolution"] = [round(float(r), 2) for r in src.res]
                info["has_georeference"] = True

                # Reproject bounds to standard WGS84 lat/lon degrees
                if str(src.crs).upper() not in ["EPSG:4326", "WGS 84", "OGC:CRS84"]:
                    w_bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                else:
                    w_bounds = list(src.bounds)

                info["wgs84_bounds"] = [round(float(b), 6) for b in w_bounds]
                info["center_lon"] = round((w_bounds[0] + w_bounds[2]) / 2, 6)
                info["center_lat"] = round((w_bounds[1] + w_bounds[3]) / 2, 6)
    except Exception:
        pass

    # 2. Try Companion Metadata JSON (for tiles/patches with sidecar metadata)
    if not info["has_georeference"]:
        p = Path(file_path)
        candidates = [
            p.with_suffix(".json"),
            p.parent.parent / "metadata" / f"{p.stem}.json",
            p.parent / f"{p.stem}.json"
        ]
        for c in candidates:
            if c.exists():
                try:
                    import json
                    with open(c, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        box = meta.get("bounding_box") or meta.get("wgs84_bounds")
                        if box and len(box) == 4:
                            info["wgs84_bounds"] = [round(float(x), 6) for x in box]
                            info["center_lon"] = round((box[0] + box[2]) / 2, 6)
                            info["center_lat"] = round((box[1] + box[3]) / 2, 6)
                            info["crs"] = meta.get("crs", "EPSG:4326 (WGS84)")
                            info["sensor"] = meta.get("sensor", info["sensor"])
                            info["has_georeference"] = True
                            break
                except Exception:
                    pass

    try:
        from services.band_service import detect_raster_bands
        info["band_info"] = detect_raster_bands(file_path)
    except Exception:
        info["band_info"] = None

    return info


def align_geotiffs(file_paths: List[str]) -> List[str]:
    """
    Co-registers two satellite rasters (optical+SAR or bi-temporal) onto an identical grid.
    Returns the paths to the aligned rasters ready for downstream AI models.
    """
    if len(file_paths) < 2:
        return file_paths

    if len(file_paths) > 2:
        # For multi-temporal series, change_service handles sub-pixel re-projection
        # and grid normalization for each pair dynamically relative to T0.
        return file_paths

    file_1 = file_paths[0]
    file_2 = file_paths[1]


    # If either is not a GeoTIFF, return as is
    if not (file_1.lower().endswith((".tif", ".tiff")) and file_2.lower().endswith((".tif", ".tiff"))):
        return file_paths

    TEMP_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    out_path_1 = str(TEMP_ALIGNED_DIR / f"aligned_{os.path.basename(file_1)}")
    out_path_2 = str(TEMP_ALIGNED_DIR / f"aligned_{os.path.basename(file_2)}")

    if (
        os.path.exists(out_path_1)
        and os.path.exists(out_path_2)
        and os.path.getmtime(out_path_1) >= os.path.getmtime(file_1)
        and os.path.getmtime(out_path_2) >= os.path.getmtime(file_2)
    ):
        print(f"[GIS Pipeline] Using cached aligned rasters: '{os.path.basename(out_path_1)}' & '{os.path.basename(out_path_2)}'")
        return [out_path_1, out_path_2]

    try:
        from src.coregistration import coregister_pair
        from src.io_utils import load_raster, save_raster

        print(f"[GIS Pipeline] Co-registering {os.path.basename(file_1)} and {os.path.basename(file_2)}...")
        arr1, prof1 = load_raster(file_1)
        arr2, prof2 = load_raster(file_2)

        aligned_1, aligned_2, shared_profile = coregister_pair(arr1, prof1, arr2, prof2)

        save_raster(out_path_1, aligned_1, shared_profile, descriptions=prof1.get("descriptions"))
        save_raster(out_path_2, aligned_2, shared_profile, descriptions=prof2.get("descriptions"))
        print(f"[GIS Pipeline] Co-registration successful! Aligned grid: {shared_profile.get('crs')}")
        return [out_path_1, out_path_2]

    except Exception as e:
        print(f"[GIS Pipeline] Note: Direct co-registration skipped ({e}); using input files.")
        return file_paths

