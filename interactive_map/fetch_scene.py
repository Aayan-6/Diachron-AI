"""
AOI Satellite Scene Fetcher Orchestrator.
Main entry point connecting STAC search, windowed COG streaming,
preview rendering, and model radiometric normalization.
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import DEFAULT_MAX_CLOUD_COVER, FETCHED_DIR
from .stac_search import (
    search_sentinel2,
    search_sentinel1,
    get_best_scene,
    extract_item_metadata
)
from .cog_reader import (
    fetch_aoi_raster,
    compute_bbox_area_km2,
    validate_aoi
)
from .preview_generator import generate_preview, generate_web_overlay
from .preprocessing import preprocess_for_model


def fetch_scene(
    bbox: List[float],
    sensor: str = "sentinel-2",
    max_cloud_cover: int = DEFAULT_MAX_CLOUD_COVER,
    date_range: Optional[str] = None,
    preprocess: bool = True,
    generate_png: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Fetches satellite imagery covering an arbitrary bounding box on Earth.
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84 decimal degrees.
        sensor: "sentinel-2" (optical) or "sentinel-1" (SAR).
        max_cloud_cover: Maximum acceptable cloud coverage percent (Sentinel-2 only).
        date_range: Optional ISO interval, e.g. "2026-06-01T00:00:00Z/2026-09-12T00:00:00Z".
        preprocess: Whether to produce a normalized, model-ready GeoTIFF.
        generate_png: Whether to generate a contrast-enhanced PNG web preview.
        output_dir: Custom output directory (defaults to interactive_map/data/fetched).
        
    Returns:
        Dict containing execution status, paths to generated artifacts, and metadata.
    """
    t0 = time.time()
    sensor_normalized = sensor.lower().strip()
    
    # 1. Validate bounding box coordinates & area limit
    try:
        validate_aoi(bbox)
    except ValueError as ve:
        return {
            "status": "error",
            "error_type": "ValidationError",
            "message": str(ve)
        }

    area_sq_km = compute_bbox_area_km2(bbox)

    # 2. STAC Search
    t_search_start = time.time()
    if sensor_normalized in ["sentinel-1", "s1", "sar"]:
        sensor_type = "sentinel-1"
        items = search_sentinel1(bbox=bbox, date_range=date_range)
    else:
        sensor_type = "sentinel-2"
        items = search_sentinel2(
            bbox=bbox,
            max_cloud_cover=max_cloud_cover,
            date_range=date_range
        )
    t_search = time.time() - t_search_start

    if not items:
        # Check local offline archive for any local scene intersecting this bbox (Air-Gapped offline fallback)
        try:
            from services import archive_indexer
            from services.archive_indexer import _bboxes_intersect
            indexer = archive_indexer.get_archive_indexer()
            local_matches = [t for t in indexer.tiles if _bboxes_intersect(t.bbox, bbox)]
            if local_matches:
                best_local = local_matches[0]
                local_path = Path(best_local.file_path)
                if local_path.exists():
                    web_overlay = None
                    try:
                        from utils import gis_pipeline
                        web_overlay = gis_pipeline.build_web_overlay(str(local_path), output_dir=str(local_path.parent))
                    except Exception:
                        pass
                    return {
                        "status": "success",
                        "scene_id": best_local.tile_id,
                        "sensor": best_local.sensor,
                        "raw_geotiff_path": str(local_path),
                        "preview_png_path": str(best_local.thumbnail_url or ""),
                        "web_overlay": web_overlay,
                        "metadata": {
                            "date": best_local.date_str,
                            "cloud_cover": best_local.cloud_cover,
                            "sensor": best_local.sensor,
                            "bounds": best_local.bbox,
                            "source": "Local Archive / Ingested Folder (Offline)",
                        },
                        "total_latency_s": round(time.time() - t0, 2),
                    }
        except Exception:
            pass

        return {
            "status": "not_found",
            "message": f"No {sensor_type} scenes found for bbox {bbox}. (If offline, ensure you have ingested local GeoTIFFs covering this area via the Archive console)",
            "bbox": bbox,
            "sensor": sensor_type,
            "search_latency_s": round(t_search, 2)
        }

    best_item = get_best_scene(items, sensor=sensor_type)
    if not best_item:
        return {
            "status": "not_found",
            "message": f"Could not determine suitable scene from {len(items)} candidates.",
            "sensor": sensor_type
        }

    scene_meta = extract_item_metadata(best_item, sensor=sensor_type)

    # 3. Windowed COG Pixel Streaming
    t_stream_start = time.time()
    try:
        raw_geotiff_path, fetch_meta = fetch_aoi_raster(
            item=best_item,
            bbox=bbox,
            sensor=sensor_type,
            output_dir=output_dir
        )
    except Exception as e:
        return {
            "status": "error",
            "error_type": "DownloadError",
            "message": f"Failed streaming pixels from remote COG: {e}",
            "scene_id": best_item.id,
            "sensor": sensor_type
        }
    t_stream = time.time() - t_stream_start

    # 4. Web Preview PNG Generation & Web Mercator Slippy Map Overlay
    t_preview_start = time.time()
    preview_png_path = None
    web_overlay = None
    if generate_png:
        try:
            preview_png_path = generate_preview(
                geotiff_path=raw_geotiff_path,
                sensor_type=sensor_type
            )
            web_overlay = generate_web_overlay(
                geotiff_path=raw_geotiff_path,
                sensor_type=sensor_type,
                output_dir=Path(raw_geotiff_path).parent
            )
        except Exception as e:
            print(f"[Warning] Preview generation failed: {e}")
    t_preview = time.time() - t_preview_start

    # 5. Model-Ready Normalization
    t_proc_start = time.time()
    processed_geotiff_path = None
    if preprocess:
        try:
            processed_geotiff_path = preprocess_for_model(
                raw_geotiff_path=raw_geotiff_path,
                sensor_type=sensor_type
            )
        except Exception as e:
            print(f"[Warning] Model preprocessing failed: {e}")
    t_proc = time.time() - t_proc_start

    total_latency = time.time() - t0

    bounds_wgs84 = web_overlay.get("wgs84_bounds") if web_overlay else bbox

    return {
        "status": "success",
        "scene_id": best_item.id,
        "acquisition_date": scene_meta.get("datetime"),
        "cloud_cover": scene_meta.get("cloud_cover"),
        "sensor": "Sentinel-2 (Optical)" if sensor_type == "sentinel-2" else "Sentinel-1 (SAR)",
        "bbox": bbox,
        "wgs84_bounds": bounds_wgs84,
        "area_sq_km": round(area_sq_km, 2),
        "crs": fetch_meta.get("crs"),
        "bands": fetch_meta.get("bands"),
        "pixel_dimensions": fetch_meta.get("pixel_dimensions"),
        "raw_geotiff_path": raw_geotiff_path,
        "processed_geotiff_path": processed_geotiff_path,
        "preview_png_path": preview_png_path,
        "web_overlay_path": web_overlay.get("png_path") if web_overlay else preview_png_path,
        "web_overlay_name": web_overlay.get("png_name") if web_overlay else (Path(preview_png_path).name if preview_png_path else None),
        "file_size_mb": fetch_meta.get("file_size_mb"),
        "timings": {
            "search_s": round(t_search, 2),
            "stream_s": round(t_stream, 2),
            "preview_s": round(t_preview, 2),
            "preprocess_s": round(t_proc, 2),
            "total_s": round(total_latency, 2)
        }
    }


def fetch_temporal_stack(
    bbox: List[float],
    sensor: str = "sentinel-2",
    max_cloud_cover: int = DEFAULT_MAX_CLOUD_COVER,
    date_range: Optional[str] = None,
    max_scenes: int = 6,
    min_interval_days: int = 14,
    preprocess: bool = True,
    generate_png: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fetches an ordered multi-temporal stack of satellite imagery covering an AOI bounding box.

    Finds clear scenes across the temporal window, ensures minimum spacing between acquisitions,
    streams the raster windows, and returns them ordered chronologically (T0 -> Tn).
    """
    t0 = time.time()
    sensor_normalized = sensor.lower().strip()

    try:
        validate_aoi(bbox)
    except ValueError as ve:
        return {
            "status": "error",
            "error_type": "ValidationError",
            "message": str(ve),
        }

    area_sq_km = compute_bbox_area_km2(bbox)

    search_limit = max(max_scenes * 12, 60)
    if sensor_normalized in ["sentinel-1", "s1", "sar"]:
        sensor_type = "sentinel-1"
        items = search_sentinel1(bbox=bbox, date_range=date_range, limit=search_limit)
    else:
        sensor_type = "sentinel-2"
        items = search_sentinel2(
            bbox=bbox,
            max_cloud_cover=max_cloud_cover,
            date_range=date_range,
            limit=search_limit,
        )

    if not items:
        # Check local offline archive for temporal stack (Air-Gapped offline fallback)
        try:
            from services import archive_indexer
            indexer = archive_indexer.get_archive_indexer()
            local_stack = indexer.get_location_temporal_stack(bbox)
            if local_stack and len(local_stack) >= 1:
                scenes_payload = []
                for s_idx, ls in enumerate(local_stack[:max_scenes]):
                    scenes_payload.append({
                        "step_index": s_idx,
                        "date": ls.get("date"),
                        "acquisition_date": ls.get("date"),
                        "raw_geotiff_path": ls.get("file_path"),
                        "preview_png_path": ls.get("thumbnail_url"),
                        "web_overlay": {
                            "image_url": ls.get("thumbnail_url"),
                            "wgs84_bounds": ls.get("bbox"),
                            "source_crs": "EPSG:4326"
                        },
                        "metadata": {
                            "date": ls.get("date"),
                            "sensor": ls.get("sensor"),
                            "bounds": ls.get("bbox"),
                            "source": "Local Archive / Ingested Folder (Offline)",
                        }
                    })
                return {
                    "status": "success",
                    "bbox": bbox,
                    "sensor": sensor_type,
                    "scene_count": len(scenes_payload),
                    "scenes": scenes_payload,
                    "is_local_offline": True,
                }
        except Exception:
            pass

        return {
            "status": "not_found",
            "message": f"No {sensor_type} scenes found for bbox {bbox}. (If offline, ensure you have ingested local GeoTIFFs covering this area via the Archive console)",
            "bbox": bbox,
            "sensor": sensor_type,
            "scenes": [],
        }

    # Extract metadata and timestamps for selection
    dated_items = []
    for item in items:
        dt = item.datetime
        if not dt:
            continue
        meta = extract_item_metadata(item, sensor=sensor_type)
        dated_items.append((dt, meta.get("cloud_cover") or 0.0, item, meta))

    # Deduplicate scenes on the exact same date (take lowest cloud cover)
    day_bucket: Dict[str, Any] = {}
    for dt, cc, item, meta in dated_items:
        day_key = dt.strftime("%Y-%m-%d")
        if day_key not in day_bucket or cc < day_bucket[day_key][1]:
            day_bucket[day_key] = (dt, cc, item, meta)

    unique_dated_items = sorted(day_bucket.values(), key=lambda x: x[0])

    # Calculate effective interval: adapt if min_interval_days is wider than total span allows
    effective_interval = min_interval_days
    if len(unique_dated_items) >= max_scenes and max_scenes > 1:
        total_span = abs((unique_dated_items[-1][0] - unique_dated_items[0][0]).total_seconds()) / 86400.0
        max_possible_interval = total_span / (max_scenes - 1)
        if effective_interval > max_possible_interval * 1.15:
            effective_interval = max(5, int(max_possible_interval * 0.85))

    # Select temporally spaced scenes
    selected = []
    for dt, cc, item, meta in unique_dated_items:
        if not selected:
            selected.append((dt, cc, item, meta))
            continue

        last_dt = selected[-1][0]
        day_diff = abs((dt - last_dt).total_seconds()) / 86400.0

        if day_diff >= effective_interval:
            selected.append((dt, cc, item, meta))
        else:
            # If this new scene is noticeably clearer, replace the previous close scene
            if cc < selected[-1][1] - 5.0:
                selected[-1] = (dt, cc, item, meta)

        if len(selected) >= max_scenes:
            break

    # If filtering reduced items below desired max_scenes and more clear items are available,
    # fill intermediate gaps with the clearest candidates
    if len(selected) < max_scenes and len(unique_dated_items) > len(selected):
        selected_ids = {s[2].id for s in selected}
        remaining = [cand for cand in unique_dated_items if cand[2].id not in selected_ids]
        remaining.sort(key=lambda x: x[1])
        for cand in remaining:
            selected.append(cand)
            if len(selected) >= max_scenes:
                break
        selected.sort(key=lambda x: x[0])

    # If filtering reduced items below 2 and more items are available, relax interval
    if len(selected) < 2 and len(unique_dated_items) >= 2:
        selected = [unique_dated_items[0], unique_dated_items[-1]]

    fetched_scenes = []
    for idx, (dt, cc, item, meta) in enumerate(selected):
        try:
            raw_path, fetch_meta = fetch_aoi_raster(
                item=item,
                bbox=bbox,
                sensor=sensor_type,
                output_dir=output_dir,
            )

            preview_path = None
            web_overlay = None
            if generate_png:
                try:
                    preview_path = generate_preview(geotiff_path=raw_path, sensor_type=sensor_type)
                    web_overlay = generate_web_overlay(
                        geotiff_path=raw_path,
                        sensor_type=sensor_type,
                        output_dir=Path(raw_path).parent,
                    )
                except Exception as p_err:
                    print(f"[Warning] Preview failed for scene {item.id}: {p_err}")

            proc_path = None
            if preprocess:
                try:
                    proc_path = preprocess_for_model(raw_geotiff_path=raw_path, sensor_type=sensor_type)
                except Exception as m_err:
                    print(f"[Warning] Preprocessing failed for scene {item.id}: {m_err}")

            bounds_wgs84 = web_overlay.get("wgs84_bounds") if web_overlay else bbox
            overlay_name = web_overlay.get("png_name") if web_overlay else (Path(preview_path).name if preview_path else None)

            fetched_scenes.append({
                "step_index": idx,
                "scene_id": item.id,
                "acquisition_date": meta.get("datetime") or dt.strftime("%Y-%m-%d"),
                "cloud_cover": meta.get("cloud_cover"),
                "raw_geotiff_path": raw_path,
                "processed_geotiff_path": proc_path or raw_path,
                "preview_png_path": preview_path,
                "web_overlay_path": web_overlay.get("png_path") if web_overlay else preview_path,
                "web_overlay_name": overlay_name,
                "wgs84_bounds": bounds_wgs84,
                "crs": fetch_meta.get("crs"),
            })
        except Exception as e:
            print(f"[Error] Failed streaming scene {item.id}: {e}")
            continue

    total_latency = time.time() - t0

    return {
        "status": "success" if len(fetched_scenes) >= 2 else ("partial" if fetched_scenes else "error"),
        "sensor": "Sentinel-2 (Optical)" if sensor_type == "sentinel-2" else "Sentinel-1 (SAR)",
        "bbox": bbox,
        "area_sq_km": round(area_sq_km, 2),
        "count": len(fetched_scenes),
        "scenes": fetched_scenes,
        "total_latency_s": round(total_latency, 2),
    }

