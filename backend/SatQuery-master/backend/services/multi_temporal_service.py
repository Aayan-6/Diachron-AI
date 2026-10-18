from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio

from services import change_service
from services.change_service import (
    OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
    _compute_spatial_analysis,
)
from services import band_service
from services import spectral_indices
from utils.markdown_formatter import format_markdown_table, sanitize_markdown

logger = logging.getLogger(__name__)


def extract_scene_date(file_path: Union[str, Path]) -> datetime:
    """
    Extracts the acquisition date of a satellite scene.
    Checks:
    1. Rasterio TIFFTAG_DATETIME or datetime metadata tags.
    2. Filename patterns: YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD, or YYYY.
    3. File creation / modification timestamp as fallback.
    """
    p = Path(file_path)

    # 1. Inspect Rasterio embedded metadata tags
    if p.exists() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
        try:
            with rasterio.open(str(p)) as ds:
                tags = ds.tags()
                for key in ("TIFFTAG_DATETIME", "datetime", "acquisition_date", "time"):
                    val = tags.get(key)
                    if val:
                        # Try parsing ISO or TIFF format: "YYYY:MM:DD HH:MM:SS" or "YYYY-MM-DD"
                        clean = val.replace(":", "-")[:10]
                        try:
                            return datetime.strptime(clean, "%Y-%m-%d")
                        except Exception:
                            pass
        except Exception:
            pass

    # 2. Filename regex matching
    # Strip any leading 32-char hex hash prefix if present (e.g. from temp upload uuid)
    clean_name = re.sub(r"^[a-f0-9]{32}_", "", p.stem.lower())

    # Pattern: 8-digit date like 20240129 or 20260112 (year 2014-2030)
    m8 = re.search(r"(?:^|[^0-9])(201[4-9]|202\d|2030)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?:$|[^0-9])", clean_name)
    if m8:
        try:
            return datetime(int(m8.group(1)), int(m8.group(2)), int(m8.group(3)))
        except ValueError:
            pass

    # Pattern: hyphen / underscore separated like 2024-01-29 or 2024_01_29
    m_sep = re.search(r"(?:^|[^0-9])(201[4-9]|202\d|2030)[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])(?:$|[^0-9])", clean_name)
    if m_sep:
        try:
            return datetime(int(m_sep.group(1)), int(m_sep.group(2)), int(m_sep.group(3)))
        except ValueError:
            pass

    # Pattern: standalone 4-digit year like 2018 or 2025 (bounded by non-alphanumeric)
    m_yr = re.search(r"(?:^|[^a-z0-9])(201[4-9]|202\d|2030)(?:$|[^a-z0-9])", clean_name)
    if m_yr:
        try:
            return datetime(int(m_yr.group(1)), 1, 1)
        except ValueError:
            pass


    # 3. Fallback: file modification time
    if p.exists():
        return datetime.fromtimestamp(p.stat().st_mtime)

    return datetime.now()


def sort_scenes_chronologically(file_paths: List[str]) -> List[Tuple[datetime, Path]]:
    """
    Sorts a collection of scene file paths by acquisition date in ascending order (T0 -> Tk).
    """
    dated: List[Tuple[datetime, Path]] = []
    for fp in file_paths:
        p = Path(fp)
        dt = extract_scene_date(p)
        dated.append((dt, p))

    dated.sort(key=lambda item: item[0])
    return dated


def _resolve_scene_photo_overlay(scene_path: Path) -> Dict[str, Any]:
    """
    Finds or builds a georeferenced web-mercator photo overlay PNG for a satellite scene.
    Returns: {"image_url": str, "nir_image_url": Optional[str], "wgs84_bounds": List[float], "crs": str}
    """
    from services.archive_indexer import _find_thumbnail_url
    from utils import gis_pipeline

    info = gis_pipeline.get_geotiff_info(str(scene_path))
    bounds = info.get("wgs84_bounds") or [77.20, 28.55, 77.36, 28.71]
    crs_str = info.get("crs", "EPSG:4326")

    url = _find_thumbnail_url(scene_path)

    clean_stem = re.sub(r"^[a-f0-9]{32}_", "", scene_path.stem)
    nir_candidate = scene_path.parent / f"nir_web_{clean_stem}.png"
    nir_url = f"/static/{nir_candidate.name}" if nir_candidate.exists() else None

    if url:
        return {
            "image_url": url,
            "nir_image_url": nir_url,
            "wgs84_bounds": bounds,
            "crs": crs_str,
        }

    # On-demand build
    try:
        ov = gis_pipeline.build_web_overlay(str(scene_path), output_dir=str(scene_path.parent))
        if ov and ov.get("png_name"):
            url = f"/static/{ov['png_name']}"
            if ov.get("png_nir_name"):
                nir_url = f"/static/{ov['png_nir_name']}"
            if ov.get("wgs84_bounds"):
                bounds = ov["wgs84_bounds"]
            return {
                "image_url": url,
                "nir_image_url": nir_url,
                "wgs84_bounds": bounds,
                "crs": ov.get("source_crs", crs_str),
            }
    except Exception as e:
        logger.warning(f"Could not build web overlay for {scene_path.name}: {e}")

    return {
        "image_url": None,
        "nir_image_url": None,
        "wgs84_bounds": bounds,
        "crs": crs_str,
    }


def _extract_outer_ring(coords_raw: Any) -> List[Tuple[float, float]]:
    """
    Safely flattens and extracts the outer boundary ring from arbitrary
    Polygon, MultiPolygon, or nested coordinate structures into a clean List[Tuple[float, float]].
    """
    if not coords_raw:
        return []

    curr = coords_raw
    # Keep unwrapping until we reach a list of 2-element coordinates
    while (
        isinstance(curr, (list, tuple))
        and len(curr) > 0
        and isinstance(curr[0], (list, tuple))
        and len(curr[0]) > 0
        and isinstance(curr[0][0], (list, tuple))
    ):
        curr = curr[0]

    points: List[Tuple[float, float]] = []
    if isinstance(curr, (list, tuple)):
        for item in curr:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    px = float(item[0])
                    py = float(item[1])
                    points.append((px, py))
                except (ValueError, TypeError):
                    continue
    return points


def _point_in_polygon(x: float, y: float, poly_coords: Any) -> bool:
    """Ray-casting point in polygon algorithm."""
    points = _extract_outer_ring(poly_coords)
    if len(points) < 3:
        return False
    n = len(points)
    inside = False
    p1x, p1y = points[0][0], points[0][1]
    for i in range(n + 1):
        p2x, p2y = points[i % n][0], points[i % n][1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _get_polygon_centroid_and_bbox(coords_input: Any) -> Tuple[Tuple[float, float], Tuple[float, float, float, float]]:
    """
    Returns ((center_x, center_y), (min_x, min_y, max_x, max_y)) from coordinates.
    """
    points = _extract_outer_ring(coords_input)
    if not points:
        return (0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    c_x = sum(xs) / float(len(xs))
    c_y = sum(ys) / float(len(ys))
    return (c_x, c_y), (min(xs), min(ys), max(xs), max(ys))


def _bbox_overlap(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> bool:
    """Checks if two [min_x, min_y, max_x, max_y] bounding boxes intersect."""
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])



def check_candidate_presence_in_earlier_pair(
    candidate_centroid: Tuple[float, float],
    candidate_bbox: Tuple[float, float, float, float],
    candidate_coords: List[List[float]],
    earlier_polygons: List[Dict[str, Any]],
    earlier_mask_path: Optional[Path] = None,
) -> bool:
    """
    Determines whether a change candidate region was already detected in an earlier temporal pair.
    Uses spatial overlap with earlier detected polygons and/or direct change mask sampling.
    """
    cx, cy = candidate_centroid

    # 1. Fast polygon geometry / bbox overlap check
    for ep in earlier_polygons:
        geom = ep.get("geometry", {})
        coords_raw = geom.get("coordinates")
        ring = _extract_outer_ring(coords_raw)
        if not ring:
            continue

        e_center, e_bbox = _get_polygon_centroid_and_bbox(ring)
        if _bbox_overlap(candidate_bbox, e_bbox):
            # Centroid inside earlier polygon or earlier centroid inside candidate
            if _point_in_polygon(cx, cy, ring):
                return True
            if _point_in_polygon(e_center[0], e_center[1], candidate_coords):
                return True


    # 2. Raster mask sampling at centroid if mask is available
    if earlier_mask_path and earlier_mask_path.exists():
        try:
            with rasterio.open(str(earlier_mask_path)) as src:
                # Sample mask value at candidate centroid (WGS84 or dataset CRS)
                # Need dataset row, col
                row, col = src.index(cx, cy)
                if 0 <= row < src.height and 0 <= col < src.width:
                    window = rasterio.windows.Window(max(0, col - 2), max(0, row - 2), 5, 5)
                    data = src.read(1, window=window)
                    if np.any(data > 0):
                        return True
        except Exception:
            pass

    return False


def _read_scene_bands(file_path: Path) -> Dict[str, np.ndarray]:
    """
    Reads available optical bands (red, green, blue, nir) into float32 numpy arrays.
    """
    band_info = band_service.detect_raster_bands(file_path)
    bands_dict: Dict[str, np.ndarray] = {}
    try:
        with rasterio.open(str(file_path)) as src:
            for bname in ("red", "green", "blue", "nir", "swir1"):
                bidx = band_info["bands"].get(bname)
                if bidx and 1 <= bidx <= src.count:
                    arr = src.read(bidx).astype(np.float32)
                    bands_dict[bname] = arr
    except Exception as e:
        logger.warning(f"Could not read spectral bands from {file_path.name}: {e}")

    return bands_dict


def synthesize_multi_temporal_markdown(
    baseline_date: datetime,
    latest_date: datetime,
    scenes_meta: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    dominant_type: str,
    total_changed_ha: float,
    cdqag_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates a clear and comprehensive GitHub-Flavored Markdown report for multi-temporal analysis,
    including CDQAG query-conditioned spatial grounding when a specific target concept is queried.
    """
    date_span_days = abs((latest_date - baseline_date).days)
    n_scenes = len(scenes_meta)

    # 1. Timeline Table
    timeline_headers = ["Step", "Acquisition Date", "Delta Area (ha)", "Change %", "Active Polygons", "Earliest Breakpoints"]
    timeline_rows = []
    for item in timeline:
        is_base = item.get("is_baseline", False)
        step_str = f"T{item['step']} (Baseline)" if is_base else f"T{item['step']}"
        date_str = item["date"]
        area_str = f"{item['change_area_ha']:.2f} ha" if not is_base else "- (0.0 ha)"
        pct_str = f"{item['change_fraction']*100:.2f}%" if not is_base else "0.0%"
        poly_str = str(item["n_polygons"]) if not is_base else "-"
        bp_str = f"{item.get('new_breakpoints', 0)} new" if not is_base else "Baseline"
        timeline_rows.append([step_str, date_str, area_str, pct_str, poly_str, bp_str])

    timeline_table_md = format_markdown_table(timeline_headers, timeline_rows)

    # 2. Candidates Breakdown Table
    has_target = bool(cdqag_summary and cdqag_summary.get("is_targeted"))
    if has_target:
        cand_headers = ["ID", "Change Type", "Earliest Observation", "Area (ha)", "Confidence", "Query Alignment"]
        cand_rows = []
        for c in candidates[:10]:
            alignment_str = f"🎯 Match ({int(c.get('semantic_relevance', 0.9)*100)}%)" if c.get("query_matched") else "Background"
            cand_rows.append([
                f"#{c['id']}",
                c["change_type"],
                f"**{c['earliest_observation']}**",
                f"{c['area_ha']:.2f} ha",
                f"{c['confidence']*100:.1f}%",
                alignment_str,
            ])
    else:
        cand_headers = ["ID", "Change Type", "Earliest Observation", "Latest Confirmation", "Area (ha)", "Confidence"]
        cand_rows = []
        for c in candidates[:10]:  # Top 10 prominent candidates
            cand_rows.append([
                f"#{c['id']}",
                c["change_type"],
                f"**{c['earliest_observation']}**",
                c["latest_observation"],
                f"{c['area_ha']:.2f} ha",
                f"{c['confidence']*100:.1f}%",
            ])

    cand_table_md = format_markdown_table(cand_headers, cand_rows) if cand_rows else "_No significant persistent change clusters identified._"

    # 3. CDQAG Grounding Section
    cdqag_section = ""
    if has_target and cdqag_summary:
        top_match = next((c for c in candidates if c.get("query_matched")), None)
        top_reason = f"\n- **Top Visual Confirmation:** Candidate #{top_match['id']} — *\"{top_match.get('grounding_reason')}\"*" if top_match and top_match.get("grounding_reason") else ""
        cdqag_section = rf"""---

#### 🎯 Query-Conditioned Spatio-Temporal Grounding (CDQAG)

- **Target Concept:** `{cdqag_summary.get('target_concept')}`
- **Query Alignment:** **{cdqag_summary.get('matched_count', 0)} of {cdqag_summary.get('total_proposals', 0)} proposals** matched the query footprint ({cdqag_summary.get('matched_area_ha', 0):.2f} ha matched).
- **Background Filtering:** {cdqag_summary.get('background_count', 0)} unaligned or seasonal background proposals were separated into the background pool.{top_reason}
"""

    # 4. Narrative Synthesis
    markdown = rf"""### Multi-Temporal Earth-Observation Analysis Report

**Temporal Baseline ($T_0$):** `{baseline_date.strftime('%Y-%m-%d')}`  
**Latest Acquisition ($T_k$):** `{latest_date.strftime('%Y-%m-%d')}`  
**Temporal Span:** `{date_span_days} days` ({n_scenes} sequential observations)  
**Total Cumulative Change:** `{total_changed_ha:.2f} ha`  
**Dominant Physical Process:** `{dominant_type}`

---

#### 1. Chronological Timeline Progression

{timeline_table_md}

---

#### 2. Key Change Candidates & Earliest Observation Dates

{cand_table_md}
{cdqag_section}
---

#### 3. Analytical Interpretation & Trajectory Insights

- **Earliest Change Emergence:** The primary change breakpoints were first confirmed on **`{candidates[0]['earliest_observation'] if candidates else baseline_date.strftime('%Y-%m-%d')}`**, demonstrating that physical ground transformation initiated well before the latest observation.
- **Physical Classification:** Analysis of multi-spectral signatures ($\Delta\text{{NDVI}}$, $\Delta\text{{NDWI}}$, and surface reflectance) indicates that the predominant change is **{dominant_type.lower()}**.
- **Interactive Map:** Scrub the **Timeline Slider** above the map to view the step-by-step visual progression and inspect the exact polygon footprints for each temporal window.
"""
    return sanitize_markdown(markdown)


async def run_inference(
    query: str,
    file_paths: List[str],
    tracer: Any,
    target_concept: Optional[str] = None,
    phenomenon: Optional[str] = None,
) -> Tuple[str, List[Any], List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Public Multi-Temporal Change Engine entrypoint used by core.agent.
    Accepts 2 or more files, arranges them chronologically, runs sequential pair
    analyses relative to T0, tracks earliest observations, and classifies change types.

    Returns:
        (text_answer: str, visual_evidence: List[Any], timeline: List[Dict], candidates: List[Dict])
    """
    tracer.append_log(f"Multi-Temporal Engine received {len(file_paths)} scenes for analysis")

    if len(file_paths) < 2:
        return (
            "Multi-temporal change analysis requires at least 2 sequential satellite images.",
            [],
            [],
            [],
            None,
        )

    # 1. Sort scenes chronologically
    dated_scenes = sort_scenes_chronologically(file_paths)
    baseline_dt, baseline_path = dated_scenes[0]
    latest_dt, latest_path = dated_scenes[-1]

    tracer.append_log(
        f"Multi-Temporal timeline established: T0={baseline_dt.strftime('%Y-%m-%d')} ({baseline_path.name}) "
        f"to Tk={latest_dt.strftime('%Y-%m-%d')} ({latest_path.name}) across {len(dated_scenes)} scenes"
    )

    # 2. Sequential pairwise analysis (T0 -> Ti)
    pair_results: List[Dict[str, Any]] = []
    evidence: List[Any] = []
    timeline: List[Dict[str, Any]] = []

    # Baseline entry (T0)
    t0_meta = _resolve_scene_photo_overlay(baseline_path)
    timeline.append({
        "step": 0,
        "step_index": 0,
        "date": baseline_dt.strftime("%Y-%m-%d"),
        "filename": baseline_path.name,
        "is_baseline": True,
        "change_area_ha": 0.0,
        "change_fraction": 0.0,
        "n_polygons": 0,
        "overlay_url": None,
        "image_url": t0_meta["image_url"],
        "imageUrl": t0_meta["image_url"],
        "nir_image_url": t0_meta["nir_image_url"],
        "nirImageUrl": t0_meta["nir_image_url"],
        "wgs84_bounds": t0_meta["wgs84_bounds"],
        "new_breakpoints": 0,
    })

    # Read baseline optical bands for spectral index calculations
    t0_bands = await asyncio.to_thread(_read_scene_bands, baseline_path)

    for i in range(1, len(dated_scenes)):
        ti_dt, ti_path = dated_scenes[i]
        date_str = ti_dt.strftime("%Y-%m-%d")
        tracer.append_log(f"step 3.{i}: analyzing temporal step T0 ({baseline_dt.strftime('%Y-%m-%d')}) -> T{i} ({date_str})")

        try:
            spatial_data = await asyncio.to_thread(
                _compute_spatial_analysis,
                query,
                baseline_path,
                ti_path,
                tracer,
            )

            if isinstance(spatial_data, tuple):
                logger.warning(f"Spatial analysis returned tuple for pair {i}: {spatial_data}")
                continue

            overlay_url = None
            mask_overlay_path = spatial_data.get("mask_overlay_path")
            mask_overlay = spatial_data.get("mask_overlay")
            if mask_overlay_path and mask_overlay_path.exists():
                relative = mask_overlay_path.relative_to(OUTPUT_DIR.parent).as_posix()
                overlay_url = f"/static/outputs/{relative}"
                evidence.append({
                    "type": "ImageOverlay",
                    "label": f"Change at {date_str} (T{i})",
                    "url": overlay_url,
                    "wgs84_bounds": mask_overlay.get("wgs84_bounds") if mask_overlay else None,
                    "opacity": 0.7,
                    "step_index": i,
                    "acquisition_date": date_str,
                })

            pair_results.append({
                "step": i,
                "date": date_str,
                "dt": ti_dt,
                "path": ti_path,
                "spatial_data": spatial_data,
                "overlay_url": overlay_url,
            })

            ti_meta = _resolve_scene_photo_overlay(ti_path)
            timeline.append({
                "step": i,
                "step_index": i,
                "date": date_str,
                "filename": ti_path.name,
                "is_baseline": False,
                "change_area_ha": round(spatial_data.get("changed_area_ha", 0.0), 3),
                "change_fraction": round(spatial_data.get("change_fraction", 0.0), 4),
                "n_polygons": len(spatial_data.get("polygons", [])),
                "overlay_url": overlay_url,
                "image_url": ti_meta["image_url"],
                "imageUrl": ti_meta["image_url"],
                "nir_image_url": ti_meta["nir_image_url"],
                "nirImageUrl": ti_meta["nir_image_url"],
                "wgs84_bounds": ti_meta["wgs84_bounds"],
                "new_breakpoints": 0,
            })

        except Exception as exc:
            tracer.append_log(f"Pair T0->T{i} ({date_str}) failed: {exc}")
            logger.error(f"Error computing pair {baseline_path.name} -> {ti_path.name}: {exc}", exc_info=True)

    if not pair_results:
        return (
            "Multi-temporal analysis could not process the provided scene stack.",
            [],
            timeline,
            [],
            None,
        )

    # 3. Candidate Tracking & Earliest Observation Detection
    # Final pair gives the full set of cumulative change polygons
    final_pair = pair_results[-1]
    final_polygons = final_pair["spatial_data"].get("polygons", [])
    tk_bands = await asyncio.to_thread(_read_scene_bands, final_pair["path"])

    candidates: List[Dict[str, Any]] = []
    type_counts: Dict[str, int] = {}
    breakpoint_counts: Dict[str, int] = {item["date"]: 0 for item in timeline}

    source_crs = None
    try:
        with rasterio.open(str(final_pair["path"])) as ds:
            source_crs = ds.crs
    except Exception:
        pass

    # Optical quality mask & SAR arbiter evaluation (ISRO PS 26167 §2.2.3)
    q_res = None
    try:
        from services import quality_mask_service, sar_arbiter_service
        q_res = quality_mask_service.detect_optical_quality_mask(final_pair["path"])
    except Exception as q_err:
        logger.warning(f"Could not compute optical quality mask: {q_err}")

    for idx, poly in enumerate(final_polygons, start=1):
        geom = poly.get("geometry", {})
        coords_raw = geom.get("coordinates")
        ring = _extract_outer_ring(coords_raw)
        if not ring:
            continue

        centroid, bbox = _get_polygon_centroid_and_bbox(ring)
        area_ha = float(poly.get("properties", {}).get("area_ha", 0.0))

        # Check earlier pairs to find earliest appearance
        earliest_step = final_pair["step"]
        earliest_date_str = final_pair["date"]
        trajectory = []

        for p_info in pair_results:
            step_idx = p_info["step"]
            d_str = p_info["date"]
            earlier_polys = p_info["spatial_data"].get("polygons", [])
            earlier_mask_path = p_info["spatial_data"].get("mask_path")

            detected = check_candidate_presence_in_earlier_pair(
                candidate_centroid=centroid,
                candidate_bbox=bbox,
                candidate_coords=ring,
                earlier_polygons=earlier_polys,
                earlier_mask_path=earlier_mask_path,
            )

            trajectory.append({
                "step": step_idx,
                "date": d_str,
                "detected": detected,
            })

            if detected and step_idx < earliest_step:
                earliest_step = step_idx
                earliest_date_str = d_str

        # Update breakpoint count for timeline
        if earliest_date_str in breakpoint_counts:
            breakpoint_counts[earliest_date_str] += 1

        poly_props = poly.get("properties", {})
        spec_dict = poly_props.get("spectral_metrics", {}) if isinstance(poly_props.get("spectral_metrics"), dict) else {}
        d_ndvi = float(spec_dict.get("delta_ndvi", poly_props.get("delta_ndvi", 0.0)))
        d_ndwi = float(spec_dict.get("delta_ndwi", poly_props.get("delta_ndwi", 0.0)))
        d_bright = float(spec_dict.get("delta_brightness", poly_props.get("delta_brightness", 0.0)))

        # Classify change type using spectral indices
        detected_change_orig = poly_props.get("detected_change", "General Change")
        spectral_metrics = spectral_indices.classify_change_type(
            delta_ndvi=d_ndvi,
            delta_ndwi=d_ndwi,
            delta_brightness=d_bright,
        )
        change_type = spectral_metrics[0]

        # If spectral metrics default to generic, fallback to detected change tag
        if change_type == "Spectral Surface Modification" and detected_change_orig:
            if "Bright" in detected_change_orig:
                change_type = "Surface Brightening / Earthwork"
            elif "Dark" in detected_change_orig:
                change_type = "Surface Darkening / Inundation"
            elif "Vegetation" in detected_change_orig:
                change_type = "Vegetation Modification"
            elif detected_change_orig != "General Change":
                change_type = detected_change_orig

        # Multi-Sensor Quality & SAR Arbitration (§2.2.3)
        cloud_occlusion = 0.0
        sar_arb_res = None
        if q_res is not None:
            try:
                poly_q = quality_mask_service.compute_polygon_quality_metrics(geom, q_res)
                cloud_occlusion = poly_q.get("cloud_overlap", 0.0)
                poly_props["is_cloud_confounded"] = poly_q.get("is_confounded", False)

                from services import sar_arbiter_service
                base_dt_str = timeline[0]["date"] if timeline else None
                sar_arb_res = sar_arbiter_service.arbitrate_candidate_change(
                    candidate={"geometry": geom, "centroid": centroid},
                    cloud_occlusion_fraction=cloud_occlusion,
                    target_date_str=final_pair["date"],
                    baseline_date_str=base_dt_str,
                )
                if sar_arb_res:
                    poly_props["sar_confidence_adjustment"] = sar_arb_res.get("confidence_adjustment", 0.0)
            except Exception as arb_err:
                logger.warning(f"SAR arbitration error for candidate {idx}: {arb_err}")

        # Compute dynamic multi-factored confidence score (0.10 to 0.98)
        confidence = spectral_indices.compute_multifactored_confidence(
            props=poly_props,
            trajectory=trajectory,
        )
        poly_props["confidence"] = confidence

        type_counts[change_type] = type_counts.get(change_type, 0) + 1

        # Resolve verified WGS84 coordinates for web map rendering and glide
        c_wgs84 = poly_props.get("centroid_wgs84")
        if c_wgs84 and len(c_wgs84) >= 2 and -180.0 <= c_wgs84[0] <= 180.0 and -90.0 <= c_wgs84[1] <= 90.0:
            final_centroid = [round(c_wgs84[0], 6), round(c_wgs84[1], 6)]
        else:
            final_centroid = [round(centroid[0], 6), round(centroid[1], 6)]

        geom_wgs84 = geom
        if (
            source_crs
            and source_crs.to_epsg() != 4326
            and ring
            and (abs(ring[0][0]) > 180.0 or abs(ring[0][1]) > 90.0)
        ):
            try:
                geom_wgs84 = change_service._transform_geojson_to_wgs84(geom, source_crs)
                wgs84_ring = _extract_outer_ring(geom_wgs84.get("coordinates"))
                if wgs84_ring and not (c_wgs84 and -180.0 <= c_wgs84[0] <= 180.0 and -90.0 <= c_wgs84[1] <= 90.0):
                    (wx, wy), _ = _get_polygon_centroid_and_bbox(wgs84_ring)
                    final_centroid = [round(wx, 6), round(wy, 6)]
            except Exception as tr_err:
                logger.warning(f"Error projecting polygon to WGS84: {tr_err}")

        candidate_record = {
            "id": idx,
            "change_type": change_type,
            "earliest_observation": earliest_date_str,
            "earliest_step": earliest_step,
            "latest_observation": final_pair["date"],
            "area_ha": round(area_ha, 3),
            "confidence": round(confidence, 3),
            "centroid": final_centroid,
            "trajectory": trajectory,
            "geometry": geom_wgs84,
            "cloud_occlusion": round(cloud_occlusion, 3),
            "false_alarm_flag": sar_arb_res.get("false_alarm_flag", "NONE") if sar_arb_res else "NONE",
            "sar_verification_status": sar_arb_res.get("status", "NOT_APPLICABLE") if sar_arb_res else "NOT_APPLICABLE",
            "sar_verification_badge": sar_arb_res.get("sar_verification_badge") if sar_arb_res else None,
            "sar_delta_db": sar_arb_res.get("sar_delta_db", 0.0) if sar_arb_res else 0.0,
            "suppression_reason": sar_arb_res.get("reason") if sar_arb_res else None,
        }
        candidates.append(candidate_record)

    # Sort candidates by area descending
    candidates.sort(key=lambda c: c["area_ha"], reverse=True)

    # Update timeline items with new breakpoint counts
    for item in timeline:
        d = item["date"]
        item["new_breakpoints"] = breakpoint_counts.get(d, 0)

    # Determine dominant change type
    dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "Surface Modification"
    total_changed_ha = float(final_pair["spatial_data"].get("changed_area_ha", 0.0))

    # 4. Language-Conditioned Spatio-Temporal Grounding & Re-ranking (CDQAG Engine)
    from services import cdqag_service

    if not target_concept or not phenomenon:
        try:
            from core.query_planner import parse_query_plan
            plan = parse_query_plan(query)
            target_concept = target_concept or plan.target_concept
            phenomenon = phenomenon or plan.phenomenon
        except Exception:
            pass

    tracer.append_log(
        f"CDQAG Engine: evaluating candidates for target concept '{target_concept or 'general'}'"
    )
    candidates, cdqag_summary = cdqag_service.batch_verify_change_candidates(
        candidates=candidates,
        scene_paths=[p for _, p in dated_scenes],
        target_concept=target_concept,
        phenomenon=phenomenon,
        query=query,
    )

    # 5. Build unified Multi-Temporal GeoJSON FeatureCollection in WGS84
    multi_temporal_features = []
    is_targeted = bool(cdqag_summary and cdqag_summary.get("is_targeted"))
    for c in candidates:
        match_tag = " [Query Matched]" if (is_targeted and c.get("query_matched")) else ""
        multi_temporal_features.append({
            "type": "Feature",
            "geometry": c["geometry"],
            "properties": {
                "id": c["id"],
                "area_ha": c["area_ha"],
                "change_type": c["change_type"],
                "earliest_observation": c["earliest_observation"],
                "latest_observation": c["latest_observation"],
                "confidence": c["confidence"],
                "center": c["centroid"],
                "centroid_wgs84": c["centroid"],
                "trajectory": c["trajectory"],
                "query_matched": c.get("query_matched", True),
                "semantic_relevance": c.get("semantic_relevance", 1.0),
                "grounding_reason": c.get("grounding_reason", ""),
                "grounding_bbox": c.get("grounding_bbox"),
                "label": f"{c['change_type']} (First seen: {c['earliest_observation']}){match_tag}",
            },
        })

    multi_temporal_geojson = {
        "type": "FeatureCollection",
        "features": multi_temporal_features,
    }
    # Prepend GeoJSON so it is primary in visual evidence
    evidence.insert(0, multi_temporal_geojson)

    # 6. Synthesize Markdown Answer
    text_answer = synthesize_multi_temporal_markdown(
        baseline_date=baseline_dt,
        latest_date=latest_dt,
        scenes_meta=[{"date": dt.strftime("%Y-%m-%d"), "path": str(p)} for dt, p in dated_scenes],
        timeline=timeline,
        candidates=candidates,
        dominant_type=dominant_type,
        total_changed_ha=total_changed_ha,
        cdqag_summary=cdqag_summary,
    )

    tracer.append_log(
        f"Multi-Temporal analysis complete: {len(candidates)} candidates tracked across {len(timeline)} observations"
    )

    # Construct complete SceneOverlay for MapViewport auto-mounting
    overall_bounds = t0_meta.get("wgs84_bounds") or [77.20, 28.55, 77.36, 28.71]
    center_lon = round((overall_bounds[0] + overall_bounds[2]) / 2.0, 6)
    center_lat = round((overall_bounds[1] + overall_bounds[3]) / 2.0, 6)
    sensor_name = "Sentinel-1" if ("s1" in baseline_path.name.lower() or "grd" in baseline_path.name.lower()) else "Sentinel-2"

    scene_items = []
    for idx, (dt, p) in enumerate(dated_scenes):
        smeta = _resolve_scene_photo_overlay(p)
        scene_items.append({
            "step_index": idx,
            "filename": p.name,
            "image_url": smeta["image_url"],
            "imageUrl": smeta["image_url"],
            "nir_image_url": smeta["nir_image_url"],
            "nirImageUrl": smeta["nir_image_url"],
            "date": dt.strftime("%Y-%m-%d"),
            "acquisition_date": dt.strftime("%Y-%m-%d"),
            "sensor": sensor_name,
        })

    scene_overlay_payload = {
        "datasetId": f"multitemporal_{int(datetime.now().timestamp())}",
        "name": f"{sensor_name} Multi-Temporal ({len(dated_scenes)} Epochs)",
        "sensor": sensor_name,
        "mode": "multi-temporal",
        "bounds": overall_bounds,
        "center": [center_lon, center_lat],
        "crs": t0_meta.get("crs", "EPSG:4326"),
        "resolution": "10.0m GSD",
        "t1ImageUrl": scene_items[0]["image_url"],
        "t2ImageUrl": scene_items[-1]["image_url"],
        "t1Filename": scene_items[0]["filename"],
        "t2Filename": scene_items[-1]["filename"],
        "scenes": scene_items,
    }

    return (
        text_answer,
        evidence,
        timeline,
        candidates,
        scene_overlay_payload,
    )
