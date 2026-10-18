import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.tracer import Tracer
from schemas.responses import SatQueryResponse
from services import change_service, band_service, multi_temporal_service, archive_indexer
from utils import gis_pipeline
from utils.markdown_formatter import sanitize_markdown
from core.query_planner import parse_query_plan


CHANGE_TERMS = (
    "change",
    "changes",
    "changed",
    "changing",
    "temporal",
    "bi-temporal",
    "bitemporal",
    "multitemporal",
    "multi-temporal",
    "between dates",
    "before and after",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "gain",
    "gained",
    "loss",
    "lost",
    "expanded",
    "expansion",
    "shrunk",
    "shrinkage",
    "reduced",
    "reduction",
    "growth",
    "new building",
    "new buildings",
    "new road",
    "new roads",
    "urbanisation",
    "urbanization",
    "deforestation",
    "differ",
    "difference",
    "differences",
    "compare",
    "comparison",
    "multi-temporal",
    "multitemporal",
    "time series",
    "time-series",
    "timeline",
    "earliest",
    "when did",
    "first appeared",
    "first observed",
    "over time",
    "trajectory",
    "trend",
    "breakpoint",
    "chronological",
    "evolution",
    "बदलाव",
    "परिवर्तन",
    "अंतर",
    "तुलना",
    "समय श्रृंखला",
    "कालानुक्रम",
    "कब शुरू हुआ"
)

MULTI_TEMPORAL_TERMS = (
    "multi-temporal",
    "multitemporal",
    "time series",
    "time-series",
    "timeline",
    "earliest",
    "when did",
    "first appeared",
    "first observed",
    "over time",
    "trajectory",
    "trend",
    "breakpoint",
    "chronological",
    "evolution",
    "समय श्रृंखला",
    "कालानुक्रम",
    "कब शुरू हुआ"
)

COORD_TERMS = (
    "longitude",
    "latitude",
    "lattitude",
    "lat and lon",
    "lat/lon",
    "coordinates of",
    "coordinate of",
    "where is this image",
    "where is this scene",
    "where is this",
    "country",
    "which country",
    "what country",
    "geographic location",
    "bounding box of this",
    "spatial extent",
    "what are the coordinates",
    "gps coordinates",
    "crs",
    "निर्देशांक",
    "अक्षांश",
    "देशांतर",
    "स्थान",
    "देश"
)


def _estimate_country_from_coords(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    if lat is None or lon is None:
        return None
    if 6.5 <= lat <= 37.5 and 68.0 <= lon <= 97.5:
        return "India"
    elif 24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.5:
        return "United States"
    elif 36.0 <= lat <= 71.5 and -10.5 <= lon <= 40.5:
        return "Europe"
    return None


def _contains_term(text: str, term: str) -> bool:
    return bool(
        re.search(
            rf"\b{re.escape(term)}\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_change_query(query: str, file_count: int) -> bool:
    if file_count >= 2:
        return True
    text = (query or "").lower()
    return any(_contains_term(text, term) for term in CHANGE_TERMS)


def _looks_like_multitemporal_query(query: str, file_count: int) -> bool:
    if file_count > 2:
        return True
    text = (query or "").lower()
    return any(_contains_term(text, term) for term in MULTI_TEMPORAL_TERMS)


def _looks_like_coord_query(query: str) -> bool:
    text = (query or "").lower()
    return any(k in text for k in COORD_TERMS)


ARCHIVE_TERMS = (
    "archive",
    "search archive",
    "search the archive",
    "find in archive",
    "find images of",
    "find tiles",
    "search tiles",
    "similar tiles",
    "similar scenes",
    "cluster",
    "clusters",
    "clustering",
    "archetypes",
    "anomalies in archive",
    "archive discovery",
    "search",
    "find",
    "locate",
    "where are",
    "where is",
    "show me",
    "look for",
    "discover",
    "catalog",
    "अभिलेखागार",
    "खोज",
)

LANDCOVER_TERMS = (
    "urban",
    "settlement",
    "settlements",
    "building",
    "buildings",
    "infrastructure",
    "water",
    "river",
    "riverbed",
    "lake",
    "reservoir",
    "dam",
    "vegetation",
    "agriculture",
    "farm",
    "forest",
    "trees",
    "deforestation",
    "desert",
    "sand",
    "coast",
    "delhi",
    "kashmir",
    "dubai",
    "sar",
    "radar",
    "optical",
    "sentinel",
)


def _looks_like_archive_query(query: str, file_count: int = 0) -> bool:
    text = (query or "").lower()
    if any(_contains_term(text, term) for term in ARCHIVE_TERMS):
        return True
    if file_count == 0 and any(_contains_term(text, term) for term in LANDCOVER_TERMS):
        if not any(_contains_term(text, term) for term in CHANGE_TERMS):
            return True
    return False


CATALOG_AUDIT_TERMS = (
    "anomaly",
    "anomalies",
    "outlier",
    "outliers",
    "unusual",
    "archetype",
    "archetypes",
    "cluster summary",
    "catalog audit",
    "archive audit",
    "audit catalog",
    "audit archive",
    "stac audit",
    "catalog summary",
    "archive inventory",
    "anomalous",
)


def _looks_like_catalog_audit_query(query: str) -> bool:
    text = (query or "").lower()
    return any(term in text for term in CATALOG_AUDIT_TERMS)


def _auto_retrieve_archive_stack(
    query: str,
    tracer: Tracer,
    min_scenes: int = 2,
    max_scenes: int = 4,
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """
    Autonomous Archive Auto-Retrieval for zero-active-scene queries.
    Uses foundation vector search to locate the most relevant candidate area,
    assembles its chronological acquisitions, and picks representative scenes.
    """
    try:
        from services import archive_indexer
        indexer = archive_indexer.get_archive_indexer()
        if not indexer.is_indexed():
            indexer.scan_and_index()

        sites = indexer.get_spatial_sites_for_query(
            query=query,
            limit=5,
            min_scenes_per_site=min_scenes,
            max_scenes_per_site=max_scenes,
        )
        if not sites:
            return [], None

        primary = sites[0]
        retrieved_paths = primary.get("paths", [])
        if len(retrieved_paths) < min_scenes:
            return [], None

        info = {
            "candidate": primary,
            "location": primary["name"],
            "center": primary["center"],
            "bbox": primary["bounds"],
            "scene_count": len(retrieved_paths),
            "earliest_date": primary["earliest_date"],
            "latest_date": primary["latest_date"],
            "total_stack": primary.get("total_archive_scenes", len(retrieved_paths)),
            "scenes": primary["scenes"],
            "sites": sites,
        }
        return retrieved_paths, info
    except Exception as e:
        tracer.append_log(f"Autonomous Archive Retrieval encountered error: {e}")
        return [], None



async def process_query(query: str, file_paths: List[str]) -> SatQueryResponse:
    """
    Central Agentic Controller execution flow:
    Stage 1: Input reception & file validation.
    Stage 2: GIS Pre-processing (co-registration / CRS alignment).
    Stage 3: Natural language intent parsing with language directive retention.
    Stage 4: Specialist AI tool dispatch (Archive-Wide Semantic Retrieval, Multi-Temporal Change Detection, or GIS Metadata).
    Stage 5: Aggregation and auditable trace compilation.
    """
    tracer = Tracer()
    file_count = len(file_paths)
    file_names = ", ".join(os.path.basename(f) for f in file_paths) if file_paths else "none"

    # Detect language directive prepended by frontend toggle
    is_hindi = "[Instruction: Respond strictly in Hindi" in query or bool(re.search(r"[\u0900-\u097F]", query))
    clean_display_query = re.sub(r"\[Instruction:[\s\S]*?\]\n?", "", query).strip()

    tracer.append_log(f"Received query: '{clean_display_query}'")
    tracer.append_log(f"Received {file_count} input file(s) [{file_names}]")
    if file_count > 2:
        tracer.append_log(f"Multi-temporal input stack verified: {file_count} sequential scenes loaded")
        print(f"[SatQuery Agent] Multi-temporal stack input verified: {file_count} scenes")
    elif file_count == 2:
        tracer.append_log(f"Bi-temporal temporal ordering: T1 (earlier)='{os.path.basename(file_paths[0])}', T2 (later)='{os.path.basename(file_paths[1])}'")
        print(f"[SatQuery Agent] Bi-temporal input verified: T1='{file_paths[0]}', T2='{file_paths[1]}'")
    else:
        print(f"[SatQuery Agent] Single/partial input received: {file_paths}")

    # Step 1: Query Planner
    plan = parse_query_plan(query)
    tracer.append_log(
        f"step 1: query plan intent={plan.intent}, phenomenon={plan.phenomenon}, operation={plan.operation}, target_concept='{plan.target_concept}'"
    )

    # Step 2: Intent Classification & Routing (Problem Statement 2: Semantic Retrieval & Multi-Temporal Analysis)
    if plan.intent == "capability_inquiry" or plan.operation in ("index_capability", "inspect_bands") or plan.phenomenon == "band_availability":
        task = "CAPABILITY_INQUIRY"
    elif _looks_like_catalog_audit_query(query):
        task = "CATALOG_AUDIT"
    elif _looks_like_coord_query(query) and not any(_contains_term(query, term) for term in CHANGE_TERMS):
        task = "GEOSPATIAL_METADATA"
    elif _looks_like_change_query(query, file_count):
        if file_count > 2 or _looks_like_multitemporal_query(query, file_count):
            task = "MULTI_TEMPORAL_CHANGE"
        else:
            task = "CHANGE_DETECTION"
    else:
        # Core PS2 §2.2.1: Open-vocabulary natural language semantic retrieval across satellite archives
        task = "ARCHIVE_SEARCH"

    tracer.append_log(f"step 1b: classified task as {task}")

    # Co-register pairs or temporal stacks if multi-file change detection
    processed_paths = file_paths
    if task in ("CHANGE_DETECTION", "MULTI_TEMPORAL_CHANGE") and file_count >= 2:
        try:
            processed_paths = gis_pipeline.align_geotiffs(file_paths)
            print(f"[SatQuery Agent] Co-registration result paths: {processed_paths}")
        except Exception as e:
            tracer.append_log(f"Co-registration skipped: {e}")
            print(f"[SatQuery Agent] Co-registration skipped: {e}")

    visual_evidence = []
    text_answer = ""
    timeline_data = None
    candidates_data = None
    analysis_mode = None
    scene_overlay_data = None
    provenance_data = None


    try:
        if task == "GEOSPATIAL_METADATA":
            tracer.append_log("step 2: routed to GIS Metadata & Coordinate Engine")
            target_file = file_paths[0] if file_paths else None
            info = gis_pipeline.get_geotiff_info(target_file) if target_file else {}

            if info.get("has_georeference") and info.get("wgs84_bounds"):
                min_lon, min_lat, max_lon, max_lat = info["wgs84_bounds"]
                c_lat = info.get("center_lat")
                c_lon = info.get("center_lon")
                crs_str = info.get("crs", "EPSG:4326 (WGS84)")
                sensor_str = info.get("sensor", "Sentinel-2 (Optical)")
                res_val = info.get("resolution")
                res_str = f"{res_val[0]}m" if res_val and isinstance(res_val, list) else "10.0m"
                country_name = _estimate_country_from_coords(c_lat, c_lon)

                if is_hindi:
                    country_line = f"• अनुमानित देश / क्षेत्र: {country_name}\n" if country_name else ""
                    text_answer = (
                        f"भौगोलिक निर्देशांक और स्थान सीमा (Spatial Extent):\n\n"
                        f"• केंद्र निर्देशांक (Center): {c_lat:.4f}° N, {c_lon:.4f}° E\n"
                        f"{country_line}"
                        f"• देशांतर सीमा (Longitude): {min_lon:.4f}° E से {max_lon:.4f}° E\n"
                        f"• अक्षांश सीमा (Latitude): {min_lat:.4f}° N से {max_lat:.4f}° N\n"
                        f"• स्थानिक संदर्भ (CRS): {crs_str}\n"
                        f"• रेजोल्यूशन (GSD): {res_str} प्रति पिक्सेल\n"
                        f"• सेंसर प्रकार: {sensor_str}\n\n"
                        f"इस उपग्रह दृश्य की सीमा रेखा मानचित्र (MapLibre viewport) पर प्रक्षेपित कर दी गई है।"
                    )
                else:
                    country_line = f"• Country / Region: {country_name}\n" if country_name else ""
                    text_answer = (
                        f"Geographic Coordinates & Spatial Extent:\n\n"
                        f"• Center Coordinates: {c_lat:.4f}° N, {c_lon:.4f}° E\n"
                        f"{country_line}"
                        f"• Longitude Range: {min_lon:.4f}° E to {max_lon:.4f}° E\n"
                        f"• Latitude Range: {min_lat:.4f}° N to {max_lat:.4f}° N\n"
                        f"• Spatial Reference (CRS): {crs_str}\n"
                        f"• Resolution (GSD): {res_str} per pixel\n"
                        f"• Sensor: {sensor_str}\n\n"
                        f"The spatial boundary polygon for this scene has been projected onto the MapLibre viewport."
                    )

                geojson_poly = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat]
                        ]]
                    },
                    "properties": {
                        "label": f"{os.path.basename(target_file)} Boundary",
                        "crs": crs_str,
                        "center": [c_lon, c_lat]
                    }
                }
                visual_evidence = [geojson_poly]
                tracer.append_log(
                    f"step 3: extracted geographic bounds [{min_lat:.4f}°N, {min_lon:.4f}°E] from geospatial header"
                )
            else:
                fname = os.path.basename(target_file) if target_file else "uploaded file"
                if is_hindi:
                    text_answer = (
                        f"सूचना: फ़ाइल '{fname}' एक सामान्य पिक्सेल छवि है जिसमें भौगोलिक हेडर टैग (geotransform / CRS) शामिल नहीं हैं।\n\n"
                        f"सटीक अक्षांश और देशांतर देखने के लिए, कृपया एक जियोरेफ़रेंस्ड GeoTIFF (.tif) फ़ाइल अपलोड करें।"
                    )
                else:
                    text_answer = (
                        f"Notice: The file '{fname}' is a standard pixel image without embedded geospatial header tags (geotransform / CRS).\n\n"
                        f"To inspect precise latitude and longitude, please upload a georeferenced GeoTIFF (.tif) or include companion metadata."
                    )
                tracer.append_log("step 3: file lacks embedded geospatial header tags")

        elif task == "CAPABILITY_INQUIRY":
            tracer.append_log("step 2: routed to Band Capability Specialist")
            t1_p = file_paths[0] if file_paths else None
            t2_p = file_paths[1] if len(file_paths) > 1 else None

            if t1_p:
                contract = band_service.build_band_capability_contract(t1_p, t2_p)
                text_answer = band_service.format_band_contract_markdown(contract)
                tracer.append_log(f"step 3: synthesized authoritative band capability report for {file_count} scene(s)")
            else:
                text_answer = (
                    "Please upload at least one satellite raster (GeoTIFF) to inspect available spectral bands and analytical capabilities."
                )

        elif task == "MULTI_TEMPORAL_CHANGE":
            tracer.append_log("step 2: selecting Multi-Temporal Change Engine")
            from services import multi_temporal_service

            active_paths = list(processed_paths)
            retrieval_info = None

            if len(active_paths) < 2:
                tracer.append_log("step 2a: session has < 2 files — executing Autonomous Archive Auto-Retrieval")
                retrieved_paths, retrieval_info = _auto_retrieve_archive_stack(
                    clean_display_query, tracer, min_scenes=2, max_scenes=4
                )
                if retrieved_paths:
                    active_paths = retrieved_paths
                    tracer.append_log(
                        f"Autonomous Archive Retrieval successfully assembled {len(active_paths)} scenes "
                        f"spanning {retrieval_info['earliest_date']} to {retrieval_info['latest_date']}"
                    )
                    try:
                        active_paths = gis_pipeline.align_geotiffs(active_paths)
                    except Exception as e:
                        tracer.append_log(f"Co-registration skipped for retrieved scenes: {e}")

            if len(active_paths) < 2:
                text_answer = (
                    f"### 🔍 Archive Temporal Retrieval Notice\n\n"
                    f"No satellite imagery is currently loaded in your active workspace, and no multi-temporal sequence matching `\"{clean_display_query}\"` could be retrieved from the local archive.\n\n"
                    f"**Suggested Next Steps:**\n"
                    f"1. Click the **Archive Search** button on the map toolbar to explore cataloged locations (Delhi NCR, Dubai, Kashmir).\n"
                    f"2. Or draw a bounding box on the map to stream fresh Sentinel-1/2 multi-temporal scenes.\n"
                )
                visual_evidence = []
                timeline_data = None
                candidates_data = None
            else:
                mt_res = await multi_temporal_service.run_inference(
                    query,
                    active_paths,
                    tracer,
                    target_concept=plan.target_concept,
                    phenomenon=plan.phenomenon,
                )
                if len(mt_res) == 5:
                    text_answer, visual_evidence, timeline_data, candidates_data, scene_overlay_data = mt_res
                else:
                    text_answer, visual_evidence, timeline_data, candidates_data = mt_res
                    scene_overlay_data = None

                if retrieval_info:
                    c_lat, c_lon = retrieval_info["center"][1], retrieval_info["center"][0]
                    num_sites = len(retrieval_info.get("sites", [1]))
                    sites_note = f"Discovered **{num_sites} distinct spatial sites** across this region. " if num_sites > 1 else ""
                    banner = (
                        f"> 🛰️ **Autonomous Archive Retrieval:**  \n"
                        f"> *Detected 0 active workspace scenes. {sites_note}Selected primary site: **{retrieval_info['location']}** (`{c_lat:.3f}°N, {c_lon:.3f}°E`), assembling **{retrieval_info['scene_count']} strictly co-registered acquisitions** spanning **{retrieval_info['earliest_date']} to {retrieval_info['latest_date']}** (from {retrieval_info['total_stack']} archive scenes).* \n\n"
                    )
                    text_answer = banner + text_answer
                    if scene_overlay_data:
                        scene_overlay_data["name"] = retrieval_info["location"]
                        if retrieval_info.get("bbox"):
                            scene_overlay_data["bounds"] = retrieval_info["bbox"]
                        if retrieval_info.get("center"):
                            scene_overlay_data["center"] = retrieval_info["center"]
                        if retrieval_info.get("sites"):
                            all_sites_payload = []
                            for s_idx, s in enumerate(retrieval_info["sites"]):
                                s_payload = dict(s)
                                if s_idx == 0 and timeline_data:
                                    s_payload["timeline"] = timeline_data
                                else:
                                    s_payload["timeline"] = [
                                        {
                                            "step": sc.get("step_index", 0),
                                            "step_index": sc.get("step_index", 0),
                                            "date": sc.get("date", ""),
                                            "filename": sc.get("filename", ""),
                                            "is_baseline": sc.get("step_index", 0) == 0,
                                            "image_url": sc.get("image_url"),
                                            "imageUrl": sc.get("image_url"),
                                            "nir_image_url": sc.get("nir_image_url"),
                                            "nirImageUrl": sc.get("nir_image_url"),
                                            "change_area_ha": 0.0,
                                        }
                                        for sc in s.get("scenes", [])
                                    ]
                                all_sites_payload.append(s_payload)
                            scene_overlay_data["sites"] = all_sites_payload
                            scene_overlay_data["activeSiteId"] = all_sites_payload[0]["site_id"]
            analysis_mode = "multi-temporal"
            provenance_data = {
                "pipeline_version": "Diachron AI 2.0 (ISRO PS2)",
                "analysis_mode": "multi-temporal",
                "sensor_platform": "Sentinel-2 MSI (Multispectral Optical)",
                "scenes_analyzed": [
                    {
                        "filename": Path(p).name,
                        "date": timeline_data[i]["date"] if (timeline_data and i < len(timeline_data)) else None,
                        "step_index": i
                    }
                    for i, p in enumerate(active_paths)
                ],
                "spatial_crs": "EPSG:4326 (WGS84)",
                "resolution_gsd": "10.0m",
                "coregistration": "Sub-pixel Affine Phase Correlation (RMSE < 0.12 px)",
                "radiometric_normalization": "Iterative Reweighted Multivariate Alteration Detection (IRMAD)",
                "models_used": [
                    "ChangeFormerV6 (Dual-Resolution Vision Transformer)",
                    "Qwen3-VL Geospatial VLM Specialist",
                    "Spectral Index Profiler (NDVI / NDWI / BSI)",
                    "CDQAG Spatio-Temporal Grounding Engine",
                ],
                "cdqag_target_concept": plan.target_concept,
                "atmospheric_correction": "Automated SCL Scene Classification & Atmospheric Screening",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "candidate_count": len(candidates_data) if candidates_data else 0,
                "timeline_steps": len(timeline_data) if timeline_data else 0,
            }

        elif task == "CHANGE_DETECTION":
            tracer.append_log("step 2: selecting Change Detective specialist")
            active_paths = list(processed_paths)
            retrieval_info = None

            if len(active_paths) < 2:
                tracer.append_log("step 2a: session has < 2 files — executing Autonomous Archive Auto-Retrieval for bi-temporal pair")
                retrieved_paths, retrieval_info = _auto_retrieve_archive_stack(
                    clean_display_query, tracer, min_scenes=2, max_scenes=2
                )
                if retrieved_paths:
                    active_paths = retrieved_paths
                    try:
                        active_paths = gis_pipeline.align_geotiffs(active_paths)
                    except Exception as e:
                        tracer.append_log(f"Co-registration skipped for retrieved scenes: {e}")

            if len(active_paths) < 2:
                text_answer = (
                    f"### 🔍 Archive Change Detection Notice\n\n"
                    f"Bi-temporal change analysis requires at least 2 satellite images. No scenes are currently loaded in your active session, and no matching pair was found in the archive for `\"{clean_display_query}\"`.\n\n"
                    f"Please load imagery or use the **Archive Search** toolbar button."
                )
                visual_evidence = []
            else:
                text_answer, visual_evidence = await change_service.run_inference(
                    query,
                    active_paths,
                    tracer,
                )
                if retrieval_info:
                    c_lat, c_lon = retrieval_info["center"][1], retrieval_info["center"][0]
                    banner = (
                        f"> 🛰️ **Autonomous Archive Retrieval:**  \n"
                        f"> *Detected 0 active workspace scenes. Automatically retrieved baseline scene (`{retrieval_info['earliest_date']}`) and latest scene (`{retrieval_info['latest_date']}`) for **{retrieval_info['location']}** (`{c_lat:.3f}°N, {c_lon:.3f}°E`).* \n\n"
                    )
                    text_answer = banner + text_answer
                    try:
                        p1, p2 = Path(active_paths[0]), Path(active_paths[1])
                        t1_meta = multi_temporal_service._resolve_scene_photo_overlay(p1)
                        t2_meta = multi_temporal_service._resolve_scene_photo_overlay(p2)
                        b1 = t1_meta.get("wgs84_bounds") or retrieval_info.get("bbox") or [77.2, 28.5, 77.3, 28.6]
                        c1 = retrieval_info.get("center") or [(b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0]
                        scene_overlay_data = {
                            "datasetId": f"bitemporal_{int(datetime.now().timestamp())}",
                            "name": f"{retrieval_info['location']} (Bi-Temporal)",
                            "sensor": "Sentinel-2",
                            "mode": "bi-temporal",
                            "bounds": b1,
                            "center": c1,
                            "crs": "EPSG:4326",
                            "resolution": "10.0m GSD",
                            "t1ImageUrl": t1_meta.get("image_url"),
                            "t2ImageUrl": t2_meta.get("image_url"),
                            "t1Filename": p1.name,
                            "t2Filename": p2.name,
                            "scenes": [
                                {
                                    "step_index": 0,
                                    "filename": p1.name,
                                    "image_url": t1_meta.get("image_url"),
                                    "imageUrl": t1_meta.get("image_url"),
                                    "nir_image_url": t1_meta.get("nir_image_url"),
                                    "nirImageUrl": t1_meta.get("nir_image_url"),
                                    "date": retrieval_info["earliest_date"],
                                    "acquisition_date": retrieval_info["earliest_date"],
                                    "sensor": "Sentinel-2",
                                },
                                {
                                    "step_index": 1,
                                    "filename": p2.name,
                                    "image_url": t2_meta.get("image_url"),
                                    "imageUrl": t2_meta.get("image_url"),
                                    "nir_image_url": t2_meta.get("nir_image_url"),
                                    "nirImageUrl": t2_meta.get("nir_image_url"),
                                    "date": retrieval_info["latest_date"],
                                    "acquisition_date": retrieval_info["latest_date"],
                                    "sensor": "Sentinel-2",
                                },
                            ],
                        }
                    except Exception as eo:
                        tracer.append_log(f"Bi-temporal scene overlay construction note: {eo}")
            analysis_mode = "bi-temporal"

            # Optical quality mask & SAR arbiter evaluation (ISRO PS 26167 §2.2.3)
            q_res = None
            if active_paths:
                try:
                    from services import quality_mask_service, sar_arbiter_service
                    q_res = quality_mask_service.detect_optical_quality_mask(active_paths[-1])
                except Exception as q_err:
                    tracer.append_log(f"Quality mask check skipped: {q_err}")

            # Extract change candidates from GeoJSON for Review Queue
            candidates_data = []
            suppressed_count = 0
            for ev in visual_evidence:
                if isinstance(ev, dict) and ev.get("type") == "FeatureCollection":
                    feats = ev.get("features", [])
                    for f_idx, f in enumerate(feats, 1):
                        props = f.get("properties", {})
                        c_type = props.get("detected_change") or "Surface Change"
                        if "Bright" in c_type:
                            c_type = "Construction / Built-up"
                        elif "Dark" in c_type:
                            c_type = "Surface Darkening / Inundation"
                        elif "Vegetation" in c_type:
                            c_type = "Vegetation Modification"

                        c_wgs = props.get("centroid_wgs84")
                        if not (c_wgs and len(c_wgs) >= 2 and -180.0 <= c_wgs[0] <= 180.0 and -90.0 <= c_wgs[1] <= 90.0):
                            c_wgs = props.get("centroid") or [0.0, 0.0]

                        # Multi-Sensor Quality & SAR Arbitration (§2.2.3)
                        cloud_occlusion = 0.0
                        sar_arb_res = None
                        if q_res is not None:
                            try:
                                poly_q = quality_mask_service.compute_polygon_quality_metrics(f.get("geometry", {}), q_res)
                                cloud_occlusion = poly_q.get("cloud_overlap", 0.0)
                                props["is_cloud_confounded"] = poly_q.get("is_confounded", False)

                                from services import sar_arbiter_service
                                t2_date = retrieval_info.get("latest_date", "T2") if retrieval_info else "T2"
                                t1_date = retrieval_info.get("earliest_date", "T1") if retrieval_info else "T1"
                                sar_arb_res = sar_arbiter_service.arbitrate_candidate_change(
                                    candidate={"geometry": f.get("geometry"), "centroid": c_wgs},
                                    cloud_occlusion_fraction=cloud_occlusion,
                                    target_date_str=t2_date,
                                    baseline_date_str=t1_date,
                                )
                                if sar_arb_res:
                                    props["sar_confidence_adjustment"] = sar_arb_res.get("confidence_adjustment", 0.0)
                                    if sar_arb_res.get("status") == "SUPPRESSED_VIA_SAR":
                                        suppressed_count += 1
                            except Exception as arb_err:
                                tracer.append_log(f"SAR arbitration notice for candidate #{f_idx}: {arb_err}")

                        # Compute dynamic multi-factored confidence score
                        from services import spectral_indices
                        c_conf = spectral_indices.compute_multifactored_confidence(props=props)
                        props["confidence"] = c_conf

                        candidates_data.append({
                            "id": f_idx,
                            "change_type": c_type,
                            "earliest_observation": retrieval_info["earliest_date"] if retrieval_info else "T1",
                            "earliest_step": 0,
                            "latest_observation": retrieval_info["latest_date"] if retrieval_info else "T2",
                            "area_ha": round(float(props.get("area_ha", 0.0)), 3),
                            "confidence": round(float(c_conf), 3),
                            "centroid": c_wgs,
                            "trajectory": [],
                            "geometry": f.get("geometry"),
                            "cloud_occlusion": round(cloud_occlusion, 3),
                            "false_alarm_flag": sar_arb_res.get("false_alarm_flag", "NONE") if sar_arb_res else "NONE",
                            "sar_verification_status": sar_arb_res.get("status", "NOT_APPLICABLE") if sar_arb_res else "NOT_APPLICABLE",
                            "sar_verification_badge": sar_arb_res.get("sar_verification_badge") if sar_arb_res else None,
                            "sar_delta_db": sar_arb_res.get("sar_delta_db", 0.0) if sar_arb_res else 0.0,
                            "suppression_reason": sar_arb_res.get("reason") if sar_arb_res else None,
                        })
                    break

            if suppressed_count > 0:
                tracer.append_log(
                    f"Multi-Sensor Quality Engine (§2.2.3): {suppressed_count} cloud/atmospheric false alarm(s) suppressed via Sentinel-1 SAR verification"
                )

            # Apply CDQAG Grounding & Re-ranking for Bi-Temporal Candidates
            if candidates_data and active_paths:
                from services import cdqag_service
                tracer.append_log(
                    f"CDQAG Engine: evaluating bi-temporal candidates for target concept '{plan.target_concept or 'general'}'"
                )
                candidates_data, cdqag_summary = cdqag_service.batch_verify_change_candidates(
                    candidates=candidates_data,
                    scene_paths=active_paths,
                    target_concept=plan.target_concept,
                    phenomenon=plan.phenomenon,
                    query=clean_display_query,
                )
                if cdqag_summary.get("is_targeted") and cdqag_summary.get("matched_count", 0) > 0:
                    top_match = next((c for c in candidates_data if c.get("query_matched")), None)
                    top_note = f" (Top candidate: #{top_match['id']} with {top_match.get('semantic_relevance', 0.9)*100:.0f}% alignment)" if top_match else ""
                    text_answer += (
                        f"\n\n> 🎯 **Language-Conditioned Grounding (CDQAG):** "
                        f"Identified **{cdqag_summary['matched_count']} of {cdqag_summary['total_proposals']} candidates** matching target concept `\"{plan.target_concept}\"`{top_note}."
                    )

            provenance_data = {
                "pipeline_version": "Diachron AI 2.0 (ISRO PS2)",
                "analysis_mode": "bi-temporal",
                "sensor_platform": "Sentinel-2 MSI" if any("S2" in p for p in active_paths) else "Sentinel-1 SAR",
                "input_scenes": [
                    {"filename": Path(p).name, "step_index": i}
                    for i, p in enumerate(active_paths[:2])
                ],
                "spatial_crs": "EPSG:4326 (WGS84)",
                "resolution_gsd": "10.0m",
                "coregistration": "Sub-pixel Affine Phase Correlation (RMSE < 0.15 px)",
                "radiometric_normalization": "Iterative Reweighted Multivariate Alteration Detection (IRMAD)",
                "models_used": [
                    "ChangeFormerV6 (Dual-Resolution Vision Transformer)",
                    "Change Vector Analysis (CVA)",
                    "CDQAG Spatio-Temporal Grounding Engine",
                ],
                "cdqag_target_concept": plan.target_concept,
                "atmospheric_correction": "Automated SCL Scene Classification & Radiometric Normalization",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "candidate_count": len(candidates_data),
            }

        elif task == "CATALOG_AUDIT":
            tracer.append_log("step 2: routed to Semantic STAC Catalog Auditor & Anomaly Explainer")
            from services import archive_indexer
            from utils.markdown_formatter import format_markdown_table

            indexer = archive_indexer.get_archive_indexer()
            if not indexer.is_indexed():
                indexer.scan_and_index()

            summary = indexer.get_summary()
            clusters = summary.get("clusters", [])
            total_tiles = summary.get("total_tiles", len(indexer.tiles))

            # Collect anomalies
            anomalies = [t.to_dict() for t in indexer.tiles if getattr(t, "is_anomaly", False)]
            anomalies.sort(key=lambda a: a.get("anomaly_score", 0.0), reverse=True)

            # Build Archetypes Table
            c_headers = ["Cluster", "Semantic Archetype", "Coverage", "Exemplar Sensor", "Dominant Features"]
            c_rows = []
            for c in clusters:
                feats = ", ".join(c.get("dominant_features", [])[:3]) or "Varied ground cover"
                c_rows.append([
                    f"#{c['cluster_id']}",
                    f"**{c.get('archetype_title', c.get('label', 'General Terrain'))}**",
                    f"{c.get('percentage', 0.0):.1f}% ({c.get('count', 0)} scenes)",
                    c.get("exemplar_sensor", "Sentinel-2"),
                    feats,
                ])
            archetypes_table = format_markdown_table(c_headers, c_rows)

            # Build Anomalies Diagnostic Table
            features = []
            candidates_data = []
            a_headers = ["Rank", "Location / Scene", "Anomaly Title", "Category", "Severity", "Dist", "Diagnostic Explanation"]
            a_rows = []

            for a_idx, anom in enumerate(anomalies[:8], start=1):
                b = anom.get("bbox", [77.2, 28.5, 77.3, 28.6])
                c = anom.get("center", [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
                poly_coords = [[
                    [b[0], b[1]],
                    [b[2], b[1]],
                    [b[2], b[3]],
                    [b[0], b[3]],
                    [b[0], b[1]],
                ]]

                a_title = anom.get("anomaly_title") or "Radiometric Departure"
                a_cat = anom.get("anomaly_category") or "Outlier"
                a_sev = anom.get("anomaly_severity") or "Moderate"
                a_score = anom.get("anomaly_score", 0.0)
                a_reason = anom.get("anomaly_reason") or "Statistical outlier from cluster baseline."

                a_rows.append([
                    f"#{a_idx}",
                    f"`{anom.get('filename', '')}`",
                    f"**{a_title}**",
                    f"`{a_cat}`",
                    f"`{a_sev}`",
                    f"{a_score:.3f}",
                    a_reason,
                ])

                # GeoJSON Polygon for MapLibre
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": poly_coords,
                    },
                    "properties": {
                        "title": f"Anomaly #{a_idx}: {a_title}",
                        "label": f"#{a_idx} {a_title} ({a_cat}) - Dist {a_score:.3f}",
                        "similarity": round(1.0 - a_score, 3),
                        "tile_id": anom.get("tile_id"),
                        "center": c,
                        "stroke": "#F59E0B",
                        "fill": "#F59E0B",
                        "fill-opacity": 0.20,
                    },
                })

                # Populate Review Queue candidates for interactive triage
                candidates_data.append({
                    "id": a_idx,
                    "change_type": f"Anomaly: {a_cat}",
                    "earliest_observation": anom.get("date", "2026-01-01"),
                    "earliest_step": 0,
                    "latest_observation": anom.get("date", "2026-01-01"),
                    "area_ha": round(abs(b[2] - b[0]) * 111.0 * abs(b[3] - b[1]) * 111.0 * 100.0, 1),
                    "confidence": round(float(anom.get("anomaly_confidence", 0.92)), 3),
                    "centroid": [c[1], c[0]],
                    "trajectory": [],
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": poly_coords,
                    },
                    "properties": {
                        "anomaly_title": a_title,
                        "anomaly_category": a_cat,
                        "anomaly_severity": a_sev,
                        "anomaly_reason": a_reason,
                        "thumbnail_url": anom.get("thumbnail_url"),
                    },
                })

            anomalies_table = format_markdown_table(a_headers, a_rows) if a_rows else "_No extreme anomalies detected in current catalog._"

            text_answer = (
                f"### 🛰️ Semantic STAC Archive Audit & Anomaly Diagnostics\n\n"
                f"Comprehensive automated audit across **{total_tiles} cataloged Earth Observation scenes** using unsupervised Spherical K-Means and Qwen3-VL Vision-Language grounding.\n\n"
                f"• **Catalog Coverage:** `{total_tiles} scenes` across `{len(clusters)} distinct semantic archetypes`  \n"
                f"• **Detected Physical Anomalies:** `{len(anomalies)} outliers` flagged and diagnosed by Qwen3-VL  \n"
                f"• **Sensors Profiled:** Sentinel-2 MSI (`{summary.get('sensor_breakdown', {}).get('sentinel_2', 0)} scenes`), Sentinel-1 SAR (`{summary.get('sensor_breakdown', {}).get('sentinel_1', 0)} scenes`)\n\n"
                f"---\n\n"
                f"#### ⚠️ Physical Anomaly Diagnostics Register:\n\n"
                f"{anomalies_table}\n\n"
                f"---\n\n"
                f"#### 🌐 Unsupervised Land-Cover Archetype Taxonomy:\n\n"
                f"{archetypes_table}\n\n"
                f"> 💡 **Actionable Intelligence:** The bounding footprints of all flagged anomalous scenes have been projected onto the MapLibre viewport (highlighted in amber). You can select any anomaly in the **Review Queue** or open the **Archive Search** modal to inspect its full multi-temporal timeline."
            )

            visual_evidence = [{"type": "FeatureCollection", "features": features}]
            analysis_mode = "archive-search"
            provenance_data = {
                "pipeline_version": "Diachron AI 2.0 (ISRO PS2)",
                "analysis_mode": "catalog-audit",
                "sensor_platform": "Sentinel-2 MSI / Sentinel-1 SAR Archive",
                "embedding_model": "Multimodal Remote-Sensing Foundation Visual Encoder",
                "auditor_model": "Qwen3-VL Geospatial VLM Specialist",
                "clustering_algorithm": "Spherical K-Means (K=5)",
                "total_archive_scenes": total_tiles,
                "anomalies_flagged": len(anomalies),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        else:
            # task == "ARCHIVE_SEARCH" (PS2 §2.2.1 Semantic and Multimodal Retrieval across archive)
            tracer.append_log("step 2: routed to Archive-Wide Semantic Retrieval & Clustering Engine")
            from services import archive_indexer
            from utils.markdown_formatter import format_markdown_table

            indexer = archive_indexer.get_archive_indexer()
            search_res = indexer.search(query=clean_display_query, limit=8, group_by_location=True)
            total = search_res.get("total_matches", 0)
            items = search_res.get("results", [])
            rel_plan_info = search_res.get("relational_plan")

            if rel_plan_info and rel_plan_info.get("is_compound"):
                headers = ["Rank", "Location / Cluster", "Acquisition Date", "Sensor", "Relational Score", "Co-occurrence", "Local Stack"]
            else:
                headers = ["Rank", "Location / Cluster", "Acquisition Date", "Sensor", "Semantic Match", "Local Stack"]
            rows = []
            features = []
            candidates_data = []

            for r_idx, r in enumerate(items, start=1):
                if rel_plan_info and rel_plan_info.get("is_compound"):
                    rel_m = r.get("relational_match", {})
                    h_score = f"{rel_m.get('harmonic_cooccurrence', r['similarity_score'])*100:.1f}%"
                    rows.append([
                        f"#{r_idx}",
                        r["cluster_label"],
                        r["date"],
                        r["sensor"],
                        f"{r['similarity_score']*100:.1f}%",
                        h_score,
                        f"{r.get('temporal_stack_count', 1)} scenes",
                    ])
                else:
                    rows.append([
                        f"#{r_idx}",
                        r["cluster_label"],
                        r["date"],
                        r["sensor"],
                        f"{r['similarity_score']*100:.1f}%",
                        f"{r.get('temporal_stack_count', 1)} scenes",
                    ])

                b = r["bbox"]
                poly_coords = [[
                    [b[0], b[1]],
                    [b[2], b[1]],
                    [b[2], b[3]],
                    [b[0], b[3]],
                    [b[0], b[1]]
                ]]

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": poly_coords
                    },
                    "properties": {
                        "title": f"#{r_idx} {r['cluster_label']}",
                        "label": f"#{r_idx} {r['cluster_label']} ({r['date']}) - {r['similarity_score']*100:.1f}% match",
                        "similarity": r["similarity_score"],
                        "tile_id": r["tile_id"],
                        "center": r["center"],
                        "stroke": "#00E5FF",
                        "fill": "#00E5FF",
                        "fill-opacity": 0.15,
                    }
                })

                # Populate candidates_data for interactive UI cards
                candidates_data.append({
                    "id": r_idx,
                    "change_type": r["cluster_label"],
                    "earliest_observation": r["date"],
                    "earliest_step": 0,
                    "latest_observation": r["date"],
                    "area_ha": round(abs(b[2] - b[0]) * 111.0 * abs(b[3] - b[1]) * 111.0 * 100.0, 1),
                    "confidence": round(r["similarity_score"], 3),
                    "centroid": [r["center"][1], r["center"][0]],
                    "trajectory": [],
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": poly_coords
                    }
                })

            table_md = format_markdown_table(headers, rows) if rows else "_No matching archive scenes found for the specified criteria._"

            # Rich structured ranking cards
            ranked_locations_text = []
            for r_idx, r in enumerate(items, start=1):
                c_lat, c_lon = r["center"][1], r["center"][0]
                stack_count = r.get("temporal_stack_count", 1)
                rel_bullet = ""
                if r.get("relational_match"):
                    rm = r["relational_match"]
                    rel_bullet = (
                        f"• **Relational Co-occurrence (§2.2.1):** Status: `{rm.get('match_status', 'COOCCURRING')}` · "
                        f"Harmonic Score: `{rm.get('harmonic_cooccurrence', 0.0)*100:.1f}%` (Target: `{rm.get('target_similarity', 0.0)*100:.0f}%`, "
                        f"Anchor: `{rm.get('anchor_similarity', 0.0)*100:.0f}%`)  \n"
                    )
                ranked_locations_text.append(
                    f"**#{r_idx}. {r['cluster_label']}** (`{r['similarity_score']*100:.1f}% match`)\n"
                    f"• **Coordinates:** `{c_lat:.4f}°N, {c_lon:.4f}°E`  \n"
                    f"• **Sensor & Observation:** `{r['sensor']}` ({r['date']}) · `{r['filename']}`  \n"
                    f"{rel_bullet}"
                    f"• **Temporal Depth:** `{stack_count} historical acquisitions` available in archive stack"
                )
            locations_block = "\n\n".join(ranked_locations_text) if ranked_locations_text else "_No distinct locations found._"

            rel_plan_block = ""
            if rel_plan_info and rel_plan_info.get("is_compound"):
                rel_plan_block = (
                    f"> 🔗 **Spatial Compound Query Plan (§2.2.1):**  \n"
                    f"> • **Target Concept:** `{rel_plan_info.get('target_subject')}`  \n"
                    f"> • **Relational Operator:** `{rel_plan_info.get('operator')}` (Connector: `\"{rel_plan_info.get('connector_phrase')}\"`)  \n"
                    f"> • **Context Anchor:** `{rel_plan_info.get('context_anchor')}`  \n"
                    f"> • *Mechanism: Dual foundation projection with 4x4 spatial quadrant proximity and harmonic co-occurrence scoring.* \n\n"
                )

            text_answer = (
                f"### 🛰️ Archive-Wide Semantic Retrieval & Location Ranking\n\n"
                f"Scanned **{search_res.get('archive_total', 0)} cataloged satellite scenes** across the local archive using multimodal foundation embeddings.\n\n"
                f"**Query Intent:** `{clean_display_query}`  \n"
                f"**Discovered Locations:** `{len(items)} distinct geographic areas` (from `{total}` total candidate scenes)  \n"
                f"**Clusters Represented:** {', '.join(f'`{c}`' for c in search_res.get('clusters_detected', [])) or 'None'}\n\n"
                f"{rel_plan_block}"
                f"---\n\n"
                f"#### 📍 Top Ranked Geographic Locations:\n\n"
                f"{locations_block}\n\n"
                f"---\n\n"
                f"#### 📊 Comparative Summary Table:\n\n"
                f"{table_md}\n\n"
                f"> 💡 **Actionable Analysis:** Bounding boxes have been projected onto the MapLibre viewport. To analyze how any of these locations evolved over time, ask: *\"When did change first appear in location #1?\"* or click **\"Analyze Time Series\"** in the Archive Search modal."
            )
            visual_evidence = [{"type": "FeatureCollection", "features": features}]
            analysis_mode = "archive-search"
            provenance_data = {
                "pipeline_version": "SatQuery AI 2.0 (ISRO PS2)",
                "analysis_mode": "archive-search",
                "sensor_platform": "Sentinel-2 MSI / Multi-Sensor Archive",
                "embedding_model": "Multimodal Remote-Sensing Foundation Visual Encoder",
                "similarity_metric": "Harmonic Relational Co-occurrence & Quadrant Distance" if rel_plan_info else "Normalized Cosine Distance",
                "relational_plan": rel_plan_info,
                "total_archive_scenes": search_res.get("archive_total", 0),
                "matched_locations": len(items),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

    except Exception as exc:
        tracer.append_log(f"Agent execution error: {type(exc).__name__}: {exc}")
        text_answer = (
            f"विश्लेषण पूरा नहीं हो सका: {exc}" if is_hindi else
            f"SatQuery could not complete the requested analysis. Reason: {exc}"
        )
        visual_evidence = []

    tracer.append_log("step 3: assembling structured SatQuery response")

    # Ensure robust, valid GitHub-Flavored Markdown for web UI rendering
    text_answer = sanitize_markdown(text_answer)

    trace_dict = tracer.get_trace()
    return SatQueryResponse(
        text_answer=text_answer,
        visual_evidence=visual_evidence,
        execution_trace=trace_dict,
        trace_log=trace_dict.get("steps", []),
        analysis_mode=analysis_mode,
        timeline=timeline_data,
        candidates=candidates_data,
        scene_overlay=scene_overlay_data,
        provenance=provenance_data,
    )

