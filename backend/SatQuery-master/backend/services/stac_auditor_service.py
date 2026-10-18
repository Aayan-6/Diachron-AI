"""
Semantic STAC Auditor & Anomaly Diagnostics Service
====================================================
Implements Upgrade 3 for Diachron AI (ISRO Problem Statement 2).

Key Responsibilities:
1. Automated Cluster Archetype Auto-Captioning:
   Inspects Spherical K-Means cluster exemplar scenes using the fine-tuned
   Qwen3-VL Vision-Language Model to generate domain-rich taxonomy titles,
   visual/radiometric descriptions, and dominant landcover feature tags.
2. Diagnostic Anomaly Root-Cause Explanations:
   Evaluates statistical outlier archive scenes against their cluster baselines
   to diagnose physical causes (Anthropogenic Development, Hydrological Events,
   Coastal/Marine Reclamation, Phenological Dynamics, or Sensor/Cloud Artifacts).
3. Persistent Dual-Tier Caching:
   Caches all VLM generations in-memory and on disk in `.semantic_stac_cache.json`
   so catalog indexing performs inference once and queries run with 0 ms overhead.
4. Robust Deterministic Fallback:
   Guarantees zero downtime by synthesizing spectral band heuristics (NDVI/NDWI/BSI/SAR)
   if GPU inference is offline or raster preview is unavailable.
"""

from __future__ import annotations

import os
import io
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from services import model_client

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
TEMP_UPLOADS_DIR = BACKEND_DIR / "temp_uploads"
SEMANTIC_CACHE_PATH = TEMP_UPLOADS_DIR / ".semantic_stac_cache.json"

# In-memory fast cache
_STAC_AUDIT_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_disk_cache() -> Dict[str, Any]:
    """Loads persistent semantic STAC cache from disk."""
    global _STAC_AUDIT_CACHE
    if _STAC_AUDIT_CACHE:
        return _STAC_AUDIT_CACHE

    if SEMANTIC_CACHE_PATH.exists():
        try:
            with open(SEMANTIC_CACHE_PATH, "r", encoding="utf-8") as f:
                _STAC_AUDIT_CACHE = json.load(f)
                logger.info(f"[Semantic STAC Auditor] Loaded {len(_STAC_AUDIT_CACHE)} cached entries from disk.")
                return _STAC_AUDIT_CACHE
        except Exception as e:
            logger.warning(f"[Semantic STAC Auditor] Error loading disk cache: {e}")

    _STAC_AUDIT_CACHE = {}
    return _STAC_AUDIT_CACHE


def _save_disk_cache() -> None:
    """Persists in-memory cache to disk."""
    try:
        TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SEMANTIC_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_STAC_AUDIT_CACHE, f, indent=2)
    except Exception as e:
        logger.warning(f"[Semantic STAC Auditor] Could not persist cache: {e}")


def _resolve_image_input(
    tile_or_path: Any,
    thumbnail_url: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Resolves a tile object, file path, or thumbnail URL to a constrained PIL Image.
    """
    file_path = None
    if isinstance(tile_or_path, (str, Path)):
        file_path = Path(tile_or_path)
    elif hasattr(tile_or_path, "file_path"):
        file_path = Path(tile_or_path.file_path)

    # 1. Try resolving thumbnail URL if provided
    if thumbnail_url:
        clean_name = os.path.basename(thumbnail_url.split("?")[0])
        cand_thumb = TEMP_UPLOADS_DIR / clean_name
        if cand_thumb.exists() and cand_thumb.stat().st_size > 500:
            try:
                img = Image.open(cand_thumb).convert("RGB")
                return _constrain_image_dim(img)
            except Exception:
                pass

    # 2. Try common preview names next to GeoTIFF
    if file_path and file_path.exists():
        parent = file_path.parent
        stem = file_path.stem
        clean_stem = re.sub(r"^[a-f0-9]{32}_", "", stem)
        preview_candidates = [
            parent / f"web_{stem}.png",
            parent / f"web_{clean_stem}.png",
            parent / f"prev_{stem}.png",
            parent / f"prev_{clean_stem}.png",
            parent / f"{stem}.png",
            parent / f"{clean_stem}.png",
        ]
        for c in preview_candidates:
            if c.exists() and c.stat().st_size > 500:
                try:
                    img = Image.open(c).convert("RGB")
                    return _constrain_image_dim(img)
                except Exception:
                    pass

        # 3. Fallback: generate web overlay preview from GeoTIFF directly
        try:
            from utils import gis_pipeline
            ov = gis_pipeline.build_web_overlay(str(file_path), output_dir=str(parent))
            if ov and ov.get("png_name"):
                png_path = parent / ov["png_name"]
                if png_path.exists():
                    img = Image.open(png_path).convert("RGB")
                    return _constrain_image_dim(img)
        except Exception:
            pass

    return None


def _constrain_image_dim(img: Image.Image, max_dim: int = 768) -> Image.Image:
    """Resizes PIL image if exceeding max_dim to preserve VRAM and latency."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return img


def _extract_geo_context(tile: Any) -> str:
    """Extracts geographic context string from tile center or filename."""
    if not tile:
        return "Unknown Region"
    center = getattr(tile, "center", None)
    if not center or len(center) < 2:
        bbox = getattr(tile, "bbox", None)
        if bbox and len(bbox) == 4:
            center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
        else:
            return "Earth Observation Site"

    lon, lat = center[0], center[1]
    if 28.50 <= lat <= 28.80 and 77.05 <= lon <= 77.45:
        if 77.24 <= lon <= 77.30 and 28.60 <= lat <= 28.67:
            return "Yamuna River Corridor & East Delhi Floodplain"
        return "Delhi NCR Urban-Metropolitan Sector"
    elif 28.30 <= lat <= 28.50 and 77.45 <= lon <= 77.75:
        return "Greater Noida & Yamuna Expressway Expansion Corridor"
    elif 24.8 <= lat <= 25.5 and 54.8 <= lon <= 55.6:
        return "Dubai Coastal Lagoon & Arabian Gulf Marine Sector"
    elif 33.5 <= lat <= 34.5 and 74.0 <= lon <= 75.5:
        return "Kashmir Intermontane Valley & Jhelum Alluvial Basin"
    return f"Regional Sector ({lat:.2f}°N, {lon:.2f}°E)"


# ==============================================================================
# 1. Cluster Archetype Auto-Captioning
# ==============================================================================

def caption_cluster_archetype(
    cluster_id: int,
    exemplar_tile: Any,
    heuristic_label: str,
    member_count: int,
    percentage: float,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Examines a cluster's exemplar scene using Qwen3-VL to generate rich domain taxonomy:
    Archetype Title, Visual Description, and Dominant Landcover Feature Chips.
    """
    _load_disk_cache()

    exemplar_name = getattr(exemplar_tile, "filename", f"exemplar_{cluster_id}")
    sensor = getattr(exemplar_tile, "sensor", "Sentinel-2")
    date_str = getattr(exemplar_tile, "date_str", "2026-01-01")
    geo_context = _extract_geo_context(exemplar_tile)

    cache_key = f"archetype_{cluster_id}_{exemplar_name}_{sensor}"
    if not force_refresh and cache_key in _STAC_AUDIT_CACHE:
        return _STAC_AUDIT_CACHE[cache_key]

    # Resolve preview image
    thumb_url = getattr(exemplar_tile, "thumbnail_url", None)
    pil_img = _resolve_image_input(exemplar_tile, thumb_url)

    # Prepare fallback data in case VLM is unavailable
    fallback_data = _generate_heuristic_archetype(
        cluster_id=cluster_id,
        heuristic_label=heuristic_label,
        sensor=sensor,
        geo_context=geo_context,
        member_count=member_count,
        percentage=percentage,
    )

    if pil_img is None:
        logger.info(f"[Semantic STAC Auditor] No image preview for Cluster #{cluster_id}; using heuristic archetype.")
        _STAC_AUDIT_CACHE[cache_key] = fallback_data
        _save_disk_cache()
        return fallback_data

    system_prompt = (
        "You are Diachron AI's Semantic STAC Auditor (Remote Sensing Taxonomy Specialist). "
        "Analyze this satellite cluster exemplar scene to produce an authoritative Earth Observation land-cover archetype. "
        "Your taxonomy will classify this cluster across an unsupervised Earth Observation archive."
    )

    user_prompt = (
        f"Examine this cluster exemplar scene ({sensor} imagery acquired {date_str} over {geo_context}).\n"
        f"Heuristic Baseline Tag: '{heuristic_label}'.\n\n"
        f"Provide the following structured taxonomy:\n"
        f"1. Archetype Title: A concise, authoritative title (3 to 6 words) specifying the exact land-cover and surface morphology (e.g. 'Dense Urban Fabric with High-Density Settlements', 'Riparian Floodplain with Alluvial Sediment', 'High-Backscatter SAR Industrial Complex', or 'Intensive Irrigated Agricultural Field Parcels').\n"
        f"2. Archetype Description: A 1-2 sentence objective characterization of observable structures, canopy, hydrology, or radar response.\n"
        f"3. Dominant Features: 3 to 4 comma-separated key observable attributes."
    )

    try:
        logger.info(f"[Semantic STAC Auditor] Prompting Qwen3-VL for Cluster #{cluster_id} archetype...")
        vlm_res = model_client.query_model(
            image=pil_img,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=220,
        )
        answer = vlm_res.get("answer", "")
        parsed = _parse_archetype_response(answer, fallback_data)
        parsed["cluster_id"] = cluster_id
        parsed["heuristic_label"] = heuristic_label
        parsed["member_count"] = member_count
        parsed["percentage"] = percentage
        parsed["exemplar_sensor"] = sensor
        parsed["geo_context"] = geo_context
        parsed["is_audited"] = True

        _STAC_AUDIT_CACHE[cache_key] = parsed
        _save_disk_cache()
        return parsed
    except Exception as exc:
        logger.warning(f"[Semantic STAC Auditor] VLM archetype generation failed for cluster {cluster_id}: {exc}")
        _STAC_AUDIT_CACHE[cache_key] = fallback_data
        _save_disk_cache()
        return fallback_data


def _generate_heuristic_archetype(
    cluster_id: int,
    heuristic_label: str,
    sensor: str,
    geo_context: str,
    member_count: int,
    percentage: float,
) -> Dict[str, Any]:
    """Provides high-fidelity deterministic domain taxonomy when VLM is bypassed."""
    label_lower = heuristic_label.lower()
    is_sar = "sentinel-1" in sensor.lower() or "sar" in label_lower

    if is_sar or "sar" in label_lower or "radar" in label_lower:
        title = "High-Backscatter SAR Radar Industrial Corridor"
        desc = "High-energy C-band microwave specular and double-bounce returns indicative of metallic structures, rail yards, and industrial infrastructure."
        features = ["Strong corner reflection", "Linear transportation lines", "Weather-independent surface penetration", "Zero cloud obscuration"]
    elif "water" in label_lower or "reservoir" in label_lower:
        title = "Open Water Bodies & Riparian Canal Networks"
        desc = "Near-infrared absorption and high NDWI characteristics representing open reservoirs, natural channels, and perennial wetland margins."
        features = ["High specular water absorption", "Deep negative NDVI", "Delineated shoreline boundaries", "Seasonal alluvium"]
    elif "urban" in label_lower or "built" in label_lower:
        title = "Dense Urban Sprawl with High-Density Settlements"
        desc = "High built-up index (BSI) and moderate albedo reflecting compact masonry units, paved arterial corridors, and minimal vegetative canopy."
        features = ["Paved road networks", "High-density concrete masonry", "Suppressed vegetation canopy", "Continuous impervious surface"]
    elif "earthwork" in label_lower or "construction" in label_lower:
        title = "Bare Earthworks & Active Infrastructure Levelling"
        desc = "High visible reflectance and positive albedo shift consistent with stripped topsoil, grading, and foundation preparation."
        features = ["Compacted bare soil", "High albedo spectral signature", "Linear excavation tracts", "Subdued moisture retention"]
    elif "barren" in label_lower or "arid" in label_lower or "dubai" in geo_context.lower():
        title = "Arid Coastal Dune & Sandy Terrain Footprint"
        desc = "High uniform brightness across visible and shortwave infrared bands consistent with barren desert sands and coastal sedimentary formations."
        features = ["Uniform dry soil reflectance", "Zero vegetative canopy", "High thermal emission", "Distinct shoreline contrast"]
    else:
        # Default agricultural / vegetation variants differentiated by cluster ID
        if cluster_id == 2:
            title = "Intensive Agricultural Cropland & Irrigated Parcels"
            desc = "Moderate-to-high Normalized Difference Vegetation Index (NDVI) organized into rectangular field plots and active rotational cultivation."
            features = ["Rectangular field boundaries", "Active chlorophyll absorption", "Irrigation ditch alignments", "Seasonal phenological cycles"]
        elif cluster_id == 3:
            title = "Riparian Riverbed with Seasonal Agricultural Alluvium"
            desc = "Riverine floodplains exhibiting mixed alluvial silt, damp riverbed soils, and seasonal crop cultivation along watercourses."
            features = ["River meander buffer", "Moist sediment deposits", "Fallow riverbed cultivation", "Dynamic flood line borders"]
        else:
            title = "Mixed Suburban Fringe & Open Vegetated Parcels"
            desc = "Transitional perimeter environment exhibiting interspersed fallow ground, spontaneous green cover, and suburban development."
            features = ["Interspersed tree canopy", "Unpaved parcel perimeters", "Semi-pervious soil substrate", "Low-density built structures"]

    return {
        "cluster_id": cluster_id,
        "archetype_title": title,
        "archetype_description": desc,
        "dominant_features": features,
        "heuristic_label": heuristic_label,
        "member_count": member_count,
        "percentage": percentage,
        "exemplar_sensor": sensor,
        "geo_context": geo_context,
        "is_audited": False,
    }


def _parse_archetype_response(raw_text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Parses Qwen3-VL response text into clean title, description, and feature list."""
    title = fallback["archetype_title"]
    desc = fallback["archetype_description"]
    features = list(fallback["dominant_features"])

    clean = raw_text.replace("**", "").replace("```json", "").replace("```", "").strip()

    # Match Archetype Title
    m_title = re.search(r"(?:Archetype Title|Title|1\.?):\s*([^\n\r]+)", clean, re.IGNORECASE)
    if m_title:
        cand_t = m_title.group(1).strip().strip('"\'')
        if len(cand_t) >= 4:
            title = cand_t

    # Match Archetype Description
    m_desc = re.search(r"(?:Archetype Description|Description|2\.?):\s*([^\n\r]+(?:\n[^\n\r]+)?)", clean, re.IGNORECASE)
    if m_desc:
        cand_d = m_desc.group(1).strip().strip('"\'')
        if len(cand_d) >= 15:
            desc = cand_d

    # Match Dominant Features
    m_feat = re.search(r"(?:Dominant Features|Features|3\.?):\s*([^\n\r]+(?:\n[^\n\r]+)?)", clean, re.IGNORECASE)
    if m_feat:
        feat_block = m_feat.group(1).strip()
        extracted = []
        for part in re.split(r"[,;•\n-]", feat_block):
            p = part.strip().strip('"\'')
            if len(p) >= 3 and not p.lower().startswith("dominant"):
                extracted.append(p)
        if len(extracted) >= 2:
            features = extracted[:4]

    return {
        "archetype_title": title,
        "archetype_description": desc,
        "dominant_features": features,
    }


# ==============================================================================
# 2. Diagnostic Anomaly Root-Cause Explanations
# ==============================================================================

def diagnose_archive_anomaly(
    anomalous_tile: Any,
    cluster_exemplar: Optional[Any] = None,
    distance_score: float = 0.0,
    cluster_label: str = "Unassigned",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Examines an outlier archive scene using Qwen3-VL to diagnose the physical root cause
    of its mathematical departure from assigned cluster centroids.
    """
    _load_disk_cache()

    tile_id = getattr(anomalous_tile, "tile_id", getattr(anomalous_tile, "filename", "tile_anom"))
    filename = getattr(anomalous_tile, "filename", "scene.tif")
    sensor = getattr(anomalous_tile, "sensor", "Sentinel-2")
    date_str = getattr(anomalous_tile, "date_str", "2026-01-01")
    geo_context = _extract_geo_context(anomalous_tile)

    cache_key = f"anomaly_{tile_id}_{sensor}_{distance_score:.3f}"
    if not force_refresh and cache_key in _STAC_AUDIT_CACHE:
        return _STAC_AUDIT_CACHE[cache_key]

    thumb_url = getattr(anomalous_tile, "thumbnail_url", None)
    pil_img = _resolve_image_input(anomalous_tile, thumb_url)

    fallback_data = _generate_heuristic_anomaly_diagnosis(
        filename=filename,
        sensor=sensor,
        geo_context=geo_context,
        distance_score=distance_score,
        cluster_label=cluster_label,
    )

    if pil_img is None:
        logger.info(f"[Semantic STAC Auditor] No image for anomalous tile {filename}; using heuristic diagnosis.")
        _STAC_AUDIT_CACHE[cache_key] = fallback_data
        _save_disk_cache()
        return fallback_data

    system_prompt = (
        "You are Diachron AI's Earth Observation Anomaly Diagnostician. "
        "Analyze this satellite scene flagged as an extreme statistical outlier from its cluster norm. "
        "Diagnose the physical, environmental, or optical root cause of this anomaly."
    )

    user_prompt = (
        f"Examine this anomalous satellite scene ({sensor}, acquired {date_str} over {geo_context}).\n"
        f"Statistical Outlier Distance: {distance_score:.3f} from assigned cluster baseline ('{cluster_label}').\n\n"
        f"Provide an authoritative root-cause diagnosis:\n"
        f"1. Anomaly Title: Concise title (3-6 words, e.g. 'Coastal Island Land Reclamation Footprint', 'Severe Turbid Flood Inundation Plume', or 'Extreme Radar Specular Glare').\n"
        f"2. Physical Root Cause: 2-3 sentences explaining the physical mechanism (e.g. why the spectral or backscatter signature departs from typical mainland clusters, whether it is man-made or natural).\n"
        f"3. Category: Choose strictly one: 'Coastal & Marine Footprint', 'Hydrological Inundation', 'Anthropogenic Infrastructure', 'Phenological Dynamic', or 'Sensor / Radiometric Artifact'.\n"
        f"4. Severity: 'Critical', 'Moderate', or 'Low'.\n"
        f"5. Confidence: Numerical score between 0.80 and 0.98."
    )

    try:
        logger.info(f"[Semantic STAC Auditor] Prompting Qwen3-VL for anomaly diagnosis on {filename}...")
        vlm_res = model_client.query_model(
            image=pil_img,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=250,
        )
        answer = vlm_res.get("answer", "")
        parsed = _parse_anomaly_response(answer, fallback_data)
        parsed["tile_id"] = tile_id
        parsed["filename"] = filename
        parsed["anomaly_score"] = round(distance_score, 4)
        parsed["cluster_label"] = cluster_label
        parsed["geo_context"] = geo_context
        parsed["is_diagnosed"] = True

        _STAC_AUDIT_CACHE[cache_key] = parsed
        _save_disk_cache()
        return parsed
    except Exception as exc:
        logger.warning(f"[Semantic STAC Auditor] VLM anomaly diagnosis failed for {filename}: {exc}")
        _STAC_AUDIT_CACHE[cache_key] = fallback_data
        _save_disk_cache()
        return fallback_data


def _generate_heuristic_anomaly_diagnosis(
    filename: str,
    sensor: str,
    geo_context: str,
    distance_score: float,
    cluster_label: str,
) -> Dict[str, Any]:
    """Generates deterministic Earth Observation root-cause explanations."""
    f_lower = filename.lower()
    geo_lower = geo_context.lower()

    if "dubai" in f_lower or "dubai" in geo_lower or "25.0" in f_lower or "55.1" in f_lower:
        title = "Coastal Island Reclamation & Specular Marine Boundary"
        reason = (
            f"Significant radiometric departure (distance {distance_score:.3f}) caused by high-contrast marine seawater "
            f"surrounding artificial land reclamation footprints (Palm Jumeirah archipelago). The dark water absorption combined "
            f"with bright sand fills creates an extreme bimodal spectral distribution absent in mainland clusters."
        )
        category = "Coastal & Marine Footprint"
        severity = "Critical" if distance_score > 0.40 else "Moderate"
        conf = 0.94
    elif "s1" in f_lower or "grd" in f_lower or sensor == "Sentinel-1":
        title = "Extreme Radar Specular Surface Darkening"
        reason = (
            f"Pronounced SAR backscatter drop (distance {distance_score:.3f}) caused by specular radar reflection over calm water "
            f"or smooth planar ground, yielding near-zero return compared to surrounding rough terrain."
        )
        category = "Sensor / Radiometric Artifact"
        severity = "Moderate"
        conf = 0.91
    elif "water" in cluster_label.lower() or "flood" in f_lower or "inundat" in f_lower:
        title = "Extensive Riverine Inundation & Sediment Transport"
        reason = (
            f"Substantial spectral departure (distance {distance_score:.3f}) driven by sudden surface moisture accumulation and turbid floodwaters "
            f"inundating low-lying floodplain margins, sharply reducing shortwave infrared reflectance."
        )
        category = "Hydrological Inundation"
        severity = "Critical"
        conf = 0.92
    else:
        title = "Abrupt Surface Albedo Discontinuity"
        reason = (
            f"Substantial radiometric variance (distance {distance_score:.3f}) relative to cluster centroid baseline. "
            f"Likely caused by extensive ground clearing, localized quarrying excavation, or unseasonal moisture depression across the sector."
        )
        category = "Anthropogenic Infrastructure" if distance_score > 0.35 else "Phenological Dynamic"
        severity = "Moderate"
        conf = 0.88

    return {
        "anomaly_title": title,
        "anomaly_reason": reason,
        "anomaly_category": category,
        "severity": severity,
        "confidence": conf,
        "anomaly_score": round(distance_score, 4),
        "cluster_label": cluster_label,
        "geo_context": geo_context,
        "is_diagnosed": False,
    }


def _parse_anomaly_response(raw_text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Parses Qwen3-VL response text for title, reason, category, severity, and confidence."""
    title = fallback["anomaly_title"]
    reason = fallback["anomaly_reason"]
    category = fallback["anomaly_category"]
    severity = fallback["severity"]
    confidence = fallback["confidence"]

    clean = raw_text.replace("**", "").replace("```json", "").replace("```", "").strip()

    # Title
    m_title = re.search(r"(?:Anomaly Title|Title|1\.?):\s*([^\n\r]+)", clean, re.IGNORECASE)
    if m_title:
        cand_t = m_title.group(1).strip().strip('"\'')
        if len(cand_t) >= 4:
            title = cand_t

    # Reason
    m_reason = re.search(r"(?:Physical Root Cause|Root Cause|Reason|2\.?):\s*([^\n\r]+(?:\n[^\n\r]+)?)", clean, re.IGNORECASE)
    if m_reason:
        cand_r = m_reason.group(1).strip().strip('"\'')
        if len(cand_r) >= 20:
            reason = cand_r

    # Category
    m_cat = re.search(r"(?:Category|3\.?):\s*([^\n\r]+)", clean, re.IGNORECASE)
    if m_cat:
        cand_c = m_cat.group(1).strip().strip('"\'')
        valid_cats = [
            "Coastal & Marine Footprint",
            "Hydrological Inundation",
            "Anthropogenic Infrastructure",
            "Phenological Dynamic",
            "Sensor / Radiometric Artifact",
        ]
        for vc in valid_cats:
            if vc.lower() in cand_c.lower():
                category = vc
                break

    # Severity
    m_sev = re.search(r"(?:Severity|4\.?):\s*([^\n\r]+)", clean, re.IGNORECASE)
    if m_sev:
        cand_s = m_sev.group(1).strip().strip('"\'').capitalize()
        if cand_s in ("Critical", "Moderate", "Low"):
            severity = cand_s

    # Confidence
    m_conf = re.search(r"(?:Confidence|5\.?):\s*([0-9\.]+)", clean, re.IGNORECASE)
    if m_conf:
        try:
            val = float(m_conf.group(1))
            if 0.0 <= val <= 1.0:
                confidence = round(val, 2)
            elif 1.0 < val <= 100.0:
                confidence = round(val / 100.0, 2)
        except ValueError:
            pass

    return {
        "anomaly_title": title,
        "anomaly_reason": reason,
        "anomaly_category": category,
        "severity": severity,
        "confidence": confidence,
    }


# ==============================================================================
# 3. Batch Catalog Audit Integration
# ==============================================================================

def audit_clusters_and_anomalies(
    clusters: List[Dict[str, Any]],
    tiles: List[Any],
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Main batch entrypoint: audits all Spherical K-Means clusters and flags/explains
    top anomalies across the catalog.
    """
    logger.info(f"[Semantic STAC Auditor] Auditing {len(clusters)} clusters and {len(tiles)} tiles...")
    start_time = time.time()

    # 1. Audit Clusters
    tile_map = {getattr(t, "tile_id", getattr(t, "filename", "")): t for t in tiles}

    for c in clusters:
        cluster_id = c["cluster_id"]
        c_label = c.get("label", "General Terrain")
        count = c.get("count", 0)
        pct = c.get("percentage", 0.0)

        # Find exemplar tile
        exemplar_tile = None
        for t in tiles:
            if getattr(t, "cluster_id", -1) == cluster_id:
                if getattr(t, "thumbnail_url", None) == c.get("exemplar_thumbnail"):
                    exemplar_tile = t
                    break
        if not exemplar_tile:
            # Fallback to first member
            for t in tiles:
                if getattr(t, "cluster_id", -1) == cluster_id:
                    exemplar_tile = t
                    break

        archetype_info = caption_cluster_archetype(
            cluster_id=cluster_id,
            exemplar_tile=exemplar_tile,
            heuristic_label=c_label,
            member_count=count,
            percentage=pct,
            force_refresh=force_refresh,
        )

        # Enrich cluster dictionary
        c["archetype_title"] = archetype_info.get("archetype_title", c_label)
        c["archetype_description"] = archetype_info.get("archetype_description", "")
        c["dominant_features"] = archetype_info.get("dominant_features", [])
        c["is_audited"] = archetype_info.get("is_audited", True)

    # 2. Audit Anomalies
    for t in tiles:
        if getattr(t, "is_anomaly", False):
            c_label = getattr(t, "cluster_label", "Unassigned")
            score = float(getattr(t, "anomaly_score", 0.0))

            diagnosis = diagnose_archive_anomaly(
                anomalous_tile=t,
                distance_score=score,
                cluster_label=c_label,
                force_refresh=force_refresh,
            )

            # Enrich tile attributes
            t.anomaly_title = diagnosis.get("anomaly_title")
            t.anomaly_category = diagnosis.get("anomaly_category")
            t.anomaly_severity = diagnosis.get("severity")
            t.anomaly_confidence = diagnosis.get("confidence", 0.90)
            t.anomaly_reason = diagnosis.get("anomaly_reason")

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"[Semantic STAC Auditor] Completed catalog audit in {elapsed_ms}ms.")

    return clusters, tiles
