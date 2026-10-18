from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services import archive_indexer

logger = logging.getLogger(__name__)

router = APIRouter()


class ArchiveSearchRequest(BaseModel):
    query: Optional[str] = Field(
        None,
        description="Free-text natural language query (e.g., 'water bodies', 'newly built structures near a river').",
    )
    search_mode: str = Field(
        "text",
        description="'text' (text-to-image) or 'image' (image-to-image similarity).",
    )
    reference_tile_id: Optional[str] = Field(
        None,
        description="Tile ID of the query reference tile for image-to-image similarity search.",
    )
    reference_file_path: Optional[str] = Field(
        None,
        description="Optional path to a custom uploaded query reference raster.",
    )
    sensor: Optional[str] = Field(
        "all",
        description="'all', 'sentinel-2' (optical), or 'sentinel-1' (SAR radar).",
    )
    bbox: Optional[List[float]] = Field(
        None,
        description="Optional [min_lon, min_lat, max_lon, max_lat] spatial bounding box filter.",
    )
    start_date: Optional[str] = Field(
        None,
        description="Earliest acquisition date in YYYY-MM-DD.",
    )
    end_date: Optional[str] = Field(
        None,
        description="Latest acquisition date in YYYY-MM-DD.",
    )
    max_cloud_cover: Optional[float] = Field(
        30.0,
        description="Maximum cloud coverage percentage for optical scenes (0-100).",
    )
    min_similarity: float = Field(
        0.0,
        description="Minimum cosine similarity threshold (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )
    cluster_id: Optional[int] = Field(
        None,
        description="Optional filter by cluster ID.",
    )
    limit: int = Field(
        15,
        description="Maximum number of candidate results to return.",
        ge=1,
        le=100,
    )
    group_by_location: bool = Field(
        True,
        description="Whether to deduplicate/group scenes into spatial location archetypes.",
    )


class TimelineStackRequest(BaseModel):
    bbox: List[float] = Field(
        ...,
        description="[min_lon, min_lat, max_lon, max_lat] in decimal degrees.",
        min_items=4,
        max_items=4,
    )


@router.post("/search")
async def search_archive(request: ArchiveSearchRequest) -> Dict[str, Any]:
    """
    Archive-wide multimodal semantic retrieval.
    Supports free-text natural language queries and image-to-image visual similarity.
    Applies spatial AOI, temporal window, and sensor filtering.
    """
    indexer = archive_indexer.get_archive_indexer()
    results = indexer.search(
        query=request.query,
        search_mode=request.search_mode,
        reference_tile_id=request.reference_tile_id,
        reference_file_path=request.reference_file_path,
        sensor=request.sensor,
        bbox=request.bbox,
        start_date=request.start_date,
        end_date=request.end_date,
        max_cloud_cover=request.max_cloud_cover,
        min_similarity=request.min_similarity,
        cluster_id=request.cluster_id,
        limit=request.limit,
        group_by_location=request.group_by_location,
    )
    return results


@router.get("/clusters")
async def get_archive_clusters() -> Dict[str, Any]:
    """
    Retrieves archive cluster distribution, semantic archetype labels,
    exemplar thumbnails, and top detected physical anomalies.
    """
    indexer = archive_indexer.get_archive_indexer()
    if not indexer.is_indexed():
        indexer.scan_and_index()

    summary = indexer.get_summary()

    # Extract top anomalies
    anomalies = [
        t.to_dict() for t in indexer.tiles if t.is_anomaly
    ]
    anomalies.sort(key=lambda a: a["anomaly_score"], reverse=True)

    return {
        "total_tiles": summary["total_tiles"],
        "clusters": summary["clusters"],
        "sensor_breakdown": summary["sensor_breakdown"],
        "anomalies_count": len(anomalies),
        "anomalies": anomalies[:8],
    }


@router.post("/index/rebuild")
async def rebuild_archive_index() -> Dict[str, Any]:
    """
    Forces an asynchronous or immediate rebuild of the archive vector embeddings,
    k-means clustering, and spatial index.
    """
    indexer = archive_indexer.get_archive_indexer()
    summary = indexer.scan_and_index(force_rebuild=True)
    return {
        "status": "success",
        "message": f"Archive index rebuilt successfully across {summary['total_tiles']} scenes.",
        "summary": summary,
    }


@router.get("/tile/{tile_id}/timeline")
async def get_tile_timeline(tile_id: str) -> Dict[str, Any]:
    """
    Given a tile ID, looks up its geographic footprint and returns the full chronological
    stack of observations (T0 -> Tk) at that location for multi-temporal inspection.
    """
    indexer = archive_indexer.get_archive_indexer()
    target_tile = None
    for t in indexer.tiles:
        if t.tile_id == tile_id:
            target_tile = t
            break

    if not target_tile:
        raise HTTPException(status_code=404, detail=f"Tile '{tile_id}' not found in archive.")

    stack = indexer.get_location_temporal_stack(target_tile.bbox)
    return {
        "tile_id": tile_id,
        "filename": target_tile.filename,
        "date": target_tile.date_str,
        "bbox": target_tile.bbox,
        "n_scenes": len(stack),
        "timeline_stack": stack,
    }


@router.post("/location/timeline")
async def get_location_timeline(req: TimelineStackRequest) -> Dict[str, Any]:
    """
    Returns all scenes intersecting the provided bounding box, sorted chronologically.
    Directly feeds into the Multi-Temporal Timeline Slider.
    """
    indexer = archive_indexer.get_archive_indexer()
    stack = indexer.get_location_temporal_stack(req.bbox)
    return {
        "bbox": req.bbox,
        "n_scenes": len(stack),
        "timeline_stack": stack,
    }


class IngestFolderRequest(BaseModel):
    folder_path: str = Field(..., description="Absolute local directory path (e.g. E:\\Judge_Data or C:\\Images)")
    scope_as_active: bool = Field(True, description="Whether to isolate the active archive scope strictly to this folder")


@router.post("/ingest-folder")
async def ingest_folder(req: IngestFolderRequest) -> Dict[str, Any]:
    """
    Ingests and indexes an arbitrary local directory (e.g., from a USB pen drive).
    Computes visual embeddings, extracts metadata, and scopes the archive view to only these scenes.
    Works 100% offline or online.
    """
    indexer = archive_indexer.get_archive_indexer()
    try:
        res = indexer.ingest_custom_folder(req.folder_path, scope_as_active=req.scope_as_active)
        return res
    except Exception as e:
        logger.error(f"Error ingesting folder '{req.folder_path}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload-folder")
async def upload_folder_files(
    files: List[Any],
) -> Dict[str, Any]:
    """Uploads multiple files into a dedicated subfolder and ingests them."""
    import time
    import shutil
    from services.archive_indexer import TEMP_UPLOADS_DIR
    target_sub = TEMP_UPLOADS_DIR / "pen_drive_imports" / f"import_{int(time.time())}"
    target_sub.mkdir(parents=True, exist_ok=True)

    for f in files:
        fname = getattr(f, "filename", "unnamed.tif")
        f_path = target_sub / fname
        with open(f_path, "wb") as out:
            shutil.copyfileobj(f.file, out)

    indexer = archive_indexer.get_archive_indexer()
    try:
        res = indexer.ingest_custom_folder(target_sub, scope_as_active=True)
        return res
    except Exception as e:
        logger.error(f"Error ingesting uploaded folder files: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset-scope")
async def reset_archive_scope() -> Dict[str, Any]:
    """
    Resets the archive view from an ingested custom folder back to the standard catalog.
    """
    indexer = archive_indexer.get_archive_indexer()
    return indexer.reset_to_default_archive()


@router.get("/network-status")
async def get_archive_network_status() -> Dict[str, Any]:
    """
    Returns connectivity state, indicating whether the platform is in online or air-gapped offline mode.
    """
    from core import network_state
    return network_state.get_network_status()

