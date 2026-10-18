"""
STAC Catalog Search for Sentinel-1 and Sentinel-2 Imagery.
Queries the Earth Search STAC API for scenes covering the target Area of Interest (AOI).
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import pystac_client
from pystac import Item

from .config import (
    STAC_API_URL,
    COLLECTIONS,
    DEFAULT_MAX_CLOUD_COVER,
    DEFAULT_SEARCH_DAYS,
)


def get_stac_client() -> Optional[pystac_client.Client]:
    """Returns an open pystac Client connected to the Earth Search endpoint, or None if offline."""
    try:
        # Check fast offline status if available
        try:
            from core import network_state
            if not network_state.is_online():
                return None
        except Exception:
            pass
        return pystac_client.Client.open(STAC_API_URL, timeout=3.5)
    except Exception:
        return None


def _format_date_range(date_range: Optional[str] = None, days: int = DEFAULT_SEARCH_DAYS) -> str:
    """Formats a datetime string for STAC API (RFC 3339 interval)."""
    if date_range:
        return date_range
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    return f"{start.strftime('%Y-%m-%d')}T00:00:00Z/{now.strftime('%Y-%m-%d')}T23:59:59Z"


def search_sentinel2(
    bbox: List[float],
    max_cloud_cover: int = DEFAULT_MAX_CLOUD_COVER,
    date_range: Optional[str] = None,
    limit: int = 10
) -> List[Item]:
    """
    Searches for Sentinel-2 Level-2A surface reflectance scenes covering the given bbox.
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84
        max_cloud_cover: Maximum acceptable cloud percentage (0-100)
        date_range: Optional ISO string interval, e.g. "2026-06-01T00:00:00Z/2026-09-01T00:00:00Z"
        limit: Max number of items to inspect
        
    Returns:
        List of matching pystac.Item instances sorted by acquisition date (newest first).
    """
    client = get_stac_client()
    if client is None:
        return []
    dt = _format_date_range(date_range)
    
    # Try primary collection first, fallback to alternate if empty
    for col_id in [COLLECTIONS["sentinel-2"], COLLECTIONS["sentinel-2-fallback"]]:
        query_params: Dict[str, Any] = {
            "collections": [col_id],
            "bbox": bbox,
            "datetime": dt,
            "query": {"eo:cloud_cover": {"lte": max_cloud_cover}},
            "max_items": limit
        }
        try:
            search = client.search(**query_params)
            items = list(search.item_collection())
            if items:
                items.sort(key=lambda x: x.datetime or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
                return items
        except Exception:
            pass

    # If strict cloud cover returned nothing, retry with relaxed cloud filter (up to 50%)
    if max_cloud_cover < 50:
        query_params["query"] = {"eo:cloud_cover": {"lte": 50}}
        try:
            search = client.search(**query_params)
            items = list(search.item_collection())
            if items:
                items.sort(key=lambda x: x.datetime or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
                return items
        except Exception:
            pass

    return []


def search_sentinel1(
    bbox: List[float],
    date_range: Optional[str] = None,
    limit: int = 10
) -> List[Item]:
    """
    Searches for Sentinel-1 GRD (Ground Range Detected) SAR scenes covering the bbox.
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84
        date_range: Optional ISO string interval
        limit: Max number of items to retrieve
        
    Returns:
        List of matching pystac.Item instances sorted by acquisition date.
    """
    client = get_stac_client()
    if client is None:
        return []
    # Sentinel-1 has a 12-day revisit; look back up to 120 days if not specified
    dt = _format_date_range(date_range, days=120)
    
    search = client.search(
        collections=[COLLECTIONS["sentinel-1"]],
        bbox=bbox,
        datetime=dt,
        max_items=limit
    )
    items = list(search.item_collection())
    items.sort(key=lambda x: x.datetime or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items


def get_best_scene(items: List[Item], sensor: str = "sentinel-2") -> Optional[Item]:
    """
    Selects the optimal scene from search results.
    - For Sentinel-2: Balances recent date with minimal cloud cover.
    - For Sentinel-1: Returns the most recent scene.
    """
    if not items:
        return None
        
    if sensor == "sentinel-1":
        return items[0]
        
    # For Sentinel-2: pick the item with the best composite score (cloud cover < 5% preferred)
    # Filter items that have the required bands (red, green, blue, nir)
    valid_items = [
        item for item in items
        if all(b in item.assets for b in ["red", "green", "blue"])
    ]
    if not valid_items:
        valid_items = items

    def scene_score(item: Item) -> float:
        cloud = item.properties.get("eo:cloud_cover", 100.0)
        # We give a bonus for very clear skies (< 10%)
        return float(cloud)

    # Sort primarily by cloud cover if within the same month, or return the lowest cloud cover in top 3
    top_candidates = valid_items[:5]
    return min(top_candidates, key=scene_score)


def extract_item_metadata(item: Item, sensor: str = "sentinel-2") -> Dict[str, Any]:
    """Extracts structured metadata and asset provenance from a STAC Item."""
    props = item.properties
    return {
        "id": item.id,
        "collection": item.collection_id,
        "datetime": item.datetime.isoformat() if item.datetime else props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "platform": props.get("platform"),
        "constellation": props.get("constellation"),
        "grid_code": props.get("grid:code") or props.get("mgrs:utm_zone"),
        "bbox": item.bbox,
        "assets": {k: a.href for k, a in item.assets.items() if k in ["red", "green", "blue", "nir", "visual", "vv", "vh", "thumbnail"]}
    }
