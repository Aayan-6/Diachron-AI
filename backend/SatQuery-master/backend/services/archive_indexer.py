from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from services import embedding_service, multi_temporal_service
from utils import gis_pipeline

logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
TEMP_UPLOADS_DIR = BACKEND_DIR / "temp_uploads"
INDEX_CACHE_PATH = TEMP_UPLOADS_DIR / ".archive_index.json"
EMBEDDINGS_CACHE_PATH = TEMP_UPLOADS_DIR / ".archive_embeddings.npy"

DEFAULT_NUM_CLUSTERS = 5


def _extract_bbox_from_filename(name: str) -> Optional[List[float]]:
    """Fallback: extracts [lon, lat] from filenames like '..._77.28_28.64.tif' and expands to a bbox."""
    m = re.search(r"(\d+\.\d+)[_,-](\d+\.\d+)", name)
    if m:
        try:
            lon = float(m.group(1))
            lat = float(m.group(2))
            delta = 0.04  # roughly ~4km AOI radius
            return [
                round(lon - delta, 6),
                round(lat - delta, 6),
                round(lon + delta, 6),
                round(lat + delta, 6),
            ]
        except ValueError:
            pass
    return None


def _calculate_bbox_iou(b1: List[float], b2: List[float]) -> float:
    if len(b1) < 4 or len(b2) < 4:
        return 0.0
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _are_same_spatial_site(
    b1: List[float],
    b2: List[float],
    c1: Optional[List[float]] = None,
    c2: Optional[List[float]] = None,
) -> bool:
    """
    Determines if two scenes belong to the exact same localized ground footprint.
    Requires significant spatial IoU or close center proximity.
    """
    iou = _calculate_bbox_iou(b1, b2)
    if iou >= 0.25:
        return True
    if c1 and c2:
        dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        if dist < 0.045:
            return True
    return False


def _get_site_geographical_label(center: List[float], cluster_label: str = "") -> str:
    """Provides a clear, human-recognizable geographical name for a spatial cluster."""
    lon, lat = center[0], center[1]
    # Delhi NCR Sub-regions
    if 28.50 <= lat <= 28.80 and 77.05 <= lon <= 77.45:
        if lon >= 77.29 and 28.58 <= lat <= 28.67:
            return "East Delhi Core (Patparganj / Anand Vihar)"
        elif lon >= 77.33 and lat >= 28.68:
            return "Hindon Airfield & Ghaziabad Corridor"
        elif 77.24 <= lon <= 77.29 and 28.60 <= lat <= 28.66:
            return "Yamuna River Corridor (Mayur Vihar)"
        elif 77.18 <= lon <= 77.24 and 28.60 <= lat <= 28.66:
            return "Central Delhi (Connaught Place / Civil Lines)"
        elif lon <= 77.18 and lat >= 28.62:
            return "West Delhi (Karol Bagh / Rajouri)"
        elif lat <= 28.60 and lon <= 77.20:
            return "South Delhi (Vasant Kunj / Mehrauli)"
        elif lat >= 28.68:
            return "North Delhi (Wazirabad / Rohini)"
        return "Delhi NCR Urban Sector"
    elif 28.30 <= lat <= 28.50 and 77.45 <= lon <= 77.75:
        return "Greater Noida & Yamuna Expressway"
    elif 33.5 <= lat <= 34.5 and 74.0 <= lon <= 75.5:
        return "Kashmir Valley (Srinagar & Pulwama Basin)"
    elif 24.8 <= lat <= 25.5 and 54.8 <= lon <= 55.6:
        if lat < 25.20:
            return "Dubai Coastal Sector (Palm Jumeirah)"
        return "Dubai Creek & Downtown"

    if cluster_label and cluster_label != "Unassigned":
        return f"{cluster_label} Sector ({lat:.3f}°N, {lon:.3f}°E)"
    return f"Observation Site ({lat:.3f}°N, {lon:.3f}°E)"


def _bboxes_intersect(b1: List[float], b2: List[float]) -> bool:
    """Checks whether two [min_lon, min_lat, max_lon, max_lat] bounding boxes intersect."""
    if len(b1) < 4 or len(b2) < 4:
        return True
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


def _find_thumbnail_url(tif_path: Path) -> Optional[str]:
    """Finds matching preview PNG or web overlay PNG for a GeoTIFF, building it if missing."""
    tif_path = Path(tif_path)
    stem = tif_path.stem
    parent = tif_path.parent
    clean_stem = re.sub(r"^[a-f0-9]{32}_", "", stem)

    # Candidate preview filenames
    candidates = [
        parent / f"web_{stem}.png",
        parent / f"web_{clean_stem}.png",
        parent / f"preview_{stem}.png",
        parent / f"preview_{clean_stem}.png",
        parent / f"prev_{stem}.png",
        parent / f"prev_{clean_stem}.png",
        parent / f"{stem}.png",
        parent / f"{clean_stem}.png",
    ]

    for c in candidates:
        if c.exists() and c.stat().st_size > 500:
            return f"/static/{c.name}"

    # If stem has prefixes like 'proc_raw_', try checking 'web_raw_...'
    if clean_stem.startswith("proc_"):
        raw_stem = clean_stem.replace("proc_", "")
        for c in (parent / f"web_{raw_stem}.png", parent / f"preview_{raw_stem}.png", parent / f"prev_{raw_stem}.png"):
            if c.exists() and c.stat().st_size > 500:
                return f"/static/{c.name}"

    # On-demand fallback: generate web overlay PNG
    try:
        from utils import gis_pipeline
        ov = gis_pipeline.build_web_overlay(str(tif_path), output_dir=str(parent))
        if ov and ov.get("png_name"):
            return f"/static/{ov['png_name']}"
    except Exception:
        pass

    return None


class ArchiveTile:
    """Represents a single cataloged satellite scene or tile."""

    def __init__(
        self,
        tile_id: str,
        filename: str,
        file_path: str,
        date_str: str,
        sensor: str,
        bbox: List[float],
        center: List[float],
        cloud_cover: float,
        thumbnail_url: Optional[str],
        embedding: np.ndarray,
        cluster_id: int = -1,
        cluster_label: str = "Unassigned",
        anomaly_score: float = 0.0,
        is_anomaly: bool = False,
        anomaly_reason: Optional[str] = None,
        anomaly_title: Optional[str] = None,
        anomaly_category: Optional[str] = None,
        anomaly_severity: Optional[str] = None,
        anomaly_confidence: float = 0.0,
        archetype_title: Optional[str] = None,
    ):
        self.tile_id = tile_id
        self.filename = filename
        self.file_path = file_path
        self.date_str = date_str
        self.sensor = sensor
        self.bbox = bbox
        self.center = center
        self.cloud_cover = cloud_cover
        self.thumbnail_url = thumbnail_url
        self.embedding = embedding
        self.cluster_id = cluster_id
        self.cluster_label = cluster_label
        self.anomaly_score = anomaly_score
        self.is_anomaly = is_anomaly
        self.anomaly_reason = anomaly_reason
        self.anomaly_title = anomaly_title
        self.anomaly_category = anomaly_category
        self.anomaly_severity = anomaly_severity
        self.anomaly_confidence = anomaly_confidence
        self.archetype_title = archetype_title

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "date": self.date_str,
            "sensor": self.sensor,
            "bbox": self.bbox,
            "center": self.center,
            "cloud_cover": round(self.cloud_cover, 1),
            "thumbnail_url": self.thumbnail_url,
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "archetype_title": self.archetype_title or self.cluster_label,
            "anomaly_score": round(self.anomaly_score, 4),
            "is_anomaly": self.is_anomaly,
            "anomaly_reason": self.anomaly_reason,
            "anomaly_title": self.anomaly_title,
            "anomaly_category": self.anomaly_category,
            "anomaly_severity": self.anomaly_severity,
            "anomaly_confidence": round(self.anomaly_confidence, 3) if self.anomaly_confidence else 0.0,
        }


class ArchiveIndexer:
    """
    Maintains the local Earth-Observation archive index, vector embeddings,
    spherical k-means clustering model, and multi-temporal spatial groupings.
    """

    def __init__(self, data_dir: Path = TEMP_UPLOADS_DIR):
        self.data_dir = data_dir
        self.tiles: List[ArchiveTile] = []
        self.embedding_matrix: np.ndarray = np.empty((0, embedding_service.EMBEDDING_DIM), dtype=np.float32)
        self.clusters: List[Dict[str, Any]] = []
        self._is_indexed = False
        self._lock = asyncio.Lock()

        # Pen-drive / local folder ingestion state
        self.active_folder_path: Optional[str] = None
        self.active_folder_name: Optional[str] = None
        self.is_scoped_to_folder: bool = False
        self._default_tiles: List[ArchiveTile] = []
        self._default_embedding_matrix: Optional[np.ndarray] = None
        self._default_clusters: List[Dict[str, Any]] = []

    def is_indexed(self) -> bool:
        return self._is_indexed and len(self.tiles) > 0

    def get_total_count(self) -> int:
        return len(self.tiles)

    def scan_and_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Synchronously scans the archive directory, reads/generates embeddings,
        clusters the archive, and persists index caches.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tif_files = sorted(list(self.data_dir.glob("*.tif")) + list(self.data_dir.glob("*.tiff")))

        # Filter out temporary intermediate files or aligned masks
        clean_files: List[Path] = []
        for f in tif_files:
            if "mask" in f.name.lower() or "aligned" in f.name.lower() or f.stat().st_size < 10000:
                continue
            clean_files.append(f)

        logger.info(f"ArchiveIndexer scanning {len(clean_files)} candidate GeoTIFF scenes in {self.data_dir}")

        cached_index: Dict[str, Any] = {}
        if not force_rebuild and INDEX_CACHE_PATH.exists() and EMBEDDINGS_CACHE_PATH.exists():
            try:
                with open(INDEX_CACHE_PATH, "r", encoding="utf-8") as f:
                    cached_index = json.load(f)
                cached_matrix = np.load(str(EMBEDDINGS_CACHE_PATH))
                # Check if file count and mtimes match
                cached_files = cached_index.get("files", {})
                all_match = True
                if len(cached_files) == len(clean_files) and cached_matrix.shape[0] == len(clean_files):
                    for p in clean_files:
                        if p.name not in cached_files or abs(p.stat().st_mtime - cached_files[p.name].get("mtime", 0)) > 1.0:
                            all_match = False
                            break
                    if all_match:
                        logger.info("ArchiveIndexer: Loading existing cached index and embeddings.")
                        self._load_from_cache(cached_index, cached_matrix)
                        return self.get_summary()
            except Exception as e:
                logger.warning(f"Could not load archive cache: {e}. Rebuilding index...")

        # Recompute / update index
        tiles_list: List[ArchiveTile] = []
        embeddings_list: List[np.ndarray] = []
        files_meta: Dict[str, Any] = {}

        for p in clean_files:
            try:
                # Extract acquisition date
                dt = multi_temporal_service.extract_scene_date(p)
                date_str = dt.strftime("%Y-%m-%d")

                # Extract sensor
                name_l = p.name.lower()
                is_sar = "s1" in name_l or "grd" in name_l or "sar" in name_l
                sensor = "Sentinel-1" if is_sar else "Sentinel-2"

                # Extract spatial bounds
                info = gis_pipeline.get_geotiff_info(str(p))
                wgs84_bounds = info.get("wgs84_bounds")
                if not wgs84_bounds or len(wgs84_bounds) < 4:
                    wgs84_bounds = _extract_bbox_from_filename(p.name) or [77.20, 28.55, 77.36, 28.71]

                c_lon = round((wgs84_bounds[0] + wgs84_bounds[2]) / 2.0, 6)
                c_lat = round((wgs84_bounds[1] + wgs84_bounds[3]) / 2.0, 6)

                cloud_cover = 0.0 if is_sar else 8.5
                thumb_url = _find_thumbnail_url(p)

                # Compute or reuse embedding
                emb = embedding_service.embed_raster_tile(p)
                tile_id = f"tile_{hashlib.md5(p.name.encode('utf-8')).hexdigest()[:10]}"

                tile = ArchiveTile(
                    tile_id=tile_id,
                    filename=p.name,
                    file_path=str(p.resolve()),
                    date_str=date_str,
                    sensor=sensor,
                    bbox=wgs84_bounds,
                    center=[c_lon, c_lat],
                    cloud_cover=cloud_cover,
                    thumbnail_url=thumb_url,
                    embedding=emb,
                )
                tiles_list.append(tile)
                embeddings_list.append(emb)

                files_meta[p.name] = {
                    "mtime": p.stat().st_mtime,
                    "tile_id": tile_id,
                    "date": date_str,
                    "sensor": sensor,
                    "bbox": wgs84_bounds,
                    "thumb": thumb_url,
                }
            except Exception as exc:
                logger.error(f"Error indexing tile {p.name}: {exc}", exc_info=True)

        if not tiles_list:
            logger.warning("No valid scenes indexed in archive.")
            self.tiles = []
            self.embedding_matrix = np.empty((0, embedding_service.EMBEDDING_DIM), dtype=np.float32)
            self._is_indexed = True
            return self.get_summary()

        self.tiles = tiles_list
        self.embedding_matrix = np.vstack(embeddings_list).astype(np.float32)

        # Run Spherical k-Means Clustering & Anomaly Detection
        self._compute_clusters_and_anomalies()

        # Save to disk cache
        try:
            np.save(str(EMBEDDINGS_CACHE_PATH), self.embedding_matrix)
            cache_payload = {
                "built_at": datetime.now().isoformat(),
                "total_tiles": len(self.tiles),
                "files": files_meta,
                "tiles_data": [t.to_dict() for t in self.tiles],
                "clusters": self.clusters,
            }
            with open(INDEX_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f, indent=2)
            logger.info(f"Archive index successfully persisted to {INDEX_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Could not persist archive cache: {e}")

        self._is_indexed = True
        return self.get_summary()

    def _load_from_cache(self, cached_index: Dict[str, Any], cached_matrix: np.ndarray):
        tiles_data = cached_index.get("tiles_data", [])
        self.tiles = []
        self.embedding_matrix = cached_matrix.astype(np.float32)

        for i, td in enumerate(tiles_data):
            emb = self.embedding_matrix[i] if i < self.embedding_matrix.shape[0] else np.zeros(embedding_service.EMBEDDING_DIM)
            tile = ArchiveTile(
                tile_id=td["tile_id"],
                filename=td["filename"],
                file_path=td["file_path"],
                date_str=td["date"],
                sensor=td["sensor"],
                bbox=td["bbox"],
                center=td["center"],
                cloud_cover=td["cloud_cover"],
                thumbnail_url=td.get("thumbnail_url"),
                embedding=emb,
                cluster_id=td.get("cluster_id", -1),
                cluster_label=td.get("cluster_label", "Unassigned"),
                anomaly_score=td.get("anomaly_score", 0.0),
                is_anomaly=td.get("is_anomaly", False),
                anomaly_reason=td.get("anomaly_reason"),
                anomaly_title=td.get("anomaly_title"),
                anomaly_category=td.get("anomaly_category"),
                anomaly_severity=td.get("anomaly_severity"),
                anomaly_confidence=td.get("anomaly_confidence", 0.0),
                archetype_title=td.get("archetype_title"),
            )
            self.tiles.append(tile)

        self.clusters = cached_index.get("clusters", [])
        self._is_indexed = True

    def _compute_clusters_and_anomalies(self, n_clusters: int = DEFAULT_NUM_CLUSTERS):
        """
        Executes Spherical k-Means on the embedding matrix and flags anomaly outliers.
        Enriches clusters with VLM archetypes and explains physical anomalies (Upgrade 3).
        """
        n_samples = len(self.tiles)
        if n_samples == 0:
            self.clusters = []
            return

        if n_samples == 1:
            self.tiles[0].cluster_id = 0
            self.tiles[0].cluster_label = "Local Ingested Scene"
            self.tiles[0].anomaly_score = 0.0
            self.clusters = [{
                "cluster_id": 0,
                "label": "Local Ingested Scene",
                "count": 1,
                "percentage": 100.0,
                "exemplar_thumbnail": self.tiles[0].thumbnail_url,
                "exemplar_date": self.tiles[0].date_str,
                "exemplar_sensor": self.tiles[0].sensor,
            }]
            return

        k = max(2, min(n_clusters, n_samples))
        # Deterministic initialization: select k spaced elements
        rng = np.random.RandomState(42)
        init_indices = rng.choice(n_samples, size=k, replace=False)
        centroids = self.embedding_matrix[init_indices].copy()

        labels = np.zeros(n_samples, dtype=int)
        # Spherical k-means iterations
        for _ in range(15):
            # Compute cosine similarities to centroids: (N x K)
            sims = np.dot(self.embedding_matrix, centroids.T)
            labels = np.argmax(sims, axis=1)

            # Recompute centroids
            for ci in range(k):
                members = self.embedding_matrix[labels == ci]
                if members.shape[0] > 0:
                    mean_vec = np.mean(members, axis=0)
                    norm = float(np.linalg.norm(mean_vec))
                    centroids[ci] = mean_vec / max(norm, 1e-6)

        # Assign cluster metadata & calculate anomaly distances
        cluster_info: List[Dict[str, Any]] = []
        anomaly_scores = np.zeros(n_samples, dtype=np.float32)

        for ci in range(k):
            member_indices = np.where(labels == ci)[0]
            count = int(len(member_indices))
            if count == 0:
                continue

            c_vec = centroids[ci]
            # Determine cluster semantic archetype label from centroid weights
            label = self._label_cluster_centroid(c_vec)

            # Exemplar tile is the one with highest cosine similarity to centroid
            member_sims = np.dot(self.embedding_matrix[member_indices], c_vec)
            best_local_idx = int(np.argmax(member_sims))
            exemplar_tile = self.tiles[member_indices[best_local_idx]]

            # Anomaly distance: 1 - cosine_sim
            for idx in member_indices:
                dist = 1.0 - float(np.dot(self.embedding_matrix[idx], c_vec))
                anomaly_scores[idx] = max(0.0, dist)
                self.tiles[idx].cluster_id = ci
                self.tiles[idx].cluster_label = label
                self.tiles[idx].anomaly_score = dist

            cluster_info.append({
                "cluster_id": ci,
                "label": label,
                "count": count,
                "percentage": round((count / n_samples) * 100.0, 1),
                "exemplar_thumbnail": exemplar_tile.thumbnail_url,
                "exemplar_date": exemplar_tile.date_str,
                "exemplar_sensor": exemplar_tile.sensor,
            })

        # Flag top anomalies (distance >= 85th percentile or distance >= 0.20)
        thresh = float(np.percentile(anomaly_scores, 85)) if len(anomaly_scores) > 3 else 0.25
        cutoff = min(0.25, max(0.18, thresh))
        for i, tile in enumerate(self.tiles):
            if anomaly_scores[i] >= cutoff:
                tile.is_anomaly = True

        # Upgrade 3: Audit clusters & anomalies using Semantic STAC Auditor (Qwen3-VL / heuristics)
        try:
            from services import stac_auditor_service
            cluster_info, self.tiles = stac_auditor_service.audit_clusters_and_anomalies(
                clusters=cluster_info,
                tiles=self.tiles,
            )
            # Map archetype title onto every tile
            archetype_lookup = {c["cluster_id"]: c.get("archetype_title", c["label"]) for c in cluster_info}
            for tile in self.tiles:
                tile.archetype_title = archetype_lookup.get(tile.cluster_id, tile.cluster_label)
        except Exception as e:
            logger.warning(f"Semantic STAC Auditor encountered error: {e}", exc_info=True)

        self.clusters = cluster_info

    def _label_cluster_centroid(self, c_vec: np.ndarray) -> str:
        """Determines semantic archetype label from 512-dim centroid feature components."""
        # Concept slots:
        # 320: water, 321: urban, 322: vegetation, 323: earthwork, 324: barren, 325: industrial, 326: sar
        scores = {
            "Water Bodies & Reservoirs": c_vec[320] + c_vec[0] * 2.0,  # NDWI
            "Dense Urban & Built-Up": c_vec[321] + c_vec[10] * 1.5,     # BSI
            "Vegetation & Agriculture": c_vec[322] + c_vec[5] * 2.0,   # NDVI
            "Earthworks & Construction": c_vec[323] + c_vec[15] * 1.5, # Brightness
            "Barren Land & Arid": c_vec[324],
            "SAR Radar & Industrial Infrastructure": c_vec[326] * 2.0 + c_vec[20] * 1.5,
        }
        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0.05 else "General Surface Terrain"

    def search(
        self,
        query: Optional[str] = None,
        search_mode: str = "text",
        reference_tile_id: Optional[str] = None,
        reference_file_path: Optional[str] = None,
        sensor: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_cloud_cover: Optional[float] = None,
        min_similarity: float = 0.0,
        cluster_id: Optional[int] = None,
        limit: int = 15,
        group_by_location: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes archive-wide semantic retrieval with spatial, temporal, and sensor filters.
        """
        if not self.is_indexed():
            self.scan_and_index()

        if len(self.tiles) == 0:
            return {"total_matches": 0, "results": [], "clusters_detected": []}

        # 1. Obtain query embedding vector and check for generalized complex/relational query
        rel_plan = None
        rel_target_vec = None
        rel_anchor_vec = None
        relational_scores = None
        relational_metas = {}

        if search_mode == "image":
            # Image-to-image similarity search
            target_path = None
            if reference_tile_id:
                for t in self.tiles:
                    if t.tile_id == reference_tile_id:
                        target_path = t.file_path
                        break
            elif reference_file_path:
                target_path = reference_file_path

            if target_path and Path(target_path).exists():
                q_vec = embedding_service.embed_raster_tile(target_path)
            else:
                q_vec = embedding_service.embed_text_query(query or "")
        else:
            # Text-to-image search: Check for open-vocabulary complex / relational query (ISRO PS 26167 §2.2.1)
            if query and query.strip():
                try:
                    from services import complex_query_service
                    rel_plan = complex_query_service.parse_complex_query(query)
                    if rel_plan.is_compound:
                        logger.info(
                            f"[ComplexQueryEngine] Decomposed complex query '{query}' -> "
                            f"Target: '{rel_plan.target_subject}', Op: {rel_plan.operator}, Anchor: '{rel_plan.context_anchor}'"
                        )
                        rel_target_vec = embedding_service.embed_text_query(rel_plan.target_subject)
                        rel_anchor_vec = embedding_service.embed_text_query(rel_plan.context_anchor)
                        relational_scores = np.zeros(len(self.tiles), dtype=np.float32)
                        for idx, tile in enumerate(self.tiles):
                            rel_score, rel_meta = complex_query_service.score_relational_scene(
                                scene_vector=tile.embedding,
                                target_vector=rel_target_vec,
                                anchor_vector=rel_anchor_vec,
                                plan=rel_plan
                            )
                            relational_scores[idx] = rel_score
                            relational_metas[idx] = rel_meta
                except Exception as e:
                    logger.warning(f"Complex query decomposition failed, falling back to atomic query: {e}")
                    rel_plan = None

            q_vec = embedding_service.embed_text_query(query or "")

        # 2. Batch Cosine Similarities: (N,)
        similarities = embedding_service.batch_cosine_similarity(q_vec, self.embedding_matrix)

        # 3. Apply GIS, Temporal, and Sensor Filters
        matched_candidates: List[Dict[str, Any]] = []

        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        for idx, tile in enumerate(self.tiles):
            if relational_scores is not None:
                score = float(relational_scores[idx])
                rel_meta = relational_metas.get(idx)
            else:
                score = float(similarities[idx])
                rel_meta = None

            if score < min_similarity:
                continue

            # Sensor filter
            if sensor and sensor.lower() not in ("all", ""):
                if sensor.lower() not in tile.sensor.lower():
                    continue

            # Temporal filter
            if start_dt or end_dt:
                try:
                    t_dt = datetime.strptime(tile.date_str, "%Y-%m-%d")
                    if start_dt and t_dt < start_dt:
                        continue
                    if end_dt and t_dt > end_dt:
                        continue
                except ValueError:
                    pass

            # Spatial AOI Bbox filter
            if bbox and len(bbox) == 4:
                if not _bboxes_intersect(tile.bbox, bbox):
                    continue

            # Cloud cover filter
            if max_cloud_cover is not None and tile.sensor == "Sentinel-2":
                if tile.cloud_cover > max_cloud_cover:
                    continue

            # Cluster filter
            if cluster_id is not None and tile.cluster_id != cluster_id:
                continue

            # Calculate temporal stack for this tile's location
            stack_count = sum(1 for other in self.tiles if _bboxes_intersect(tile.bbox, other.bbox))

            record = tile.to_dict()
            record["similarity_score"] = round(score, 4)
            record["temporal_stack_count"] = stack_count
            if rel_meta:
                record["relational_match"] = rel_meta
            matched_candidates.append(record)

        # Sort descending by similarity score (or by date if cluster browsing without query)
        if cluster_id is not None and not (query and query.strip()):
            matched_candidates.sort(key=lambda x: x["date"], reverse=True)
        else:
            matched_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Optional location deduplication / grouping (disabled when explicitly browsing a specific cluster)
        if group_by_location and cluster_id is None and len(matched_candidates) > limit:
            # Group into top representations per location
            seen_centers = []
            grouped = []
            for item in matched_candidates:
                c = item["center"]
                is_duplicate = False
                for sc in seen_centers:
                    # within ~2km
                    if abs(c[0] - sc[0]) < 0.03 and abs(c[1] - sc[1]) < 0.03:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    grouped.append(item)
                    seen_centers.append(c)
                if len(grouped) >= limit:
                    break
            results = grouped
        else:
            results = matched_candidates[:limit]

        clusters_seen = sorted(list(set(r["cluster_label"] for r in results)))

        rel_plan_dict = None
        if rel_plan and rel_plan.is_compound:
            rel_plan_dict = {
                "is_compound": True,
                "operator": rel_plan.operator,
                "target_subject": rel_plan.target_subject,
                "context_anchor": rel_plan.context_anchor,
                "connector_phrase": rel_plan.connector_phrase,
                "explanation": rel_plan.explanation,
            }

        return {
            "total_matches": len(matched_candidates),
            "results": results,
            "clusters_detected": clusters_seen,
            "archive_total": len(self.tiles),
            "relational_plan": rel_plan_dict,
        }

    def get_location_temporal_stack(self, bbox: List[float]) -> List[Dict[str, Any]]:
        """
        Retrieves all scenes covering the specified bbox with high spatial overlap,
        strictly preserving spatial co-registration for time-series analysis.
        """
        if not self.is_indexed():
            self.scan_and_index()

        stack = []
        c_target = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
        for t in self.tiles:
            if _are_same_spatial_site(t.bbox, bbox, t.center, c_target):
                stack.append(t.to_dict())

        # If strict check yielded < 2 scenes, allow slightly relaxed overlap (>= 0.15) as fallback
        if len(stack) < 2:
            for t in self.tiles:
                if t.to_dict() not in stack:
                    iou = _calculate_bbox_iou(t.bbox, bbox)
                    if iou >= 0.15:
                        stack.append(t.to_dict())

        # Sort chronologically
        stack.sort(key=lambda item: item["date"])
        return stack

    def get_spatial_sites_for_query(
        self,
        query: str,
        limit: int = 5,
        min_scenes_per_site: int = 2,
        max_scenes_per_site: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Discovers distinct geographic ground sites matching a user query across the archive.
        Clusters tiles into strictly co-registered spatial footprints (IoU >= 0.25 or dist < 4.5km),
        and packages each site with its own isolated chronological timeline stack.
        """
        if not self.is_indexed():
            self.scan_and_index()

        q_lower = query.lower()
        mentioned_years = set(re.findall(r"\b(201[4-9]|202\d|2030)\b", query))

        # 1. Cluster all archive tiles into distinct, strictly co-registered spatial sites
        raw_sites: List[Dict[str, Any]] = []
        for t in self.tiles:
            matched = False
            for s in raw_sites:
                rep = s["rep"]
                if _are_same_spatial_site(t.bbox, rep.bbox, t.center, rep.center):
                    s["tiles"].append(t)
                    matched = True
                    break
            if not matched:
                raw_sites.append({"rep": t, "tiles": [t]})

        # 2. Score and rank sites for the given query
        scored_sites: List[Tuple[float, Dict[str, Any]]] = []

        is_delhi_query = "delhi" in q_lower or "ncr" in q_lower
        is_east_delhi = "east delhi" in q_lower or ("east" in q_lower and "delhi" in q_lower)
        is_west_delhi = "west delhi" in q_lower or ("west" in q_lower and "delhi" in q_lower) or "karol bagh" in q_lower or "rajouri" in q_lower
        is_noida_query = "noida" in q_lower or "greater noida" in q_lower or "yamuna expressway" in q_lower
        is_dubai_query = "dubai" in q_lower or "uae" in q_lower or "jumeirah" in q_lower or "palm" in q_lower
        is_kashmir_query = "kashmir" in q_lower or "srinagar" in q_lower or "pulwama" in q_lower

        # Compute query text embedding or relational site scoring
        rel_plan = None
        rel_target_vec = None
        rel_anchor_vec = None
        try:
            from services import complex_query_service
            rel_plan = complex_query_service.parse_complex_query(query)
            if rel_plan.is_compound:
                rel_target_vec = embedding_service.embed_text_query(rel_plan.target_subject)
                rel_anchor_vec = embedding_service.embed_text_query(rel_plan.context_anchor)
        except Exception:
            rel_plan = None

        q_emb = embedding_service.embed_text_query(query)

        for s in raw_sites:
            tiles = s["tiles"]
            rep = s["rep"]
            c = rep.center
            lon, lat = c[0], c[1]

            # Regional keyword relevance
            relevance_score = 0.0
            if is_east_delhi and (28.58 <= lat <= 28.67 and lon >= 77.29):
                relevance_score += 15.0
            elif is_west_delhi and (lon <= 77.22 and 28.58 <= lat <= 28.72):
                relevance_score += 15.0
            elif is_noida_query and (28.30 <= lat <= 28.50 and 77.45 <= lon <= 77.75):
                relevance_score += 15.0
            elif is_dubai_query and (24.8 <= lat <= 25.5 and 54.8 <= lon <= 55.6):
                relevance_score += 15.0
            elif is_kashmir_query and (33.5 <= lat <= 34.5 and 74.0 <= lon <= 75.5):
                relevance_score += 15.0
            elif is_delhi_query and (28.3 <= lat <= 28.85 and 76.9 <= lon <= 77.7):
                relevance_score += 6.0

            # Semantic similarity across tiles in site (using relational score if compound)
            if rel_plan and rel_plan.is_compound and rel_target_vec is not None and rel_anchor_vec is not None:
                site_sims = [
                    complex_query_service.score_relational_scene(
                        t.embedding, rel_target_vec, rel_anchor_vec, rel_plan
                    )[0] for t in tiles
                ]
            else:
                site_sims = [float(np.dot(t.embedding, q_emb)) for t in tiles]
            max_sim = max(site_sims) if site_sims else 0.0
            relevance_score += max_sim * 3.0

            # Temporal bonus: sites with more distinct dates are much better candidates
            distinct_dates = set(t.date_str for t in tiles)
            relevance_score += min(len(distinct_dates), 10) * 0.7

            # Determine preferred sensor modality (strictly prevent invalid cross-modal pairs in change detection)
            wants_sar = "sar" in q_lower or "radar" in q_lower or "backscatter" in q_lower
            s2_tiles = [t for t in tiles if t.sensor == "Sentinel-2"]
            s1_tiles = [t for t in tiles if t.sensor == "Sentinel-1"]

            if wants_sar and len(s1_tiles) >= min_scenes_per_site:
                modal_tiles = s1_tiles
            elif len(s2_tiles) >= min_scenes_per_site:
                modal_tiles = s2_tiles
            elif len(s1_tiles) >= min_scenes_per_site:
                modal_tiles = s1_tiles
            else:
                modal_tiles = s2_tiles if s2_tiles else tiles

            # Deduplicate tiles by date within chosen modality
            by_date: Dict[str, ArchiveTile] = {}
            for t in modal_tiles:
                d = t.date_str
                if d not in by_date:
                    by_date[d] = t

            sorted_unique_tiles = sorted(by_date.values(), key=lambda t: t.date_str)

            # Skip sites that don't meet minimum scene threshold if others are available
            if len(sorted_unique_tiles) < min_scenes_per_site and not (is_delhi_query and "delhi" in rep.filename.lower()):
                continue

            # Select up to max_scenes_per_site representative scenes
            if len(sorted_unique_tiles) > max_scenes_per_site:
                selected: List[ArchiveTile] = []
                # Include mentioned years
                if mentioned_years:
                    for t in sorted_unique_tiles:
                        if t.date_str[:4] in mentioned_years and t not in selected:
                            selected.append(t)

                # Always include earliest (baseline) and latest
                if sorted_unique_tiles[0] not in selected:
                    selected.insert(0, sorted_unique_tiles[0])
                if sorted_unique_tiles[-1] not in selected:
                    selected.append(sorted_unique_tiles[-1])

                # Sample intermediate evenly if needed
                if len(selected) < max_scenes_per_site:
                    remaining = [t for t in sorted_unique_tiles if t not in selected]
                    needed = max_scenes_per_site - len(selected)
                    if remaining and needed > 0:
                        step = len(remaining) / float(needed + 1)
                        for i in range(1, needed + 1):
                            idx = int(round(i * step))
                            if idx < len(remaining):
                                selected.append(remaining[idx])

                selected.sort(key=lambda t: t.date_str)
                final_tiles = selected[:max_scenes_per_site]
            else:
                final_tiles = sorted_unique_tiles

            # Resolve site label
            site_label = _get_site_geographical_label(c, rep.cluster_label)

            # Assemble scene metadata with thumbnails/web overlays
            site_scenes = []
            for idx, ft in enumerate(final_tiles):
                thumb = ft.thumbnail_url or _find_thumbnail_url(ft.file_path)
                site_scenes.append({
                    "step_index": idx,
                    "filename": ft.filename,
                    "file_path": str(ft.file_path),
                    "image_url": thumb,
                    "imageUrl": thumb,
                    "date": ft.date_str,
                    "acquisition_date": ft.date_str,
                    "sensor": ft.sensor,
                    "wgs84_bounds": ft.bbox,
                })

            site_obj = {
                "site_id": f"site_{rep.center[0]:.3f}_{rep.center[1]:.3f}".replace(".", "_").replace("-", "neg"),
                "name": site_label,
                "label": site_label,
                "center": [round(c[0], 6), round(c[1], 6)],
                "bounds": rep.bbox,
                "scene_count": len(site_scenes),
                "total_archive_scenes": len(tiles),
                "earliest_date": site_scenes[0]["date"] if site_scenes else "",
                "latest_date": site_scenes[-1]["date"] if site_scenes else "",
                "date_range": f"{site_scenes[0]['date'][:4]} - {site_scenes[-1]['date'][:4]}" if site_scenes else "",
                "dominant_type": rep.cluster_label,
                "t1ImageUrl": site_scenes[0]["image_url"] if site_scenes else "",
                "t2ImageUrl": site_scenes[-1]["image_url"] if len(site_scenes) > 1 else "",
                "scenes": site_scenes,
                "paths": [s["file_path"] for s in site_scenes],
            }
            scored_sites.append((relevance_score, site_obj))

        scored_sites.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_sites[:limit]]

    def ingest_custom_folder(
        self,
        folder_path: Union[str, Path],
        scope_as_active: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingests and indexes an arbitrary local directory (e.g. from a USB pen drive or local folder).
        Discovers all GeoTIFFs (*.tif, *.tiff) and images (*.png, *.jpg).
        Generates georeference metadata, preview overlays, and visual embeddings.
        If scope_as_active=True, restricts the active search and analysis strictly to this folder.
        """
        target_dir = Path(folder_path).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            raise ValueError(f"Directory '{folder_path}' does not exist or is not a valid folder.")

        # Find all raster files
        patterns = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
        candidate_files: List[Path] = []
        for pat in patterns:
            candidate_files.extend(list(target_dir.glob(pat)))
            candidate_files.extend(list(target_dir.glob(pat.upper())))

        # Deduplicate & filter out tiny masks
        candidate_files = sorted(list({p.resolve() for p in candidate_files}))
        clean_files = [f for f in candidate_files if "mask" not in f.name.lower() and f.stat().st_size > 5000]

        if not clean_files:
            raise ValueError(f"No valid satellite GeoTIFF or image rasters found in '{folder_path}'.")

        # Save default archive if not already saved
        if not self._default_tiles and self.tiles:
            self._default_tiles = list(self.tiles)
            self._default_embedding_matrix = self.embedding_matrix.copy()
            self._default_clusters = list(self.clusters)

        new_tiles: List[ArchiveTile] = []
        new_embeddings: List[np.ndarray] = []

        for p in clean_files:
            try:
                # 1. Date
                dt = multi_temporal_service.extract_scene_date(p)
                date_str = dt.strftime("%Y-%m-%d")

                # 2. Sensor
                name_l = p.name.lower()
                is_sar = "s1" in name_l or "grd" in name_l or "sar" in name_l or "risat" in name_l
                sensor = "Sentinel-1" if is_sar else "Sentinel-2"

                # 3. Spatial bounds & projection
                info = gis_pipeline.get_geotiff_info(str(p))
                wgs84_bounds = info.get("wgs84_bounds")
                if not wgs84_bounds or len(wgs84_bounds) < 4:
                    wgs84_bounds = _extract_bbox_from_filename(p.name) or [77.20, 28.55, 77.36, 28.71]

                c_lon = round((wgs84_bounds[0] + wgs84_bounds[2]) / 2.0, 6)
                c_lat = round((wgs84_bounds[1] + wgs84_bounds[3]) / 2.0, 6)

                # 4. Preview overlay URL
                thumb_url = _find_thumbnail_url(p)
                if not thumb_url:
                    try:
                        ov = gis_pipeline.build_web_overlay(str(p), output_dir=str(TEMP_UPLOADS_DIR))
                        if ov and ov.get("png_name"):
                            thumb_url = f"/static/{ov['png_name']}"
                    except Exception:
                        thumb_url = None

                # 5. Visual Embedding
                emb = embedding_service.embed_raster_tile(str(p))

                tile_id = f"custom_{p.stem[:24]}_{hashlib.md5(p.name.encode('utf-8')).hexdigest()[:8]}"
                tile = ArchiveTile(
                    tile_id=tile_id,
                    filename=p.name,
                    file_path=str(p),
                    date_str=date_str,
                    sensor=sensor,
                    bbox=wgs84_bounds,
                    center=[c_lon, c_lat],
                    cloud_cover=0.0,
                    thumbnail_url=thumb_url,
                    embedding=emb,
                )
                new_tiles.append(tile)
                new_embeddings.append(emb)
            except Exception as err:
                logger.warning(f"Error indexing custom file {p.name}: {err}")

        if not new_tiles:
            raise ValueError(f"Failed to extract valid geospatial data from files in '{folder_path}'.")

        if scope_as_active:
            self.tiles = new_tiles
            self.embedding_matrix = np.vstack(new_embeddings).astype(np.float32)
            self.active_folder_path = str(target_dir)
            self.active_folder_name = target_dir.name
            self.is_scoped_to_folder = True
            self._compute_clusters_and_anomalies()
        else:
            self.tiles.extend(new_tiles)
            self.embedding_matrix = np.vstack([self.embedding_matrix, *new_embeddings]).astype(np.float32)
            self._compute_clusters_and_anomalies()

        logger.info(
            f"ArchiveIndexer: Ingested {len(new_tiles)} scenes from custom folder '{target_dir.name}' "
            f"(scoped={scope_as_active})"
        )

        return {
            "status": "success",
            "folder_path": str(target_dir),
            "folder_name": target_dir.name,
            "indexed_count": len(new_tiles),
            "is_scoped": self.is_scoped_to_folder,
            "summary": self.get_summary(),
        }

    def reset_to_default_archive(self) -> Dict[str, Any]:
        """Resets the archive view back to the standard catalog."""
        if self._default_tiles:
            self.tiles = list(self._default_tiles)
            if self._default_embedding_matrix is not None:
                self.embedding_matrix = self._default_embedding_matrix.copy()
            self.clusters = list(self._default_clusters)

        self.active_folder_path = None
        self.active_folder_name = None
        self.is_scoped_to_folder = False
        return {
            "status": "success",
            "message": "Reset archive to standard catalog.",
            "summary": self.get_summary(),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "indexed": self._is_indexed,
            "total_tiles": len(self.tiles),
            "clusters": self.clusters,
            "sensor_breakdown": {
                "sentinel_2": sum(1 for t in self.tiles if t.sensor == "Sentinel-2"),
                "sentinel_1": sum(1 for t in self.tiles if t.sensor == "Sentinel-1"),
            },
            "anomalies_count": sum(1 for t in self.tiles if t.is_anomaly),
            "audited_archetypes_count": sum(1 for c in self.clusters if c.get("is_audited")),
            "active_folder_path": self.active_folder_path,
            "active_folder_name": self.active_folder_name,
            "is_scoped_to_folder": self.is_scoped_to_folder,
        }


# Global indexer singleton instance
_indexer_instance: Optional[ArchiveIndexer] = None


def get_archive_indexer() -> ArchiveIndexer:
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = ArchiveIndexer()
        _indexer_instance.scan_and_index()
    return _indexer_instance
