/**
 * Typed API client for Archive-Wide Multimodal Semantic Retrieval & Clustering.
 * PS2 §2.2.1 (Semantic Search) & §2.2.4 (Semantic Representation & Clustering).
 */

import { API_BASE_URL, resolveAssetUrl } from './api';

export interface ArchiveSearchRequest {
  query?: string;
  search_mode?: 'text' | 'image';
  reference_tile_id?: string | null;
  reference_file_path?: string | null;
  sensor?: 'all' | 'sentinel-2' | 'sentinel-1';
  bbox?: [number, number, number, number] | null;
  start_date?: string | null;
  end_date?: string | null;
  max_cloud_cover?: number;
  min_similarity?: number;
  cluster_id?: number | null;
  limit?: number;
  group_by_location?: boolean;
}

export interface RelationalMatchInfo {
  target_similarity: number;
  anchor_similarity: number;
  harmonic_cooccurrence: number;
  spatial_quadrant_distance: number;
  relational_multiplier: number;
  match_status: string;
  is_relational_match: boolean;
  target_subject: string;
  context_anchor: string;
  operator: string;
}

export interface RelationalQueryPlanInfo {
  is_compound: boolean;
  operator: string;
  target_subject: string;
  context_anchor: string;
  connector_phrase: string;
  explanation: string;
}

export interface ArchiveTileResult {
  tile_id: string;
  filename: string;
  file_path: string;
  date: string;
  sensor: string;
  bbox: [number, number, number, number];
  center: [number, number];
  cloud_cover: number;
  thumbnail_url: string | null;
  cluster_id: number;
  cluster_label: string;
  archetype_title?: string | null;
  anomaly_score: number;
  is_anomaly: boolean;
  anomaly_reason?: string | null;
  anomaly_title?: string | null;
  anomaly_category?: string | null;
  anomaly_severity?: string | null;
  anomaly_confidence?: number;
  similarity_score: number;
  temporal_stack_count: number;
  relational_match?: RelationalMatchInfo | null;
}

export interface ArchiveSearchResponse {
  total_matches: number;
  results: ArchiveTileResult[];
  clusters_detected: string[];
  archive_total: number;
  relational_plan?: RelationalQueryPlanInfo | null;
}

export interface ArchiveClusterItem {
  cluster_id: number;
  label: string;
  archetype_title?: string;
  archetype_description?: string;
  dominant_features?: string[];
  is_audited?: boolean;
  count: number;
  percentage: number;
  exemplar_thumbnail: string | null;
  exemplar_date: string;
  exemplar_sensor: string;
}

export interface ArchiveClustersResponse {
  total_tiles: number;
  clusters: ArchiveClusterItem[];
  sensor_breakdown: {
    sentinel_2: number;
    sentinel_1: number;
  };
  anomalies_count: number;
  audited_archetypes_count?: number;
  anomalies: ArchiveTileResult[];
  active_folder_path?: string | null;
  active_folder_name?: string | null;
  is_scoped_to_folder?: boolean;
}

export interface LocationTimelineResponse {
  bbox: [number, number, number, number];
  n_scenes: number;
  timeline_stack: ArchiveTileResult[];
}

/**
 * Searches the archive using semantic text or image-to-image similarity.
 */
export async function searchArchive(
  request: ArchiveSearchRequest
): Promise<ArchiveSearchResponse> {
  const res = await fetch(`${API_BASE_URL}/archive/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    throw new Error(`Archive search failed with HTTP ${res.status}`);
  }

  const data: ArchiveSearchResponse = await res.json();
  // Resolve thumbnail URLs to full backend URLs
  data.results = data.results.map((r) => ({
    ...r,
    thumbnail_url: (r.thumbnail_url ? resolveAssetUrl(r.thumbnail_url) : null) ?? null,
  }));
  return data;
}

/**
 * Retrieves archive cluster summaries, semantic archetypes, and detected anomalies.
 */
export async function fetchArchiveClusters(): Promise<ArchiveClustersResponse> {
  const res = await fetch(`${API_BASE_URL}/archive/clusters`);
  if (!res.ok) {
    throw new Error(`Failed to fetch clusters: HTTP ${res.status}`);
  }

  const data: ArchiveClustersResponse = await res.json();
  data.clusters = data.clusters.map((c) => ({
    ...c,
    exemplar_thumbnail: (c.exemplar_thumbnail
      ? resolveAssetUrl(c.exemplar_thumbnail)
      : null) ?? null,
  }));
  data.anomalies = data.anomalies.map((a) => ({
    ...a,
    thumbnail_url: (a.thumbnail_url ? resolveAssetUrl(a.thumbnail_url) : null) ?? null,
  }));
  return data;
}

/**
 * Re-indexes all GeoTIFFs in the archive and updates vector embeddings.
 */
export async function rebuildArchiveIndex(): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/archive/index/rebuild`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error(`Rebuild index failed: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Retrieves the chronological temporal stack for a given tile's location.
 */
export async function fetchTileTimeline(tileId: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/archive/tile/${encodeURIComponent(tileId)}/timeline`);
  if (!res.ok) {
    throw new Error(`Failed to fetch tile timeline: HTTP ${res.status}`);
  }
  const data = await res.json();
  if (data.timeline_stack) {
    data.timeline_stack = data.timeline_stack.map((t: ArchiveTileResult) => ({
      ...t,
      thumbnail_url: t.thumbnail_url ? resolveAssetUrl(t.thumbnail_url) : null,
    }));
  }
  return data;
}

/**
 * Retrieves all scenes intersecting an arbitrary bounding box, chronologically sorted.
 */
export async function fetchLocationTimeline(
  bbox: [number, number, number, number]
): Promise<LocationTimelineResponse> {
  const res = await fetch(`${API_BASE_URL}/archive/location/timeline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bbox }),
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch location timeline: HTTP ${res.status}`);
  }
  const data: LocationTimelineResponse = await res.json();
  data.timeline_stack = data.timeline_stack.map((t) => ({
    ...t,
    thumbnail_url: (t.thumbnail_url ? resolveAssetUrl(t.thumbnail_url) : null) ?? null,
  }));
  return data;
}

/**
 * Ingests an arbitrary local folder (e.g., from a USB pen drive) into the archive.
 * Works both online and 100% air-gapped offline.
 */
export async function ingestCustomFolder(
  folderPath: string,
  scopeAsActive = true
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/archive/ingest-folder`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      folder_path: folderPath,
      scope_as_active: scopeAsActive,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to ingest folder' }));
    throw new Error(err.detail || `Folder ingestion failed with HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Resets the active archive scope back to the default catalog.
 */
export async function resetArchiveScope(): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/archive/reset-scope`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error(`Reset archive scope failed: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Queries backend network connectivity state.
 */
export async function fetchNetworkStatus(): Promise<{
  is_online: boolean;
  mode: 'online' | 'offline_airgapped';
  forced_offline: boolean;
  timestamp: number;
  description: string;
}> {
  try {
    const res = await fetch(`${API_BASE_URL}/archive/network-status`);
    if (!res.ok) {
      return {
        is_online: false,
        mode: 'offline_airgapped',
        forced_offline: false,
        timestamp: Date.now(),
        description: 'Local offline air-gapped mode active.',
      };
    }
    return await res.json();
  } catch {
    return {
      is_online: false,
      mode: 'offline_airgapped',
      forced_offline: false,
      timestamp: Date.now(),
      description: 'Air-gapped operation active.',
    };
  }
}
