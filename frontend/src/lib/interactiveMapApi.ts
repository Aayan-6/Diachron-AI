/**
 * Typed API client for Interactive Map STAC + COG Satellite Imagery Streaming.
 */

import { API_BASE_URL, resolveAssetUrl } from './api';
import type { SceneDataset, SceneOverlay } from './types';

export interface FetchAOIRequest {
  bbox: [number, number, number, number]; // [min_lon, min_lat, max_lon, max_lat] in WGS84
  sensor: 'sentinel-2' | 'sentinel-1';
  max_cloud_cover?: number;
  date_range?: string | null;
}

export interface FetchAOIResponse {
  status: 'success' | 'error' | 'not_found';
  dataset_id: string;
  name: string;
  scene_id?: string;
  sensor: string;
  mode: 'single';
  georeferenced: boolean;
  wgs84_bounds: [number, number, number, number];
  center: [number, number];
  crs: string;
  resolution: string;
  area_sq_km: number;
  t1_image_url: string;
  t1_filename: string;
  acquisition_date: string;
  cloud_cover?: number | null;
  file_size_mb?: number | null;
  bands: string[];
  suggested_queries?: string[];
  timings?: Record<string, number>;
}

export interface InteractiveMapHealth {
  status: 'online' | 'degraded' | 'offline';
  module_loaded: boolean;
  error?: string | null;
  stac_endpoint?: string | null;
  max_aoi_cap_sq_km: number;
  supported_sensors: string[];
}

/**
 * Checks connectivity to the backend STAC/COG engine.
 */
export async function checkInteractiveMapHealth(): Promise<InteractiveMapHealth> {
  const res = await fetch(`${API_BASE_URL}/interactive-map/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Streams satellite imagery covering the specified AOI bounding box.
 */
export async function fetchAOISatelliteScene(req: FetchAOIRequest): Promise<FetchAOIResponse> {
  const res = await fetch(`${API_BASE_URL}/interactive-map/fetch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errorDetail;
    } catch {
      // fallback
    }
    throw new Error(errorDetail);
  }

  const data: FetchAOIResponse = await res.json();
  return data;
}

/**
 * Converts a successful STAC fetch response into a SceneDataset for AppState.
 */
export function aoiResponseToSceneDataset(res: FetchAOIResponse): SceneDataset {
  const sizeMb = typeof res.file_size_mb === 'number' && !isNaN(res.file_size_mb) ? res.file_size_mb : null;
  const areaKm = typeof res.area_sq_km === 'number' && !isNaN(res.area_sq_km) ? res.area_sq_km : 0;
  return {
    name: res.name,
    sizeLabel: sizeMb && sizeMb > 0 ? `${sizeMb.toFixed(1)} MB` : `${areaKm} km²`,
    sensor: res.sensor,
    mode: res.mode,
    crs: res.crs,
    resolution: res.resolution,
    syncedWithBackend: true,
    georeferenced: true,
    areaSqKm: areaKm,
  };
}

/**
 * Converts a successful STAC fetch response into a SceneOverlay for MapViewport.
 */
export function aoiResponseToSceneOverlay(res: FetchAOIResponse): SceneOverlay {
  return {
    datasetId: res.dataset_id,
    name: res.name,
    sensor: res.sensor,
    mode: res.mode,
    bounds: res.wgs84_bounds,
    center: res.center,
    crs: res.crs,
    resolution: res.resolution,
    areaSqKm: res.area_sq_km,
    t1ImageUrl: resolveAssetUrl(res.t1_image_url) ?? res.t1_image_url,
    t1Filename: res.t1_filename,
  };
}

export interface FetchBiTemporalAOIRequest {
  bbox: [number, number, number, number];
  sensor: 'sentinel-2' | 'sentinel-1';
  max_cloud_cover?: number;
  date_t1: string;
  date_t2: string;
}

export interface FetchBiTemporalAOIResponse {
  status: 'success' | 'error' | 'not_found';
  dataset_id: string;
  name: string;
  sensor: string;
  mode: 'bi-temporal';
  georeferenced: boolean;
  wgs84_bounds: [number, number, number, number];
  center: [number, number];
  crs: string;
  resolution: string;
  area_sq_km: number;
  interval_days: number;
  t1_image_url: string;
  t2_image_url: string;
  t1_filename: string;
  t2_filename: string;
  t1: {
    requested_date: string;
    acquisition_date: string;
    scene_id?: string;
    cloud_cover?: number | null;
    bands: string[];
    file_size_mb?: number | null;
    filename: string;
    image_url: string;
  };
  t2: {
    requested_date: string;
    acquisition_date: string;
    scene_id?: string;
    cloud_cover?: number | null;
    bands: string[];
    file_size_mb?: number | null;
    filename: string;
    image_url: string;
  };
  band_contract?: any;
  compatibility?: {
    compatible: boolean;
    same_aoi: boolean;
    same_sensor: boolean;
    interval_days: number;
    warnings: string[];
  };
  suggested_queries?: string[];
}

/**
 * Streams a bi-temporal pair of satellite scenes for the identical AOI bounding box across two dates.
 */
export async function fetchBiTemporalAOIScenes(req: FetchBiTemporalAOIRequest): Promise<FetchBiTemporalAOIResponse> {
  const res = await fetch(`${API_BASE_URL}/interactive-map/fetch-bitemporal`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errorDetail;
    } catch {
      // fallback
    }
    throw new Error(errorDetail);
  }

  const data: FetchBiTemporalAOIResponse = await res.json();
  return data;
}

/**
 * Converts a successful bi-temporal STAC fetch response into a SceneDataset for AppState.
 */
export function aoiPairResponseToSceneDataset(res: FetchBiTemporalAOIResponse): SceneDataset {
  const sizeMb = ((res.t1?.file_size_mb || 0) + (res.t2?.file_size_mb || 0));
  const areaKm = typeof res.area_sq_km === 'number' && !isNaN(res.area_sq_km) ? res.area_sq_km : 0;
  return {
    name: res.name,
    sizeLabel: sizeMb > 0 ? `${sizeMb.toFixed(1)} MB` : `${areaKm} km²`,
    sensor: res.sensor,
    mode: 'bi-temporal',
    crs: res.crs,
    resolution: res.resolution,
    syncedWithBackend: true,
    georeferenced: true,
    areaSqKm: areaKm,
    datasetId: res.dataset_id,
    t1Filename: res.t1_filename,
    t2Filename: res.t2_filename,
    bandContract: res.band_contract,
  };
}

/**
 * Converts a successful bi-temporal STAC fetch response into a SceneOverlay for MapViewport.
 */
export function aoiPairResponseToSceneOverlay(res: FetchBiTemporalAOIResponse): SceneOverlay {
  return {
    datasetId: res.dataset_id,
    name: res.name,
    sensor: res.sensor,
    mode: 'bi-temporal',
    bounds: res.wgs84_bounds,
    center: res.center,
    crs: res.crs,
    resolution: res.resolution,
    areaSqKm: res.area_sq_km,
    t1ImageUrl: resolveAssetUrl(res.t1_image_url) ?? res.t1_image_url,
    t2ImageUrl: resolveAssetUrl(res.t2_image_url) ?? res.t2_image_url,
    t1Filename: res.t1_filename,
    t2Filename: res.t2_filename,
    bandContract: res.band_contract,
  };
}

export interface FetchTemporalStackRequest {
  bbox: [number, number, number, number];
  sensor: 'sentinel-2' | 'sentinel-1';
  max_cloud_cover?: number;
  date_range?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  max_scenes?: number;
  min_interval_days?: number;
}

export interface FetchTemporalStackResponse {
  status: 'success' | 'partial' | 'error';
  sensor: string;
  bbox: [number, number, number, number];
  area_sq_km: number;
  n_scenes: number;
  count?: number;
  date_span_days: number;
  scenes: {
    step_index: number;
    scene_id: string;
    acquisition_date: string;
    date?: string;
    cloud_cover?: number | null;
    filename: string;
    image_url: string;
    imageUrl?: string;
    nir_image_url?: string;
    nirImageUrl?: string;
    crs?: string;
    wgs84_bounds?: [number, number, number, number];
  }[];
  suggested_queries?: string[];
}

export async function fetchAOITemporalStack(req: FetchTemporalStackRequest): Promise<FetchTemporalStackResponse> {
  const targetDateRange =
    req.date_range ||
    (req.start_date && req.end_date ? `${req.start_date}T00:00:00Z/${req.end_date}T23:59:59Z` : undefined);

  const payload = {
    ...req,
    date_range: targetDateRange,
  };

  const res = await fetch(`${API_BASE_URL}/interactive-map/fetch-temporal-stack`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errorDetail;
    } catch {
      // fallback
    }
    throw new Error(errorDetail);
  }

  return res.json();
}

export function aoiStackResponseToSceneDataset(res: FetchTemporalStackResponse): SceneDataset {
  const firstScene = res.scenes[0];
  const lastScene = res.scenes[res.scenes.length - 1];
  const compName = res.scenes.map((s) => s.filename).join(':::');
  return {
    name: compName,
    sizeLabel: `${res.scenes.length} scenes · ${res.date_span_days}d`,
    sensor: res.sensor,
    mode: 'multi-temporal',
    crs: firstScene?.crs || 'EPSG:4326 (WGS84)',
    resolution: '10.0m GSD',
    syncedWithBackend: true,
    georeferenced: true,
    areaSqKm: res.area_sq_km,
    datasetId: `stack_${Date.now()}`,
    t1Filename: firstScene?.filename,
    t2Filename: lastScene?.filename,
    scenes: res.scenes.map((s) => ({
      step_index: s.step_index,
      filename: s.filename,
      image_url: resolveAssetUrl(s.image_url) ?? s.image_url,
      acquisition_date: s.acquisition_date,
    })),
  };
}

export function aoiStackResponseToSceneOverlay(res: FetchTemporalStackResponse): SceneOverlay {
  const firstScene = res.scenes[0];
  const lastScene = res.scenes[res.scenes.length - 1];
  const b = firstScene?.wgs84_bounds || res.bbox;
  const center: [number, number] = [
    roundCoord((b[0] + b[2]) / 2),
    roundCoord((b[1] + b[3]) / 2),
  ];
  return {
    datasetId: `stack_${Date.now()}`,
    name: `AOI Multi-Temporal (${res.scenes.length} Scenes)`,
    sensor: res.sensor,
    mode: 'multi-temporal',
    bounds: b,
    center,
    crs: firstScene?.crs || 'EPSG:4326',
    resolution: '10.0m GSD',
    areaSqKm: res.area_sq_km,
    t1ImageUrl: resolveAssetUrl(firstScene?.image_url) ?? (firstScene?.image_url || ''),
    t2ImageUrl: resolveAssetUrl(lastScene?.image_url) ?? (lastScene?.image_url || ''),
    t1Filename: firstScene?.filename,
    t2Filename: lastScene?.filename,
    scenes: res.scenes.map((s) => ({
      step_index: s.step_index,
      filename: s.filename,
      image_url: resolveAssetUrl(s.image_url) ?? s.image_url,
      acquisition_date: s.acquisition_date,
    })),
  };
}

function roundCoord(v: number): number {
  return Math.round(v * 1000000) / 1000000;
}

