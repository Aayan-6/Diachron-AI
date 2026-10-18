/**
 * Diachron AI — backend client.
 *
 * The backend is a FastAPI app (see backend/SatQuery-master/backend) exposing:
 *
 *   GET  /api/v1/status            → host / GPU / specialist availability
 *   POST /api/v1/satquery          → multipart: query, files[], dataset_name
 *   POST /api/v1/imagery/upload    → multipart: files[], sensor_type
 *   GET  /api/v1/imagery/presets   → demo scenes with georeferenced overlays
 *   /static/**                     → raster previews & generated masks
 *
 * Requests default to same-origin relative paths so Vite's dev proxy
 * (vite.config.ts) forwards them to 127.0.0.1:8000. That keeps the app working
 * when it is opened from another machine on the LAN, which the previous
 * hard-coded http://localhost:8000 base could never do. Set VITE_API_BASE_URL
 * to point at a different host (e.g. a tunnelled GPU box).
 */

import type {
  BandCapabilityContract,
  CandidateAttributionDossier,
  ChangeCandidate,
  ImageOverlayEvidence,
  ProvenanceRecord,
  SceneOverlay,
  TimelineObservation,
} from './types';

const RAW_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api/v1').trim();


/** Normalised API root, never with a trailing slash. */
export const API_BASE_URL = RAW_BASE.replace(/\/+$/, '');

/** Origin serving /static assets. Empty string means "same origin". */
export const ASSET_ORIGIN = (() => {
  if (!/^https?:\/\//i.test(API_BASE_URL)) return '';
  try {
    return new URL(API_BASE_URL).origin;
  } catch {
    return '';
  }
})();

/**
 * The backend hands back absolute `http://localhost:8000/static/...` URLs.
 * Those only resolve on the machine running the backend, so rewrite them onto
 * whatever origin this client is actually talking to.
 */
export function resolveAssetUrl(url?: string | null): string | undefined {
  if (!url) return undefined;
  if (url.startsWith('blob:') || url.startsWith('data:')) return url;

  const staticIndex = url.indexOf('/static/');
  if (staticIndex === -1) {
    if (/^https?:\/\//i.test(url)) return url;
    return ASSET_ORIGIN + (url.startsWith('/') ? url : '/' + url);
  }
  return ASSET_ORIGIN + url.slice(staticIndex);
}

/* -------------------------------------------------------------------------- */
/* Errors                                                                      */
/* -------------------------------------------------------------------------- */

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(message: string, status: number, detail = '') {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }

  /** True when the backend process could not be reached at all. */
  get isOffline(): boolean {
    return this.status === 0;
  }
}

/** Pulls FastAPI's {"detail": ...} out of an error body when present. */
async function readErrorDetail(res: Response): Promise<string> {
  try {
    const text = await res.text();
    if (!text) return res.statusText;
    try {
      const parsed = JSON.parse(text);
      const detail = parsed?.detail ?? parsed?.message;
      if (typeof detail === 'string') return detail;
      if (detail) return JSON.stringify(detail);
    } catch {
      /* body was not JSON — fall through to raw text */
    }
    return text.slice(0, 400);
  } catch {
    return res.statusText;
  }
}

interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

const DEFAULT_TIMEOUT_MS = 20000;

/** fetch + timeout + uniform error mapping. */
async function request<T>(
  path: string,
  init: RequestInit = {},
  options: RequestOptions = {}
): Promise<T> {
  const { signal, timeoutMs = DEFAULT_TIMEOUT_MS } = options;
  const controller = new AbortController();
  const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null;
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort, { once: true });

  try {
    const res = await fetch(API_BASE_URL + path, {
      ...init,
      signal: controller.signal,
      headers: { Accept: 'application/json', ...(init.headers ?? {}) },
    });

    if (!res.ok) {
      throw new ApiError(
        'Request to ' + path + ' failed (HTTP ' + res.status + ')',
        res.status,
        await readErrorDetail(res)
      );
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    // Caller cancelled deliberately — let that propagate untouched.
    if (signal?.aborted) throw err;
    if ((err as Error)?.name === 'AbortError') {
      throw new ApiError(
        'Request to ' + path + ' timed out',
        0,
        'No response within ' + Math.round(timeoutMs / 1000) + 's.'
      );
    }
    const reason = (err as Error)?.message ?? 'Network error';
    throw new ApiError(
      'Cannot reach the Diachron AI backend',
      0,
      reason + ' — is the FastAPI server running and reachable at ' + (ASSET_ORIGIN || 'this origin') + '?'
    );
  } finally {
    if (timer) clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

/* -------------------------------------------------------------------------- */
/* System status                                                               */
/* -------------------------------------------------------------------------- */

export interface SystemStatusResponse {
  service: string;
  status: string;
  execution_mode: string;
  model_engine: {
    model_name?: string;
    model_checkpoint?: string;
    modalities_supported?: string[];
    remote_service_url?: string | null;
    remote_host_status?: string | null;
    remote_vram?: string | null;
    local_cuda_available?: boolean;
    local_device?: string;
    local_vram_gb?: number;
  };
  network: {
    lan_ip: string;
    port: number;
    api_endpoint: string;
  };
  specialists: Record<string, string>;
}

export function fetchSystemStatus(options?: RequestOptions): Promise<SystemStatusResponse> {
  return request<SystemStatusResponse>('/status', { method: 'GET' }, { timeoutMs: 6000, ...options });
}

/** Human-readable name of whatever hardware is actually serving inference. */
export function describeCompute(status: SystemStatusResponse | null): string {
  if (!status) return 'Unknown';
  const engine = status.model_engine ?? {};
  if (engine.remote_service_url) {
    return engine.remote_vram ? 'Remote GPU · ' + engine.remote_vram : 'Remote GPU host';
  }
  if (engine.local_cuda_available && engine.local_device) {
    const modelTag = engine.model_name ? ` · ${engine.model_name}` : '';
    return engine.local_vram_gb
      ? `${engine.local_device}${modelTag} · ${engine.local_vram_gb} GB VRAM`
      : `${engine.local_device}${modelTag}`;
  }
  return engine.local_device || 'CPU (no CUDA device)';
}

/* -------------------------------------------------------------------------- */
/* Query                                                                       */
/* -------------------------------------------------------------------------- */

export interface SatQueryResponse {
  text_answer: string;
  /** GeoJSON features, mask URLs, or raw pixel boxes, depending on the tool. */
  visual_evidence: unknown[];
  execution_trace?: {
    steps?: string[];
    logs?: string[];
    summary?: string;
    [key: string]: unknown;
  };
  trace_log?: string[];
  /** Not in the current schema; tolerated if a service starts emitting it. */
  response?: string;
  confidence?: number;
  analysis_mode?: 'single' | 'bi-temporal' | 'multi-temporal';
  timeline?: TimelineObservation[];
  candidates?: ChangeCandidate[];
  scene_overlay?: SceneOverlay;
  provenance?: ProvenanceRecord;
}


export interface SubmitQueryArgs {
  query: string;
  files?: File[] | null;
  datasetName?: string | null;
  t1Filename?: string | null;
  t2Filename?: string | null;
  datasetId?: string | null;
  beforeFile?: File | null;
  afterFile?: File | null;
  signal?: AbortSignal;
}

/**
 * Inference is the slow path — a cold ChangeFormer pass over a bi-temporal
 * pair runs for minutes, so this gets a long ceiling rather than the default.
 */
const QUERY_TIMEOUT_MS = 10 * 60000;

export async function submitSatQuery(args: SubmitQueryArgs): Promise<SatQueryResponse> {
  const {
    query,
    files,
    datasetName,
    t1Filename,
    t2Filename,
    datasetId,
    beforeFile,
    afterFile,
    signal,
  } = args;
  const form = new FormData();
  form.append('query', query);

  if (beforeFile) form.append('before_file', beforeFile);
  if (afterFile) form.append('after_file', afterFile);

  const fileArray = files ?? [];
  fileArray.forEach((file, idx) => {
    form.append('files', file);
    if (fileArray.length >= 2) {
      if (idx === 0 && !beforeFile) form.append('before_file', file);
      if (idx === 1 && !afterFile) form.append('after_file', file);
    }
  });

  if (datasetName) form.append('dataset_name', datasetName);
  if (t1Filename) form.append('t1_filename', t1Filename);
  if (t2Filename) form.append('t2_filename', t2Filename);
  if (datasetId) form.append('dataset_id', datasetId);

  console.log('[Diachron Client] Dispatching query:', {
    query: query.slice(0, 80),
    filesCount: fileArray.length,
    t1: t1Filename || fileArray[0]?.name || beforeFile?.name,
    t2: t2Filename || fileArray[1]?.name || afterFile?.name,
    datasetName,
    datasetId,
  });

  return request<SatQueryResponse>(
    '/satquery',
    { method: 'POST', body: form },
    { signal, timeoutMs: QUERY_TIMEOUT_MS }
  );
}

/** Pulls trace lines out of whichever field the backend populated. */
export function extractTraceLines(res: SatQueryResponse): string[] {
  const candidates = [res.execution_trace?.steps, res.execution_trace?.logs, res.trace_log];
  for (const candidate of candidates) {
    if (Array.isArray(candidate) && candidate.length > 0) {
      return candidate.filter((line): line is string => typeof line === 'string');
    }
  }
  return [];
}

/**
 * `visual_evidence` is a heterogeneous list. Only GeoJSON entries can go on
 * the map; pixel-space boxes and mask URLs are handled elsewhere.
 */
export function extractGeoJson(res: SatQueryResponse): any | null {
  for (const item of res.visual_evidence ?? []) {
    if (item && typeof item === 'object' && 'type' in (item as Record<string, unknown>)) {
      const kind = (item as Record<string, unknown>).type;
      if (kind === 'FeatureCollection' || kind === 'Feature') return item;
    }
  }
  return null;
}

/**
 * Pull the backend's georeferenced raster overlay out of `visual_evidence`.
 *
 * The change detector renders its mask as a PNG already warped to Web
 * Mercator, so the map can place it by its corner coordinates and have it land
 * on the right ground at every zoom.
 */
export function extractImageOverlay(res: SatQueryResponse): ImageOverlayEvidence | null {
  const items = res.visual_evidence ?? [];
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    if (!item || typeof item !== 'object') continue;
    const candidate = item as Record<string, unknown>;
    if (candidate.type !== 'ImageOverlay') continue;

    const bounds = candidate.wgs84_bounds;
    const url = candidate.url;
    if (typeof url !== 'string' || !Array.isArray(bounds) || bounds.length !== 4) continue;

    return {
      label: typeof candidate.label === 'string' ? candidate.label : 'Overlay',
      url: resolveAssetUrl(url) ?? url,
      bounds: bounds.map(Number) as [number, number, number, number],
      opacity: typeof candidate.opacity === 'number' ? candidate.opacity : 0.7,
    };
  }
  return null;
}

/**
 * Extracts all georeferenced raster overlays from `visual_evidence`, ordered chronologically.
 */
export function extractAllImageOverlays(res: SatQueryResponse): ImageOverlayEvidence[] {
  const items = res.visual_evidence ?? [];
  const overlays: ImageOverlayEvidence[] = [];
  for (const item of items) {
    if (!item || typeof item !== 'object') continue;
    const candidate = item as Record<string, unknown>;
    if (candidate.type !== 'ImageOverlay') continue;

    const bounds = candidate.wgs84_bounds;
    const url = candidate.url;
    if (typeof url !== 'string' || !Array.isArray(bounds) || bounds.length !== 4) continue;

    overlays.push({
      label: typeof candidate.label === 'string' ? candidate.label : 'Overlay',
      url: resolveAssetUrl(url) ?? url,
      bounds: bounds.map(Number) as [number, number, number, number],
      opacity: typeof candidate.opacity === 'number' ? candidate.opacity : 0.7,
    });
  }
  return overlays;
}

/** Number of drawn features in the evidence payload, for the message footer. */
export function countEvidenceFeatures(res: SatQueryResponse): number {
  const geo = extractGeoJson(res);
  if (geo?.type === 'FeatureCollection') return geo.features?.length ?? 0;
  if (geo?.type === 'Feature') return 1;
  return res.visual_evidence?.length ?? 0;
}

/* -------------------------------------------------------------------------- */
/* Imagery                                                                     */
/* -------------------------------------------------------------------------- */

export interface ImageryUploadResponse {
  dataset_id: string;
  name: string;
  sensor: string;
  mode: 'single' | 'bi-temporal' | 'multi-temporal';
  /** False when the raster has no CRS; bounds and center are then null. */
  georeferenced?: boolean;
  wgs84_bounds: [number, number, number, number] | null;
  center: [number, number] | null;
  crs: string;
  resolution: string;
  area_sq_km: number;
  t1_image_url: string;
  t2_image_url?: string | null;
  t1_filename?: string;
  t2_filename?: string | null;
  scenes?: {
    step_index: number;
    filename: string;
    image_url: string;
  }[];
  band_contract?: BandCapabilityContract;
  t1_nir_image_url?: string | null;
  t2_nir_image_url?: string | null;
}


export function uploadImagery(
  files: File[],
  sensorType?: string,
  signal?: AbortSignal
): Promise<ImageryUploadResponse> {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  if (sensorType) form.append('sensor_type', sensorType);

  return request<ImageryUploadResponse>(
    '/imagery/upload',
    { method: 'POST', body: form },
    { signal, timeoutMs: 5 * 60000 }
  );
}

export interface ImageryPreset {
  id: string;
  name: string;
  displayName?: string;
  sensor: string;
  mode: 'single' | 'bi-temporal';
  wgs84_bounds: [number, number, number, number];
  center: [number, number];
  crs: string;
  resolution: string;
  area_sq_km: number;
  t1_image_url: string;
  t2_image_url?: string | null;
  band_contract?: BandCapabilityContract;
  t1_nir_image_url?: string | null;
  t2_nir_image_url?: string | null;
}

/**
 * Demo scenes. The frontend used to hard-code this list; it now comes from the
 * backend so the two can never drift and the preview URLs point at files the
 * server actually has.
 */
export function fetchImageryPresets(signal?: AbortSignal): Promise<ImageryPreset[]> {
  return request<ImageryPreset[]>('/imagery/presets', { method: 'GET' }, { signal, timeoutMs: 8000 });
}

export function fetchImageryCapabilities(
  datasetName: string,
  signal?: AbortSignal
): Promise<BandCapabilityContract> {
  return request<BandCapabilityContract>(
    `/imagery/capabilities?dataset_name=${encodeURIComponent(datasetName)}`,
    { method: 'GET' },
    { signal, timeoutMs: 8000 }
  );
}

export interface FetchAttributionArgs {
  candidateId: number | string;
  candidate: ChangeCandidate;
  scenePaths?: string[];
  datasetName?: string | null;
  datasetId?: string | null;
  t1Filename?: string | null;
  t2Filename?: string | null;
  query?: string | null;
  forceRefresh?: boolean;
}

/**
 * Trajectory Change Captioning & Physical Attribution (Upgrade 1).
 * Invokes Qwen3-VL on GPU to generate an attribution dossier for a candidate.
 */
export function fetchCandidateAttribution(
  args: FetchAttributionArgs,
  signal?: AbortSignal
): Promise<CandidateAttributionDossier> {
  const { candidateId, candidate, scenePaths, datasetName, datasetId, t1Filename, t2Filename, query, forceRefresh } = args;
  return request<CandidateAttributionDossier>(
    `/candidates/${candidateId}/attribution`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        candidate,
        scene_paths: scenePaths,
        dataset_name: datasetName,
        dataset_id: datasetId,
        t1_filename: t1Filename,
        t2_filename: t2Filename,
        query,
        force_refresh: forceRefresh ?? false,
      }),
    },
    { signal, timeoutMs: 60000 }
  );
}
