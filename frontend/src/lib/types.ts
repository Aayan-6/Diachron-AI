/** Shared domain types for the Diachron AI workspace. */

export type SensorKind = 'optical' | 'sar' | 'fused' | 'unknown';

export type TraceStage = 'INTENT' | 'PREPARE' | 'ROUTE' | 'INFER' | 'SYNTHESIZE';

export type TraceStatus = 'success' | 'warning' | 'error';

/**
 * One line of the backend's execution trace, parsed into something displayable.
 *
 * Everything here is derived from what the backend actually emitted. Fields the
 * backend does not report (per-step latency, cryptographic audit hashes) are
 * deliberately absent rather than fabricated — the previous UI invented both
 * and presented them as verified telemetry.
 */
export interface TraceStep {
  id: string;
  stage: TraceStage;
  /** Short human-readable summary shown in the list. */
  action: string;
  /** The specialist the line refers to, when it names one. */
  tool: string | null;
  /** Original untouched log line. */
  raw: string;
  status: TraceStatus;
  /** Only set when the backend printed a confidence in the line. */
  confidence?: number;
}

export interface BandCapabilityInfo {
  filename?: string;
  bands_count?: number;
  descriptions?: (string | null)[];
  has_nir?: boolean;
  nir_band_index?: number;
  has_red?: boolean;
  red_band_index?: number;
  has_green?: boolean;
  has_blue?: boolean;
  can_false_color_nir?: boolean;
  can_ndvi?: boolean;
}

export interface BandCapabilityContract {
  t1?: BandCapabilityInfo;
  t2?: BandCapabilityInfo | null;
  joint?: {
    common_nir?: boolean;
    common_ndvi?: boolean;
    can_delta_ndvi?: boolean;
    reason?: string;
  };
  capabilities?: {
    t1_false_color_nir?: boolean;
    t2_false_color_nir?: boolean;
    joint_false_color_nir?: boolean;
    joint_delta_ndvi?: boolean;
  };
  indicators?: {
    t1?: string;
    t2?: string;
    comparison_note?: string;
  };
}

export interface TimelineObservation {
  step?: number;
  step_index?: number;
  date: string;
  filename?: string;
  image_url?: string | null;
  imageUrl?: string | null;
  is_baseline?: boolean;
  change_area_ha?: number;
  change_fraction?: number;
  n_polygons?: number;
  overlay_url?: string | null;
  new_breakpoints?: number;
}

export interface CandidateAttributionDossier {
  candidate_id: number;
  summary: string;
  trajectory_narrative: string;
  vlm_analysis?: string;
  attribution: string;
  is_anthropogenic: boolean;
  confidence_score: number;
  physical_drivers: string[];
  triplet_image_url?: string;
  dates?: {
    t0: string;
    t_onset: string;
    tk: string;
  };
  metrics?: {
    area_ha?: number;
    delta_ndvi?: number;
    delta_ndwi?: number;
    delta_brightness?: number;
  };
  latency_ms?: number;
  generated_at?: string;
}

export interface ChangeCandidate {
  id: number;
  change_type: string;
  earliest_observation: string;
  earliest_step: number;
  latest_observation: string;
  area_ha: number;
  confidence: number;
  centroid: [number, number];
  trajectory: { step: number; date: string; detected: boolean }[];
  geometry?: any;
  attribution_dossier?: CandidateAttributionDossier | null;
  query_matched?: boolean;
  semantic_relevance?: number;
  grounding_reason?: string;
  grounding_bbox?: [number, number, number, number] | null;
  cloud_occlusion?: number;
  false_alarm_flag?: 'CLOUD_SHADOW_ARTIFACT' | 'PHENOLOGICAL_VEGETATION' | 'VERIFIED_CHANGE' | 'NONE';
  sar_verification_status?: 'CONFIRMED_VIA_SAR' | 'SUPPRESSED_VIA_SAR' | 'NO_SAR_AVAILABLE' | 'NOT_APPLICABLE' | string;
  sar_verification_badge?: string | null;
  sar_delta_db?: number;
  suppression_reason?: string | null;
  original_rank?: number;
  new_rank?: number;
  rank_delta?: number;
  rank_movement?: 'PROMOTED' | 'DEMOTED' | 'STABLE';
  feedback_boost?: number;
  feedback_reason?: string;
}

export interface SceneItem {
  step?: number;
  step_index?: number;
  filename?: string;
  image_url?: string;
  imageUrl?: string;
  nirImageUrl?: string;
  nir_image_url?: string;
  date?: string;
  acquisition_date?: string;
  id?: string;
  sensor?: string;
  cloud_cover?: number | null;
  wgs84_bounds?: [number, number, number, number];
}

export interface SpatialSite {
  site_id: string;
  name: string;
  label?: string;
  center: [number, number];
  bounds: [number, number, number, number];
  scene_count: number;
  total_archive_scenes?: number;
  date_range: string;
  earliest_date?: string;
  latest_date?: string;
  dominant_type?: string;
  t1ImageUrl?: string;
  t2ImageUrl?: string;
  scenes: SceneItem[];
  timeline?: TimelineObservation[];
}

export interface SceneOverlay {
  datasetId?: string;
  name: string;
  sensor: string;
  mode: 'single' | 'bi-temporal' | 'multi-temporal';
  /** [minLon, minLat, maxLon, maxLat] in WGS84. */
  bounds: [number, number, number, number];
  /** [lon, lat] */
  center: [number, number];
  crs: string;
  resolution: string;
  areaSqKm?: number;
  t1ImageUrl: string;
  t2ImageUrl?: string | null;
  t1Filename?: string;
  t2Filename?: string | null;
  scenes?: SceneItem[];
  bandContract?: BandCapabilityContract;
  t1NirImageUrl?: string | null;
  t2NirImageUrl?: string | null;
  sites?: SpatialSite[];
  activeSiteId?: string;
}

export interface SceneDataset {
  name: string;
  sizeLabel: string;
  sensor: string;
  mode: 'single' | 'bi-temporal' | 'multi-temporal';
  crs: string;
  resolution: string;
  /** Present only for locally uploaded scenes that still need sending. */
  files?: File[];
  /** True once the backend has ingested these bytes and knows the name. */
  syncedWithBackend: boolean;
  areaSqKm?: number;
  /**
   * False when the raster carries no CRS. Such a file cannot be placed on a
   * map at all, and the UI must say so rather than guess a location.
   */
  georeferenced: boolean;
  datasetId?: string;
  t1Filename?: string;
  t2Filename?: string | null;
  scenes?: SceneItem[];
  bandContract?: BandCapabilityContract;
  sites?: SpatialSite[];
  activeSiteId?: string;
}


export interface TemporalSceneMetadata {
  requestedDate?: string;
  acquisitionDate?: string;
  sceneId?: string;
  sensor?: string;
  cloudCover?: number | null;
  bands?: string[];
  fileSizeMb?: number | null;
  filename: string;
  imageUrl: string;
}

export interface AOITemporalState {
  bbox: [number, number, number, number] | null;
  areaKm2: number;
  date1: string;
  date2: string;
  sensor: 'sentinel-2' | 'sentinel-1';
  maxCloudCover?: number;
  t1Scene: TemporalSceneMetadata | null;
  t2Scene: TemporalSceneMetadata | null;
  intervalDays: number;
  bandContract?: BandCapabilityContract | null;
  compatibility?: {
    compatible: boolean;
    same_aoi: boolean;
    same_sensor: boolean;
    interval_days: number;
    warnings: string[];
  } | null;
  isStale: boolean;
}

export type MessageRole = 'user' | 'assistant';

export type MessageStatus = 'pending' | 'complete' | 'error' | 'cancelled';

/** A georeferenced raster the backend rendered for display on the map. */
export interface ImageOverlayEvidence {
  label: string;
  url: string;
  /** [minLon, minLat, maxLon, maxLat] of a Web-Mercator-warped PNG. */
  bounds: [number, number, number, number];
  opacity: number;
}

export interface MessageEvidence {
  geoJson: any | null;
  featureCount: number;
  /** Total hectares across change features, when the payload reports areas. */
  areaHa?: number;
  /** e.g. the change mask, already warped so it registers on the basemap. */
  overlay?: ImageOverlayEvidence | null;
  /** All sequential overlays in a multi-temporal time series. */
  allOverlays?: ImageOverlayEvidence[];
  /** Chronological observations per scene date. */
  timeline?: TimelineObservation[];
  /** Tracked change candidates with earliest appearance and trajectory. */
  candidates?: ChangeCandidate[];
  /** Structured analytical provenance record. */
  provenance?: ProvenanceRecord;
}

export type AnalystVerdict = 'pending' | 'confirmed' | 'rejected' | 'flagged';

export interface AnalystDecision {
  candidate_id: number | string;
  verdict: AnalystVerdict;
  notes?: string;
  reviewed_at?: string;
  analyst_id?: string;
}

export interface ProvenanceRecord {
  pipeline_version?: string;
  analysis_mode?: string;
  sensor_platform?: string;
  scenes_analyzed?: Array<{ filename: string; date?: string; cloud_cover?: number; step_index?: number }>;
  spatial_crs?: string;
  resolution_gsd?: string;
  coregistration?: string;
  radiometric_normalization?: string;
  models_used?: string[];
  quality_screening?: string;
  atmospheric_correction?: string;
  timestamp?: string;
  candidate_count?: number;
  timeline_steps?: number;
  [key: string]: any;
}


export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  status: MessageStatus;
  datasetName?: string;
  sensor?: string;
  /** Assistant turns only. */
  trace?: TraceStep[];
  evidence?: MessageEvidence;
  /** Wall-clock time measured by the client, in ms. */
  elapsedMs?: number;
  /** Populated when status === 'error'. */
  errorDetail?: string;
  /** The user text that produced this turn, so a failed turn can be retried. */
  sourceQuery?: string;
}

export interface Session {
  id: string;
  title: string;
  startedAt: string;
  messages: ChatMessage[];
  datasetName?: string;
}

export type BackendPhase = 'checking' | 'online' | 'offline';
