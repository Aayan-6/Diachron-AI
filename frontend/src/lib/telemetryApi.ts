/**
 * Typed API Client for System Telemetry & ISRO PS 26167 §2.3 Evaluation Deliverables.
 */

import { API_BASE_URL } from './api';

export interface HardwareTelemetry {
  operating_system: string;
  python_version: string;
  cpu_model: string;
  cpu_cores_physical: number;
  cpu_cores_logical: number;
  ram_total_gb: number;
  ram_used_gb: number;
  ram_utilization_pct: number;
  cuda_available: boolean;
  cuda_version: string;
  torch_version: string;
  gpu_name: string;
  gpu_device_count: number;
  vram_total_gb: number;
  vram_allocated_gb: number;
  vram_reserved_gb: number;
  vram_utilization_pct: number;
}

export interface ArchiveCoverageTelemetry {
  total_scenes: number;
  optical_sentinel_2_count: number;
  radar_sentinel_1_count: number;
  clusters_count: number;
  anomalies_flagged: number;
  bounding_box_wgs84: [number, number, number, number];
  bounding_box_extent_area_km2: number;
  cumulative_tile_footprint_km2: number;
  average_tile_area_km2: number;
  primary_focal_area: string;
  spatial_resolution_gsd_m: number;
}

export interface StorageFootprintTelemetry {
  index_metadata_json_kb: number;
  vector_embeddings_matrix_kb: number;
  raster_imagery_cache_mb: number;
  gis_preprocessed_data_mb: number;
  deep_learning_weights_mb: number;
  audit_logs_footprint_kb: number;
  total_system_storage_gb: number;
  offline_sovereignty: string;
}

export interface LatencyBenchmarksTelemetry {
  index_build_cold_seconds: number;
  incremental_ingest_per_scene_seconds: number;
  cache_index_reload_ms: number;
  atomic_vector_retrieval_ms: number;
  complex_relational_query_ms: number;
  optical_quality_screening_ms: number;
  sentinel1_sar_arbitration_ms: number;
  rocchio_active_feedback_rerank_ms: number;
  changeformer_siamese_inference_ms: number;
  qwen3_vl_reasoning_gpu_seconds: number;
  qwen3_vl_reasoning_cpu_seconds: number;
  end_to_end_p95_pipeline_seconds: number;
}

export interface RubricComplianceItem {
  status: 'COMPLIANT' | 'PARTIAL' | 'NON_COMPLIANT';
  solution: string;
}

export interface ProvenanceAndCompliance {
  problem_statement_id: string;
  competition_title: string;
  vision_language_backbone: {
    model_name: string;
    adaptation_method: string;
    lora_rank: number;
    lora_alpha: number;
    target_modules: string[];
    benchmark_dataset: string;
    evaluation_accuracy: string;
  };
  change_detection_engine: {
    architecture: string;
    backbone: string;
    output_resolution: string;
  };
  sensor_lineage: Array<{
    sensor: string;
    modality: string;
    level: string;
    bands_processed?: string[];
    polarizations?: string[];
    role?: string;
  }>;
  rubric_compliance: Record<string, RubricComplianceItem>;
}

export interface SystemTelemetryData {
  timestamp_utc: string;
  system_status: string;
  hardware: HardwareTelemetry;
  archive_coverage: ArchiveCoverageTelemetry;
  storage_footprint: StorageFootprintTelemetry;
  latency_benchmarks: LatencyBenchmarksTelemetry;
  provenance_and_compliance: ProvenanceAndCompliance;
}

export async function fetchSystemTelemetry(): Promise<SystemTelemetryData> {
  const resp = await fetch(`${API_BASE_URL}/system/telemetry`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch system telemetry: ${resp.statusText}`);
  }
  return resp.json();
}

export async function fetchEvaluationReportMarkdown(): Promise<string> {
  const resp = await fetch(`${API_BASE_URL}/system/evaluation-report`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch evaluation report: ${resp.statusText}`);
  }
  return resp.text();
}

export async function triggerLiveBenchmark(): Promise<{
  status: string;
  message: string;
  latency_benchmarks: LatencyBenchmarksTelemetry;
  device?: string;
  vram_gb?: number;
}> {
  const resp = await fetch(`${API_BASE_URL}/system/benchmark`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!resp.ok) {
    throw new Error(`Live benchmark trigger failed: ${resp.statusText}`);
  }
  return resp.json();
}
