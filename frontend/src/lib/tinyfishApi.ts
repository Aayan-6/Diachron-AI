/**
 * Client library for TinyFish Spatio-Temporal Ground-Truth Sentinel
 */

export interface SpatialContext {
  display_name: string;
  village?: string | null;
  town_or_city?: string | null;
  subdistrict?: string | null;
  district?: string | null;
  state?: string | null;
  country: string;
}

export interface TimelineMilestone {
  date: string;
  event: string;
  source: string;
}

export interface CorroboratingSource {
  title: string;
  url: string;
  snippet: string;
  domain: string;
  date: string;
}

export interface AttributionReport {
  causality_score: number;
  classification: string;
  classification_label: string;
  summary: string;
  confidence_grade: 'High' | 'Moderate' | 'Low';
  timeline: TimelineMilestone[];
  corroborating_sources: CorroboratingSource[];
}

export interface InvestigationResponse {
  status: 'success' | 'error';
  spatial_context: SpatialContext;
  satellite_summary: {
    change_type: string;
    area_hectares: number;
    t1_date: string;
    t2_date: string;
    coordinates: { lat: number; lon: number };
  };
  attribution: AttributionReport;
  total_sources_scanned: number;
  investigation_timestamp: string;
}

export interface InvestigationParams {
  lat: number;
  lon: number;
  change_type?: string;
  area_hectares?: number;
  t1_date?: string;
  t2_date?: string;
}

const API_BASE = '/api/v1/tinyfish';

export async function runTinyFishInvestigation(
  params: InvestigationParams
): Promise<InvestigationResponse> {
  const response = await fetch(`${API_BASE}/investigate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`TinyFish investigation failed: ${errorText}`);
  }

  return response.json();
}

export function streamTinyFishInvestigation(
  params: InvestigationParams,
  onStep: (stepText: string) => void,
  onComplete: (data: any) => void,
  onError: (err: any) => void
): () => void {
  const query = new URLSearchParams({
    lat: String(params.lat),
    lon: String(params.lon),
    change_type: params.change_type || 'Vegetation Loss',
    area_hectares: String(params.area_hectares || 15.0),
    t1_date: params.t1_date || '2025-01-01',
    t2_date: params.t2_date || '2025-08-01',
  }).toString();

  const eventSource = new EventSource(`${API_BASE}/stream-investigation?${query}`);

  eventSource.addEventListener('step', (e) => {
    onStep(e.data);
  });

  eventSource.addEventListener('complete', (e) => {
    try {
      const parsed = JSON.parse(e.data);
      onComplete(parsed);
    } catch {
      onComplete(e.data);
    }
    eventSource.close();
  });

  eventSource.onerror = (e) => {
    onError(e);
    eventSource.close();
  };

  return () => {
    eventSource.close();
  };
}
