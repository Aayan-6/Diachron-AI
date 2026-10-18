import type { ChangeCandidate, ProvenanceRecord, AnalystDecision } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api/v1').trim();

export interface ExportPayloadArgs {
  title?: string;
  query?: string;
  candidates: ChangeCandidate[];
  provenance: ProvenanceRecord | null;
  analystVerdicts: Record<string, AnalystDecision>;
  timeline?: any[];
  geoJson?: any;
}

/**
 * Downloads a file triggered in the browser.
 */
function triggerBrowserDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/**
 * Export GeoJSON FeatureCollection with enriched analyst triage and provenance.
 */
export async function exportEnrichedGeoJson(args: ExportPayloadArgs) {
  const { title = 'Diachron_Intelligence_Export', candidates, provenance, analystVerdicts, geoJson } = args;

  // Extract features from either passed geoJson or candidate geometries
  let features = [];
  if (geoJson && geoJson.features) {
    features = geoJson.features;
  } else if (candidates && candidates.length > 0) {
    features = candidates.map((c) => ({
      type: 'Feature',
      geometry: c.geometry,
      properties: {
        id: c.id,
        change_type: c.change_type,
        area_ha: c.area_ha,
        confidence: c.confidence,
        earliest_observation: c.earliest_observation,
        latest_observation: c.latest_observation,
        centroid: c.centroid,
        query_matched: c.query_matched ?? null,
        semantic_relevance: c.semantic_relevance ?? null,
        grounding_reason: c.grounding_reason ?? null,
        attribution_dossier: c.attribution_dossier || null,
        is_anthropogenic: c.attribution_dossier?.is_anthropogenic ?? null,
        physical_attribution: c.attribution_dossier?.attribution ?? null,
      },
    }));
  }

  try {
    const res = await fetch(`${API_BASE}/export/geojson`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        features,
        provenance,
        analyst_verdicts: analystVerdicts,
      }),
    });

    if (res.ok) {
      const blob = await res.blob();
      triggerBrowserDownload(blob, `${title.toLowerCase().replace(/[^a-z0-9]/g, '_')}_${Date.now()}.geojson`);
      return;
    }
  } catch (err) {
    console.warn('Backend GeoJSON export endpoint unreachable, generating client-side fallback:', err);
  }

  // Client-side fallback if server endpoint is unavailable
  const fallbackPayload = {
    type: 'FeatureCollection',
    name: title,
    metadata: {
      exported_at: new Date().toISOString(),
      standard: 'RFC 7946 GeoJSON / ISRO PS2',
      feature_count: features.length,
      provenance: provenance || { pipeline_version: 'Diachron AI 2.0' },
    },
    features: features.map((f: any, idx: number) => {
      const cId = String(f.properties?.id ?? idx + 1);
      const dec = analystVerdicts[cId];
      return {
        ...f,
        properties: {
          ...f.properties,
          analyst_verdict: dec?.verdict || 'unreviewed',
          analyst_notes: dec?.notes || '',
          reviewed_at: dec?.reviewed_at || null,
        },
      };
    }),
  };

  const blob = new Blob([JSON.stringify(fallbackPayload, null, 2)], { type: 'application/geo+json' });
  triggerBrowserDownload(blob, `${title.toLowerCase().replace(/[^a-z0-9]/g, '_')}_${Date.now()}.geojson`);
}

/**
 * Export machine-readable W3C PROV / ISO 19115 Audit Log.
 */
export async function exportAuditJson(args: ExportPayloadArgs) {
  const { query = '', candidates, provenance, analystVerdicts } = args;

  try {
    const res = await fetch(`${API_BASE}/export/audit-json`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: `session_${Date.now()}`,
        query,
        candidates,
        provenance,
        analyst_verdicts: analystVerdicts,
      }),
    });

    if (res.ok) {
      const blob = await res.blob();
      triggerBrowserDownload(blob, `diachron_audit_trail_${Date.now()}.json`);
      return;
    }
  } catch (err) {
    console.warn('Backend audit JSON export error, using fallback:', err);
  }

  // Client fallback
  const fallback = {
    prov_standard: 'W3C PROV-O / ISO 19115-1:2014',
    system: 'Diachron AI Remote Sensing Decision Support',
    export_timestamp: new Date().toISOString(),
    query,
    provenance,
    analyst_decisions: analystVerdicts,
    candidates,
  };
  const blob = new Blob([JSON.stringify(fallback, null, 2)], { type: 'application/json' });
  triggerBrowserDownload(blob, `diachron_audit_trail_${Date.now()}.json`);
}

/**
 * Opens print-ready HTML Intelligence Dossier in a new browser window.
 */
export async function openIntelligenceDossier(args: ExportPayloadArgs) {
  const { title = 'Earth-Observation Change Analysis Intelligence Brief', query = '', candidates, provenance, analystVerdicts, timeline } = args;

  try {
    const res = await fetch(`${API_BASE}/export/intelligence-dossier`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        query,
        candidates,
        provenance,
        analyst_verdicts: analystVerdicts,
        timeline,
      }),
    });

    if (res.ok) {
      const htmlText = await res.text();
      const printWin = window.open('', '_blank');
      if (printWin) {
        printWin.document.open();
        printWin.document.write(htmlText);
        printWin.document.close();
        return;
      }
    }
  } catch (err) {
    console.warn('Failed to open intelligence dossier via API:', err);
  }
}

/**
 * Persists analyst decisions to server audit storage.
 */
export async function persistAuditDecisions(sessionId: string, query: string, decisions: Record<string, AnalystDecision>, provenance: ProvenanceRecord | null) {
  try {
    const res = await fetch(`${API_BASE}/analyst/audit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        query,
        decisions,
        provenance,
      }),
    });
    return res.ok;
  } catch {
    return false;
  }
}
