import type { SensorKind, TraceStage, TraceStatus, TraceStep } from './types';

/**
 * Turns the backend's flat list of log strings into structured steps.
 *
 * The backend (core/agent.py + core/tracer.py) emits free-form lines such as:
 *
 *   step 0: received query '...' with 2 input file(s) [a.tif, b.tif]
 *   step 1.2: agentic controller classified intent as CHANGE_DETECTION (confidence: 96.2%)
 *   step 2: routed to Change Detective & ChangeFormer AI specialist
 *   Change Detective completed successfully
 *
 * The previous parser keyed off `stageTag.includes('1')`, which also matched
 * "step 1.2", "step 10" and any line containing a 1 — so stages were routinely
 * mislabelled. This classifies on content instead.
 */

const STEP_PREFIX = /^\s*step\s+[\d.]+\s*:\s*/i;
const CONFIDENCE = /confidence\s*[:=]\s*([\d.]+)\s*%/i;

interface StageRule {
  stage: TraceStage;
  patterns: RegExp[];
}

// Ordered: the first rule that matches wins, so the more specific phases
// (routing, inference) are tested before the generic preparation phase.
const STAGE_RULES: StageRule[] = [
  { stage: 'INTENT', patterns: [/received query/i, /classified intent/i, /intent/i] },
  { stage: 'ROUTE', patterns: [/routed to/i, /dispatch/i] },
  {
    stage: 'INFER',
    patterns: [
      /inference/i,
      /changeformer/i,
      /change detective/i,
      /archive-wide/i,
      /semantic retrieval/i,
      /clustering/i,
      /extracted geographic bounds/i,
      /detected/i,
    ],
  },
  {
    stage: 'SYNTHESIZE',
    patterns: [/completed/i, /synthes/i, /packag/i, /aggregat/i, /visual evidence/i],
  },
  {
    stage: 'PREPARE',
    patterns: [
      /co-?registration/i,
      /alignment/i,
      /validated/i,
      /pre-?processing/i,
      /\bcrs\b/i,
      /geospatial header/i,
      /temporal pair/i,
    ],
  },
];

const TOOL_RULES: Array<[RegExp, string]> = [
  [/changeformer|change detective/i, 'ChangeFormerV6'],
  [/archive|semantic retrieval|remoteclip|spherical k-means/i, 'Semantic Archive Indexer'],
  [/multi-temporal|trajectory|onset date|breakpoint/i, 'Multi-Temporal Engine'],
  [/gis metadata|coordinate engine|geospatial header|co-?registration|rasterio|gdal/i, 'GIS Pipeline'],
  [/quality|masking|radiometric|scl|conflation/i, 'False-Alarm Mitigator'],
  [/agentic controller|classified intent|received query/i, 'Agentic Controller'],
];

function classifyStage(line: string): TraceStage {
  for (const rule of STAGE_RULES) {
    if (rule.patterns.some((pattern) => pattern.test(line))) return rule.stage;
  }
  return 'PREPARE';
}

function classifyStatus(line: string): TraceStatus {
  if (/\b(failed|error|exception|could not|unable)\b/i.test(line)) return 'error';
  if (/\b(warning|unavailable|notice|lacks|fallback)\b/i.test(line)) return 'warning';
  return 'success';
}

function classifyTool(line: string): string | null {
  for (const [pattern, name] of TOOL_RULES) {
    if (pattern.test(line)) return name;
  }
  return null;
}

/** Trims the `step N:` prefix and clips to a headline-length summary. */
function summarise(line: string): string {
  const body = line.replace(STEP_PREFIX, '').trim();
  if (!body) return 'Pipeline stage';
  const firstClause = body.split(/\s+[[(]/)[0].trim() || body;
  const headline = firstClause.length > 76 ? firstClause.slice(0, 73).trimEnd() + '…' : firstClause;
  return headline.charAt(0).toUpperCase() + headline.slice(1);
}

export function parseTrace(lines: string[], seed = ''): TraceStep[] {
  return lines
    .filter((line) => typeof line === 'string' && line.trim().length > 0)
    .map((raw, index) => {
      const confidenceMatch = raw.match(CONFIDENCE);
      return {
        id: `trace-${seed}-${index}`,
        stage: classifyStage(raw),
        action: summarise(raw),
        tool: classifyTool(raw),
        raw,
        status: classifyStatus(raw),
        confidence: confidenceMatch ? Number(confidenceMatch[1]) / 100 : undefined,
      } satisfies TraceStep;
    });
}

export const STAGE_LABELS: Record<TraceStage, string> = {
  INTENT: 'Intent',
  PREPARE: 'Prepare',
  ROUTE: 'Route',
  INFER: 'Infer',
  SYNTHESIZE: 'Synthesize',
};

export const STAGE_ACCENTS: Record<TraceStage, string> = {
  INTENT: 'text-violet',
  PREPARE: 'text-ink-muted',
  ROUTE: 'text-amber',
  INFER: 'text-accent',
  SYNTHESIZE: 'text-teal',
};

/* -------------------------------------------------------------------------- */
/* Misc helpers                                                                */
/* -------------------------------------------------------------------------- */

export function sensorKind(sensor?: string | null): SensorKind {
  const value = (sensor ?? '').toLowerCase();
  if (!value) return 'unknown';
  if (value.includes('fused') || (value.includes('optical') && value.includes('sar'))) return 'fused';
  if (value.includes('sar') || value.includes('sentinel-1') || value.includes('radar')) return 'sar';
  if (value.includes('optical') || value.includes('sentinel-2')) return 'optical';
  return 'unknown';
}

export function formatDuration(ms?: number): string {
  if (ms === undefined || Number.isNaN(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.round((ms % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatClock(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

/** Sums `area_ha` across GeoJSON features when the backend reports it. */
export function sumAreaHa(geoJson: any): number | undefined {
  const features: any[] =
    geoJson?.type === 'FeatureCollection'
      ? geoJson.features ?? []
      : geoJson?.type === 'Feature'
        ? [geoJson]
        : [];

  let total = 0;
  let found = false;
  for (const feature of features) {
    const value = Number(feature?.properties?.area_ha);
    if (Number.isFinite(value)) {
      total += value;
      found = true;
    }
  }
  return found ? Math.round(total * 100) / 100 : undefined;
}
