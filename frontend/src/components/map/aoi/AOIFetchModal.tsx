import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Satellite,
  Radar,
  CloudSun,
  Loader2,
  X,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Calendar,
  Clock,
  ArrowRight,
  Eye,
  AlertTriangle,
  Check,
  ShieldCheck,
  SplitSquareVertical,
  Activity,
} from 'lucide-react';
import {
  fetchAOISatelliteScene,
  fetchBiTemporalAOIScenes,
  fetchAOITemporalStack,
  type FetchAOIResponse,
  type FetchBiTemporalAOIResponse,
  type FetchTemporalStackResponse,
  aoiPairResponseToSceneDataset,
  aoiPairResponseToSceneOverlay,
  aoiStackResponseToSceneDataset,
  aoiStackResponseToSceneOverlay,
} from '../../../lib/interactiveMapApi';
import { useApp } from '../../../context/AppState';

interface AOIFetchModalProps {
  bbox: [number, number, number, number] | null;
  areaKm2: number;
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (response: FetchAOIResponse) => void;
  onSuccessBiTemporal?: (response: FetchBiTemporalAOIResponse) => void;
  onRunChangeFormer?: (response: FetchBiTemporalAOIResponse) => void;
}

type AnalysisMode = 'single' | 'bitemporal' | 'multitemporal';
type SensorChoice = 'sentinel-2' | 'sentinel-1';

const SINGLE_STREAM_STEPS = [
  'Connecting to AWS STAC Catalog...',
  'Locating closest orbital pass...',
  'Streaming 10m COG pixels via HTTP Range Requests...',
  'Generating Web Mercator warped overlay...',
];

const BITEMPORAL_STREAM_STEPS = [
  'Connecting to AWS STAC Catalog...',
  'Locating closest orbital pass for T1 (Before)...',
  'Locating closest orbital pass for T2 (After)...',
  'Streaming 10m COG pixels for both temporal acquisitions...',
  'Generating Web Mercator warped overlays...',
  'Evaluating spectral band capability contract...',
];

const MULTITEMPORAL_STREAM_STEPS = [
  'Querying STAC catalog across temporal window...',
  'Filtering cloud-free passes with minimum interval spacing...',
  'Streaming 10m COG pixels for time series stack...',
  'Co-registering spatial bounding boxes & warping overlays...',
  'Building chronological multi-temporal stack...',
];

function calculateIntervalDays(d1Str: string, d2Str: string): number {
  if (!d1Str || !d2Str) return 0;
  const d1 = new Date(d1Str);
  const d2 = new Date(d2Str);
  if (isNaN(d1.getTime()) || isNaN(d2.getTime())) return 0;
  const diffTime = d2.getTime() - d1.getTime();
  return Math.max(0, Math.round(diffTime / (1000 * 60 * 60 * 24)));
}

function formatDateDisplay(dateStr?: string): string {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  return isNaN(d.getTime())
    ? dateStr
    : d.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
}

export const AOIFetchModal: React.FC<AOIFetchModalProps> = ({
  bbox,
  areaKm2,
  isOpen,
  onClose,
  onSuccess,
  onSuccessBiTemporal,
  onRunChangeFormer,
}) => {
  const navigate = useNavigate();
  const { setScene, openDock, sendQuery, setAoiTemporal } = useApp();

  const [mode, setMode] = useState<AnalysisMode>('bitemporal');
  const [sensor, setSensor] = useState<SensorChoice>('sentinel-2');
  const [maxCloudCover, setMaxCloudCover] = useState<number>(20);

  // Bi-temporal dates (Default: 1-year interval)
  const [dateT1, setDateT1] = useState<string>('2024-01-15');
  const [dateT2, setDateT2] = useState<string>('2025-01-15');

  // Loading & Execution State
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Fetched bi-temporal result preview
  const [fetchedPair, setFetchedPair] = useState<FetchBiTemporalAOIResponse | null>(null);

  // Multi-temporal stack state
  const [fetchedStack, setFetchedStack] = useState<FetchTemporalStackResponse | null>(null);
  const [stackStartDate, setStackStartDate] = useState<string>('2024-01-01');
  const [stackEndDate, setStackEndDate] = useState<string>('2025-06-30');
  const [stackMaxScenes, setStackMaxScenes] = useState<number>(4);
  const [stackMinIntervalDays, setStackMinIntervalDays] = useState<number>(30);

  const stepIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const prevBboxRef = useRef<[number, number, number, number] | null>(null);

  // Stale data invalidation: if bbox changes, clear fetched results
  useEffect(() => {
    if (bbox) {
      if (
        prevBboxRef.current &&
        (prevBboxRef.current[0] !== bbox[0] ||
          prevBboxRef.current[1] !== bbox[1] ||
          prevBboxRef.current[2] !== bbox[2] ||
          prevBboxRef.current[3] !== bbox[3])
      ) {
        setFetchedPair(null);
        setFetchedStack(null);
        setError(null);
      }
      prevBboxRef.current = bbox;
    }
  }, [bbox]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (stepIntervalRef.current) {
        clearInterval(stepIntervalRef.current);
      }
    };
  }, []);

  if (!isOpen || !bbox) return null;

  const [minLon, minLat, maxLon, maxLat] = bbox;
  const intervalDays = calculateIntervalDays(dateT1, dateT2);

  // Validation
  const hasBothDates = Boolean(dateT1 && dateT2);
  const isDateOrderValid = hasBothDates && new Date(dateT1).getTime() < new Date(dateT2).getTime();
  const isSameDate = hasBothDates && dateT1 === dateT2;

  // Stale invalidation handler when user alters dates
  const handleDateT1Change = (newDate: string) => {
    setDateT1(newDate);
    if (fetchedPair) {
      setFetchedPair(null);
    }
    setError(null);
  };

  const handleDateT2Change = (newDate: string) => {
    setDateT2(newDate);
    if (fetchedPair) {
      setFetchedPair(null);
    }
    setError(null);
  };

  const handleSensorChange = (newSensor: SensorChoice) => {
    setSensor(newSensor);
    if (fetchedPair) {
      setFetchedPair(null);
    }
    setError(null);
  };

  const applyPresetInterval = (gapMonths: number) => {
    const end = new Date(dateT2 || '2025-01-15');
    if (isNaN(end.getTime())) return;
    const start = new Date(end);
    start.setMonth(start.getMonth() - gapMonths);
    const startStr = start.toISOString().split('T')[0];
    handleDateT1Change(startStr);
  };

  // Start single-scene fetch (existing workflow)
  const handleStartSingleFetch = async () => {
    setIsStreaming(true);
    setError(null);
    setCurrentStepIndex(0);

    if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
    stepIntervalRef.current = setInterval(() => {
      setCurrentStepIndex((prev) => (prev < SINGLE_STREAM_STEPS.length - 1 ? prev + 1 : prev));
    }, 2800);

    try {
      const response = await fetchAOISatelliteScene({
        bbox,
        sensor,
        max_cloud_cover: sensor === 'sentinel-2' ? maxCloudCover : undefined,
      });
      if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
      setIsStreaming(false);
      onSuccess(response);
    } catch (err: any) {
      if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
      setIsStreaming(false);
      setError(err?.message || 'Failed to stream satellite scene from STAC archive.');
    }
  };

  // Start bi-temporal fetch (new workflow)
  const handleStartBiTemporalFetch = async () => {
    if (!isDateOrderValid) {
      setError('Invalid date selection: T2 (After) must be strictly later than T1 (Before).');
      return;
    }

    setIsStreaming(true);
    setError(null);
    setFetchedPair(null);
    setCurrentStepIndex(0);

    if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
    stepIntervalRef.current = setInterval(() => {
      setCurrentStepIndex((prev) => (prev < BITEMPORAL_STREAM_STEPS.length - 1 ? prev + 1 : prev));
    }, 2800);

    try {
      const response = await fetchBiTemporalAOIScenes({
        bbox,
        sensor,
        max_cloud_cover: sensor === 'sentinel-2' ? maxCloudCover : undefined,
        date_t1: dateT1,
        date_t2: dateT2,
      });

      if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
      setIsStreaming(false);
      setFetchedPair(response);

      // Record in global AppState
      setAoiTemporal({
        bbox,
        areaKm2,
        date1: dateT1,
        date2: dateT2,
        sensor,
        t1Scene: {
          requestedDate: response.t1.requested_date,
          acquisitionDate: response.t1.acquisition_date,
          sceneId: response.t1.scene_id,
          sensor: response.sensor,
          cloudCover: response.t1.cloud_cover,
          bands: response.t1.bands,
          fileSizeMb: response.t1.file_size_mb,
          filename: response.t1.filename,
          imageUrl: response.t1.image_url,
        },
        t2Scene: {
          requestedDate: response.t2.requested_date,
          acquisitionDate: response.t2.acquisition_date,
          sceneId: response.t2.scene_id,
          sensor: response.sensor,
          cloudCover: response.t2.cloud_cover,
          bands: response.t2.bands,
          fileSizeMb: response.t2.file_size_mb,
          filename: response.t2.filename,
          imageUrl: response.t2.image_url,
        },
        intervalDays: response.interval_days,
        bandContract: response.band_contract,
        compatibility: response.compatibility,
        isStale: false,
      });
    } catch (err: any) {
      if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
      setIsStreaming(false);
      setError(err?.message || 'Failed to retrieve bi-temporal satellite pair for this AOI.');
    }
  };

  // Action: Load pair on MapLibre map for interactive Swipe / Fade comparison
  const handlePreviewOnMap = () => {
    if (!fetchedPair) return;
    const ds = aoiPairResponseToSceneDataset(fetchedPair);
    const ov = aoiPairResponseToSceneOverlay(fetchedPair);
    setScene(ds, ov);
    if (onSuccessBiTemporal) {
      onSuccessBiTemporal(fetchedPair);
    }
    onClose();
  };

  // Action: Detect Changes via ChangeFormer V6 pipeline
  const handleRunChangeFormer = () => {
    if (!fetchedPair) return;
    const ds = aoiPairResponseToSceneDataset(fetchedPair);
    const ov = aoiPairResponseToSceneOverlay(fetchedPair);
    setScene(ds, ov);

    if (onRunChangeFormer) {
      onRunChangeFormer(fetchedPair);
    } else {
      // Direct integration: navigate to conversation workspace with map dock open
      onClose();
      navigate('/console');
      openDock('map');
      const queryText =
        sensor === 'sentinel-1'
          ? `Run ChangeFormer V6 bi-temporal change detection to detect terrain, moisture, and structural changes between T1 (${formatDateDisplay(fetchedPair.t1.acquisition_date)}) and T2 (${formatDateDisplay(fetchedPair.t2.acquisition_date)}).`
          : `Detect and analyze all bi-temporal changes between T1 (${formatDateDisplay(fetchedPair.t1.acquisition_date)}) and T2 (${formatDateDisplay(fetchedPair.t2.acquisition_date)}) in this area of interest using ChangeFormer V6.`;
      sendQuery(queryText);
    }
  };

  // Start multi-temporal stack fetch (ISRO PS2 Workflow)
  const handleStartMultiTemporalFetch = async () => {
    setIsStreaming(true);
    setError(null);
    setFetchedStack(null);
    setCurrentStepIndex(0);

    if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
    stepIntervalRef.current = setInterval(() => {
      setCurrentStepIndex((prev) => (prev < MULTITEMPORAL_STREAM_STEPS.length - 1 ? prev + 1 : prev));
    }, 2800);

    try {
      const response = await fetchAOITemporalStack({
        bbox,
        sensor,
        max_cloud_cover: sensor === 'sentinel-2' ? maxCloudCover : undefined,
        start_date: stackStartDate,
        end_date: stackEndDate,
        max_scenes: stackMaxScenes,
        min_interval_days: stackMinIntervalDays,
      });

      if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
      setIsStreaming(false);
      setFetchedStack(response);
    } catch (err: any) {
      if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
      setIsStreaming(false);
      setError(err?.message || 'Failed to retrieve multi-temporal satellite stack for this AOI.');
    }
  };

  // Action: Load multi-temporal stack on map with timeline scrubber
  const handlePreviewStackOnMap = () => {
    if (!fetchedStack) return;
    const ds = aoiStackResponseToSceneDataset(fetchedStack);
    const ov = aoiStackResponseToSceneOverlay(fetchedStack);
    setScene(ds, ov);
    onClose();
    openDock('map');
  };

  // Action: Run multi-temporal change detection & spectral index tracking
  const handleRunMultiTemporalAnalysis = () => {
    if (!fetchedStack) return;
    const ds = aoiStackResponseToSceneDataset(fetchedStack);
    const ov = aoiStackResponseToSceneOverlay(fetchedStack);
    setScene(ds, ov);
    onClose();
    navigate('/console');
    openDock('map');
    const firstDate = fetchedStack.scenes[0]?.acquisition_date || fetchedStack.scenes[0]?.date || stackStartDate;
    const lastDate = fetchedStack.scenes[fetchedStack.scenes.length - 1]?.acquisition_date || fetchedStack.scenes[fetchedStack.scenes.length - 1]?.date || stackEndDate;
    const sceneCount = fetchedStack.count || fetchedStack.n_scenes || fetchedStack.scenes.length;
    sendQuery(
      `Analyze multi-temporal time series changes across all ${sceneCount} observations from ${firstDate} to ${lastDate}. Track earliest observation dates, change trajectories, and spectral classification.`
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-space-black/65 backdrop-blur-sm animate-in fade-in duration-150 overflow-y-auto">
      <div className="relative w-full max-w-xl rounded-2xl border border-line bg-surface/95 p-5 shadow-2xl backdrop-blur-xl text-ink font-sans my-auto max-h-[92vh] overflow-y-auto scrollbar-slim">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-line pb-3 mb-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-accent/15 text-accent shadow-[0_0_12px_rgba(0,242,255,0.2)]">
              {mode === 'multitemporal' ? (
                <Clock size={18} className="text-cyan-400" />
              ) : mode === 'bitemporal' ? (
                <SplitSquareVertical size={18} />
              ) : (
                <Satellite size={18} />
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold tracking-tight text-ink">
                {mode === 'multitemporal'
                  ? 'Multi-Temporal Time Series (ISRO PS2)'
                  : mode === 'bitemporal'
                  ? 'Bi-Temporal Satellite Analysis'
                  : 'Stream Satellite Imagery'}
              </h3>
              <p className="text-[11px] text-ink-muted">
                {mode === 'multitemporal'
                  ? 'Chronological N-Pass Stack • Earliest Observation • Spectral Trajectory'
                  : mode === 'bitemporal'
                  ? 'Spatial AOI Constraint • Two Dates • ChangeFormer V6'
                  : 'AWS Open Data STAC • 10m Cloud-Optimized GeoTIFF'}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isStreaming}
            className="rounded p-1 text-ink-faint hover:text-ink transition-colors cursor-pointer disabled:opacity-40"
          >
            <X size={16} />
          </button>
        </div>

        {/* Workflow Mode Tabs: Single Scene vs Bi-Temporal vs Multi-Temporal */}
        <div className="mb-4 grid grid-cols-3 gap-1.5 rounded-xl bg-surface-3 p-1 border border-line">
          <button
            type="button"
            disabled={isStreaming}
            onClick={() => {
              setMode('single');
              setError(null);
            }}
            className={`flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition-all cursor-pointer ${
              mode === 'single'
                ? 'bg-surface text-ink shadow-sm border border-line'
                : 'text-ink-muted hover:text-ink'
            }`}
          >
            <Satellite size={13} className={mode === 'single' ? 'text-accent' : ''} />
            <span className="truncate">Single Scene</span>
          </button>

          <button
            type="button"
            disabled={isStreaming}
            onClick={() => {
              setMode('bitemporal');
              setError(null);
            }}
            className={`flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-semibold transition-all cursor-pointer ${
              mode === 'bitemporal'
                ? 'bg-accent/15 text-accent shadow-sm border border-accent/30'
                : 'text-ink-muted hover:text-ink'
            }`}
          >
            <SplitSquareVertical size={13} className={mode === 'bitemporal' ? 'text-accent' : ''} />
            <span className="truncate">Bi-Temporal</span>
          </button>

          <button
            type="button"
            disabled={isStreaming}
            onClick={() => {
              setMode('multitemporal');
              setError(null);
            }}
            className={`flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-semibold transition-all cursor-pointer ${
              mode === 'multitemporal'
                ? 'bg-cyan-500/20 text-cyan-400 shadow-sm border border-cyan-500/40'
                : 'text-ink-muted hover:text-ink'
            }`}
          >
            <Clock size={13} className={mode === 'multitemporal' ? 'text-cyan-400' : ''} />
            <span className="truncate">Multi-Temporal</span>
          </button>
        </div>

        {/* Spatial AOI Constraint Banner (Reused for BOTH T1 and T2) */}
        <div className="mb-4 rounded-xl border border-line bg-surface-2 p-3 text-[11px] font-mono">
          <div className="flex items-center justify-between text-ink-muted mb-1">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-accent" />
              SHARED SPATIAL AOI CONSTRAINT
            </span>
            <span className="text-accent font-bold">{areaKm2} km²</span>
          </div>
          <div className="text-ink-faint text-[10px] truncate">
            BBox: [{minLon}, {minLat}, {maxLon}, {maxLat}]
          </div>
          {mode === 'bitemporal' && (
            <div className="mt-1.5 pt-1.5 border-t border-line/60 text-[10px] text-ink-muted flex items-center gap-1">
              <ShieldCheck size={12} className="text-accent shrink-0" />
              <span>Exact same spatial bounding box enforces identical bounds for T1 and T2 crops.</span>
            </div>
          )}
        </div>

        {/* Sensor Selection Toggle */}
        <div className="mb-4">
          <label className="block text-[11px] font-medium text-ink-muted mb-1.5 uppercase tracking-wider font-mono">
            1. SENSOR TYPE
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              disabled={isStreaming}
              onClick={() => handleSensorChange('sentinel-2')}
              className={`flex flex-col items-start gap-1 p-2.5 rounded-xl border text-left transition-all cursor-pointer ${
                sensor === 'sentinel-2'
                  ? 'border-accent bg-accent/10 shadow-[0_0_12px_rgba(0,242,255,0.15)]'
                  : 'border-line bg-surface-2 hover:border-line-strong'
              }`}
            >
              <div className="flex items-center gap-1.5 text-xs font-semibold text-ink">
                <Satellite size={14} className={sensor === 'sentinel-2' ? 'text-accent' : 'text-ink-muted'} />
                <span>Sentinel-2</span>
              </div>
              <span className="text-[10px] text-ink-muted leading-tight">
                Optical • 10m True Color & NIR
              </span>
            </button>

            <button
              type="button"
              disabled={isStreaming}
              onClick={() => handleSensorChange('sentinel-1')}
              className={`flex flex-col items-start gap-1 p-2.5 rounded-xl border text-left transition-all cursor-pointer ${
                sensor === 'sentinel-1'
                  ? 'border-accent-warm bg-accent-warm/10 shadow-[0_0_12px_rgba(255,170,0,0.15)]'
                  : 'border-line bg-surface-2 hover:border-line-strong'
              }`}
            >
              <div className="flex items-center gap-1.5 text-xs font-semibold text-ink">
                <Radar size={14} className={sensor === 'sentinel-1' ? 'text-accent-warm' : 'text-ink-muted'} />
                <span>Sentinel-1</span>
              </div>
              <span className="text-[10px] text-ink-muted leading-tight">
                SAR Radar • All-Weather C-Band
              </span>
            </button>
          </div>
        </div>

        {/* Optical Cloud Cover Slider (Optical only) */}
        {sensor === 'sentinel-2' && (
          <div className="mb-4 rounded-xl border border-line bg-surface-2 p-3">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-[11px] text-ink-muted flex items-center gap-1.5">
                <CloudSun size={13} className="text-accent" />
                Max Cloud Coverage
              </span>
              <span className="font-mono text-xs font-bold text-ink">{maxCloudCover}%</span>
            </div>
            <input
              type="range"
              min={1}
              max={60}
              value={maxCloudCover}
              disabled={isStreaming}
              onChange={(e) => {
                setMaxCloudCover(Number(e.target.value));
                if (fetchedPair) setFetchedPair(null);
              }}
              className="w-full h-1.5 bg-surface-3 rounded-lg appearance-none cursor-pointer accent-accent"
            />
            <div className="flex justify-between text-[9px] font-mono text-ink-faint mt-1">
              <span>Clear Sky (1%)</span>
              <span>Cloud Tolerant (60%)</span>
            </div>
          </div>
        )}

        {/* BI-TEMPORAL DATE SELECTION UI */}
        {mode === 'bitemporal' && (
          <div className="mb-4 rounded-xl border border-line bg-surface-2 p-3.5 space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-[11px] font-medium text-ink-muted uppercase tracking-wider font-mono flex items-center gap-1.5">
                <Calendar size={12} className="text-accent" />
                2. SELECT TEMPORAL DATES (T1 & T2)
              </label>

              {/* Quick Presets */}
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => applyPresetInterval(6)}
                  className="rounded px-1.5 py-0.5 text-[10px] font-mono bg-surface-3 border border-line text-ink-muted hover:text-accent hover:border-accent/40 cursor-pointer"
                >
                  6 Mos
                </button>
                <button
                  type="button"
                  onClick={() => applyPresetInterval(12)}
                  className="rounded px-1.5 py-0.5 text-[10px] font-mono bg-surface-3 border border-line text-ink-muted hover:text-accent hover:border-accent/40 cursor-pointer"
                >
                  1 Year
                </button>
                <button
                  type="button"
                  onClick={() => applyPresetInterval(24)}
                  className="rounded px-1.5 py-0.5 text-[10px] font-mono bg-surface-3 border border-line text-ink-muted hover:text-accent hover:border-accent/40 cursor-pointer"
                >
                  2 Years
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {/* T1 Date Picker */}
              <div>
                <span className="block text-[10px] font-mono text-ink-faint mb-1">
                  T1 (BEFORE / EARLIER)
                </span>
                <div className="relative">
                  <input
                    type="date"
                    value={dateT1}
                    disabled={isStreaming}
                    onChange={(e) => handleDateT1Change(e.target.value)}
                    className="w-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs font-mono text-ink focus:border-accent focus:outline-none"
                  />
                </div>
              </div>

              {/* T2 Date Picker */}
              <div>
                <span className="block text-[10px] font-mono text-ink-faint mb-1">
                  T2 (AFTER / LATER)
                </span>
                <div className="relative">
                  <input
                    type="date"
                    value={dateT2}
                    disabled={isStreaming}
                    onChange={(e) => handleDateT2Change(e.target.value)}
                    className="w-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs font-mono text-ink focus:border-accent focus:outline-none"
                  />
                </div>
              </div>
            </div>

            {/* Timeline & Interval Indicator */}
            <div className="rounded-lg bg-surface-3/80 p-2 border border-line/60 flex items-center justify-between text-[11px] font-mono">
              <div className="flex items-center gap-2">
                <Clock size={12} className="text-accent shrink-0" />
                <span className="text-ink-muted">Time interval:</span>
                <span className={`font-bold ${isDateOrderValid ? 'text-accent' : 'text-danger'}`}>
                  {isDateOrderValid ? `${intervalDays} days` : 'Invalid date order'}
                </span>
              </div>

              {isDateOrderValid && (
                <span className="text-[10px] text-ink-faint hidden sm:inline">
                  {formatDateDisplay(dateT1)} → {formatDateDisplay(dateT2)}
                </span>
              )}
            </div>

            {/* Validation alert if ordering is wrong */}
            {isSameDate && (
              <div className="flex items-center gap-1.5 text-[11px] text-danger font-mono bg-danger/10 p-2 rounded-lg border border-danger/30">
                <AlertTriangle size={13} />
                <span>T1 and T2 cannot be the same date. Please choose two distinct dates.</span>
              </div>
            )}
            {hasBothDates && !isDateOrderValid && !isSameDate && (
              <div className="flex items-center gap-1.5 text-[11px] text-danger font-mono bg-danger/10 p-2 rounded-lg border border-danger/30">
                <AlertTriangle size={13} />
                <span>The second date (T2) must be strictly later than the first date (T1).</span>
              </div>
            )}
          </div>
        )}

        {/* MULTI-TEMPORAL STACK PARAMETERS (ISRO PS2) */}
        {mode === 'multitemporal' && (
          <div className="mb-4 rounded-xl border border-cyan-500/30 bg-cyan-950/20 p-3.5 space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-[11px] font-semibold text-cyan-400 uppercase tracking-wider font-mono flex items-center gap-1.5">
                <Clock size={12} className="text-cyan-400" />
                2. TEMPORAL STACK PARAMETERS (ISRO PS2)
              </label>
              <span className="text-[10px] font-mono text-cyan-400/80 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                N-Observation Time Series
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <span className="block text-[10px] font-mono text-ink-faint mb-1">
                  START DATE (BASELINE)
                </span>
                <input
                  type="date"
                  value={stackStartDate}
                  disabled={isStreaming}
                  onChange={(e) => {
                    setStackStartDate(e.target.value);
                    setFetchedStack(null);
                    setError(null);
                  }}
                  className="w-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs font-mono text-ink focus:border-accent focus:outline-none"
                />
              </div>

              <div>
                <span className="block text-[10px] font-mono text-ink-faint mb-1">
                  END DATE (LATEST)
                </span>
                <input
                  type="date"
                  value={stackEndDate}
                  disabled={isStreaming}
                  onChange={(e) => {
                    setStackEndDate(e.target.value);
                    setFetchedStack(null);
                    setError(null);
                  }}
                  className="w-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs font-mono text-ink focus:border-accent focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 pt-1">
              <div>
                <div className="flex items-center justify-between text-[10px] font-mono text-ink-faint mb-1">
                  <span>MAX SCENES IN STACK</span>
                  <span className="text-cyan-400 font-bold">{stackMaxScenes} scenes</span>
                </div>
                <input
                  type="range"
                  min={2}
                  max={8}
                  step={1}
                  value={stackMaxScenes}
                  disabled={isStreaming}
                  onChange={(e) => {
                    setStackMaxScenes(parseInt(e.target.value, 10));
                    setFetchedStack(null);
                  }}
                  className="w-full accent-cyan-400"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-[10px] font-mono text-ink-faint mb-1">
                  <span>MIN INTERVAL BETWEEN PASSES</span>
                  <span className="text-cyan-400 font-bold">{stackMinIntervalDays} days</span>
                </div>
                <input
                  type="range"
                  min={10}
                  max={90}
                  step={5}
                  value={stackMinIntervalDays}
                  disabled={isStreaming}
                  onChange={(e) => {
                    setStackMinIntervalDays(parseInt(e.target.value, 10));
                    setFetchedStack(null);
                  }}
                  className="w-full accent-cyan-400"
                />
              </div>
            </div>
          </div>
        )}

        {/* Error Banner */}
        {error && (
          <div className="mb-4 flex items-start gap-2 rounded-xl border border-danger/40 bg-danger/10 p-3 text-xs text-danger">
            <AlertCircle size={14} className="shrink-0 mt-0.5" />
            <div className="flex-1 text-[11px] leading-relaxed">{error}</div>
          </div>
        )}

        {/* Multi-step Streaming Progress Indicator */}
        {isStreaming && (
          <div className="mb-4 rounded-xl border border-accent/30 bg-accent/5 p-3 font-mono text-xs">
            <div className="flex items-center gap-2 text-accent font-semibold mb-2">
              <Loader2 size={13} className="animate-spin" />
              <span>
                {mode === 'multitemporal'
                  ? 'Retrieving Multi-Temporal Satellite Stack...'
                  : mode === 'bitemporal'
                  ? 'Retrieving Bi-Temporal Satellite Pair...'
                  : 'Streaming COG Pixels...'}
              </span>
            </div>
            <div className="space-y-1.5 text-[10px]">
              {(mode === 'multitemporal'
                ? MULTITEMPORAL_STREAM_STEPS
                : mode === 'bitemporal'
                ? BITEMPORAL_STREAM_STEPS
                : SINGLE_STREAM_STEPS
              ).map((step, idx) => {
                  const isPassed = idx < currentStepIndex;
                  const isCurrent = idx === currentStepIndex;
                  return (
                    <div
                      key={step}
                      className={`flex items-center gap-2 ${
                        isPassed
                          ? 'text-accent'
                          : isCurrent
                          ? 'text-white font-bold animate-pulse'
                          : 'text-ink-faint'
                      }`}
                    >
                      {isPassed ? (
                        <CheckCircle2 size={11} className="text-accent" />
                      ) : (
                        <div className="h-1.5 w-1.5 rounded-full bg-current" />
                      )}
                      <span>{step}</span>
                    </div>
                  );
                }
              )}
            </div>
          </div>
        )}

        {/* T1 / T2 SIDE-BY-SIDE PREVIEW CARDS (Once fetched in Bi-Temporal mode) */}
        {mode === 'bitemporal' && fetchedPair && !isStreaming && (
          <div className="mb-4 rounded-xl border border-accent/40 bg-surface-2 p-3.5 space-y-3 animate-in fade-in zoom-in-98 duration-200">
            <div className="flex items-center justify-between border-b border-line pb-2">
              <div className="flex items-center gap-1.5">
                <CheckCircle2 size={14} className="text-accent" />
                <span className="text-xs font-mono font-semibold text-accent uppercase">
                  Bi-Temporal Pair Ready
                </span>
              </div>
              <span className="text-[10px] font-mono text-ink-muted">
                {fetchedPair.interval_days} Days Elapsed
              </span>
            </div>

            {/* Side-by-Side Cards */}
            <div className="grid grid-cols-2 gap-3">
              {/* T1 Card */}
              <div className="rounded-xl border border-line bg-surface p-2.5 space-y-1.5">
                <div className="flex items-center justify-between text-[10px] font-mono text-ink-muted">
                  <span className="font-bold text-accent">T1 (BEFORE)</span>
                  <span>{fetchedPair.t1.cloud_cover !== null ? `${fetchedPair.t1.cloud_cover?.toFixed(1)}% cloud` : 'SAR'}</span>
                </div>

                <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-line bg-surface-3">
                  <img
                    src={fetchedPair.t1_image_url}
                    alt="T1 Preview"
                    className="h-full w-full object-cover"
                  />
                  <div className="absolute bottom-1 left-1 rounded bg-space-black/75 px-1 py-0.5 text-[9px] font-mono text-accent">
                    {formatDateDisplay(fetchedPair.t1.acquisition_date)}
                  </div>
                </div>

                <div className="text-[10px] font-mono space-y-0.5 text-ink-faint">
                  <div className="truncate" title={fetchedPair.t1_filename}>
                    File: {fetchedPair.t1_filename}
                  </div>
                  <div>Req: {fetchedPair.t1.requested_date}</div>
                  <div>Bands: {fetchedPair.t1.bands?.slice(0, 4).join(', ') || 'RGB'}</div>
                </div>
              </div>

              {/* T2 Card */}
              <div className="rounded-xl border border-line bg-surface p-2.5 space-y-1.5">
                <div className="flex items-center justify-between text-[10px] font-mono text-ink-muted">
                  <span className="font-bold text-accent-warm">T2 (AFTER)</span>
                  <span>{fetchedPair.t2.cloud_cover !== null ? `${fetchedPair.t2.cloud_cover?.toFixed(1)}% cloud` : 'SAR'}</span>
                </div>

                <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-line bg-surface-3">
                  <img
                    src={fetchedPair.t2_image_url}
                    alt="T2 Preview"
                    className="h-full w-full object-cover"
                  />
                  <div className="absolute bottom-1 left-1 rounded bg-space-black/75 px-1 py-0.5 text-[9px] font-mono text-accent-warm">
                    {formatDateDisplay(fetchedPair.t2.acquisition_date)}
                  </div>
                </div>

                <div className="text-[10px] font-mono space-y-0.5 text-ink-faint">
                  <div className="truncate" title={fetchedPair.t2_filename}>
                    File: {fetchedPair.t2_filename}
                  </div>
                  <div>Req: {fetchedPair.t2.requested_date}</div>
                  <div>Bands: {fetchedPair.t2.bands?.slice(0, 4).join(', ') || 'RGB'}</div>
                </div>
              </div>
            </div>

            {/* Compatibility Summary Badges */}
            <div className="rounded-lg bg-surface p-2 border border-line text-[10px] font-mono space-y-1">
              <div className="flex flex-wrap gap-2 text-ink-muted">
                <span className="inline-flex items-center gap-1 text-accent">
                  <Check size={11} /> Same Spatial AOI
                </span>
                <span className="inline-flex items-center gap-1 text-accent">
                  <Check size={11} /> Sensor: {fetchedPair.sensor.split(' ')[0]}
                </span>
                <span className="inline-flex items-center gap-1 text-accent">
                  <Check size={11} /> Web Mercator Warped
                </span>
              </div>

              {fetchedPair.compatibility?.warnings && fetchedPair.compatibility.warnings.length > 0 && (
                <div className="text-amber pt-1 border-t border-line/60">
                  {fetchedPair.compatibility.warnings.map((w, idx) => (
                    <div key={idx} className="flex items-start gap-1">
                      <AlertTriangle size={11} className="shrink-0 mt-0.5" />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* MULTI-TEMPORAL STACK PREVIEW CARDS (ISRO PS2) */}
        {mode === 'multitemporal' && fetchedStack && !isStreaming && (
          <div className="mb-4 rounded-xl border border-cyan-500/40 bg-surface-2 p-3.5 space-y-3 animate-in fade-in zoom-in-98 duration-200">
            <div className="flex items-center justify-between border-b border-line pb-2">
              <div className="flex items-center gap-1.5">
                <CheckCircle2 size={14} className="text-cyan-400" />
                <span className="text-xs font-mono font-semibold text-cyan-400 uppercase">
                  Multi-Temporal Stack Ready ({fetchedStack.count || fetchedStack.n_scenes || fetchedStack.scenes.length} Scenes)
                </span>
              </div>
              <span className="text-[10px] font-mono text-ink-muted">
                {fetchedStack.scenes[0]?.acquisition_date || fetchedStack.scenes[0]?.date} ➔ {fetchedStack.scenes[fetchedStack.scenes.length - 1]?.acquisition_date || fetchedStack.scenes[fetchedStack.scenes.length - 1]?.date}
              </span>
            </div>

            {/* Scenes Horizontal Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {fetchedStack.scenes.map((scene, idx) => (
                <div key={idx} className="rounded-lg border border-line bg-surface p-1.5 space-y-1">
                  <div className="flex items-center justify-between text-[9px] font-mono text-ink-muted">
                    <span className="font-bold text-cyan-400">T{idx + 1} {idx === 0 ? '(Base)' : ''}</span>
                    <span>{scene.cloud_cover !== null ? `${scene.cloud_cover?.toFixed(0)}%` : 'SAR'}</span>
                  </div>
                  <div className="relative aspect-video w-full overflow-hidden rounded border border-line bg-surface-3">
                    <img
                      src={scene.image_url}
                      alt={`Observation ${idx + 1}`}
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <div className="text-[9px] font-mono text-ink-muted truncate" title={scene.acquisition_date || scene.date || scene.filename}>
                    {scene.acquisition_date || scene.date || scene.filename}
                  </div>
                </div>
              ))}
            </div>

            {/* Stack Info Summary */}
            <div className="rounded-lg bg-surface p-2 border border-line text-[10px] font-mono flex items-center justify-between text-ink-muted">
              <span>Sensor: {fetchedStack.sensor}</span>
              <span className="text-cyan-400 font-semibold">{fetchedStack.area_sq_km} km² AOI</span>
              <span>10m Spatial Resolution</span>
            </div>
          </div>
        )}

        {/* Modal Action Buttons */}
        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-line">
          <button
            type="button"
            disabled={isStreaming}
            onClick={onClose}
            className="rounded-lg border border-line px-3 py-1.5 text-xs text-ink-muted hover:text-ink transition-colors cursor-pointer disabled:opacity-40"
          >
            Cancel
          </button>

          <div className="flex items-center gap-2">
            {mode === 'single' ? (
              /* Single Scene Fetch Button */
              <button
                type="button"
                disabled={isStreaming}
                onClick={handleStartSingleFetch}
                className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-4 py-1.5 text-xs font-semibold text-space-black shadow-lg hover:bg-accent/90 transition-all cursor-pointer disabled:opacity-50"
              >
                {isStreaming ? (
                  <>
                    <Loader2 size={13} className="animate-spin" />
                    <span>Streaming...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={13} />
                    <span>Fetch Satellite Scene</span>
                  </>
                )}
              </button>
            ) : mode === 'multitemporal' ? (
              fetchedStack && !isStreaming ? (
                <>
                  <button
                    type="button"
                    onClick={handlePreviewStackOnMap}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-xs font-medium text-ink hover:border-cyan-400/40 hover:text-cyan-400 transition-all cursor-pointer"
                    title="Switch Map to multi-temporal timeline slider for this stack"
                  >
                    <Eye size={13} />
                    <span>Inspect Stack on Map</span>
                  </button>

                  <button
                    type="button"
                    onClick={handleRunMultiTemporalAnalysis}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-400 px-4 py-1.5 text-xs font-semibold text-space-black shadow-[0_0_15px_rgba(0,242,255,0.3)] hover:bg-cyan-300 transition-all cursor-pointer"
                  >
                    <Activity size={13} />
                    <span>Analyze Multi-Temporal Changes</span>
                    <ArrowRight size={12} />
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  disabled={isStreaming}
                  onClick={handleStartMultiTemporalFetch}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-400 px-4 py-1.5 text-xs font-semibold text-space-black shadow-lg hover:bg-cyan-300 transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {isStreaming ? (
                    <>
                      <Loader2 size={13} className="animate-spin" />
                      <span>Streaming Time Series Stack...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={13} />
                      <span>Fetch Multi-Temporal Stack</span>
                    </>
                  )}
                </button>
              )
            ) : fetchedPair && !isStreaming ? (
              /* Bi-Temporal Post-Fetch Actions */
              <>
                <button
                  type="button"
                  onClick={handlePreviewOnMap}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-xs font-medium text-ink hover:border-accent/40 hover:text-accent transition-all cursor-pointer"
                  title="Switch Map to Swipe / Fade mode for this pair"
                >
                  <Eye size={13} />
                  <span>Preview on Map</span>
                </button>

                <button
                  type="button"
                  onClick={handleRunChangeFormer}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-4 py-1.5 text-xs font-semibold text-space-black shadow-[0_0_15px_rgba(0,242,255,0.3)] hover:bg-accent/90 transition-all cursor-pointer"
                >
                  <Activity size={13} />
                  <span>Detect Changes (ChangeFormer V6)</span>
                  <ArrowRight size={12} />
                </button>
              </>
            ) : (
              /* Bi-Temporal Fetch Button */
              <button
                type="button"
                disabled={isStreaming || !isDateOrderValid}
                onClick={handleStartBiTemporalFetch}
                className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-4 py-1.5 text-xs font-semibold text-space-black shadow-lg hover:bg-accent/90 transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {isStreaming ? (
                  <>
                    <Loader2 size={13} className="animate-spin" />
                    <span>Streaming Pair...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={13} />
                    <span>Fetch Satellite Pair</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
