import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  Globe,
  ExternalLink,
  Terminal,
  RefreshCw,
  MapPin,
  Calendar,
  FileText,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import {
  runTinyFishInvestigation,
  streamTinyFishInvestigation,
} from '../../lib/tinyfishApi';
import type { InvestigationResponse } from '../../lib/tinyfishApi';
import { useApp } from '../../context/AppState';

interface GroundTruthSentinelPanelProps {
  initialLat?: number;
  initialLon?: number;
  initialChangeType?: string;
}

export const GroundTruthSentinelPanel: React.FC<GroundTruthSentinelPanelProps> = ({
  initialLat,
  initialLon,
  initialChangeType,
}) => {
  const { aoiTemporal } = useApp();

  // If AOI bbox exists, compute center lat/lon
  const computedLat = aoiTemporal?.bbox
    ? (aoiTemporal.bbox[1] + aoiTemporal.bbox[3]) / 2
    : 18.922;
  const computedLon = aoiTemporal?.bbox
    ? (aoiTemporal.bbox[0] + aoiTemporal.bbox[2]) / 2
    : 72.834;

  const [lat, setLat] = useState<number>(initialLat ?? computedLat);
  const [lon, setLon] = useState<number>(initialLon ?? computedLon);
  const [changeType, setChangeType] = useState<string>(
    initialChangeType ?? 'Water level decline / Surface change'
  );
  const [areaHa, setAreaHa] = useState<number>(24.5);
  const [t1Date, setT1Date] = useState<string>('2025-01-01');
  const [t2Date, setT2Date] = useState<string>('2025-08-01');

  const [loading, setLoading] = useState<boolean>(false);
  const [streamLogs, setStreamLogs] = useState<string[]>([]);
  const [report, setReport] = useState<InvestigationResponse | null>(null);
  const [showTerminal, setShowTerminal] = useState<boolean>(true);
  const [showConfig, setShowConfig] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const logsEndRef = useRef<HTMLDivElement>(null);

  const handleStartInvestigation = () => {
    setLoading(true);
    setError(null);
    setStreamLogs([]);
    setReport(null);

    const cleanup = streamTinyFishInvestigation(
      {
        lat,
        lon,
        change_type: changeType,
        area_hectares: areaHa,
        t1_date: t1Date,
        t2_date: t2Date,
      },
      (stepText) => {
        setStreamLogs((prev) => [...prev, stepText]);
      },
      (completedData) => {
        setLoading(false);
        if (completedData && completedData.attribution) {
          setReport({
            status: 'success',
            spatial_context: completedData.spatial_context,
            satellite_summary: {
              change_type: changeType,
              area_hectares: areaHa,
              t1_date: t1Date,
              t2_date: t2Date,
              coordinates: { lat, lon },
            },
            attribution: completedData.attribution,
            total_sources_scanned: completedData.attribution.corroborating_sources?.length || 5,
            investigation_timestamp: new Date().toISOString(),
          });
        }
      },
      (_err) => {
        setLoading(false);
        setError('Streaming connection closed. Falling back to direct API...');
        runTinyFishInvestigation({
          lat,
          lon,
          change_type: changeType,
          area_hectares: areaHa,
          t1_date: t1Date,
          t2_date: t2Date,
        })
          .then((res) => {
            setReport(res);
            setError(null);
          })
          .catch((e) => setError(e.message));
      }
    );

    return cleanup;
  };

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [streamLogs]);

  // Run initial investigation on mount if empty
  useEffect(() => {
    if (!report && !loading) {
      handleStartInvestigation();
    }
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-emerald-400 border-emerald-500/40 bg-emerald-950/20';
    if (score >= 60) return 'text-amber-400 border-amber-500/40 bg-amber-950/20';
    return 'text-rose-400 border-rose-500/40 bg-rose-950/20';
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto bg-neutral-950 text-neutral-100 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 bg-cyan-950/40 border border-cyan-500/30 rounded-lg text-cyan-400">
            <Globe className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <h2 className="text-sm font-semibold tracking-wide text-neutral-100 flex items-center gap-1.5">
              Ground-Truth Sentinel
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                TinyFish AI
              </span>
            </h2>
            <p className="text-xs text-neutral-400">
              Autonomous Web Agent Corroboration for Satellite Change
            </p>
          </div>
        </div>
        <button
          onClick={handleStartInvestigation}
          disabled={loading}
          className="flex items-center space-x-1.5 px-3 py-1.5 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-50 text-xs font-medium rounded-md border border-neutral-700 transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>{loading ? 'Investigating...' : 'Re-Run'}</span>
        </button>
      </div>

      {/* Target Parameters Bar */}
      <div className="grid grid-cols-2 gap-2 text-xs bg-neutral-900/60 p-2.5 rounded-lg border border-neutral-800/80">
        <div className="flex items-center gap-1.5 text-neutral-400">
          <MapPin className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <span className="truncate">
            [{lat.toFixed(3)}, {lon.toFixed(3)}]
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-neutral-400">
          <Calendar className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span className="truncate">
            {t1Date} → {t2Date}
          </span>
        </div>
        <div className="col-span-2 flex items-center justify-between pt-1 border-t border-neutral-800/50">
          <span className="truncate text-neutral-300 font-medium">
            Phenomenon: <span className="text-cyan-300">{changeType}</span> ({areaHa} ha)
          </span>
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="text-[10px] text-cyan-400 hover:text-cyan-300 underline font-mono shrink-0 ml-2"
          >
            {showConfig ? 'Hide Config' : 'Edit Target'}
          </button>
        </div>
      </div>

      {showConfig && (
        <div className="p-3 bg-neutral-900/90 border border-neutral-800 rounded-lg space-y-2 text-xs">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-neutral-400 block mb-0.5">Latitude</label>
              <input
                type="number"
                step="0.001"
                value={lat}
                onChange={(e) => setLat(parseFloat(e.target.value) || 0)}
                className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
              />
            </div>
            <div>
              <label className="text-[10px] text-neutral-400 block mb-0.5">Longitude</label>
              <input
                type="number"
                step="0.001"
                value={lon}
                onChange={(e) => setLon(parseFloat(e.target.value) || 0)}
                className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
              />
            </div>
          </div>
          <div>
            <label className="text-[10px] text-neutral-400 block mb-0.5">Change Phenomenon</label>
            <input
              type="text"
              value={changeType}
              onChange={(e) => setChangeType(e.target.value)}
              className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
            />
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="text-[10px] text-neutral-400 block mb-0.5">Area (ha)</label>
              <input
                type="number"
                value={areaHa}
                onChange={(e) => setAreaHa(parseFloat(e.target.value) || 0)}
                className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-200"
              />
            </div>
            <div>
              <label className="text-[10px] text-neutral-400 block mb-0.5">Start Date</label>
              <input
                type="date"
                value={t1Date}
                onChange={(e) => setT1Date(e.target.value)}
                className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-200 text-[11px]"
              />
            </div>
            <div>
              <label className="text-[10px] text-neutral-400 block mb-0.5">End Date</label>
              <input
                type="date"
                value={t2Date}
                onChange={(e) => setT2Date(e.target.value)}
                className="w-full bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-neutral-200 text-[11px]"
              />
            </div>
          </div>
        </div>
      )}


      {/* Live Stream Terminal */}
      <div className="border border-neutral-800 rounded-lg overflow-hidden bg-black/70">
        <div
          onClick={() => setShowTerminal(!showTerminal)}
          className="flex items-center justify-between px-3 py-2 bg-neutral-900/80 border-b border-neutral-800 cursor-pointer text-xs font-mono text-neutral-400 hover:text-neutral-200"
        >
          <div className="flex items-center space-x-2">
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            <span>Autonomous Agent Live Terminal</span>
            {loading && <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />}
          </div>
          {showTerminal ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </div>
        {showTerminal && (
          <div className="p-3 font-mono text-[11px] leading-relaxed max-h-48 overflow-y-auto space-y-1 text-neutral-300 bg-neutral-950/90">
            {streamLogs.length === 0 && !loading && (
              <div className="text-neutral-600 italic">No agent logs yet. Click 'Re-Run' to dispatch investigator.</div>
            )}
            {streamLogs.map((log, idx) => (
              <div key={idx} className="flex items-start gap-1.5">
                <span className="text-neutral-600 select-none">&gt;</span>
                <span
                  className={
                    log.includes('📍')
                      ? 'text-cyan-300'
                      : log.includes('🔍')
                      ? 'text-amber-300'
                      : log.includes('⚡')
                      ? 'text-emerald-300'
                      : 'text-neutral-300'
                  }
                >
                  {log}
                </span>
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-1.5 text-cyan-400/80 animate-pulse">
                <span>&gt;</span>
                <span>TinyFish Agent browsing regional portals...</span>
              </div>
            )}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>

      {error && (
        <div className="p-2.5 bg-rose-950/20 border border-rose-500/30 rounded-lg text-xs text-rose-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Attribution Card */}
      {report && (
        <div className="space-y-4">
          {/* Causality Confidence Scorecard */}
          <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-900/40 relative overflow-hidden">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs uppercase font-mono tracking-wider text-neutral-400">
                Cross-Modal Causality Score
              </span>
              <span
                className={`text-xs px-2.5 py-1 rounded-full font-semibold border ${getScoreColor(
                  report.attribution.causality_score
                )}`}
              >
                {report.attribution.causality_score}% Confidence
              </span>
            </div>

            <div className="text-sm font-semibold text-neutral-100 flex items-center gap-2 mb-1.5">
              <ShieldCheck className="w-4 h-4 text-cyan-400 shrink-0" />
              <span>{report.attribution.classification_label}</span>
            </div>

            <p className="text-xs text-neutral-300 leading-relaxed bg-black/20 p-2.5 rounded-lg border border-neutral-800/40">
              {report.attribution.summary}
            </p>
          </div>

          {/* Chronological Timeline */}
          {report.attribution.timeline && report.attribution.timeline.length > 0 && (
            <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-900/40">
              <h3 className="text-xs font-semibold text-neutral-300 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-amber-400" />
                Spatio-Temporal Event Chronology
              </h3>
              <div className="relative pl-4 space-y-3 before:absolute before:left-1.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-neutral-800">
                {report.attribution.timeline.map((item, idx) => (
                  <div key={idx} className="relative text-xs">
                    <div className="absolute -left-[19px] top-1 w-2.5 h-2.5 rounded-full border-2 border-neutral-900 bg-cyan-400" />
                    <div className="font-mono text-[10px] text-cyan-400 font-medium">
                      {item.date}
                    </div>
                    <div className="text-neutral-200 mt-0.5 font-medium">{item.event}</div>
                    <div className="text-[10px] text-neutral-500 mt-0.5">Source: {item.source}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Corroborating Evidence Sources */}
          {report.attribution.corroborating_sources &&
            report.attribution.corroborating_sources.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-neutral-300 uppercase tracking-wider flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-emerald-400" />
                  Corroborating Web Evidence ({report.attribution.corroborating_sources.length})
                </h3>
                <div className="space-y-2">
                  {report.attribution.corroborating_sources.map((src, idx) => (
                    <a
                      key={idx}
                      href={src.url}
                      target="_blank"
                      rel="noreferrer"
                      className="block p-2.5 rounded-lg border border-neutral-800/80 bg-neutral-900/30 hover:bg-neutral-900/70 hover:border-cyan-500/30 transition group"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <h4 className="text-xs font-medium text-neutral-200 group-hover:text-cyan-300 transition line-clamp-1">
                          {src.title}
                        </h4>
                        <ExternalLink className="w-3 h-3 text-neutral-500 group-hover:text-cyan-400 shrink-0 mt-0.5" />
                      </div>
                      <p className="text-[11px] text-neutral-400 mt-1 line-clamp-2 leading-snug">
                        {src.snippet}
                      </p>
                      <div className="flex items-center justify-between text-[10px] text-neutral-500 mt-1.5 pt-1.5 border-t border-neutral-800/40">
                        <span>{src.domain}</span>
                        <span>{src.date}</span>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
        </div>
      )}
    </div>
  );
};
