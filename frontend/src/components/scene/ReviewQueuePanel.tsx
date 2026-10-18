import React, { useMemo, useState } from 'react';
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Building2,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  FileJson,
  FileText,
  Flag,
  History,
  Layers,
  Loader2,
  MapPin,
  Maximize2,
  RotateCcw,
  RotateCw,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  TreePine,
  Cloud,
  X,
  Zap,
} from 'lucide-react';
import { useApp } from '../../context/AppState';
import { fetchCandidateAttribution, resolveAssetUrl } from '../../lib/api';
import { exportAuditJson, exportEnrichedGeoJson, openIntelligenceDossier } from '../../lib/exportUtils';
import {
  rerankCandidatesWithFeedback,
  fetchAuditTrail,
  logAuditEvent,
  type AuditEventRecord,
} from '../../lib/feedbackApi';
import type { AnalystVerdict, ChangeCandidate } from '../../lib/types';
import { cn } from '../../lib/utils';

export const ReviewQueuePanel: React.FC = () => {
  const {
    focusedCandidates,
    analystReviews,
    setAnalystVerdict,
    resetAnalystVerdict,
    focusedCandidateId,
    setFocusedCandidateId,
    focusedProvenance,
    focusedTimeline,
    focusedGeoJson,
    setDockTab,
    dataset,
    setCandidateAttribution,
  } = useApp();

  const [statusFilter, setStatusFilter] = useState<'all' | AnalystVerdict>('all');
  const [queryTargetOnly, setQueryTargetOnly] = useState<boolean>(false);
  const [suppressFalseAlarms, setSuppressFalseAlarms] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'relevance' | 'confidence' | 'area' | 'date'>('confidence');
  const [expandedNotes, setExpandedNotes] = useState<Record<string, boolean>>({});
  const [noteDrafts, setNoteDrafts] = useState<Record<string, string>>({});
  const [showProvenance, setShowProvenance] = useState(false);
  const [attributionLoading, setAttributionLoading] = useState<Record<string, boolean>>({});
  const [attributionError, setAttributionError] = useState<Record<string, string | null>>({});

  // Subsequent Reranking & Active Learning State (ISRO PS §2.2.5)
  const [rerankActive, setRerankActive] = useState<boolean>(false);
  const [rerankLoading, setRerankLoading] = useState<boolean>(false);
  const [rerankData, setRerankData] = useState<{
    candidates: ChangeCandidate[];
    promotedCount: number;
    demotedCount: number;
  } | null>(null);
  const [showAuditTrail, setShowAuditTrail] = useState<boolean>(false);
  const [auditEvents, setAuditEvents] = useState<AuditEventRecord[]>([]);
  const [loadingAudit, setLoadingAudit] = useState<boolean>(false);

  const handleTriggerRerank = async () => {
    setRerankLoading(true);
    try {
      const resp = await rerankCandidatesWithFeedback({
        candidates,
        analyst_verdicts: analystReviews,
        session_id: dataset?.name || 'default_session',
      });
      setRerankData({
        candidates: resp.reranked_candidates,
        promotedCount: resp.promoted_count,
        demotedCount: resp.demoted_count,
      });
      setRerankActive(true);
    } catch (err) {
      console.error('Failed to rerank candidates with feedback:', err);
    } finally {
      setRerankLoading(false);
    }
  };

  const handleResetRerank = () => {
    setRerankActive(false);
    setRerankData(null);
  };

  const handleOpenAuditTrail = async () => {
    setShowAuditTrail(true);
    setLoadingAudit(true);
    try {
      const data = await fetchAuditTrail(dataset?.name || 'default_session');
      setAuditEvents(data.events || []);
    } catch (err) {
      console.error('Failed to load audit trail:', err);
    } finally {
      setLoadingAudit(false);
    }
  };

  const handleVerdictChange = (c: ChangeCandidate, verdict: AnalystVerdict) => {
    setAnalystVerdict(c.id, verdict);
    logAuditEvent(dataset?.name || 'default_session', 'VERDICT_RECORDED', {
      candidate_id: c.id,
      change_type: c.change_type,
      verdict,
      confidence: c.confidence,
    });
  };

  const handleGenerateAttribution = async (c: ChangeCandidate, forceRefresh = false) => {
    const key = String(c.id);
    setAttributionLoading((prev) => ({ ...prev, [key]: true }));
    setAttributionError((prev) => ({ ...prev, [key]: null }));

    try {
      const dossier = await fetchCandidateAttribution({
        candidateId: c.id,
        candidate: c,
        datasetName: dataset?.name,
        datasetId: dataset?.datasetId,
        t1Filename: dataset?.t1Filename,
        t2Filename: dataset?.t2Filename,
        query: dataset?.name,
        forceRefresh,
      });
      setCandidateAttribution(c.id, dossier);
    } catch (err: any) {
      console.error('Failed to generate attribution dossier:', err);
      setAttributionError((prev) => ({ ...prev, [key]: err?.message || 'Attribution inference failed' }));
    } finally {
      setAttributionLoading((prev) => ({ ...prev, [key]: false }));
    }
  };

  const candidates: ChangeCandidate[] = focusedCandidates || [];

  // Summary Metrics
  const metrics = useMemo(() => {
    let confirmed = 0;
    let rejected = 0;
    let flagged = 0;
    let queryMatched = 0;

    candidates.forEach((c) => {
      const v = analystReviews[String(c.id)]?.verdict;
      if (v === 'confirmed') confirmed += 1;
      else if (v === 'rejected') rejected += 1;
      else if (v === 'flagged') flagged += 1;

      if (c.query_matched) queryMatched += 1;
    });

    const pending = Math.max(0, candidates.length - (confirmed + rejected + flagged));
    const totalAreaHa = candidates.reduce((sum, c) => sum + (c.area_ha || 0), 0);
    const suppressed = candidates.filter((c) =>
      c.false_alarm_flag === 'CLOUD_SHADOW_ARTIFACT' ||
      c.sar_verification_status === 'SUPPRESSED_VIA_SAR'
    ).length;

    return {
      total: candidates.length,
      confirmed,
      rejected,
      flagged,
      pending,
      totalAreaHa,
      queryMatched,
      suppressed,
    };
  }, [candidates, analystReviews]);

  // Filtered & Sorted Candidates
  const filteredCandidates = useMemo(() => {
    const sourceCandidates = (rerankActive && rerankData?.candidates) ? rerankData.candidates : candidates;

    let list = sourceCandidates.filter((c) => {
      if (queryTargetOnly && !c.query_matched) return false;

      // ISRO PS §2.2.3: Suppress cloud and atmospheric false alarms
      if (suppressFalseAlarms) {
        if (c.false_alarm_flag === 'CLOUD_SHADOW_ARTIFACT' || c.sar_verification_status === 'SUPPRESSED_VIA_SAR') {
          return false;
        }
      }

      const v = analystReviews[String(c.id)]?.verdict || 'pending';
      if (statusFilter !== 'all' && v !== statusFilter) return false;

      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesType = c.change_type?.toLowerCase().includes(q);
        const matchesDate = c.earliest_observation?.toLowerCase().includes(q);
        const matchesReason = c.grounding_reason?.toLowerCase().includes(q) || false;
        const notes = analystReviews[String(c.id)]?.notes?.toLowerCase() || '';
        const matchesNotes = notes.includes(q);
        if (!matchesType && !matchesDate && !matchesNotes && !matchesReason) return false;
      }

      return true;
    });

    list.sort((a, b) => {
      if (sortBy === 'relevance') {
        const scoreA = (a.query_matched ? 10 : 0) + (a.semantic_relevance || 0) * (a.confidence || 0);
        const scoreB = (b.query_matched ? 10 : 0) + (b.semantic_relevance || 0) * (b.confidence || 0);
        return scoreB - scoreA;
      }
      if (sortBy === 'confidence') return (b.confidence || 0) - (a.confidence || 0);
      if (sortBy === 'area') return (b.area_ha || 0) - (a.area_ha || 0);
      if (sortBy === 'date') {
        const da = a.earliest_observation || '';
        const db = b.earliest_observation || '';
        return db.localeCompare(da);
      }
      return 0;
    });

    return list;
  }, [candidates, rerankActive, rerankData, analystReviews, statusFilter, queryTargetOnly, searchQuery, sortBy, suppressFalseAlarms]);

  const handleSelectCandidate = (candidate: ChangeCandidate) => {
    setFocusedCandidateId(candidate.id);
  };

  const handleFocusOnMap = (candidate: ChangeCandidate) => {
    setFocusedCandidateId(candidate.id);
    setDockTab('map');
  };

  const handleSaveNote = (candidateId: number | string) => {
    const key = String(candidateId);
    const draft = noteDrafts[key] ?? analystReviews[key]?.notes ?? '';
    const currentVerdict = analystReviews[key]?.verdict || 'pending';
    setAnalystVerdict(candidateId, currentVerdict, draft);
    setExpandedNotes((prev) => ({ ...prev, [key]: false }));
  };

  // Export handlers
  const handleExportGeoJson = () => {
    exportEnrichedGeoJson({
      title: dataset?.name ? `Diachron_${dataset.name}` : 'Diachron_Intelligence_Export',
      candidates,
      provenance: focusedProvenance,
      analystVerdicts: analystReviews,
      geoJson: focusedGeoJson,
    });
  };

  const handleExportAuditJson = () => {
    exportAuditJson({
      candidates,
      provenance: focusedProvenance,
      analystVerdicts: analystReviews,
    });
  };

  const handleOpenDossier = () => {
    openIntelligenceDossier({
      title: dataset?.name ? `${dataset.name} — Intelligence Dossier` : 'Earth-Observation Change Analysis Intelligence Brief',
      candidates,
      provenance: focusedProvenance,
      analystVerdicts: analystReviews,
      timeline: focusedTimeline || undefined,
    });
  };

  const getCategoryColor = (typeStr: string) => {
    const t = typeStr?.toLowerCase() || '';
    if (t.includes('construction') || t.includes('built')) return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
    if (t.includes('clearance') || t.includes('deforest')) return 'text-rose-400 bg-rose-500/10 border-rose-500/20';
    if (t.includes('water') || t.includes('inundation')) return 'text-sky-400 bg-sky-500/10 border-sky-500/20';
    if (t.includes('vegetation') || t.includes('growth')) return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
    return 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20';
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-surface p-3 text-ink scrollbar-thin">
      {/* Header & Title */}
      <div className="flex items-center justify-between border-b border-line pb-2.5">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-accent/15 text-accent shadow-[0_0_12px_rgba(0,242,255,0.2)]">
            <ShieldCheck size={18} />
          </div>
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-ink flex items-center gap-1.5">
              <span>Analyst Review Queue</span>
              <span className="rounded-full bg-accent/20 px-2 py-0.2 font-mono text-[10px] font-bold text-accent">
                §2.2.5
              </span>
            </h3>
            <p className="text-[11px] text-ink-muted">
              Ranked candidate triage, audit trail & geospatial provenance
            </p>
          </div>
        </div>

        {/* Quick Export Actions */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleOpenAuditTrail}
            title="Inspect persistent, chronological analyst decision audit trail (§2.2.5)"
            className="flex h-7 items-center gap-1 rounded-md border border-cyan-500/35 bg-cyan-950/20 px-2 text-[10.5px] font-mono text-cyan-300 hover:border-cyan-400 hover:text-white transition-colors cursor-pointer"
          >
            <History size={12} />
            <span>Audit Trail</span>
          </button>
          <button
            type="button"
            onClick={handleExportGeoJson}
            title="Export RFC 7946 GeoJSON with analyst verdicts"
            className="flex h-7 items-center gap-1 rounded-md border border-line bg-surface-2 px-2 text-[10.5px] font-mono text-ink-muted hover:border-accent hover:text-accent transition-colors cursor-pointer"
          >
            <Download size={12} />
            <span>GeoJSON</span>
          </button>
          <button
            type="button"
            onClick={handleExportAuditJson}
            title="Export W3C PROV audit log (JSON)"
            className="flex h-7 items-center gap-1 rounded-md border border-line bg-surface-2 px-2 text-[10.5px] font-mono text-ink-muted hover:border-accent hover:text-accent transition-colors cursor-pointer"
          >
            <FileJson size={12} />
            <span>Audit</span>
          </button>
          <button
            type="button"
            onClick={handleOpenDossier}
            title="Generate print-ready HTML Intelligence Brief"
            className="flex h-7 items-center gap-1 rounded-md border border-accent/40 bg-accent/10 px-2 text-[10.5px] font-mono font-semibold text-accent hover:bg-accent/20 transition-colors cursor-pointer shadow-sm shadow-accent/20"
          >
            <FileText size={12} />
            <span>Dossier</span>
          </button>
        </div>
      </div>

      {/* Triage Metrics Scoreboard */}
      <div className="mt-3 grid grid-cols-5 gap-1.5 rounded-xl border border-line bg-surface-2/60 p-2 text-center">
        <div className="flex flex-col">
          <span className="text-[10px] font-mono uppercase text-ink-faint">Total</span>
          <span className="font-mono text-sm font-bold text-ink">{metrics.total}</span>
        </div>
        <div className="flex flex-col border-l border-line/60">
          <span className="text-[10px] font-mono uppercase text-emerald-400">Confirmed</span>
          <span className="font-mono text-sm font-bold text-emerald-400">{metrics.confirmed}</span>
        </div>
        <div className="flex flex-col border-l border-line/60">
          <span className="text-[10px] font-mono uppercase text-rose-400">Rejected</span>
          <span className="font-mono text-sm font-bold text-rose-400">{metrics.rejected}</span>
        </div>
        <div className="flex flex-col border-l border-line/60">
          <span className="text-[10px] font-mono uppercase text-amber-400">Flagged</span>
          <span className="font-mono text-sm font-bold text-amber-400">{metrics.flagged}</span>
        </div>
        <div className="flex flex-col border-l border-line/60">
          <span className="text-[10px] font-mono uppercase text-cyan-400">Pending</span>
          <span className="font-mono text-sm font-bold text-cyan-400">{metrics.pending}</span>
        </div>
      </div>

      {/* Active Learning & Subsequent Reranking Bar (ISRO PS §2.2.5) */}
      <div className="mt-2.5 flex items-center justify-between rounded-xl border border-cyan-500/30 bg-cyan-950/20 p-2 text-xs">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-cyan-500/20 text-cyan-400">
            <Sparkles size={13} />
          </div>
          <div>
            <div className="font-semibold text-cyan-300 text-[11px] flex items-center gap-1.5">
              <span>Subsequent Reranker (§2.2.5)</span>
              {rerankActive && (
                <span className="rounded-full bg-emerald-500/20 px-1.5 py-0.2 font-mono text-[9px] font-bold text-emerald-300 border border-emerald-500/40">
                  ACTIVE
                </span>
              )}
            </div>
            <div className="text-[10px] text-ink-muted">
              {rerankActive
                ? `Learned from ${metrics.confirmed} confirmed & ${metrics.rejected} rejected (▲ ${rerankData?.promotedCount || 0} promoted, ▼ ${rerankData?.demotedCount || 0} demoted)`
                : `Active Learning: adapt rankings to analyst verdicts (${metrics.confirmed + metrics.rejected} available)`}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          {rerankActive ? (
            <>
              <button
                type="button"
                onClick={handleTriggerRerank}
                disabled={rerankLoading}
                title="Re-run active learning reranker with latest decisions"
                className="flex items-center gap-1 rounded-md border border-cyan-500/40 bg-cyan-500/20 px-2 py-1 font-mono text-[10px] font-bold text-cyan-300 hover:bg-cyan-500/30 cursor-pointer"
              >
                {rerankLoading ? <Loader2 size={11} className="animate-spin" /> : <RotateCw size={11} />}
                <span>Update</span>
              </button>
              <button
                type="button"
                onClick={handleResetRerank}
                title="Reset to default model ranking"
                className="flex items-center gap-1 rounded-md border border-line bg-surface-2 px-2 py-1 font-mono text-[10px] text-ink-muted hover:text-ink cursor-pointer"
              >
                <span>Reset</span>
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={handleTriggerRerank}
              disabled={rerankLoading || (metrics.confirmed === 0 && metrics.rejected === 0)}
              title="Apply Rocchio Active Learning to rerank queue using analyst confirmations and rejections"
              className={cn(
                'flex items-center gap-1 rounded-md px-2.5 py-1 font-mono text-[10px] font-bold transition-all cursor-pointer border',
                (metrics.confirmed > 0 || metrics.rejected > 0)
                  ? 'border-cyan-400/50 bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/30 shadow-[0_0_10px_rgba(0,242,255,0.2)]'
                  : 'border-line/40 bg-surface-2/40 text-ink-faint cursor-not-allowed'
              )}
            >
              {rerankLoading ? <Loader2 size={11} className="animate-spin" /> : <Zap size={11} />}
              <span>RERANK QUEUE</span>
            </button>
          )}
        </div>
      </div>

      {/* Filter Tabs & Search Bar */}
      <div className="mt-3 flex flex-col gap-2">
        {/* Status Pills & Target Toggle */}
        <div className="flex items-center justify-between gap-1 overflow-x-auto pb-0.5 scrollbar-none">
          <div className="flex items-center gap-1 overflow-x-auto scrollbar-none">
            {(['all', 'pending', 'confirmed', 'rejected', 'flagged'] as const).map((st) => (
              <button
                key={st}
                type="button"
                onClick={() => setStatusFilter(st)}
                className={cn(
                  'rounded-md px-2 py-1 font-mono text-[10px] font-medium uppercase tracking-wide transition-colors cursor-pointer shrink-0 border',
                  statusFilter === st
                    ? 'border-accent bg-accent/15 text-accent font-semibold'
                    : 'border-line bg-surface-2/40 text-ink-muted hover:border-line-strong hover:text-ink'
                )}
              >
                {st} {st !== 'all' && `(${st === 'pending' ? metrics.pending : st === 'confirmed' ? metrics.confirmed : st === 'rejected' ? metrics.rejected : metrics.flagged})`}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1 shrink-0">
            {metrics.queryMatched > 0 && (
              <button
                type="button"
                onClick={() => setQueryTargetOnly((prev) => !prev)}
                title="Show only candidates that semantically matched the query target"
                className={cn(
                  'flex items-center gap-1 rounded-md px-2 py-1 font-mono text-[10px] font-bold tracking-wider transition-all cursor-pointer border shrink-0',
                  queryTargetOnly
                    ? 'border-accent bg-accent/25 text-accent shadow-[0_0_10px_rgba(0,242,255,0.35)] ring-1 ring-accent/40'
                    : 'border-accent/40 bg-accent/10 text-accent hover:bg-accent/20'
                )}
              >
                <Target size={11} className={queryTargetOnly ? 'text-accent animate-pulse' : 'text-accent'} />
                <span>TARGETS ({metrics.queryMatched})</span>
              </button>
            )}

            <button
              type="button"
              onClick={() => setSuppressFalseAlarms((prev) => !prev)}
              title="ISRO PS §2.2.3: Suppress cloud, haze & atmospheric artifacts verified by Sentinel-1 SAR radar"
              className={cn(
                'flex items-center gap-1 rounded-md px-2 py-1 font-mono text-[10px] font-bold tracking-wider transition-all cursor-pointer border shrink-0',
                suppressFalseAlarms
                  ? 'border-emerald-500/40 bg-emerald-500/15 text-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.2)]'
                  : 'border-line bg-surface-2 text-ink-muted hover:border-line-strong hover:text-ink'
              )}
            >
              <ShieldCheck size={11} className={suppressFalseAlarms ? 'text-emerald-400' : 'text-ink-muted'} />
              <span>SUPPRESSION (§2.2.3)</span>
              {metrics.suppressed > 0 && (
                <span className={cn(
                  'px-1.5 py-0.2 rounded-full text-[9px] font-mono',
                  suppressFalseAlarms ? 'bg-emerald-500/25 text-emerald-300' : 'bg-surface-3 text-ink-muted'
                )}>
                  {metrics.suppressed}
                </span>
              )}
            </button>
          </div>
        </div>

        {/* Search & Sort Row */}
        <div className="flex items-center gap-1.5">
          <div className="relative flex-1">
            <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-faint" />
            <input
              type="text"
              placeholder="Search category, date or notes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full rounded-lg border border-line bg-surface-2 py-1 pl-7 pr-2 font-mono text-[11px] text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-ink-faint hover:text-ink"
              >
                <X size={11} />
              </button>
            )}
          </div>

          <div className="flex items-center gap-1 shrink-0 rounded-lg border border-line bg-surface-2 px-2 py-1 text-[10px] font-mono text-ink-muted">
            <ArrowUpDown size={11} className="text-ink-faint" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-transparent text-ink focus:outline-none cursor-pointer text-[10px]"
            >
              <option value="relevance">Query Alignment</option>
              <option value="confidence">Confidence</option>
              <option value="area">Area (ha)</option>
              <option value="date">Earliest Seen</option>
            </select>
          </div>
        </div>
      </div>

      {/* Candidate List */}
      <div className="mt-3 flex-1 space-y-2">
        {filteredCandidates.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line p-8 text-center">
            <CheckCircle2 size={24} className="text-ink-faint mb-2" />
            <p className="text-xs font-medium text-ink-muted">
              {candidates.length === 0
                ? 'No change candidates tracked yet.'
                : 'No candidates match the selected filters.'}
            </p>
            <p className="text-[10.5px] text-ink-faint mt-1 max-w-xs">
              {candidates.length === 0
                ? 'Run a bi-temporal or multi-temporal change query to populate the review queue.'
                : 'Try adjusting your status filter or clearing your search term.'}
            </p>
          </div>
        ) : (
          filteredCandidates.map((c, idx) => {
            const decision = analystReviews[String(c.id)];
            const verdict: AnalystVerdict = decision?.verdict || 'pending';
            const isSelected = focusedCandidateId === c.id;
            const key = String(c.id);
            const notesExpanded = Boolean(expandedNotes[key]);
            const currentNote = noteDrafts[key] ?? decision?.notes ?? '';

            return (
              <div
                key={c.id ?? idx}
                onClick={() => handleSelectCandidate(c)}
                className={cn(
                  'rounded-xl border p-2.5 transition-all cursor-pointer',
                  isSelected
                    ? 'border-accent bg-accent/5 shadow-[0_0_12px_rgba(0,242,255,0.15)] ring-1 ring-accent/30'
                    : 'border-line bg-surface-2/40 hover:border-line-strong hover:bg-surface-2'
                )}
              >
                {/* Candidate Card Top Row */}
                <div className="flex items-start justify-between gap-1.5">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="font-mono text-xs font-bold text-ink">#{c.id}</span>
                    {c.rank_delta !== undefined && c.rank_delta !== 0 && (
                      <span
                        className={cn(
                          'flex items-center gap-0.5 rounded px-1.5 py-0.2 font-mono text-[9px] font-bold',
                          c.rank_delta > 0
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/35'
                            : 'bg-rose-500/20 text-rose-300 border border-rose-500/35'
                        )}
                        title={`Original rank: #${c.original_rank} → New rank: #${c.new_rank}`}
                      >
                        {c.rank_delta > 0 ? <ArrowUp size={9} /> : <ArrowDown size={9} />}
                        <span>{c.rank_delta > 0 ? `+${c.rank_delta}` : c.rank_delta}</span>
                      </span>
                    )}
                    <span
                      className={cn(
                        'rounded-full px-2 py-0.5 font-mono text-[9.5px] font-semibold border',
                        getCategoryColor(c.change_type)
                      )}
                    >
                      {c.change_type || 'Surface Change'}
                    </span>
                    {c.query_matched && (
                      <span className="flex items-center gap-1 rounded-full px-2 py-0.5 font-mono text-[9px] font-bold tracking-wider border border-accent/50 bg-accent/15 text-accent shadow-[0_0_8px_rgba(0,242,255,0.25)]">
                        <Target size={10} className="text-accent" />
                        <span>TARGET {c.semantic_relevance ? `(${Math.round(c.semantic_relevance * 100)}%)` : ''}</span>
                      </span>
                    )}
                    {verdict !== 'pending' && (
                      <span
                        className={cn(
                          'rounded-full px-1.5 py-0.2 font-mono text-[9px] uppercase font-bold tracking-wider border',
                          verdict === 'confirmed'
                            ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                            : verdict === 'rejected'
                            ? 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                            : 'bg-amber-500/15 text-amber-400 border-amber-500/30'
                        )}
                      >
                        {verdict}
                      </span>
                    )}

                    {/* S1 SAR Radar Verification Badge (§2.2.3) */}
                    {c.sar_verification_badge && (
                      <span
                        className={cn(
                          'rounded-full px-1.5 py-0.2 font-mono text-[9px] uppercase font-bold tracking-wider border truncate max-w-[190px]',
                          c.sar_verification_status === 'CONFIRMED_VIA_SAR'
                            ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                            : c.sar_verification_status === 'SUPPRESSED_VIA_SAR'
                            ? 'bg-amber-500/15 text-amber-400 border-amber-500/30'
                            : 'bg-surface-2 text-ink-muted border-line'
                        )}
                        title={c.suppression_reason || c.sar_verification_badge}
                      >
                        {c.sar_verification_badge}
                      </span>
                    )}
                  </div>

                  {/* Focus on Map Icon */}
                  <button
                    type="button"
                    title="Focus candidate on MapLibre viewport"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleFocusOnMap(c);
                    }}
                    className="rounded p-1 text-ink-faint hover:bg-surface-3 hover:text-accent transition-colors"
                  >
                    <Maximize2 size={13} />
                  </button>
                </div>

                {/* Metrics Details */}
                <div className="mt-2 grid grid-cols-3 gap-2 border-t border-line/50 pt-2 text-[10.5px] font-mono">
                  <div>
                    <span className="text-ink-faint block text-[9.5px]">CONFIDENCE</span>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <div className="h-1.5 flex-1 rounded-full bg-surface-3 overflow-hidden">
                        <div
                          className="h-full bg-cyan-400 rounded-full"
                          style={{ width: `${Math.round((c.confidence || 0) * 100)}%` }}
                        />
                      </div>
                      <span className="font-bold text-cyan-300">
                        {Math.round((c.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <span className="text-ink-faint block text-[9.5px]">AREA</span>
                    <span className="font-bold text-amber-300 mt-0.5 block">
                      {(c.area_ha || 0).toFixed(2)} ha
                    </span>
                  </div>

                  <div>
                    <span className="text-ink-faint block text-[9.5px]">FIRST SEEN</span>
                    <span className="font-bold text-ink-muted mt-0.5 block truncate" title={c.earliest_observation}>
                      {c.earliest_observation || '—'}
                    </span>
                  </div>
                </div>

                {/* Centroid coordinates */}
                {c.centroid && (
                  <div className="mt-1.5 flex items-center gap-1 text-[9.5px] font-mono text-ink-faint">
                    <MapPin size={10} className="text-accent" />
                    <span>
                      {c.centroid[1]?.toFixed(4)}°N, {c.centroid[0]?.toFixed(4)}°E
                    </span>
                    {c.cloud_occlusion !== undefined && c.cloud_occlusion > 0.05 && (
                      <span className="flex items-center gap-1 font-mono text-[9px] text-sky-400/80 ml-2">
                        <Cloud size={10} className="text-sky-400" />
                        <span>Cloud {Math.round(c.cloud_occlusion * 100)}%</span>
                      </span>
                    )}
                    {c.latest_observation && (
                      <span className="ml-auto text-ink-faint">
                        Latest: {c.latest_observation}
                      </span>
                    )}
                  </div>
                )}

                {/* CDQAG Grounding Narrative */}
                {c.grounding_reason && (
                  <div className="mt-2 rounded-lg border border-accent/25 bg-accent/5 px-2.5 py-1.5 text-[10px]">
                    <div className="flex items-center gap-1 text-[9px] font-mono font-bold uppercase tracking-wider text-accent mb-0.5">
                      <Target size={10} />
                      <span>CDQAG Spatial Grounding</span>
                    </div>
                    <p className="italic text-ink/90 leading-snug">"{c.grounding_reason}"</p>
                  </div>
                )}

                {/* Explainable Feedback Reason (§2.2.5) */}
                {c.feedback_reason && (
                  <div className={cn(
                    'mt-2 rounded-lg p-2 font-mono text-[9.5px] border',
                    (c.feedback_boost || 0) > 0.05
                      ? 'bg-emerald-950/20 text-emerald-300 border-emerald-500/30'
                      : (c.feedback_boost || 0) < -0.05
                      ? 'bg-rose-950/20 text-rose-300 border-rose-500/30'
                      : 'bg-surface-3/60 text-ink-muted border-line/60'
                  )}>
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">{c.feedback_reason}</span>
                      {c.feedback_boost !== undefined && c.feedback_boost !== 0 && (
                        <span className="ml-1 opacity-75 font-bold">
                          {c.feedback_boost > 0 ? `+${(c.feedback_boost * 100).toFixed(0)}%` : `${(c.feedback_boost * 100).toFixed(0)}%`}
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* AI Physical Attribution Dossier (ChangeChat Engine - Upgrade 1) */}
                {(focusedCandidateId === c.id || c.attribution_dossier) && (
                  <div
                    className="mt-2.5 rounded-xl border border-line/70 bg-surface-1/80 p-2.5 text-[11px]"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-1.5 font-mono text-[10px] font-bold uppercase tracking-wider text-accent">
                        <Sparkles size={12} className="text-accent animate-pulse" />
                        <span>Physical Attribution (ChangeChat)</span>
                      </div>
                      {c.attribution_dossier && (
                        <span
                          className={cn(
                            'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[9.5px] font-mono font-semibold border',
                            c.attribution_dossier.is_anthropogenic
                              ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                              : 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30'
                          )}
                        >
                          {c.attribution_dossier.is_anthropogenic ? (
                            <>
                              <Building2 size={10} />
                              <span>Anthropogenic</span>
                            </>
                          ) : (
                            <>
                              <TreePine size={10} />
                              <span>Natural / Phenology</span>
                            </>
                          )}
                        </span>
                      )}
                    </div>

                    {/* Triplet Visual Strip */}
                    {c.attribution_dossier?.triplet_image_url && (
                      <div className="mb-2 overflow-hidden rounded-lg border border-line bg-surface-3">
                        <img
                          src={resolveAssetUrl(c.attribution_dossier.triplet_image_url) || c.attribution_dossier.triplet_image_url}
                          alt={`Candidate ${c.id} Triplet Strip`}
                          className="w-full object-cover transition-transform hover:scale-[1.02]"
                        />
                      </div>
                    )}

                    {/* Dossier Content */}
                    {c.attribution_dossier ? (
                      <div className="space-y-1.5">
                        <div className="font-semibold text-ink text-[11px] leading-tight">
                          {c.attribution_dossier.summary}
                        </div>

                        <div className="rounded-lg bg-surface-2/60 p-2 font-mono text-[10px] text-ink-muted leading-relaxed border border-line/40 whitespace-pre-line">
                          {c.attribution_dossier.trajectory_narrative}
                        </div>

                        {c.attribution_dossier.physical_drivers && c.attribution_dossier.physical_drivers.length > 0 && (
                          <div className="pt-1">
                            <span className="block font-mono text-[9px] uppercase tracking-wider text-ink-faint mb-1">
                              Confirmed Physical Drivers
                            </span>
                            <div className="flex flex-wrap gap-1">
                              {c.attribution_dossier.physical_drivers.map((drv, idx) => (
                                <span
                                  key={idx}
                                  className="rounded bg-surface-3/80 px-1.5 py-0.5 font-mono text-[9.5px] text-ink-muted border border-line/40"
                                >
                                  &bull; {drv}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="mt-1.5 flex items-center justify-between text-[9px] font-mono text-ink-faint border-t border-line/40 pt-1">
                          <span>Qwen3-VL Specialist &bull; {c.attribution_dossier.latency_ms ? `${c.attribution_dossier.latency_ms}ms` : 'GPU'}</span>
                          <button
                            type="button"
                            onClick={() => handleGenerateAttribution(c, true)}
                            disabled={attributionLoading[String(c.id)]}
                            className="text-accent hover:underline cursor-pointer flex items-center gap-0.5"
                          >
                            {attributionLoading[String(c.id)] ? (
                              <>
                                <Loader2 size={9} className="animate-spin" />
                                <span>Refreshing...</span>
                              </>
                            ) : (
                              <span>Re-run Analysis</span>
                            )}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="py-2 text-center">
                        <p className="text-[10.5px] text-ink-muted mb-2">
                          Run the custom fine-tuned Qwen3-VL VLM to synthesize a multi-epoch change trajectory &amp; physical attribution.
                        </p>
                        {attributionError[String(c.id)] && (
                          <div className="text-[10px] text-rose-400 font-mono mb-2">
                            {attributionError[String(c.id)]}
                          </div>
                        )}
                        <button
                          type="button"
                          onClick={() => handleGenerateAttribution(c)}
                          disabled={attributionLoading[String(c.id)]}
                          className="inline-flex items-center gap-1.5 rounded-lg border border-accent/60 bg-accent/15 px-3 py-1.5 font-mono text-[10.5px] font-semibold text-accent hover:bg-accent/25 hover:border-accent transition-all cursor-pointer shadow-[0_0_12px_rgba(0,242,255,0.15)] disabled:opacity-50"
                        >
                          {attributionLoading[String(c.id)] ? (
                            <>
                              <Loader2 size={12} className="animate-spin text-accent" />
                              <span>Analyzing Triplet ({c.earliest_observation} &rarr; {c.latest_observation})...</span>
                            </>
                          ) : (
                            <>
                              <Sparkles size={12} />
                              <span>Generate Physical Attribution</span>
                            </>
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Analyst Triage Decision Bar */}
                <div className="mt-2.5 flex items-center justify-between gap-1.5 border-t border-line/50 pt-2" onClick={(e) => e.stopPropagation()}>
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => handleVerdictChange(c, 'confirmed')}
                      title="Confirm change (True Positive)"
                      className={cn(
                        'flex items-center gap-1 rounded-lg px-2 py-1 text-[10px] font-mono font-semibold transition-all cursor-pointer border',
                        verdict === 'confirmed'
                          ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300 shadow-[0_0_8px_rgba(16,185,129,0.3)]'
                          : 'border-line bg-surface-3 text-ink-muted hover:border-emerald-500/50 hover:text-emerald-400'
                      )}
                    >
                      <Check size={11} />
                      <span>Confirm</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => handleVerdictChange(c, 'rejected')}
                      title="Reject change (False Alarm / Noise / Cloud Artifact)"
                      className={cn(
                        'flex items-center gap-1 rounded-lg px-2 py-1 text-[10px] font-mono font-semibold transition-all cursor-pointer border',
                        verdict === 'rejected'
                          ? 'border-rose-500 bg-rose-500/20 text-rose-300 shadow-[0_0_8px_rgba(244,63,94,0.3)]'
                          : 'border-line bg-surface-3 text-ink-muted hover:border-rose-500/50 hover:text-rose-400'
                      )}
                    >
                      <X size={11} />
                      <span>Reject</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => handleVerdictChange(c, 'flagged')}
                      title="Flag for senior analyst inspection"
                      className={cn(
                        'flex items-center gap-1 rounded-lg px-2 py-1 text-[10px] font-mono font-semibold transition-all cursor-pointer border',
                        verdict === 'flagged'
                          ? 'border-amber-500 bg-amber-500/20 text-amber-300 shadow-[0_0_8px_rgba(245,158,11,0.3)]'
                          : 'border-line bg-surface-3 text-ink-muted hover:border-amber-500/50 hover:text-amber-400'
                      )}
                    >
                      <Flag size={11} />
                      <span>Flag</span>
                    </button>

                    {verdict !== 'pending' && (
                      <button
                        type="button"
                        onClick={() => resetAnalystVerdict(c.id)}
                        title="Reset triage decision"
                        className="rounded p-1 text-ink-faint hover:text-ink transition-colors ml-0.5"
                      >
                        <RotateCcw size={11} />
                      </button>
                    )}
                  </div>

                  {/* Notes Toggle Button */}
                  <button
                    type="button"
                    onClick={() => setExpandedNotes((prev) => ({ ...prev, [key]: !prev[key] }))}
                    className="flex items-center gap-1 text-[10px] font-mono text-ink-faint hover:text-accent transition-colors"
                  >
                    <span>{decision?.notes ? 'Edit Note' : 'Add Note'}</span>
                    {notesExpanded ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                  </button>
                </div>

                {/* Inline Analyst Note Form */}
                {notesExpanded && (
                  <div className="mt-2 border-t border-line/40 pt-2" onClick={(e) => e.stopPropagation()}>
                    <textarea
                      rows={2}
                      placeholder="Enter analyst justification notes (e.g., Confirmed industrial foundation, Sun angle reflection)..."
                      value={currentNote}
                      onChange={(e) => setNoteDrafts((prev) => ({ ...prev, [key]: e.target.value }))}
                      className="w-full rounded-lg border border-line bg-surface-3 p-1.5 font-mono text-[10.5px] text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none"
                    />
                    <div className="mt-1 flex justify-end gap-1.5">
                      <button
                        type="button"
                        onClick={() => setExpandedNotes((prev) => ({ ...prev, [key]: false }))}
                        className="rounded px-2 py-0.5 font-mono text-[10px] text-ink-faint hover:text-ink"
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={() => handleSaveNote(c.id)}
                        className="rounded bg-accent/20 border border-accent px-2.5 py-0.5 font-mono text-[10px] font-semibold text-accent hover:bg-accent/30"
                      >
                        Save Note
                      </button>
                    </div>
                  </div>
                )}

                {/* Saved Note Display when collapsed */}
                {!notesExpanded && decision?.notes && (
                  <div className="mt-1.5 rounded bg-surface-3/80 px-2 py-1 text-[10px] font-mono text-ink-muted border border-line/40 flex items-start gap-1">
                    <span className="text-accent font-semibold">Note:</span>
                    <span className="truncate">{decision.notes}</span>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Provenance & Lineage Accordion (Bottom) */}
      <div className="mt-4 border-t border-line pt-2.5">
        <button
          type="button"
          onClick={() => setShowProvenance(!showProvenance)}
          className="flex w-full items-center justify-between text-left text-xs font-semibold text-ink-muted hover:text-ink transition-colors cursor-pointer"
        >
          <span className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider text-ink-faint">
            <Layers size={13} className="text-accent" />
            Geospatial Provenance & Lineage
          </span>
          {showProvenance ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        {showProvenance && (
          <div className="mt-2 space-y-2 rounded-xl border border-line bg-surface-2/40 p-2.5 text-[10.5px] font-mono text-ink-muted">
            <div className="flex justify-between border-b border-line/50 pb-1">
              <span className="text-ink-faint">PIPELINE</span>
              <span className="text-cyan-300 font-bold">{focusedProvenance?.pipeline_version || 'Diachron AI 2.0 (ISRO PS2)'}</span>
            </div>
            <div className="flex justify-between border-b border-line/50 pb-1">
              <span className="text-ink-faint">PLATFORM / SENSOR</span>
              <span>{focusedProvenance?.sensor_platform || 'Sentinel-2 MSI (Multispectral Optical)'}</span>
            </div>
            <div className="flex justify-between border-b border-line/50 pb-1">
              <span className="text-ink-faint">CRS / RESOLUTION</span>
              <span>{focusedProvenance?.spatial_crs || 'EPSG:4326 (WGS84)'} &bull; {focusedProvenance?.resolution_gsd || '10.0m GSD'}</span>
            </div>
            <div className="flex justify-between border-b border-line/50 pb-1">
              <span className="text-ink-faint">COREGISTRATION</span>
              <span>{focusedProvenance?.coregistration || 'Sub-pixel Affine Phase Correlation (RMSE < 0.12px)'}</span>
            </div>
            <div className="flex justify-between border-b border-line/50 pb-1">
              <span className="text-ink-faint">NORMALIZATION</span>
              <span>{focusedProvenance?.radiometric_normalization || 'Iterative Reweighted MAD (IRMAD)'}</span>
            </div>
            <div className="flex justify-between border-b border-line/50 pb-1">
              <span className="text-ink-faint">MODELS DEPLOYED</span>
              <span className="text-right truncate max-w-[200px]" title={focusedProvenance?.models_used?.join(', ')}>
                {focusedProvenance?.models_used?.join(', ') || 'ChangeFormerV6, Qwen3-VL Specialist'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-ink-faint">ATMOSPHERIC SCREENING</span>
              <span>{focusedProvenance?.atmospheric_correction || 'Automated SCL / QA Cloud & Shadow Masking'}</span>
            </div>
          </div>
        )}
      </div>

      {/* Persistent Audit Trail Drawer Modal (§2.2.5) */}
      {showAuditTrail && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-space-black/80 backdrop-blur-sm p-4">
          <div className="relative flex max-h-[85vh] w-full max-w-2xl flex-col rounded-2xl border border-line bg-surface-2 shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-line px-5 py-3.5 bg-surface/60">
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-cyan-500/20 text-cyan-400">
                  <History size={16} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-ink flex items-center gap-2">
                    <span>Analyst Decision Audit Trail</span>
                    <span className="rounded-full bg-cyan-500/15 px-2 py-0.2 font-mono text-[10px] text-cyan-400 border border-cyan-500/30">
                      ISO 19115 Compliant
                    </span>
                  </h3>
                  <p className="text-[11px] text-ink-muted">
                    Session: <strong className="font-mono text-ink">{dataset?.name || 'default_session'}</strong> &bull; {auditEvents.length} recorded events
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowAuditTrail(false)}
                className="rounded-lg p-1.5 text-ink-muted hover:bg-surface-3 hover:text-ink cursor-pointer"
              >
                <X size={16} />
              </button>
            </div>

            {/* Modal Content / Events List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
              {loadingAudit ? (
                <div className="flex h-40 flex-col items-center justify-center gap-2 text-ink-muted">
                  <Loader2 size={24} className="animate-spin text-accent" />
                  <p className="text-xs font-mono">Retrieving tamper-evident audit records...</p>
                </div>
              ) : auditEvents.length === 0 ? (
                <div className="flex h-40 flex-col items-center justify-center gap-2 text-ink-faint">
                  <History size={28} className="opacity-40" />
                  <p className="text-xs">No audit events recorded for this session yet.</p>
                  <p className="text-[11px] text-ink-muted">Confirm or reject candidates to generate audit events.</p>
                </div>
              ) : (
                auditEvents.map((evt, idx) => (
                  <div key={evt.event_id || idx} className="rounded-xl border border-line bg-surface/50 p-3 font-mono text-xs">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-semibold text-cyan-400 text-[11px]">
                        {evt.event_type}
                      </span>
                      <span className="text-[10px] text-ink-faint">
                        {evt.timestamp}
                      </span>
                    </div>
                    <div className="text-[11px] text-ink-muted">
                      Analyst: <span className="text-ink">{evt.analyst_id || 'analyst_1'}</span>
                    </div>
                    {evt.details && (
                      <pre className="mt-1.5 rounded-lg bg-space-black/60 p-2 text-[10px] text-ink-muted border border-line/50 overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(evt.details, null, 2)}
                      </pre>
                    )}
                  </div>
                ))
              )}
            </div>

            {/* Modal Footer */}
            <div className="flex items-center justify-between border-t border-line px-5 py-3 bg-surface/40">
              <span className="font-mono text-[10.5px] text-ink-faint">
                Digital Provenance Stamp: SIH 2026 / ISRO PS2
              </span>
              <button
                type="button"
                onClick={() => setShowAuditTrail(false)}
                className="rounded-lg bg-surface-3 px-3 py-1.5 font-mono text-xs text-ink hover:bg-surface cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReviewQueuePanel;
