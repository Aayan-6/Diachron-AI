import React from 'react';
import { Calendar, Layers3, Maximize2, Ruler, Shapes, Sparkles } from 'lucide-react';
import { useApp } from '../../context/AppState';
import type { ChatMessage } from '../../lib/types';
import { cn } from '../../lib/utils';

interface EvidenceCardProps {
  message: ChatMessage;
}

/**
 * The hinge between the conversation and the map: when an answer carries
 * geographic evidence or multi-temporal timeline tracking, this sits inline
 * in the thread and pushes those features onto the scene dock.
 */
export const EvidenceCard: React.FC<EvidenceCardProps> = ({ message }) => {
  const { focusEvidence, focusedMessageId, openDock } = useApp();
  const evidence = message.evidence;

  const hasGeoJson = Boolean(evidence?.geoJson && (evidence.featureCount ?? 0) > 0);
  const hasTimeline = Boolean(evidence?.timeline && evidence.timeline.length > 1);
  const hasCandidates = Boolean(evidence?.candidates && evidence.candidates.length > 0);

  if (!hasGeoJson && !hasTimeline && !evidence?.overlay && !hasCandidates) return null;

  const isFocused = focusedMessageId === message.id;

  return (
    <div
      onClick={() => focusEvidence(message.id)}
      className={cn(
        'group mt-3 flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors cursor-pointer',
        isFocused
          ? 'border-accent/45 bg-accent/10'
          : 'border-line bg-surface-2/70 hover:border-accent/35 hover:bg-surface-3'
      )}
      role="button"
      tabIndex={0}
      aria-label="Show evidence on map"
    >
      <span
        className={cn(
          'flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border',
          isFocused ? 'border-accent/40 bg-accent/15 text-accent' : 'border-line bg-surface-3 text-accent'
        )}
        aria-hidden
      >
        {hasTimeline ? <Calendar size={16} /> : <Layers3 size={16} />}
      </span>

      <span className="min-w-0 flex-1">
        <span className="block text-[13px] font-medium text-ink">
          {hasTimeline
            ? `Multi-Temporal Series · ${evidence?.timeline?.length} Acquisitions`
            : evidence?.featureCount === 1
            ? '1 region on the map'
            : `${evidence?.featureCount ?? 0} regions on the map`}
        </span>
        <span className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 font-mono text-[10.5px] text-ink-faint">
          {evidence?.candidates && (
            <span className="inline-flex items-center gap-1 text-cyan-400">
              <Sparkles size={10} aria-hidden />
              {evidence.candidates.length} change candidates tracked
            </span>
          )}
          {hasGeoJson && (
            <span className="inline-flex items-center gap-1">
              <Shapes size={10} aria-hidden />
              GeoJSON · WGS84
            </span>
          )}
          {evidence?.areaHa !== undefined && (
            <span className="inline-flex items-center gap-1 text-amber">
              <Ruler size={10} aria-hidden />
              {evidence.areaHa.toLocaleString()} ha
            </span>
          )}
        </span>
      </span>

      <div className="flex items-center gap-1.5 shrink-0" onClick={(e) => e.stopPropagation()}>
        {hasCandidates && (
          <button
            type="button"
            onClick={() => {
              focusEvidence(message.id);
              openDock('review');
            }}
            className="flex items-center gap-1 rounded-md border border-accent/40 bg-accent/15 px-2 py-1 font-mono text-[10px] font-semibold uppercase tracking-wider text-accent hover:bg-accent/25 transition-colors cursor-pointer"
          >
            <span>Review</span>
            <span>({evidence!.candidates!.length})</span>
          </button>
        )}

        <button
          type="button"
          onClick={() => focusEvidence(message.id)}
          className={cn(
            'flex items-center gap-1 rounded-md px-2 py-1 font-mono text-[10px] uppercase tracking-wider transition-colors border',
            isFocused
              ? 'border-accent/40 bg-accent/20 text-accent font-semibold'
              : 'border-line bg-surface-3 text-ink-faint hover:text-ink hover:border-line-strong'
          )}
        >
          <Maximize2 size={11} aria-hidden />
          <span>{isFocused ? 'On map' : 'Map'}</span>
        </button>
      </div>
    </div>
  );
};

export default EvidenceCard;
