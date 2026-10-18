import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronsLeft, ChevronsRight, ClipboardCheck, Globe, Images, Map as MapIcon, Terminal } from 'lucide-react';
import MapViewport from '../map/MapViewport';
import { ScenePanel } from './ScenePanel';
import { TracePanel } from './TracePanel';
import { ReviewQueuePanel } from './ReviewQueuePanel';
import { GroundTruthSentinelPanel } from '../tinyfish/GroundTruthSentinelPanel';
import { Segmented } from '../ui/Segmented';
import { IconButton } from '../ui/Button';
import { useApp } from '../../context/AppState';
import { cn } from '../../lib/utils';

const MIN_WIDTH = 320;
const MAX_WIDTH = 780;
const DEFAULT_WIDTH = 460;
const WIDTH_KEY = 'diachron.dockWidth';


/**
 * Right-hand context dock. The map lives here rather than owning the screen:
 * it is a companion to the conversation, sized by the reader and collapsible
 * to a thin strip when the answer is all that matters.
 */
export const SceneDock: React.FC = () => {
  const { dockOpen, setDockOpen, dockTab, setDockTab, dataset, overlay, focusedGeoJson, focusedOverlay, focusedCandidates } =
    useApp();

  const tabs = useMemo(() => [
    { value: 'map' as const, label: 'Map', icon: <MapIcon size={13} /> },
    {
      value: 'review' as const,
      label: focusedCandidates && focusedCandidates.length > 0 ? `Review (${focusedCandidates.length})` : 'Review',
      icon: <ClipboardCheck size={13} />,
    },
    { value: 'trace' as const, label: 'Trace', icon: <Terminal size={13} /> },
    { value: 'scene' as const, label: 'Scene', icon: <Images size={13} /> },
    { value: 'sentinel' as const, label: 'Sentinel', icon: <Globe size={13} /> },
  ], [focusedCandidates]);

  const [width, setWidth] = useState<number>(() => {
    const stored = Number(localStorage.getItem(WIDTH_KEY));
    return Number.isFinite(stored) && stored >= MIN_WIDTH && stored <= MAX_WIDTH
      ? stored
      : DEFAULT_WIDTH;
  });
  const [dragging, setDragging] = useState(false);
  const asideRef = useRef<HTMLElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(WIDTH_KEY, String(width));
    } catch {
      /* storage unavailable — the width simply resets next session */
    }
  }, [width]);

  // Drag-to-resize. Listeners live on window so the pointer can leave the
  // 5px handle without the drag stalling.
  const startResize = useCallback((event: React.PointerEvent) => {
    event.preventDefault();
    setDragging(true);

    const onMove = (moveEvent: PointerEvent) => {
      const right = asideRef.current?.parentElement?.getBoundingClientRect().right ?? window.innerWidth;
      const next = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, right - moveEvent.clientX));
      setWidth(next);
    };
    const onUp = () => {
      setDragging(false);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }, []);

  const nudge = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowLeft') setWidth((w) => Math.min(MAX_WIDTH, w + 24));
    if (event.key === 'ArrowRight') setWidth((w) => Math.max(MIN_WIDTH, w - 24));
  };

  if (!dockOpen) {
    return (
      <aside className="flex w-12 shrink-0 flex-col items-center gap-1.5 border-l border-line bg-surface py-3">
        <IconButton label="Open scene panel" onClick={() => setDockOpen(true)}>
          <ChevronsLeft size={15} />
        </IconButton>
        <div className="my-1 h-px w-5 bg-line" aria-hidden />
        {tabs.map((tab) => (
          <IconButton
            key={tab.value}
            label={tab.label}
            active={dockTab === tab.value}
            onClick={() => {
              setDockTab(tab.value);
              setDockOpen(true);
            }}
          >
            {tab.icon}
          </IconButton>
        ))}
      </aside>
    );
  }

  return (
    <aside
      ref={asideRef}
      style={{ width }}
      className="relative flex shrink-0 flex-col border-l border-line bg-surface"
      aria-label="Scene and evidence"
    >
      {/* Resize handle */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize scene panel"
        tabIndex={0}
        onPointerDown={startResize}
        onKeyDown={nudge}
        className={cn(
          'absolute -left-[3px] top-0 z-20 h-full w-[6px] cursor-col-resize',
          'after:absolute after:left-[2px] after:h-full after:w-px after:bg-transparent after:transition-colors',
          dragging && 'after:bg-accent',
          'hover:after:bg-accent/60'
        )}
      />

      <header className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-line px-2.5">
        <Segmented
          options={tabs}
          value={dockTab}
          onChange={setDockTab}
          size="xs"
          ariaLabel="Scene panel view"
        />
        <IconButton label="Collapse scene panel" size="sm" onClick={() => setDockOpen(false)}>
          <ChevronsRight size={14} />
        </IconButton>
      </header>

      {/*
        The map is always mounted *and always laid out*.

        Hiding it with `display: none` is what a tab switch would normally do,
        but a MapLibre map in a zero-size container never completes its first
        render, so its `load` event never fires. Everything that adds a source
        or layer waits on that event, so the change-mask overlay and the
        detection outlines both silently failed to appear whenever the reader
        had opened the Scene or Trace tab first. Keeping the box and dropping
        opacity keeps the WebGL context alive and the camera where it was.
      */}
      <div className="relative min-h-0 flex-1">
        <div
          className={cn(
            'absolute inset-0 flex transition-opacity duration-150',
            dockTab === 'map' ? 'opacity-100' : 'pointer-events-none opacity-0'
          )}
          aria-hidden={dockTab !== 'map'}
        >
          <MapViewport
            density="compact"
            geoJsonData={focusedGeoJson}
            resultOverlay={focusedOverlay}
            datasetName={dataset?.name}
            sensor={dataset?.sensor}
            overlay={overlay}
          />
        </div>

        {dockTab === 'review' && (
          <div className="absolute inset-0 flex flex-col bg-surface">
            <ReviewQueuePanel />
          </div>
        )}
        {dockTab === 'trace' && (
          <div className="absolute inset-0 flex flex-col bg-surface">
            <TracePanel />
          </div>
        )}
        {dockTab === 'scene' && (
          <div className="absolute inset-0 flex flex-col bg-surface">
            <ScenePanel />
          </div>
        )}
        {dockTab === 'sentinel' && (
          <div className="absolute inset-0 flex flex-col bg-surface">
            <GroundTruthSentinelPanel />
          </div>
        )}

      </div>
    </aside>
  );
};

export default SceneDock;
