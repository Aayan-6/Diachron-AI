import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Bot,
  ArrowRight,
  X,
  CheckCircle2,
  Calendar,
} from 'lucide-react';
import { useApp } from '../../../context/AppState';
import {
  type FetchAOIResponse,
  type FetchBiTemporalAOIResponse,
  aoiResponseToSceneDataset,
  aoiResponseToSceneOverlay,
  aoiPairResponseToSceneDataset,
  aoiPairResponseToSceneOverlay,
} from '../../../lib/interactiveMapApi';

interface AOIChatbotHandoffProps {
  scene: FetchAOIResponse | FetchBiTemporalAOIResponse | null;
  onDismiss: () => void;
  onRunChangeFormer?: (scene: FetchBiTemporalAOIResponse) => void;
}

export const AOIChatbotHandoff: React.FC<AOIChatbotHandoffProps> = ({
  scene,
  onDismiss,
  onRunChangeFormer,
}) => {
  const navigate = useNavigate();
  const { setScene, openDock, sendQuery } = useApp();

  if (!scene) return null;

  const isBiTemporal = 't2_image_url' in scene;

  const handleHandoff = () => {
    if (isBiTemporal) {
      const bitemp = scene as FetchBiTemporalAOIResponse;
      const dataset = aoiPairResponseToSceneDataset(bitemp);
      const overlay = aoiPairResponseToSceneOverlay(bitemp);
      setScene(dataset, overlay);
    } else {
      const single = scene as FetchAOIResponse;
      const dataset = aoiResponseToSceneDataset(single);
      const overlay = aoiResponseToSceneOverlay(single);
      setScene(dataset, overlay);
    }

    // Switch to Workspace view for conversation with the AI model
    navigate('/console');
    openDock('map');
  };

  const handleChangeFormerAction = () => {
    if (isBiTemporal) {
      const bitemp = scene as FetchBiTemporalAOIResponse;
      if (onRunChangeFormer) {
        onRunChangeFormer(bitemp);
      } else {
        const dataset = aoiPairResponseToSceneDataset(bitemp);
        const overlay = aoiPairResponseToSceneOverlay(bitemp);
        setScene(dataset, overlay);
        navigate('/console');
        openDock('map');
        sendQuery(
          `Run ChangeFormer change detection on this bi-temporal pair (${bitemp.name}) to identify significant surface changes between ${bitemp.t1?.acquisition_date || 'T1'} and ${bitemp.t2?.acquisition_date || 'T2'}.`
        );
      }
    }
  };

  const formattedDate = (() => {
    if (isBiTemporal) {
      const bitemp = scene as FetchBiTemporalAOIResponse;
      const fmt = (dStr?: string) => {
        if (!dStr) return 'Pass';
        const d = new Date(dStr);
        return isNaN(d.getTime()) ? dStr : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
      };
      return `${fmt(bitemp.t1?.acquisition_date)} → ${fmt(bitemp.t2?.acquisition_date)}`;
    }
    const single = scene as FetchAOIResponse;
    if (!single.acquisition_date) return 'Recent Pass';
    const d = new Date(single.acquisition_date);
    return isNaN(d.getTime())
      ? 'Recent Pass'
      : d.toLocaleDateString(undefined, {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
        });
  })();

  const cloudCoverValue = (() => {
    if (isBiTemporal) {
      const bitemp = scene as FetchBiTemporalAOIResponse;
      return typeof bitemp.t1?.cloud_cover === 'number' ? bitemp.t1.cloud_cover : null;
    }
    const single = scene as FetchAOIResponse;
    return typeof single.cloud_cover === 'number' ? single.cloud_cover : null;
  })();

  return (
    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-40 w-[92%] max-w-lg animate-in slide-in-from-bottom-4 duration-300 pointer-events-auto">
      <div className="rounded-2xl border border-accent/40 bg-surface/95 p-3.5 shadow-2xl backdrop-blur-xl text-ink font-sans">
        <div className="flex items-start justify-between gap-3">
          {/* Thumbnail Preview */}
          <div className="relative h-16 w-16 shrink-0 overflow-hidden rounded-xl border border-line bg-surface-3">
            <img
              src={scene.t1_image_url}
              alt="Fetched Scene Preview"
              className="h-full w-full object-cover"
              onError={(e) => {
                // Fallback to satellite icon if preview not yet rendered
                (e.currentTarget as HTMLElement).style.display = 'none';
              }}
            />
            <div className="absolute bottom-1 right-1 rounded bg-space-black/70 px-1 py-0.2 text-[8px] font-mono text-accent">
              10m
            </div>
          </div>

          {/* Info Details */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-1">
              <span className="flex items-center gap-1 text-[11px] font-semibold text-accent font-mono">
                <CheckCircle2 size={12} />
                {isBiTemporal ? 'BI-TEMPORAL PAIR READY' : 'SCENE STREAMED'}
              </span>
              <span className="text-line">•</span>
              <span className="text-[11px] text-ink-muted truncate font-mono">
                {scene.sensor ? scene.sensor.split(' ')[0] : 'Scene'}
              </span>
            </div>

            <div className="text-xs font-semibold text-ink truncate mb-1">
              {scene.name}
            </div>

            <div className="flex items-center gap-3 text-[10px] font-mono text-ink-faint">
              <span className="flex items-center gap-1">
                <Calendar size={11} />
                {formattedDate}
              </span>
              <span>{scene.area_sq_km} km²</span>
              {typeof cloudCoverValue === 'number' && !isNaN(cloudCoverValue) && (
                <span>Clouds: {cloudCoverValue.toFixed(1)}%</span>
              )}
            </div>
          </div>

          {/* Close button */}
          <button
            type="button"
            onClick={onDismiss}
            className="rounded p-1 text-ink-faint hover:text-ink transition-colors cursor-pointer"
            title="Dismiss notification"
          >
            <X size={14} />
          </button>
        </div>

        {/* Action Buttons */}
        <div className="mt-3 pt-2.5 border-t border-line flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2">
          <span className="text-[11px] text-ink-muted hidden sm:inline">
            {isBiTemporal ? 'Compare on map or detect changes:' : 'Load this real scene into the AI assistant:'}
          </span>
          <div className="flex items-center gap-2">
            {isBiTemporal && (
              <button
                type="button"
                onClick={handleChangeFormerAction}
                className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-1.5 rounded-xl bg-surface-3 border border-accent/40 px-3 py-2 text-xs font-semibold text-accent hover:bg-surface-4 transition-all cursor-pointer shadow-sm"
              >
                <span>Run ChangeFormer V6</span>
                <ArrowRight size={13} />
              </button>
            )}
            <button
              type="button"
              onClick={handleHandoff}
              className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-4 py-2 text-xs font-semibold text-space-black shadow-[0_0_15px_rgba(0,242,255,0.3)] hover:bg-accent/90 transition-all cursor-pointer"
            >
              <Bot size={14} />
              <span>{isBiTemporal ? 'Workspace Console' : 'Analyze with Diachron AI'}</span>
              <ArrowRight size={13} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
