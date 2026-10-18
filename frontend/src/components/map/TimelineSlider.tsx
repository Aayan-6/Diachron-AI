import React, { useEffect, useState } from 'react';
import { Calendar, ChevronLeft, ChevronRight, Pause, Play, Sparkles } from 'lucide-react';
import type { TimelineObservation } from '../../lib/types';
import { cn } from '../../lib/utils';

export interface TimelineSliderProps {
  timeline: TimelineObservation[];
  currentStep: number;
  onStepChange: (step: number) => void;
  className?: string;
}

export const TimelineSlider: React.FC<TimelineSliderProps> = ({
  timeline,
  currentStep,
  onStepChange,
  className,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);

  // Playback timer
  useEffect(() => {
    if (!isPlaying || timeline.length <= 1) return;

    const interval = setInterval(() => {
      onStepChange((currentStep + 1) % timeline.length);
    }, 1600);

    return () => clearInterval(interval);
  }, [isPlaying, currentStep, timeline.length, onStepChange]);

  if (!timeline || timeline.length <= 1) return null;

  const activeObs = timeline[currentStep] || timeline[0];
  const isBaseline = activeObs.is_baseline || currentStep === 0;

  const handlePrev = () => {
    setIsPlaying(false);
    onStepChange(Math.max(0, currentStep - 1));
  };

  const handleNext = () => {
    setIsPlaying(false);
    onStepChange(Math.min(timeline.length - 1, currentStep + 1));
  };

  return (
    <div
      className={cn(
        'flex flex-col gap-2 rounded-2xl border border-white/10 bg-slate-950/85 p-3.5 shadow-2xl backdrop-blur-xl transition-all',
        className
      )}
    >
      {/* Header Info */}
      <div className="flex items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-emerald-500/20 text-emerald-400">
            <Calendar size={13} />
          </span>
          <div className="flex flex-col">
            <span className="font-semibold text-white">
              {isBaseline ? 'Baseline Reference (T0)' : `Observation Step T${activeObs.step}`}
            </span>
            <span className="font-mono text-[11px] text-slate-400">{activeObs.date}</span>
          </div>
        </div>

        {/* Change Stats */}
        <div className="flex items-center gap-2">
          {!isBaseline && (activeObs.change_area_ha ?? 0) > 0 && (
            <span className="rounded-full bg-amber-500/15 px-2.5 py-0.5 font-mono text-[11px] font-medium text-amber-300">
              +{(activeObs.change_area_ha ?? 0).toFixed(1)} ha ({(((activeObs.change_fraction ?? 0)) * 100).toFixed(1)}%)
            </span>
          )}
          {activeObs.new_breakpoints && activeObs.new_breakpoints > 0 ? (
            <span className="flex items-center gap-1 rounded-full bg-rose-500/20 px-2 py-0.5 font-mono text-[10px] font-semibold text-rose-300">
              <Sparkles size={10} />
              {activeObs.new_breakpoints} new
            </span>
          ) : null}
        </div>
      </div>

      {/* Scrub Slider */}
      <div className="relative py-1">
        <input
          type="range"
          min={0}
          max={timeline.length - 1}
          value={currentStep}
          onChange={(e) => {
            setIsPlaying(false);
            onStepChange(Number(e.target.value));
          }}
          className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-emerald-400 transition-colors focus:outline-none"
        />

        {/* Date Tick Marks */}
        <div className="mt-1 flex justify-between px-1">
          {timeline.map((obs, idx) => {
            const active = idx === currentStep;
            const past = idx <= currentStep;
            return (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  onStepChange(idx);
                }}
                className="group flex flex-col items-center gap-0.5 text-center focus:outline-none"
              >
                <span
                  className={cn(
                    'h-1.5 w-1.5 rounded-full transition-transform',
                    active
                      ? 'scale-150 bg-emerald-400 ring-2 ring-emerald-400/40'
                      : past
                      ? 'bg-slate-400'
                      : 'bg-slate-700 group-hover:bg-slate-500'
                  )}
                />
                <span
                  className={cn(
                    'font-mono text-[9px] transition-colors',
                    active ? 'font-bold text-emerald-300' : 'text-slate-500 group-hover:text-slate-300'
                  )}
                >
                  {obs.date.slice(5)}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Playback & Step Controls */}
      <div className="flex items-center justify-between border-t border-white/5 pt-2">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setIsPlaying(!isPlaying)}
            aria-label={isPlaying ? 'Pause timeline playback' : 'Play timeline animation'}
            className={cn(
              'flex h-7 items-center gap-1 rounded-lg px-2.5 text-[11px] font-medium transition-colors',
              isPlaying
                ? 'bg-amber-500/20 text-amber-300 hover:bg-amber-500/30'
                : 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30'
            )}
          >
            {isPlaying ? <Pause size={12} /> : <Play size={12} />}
            <span>{isPlaying ? 'Pause' : 'Play Timeline'}</span>
          </button>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handlePrev}
            disabled={currentStep === 0}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-slate-900 text-slate-300 transition-colors hover:bg-slate-800 disabled:opacity-40"
            aria-label="Previous observation"
          >
            <ChevronLeft size={14} />
          </button>
          <span className="font-mono text-[11px] text-slate-400 px-1">
            {currentStep + 1} / {timeline.length}
          </span>
          <button
            type="button"
            onClick={handleNext}
            disabled={currentStep === timeline.length - 1}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-slate-900 text-slate-300 transition-colors hover:bg-slate-800 disabled:opacity-40"
            aria-label="Next observation"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};
