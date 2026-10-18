import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUp, ImagePlus, Square, X, Mic, MicOff } from 'lucide-react';
import { useApp } from '../../context/AppState';
import { sensorKind } from '../../lib/trace';
import { cn } from '../../lib/utils';

const MAX_TEXTAREA_PX = 168;

const SENSOR_TONE: Record<string, string> = {
  sar: 'text-amber',
  optical: 'text-teal',
  fused: 'text-violet',
  unknown: 'text-ink-muted',
};

/** Chip showing which scene the next question will run against. */
const SceneChip: React.FC = () => {
  const { dataset, clearScene, openDock } = useApp();
  if (!dataset) return null;

  const tone = SENSOR_TONE[sensorKind(dataset.sensor)];

  return (
    <div className="mb-2 flex items-center gap-2 self-start rounded-lg border border-line bg-surface-2 py-1 pl-2 pr-1">
      <span className={cn('font-mono text-[10px] uppercase tracking-wider', tone)}>
        {dataset.mode === 'bi-temporal' ? 'T1/T2' : dataset.sensor.split(' ')[0]}
      </span>
      <button
        type="button"
        onClick={() => openDock('scene')}
        className="max-w-[22rem] truncate text-xs text-ink-muted transition-colors hover:text-ink cursor-pointer"
        title={dataset.name}
      >
        {dataset.t1Filename && dataset.t2Filename
          ? `${dataset.t1Filename} + ${dataset.t2Filename}`
          : dataset.name.includes(':::')
          ? dataset.name.split(':::').join(' + ')
          : dataset.name}
      </button>
      <button
        type="button"
        onClick={clearScene}
        aria-label="Remove active scene"
        className="rounded p-1 text-ink-faint transition-colors hover:bg-surface-4 hover:text-danger cursor-pointer"
      >
        <X size={12} />
      </button>
    </div>
  );
};

export const Composer: React.FC = () => {
  const { sendQuery, cancelQuery, isBusy, dataset, openDock, backendPhase } = useApp();
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  // Grow with the content up to a ceiling, then scroll internally.
  const resize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_PX)}px`;
  }, []);

  useEffect(resize, [value, resize]);

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      
      recognition.onresult = (event: any) => {
        let transcript = '';
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        setValue(transcript);
        requestAnimationFrame(resize);
      };

      recognition.onend = () => {
        setIsListening(false);
      };
      
      recognition.onerror = () => {
        setIsListening(false);
      };
      
      recognitionRef.current = recognition;
    }
  }, [resize]);

  const toggleMic = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in your browser (try Chrome/Edge).");
      return;
    }
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      setValue('');
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  const submit = () => {
    const text = value.trim();
    if (!text || isBusy) return;
    
    // Stop listening if sending
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
    
    sendQuery(text);
    setValue('');
    requestAnimationFrame(resize);
  };

  const offline = backendPhase === 'offline';

  return (
    <div className="border-t border-line bg-ground/85 px-4 pb-4 pt-3 backdrop-blur-sm">
      <div className="mx-auto flex w-full max-w-[46rem] flex-col">
        <SceneChip />

        <form
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
          className={cn(
            'flex items-end gap-2 rounded-2xl border bg-surface-2 p-2 transition-colors',
            'focus-within:border-accent/45 focus-within:bg-surface-2',
            offline ? 'border-danger/30' : 'border-line-strong'
          )}
        >
          <button
            type="button"
            onClick={() => openDock('scene')}
            aria-label="Add or change imagery"
            title="Add or change imagery"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-ink-faint transition-colors hover:bg-surface-3 hover:text-accent cursor-pointer"
          >
            <ImagePlus size={17} />
          </button>

          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder={
              dataset
                ? `Analyze change, pinpoint onset date, or inspect features in ${dataset.name}…`
                : 'Search archive by semantic description or analyze multi-temporal changes…'
            }
            className="scrollbar-slim max-h-[168px] min-h-[36px] flex-1 resize-none bg-transparent py-2 text-[14px] leading-relaxed text-ink outline-none placeholder:text-ink-faint"
          />

          {isBusy ? (
            <button
              type="button"
              onClick={cancelQuery}
              aria-label="Stop generating"
              title="Stop"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-line-strong bg-surface-3 text-ink transition-colors hover:border-danger/40 hover:text-danger cursor-pointer"
            >
              <Square size={13} fill="currentColor" />
            </button>
          ) : (
            <div className="flex gap-1 shrink-0">
              <button
                type="button"
                onClick={toggleMic}
                aria-label={isListening ? "Stop listening" : "Start voice input"}
                title="Voice Input"
                className={cn(
                  "flex h-9 w-9 items-center justify-center rounded-xl transition-all cursor-pointer",
                  isListening 
                    ? "bg-danger text-white animate-pulse" 
                    : "text-ink-faint hover:bg-surface-3 hover:text-accent"
                )}
              >
                {isListening ? <MicOff size={17} /> : <Mic size={17} />}
              </button>
              <button
                type="submit"
                disabled={!value.trim()}
                aria-label="Send query"
                className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent text-space-black transition-all hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-surface-4 disabled:text-ink-faint cursor-pointer"
              >
                <ArrowUp size={17} strokeWidth={2.5} />
              </button>
            </div>
          )}
        </form>

        <p className="mt-2 px-1 text-center font-mono text-[10px] text-ink-faint">
          {offline
            ? 'Backend unreachable — start the FastAPI server to run queries.'
            : 'Enter to send · Shift+Enter for a new line · Mic for voice'}
        </p>
      </div>
    </div>
  );
};

export default Composer;
