import React, { useEffect, useRef } from 'react';
import { Compass, GitCompareArrows, ScanSearch, Sparkles } from 'lucide-react';
import { MessageItem } from './MessageItem';
import { useApp } from '../../context/AppState';
import { cn } from '../../lib/utils';

interface StarterGroup {
  id: string;
  icon: React.ReactNode;
  title: string;
  tone: string;
  prompts: string[];
  /** Change detection needs two acquisitions, so it is gated on a T1/T2 pair. */
  biTemporalOnly?: boolean;
}

/**
 * Starter prompts, grouped by the three specialists the backend routes to
 * (see core/agent.py intent classification).
 */
const STARTERS: StarterGroup[] = [
  {
    id: 'semantic_retrieval',
    icon: <ScanSearch size={15} />,
    title: 'Semantic Archive Search',
    tone: 'text-teal',
    prompts: [
      'Search for newly built structures near a river.',
      'Find large vehicle concentrations on open ground.',
      'Where are dense urban settlements across the archive?',
    ],
  },
  {
    id: 'multitemporal_change',
    icon: <GitCompareArrows size={15} />,
    title: 'Multi-Temporal Change',
    tone: 'text-amber',
    prompts: [
      'When did change first appear in this area?',
      'Detect construction and land clearance between 2018 and 2026.',
      'Show me where water bodies expanded or dried up.',
    ],
  },
  {
    id: 'trajectory_taxonomy',
    icon: <Compass size={15} />,
    title: 'Trajectory & Taxonomy',
    tone: 'text-violet',
    prompts: [
      'Classify detected changes into infrastructure vs land clearance.',
      'Trace chronological trajectory across all historical passes.',
      'Pinpoint earliest onset date with verified usable imagery.',
    ],
  },
];

const EmptyThread: React.FC = () => {
  const { sendQuery, dataset, openDock, backendPhase } = useApp();
  const biTemporal = dataset?.mode === 'bi-temporal';

  return (
    <div className="flex min-h-full flex-col items-center justify-center px-4 py-10">
      <div className="w-full max-w-[46rem]">
        <div className="mb-7 text-center">
          <h1 className="text-[26px] font-semibold tracking-tight text-ink">
            What should I look at?
          </h1>
          <p className="mx-auto mt-2 max-w-md text-[13.5px] leading-relaxed text-ink-muted">
            {dataset
              ? 'Ask in plain language. The agentic controller picks the specialist and puts its evidence on the map beside you.'
              : 'Attach a Sentinel scene or pick a demo one, then ask in plain language.'}
          </p>
          {!dataset && (
            <button
              type="button"
              onClick={() => openDock('scene')}
              className="mt-4 inline-flex items-center gap-2 rounded-lg border border-accent/30 bg-accent/10 px-3 py-2 text-[13px] font-medium text-accent transition-colors hover:bg-accent/15 cursor-pointer"
            >
              <Sparkles size={14} />
              Choose imagery
            </button>
          )}
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          {STARTERS.map((group) => {
            const disabled =
              backendPhase === 'offline' || (group.biTemporalOnly && !biTemporal);
            return (
              <div
                key={group.id}
                className={cn(
                  'rounded-panel border border-line bg-surface p-3 transition-opacity',
                  disabled && 'opacity-45'
                )}
              >
                <div className="mb-2.5 flex items-center gap-2">
                  <span className={group.tone} aria-hidden>
                    {group.icon}
                  </span>
                  <h2 className="text-xs font-medium text-ink">{group.title}</h2>
                </div>
                <ul className="space-y-1">
                  {group.prompts.map((prompt) => (
                    <li key={prompt}>
                      <button
                        type="button"
                        disabled={disabled}
                        onClick={() => sendQuery(prompt)}
                        className="w-full rounded-lg px-2 py-1.5 text-left text-[12.5px] leading-snug text-ink-muted transition-colors hover:bg-surface-3 hover:text-ink disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-ink-muted cursor-pointer"
                      >
                        {prompt}
                      </button>
                    </li>
                  ))}
                </ul>
                {group.biTemporalOnly && !biTemporal && (
                  <p className="mt-2 border-t border-line pt-2 font-mono text-[10px] text-ink-faint">
                    Needs a bi-temporal pair
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export const ChatThread: React.FC = () => {
  const { messages, isBusy } = useApp();
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  // Only auto-scroll when the reader is already at the bottom, so scrolling up
  // to re-read an earlier answer is not yanked back by a streaming reply.
  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  const prevMessagesLength = useRef(messages.length);

  useEffect(() => {
    const isNewMessage = messages.length > prevMessagesLength.current;
    prevMessagesLength.current = messages.length;

    if (isNewMessage || pinnedRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
      pinnedRef.current = true;
    }
  }, [messages, isBusy]);

  return (
    <div
      ref={scrollRef}
      onScroll={handleScroll}
      className="scrollbar-slim min-h-0 flex-1 overflow-y-auto"
    >
      {messages.length === 0 ? (
        <EmptyThread />
      ) : (
        <div className="mx-auto w-full max-w-[46rem] space-y-6 px-4 py-6">
          {messages.map((message) => (
            <MessageItem key={message.id} message={message} />
          ))}
          <div ref={bottomRef} className="h-2" />
        </div>
      )}
    </div>
  );
};

export default ChatThread;
