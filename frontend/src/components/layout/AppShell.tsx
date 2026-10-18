import React, { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Activity, Globe2, History, LogOut, Map as MapIcon, MessagesSquare, Settings, User } from 'lucide-react';
import HistoryPanel from '../history/HistoryPanel';
import { SystemTelemetryModal } from '../scene/SystemTelemetryModal';
import { cn } from '../../lib/utils';
import { useApp } from '../../context/AppState';
import { describeCompute } from '../../lib/api';

interface RailItem {
  to: string;
  label: string;
  icon: React.ReactNode;
}

const PRIMARY: RailItem[] = [
  { to: '/console', label: 'Workspace', icon: <MessagesSquare size={18} /> },
  { to: '/map', label: 'Full map', icon: <MapIcon size={18} /> },
];

const SECONDARY: RailItem[] = [
  { to: '/profile', label: 'Profile', icon: <User size={18} /> },
  { to: '/settings', label: 'Settings', icon: <Settings size={18} /> },
];

/** Shared hover label — the rail is icon-only to keep chat + map wide. */
const RailTooltip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span
    role="tooltip"
    className="pointer-events-none absolute left-[calc(100%+10px)] z-50 whitespace-nowrap rounded-md border border-line-strong bg-surface-3 px-2 py-1 text-xs text-ink opacity-0 shadow-xl transition-opacity group-hover:opacity-100"
  >
    {children}
  </span>
);

const RailLink: React.FC<{ item: RailItem }> = ({ item }) => (
  <NavLink
    to={item.to}
    className={({ isActive }) =>
      cn(
        'group relative flex h-10 w-10 items-center justify-center rounded-xl transition-colors',
        isActive ? 'bg-accent/15 text-accent' : 'text-ink-faint hover:bg-surface-3 hover:text-ink'
      )
    }
  >
    {({ isActive }) => (
      <>
        {item.icon}
        {isActive && (
          <span className="absolute -left-2 h-5 w-[3px] rounded-r-full bg-accent" aria-hidden />
        )}
        <RailTooltip>{item.label}</RailTooltip>
      </>
    )}
  </NavLink>
);

/** Small live indicator for backend reachability. */
const BackendDot: React.FC = () => {
  const { backendPhase, backendStatus } = useApp();

  const tone =
    backendPhase === 'online'
      ? 'bg-ok'
      : backendPhase === 'checking'
        ? 'bg-amber animate-pulse'
        : 'bg-danger';

  const label =
    backendPhase === 'online'
      ? `Backend online · ${describeCompute(backendStatus)}`
      : backendPhase === 'checking'
        ? 'Contacting backend…'
        : 'Backend unreachable';

  return (
    <div className="group relative flex h-10 w-10 items-center justify-center">
      <span className={cn('h-2 w-2 rounded-full', tone)} aria-hidden />
      <span className="sr-only">{label}</span>
      <RailTooltip>{label}</RailTooltip>
    </div>
  );
};

/**
 * Application frame: a narrow icon rail, an optional conversation history
 * rail, then the routed surface. History is a panel rather than a page so the
 * list and the conversation it points at are on screen together.
 */
const AppShell: React.FC = () => {
  const { historyOpen, setHistoryOpen } = useApp();
  const [telemetryOpen, setTelemetryOpen] = useState(false);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-ground text-ink">
      <nav
        aria-label="Primary"
        className="flex w-[60px] shrink-0 flex-col items-center gap-1 border-r border-line bg-surface py-3"
      >
        <NavLink
          to="/"
          aria-label="Diachron AI home"
          className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl border border-accent/25 bg-accent/10 text-accent transition-colors hover:bg-accent/20"
        >
          <Globe2 size={19} />
        </NavLink>

        {PRIMARY.map((item) => (
          <RailLink key={item.to} item={item} />
        ))}

        <button
          type="button"
          aria-label="Conversation history"
          aria-pressed={historyOpen}
          onClick={() => setHistoryOpen(!historyOpen)}
          className={cn(
            'group relative flex h-10 w-10 cursor-pointer items-center justify-center rounded-xl transition-colors',
            historyOpen ? 'bg-accent/15 text-accent' : 'text-ink-faint hover:bg-surface-3 hover:text-ink'
          )}
        >
          <History size={18} />
          {historyOpen && (
            <span className="absolute -left-2 h-5 w-[3px] rounded-r-full bg-accent" aria-hidden />
          )}
          <RailTooltip>{historyOpen ? 'Hide history' : 'History'}</RailTooltip>
        </button>

        <div className="mt-auto flex flex-col items-center gap-1">
          <button
            type="button"
            aria-label="System Telemetry (§2.3)"
            onClick={() => setTelemetryOpen(true)}
            className="group relative flex h-10 w-10 cursor-pointer items-center justify-center rounded-xl text-ink-faint transition-colors hover:bg-accent/15 hover:text-accent"
          >
            <Activity size={18} />
            <RailTooltip>System Telemetry (§2.3)</RailTooltip>
          </button>
          <BackendDot />
          <div className="my-1 h-px w-6 bg-line" aria-hidden />
          {SECONDARY.map((item) => (
            <RailLink key={item.to} item={item} />
          ))}
          <NavLink
            to="/"
            aria-label="Sign out"
            className="group relative flex h-10 w-10 items-center justify-center rounded-xl text-ink-faint transition-colors hover:bg-danger/10 hover:text-danger"
          >
            <LogOut size={18} />
            <RailTooltip>Sign out</RailTooltip>
          </NavLink>
        </div>
      </nav>

      {historyOpen && <HistoryPanel />}

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Outlet />
      </main>

      <SystemTelemetryModal isOpen={telemetryOpen} onClose={() => setTelemetryOpen(false)} />
    </div>
  );
};

export default AppShell;
