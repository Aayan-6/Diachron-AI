import React, { useState } from 'react';
import { Activity, Download, PanelRight, RefreshCcw, SquarePen } from 'lucide-react';
import ChatThread from '../components/chat/ChatThread';
import Composer from '../components/chat/Composer';
import SceneDock from '../components/scene/SceneDock';
import { SystemTelemetryModal } from '../components/scene/SystemTelemetryModal';
import { Badge } from '../components/ui/Badge';
import { IconButton } from '../components/ui/Button';
import { LanguageSelector } from '../components/ui/LanguageSelector';
import { useApp } from '../context/AppState';
import { describeCompute } from '../lib/api';
import { downloadGeoJsonReport } from '../lib/report';

const BackendBadge: React.FC = () => {
  const { backendPhase, backendStatus, backendError, refreshBackend } = useApp();

  if (backendPhase === 'checking') {
    return <Badge variant="quiet" dot>connecting</Badge>;
  }

  if (backendPhase === 'offline') {
    return (
      <button
        type="button"
        onClick={refreshBackend}
        title={backendError ?? 'Backend unreachable'}
        className="cursor-pointer"
      >
        <Badge variant="error" dot>
          backend offline
        </Badge>
      </button>
    );
  }

  return (
    <Badge variant="success" dot title={describeCompute(backendStatus)}>
      {backendStatus?.execution_mode?.replace(/_/g, ' ') ?? 'online'}
    </Badge>
  );
};

/**
 * The product's centre of gravity: a conversation, with the map and the
 * execution trace as companions on the right rather than as the main event.
 * The thread is width-capped so it stays readable and never sprawls across a
 * wide display, even with the dock collapsed.
 */
const Workspace: React.FC = () => {
  const { dockOpen, setDockOpen, resetConversation, messages, refreshBackend, backendPhase } =
    useApp();
  const [telemetryOpen, setTelemetryOpen] = useState(false);

  const handlePrint = () => {
    // Determine a title for the report based on the first message
    const title = messages.find(m => m.role === 'user')?.content.substring(0, 60) ?? 'Chat Session';
    downloadGeoJsonReport(title, messages);
  };

  return (
    <div className="flex h-full min-h-0">
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-line bg-surface/60 px-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <h1 className="text-[13px] font-medium text-ink">Diachron AI workspace</h1>
            <BackendBadge />
            <button
              type="button"
              onClick={() => setTelemetryOpen(true)}
              className="flex items-center gap-1.5 px-2.5 py-0.5 text-[11px] font-medium rounded-full border border-accent/30 bg-accent/10 text-accent hover:bg-accent/20 transition-all cursor-pointer shadow-sm ml-1"
              title="Open ISRO PS 26167 §2.3 System Telemetry & Evaluation Dashboard"
            >
              <Activity size={12} className="text-accent" />
              <span>Telemetry (§2.3)</span>
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <LanguageSelector />
            
            {backendPhase === 'offline' && (
              <IconButton label="Retry backend connection" size="sm" onClick={refreshBackend}>
                <RefreshCcw size={13} />
              </IconButton>
            )}
            <IconButton
              label="Download GeoJSON report"
              size="sm"
              onClick={handlePrint}
              disabled={messages.length === 0}
            >
              <Download size={13} />
            </IconButton>
            <IconButton
              label="New conversation"
              size="sm"
              onClick={resetConversation}
              disabled={messages.length === 0}
            >
              <SquarePen size={13} />
            </IconButton>
            <IconButton
              label={dockOpen ? 'Hide scene panel' : 'Show scene panel'}
              size="sm"
              active={dockOpen}
              onClick={() => setDockOpen(!dockOpen)}
            >
              <PanelRight size={13} />
            </IconButton>
          </div>
        </header>

        <ChatThread />
        <Composer />
      </section>

      <SceneDock />
      <SystemTelemetryModal isOpen={telemetryOpen} onClose={() => setTelemetryOpen(false)} />
    </div>
  );
};

export default Workspace;
