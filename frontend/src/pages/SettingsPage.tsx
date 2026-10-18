import React from 'react';
import { CheckCircle2, Cpu, Network, RefreshCcw, Server, XCircle } from 'lucide-react';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { InlineAlert } from '../components/ui/Feedback';
import { useApp } from '../context/AppState';
import { API_BASE_URL, describeCompute } from '../lib/api';

const Field: React.FC<{ label: string; value: React.ReactNode; mono?: boolean }> = ({
  label,
  value,
  mono = true,
}) => (
  <div className="min-w-0">
    <dt className="label-caps text-ink-faint">{label}</dt>
    <dd className={`mt-1 break-words text-[12.5px] text-ink ${mono ? 'font-mono' : ''}`}>{value}</dd>
  </div>
);

const SettingsPage: React.FC = () => {
  const { backendPhase, backendStatus, backendError, refreshBackend } = useApp();

  return (
    <div className="scrollbar-slim min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl space-y-5 px-6 py-8">
        <header>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Settings</h1>
          <p className="mt-1 text-[13px] text-ink-muted">
            Connection and compute details for the FastAPI backend serving this session.
          </p>
        </header>

        {/* Connection */}
        <section className="rounded-panel border border-line bg-surface p-5">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Server size={15} className="text-accent" aria-hidden />
              <h2 className="text-sm font-medium text-ink">Backend connection</h2>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={refreshBackend}
              isLoading={backendPhase === 'checking'}
            >
              {backendPhase !== 'checking' && <RefreshCcw size={13} />}
              Re-check
            </Button>
          </div>

          <dl className="grid gap-4 sm:grid-cols-2">
            <Field label="API base" value={API_BASE_URL} />
            <Field
              label="Status"
              value={
                backendPhase === 'online' ? (
                  <span className="inline-flex items-center gap-1.5 text-ok">
                    <CheckCircle2 size={13} aria-hidden />
                    online
                  </span>
                ) : backendPhase === 'checking' ? (
                  <span className="text-amber">checking…</span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 text-danger">
                    <XCircle size={13} aria-hidden />
                    unreachable
                  </span>
                )
              }
            />
          </dl>

          {backendPhase === 'offline' && (
            <InlineAlert tone="error" title="Cannot reach the backend" className="mt-4">
              {backendError ??
                'No response from the API. Start the FastAPI server, then re-check.'}
              <pre className="mt-2 overflow-x-auto rounded-md border border-line bg-ground px-2.5 py-2 font-mono text-[11px] text-ink-muted">
                cd backend/SatQuery-master/backend{'\n'}python main.py
              </pre>
            </InlineAlert>
          )}
        </section>

        {/* Compute */}
        {backendStatus && (
          <section className="rounded-panel border border-line bg-surface p-5">
            <div className="mb-4 flex items-center gap-2">
              <Cpu size={15} className="text-accent" aria-hidden />
              <h2 className="text-sm font-medium text-ink">Inference engine</h2>
              <Badge variant="info" className="ml-auto">
                {backendStatus.execution_mode?.replace(/_/g, ' ')}
              </Badge>
            </div>

            <dl className="grid gap-4 sm:grid-cols-2">
              <Field label="Model" value={backendStatus.model_engine.model_name ?? 'Qwen3-VL-2B-Diachron'} />
              <Field
                label="Trained Modalities"
                value={
                  backendStatus.model_engine.modalities_supported?.join(' · ') ??
                  'Sentinel-1 (SAR) · Sentinel-2 (Optical)'
                }
              />
              <Field label="Compute" value={describeCompute(backendStatus)} />
              <Field
                label="CUDA available"
                value={backendStatus.model_engine.local_cuda_available ? 'yes' : 'no'}
              />
              {backendStatus.model_engine.remote_service_url && (
                <>
                  <Field
                    label="Remote host"
                    value={backendStatus.model_engine.remote_service_url}
                  />
                  <Field
                    label="Remote health"
                    value={backendStatus.model_engine.remote_host_status ?? '—'}
                  />
                </>
              )}
            </dl>
          </section>
        )}

        {/* Specialists + network */}
        {backendStatus && (
          <section className="rounded-panel border border-line bg-surface p-5">
            <div className="mb-4 flex items-center gap-2">
              <Network size={15} className="text-accent" aria-hidden />
              <h2 className="text-sm font-medium text-ink">Specialists &amp; network</h2>
            </div>

            <ul className="mb-4 space-y-1.5">
              {Object.entries(backendStatus.specialists ?? {}).map(([key, value]) => (
                <li
                  key={key}
                  className="flex items-center justify-between gap-3 rounded-lg border border-line bg-surface-2 px-3 py-2"
                >
                  <span className="font-mono text-[11.5px] text-ink-muted">
                    {key.replace(/_/g, ' ')}
                  </span>
                  <span className="truncate text-right font-mono text-[11px] text-ink">{value}</span>
                </li>
              ))}
            </ul>

            <dl className="grid gap-4 sm:grid-cols-3">
              <Field label="LAN IP" value={backendStatus.network?.lan_ip ?? '—'} />
              <Field label="Port" value={backendStatus.network?.port ?? '—'} />
              <Field label="Service" value={backendStatus.service} />
            </dl>

            <p className="mt-4 border-t border-line pt-3 font-mono text-[10.5px] leading-relaxed text-ink-faint">
              Requests go to a same-origin path by default so Vite&rsquo;s proxy can
              forward them; set VITE_API_BASE_URL to target another host.
            </p>
          </section>
        )}
      </div>
    </div>
  );
};

export default SettingsPage;
