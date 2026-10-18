import React, { useRef, useState } from 'react';
import {
  CheckCircle2,
  FileImage,
  Images,
  Loader2,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { Badge } from '../ui/Badge';
import { InlineAlert } from '../ui/Feedback';
import { presetToDataset, presetToOverlay, useApp } from '../../context/AppState';
import { ApiError, resolveAssetUrl, uploadImagery } from '../../lib/api';
import { formatBytes, sensorKind } from '../../lib/trace';
import type { SceneDataset, SceneOverlay } from '../../lib/types';
import { cn } from '../../lib/utils';

const ACCEPTED = '.tif,.tiff,.geotiff,.png,.jpg,.jpeg';

const SENSOR_TONE: Record<string, 'info' | 'warning' | 'violet' | 'default'> = {
  optical: 'info',
  sar: 'warning',
  fused: 'violet',
  unknown: 'default',
};

/** Local preview used while the backend is still ingesting the raster. */
function buildLocalOverlay(files: File[]): SceneOverlay {
  const bounds: [number, number, number, number] = [77.618, 13.022, 77.652, 13.048];
  const mode = files.length > 2 ? 'multi-temporal' : files.length === 2 ? 'bi-temporal' : 'single';
  return {
    name: files[0].name,
    sensor: 'Optical (pending ingest)',
    mode,
    bounds,
    center: [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2],
    crs: 'Unknown until ingested',
    resolution: '—',
    t1ImageUrl: URL.createObjectURL(files[0]),
    t2ImageUrl: files[1] ? URL.createObjectURL(files[1]) : undefined,
    t1Filename: files[0].name,
    t2Filename: files[1]?.name,
    scenes: files.map((f, idx) => ({
      step_index: idx,
      id: `local-${idx}`,
      filename: f.name,
      date: `T${idx + 1}`,
      image_url: URL.createObjectURL(f),
      imageUrl: URL.createObjectURL(f),
      sensor: 'Optical',
    })),
  };
}

export const ScenePanel: React.FC = () => {
  const { dataset, overlay, setScene, clearScene, presets, presetsLoading, backendPhase, activeSiteId, setActiveSiteId } = useApp();
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const ingest = async (fileList: FileList | File[]) => {
    const files = Array.from(fileList).slice(0, 10);
    if (files.length === 0) return;

    setError(null);
    setNotice(null);
    setBusy(true);

    // Show something immediately; GeoTIFFs cannot be previewed in a browser, so
    // those wait for the server-rendered RGB thumbnail.
    const isTiff = /\.tiff?$/i.test(files[0].name);
    const mode = files.length > 2 ? 'multi-temporal' : files.length === 2 ? 'bi-temporal' : 'single';
    const provisional: SceneDataset = {
      name: files.length > 1 ? files.map((f) => f.name).join(':::') : files[0].name,
      sizeLabel: formatBytes(files.reduce((sum, f) => sum + f.size, 0)),
      sensor: 'Pending ingest',
      mode,
      crs: '—',
      resolution: '—',
      files,
      syncedWithBackend: false,
      georeferenced: false,
      t1Filename: files[0].name,
      t2Filename: files[1]?.name,
      scenes: files.map((f, idx) => ({
        step_index: idx,
        id: `local-${idx}`,
        filename: f.name,
        date: `T${idx + 1}`,
        image_url: URL.createObjectURL(f),
        imageUrl: URL.createObjectURL(f),
        sensor: 'Optical',
      })),
    };
    setScene(provisional, isTiff ? null : buildLocalOverlay(files));

    try {
      const response = await uploadImagery(files);

      // The backend only returns bounds when the raster actually carries a
      // CRS. Without one there is no honest place to put it on the map, so the
      // scene is still usable for questions but gets no overlay.
      const placeable = Boolean(response.wgs84_bounds && response.center);
      if (!placeable) {
        setNotice(
          'This file has no coordinate reference system, so it cannot be placed on the map. Questions about it will still work.'
        );
      }

      const resolvedScenes = response.scenes?.map((s) => ({
        step_index: s.step_index,
        id: `scene-${s.step_index}`,
        filename: s.filename,
        date: s.filename,
        image_url: resolveAssetUrl(s.image_url) ?? s.image_url,
        imageUrl: resolveAssetUrl(s.image_url) ?? s.image_url,
        sensor: response.sensor,
      }));

      setScene(
        {
          name: response.name,
          sizeLabel: formatBytes(files.reduce((sum, f) => sum + f.size, 0)),
          sensor: response.sensor,
          mode: response.mode,
          crs: response.crs,
          resolution: response.resolution,
          files,
          // The server now holds these bytes under `name`, so later turns can
          // reference the scene without re-uploading it.
          syncedWithBackend: true,
          georeferenced: placeable,
          areaSqKm: response.area_sq_km,
          datasetId: response.dataset_id,
          t1Filename: response.t1_filename || files[0].name,
          t2Filename: response.t2_filename || files[1]?.name,
          bandContract: response.band_contract,
          scenes: resolvedScenes,
        },
        placeable
          ? {
              datasetId: response.dataset_id,
              name: response.name,
              sensor: response.sensor,
              mode: response.mode,
              bounds: response.wgs84_bounds as [number, number, number, number],
              center: response.center as [number, number],
              crs: response.crs,
              resolution: response.resolution,
              areaSqKm: response.area_sq_km,
              t1ImageUrl: resolveAssetUrl(response.t1_image_url) ?? response.t1_image_url,
              t2ImageUrl: resolveAssetUrl(response.t2_image_url),
              t1Filename: response.t1_filename,
              t2Filename: response.t2_filename,
              bandContract: response.band_contract,
              t1NirImageUrl: resolveAssetUrl(response.t1_nir_image_url),
              t2NirImageUrl: resolveAssetUrl(response.t2_nir_image_url),
              scenes: resolvedScenes,
            }
          : null
      );
    } catch (err) {
      // The scene stays selected: the query endpoint accepts raw files too, so
      // the user can still ask questions even when ingestion failed.
      const detail = err instanceof ApiError ? err.detail || err.message : String(err);
      setError(`Server-side ingest failed — ${detail} Queries will upload the file directly instead.`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="scrollbar-slim flex-1 overflow-y-auto p-3.5">
      {/* Dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files?.length) void ingest(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload satellite imagery"
        className={cn(
          'flex cursor-pointer flex-col items-center gap-2 rounded-xl border border-dashed px-4 py-6 text-center transition-colors',
          dragging
            ? 'border-accent bg-accent/10'
            : 'border-line-strong bg-surface/60 hover:border-accent/40 hover:bg-surface-2'
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) void ingest(e.target.files);
            e.target.value = '';
          }}
        />
        <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-line bg-surface-3 text-accent">
          {busy ? <Loader2 size={17} className="animate-spin" /> : <UploadCloud size={17} />}
        </span>
        <span className="text-[13px] font-medium text-ink">
          {busy ? 'Ingesting raster…' : 'Drop GeoTIFFs, pair, or temporal series'}
        </span>
        <span className="font-mono text-[10px] text-ink-faint">
          GeoTIFF · COG · PNG · JPEG — up to 10 time series files
        </span>
      </div>

      {error && (
        <InlineAlert tone="warning" title="Ingest incomplete" className="mt-3">
          {error}
        </InlineAlert>
      )}

      {notice && (
        <InlineAlert tone="info" title="Not georeferenced" className="mt-3">
          {notice}
        </InlineAlert>
      )}

      {/* Active scene */}
      {dataset && (
        <div className="mt-3 rounded-xl border border-line bg-surface p-3">
          <div className="flex items-start justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2">
              <FileImage size={15} className="shrink-0 text-accent" aria-hidden />
              <span className="truncate font-mono text-[11.5px] text-ink" title={dataset.name}>
                {dataset.mode === 'multi-temporal'
                  ? `${dataset.scenes?.length || dataset.files?.length || 'Time'} Observations: ${dataset.name.split(':::').join(' ➔ ')}`
                  : dataset.t1Filename && dataset.t2Filename
                  ? `${dataset.t1Filename} + ${dataset.t2Filename}`
                  : dataset.name.includes(':::')
                  ? dataset.name.split(':::').join(' + ')
                  : dataset.name}
              </span>
            </div>
            <button
              type="button"
              onClick={clearScene}
              aria-label="Remove scene"
              className="rounded p-1 text-ink-faint transition-colors hover:bg-surface-3 hover:text-danger cursor-pointer"
            >
              <Trash2 size={13} />
            </button>
          </div>

          <div className="mt-2.5 flex flex-wrap gap-1.5">
            <Badge variant={SENSOR_TONE[sensorKind(dataset.sensor)]}>{dataset.sensor}</Badge>
            <Badge variant="quiet">{dataset.mode}</Badge>
            {dataset.syncedWithBackend ? (
              <Badge variant="success" dot>
                ingested
              </Badge>
            ) : (
              <Badge variant="warning" dot>
                local only
              </Badge>
            )}
            {dataset.syncedWithBackend && !dataset.georeferenced && (
              <Badge variant="warning">no CRS</Badge>
            )}
          </div>

          <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2 border-t border-line pt-2.5">
            {[
              ['CRS', dataset.crs],
              ['Resolution', dataset.resolution],
              ['Size', dataset.sizeLabel],
              ['Area', dataset.areaSqKm ? `${dataset.areaSqKm.toLocaleString()} km²` : '—'],
            ].map(([label, value]) => (
              <div key={label} className="min-w-0">
                <dt className="label-caps text-ink-faint">{label}</dt>
                <dd className="mt-1 truncate font-mono text-[11px] text-ink-muted" title={String(value)}>
                  {value}
                </dd>
              </div>
            ))}
          </dl>

          {dataset.sites && dataset.sites.length > 1 && (
            <div className="mt-3 border-t border-line pt-2.5">
              <span className="label-caps text-ink-faint block mb-1.5 flex items-center justify-between">
                <span>Discovered Regional Sites ({dataset.sites.length})</span>
                <span className="text-[9.5px] text-accent font-mono">Click to focus</span>
              </span>
              <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                {dataset.sites.map((st) => {
                  const isCurrent = activeSiteId ? st.site_id === activeSiteId : st.site_id === dataset.sites![0].site_id;
                  return (
                    <button
                      key={st.site_id}
                      type="button"
                      onClick={() => setActiveSiteId(st.site_id)}
                      className={cn(
                        'w-full flex items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-[11px] font-mono transition-all border cursor-pointer',
                        isCurrent
                          ? 'border-accent bg-accent/15 text-accent font-semibold shadow-[0_0_8px_rgba(0,242,255,0.2)]'
                          : 'border-line bg-surface-2/60 text-ink-muted hover:border-accent/40 hover:text-ink'
                      )}
                    >
                      <div className="min-w-0 flex items-center gap-1.5">
                        <span className={cn('h-1.5 w-1.5 rounded-full shrink-0', isCurrent ? 'bg-accent animate-pulse' : 'bg-ink-muted')} />
                        <span className="truncate">{st.name}</span>
                      </div>
                      <span className="shrink-0 text-[10px] opacity-75">
                        {st.scene_count} Epochs
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {dataset.mode === 'multi-temporal' && dataset.scenes && dataset.scenes.length > 0 && (
            <div className="mt-3 border-t border-line pt-2.5">
              <span className="label-caps text-ink-faint block mb-1.5">Temporal Stack ({dataset.scenes.length} Scenes)</span>
              <div className="space-y-1 max-h-36 overflow-y-auto pr-1">
                {dataset.scenes.map((sc, idx) => (
                  <div
                    key={sc.id || idx}
                    className="flex items-center justify-between gap-2 rounded bg-surface-2 px-2 py-1 text-[10.5px] font-mono"
                  >
                    <span className="flex items-center gap-1.5 text-accent font-semibold">
                      <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                      T{idx + 1}
                    </span>
                    <span className="truncate text-ink-muted" title={sc.date || sc.acquisition_date || sc.filename}>
                      {sc.date || sc.acquisition_date || sc.filename}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {overlay && (
            <p className="mt-2.5 border-t border-line pt-2 font-mono text-[10px] text-ink-faint">
              Bounds {overlay.bounds.map((b) => b.toFixed(3)).join(', ')}
            </p>
          )}
        </div>
      )}

      {/* Demo scenes — served by GET /api/v1/imagery/presets */}
      <section className="mt-5">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="label-caps text-ink-faint">Demo scenes</h3>
          {presetsLoading && backendPhase === 'online' && (
            <Loader2 size={11} className="animate-spin text-ink-faint" aria-hidden />
          )}
        </div>

        <ul className="space-y-1.5">
          {presets.map((preset) => {
            const active = dataset?.name === preset.name;
            return (
              <li key={preset.id}>
                <button
                  type="button"
                  onClick={() => setScene(presetToDataset(preset), presetToOverlay(preset))}
                  className={cn(
                    'flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition-colors cursor-pointer',
                    active
                      ? 'border-accent/40 bg-accent/10'
                      : 'border-line bg-surface/60 hover:border-line-strong hover:bg-surface-2'
                  )}
                >
                  <span
                    className={cn(
                      'flex h-7 w-7 shrink-0 items-center justify-center rounded-md border',
                      active ? 'border-accent/35 text-accent' : 'border-line text-ink-faint'
                    )}
                    aria-hidden
                  >
                    {active ? <CheckCircle2 size={14} /> : <Images size={14} />}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[12.5px] text-ink">
                      {preset.displayName || preset.name}
                    </span>
                    <span className="block truncate font-mono text-[10px] text-ink-faint">
                      {preset.sensor} · {preset.mode} · {preset.area_sq_km} km²
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>

        {backendPhase === 'offline' && (
          <p className="mt-2.5 font-mono text-[10px] leading-relaxed text-ink-faint">
            Backend offline — showing bundled scene definitions. Previews will not
            load until the FastAPI server is running.
          </p>
        )}
      </section>
    </div>
  );
};

export default ScenePanel;
