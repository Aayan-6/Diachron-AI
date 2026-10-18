import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ApiError,
  countEvidenceFeatures,
  extractGeoJson,
  extractImageOverlay,
  extractAllImageOverlays,
  extractTraceLines,
  fetchImageryPresets,
  fetchSystemStatus,
  resolveAssetUrl,
  submitSatQuery,
  type ImageryPreset,
  type SystemStatusResponse,
} from '../lib/api';
import { parseTrace, sumAreaHa } from '../lib/trace';
import type {
  AnalystDecision,
  AnalystVerdict,
  AOITemporalState,
  BackendPhase,
  CandidateAttributionDossier,
  ChangeCandidate,
  ChatMessage,
  ImageOverlayEvidence,
  ProvenanceRecord,
  SceneDataset,
  SceneOverlay,
  Session,
  TimelineObservation,
  TraceStep,
} from '../lib/types';
import { useLanguage } from './LanguageContext';


/* -------------------------------------------------------------------------- */
/* Fallback demo scenes                                                       */
/* -------------------------------------------------------------------------- */

/**
 * Used only while /imagery/presets is unreachable, so the workspace still has
 * something to show. Paths are relative so they resolve through the dev proxy.
 */
const FALLBACK_PRESETS: ImageryPreset[] = [
  {
    id: 'delhi_temporal',
    name: 'Delhi_MultiYear_2018_2026_Pair.tif',
    displayName: 'Delhi NCR 2018 / 2026',
    sensor: 'Optical (Sentinel-2 Multi-Temporal)',
    mode: 'bi-temporal',
    wgs84_bounds: [77.265, 28.58, 77.365, 28.69],
    center: [77.315, 28.635],
    crs: 'EPSG:32643 (UTM 43N)',
    resolution: '10.0m GSD',
    area_sq_km: 124.5,
    t1_image_url: '/static/prev_east_delhi_2018_S2.png',
    t2_image_url: '/static/prev_delhi_20260112_S2.png',
  },
];

export function presetToOverlay(preset: ImageryPreset): SceneOverlay {
  return {
    datasetId: preset.id,
    name: preset.displayName || preset.name,
    sensor: preset.sensor,
    mode: preset.mode,
    bounds: preset.wgs84_bounds,
    center: preset.center,
    crs: preset.crs,
    resolution: preset.resolution,
    areaSqKm: preset.area_sq_km,
    t1ImageUrl: resolveAssetUrl(preset.t1_image_url) ?? preset.t1_image_url,
    t2ImageUrl: resolveAssetUrl(preset.t2_image_url),
    bandContract: preset.band_contract,
    t1NirImageUrl: resolveAssetUrl(preset.t1_nir_image_url),
    t2NirImageUrl: resolveAssetUrl(preset.t2_nir_image_url),
  };
}

export function presetToDataset(preset: ImageryPreset): SceneDataset {
  let t1: string | undefined = undefined;
  let t2: string | undefined = undefined;
  if (preset.id === 'delhi_temporal') {
    t1 = 'aligned_east_delhi_2018_S2.tif';
    t2 = 'aligned_delhi_20260112_S2.tif';
  } else if (preset.id === 'bengaluru_urban_pair') {
    t1 = 'Bengaluru_T1_Pre.png';
    t2 = 'Bengaluru_T2_Post.png';
  } else if (preset.mode === 'bi-temporal' && preset.name.includes(':::')) {
    const parts = preset.name.split(':::');
    t1 = parts[0];
    t2 = parts[1];
  }

  return {
    // The backend resolves imagery by the on-disk filename, not the pretty
    // display name, so the dataset carries `name` verbatim.
    name: preset.name,
    sizeLabel: preset.area_sq_km ? `${preset.area_sq_km} km²` : '—',
    sensor: preset.sensor,
    mode: preset.mode,
    crs: preset.crs,
    resolution: preset.resolution,
    syncedWithBackend: true,
    georeferenced: true,
    areaSqKm: preset.area_sq_km,
    datasetId: preset.id,
    t1Filename: t1,
    t2Filename: t2,
    bandContract: preset.band_contract,
  };
}

/* -------------------------------------------------------------------------- */
/* Context shape                                                               */
/* -------------------------------------------------------------------------- */

export type DockTab = 'map' | 'trace' | 'scene' | 'review';

interface AppStateValue {
  /* backend */
  backendPhase: BackendPhase;
  backendStatus: SystemStatusResponse | null;
  backendError: string | null;
  refreshBackend: () => void;

  /* imagery presets */
  presets: ImageryPreset[];
  presetsLoading: boolean;

  /* active scene */
  dataset: SceneDataset | null;
  overlay: SceneOverlay | null;
  setScene: (dataset: SceneDataset | null, overlay: SceneOverlay | null) => void;
  clearScene: () => void;

  /* AOI bi-temporal analysis state */
  aoiTemporal: AOITemporalState;
  setAoiTemporal: (patch: Partial<AOITemporalState>) => void;
  clearAoiTemporal: () => void;
  invalidateAoiTemporal: () => void;

  /* conversation */
  messages: ChatMessage[];
  isBusy: boolean;
  sendQuery: (text: string) => void;
  cancelQuery: () => void;
  retryMessage: (messageId: string) => void;
  resetConversation: () => void;

  /* evidence shown on the map */
  focusedGeoJson: any | null;
  /** Georeferenced raster the focused answer produced, e.g. a change mask. */
  focusedOverlay: ImageOverlayEvidence | null;
  focusedMessageId: string | null;
  focusEvidence: (messageId: string) => void;
  clearFocus: () => void;

  /* multi-temporal timeline */
  focusedTimeline: TimelineObservation[] | null;
  focusedCandidates: ChangeCandidate[] | null;
  activeTimelineStep: number;
  setActiveTimelineStep: (step: number) => void;

  /* multi-site spatial clusters */
  activeSiteId: string | null;
  setActiveSiteId: (siteId: string) => void;

  /* analyst review & provenance (PS2 §2.2.5) */
  analystReviews: Record<string, AnalystDecision>;
  setAnalystVerdict: (candidateId: number | string, verdict: AnalystVerdict, notes?: string) => void;
  resetAnalystVerdict: (candidateId: number | string) => void;
  focusedCandidateId: number | string | null;
  setFocusedCandidateId: (id: number | string | null) => void;
  focusedProvenance: ProvenanceRecord | null;
  setCandidateAttribution: (candidateId: number | string, dossier: CandidateAttributionDossier) => void;

  /* trace inspection */
  activeTrace: TraceStep[];
  selectedStep: TraceStep | null;
  setSelectedStep: (step: TraceStep | null) => void;

  /* right-hand dock */
  dockOpen: boolean;
  setDockOpen: (open: boolean) => void;
  dockTab: DockTab;
  setDockTab: (tab: DockTab) => void;
  openDock: (tab: DockTab) => void;

  /* history */
  historyOpen: boolean;
  setHistoryOpen: (open: boolean) => void;
  sessions: Session[];
  restoreSession: (sessionId: string) => void;
  renameSession: (sessionId: string, title: string) => void;
  deleteSession: (sessionId: string) => void;
}

const AppStateContext = createContext<AppStateValue | undefined>(undefined);

const MESSAGES_KEY = 'diachron.messages.v2';
const SESSIONS_KEY = 'diachron.sessions.v2';
const REVIEWS_KEY = 'diachron.analyst_reviews.v1';

function loadPersisted<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as T) : fallback;
  } catch {
    return fallback;
  }
}

function persist(key: string, value: unknown) {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota or private mode — the app works fine without persistence */
  }
}

const newId = (prefix: string) =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;

/* -------------------------------------------------------------------------- */
/* Provider                                                                    */
/* -------------------------------------------------------------------------- */

export const AppStateProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { language } = useLanguage();

  /* --- backend health ---------------------------------------------------- */
  const [backendPhase, setBackendPhase] = useState<BackendPhase>('checking');
  const [backendStatus, setBackendStatus] = useState<SystemStatusResponse | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [healthNonce, setHealthNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const status = await fetchSystemStatus();
        if (cancelled) return;
        setBackendStatus(status);
        setBackendPhase('online');
        setBackendError(null);
      } catch (err) {
        if (cancelled) return;
        setBackendPhase('offline');
        setBackendStatus(null);
        setBackendError(err instanceof ApiError ? err.detail || err.message : String(err));
      }
    };

    check();
    const interval = setInterval(check, 30000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [healthNonce]);

  const refreshBackend = useCallback(() => {
    setBackendPhase('checking');
    setHealthNonce((n) => n + 1);
  }, []);

  /* --- imagery presets --------------------------------------------------- */
  const [presets, setPresets] = useState<ImageryPreset[]>(FALLBACK_PRESETS);
  const [presetsLoading, setPresetsLoading] = useState(true);

  useEffect(() => {
    if (backendPhase !== 'online') return;
    const controller = new AbortController();
    setPresetsLoading(true);
    fetchImageryPresets(controller.signal)
      .then((list) => {
        if (Array.isArray(list) && list.length > 0) setPresets(list);
      })
      .catch(() => {
        /* keep the bundled fallback list */
      })
      .finally(() => setPresetsLoading(false));
    return () => controller.abort();
  }, [backendPhase]);

  /* --- active scene ------------------------------------------------------ */
  const [dataset, setDataset] = useState<SceneDataset | null>(null);
  const [overlay, setOverlay] = useState<SceneOverlay | null>(null);
  const [activeSiteId, setActiveSiteIdState] = useState<string | null>(null);

  const setScene = useCallback((next: SceneDataset | null, nextOverlay: SceneOverlay | null) => {
    setDataset(next);
    setOverlay(nextOverlay);
    if (nextOverlay?.activeSiteId) {
      setActiveSiteIdState(nextOverlay.activeSiteId);
    } else if (nextOverlay?.sites && nextOverlay.sites.length > 0) {
      setActiveSiteIdState(nextOverlay.sites[0].site_id);
    } else {
      setActiveSiteIdState(null);
    }
  }, []);

  const setActiveSiteId = useCallback((siteId: string) => {
    setActiveSiteIdState(siteId);
    setOverlay((prev) => {
      if (!prev || !prev.sites) return prev;
      const targetSite = prev.sites.find((s) => s.site_id === siteId);
      if (!targetSite) return prev;
      return {
        ...prev,
        activeSiteId: siteId,
        name: targetSite.name,
        bounds: targetSite.bounds,
        center: targetSite.center,
        t1ImageUrl: targetSite.t1ImageUrl || targetSite.scenes[0]?.image_url || prev.t1ImageUrl,
        t2ImageUrl: targetSite.t2ImageUrl || targetSite.scenes[targetSite.scenes.length - 1]?.image_url || prev.t2ImageUrl,
        scenes: targetSite.scenes,
      };
    });
    setDataset((prev) => {
      if (!prev || !prev.sites) return prev;
      const targetSite = prev.sites.find((s) => s.site_id === siteId);
      if (!targetSite) return prev;
      return {
        ...prev,
        activeSiteId: siteId,
        name: targetSite.name,
        sizeLabel: `${targetSite.scenes.length} Epochs`,
        scenes: targetSite.scenes,
      };
    });
    setActiveTimelineStep(0);
  }, []);

  const clearScene = useCallback(() => {
    setDataset(null);
    setOverlay(null);
    setActiveSiteIdState(null);
  }, []);

  /* --- AOI bi-temporal analysis state ------------------------------------ */
  const [aoiTemporal, setAoiTemporalState] = useState<AOITemporalState>({
    bbox: null,
    areaKm2: 0,
    date1: '',
    date2: '',
    sensor: 'sentinel-2',
    maxCloudCover: 20,
    t1Scene: null,
    t2Scene: null,
    intervalDays: 0,
    bandContract: null,
    compatibility: null,
    isStale: false,
  });

  const setAoiTemporal = useCallback((patch: Partial<AOITemporalState>) => {
    setAoiTemporalState((prev) => {
      // Check invalidation conditions:
      // If bbox changes, sensor changes, or either date changes: invalidate fetched imagery
      const bboxChanged = patch.bbox !== undefined && (
        (!prev.bbox && patch.bbox !== null) ||
        (prev.bbox && !patch.bbox) ||
        (prev.bbox && patch.bbox && (
          patch.bbox[0] !== prev.bbox[0] ||
          patch.bbox[1] !== prev.bbox[1] ||
          patch.bbox[2] !== prev.bbox[2] ||
          patch.bbox[3] !== prev.bbox[3]
        ))
      );
      const sensorChanged = patch.sensor !== undefined && patch.sensor !== prev.sensor;
      const date1Changed = patch.date1 !== undefined && patch.date1 !== prev.date1;
      const date2Changed = patch.date2 !== undefined && patch.date2 !== prev.date2;

      let isStale = prev.isStale;
      let t1Scene = patch.t1Scene !== undefined ? patch.t1Scene : prev.t1Scene;
      let t2Scene = patch.t2Scene !== undefined ? patch.t2Scene : prev.t2Scene;

      if (bboxChanged || sensorChanged) {
        t1Scene = null;
        t2Scene = null;
        isStale = true;
      } else if (date1Changed) {
        t1Scene = null;
        isStale = true;
      } else if (date2Changed) {
        t2Scene = null;
        isStale = true;
      }

      if (patch.t1Scene && patch.t2Scene) {
        isStale = false;
      }

      return {
        ...prev,
        ...patch,
        t1Scene,
        t2Scene,
        isStale,
      };
    });
  }, []);

  const clearAoiTemporal = useCallback(() => {
    setAoiTemporalState({
      bbox: null,
      areaKm2: 0,
      date1: '',
      date2: '',
      sensor: 'sentinel-2',
      maxCloudCover: 20,
      t1Scene: null,
      t2Scene: null,
      intervalDays: 0,
      bandContract: null,
      compatibility: null,
      isStale: false,
    });
  }, []);

  const invalidateAoiTemporal = useCallback(() => {
    setAoiTemporalState((prev) => ({
      ...prev,
      t1Scene: null,
      t2Scene: null,
      isStale: true,
    }));
  }, []);

  /* --- conversation ------------------------------------------------------ */
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadPersisted(MESSAGES_KEY, []));
  const [sessions, setSessions] = useState<Session[]>(() => loadPersisted(SESSIONS_KEY, []));
  const [isBusy, setIsBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => persist(MESSAGES_KEY, messages), [messages]);
  useEffect(() => persist(SESSIONS_KEY, sessions), [sessions]);

  /* --- evidence focus ---------------------------------------------------- */
  const [focusedMessageId, setFocusedMessageId] = useState<string | null>(null);
  const [activeTimelineStep, setActiveTimelineStep] = useState<number>(0);

  /* --- dock -------------------------------------------------------------- */
  const [dockOpen, setDockOpen] = useState(true);
  const [dockTab, setDockTab] = useState<DockTab>('map');
  const [selectedStep, setSelectedStep] = useState<TraceStep | null>(null);

  const openDock = useCallback((tab: DockTab) => {
    setDockTab(tab);
    setDockOpen(true);
  }, []);

  const patchMessage = useCallback((id: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  /**
   * Runs one turn. Files are only uploaded on the first turn for a scene —
   * afterwards the backend resolves the same raster by `dataset_name`, so
   * re-posting a 117 MB GeoTIFF on every question is avoided.
   */
  const runQuery = useCallback(
    async (text: string, assistantId: string) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setIsBusy(true);

      const startedAt = performance.now();
      const filesToSend = dataset?.files?.length ? dataset.files : null;
      const beforeFile = dataset?.files && dataset.files.length >= 2 ? dataset.files[0] : null;
      const afterFile = dataset?.files && dataset.files.length >= 2 ? dataset.files[1] : null;

      // Add instruction tag for the backend API based on active language
      const instruction =
        language === 'hi'
          ? '[Instruction: Respond strictly in Hindi language using Devanagari script]\n'
          : '[Instruction: Respond strictly in English language]\n';

      const fullBackendQuery = instruction + text;

      try {
        const response = await submitSatQuery({
          query: fullBackendQuery,
          files: filesToSend,
          beforeFile,
          afterFile,
          datasetName: dataset?.name ?? null,
          t1Filename: dataset?.t1Filename ?? null,
          t2Filename: dataset?.t2Filename ?? null,
          datasetId: dataset?.datasetId ?? null,
          signal: controller.signal,
        });

        // The bytes are on the server now; later turns can reference by name.
        if (filesToSend && dataset) {
          setDataset((prev) => (prev ? { ...prev, syncedWithBackend: true } : prev));
        }

        const geoJson = extractGeoJson(response);
        const overlay = extractImageOverlay(response);
        const allOverlays = extractAllImageOverlays(response);
        const trace = parseTrace(extractTraceLines(response), assistantId);

        // Auto-mount autonomous retrieval scene overlay (satellite photo & timeline)
        if (response.scene_overlay) {
          const resolvedSites = response.scene_overlay.sites?.map((st) => ({
            ...st,
            t1ImageUrl: resolveAssetUrl(st.t1ImageUrl) || st.t1ImageUrl,
            t2ImageUrl: resolveAssetUrl(st.t2ImageUrl) || st.t2ImageUrl,
            scenes: st.scenes.map((s) => ({
              ...s,
              image_url: resolveAssetUrl(s.image_url || s.imageUrl) || s.image_url || s.imageUrl,
              imageUrl: resolveAssetUrl(s.imageUrl || s.image_url) || s.imageUrl || s.image_url,
              nir_image_url: resolveAssetUrl(s.nir_image_url || s.nirImageUrl) || s.nir_image_url || s.nirImageUrl,
              nirImageUrl: resolveAssetUrl(s.nirImageUrl || s.nir_image_url) || s.nirImageUrl || s.nir_image_url,
            })),
            timeline: st.timeline?.map((t) => ({
              ...t,
              image_url: resolveAssetUrl(t.image_url) || t.image_url,
              imageUrl: resolveAssetUrl(t.image_url) || t.image_url,
              overlay_url: resolveAssetUrl(t.overlay_url) || t.overlay_url,
            })),
          }));

          const autoOv: SceneOverlay = {
            ...response.scene_overlay,
            t1ImageUrl: resolveAssetUrl(response.scene_overlay.t1ImageUrl) || response.scene_overlay.t1ImageUrl,
            t2ImageUrl: resolveAssetUrl(response.scene_overlay.t2ImageUrl) || response.scene_overlay.t2ImageUrl,
            t1NirImageUrl: resolveAssetUrl(response.scene_overlay.t1NirImageUrl) || response.scene_overlay.t1NirImageUrl,
            t2NirImageUrl: resolveAssetUrl(response.scene_overlay.t2NirImageUrl) || response.scene_overlay.t2NirImageUrl,
            sites: resolvedSites,
            activeSiteId: response.scene_overlay.activeSiteId || resolvedSites?.[0]?.site_id,
            scenes: response.scene_overlay.scenes?.map((s) => ({
              ...s,
              image_url: resolveAssetUrl(s.image_url || s.imageUrl) || s.image_url || s.imageUrl,
              imageUrl: resolveAssetUrl(s.imageUrl || s.image_url) || s.imageUrl || s.image_url,
              nir_image_url: resolveAssetUrl(s.nir_image_url || s.nirImageUrl) || s.nir_image_url || s.nirImageUrl,
              nirImageUrl: resolveAssetUrl(s.nirImageUrl || s.nir_image_url) || s.nirImageUrl || s.nir_image_url,
            })),
          };

          const autoDataset: SceneDataset = {
            name: autoOv.name,
            sizeLabel: `${autoOv.scenes?.length || 0} Epochs`,
            sensor: autoOv.sensor,
            mode: (autoOv.mode as any) || 'multi-temporal',
            crs: autoOv.crs || 'EPSG:4326',
            resolution: autoOv.resolution || '10.0m GSD',
            syncedWithBackend: true,
            georeferenced: true,
            datasetId: autoOv.datasetId,
            scenes: autoOv.scenes,
            sites: resolvedSites,
            activeSiteId: autoOv.activeSiteId,
          };
          setScene(autoDataset, autoOv);
          if (autoOv.activeSiteId) {
            setActiveSiteIdState(autoOv.activeSiteId);
          }
        }

        if (response.timeline && response.timeline.length > 0) {
          setActiveTimelineStep(response.timeline.length - 1);
        }

        patchMessage(assistantId, {
          content:
            response.text_answer?.trim() ||
            response.response?.trim() ||
            'The backend returned an empty answer.',
          status: 'complete',
          trace,
          elapsedMs: Math.round(performance.now() - startedAt),
          evidence: {
            geoJson,
            featureCount: countEvidenceFeatures(response),
            areaHa: sumAreaHa(geoJson),
            overlay,
            allOverlays,
            timeline: response.timeline,
            candidates: response.candidates,
            provenance: response.provenance,
          },
        });

        // Point the map at this answer, or clear it — leaving the previous
        // answer's polygons up would attribute them to the wrong reply.
        const hasMapEvidence = Boolean(geoJson) || Boolean(overlay) || Boolean(response.scene_overlay);
        setFocusedMessageId(hasMapEvidence ? assistantId : null);
        if (hasMapEvidence) {
          setDockTab('map');
          setDockOpen(true);
        }
      } catch (err) {

        const cancelled = controller.signal.aborted;
        if (cancelled) {
          patchMessage(assistantId, {
            content: 'Cancelled.',
            status: 'cancelled',
            elapsedMs: Math.round(performance.now() - startedAt),
          });
          return;
        }

        const apiError = err instanceof ApiError ? err : null;
        patchMessage(assistantId, {
          content: apiError?.message ?? 'The query could not be completed.',
          errorDetail: apiError?.detail || (err as Error)?.message || 'Unknown error',
          status: 'error',
          elapsedMs: Math.round(performance.now() - startedAt),
          sourceQuery: text,
        });
        if (apiError?.isOffline) setBackendPhase('offline');
      } finally {
        abortRef.current = null;
        setIsBusy(false);
      }
    },
    [dataset, patchMessage, language]
  );

  const sendQuery = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || abortRef.current) return;

      const assistantId = newId('a');
      const now = new Date().toISOString();

      setMessages((prev) => [
        ...prev,
        {
          id: newId('u'),
          role: 'user',
          content: trimmed,
          timestamp: now,
          status: 'complete',
          datasetName: dataset?.name,
          sensor: dataset?.sensor,
        },
        {
          id: assistantId,
          role: 'assistant',
          content: '',
          timestamp: now,
          status: 'pending',
          datasetName: dataset?.name,
          sensor: dataset?.sensor,
          sourceQuery: trimmed,
        },
      ]);

      void runQuery(trimmed, assistantId);
    },
    [dataset, runQuery]
  );

  const cancelQuery = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const retryMessage = useCallback(
    (messageId: string) => {
      const target = messages.find((m) => m.id === messageId);
      if (!target?.sourceQuery || abortRef.current) return;
      patchMessage(messageId, {
        status: 'pending',
        content: '',
        errorDetail: undefined,
        elapsedMs: undefined,
      });
      void runQuery(target.sourceQuery, messageId);
    },
    [messages, patchMessage, runQuery]
  );

  const resetConversation = useCallback(() => {
    abortRef.current?.abort();

    // Archive the thread before clearing it. Both updates are computed from
    // the current messages here rather than nesting one setState inside the
    // other's updater, which would double-archive under StrictMode.
    const firstUser = messages.find((m) => m.role === 'user');
    if (firstUser) {
      setSessions((history) => [
        {
          id: newId('s'),
          title: firstUser.content.slice(0, 90),
          startedAt: firstUser.timestamp,
          messages,
          datasetName: firstUser.datasetName,
        },
        ...history,
      ]);
    }

    setMessages([]);
    setFocusedMessageId(null);
    setSelectedStep(null);
  }, [messages]);

  const restoreSession = useCallback(
    (sessionId: string) => {
      const session = sessions.find((s) => s.id === sessionId);
      if (session) {
        setMessages(session.messages);
        setFocusedMessageId(null);
        setSelectedStep(null);
      }
    },
    [sessions]
  );

  const [historyOpen, setHistoryOpen] = useState(false);

  const renameSession = useCallback((sessionId: string, title: string) => {
    const trimmed = title.trim();
    if (!trimmed) return;
    setSessions((history) =>
      history.map((session) =>
        session.id === sessionId ? { ...session, title: trimmed.slice(0, 120) } : session
      )
    );
  }, []);

  const deleteSession = useCallback((sessionId: string) => {
    setSessions((history) => history.filter((session) => session.id !== sessionId));
  }, []);

  const focusEvidence = useCallback(
    (messageId: string) => {
      setFocusedMessageId(messageId);
      openDock('map');
    },
    [openDock]
  );

  const clearFocus = useCallback(() => setFocusedMessageId(null), []);

  /* --- derived --------------------------------------------------------- */
  const focusedGeoJson = useMemo(() => {
    if (!focusedMessageId) return null;
    return messages.find((m) => m.id === focusedMessageId)?.evidence?.geoJson ?? null;
  }, [focusedMessageId, messages]);

  const focusedTimeline = useMemo(() => {
    if (overlay?.sites && activeSiteId) {
      const activeSite = overlay.sites.find((s) => s.site_id === activeSiteId);
      if (activeSite?.timeline && activeSite.timeline.length > 0) {
        return activeSite.timeline;
      }
    }
    if (!focusedMessageId) return null;
    return messages.find((m) => m.id === focusedMessageId)?.evidence?.timeline ?? null;
  }, [focusedMessageId, messages, overlay, activeSiteId]);

  const focusedCandidates = useMemo(() => {
    if (!focusedMessageId) return null;
    return messages.find((m) => m.id === focusedMessageId)?.evidence?.candidates ?? null;
  }, [focusedMessageId, messages]);

  const focusedOverlay = useMemo(() => {
    if (!focusedMessageId) return null;
    const msg = messages.find((m) => m.id === focusedMessageId);
    if (!msg?.evidence) return null;
    if (msg.evidence.allOverlays && msg.evidence.allOverlays.length > 0 && activeTimelineStep > 0) {
      const stepOverlay = msg.evidence.allOverlays.find((ov) => ov.label?.includes(`T${activeTimelineStep}`));
      if (stepOverlay) return stepOverlay;
    }
    return msg.evidence.overlay ?? null;
  }, [focusedMessageId, messages, activeTimelineStep]);

  const activeTrace = useMemo(() => {
    const focused = focusedMessageId ? messages.find((m) => m.id === focusedMessageId) : null;
    if (focused?.trace?.length) return focused.trace;
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const trace = messages[i].trace;
      if (trace?.length) return trace;
    }
    return [];
  }, [focusedMessageId, messages]);

  /* --- analyst review state (PS2 §2.2.5) --------------------------------- */
  const [analystReviews, setAnalystReviews] = useState<Record<string, AnalystDecision>>(() => {
    try {
      const stored = localStorage.getItem(REVIEWS_KEY);
      return stored ? JSON.parse(stored) : {};
    } catch {
      return {};
    }
  });

  const [focusedCandidateId, setFocusedCandidateId] = useState<number | string | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(REVIEWS_KEY, JSON.stringify(analystReviews));
    } catch {
      /* ignore */
    }
  }, [analystReviews]);

  const setAnalystVerdict = useCallback((candidateId: number | string, verdict: AnalystVerdict, notes?: string) => {
    const key = String(candidateId);
    setAnalystReviews((prev) => {
      const existing = prev[key];
      const updated: AnalystDecision = {
        candidate_id: candidateId,
        verdict,
        notes: notes !== undefined ? notes : existing?.notes || '',
        reviewed_at: new Date().toISOString(),
        analyst_id: 'analyst_1',
      };
      return { ...prev, [key]: updated };
    });
  }, []);

  const resetAnalystVerdict = useCallback((candidateId: number | string) => {
    const key = String(candidateId);
    setAnalystReviews((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }, []);

  const setCandidateAttribution = useCallback((candidateId: number | string, dossier: CandidateAttributionDossier) => {
    setMessages((prev) =>
      prev.map((m) => {
        if (m.evidence?.candidates) {
          const updated = m.evidence.candidates.map((c) => {
            if (String(c.id) === String(candidateId)) {
              return { ...c, attribution_dossier: dossier };
            }
            return c;
          });
          return {
            ...m,
            evidence: {
              ...m.evidence,
              candidates: updated,
            },
          };
        }
        return m;
      })
    );
  }, []);

  const focusedProvenance = useMemo(() => {
    if (!focusedMessageId) {
      for (let i = messages.length - 1; i >= 0; i -= 1) {
        if (messages[i].evidence?.provenance) return messages[i].evidence!.provenance!;
      }
      return null;
    }
    return messages.find((m) => m.id === focusedMessageId)?.evidence?.provenance ?? null;
  }, [focusedMessageId, messages]);

  const value = useMemo<AppStateValue>(
    () => ({
      backendPhase,
      backendStatus,
      backendError,
      refreshBackend,
      presets,
      presetsLoading,
      dataset,
      overlay,
      setScene,
      clearScene,
      aoiTemporal,
      setAoiTemporal,
      clearAoiTemporal,
      invalidateAoiTemporal,
      messages,
      isBusy,
      sendQuery,
      cancelQuery,
      retryMessage,
      resetConversation,
      focusedGeoJson,
      focusedOverlay,
      focusedMessageId,
      focusEvidence,
      clearFocus,
      focusedTimeline,
      focusedCandidates,
      activeTimelineStep,
      setActiveTimelineStep,
      activeSiteId,
      setActiveSiteId,
      analystReviews,
      setAnalystVerdict,
      resetAnalystVerdict,
      focusedCandidateId,
      setFocusedCandidateId,
      focusedProvenance,
      setCandidateAttribution,
      activeTrace,
      selectedStep,
      setSelectedStep,
      dockOpen,
      setDockOpen,
      dockTab,
      setDockTab,
      openDock,
      historyOpen,
      setHistoryOpen,
      sessions,
      restoreSession,
      renameSession,
      deleteSession,
    }),
    [
      backendPhase,
      backendStatus,
      backendError,
      refreshBackend,
      presets,
      presetsLoading,
      dataset,
      overlay,
      setScene,
      clearScene,
      aoiTemporal,
      setAoiTemporal,
      clearAoiTemporal,
      invalidateAoiTemporal,
      messages,
      isBusy,
      sendQuery,
      cancelQuery,
      retryMessage,
      resetConversation,
      focusedGeoJson,
      focusedOverlay,
      focusedMessageId,
      focusEvidence,
      clearFocus,
      focusedTimeline,
      focusedCandidates,
      activeTimelineStep,
      activeSiteId,
      setActiveSiteId,
      analystReviews,
      setAnalystVerdict,
      resetAnalystVerdict,
      focusedCandidateId,
      focusedProvenance,
      setCandidateAttribution,
      activeTrace,
      selectedStep,
      dockOpen,
      dockTab,
      openDock,
      historyOpen,
      sessions,
      restoreSession,
      renameSession,
      deleteSession,
    ]
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
};

export function useApp(): AppStateValue {
  const context = useContext(AppStateContext);
  if (!context) throw new Error('useApp must be used inside <AppStateProvider>');
  return context;
}
