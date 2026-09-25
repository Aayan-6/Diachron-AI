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
  AOITemporalState,
  BackendPhase,
  ChatMessage,
  ImageOverlayEvidence,
  SceneDataset,
  SceneOverlay,
  Session,
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
    id: 'bengaluru_urban_pair',
    name: 'Bengaluru_Urban_Corridor_T1_T2.tif',
    displayName: 'Bengaluru Urban Corridor',
    sensor: 'Optical (Sentinel-2)',
    mode: 'bi-temporal',
    wgs84_bounds: [77.618, 13.022, 77.652, 13.048],
    center: [77.635, 13.035],
    crs: 'EPSG:4326 (WGS84)',
    resolution: '10.0m GSD',
    area_sq_km: 10.5,
    t1_image_url: '/static/Bengaluru_T1_Pre.png',
    t2_image_url: '/static/Bengaluru_T2_Post.png',
  },
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
  {
    id: 'mangalore_sar',
    name: 'Mangalore_Harbor_SAR_VV.tif',
    displayName: 'Mangalore Port (SAR)',
    sensor: 'SAR (Sentinel-1)',
    mode: 'single',
    wgs84_bounds: [74.78, 12.85, 74.88, 12.95],
    center: [74.83, 12.9],
    crs: 'EPSG:4326 (WGS84)',
    resolution: '10.0m GSD',
    area_sq_km: 118.2,
    t1_image_url: '/static/Mangalore_SAR_VV.png',
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

export type DockTab = 'map' | 'trace' | 'scene';

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

const MESSAGES_KEY = 'satquery.messages.v2';
const SESSIONS_KEY = 'satquery.sessions.v2';

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
    const interval = setInterval(check, backendPhase === 'offline' ? 5000 : 30000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [healthNonce, backendPhase]);

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

  const setScene = useCallback((next: SceneDataset | null, nextOverlay: SceneOverlay | null) => {
    setDataset(next);
    setOverlay(nextOverlay);
  }, []);

  const clearScene = useCallback(() => {
    setDataset(null);
    setOverlay(null);
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
        const trace = parseTrace(extractTraceLines(response), assistantId);

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
          },
        });

        // Point the map at this answer, or clear it — leaving the previous
        // answer's polygons up would attribute them to the wrong reply.
        const hasMapEvidence = Boolean(geoJson) || Boolean(overlay);
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

  /* --- derived ----------------------------------------------------------- */

  const focusedGeoJson = useMemo(() => {
    if (!focusedMessageId) return null;
    return messages.find((m) => m.id === focusedMessageId)?.evidence?.geoJson ?? null;
  }, [focusedMessageId, messages]);

  const focusedOverlay = useMemo(() => {
    if (!focusedMessageId) return null;
    return messages.find((m) => m.id === focusedMessageId)?.evidence?.overlay ?? null;
  }, [focusedMessageId, messages]);

  const activeTrace = useMemo(() => {
    const focused = focusedMessageId ? messages.find((m) => m.id === focusedMessageId) : null;
    if (focused?.trace?.length) return focused.trace;
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const trace = messages[i].trace;
      if (trace?.length) return trace;
    }
    return [];
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
