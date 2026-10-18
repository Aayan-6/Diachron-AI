import React, { useState, useEffect, useRef } from 'react';
import {
  Search,
  Sparkles,
  Layers,
  X,
  Loader2,
  History,
  AlertTriangle,
  RotateCw,
  Image as ImageIcon,
  ChevronRight,
  ChevronDown,
  MapPin,
  ShieldCheck,
  Check,
  SplitSquareVertical,
  FolderPlus,
  HardDrive,
  Link2,
  ThumbsUp,
  ThumbsDown,
} from 'lucide-react';
import {
  searchArchive,
  fetchArchiveClusters,
  rebuildArchiveIndex,
  fetchLocationTimeline,
  ingestCustomFolder,
  resetArchiveScope,
  fetchNetworkStatus,
  type ArchiveTileResult,
  type ArchiveClustersResponse,
  type RelationalQueryPlanInfo,
} from '../../lib/archiveApi';
import { refineArchiveSearchWithFeedback } from '../../lib/feedbackApi';
import { resolveAssetUrl } from '../../lib/api';
import { useApp } from '../../context/AppState';
import type { SceneDataset, SceneOverlay } from '../../lib/types';

interface ArchiveSearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentMapBbox?: [number, number, number, number] | null;
  onSelectTileForMap?: (tile: ArchiveTileResult) => void;
  onLoadComparisonPair?: (tile1: ArchiveTileResult, tile2: ArchiveTileResult) => void;
  onLoadTimelineStack?: (tiles: ArchiveTileResult[], initialTile: ArchiveTileResult) => void;
}

type TabType = 'search' | 'clusters' | 'anomalies';
type SearchMode = 'text' | 'image';
type SensorFilter = 'all' | 'sentinel-2' | 'sentinel-1';

const CONCEPT_PILLS = [
  'Structures near river',
  'Vehicles on open ground',
  'Buildings along highway',
  'Water bodies and canals',
  'Dense vegetation and forest',
  'High backscatter SAR radar',
  'Urban area without water',
];

const getCategoryStyle = (category?: string | null) => {
  const cat = (category || '').toLowerCase();
  if (cat.includes('coastal') || cat.includes('marine')) {
    return 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30';
  }
  if (cat.includes('hydro') || cat.includes('water') || cat.includes('flood')) {
    return 'bg-sky-500/15 text-sky-300 border-sky-500/30';
  }
  if (cat.includes('anthropogenic') || cat.includes('urban') || cat.includes('construction') || cat.includes('infrastructure')) {
    return 'bg-purple-500/15 text-purple-300 border-purple-500/30';
  }
  if (cat.includes('pheno') || cat.includes('vegetation') || cat.includes('crop') || cat.includes('dynamic')) {
    return 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30';
  }
  return 'bg-amber-500/15 text-amber-300 border-amber-500/30';
};

export const ArchiveSearchModal: React.FC<ArchiveSearchModalProps> = ({
  isOpen,
  onClose,
  currentMapBbox,
  onSelectTileForMap,
  onLoadComparisonPair,
  onLoadTimelineStack,
}) => {
  const { setScene, setActiveTimelineStep } = useApp();

  const [activeTab, setActiveTab] = useState<TabType>('search');
  const [searchMode, setSearchMode] = useState<SearchMode>('text');
  const [query, setQuery] = useState('');
  const [sensor, setSensor] = useState<SensorFilter>('all');
  const [useMapBbox, setUseMapBbox] = useState(false);
  const [minSimilarity] = useState(0.0);
  const [maxCloudCover] = useState(30);
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);

  // Multi-scene selection state
  const [selectedTiles, setSelectedTiles] = useState<ArchiveTileResult[]>([]);

  // Reference tile for image-to-image similarity
  const [referenceTile, setReferenceTile] = useState<ArchiveTileResult | null>(null);

  // Data states
  const [loading, setLoading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [results, setResults] = useState<ArchiveTileResult[]>([]);
  const [totalMatches, setTotalMatches] = useState(0);
  const [archiveTotal, setArchiveTotal] = useState(0);
  const [relationalPlan, setRelationalPlan] = useState<RelationalQueryPlanInfo | null>(null);
  const [clustersData, setClustersData] = useState<ArchiveClustersResponse | null>(null);
  const [, setError] = useState<string | null>(null);

  // Quick preview cache for cluster scenes
  const [clusterTilesCache, setClusterTilesCache] = useState<Record<number, ArchiveTileResult[]>>({});
  const [expandedClusterId, setExpandedClusterId] = useState<number | null>(null);
  const [loadingClusterTiles, setLoadingClusterTiles] = useState<number | null>(null);

  // Custom folder ingestion / offline pen-drive state
  const [showFolderModal, setShowFolderModal] = useState(false);
  const [folderPathInput, setFolderPathInput] = useState('');
  const [ingestingFolder, setIngestingFolder] = useState(false);
  const [folderIngestError, setFolderIngestError] = useState<string | null>(null);
  const [folderIngestSuccess, setFolderIngestSuccess] = useState<string | null>(null);
  const [networkInfo, setNetworkInfo] = useState<{ is_online: boolean; forced_offline: boolean } | null>(null);

  // Relevance feedback states (§2.2.5)
  const [confirmedTileIds, setConfirmedTileIds] = useState<string[]>([]);
  const [rejectedTileIds, setRejectedTileIds] = useState<string[]>([]);
  const [refiningWithFeedback, setRefiningWithFeedback] = useState<boolean>(false);
  const [feedbackRefinedExplanation, setFeedbackRefinedExplanation] = useState<string | null>(null);

  const toggleConfirmTile = (tileId: string) => {
    setRejectedTileIds((prev) => prev.filter((id) => id !== tileId));
    setConfirmedTileIds((prev) =>
      prev.includes(tileId) ? prev.filter((id) => id !== tileId) : [...prev, tileId]
    );
  };

  const toggleRejectTile = (tileId: string) => {
    setConfirmedTileIds((prev) => prev.filter((id) => id !== tileId));
    setRejectedTileIds((prev) =>
      prev.includes(tileId) ? prev.filter((id) => id !== tileId) : [...prev, tileId]
    );
  };

  const handleRefineSearchWithFeedback = async () => {
    setRefiningWithFeedback(true);
    try {
      const resp = await refineArchiveSearchWithFeedback({
        query,
        confirmed_tile_ids: confirmedTileIds,
        rejected_tile_ids: rejectedTileIds,
        limit: 30,
      });
      setResults(resp.results);
      setTotalMatches(resp.total_matches);
      setFeedbackRefinedExplanation(resp.explanation);
    } catch (err: any) {
      console.error('Failed to refine search:', err);
    } finally {
      setRefiningWithFeedback(false);
    }
  };

  const handleResetFeedback = () => {
    setConfirmedTileIds([]);
    setRejectedTileIds([]);
    setFeedbackRefinedExplanation(null);
    triggerSearch();
  };

  const debounceTimer = useRef<any>(null);

  // Fetch cluster data on mount
  useEffect(() => {
    if (isOpen) {
      loadClusters();
      triggerSearch();
      checkNetwork();
    }
  }, [isOpen]);

  const checkNetwork = async () => {
    try {
      const net = await fetchNetworkStatus();
      setNetworkInfo({ is_online: net.is_online, forced_offline: net.forced_offline });
    } catch {
      // offline or silent
    }
  };

  const handleIngestFolder = async () => {
    if (!folderPathInput.trim()) {
      setFolderIngestError('Please enter a folder path (e.g. E:\\Judges_Scenes or D:\\Data)');
      return;
    }
    setIngestingFolder(true);
    setFolderIngestError(null);
    setFolderIngestSuccess(null);
    try {
      const res = await ingestCustomFolder(folderPathInput.trim());
      setFolderIngestSuccess(`Successfully ingested ${res.scenes_indexed} scenes from ${res.active_folder_path}`);
      await loadClusters();
      await triggerSearch();
      setTimeout(() => {
        setShowFolderModal(false);
        setFolderIngestSuccess(null);
      }, 1500);
    } catch (err: any) {
      setFolderIngestError(err?.message || 'Failed to ingest folder');
    } finally {
      setIngestingFolder(false);
    }
  };

  const handleResetScope = async () => {
    try {
      await resetArchiveScope();
      await loadClusters();
      await triggerSearch();
    } catch (err: any) {
      console.warn('Failed to reset archive scope:', err);
    }
  };

  const toggleSelectTile = (tile: ArchiveTileResult) => {
    setSelectedTiles((prev) => {
      const exists = prev.some((t) => t.tile_id === tile.tile_id);
      if (exists) {
        return prev.filter((t) => t.tile_id !== tile.tile_id);
      } else {
        return [...prev, tile];
      }
    });
  };

  const isSelected = (tileId: string) => selectedTiles.some((t) => t.tile_id === tileId);

  const loadClusters = async () => {
    try {
      const data = await fetchArchiveClusters();
      setClustersData(data);
      setArchiveTotal(data.total_tiles);
    } catch (err: any) {
      console.warn('Could not load archive clusters:', err);
    }
  };

  const handleRebuildIndex = async () => {
    setRebuilding(true);
    try {
      await rebuildArchiveIndex();
      await loadClusters();
      await triggerSearch();
    } catch (err: any) {
      setError(err?.message || 'Failed to rebuild archive index');
    } finally {
      setRebuilding(false);
    }
  };

  const triggerSearch = async (
    overrideQuery?: string,
    overrideRefTile?: ArchiveTileResult | null,
    overrideClusterId?: number | null,
    overrideSensor?: SensorFilter
  ) => {
    setLoading(true);
    setError(null);
    try {
      const activeQ = overrideQuery !== undefined ? overrideQuery : query;
      const activeRef = overrideRefTile !== undefined ? overrideRefTile : referenceTile;
      const activeClusterId = overrideClusterId !== undefined ? overrideClusterId : selectedClusterId;
      const activeSensor = overrideSensor !== undefined ? overrideSensor : sensor;

      const isClusterBrowse = activeClusterId !== null && activeClusterId !== undefined;

      const resp = await searchArchive({
        query: activeQ,
        search_mode: activeRef ? 'image' : searchMode,
        reference_tile_id: activeRef ? activeRef.tile_id : undefined,
        sensor: activeSensor,
        bbox: useMapBbox && currentMapBbox ? currentMapBbox : undefined,
        min_similarity: minSimilarity,
        max_cloud_cover: maxCloudCover,
        cluster_id: activeClusterId,
        limit: isClusterBrowse ? 50 : 20,
        group_by_location: isClusterBrowse ? false : true,
      });

      setResults(resp.results);
      setTotalMatches(resp.total_matches);
      setArchiveTotal(resp.archive_total);
      setRelationalPlan(resp.relational_plan || null);
    } catch (err: any) {
      setError(err?.message || 'Archive search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleQueryChange = (val: string) => {
    setQuery(val);
    setReferenceTile(null); // clear image-to-image when text is edited
    if (!val.trim()) setRelationalPlan(null);
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    debounceTimer.current = setTimeout(() => {
      triggerSearch(val, null);
    }, 350);
  };

  const handleSelectPill = (pillText: string) => {
    setQuery(pillText);
    setReferenceTile(null);
    triggerSearch(pillText, null);
  };

  const handleImageSimilarity = (tile: ArchiveTileResult) => {
    setReferenceTile(tile);
    setSearchMode('image');
    setActiveTab('search');
    triggerSearch(undefined, tile);
  };

  const handleDirectSelectForMap = (tile: ArchiveTileResult) => {
    if (onSelectTileForMap) {
      onSelectTileForMap(tile);
    } else {
      const thumb = resolveAssetUrl(tile.thumbnail_url) || tile.thumbnail_url || '';
      const newDataset: SceneDataset = {
        name: tile.anomaly_title || tile.archetype_title || tile.filename,
        sizeLabel: tile.is_anomaly ? 'Physical Outlier' : 'Archive Scene',
        sensor: tile.sensor,
        mode: 'single',
        crs: 'EPSG:4326 (WGS84)',
        resolution: '10.0m',
        syncedWithBackend: true,
        georeferenced: true,
        datasetId: `archive-tile-${tile.tile_id}`,
        t1Filename: tile.filename,
      };
      const newOverlay: SceneOverlay = {
        datasetId: `archive-tile-${tile.tile_id}`,
        name: tile.anomaly_title || tile.archetype_title || tile.filename,
        sensor: tile.sensor,
        mode: 'single',
        bounds: tile.bbox,
        center: tile.center,
        crs: 'EPSG:4326 (WGS84)',
        resolution: '10.0m',
        t1ImageUrl: thumb,
        t1Filename: tile.filename,
      };
      setScene(newDataset, newOverlay);
    }
    onClose();
  };

  const handleLoadTimeline = async (tile: ArchiveTileResult) => {
    setLoading(true);
    try {
      const locResp = await fetchLocationTimeline(tile.bbox);
      if (locResp.timeline_stack && locResp.timeline_stack.length > 0) {
        if (onLoadTimelineStack) {
          onLoadTimelineStack(locResp.timeline_stack, tile);
        } else {
          // Direct fallback handoff into AppState
          const scenes = locResp.timeline_stack;
          const latest = scenes[scenes.length - 1];

          const newDataset: SceneDataset = {
            name: `${tile.sensor} Time Series (${scenes.length} obs)`,
            sizeLabel: 'Archive Stack',
            sensor: tile.sensor,
            mode: scenes.length >= 2 ? 'multi-temporal' : 'single',
            crs: 'EPSG:4326 (WGS84)',
            resolution: '10.0m',
            syncedWithBackend: true,
            georeferenced: true,
            datasetId: `archive-timeline-${tile.tile_id}`,
          };

          const newOverlay: SceneOverlay = {
            datasetId: `archive-timeline-${tile.tile_id}`,
            name: `${tile.sensor} Multi-Temporal Stack`,
            sensor: tile.sensor,
            mode: scenes.length >= 2 ? 'multi-temporal' : 'single',
            bounds: tile.bbox,
            center: tile.center,
            crs: 'EPSG:4326 (WGS84)',
            resolution: '10.0m',
            t1ImageUrl: resolveAssetUrl(latest.thumbnail_url || tile.thumbnail_url) || '',
            scenes: scenes.map((s, idx) => ({
              step: idx,
              step_index: idx,
              filename: s.filename,
              date: s.date,
              acquisition_date: s.date,
              sensor: s.sensor,
              cloud_cover: s.cloud_cover,
              imageUrl: resolveAssetUrl(s.thumbnail_url) || s.thumbnail_url || '',
              image_url: resolveAssetUrl(s.thumbnail_url) || s.thumbnail_url || '',
              wgs84_bounds: s.bbox,
            })),
          };

          setScene(newDataset, newOverlay);
          setActiveTimelineStep(scenes.length - 1);
        }
        onClose();
      }
    } catch (err: any) {
      console.warn('Failed to load timeline stack:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleClusterQuickPreview = async (clusterId: number) => {
    if (expandedClusterId === clusterId) {
      setExpandedClusterId(null);
      return;
    }

    setExpandedClusterId(clusterId);
    if (!clusterTilesCache[clusterId]) {
      setLoadingClusterTiles(clusterId);
      try {
        const resp = await searchArchive({
          query: '',
          cluster_id: clusterId,
          min_similarity: 0.0,
          limit: 30,
          group_by_location: false,
        });
        setClusterTilesCache((prev) => ({ ...prev, [clusterId]: resp.results }));
      } catch (e) {
        console.warn('Failed to fetch cluster scenes:', e);
      } finally {
        setLoadingClusterTiles(null);
      }
    }
  };

  const activeClusterObj = clustersData?.clusters.find((c) => c.cluster_id === selectedClusterId);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-space-black/75 p-3 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative flex h-[90vh] max-h-[820px] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-line bg-surface/95 shadow-2xl backdrop-blur-xl">
        {/* Header */}
        <div className="flex shrink-0 items-center justify-between border-b border-line px-5 py-3.5 bg-ground/50">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent/15 border border-accent/30 text-accent shadow-[0_0_12px_rgba(0,242,255,0.2)]">
              <Sparkles size={18} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold tracking-wide text-ink">
                  Archive Semantic Discovery & Clustering
                </h2>
                <span className="rounded-full bg-surface-2 border border-line px-2 py-0.5 font-mono text-[10px] text-ink-muted">
                  {archiveTotal} scenes indexed
                </span>
              </div>
              <p className="text-[11px] text-ink-faint">
                Search multi-sensor imagery by semantic meaning, explore land archetypes, and inspect anomalies
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {networkInfo && !networkInfo.is_online && (
              <span className="hidden sm:inline-flex items-center gap-1.5 rounded-full bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 font-mono text-[10px] text-amber-300">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse" />
                Offline Mode
              </span>
            )}

            {clustersData?.is_scoped_to_folder && clustersData?.active_folder_path && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-mono text-[11px]">
                <FolderPlus size={12} />
                <span className="truncate max-w-[150px] sm:max-w-[200px]" title={clustersData.active_folder_path}>
                  {clustersData.active_folder_path.split(/[\\/]/).pop() || 'Custom Folder'}
                </span>
                <button
                  type="button"
                  onClick={handleResetScope}
                  className="ml-1 text-[10px] text-emerald-300 underline hover:text-white cursor-pointer"
                  title="Reset to default multi-sensor archive"
                >
                  Reset
                </button>
              </div>
            )}

            <button
              type="button"
              onClick={() => setShowFolderModal(true)}
              title="Scope archive to a local folder or pen-drive (works offline & online)"
              className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1.5 text-xs text-emerald-400 transition-colors hover:bg-emerald-500/20 cursor-pointer"
            >
              <HardDrive size={13} />
              <span className="hidden sm:inline">Import Folder</span>
            </button>

            <button
              type="button"
              onClick={handleRebuildIndex}
              disabled={rebuilding}
              title="Rescan archive directory & recompute vector embeddings"
              className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface-2 px-2.5 py-1.5 text-xs text-ink-muted transition-colors hover:border-accent/40 hover:text-accent disabled:opacity-50 cursor-pointer"
            >
              <RotateCw size={13} className={rebuilding ? 'animate-spin text-accent' : ''} />
              <span className="hidden sm:inline">Sync Index</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg p-1.5 text-ink-muted hover:bg-surface-3 hover:text-ink transition-colors cursor-pointer"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex shrink-0 items-center gap-2 border-b border-line px-5 py-2 bg-surface-2/60">
          <button
            type="button"
            onClick={() => setActiveTab('search')}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all cursor-pointer ${
              activeTab === 'search'
                ? 'bg-accent/15 text-accent border border-accent/30 shadow-[0_0_8px_rgba(0,242,255,0.15)]'
                : 'text-ink-muted hover:text-ink hover:bg-surface-3'
            }`}
          >
            <Search size={13} />
            Semantic Retrieval
            {totalMatches > 0 && (
              <span className="ml-1 rounded-full bg-accent/20 px-1.5 py-0.2 font-mono text-[10px]">
                {totalMatches}
              </span>
            )}
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('clusters')}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all cursor-pointer ${
              activeTab === 'clusters'
                ? 'bg-accent/15 text-accent border border-accent/30 shadow-[0_0_8px_rgba(0,242,255,0.15)]'
                : 'text-ink-muted hover:text-ink hover:bg-surface-3'
            }`}
          >
            <Layers size={13} />
            Archive Clusters
            {clustersData?.clusters && (
              <span className="ml-1 rounded-full bg-surface-3 px-1.5 py-0.2 font-mono text-[10px]">
                {clustersData.clusters.length}
              </span>
            )}
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('anomalies')}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all cursor-pointer ${
              activeTab === 'anomalies'
                ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                : 'text-ink-muted hover:text-ink hover:bg-surface-3'
            }`}
          >
            <AlertTriangle size={13} />
            Physical Anomalies
            {clustersData?.anomalies_count !== undefined && clustersData.anomalies_count > 0 && (
              <span className="ml-1 rounded-full bg-amber-500/20 text-amber-300 px-1.5 py-0.2 font-mono text-[10px]">
                {clustersData.anomalies_count}
              </span>
            )}
          </button>
        </div>

        {/* Tab 1: Semantic Retrieval */}
        {activeTab === 'search' && (
          <div className="flex flex-1 min-h-0 flex-col overflow-hidden">
            {/* Search Input Bar */}
            <div className="shrink-0 p-4 border-b border-line bg-surface/40 space-y-3">
              <div className="relative flex items-center">
                <Search size={16} className="absolute left-3.5 text-ink-muted" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => handleQueryChange(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && triggerSearch()}
                  placeholder={
                    referenceTile
                      ? `Searching visual neighbors for: ${referenceTile.filename}...`
                      : "Search by meaning (e.g. 'water bodies near rivers', 'built-up structures', 'dense forest')..."
                  }
                  className="w-full rounded-xl border border-line bg-surface-2 py-2.5 pl-10 pr-24 text-xs text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
                />
                {referenceTile ? (
                  <button
                    type="button"
                    onClick={() => {
                      setReferenceTile(null);
                      setSearchMode('text');
                      triggerSearch('', null);
                    }}
                    className="absolute right-2 rounded-lg bg-surface-3 px-2 py-1 text-[10.5px] font-mono text-amber-400 hover:bg-surface border border-line cursor-pointer"
                  >
                    Clear Image Ref
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => triggerSearch()}
                    disabled={loading}
                    className="absolute right-2 rounded-lg bg-accent/20 px-3 py-1 text-xs font-medium text-accent hover:bg-accent/30 border border-accent/40 transition-colors cursor-pointer"
                  >
                    {loading ? <Loader2 size={13} className="animate-spin" /> : 'Search'}
                  </button>
                )}
              </div>

              {/* Concept Quick-Pills */}
              {!referenceTile && (
                <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-[11px] scrollbar-none">
                  <span className="shrink-0 text-ink-faint text-[10px] uppercase font-mono">Suggested:</span>
                  {CONCEPT_PILLS.map((pill) => (
                    <button
                      key={pill}
                      type="button"
                      onClick={() => handleSelectPill(pill)}
                      className={`shrink-0 rounded-full border px-2.5 py-0.5 transition-colors cursor-pointer ${
                        query === pill
                          ? 'border-accent/50 bg-accent/15 text-accent'
                          : 'border-line bg-surface-2/60 text-ink-muted hover:border-accent/30 hover:text-ink'
                      }`}
                    >
                      {pill}
                    </button>
                  ))}
                </div>
              )}

              {/* Filters Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-1 text-xs text-ink-muted">
                <div className="flex items-center gap-3">
                  {/* Sensor Toggle */}
                  <div className="flex items-center gap-1 rounded-lg border border-line bg-surface-2 p-0.5">
                    {(['all', 'sentinel-2', 'sentinel-1'] as SensorFilter[]).map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => {
                          setSensor(s);
                          setTimeout(() => triggerSearch(undefined, undefined, undefined, s), 50);
                        }}
                        className={`rounded-md px-2 py-0.5 text-[11px] capitalize transition-colors cursor-pointer ${
                          sensor === s ? 'bg-accent/20 text-accent font-medium' : 'text-ink-muted hover:text-ink'
                        }`}
                      >
                        {s === 'all' ? 'All Sensors' : s === 'sentinel-2' ? 'S2 (Optical)' : 'S1 (SAR)'}
                      </button>
                    ))}
                  </div>

                  {/* Bbox filter */}
                  {currentMapBbox && (
                    <label className="flex cursor-pointer items-center gap-1.5 text-[11.5px] select-none hover:text-ink">
                      <input
                        type="checkbox"
                        checked={useMapBbox}
                        onChange={(e) => {
                          setUseMapBbox(e.target.checked);
                          setTimeout(() => triggerSearch(), 50);
                        }}
                        className="rounded accent-[var(--color-accent)]"
                      />
                      <span>In current map view</span>
                    </label>
                  )}

                  {/* Cluster filter active pill */}
                  {selectedClusterId !== null && (
                    <div className="inline-flex items-center gap-1.5 rounded-full bg-accent/15 border border-accent/30 px-2.5 py-0.5 text-[11px] text-accent">
                      <span>Archetype: {activeClusterObj?.archetype_title || activeClusterObj?.label || `Cluster #${selectedClusterId}`}</span>
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedClusterId(null);
                          triggerSearch(undefined, undefined, null);
                        }}
                        className="hover:text-white cursor-pointer"
                        title="Clear Archetype filter"
                      >
                        <X size={11} />
                      </button>
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-3 font-mono text-[11px]">
                  {(confirmedTileIds.length > 0 || rejectedTileIds.length > 0) && (
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={handleRefineSearchWithFeedback}
                        disabled={refiningWithFeedback}
                        title="Apply Rocchio Relevance Feedback to refine search results using confirmed and rejected exemplar scenes (§2.2.5)"
                        className="flex items-center gap-1 rounded-lg border border-cyan-400/50 bg-cyan-500/20 px-2.5 py-1 text-[11px] font-mono font-bold text-cyan-300 hover:bg-cyan-500/30 transition-all cursor-pointer shadow-[0_0_12px_rgba(0,242,255,0.25)]"
                      >
                        {refiningWithFeedback ? <Loader2 size={12} className="animate-spin" /> : <Sparkles size={12} />}
                        <span>REFINE WITH FEEDBACK ({confirmedTileIds.length} 👍 / {rejectedTileIds.length} 👎)</span>
                      </button>
                      <button
                        type="button"
                        onClick={handleResetFeedback}
                        title="Clear relevance feedback"
                        className="rounded-md border border-line bg-surface-2 px-2 py-1 font-mono text-[10px] text-ink-muted hover:text-ink cursor-pointer"
                      >
                        Reset
                      </button>
                    </div>
                  )}
                  <span>Matches: <strong className="text-ink">{totalMatches}</strong></span>
                </div>
              </div>
            </div>

            {/* Results Grid */}
            <div className="flex-1 overflow-y-auto p-4">
              {/* Relational Query Decomposition Banner (ISRO PS 26167 §2.2.1) */}
              {relationalPlan && relationalPlan.is_compound && (
                <div className="mb-4 rounded-xl border border-cyan-500/35 bg-cyan-950/30 p-3.5 text-xs backdrop-blur-md shadow-[0_0_20px_rgba(0,242,255,0.08)]">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="flex h-5 w-5 items-center justify-center rounded-md bg-cyan-500/20 text-cyan-400">
                        <Link2 size={13} />
                      </span>
                      <span className="font-semibold text-cyan-300 text-xs">
                        Spatial Compound Query Decomposition (§2.2.1)
                      </span>
                    </div>
                    <span className="rounded-full bg-cyan-500/20 px-2.5 py-0.5 font-mono text-[10px] font-semibold text-cyan-300 border border-cyan-500/40">
                      OPERATOR: {relationalPlan.operator}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-ink">
                    <span className="rounded-lg bg-surface-3 px-2.5 py-1 text-cyan-200 border border-cyan-500/30 shadow-sm">
                      🎯 Target: <strong>{relationalPlan.target_subject}</strong>
                    </span>
                    <span className="text-cyan-400 font-semibold text-[11px]">──[{relationalPlan.connector_phrase}]──▶</span>
                    <span className="rounded-lg bg-surface-3 px-2.5 py-1 text-amber-200 border border-amber-500/30 shadow-sm">
                      📍 Context Anchor: <strong>{relationalPlan.context_anchor}</strong>
                    </span>
                  </div>
                  <p className="mt-2 text-[11px] text-ink-muted">
                    {relationalPlan.explanation} · Dual-concept projection with 4×4 spatial quadrant distance evaluation & harmonic co-occurrence.
                  </p>
                </div>
              )}

              {/* Relevance Feedback Refinement Active Banner (§2.2.5) */}
              {feedbackRefinedExplanation && (
                <div className="mb-4 rounded-xl border border-emerald-500/40 bg-emerald-950/25 p-3 text-xs backdrop-blur-md">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="flex h-5 w-5 items-center justify-center rounded-md bg-emerald-500/20 text-emerald-400">
                        <Sparkles size={12} />
                      </span>
                      <span className="font-semibold text-emerald-300 text-[11.5px]">
                        Rocchio Relevance Feedback Active (§2.2.5)
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={handleResetFeedback}
                      className="text-emerald-400 hover:text-white text-[10.5px] font-mono cursor-pointer"
                    >
                      Clear Feedback
                    </button>
                  </div>
                  <p className="mt-1 text-[11px] text-ink-muted">{feedbackRefinedExplanation}</p>
                </div>
              )}

              {loading ? (
                <div className="flex h-64 flex-col items-center justify-center gap-2 text-ink-muted">
                  <Loader2 size={24} className="animate-spin text-accent" />
                  <p className="text-xs">Computing semantic cosine similarities across archive...</p>
                </div>
              ) : results.length === 0 ? (
                <div className="flex h-64 flex-col items-center justify-center gap-2 text-center text-ink-faint">
                  <Search size={32} strokeWidth={1.5} className="text-ink-muted/50" />
                  <p className="text-xs text-ink-muted">No satellite scenes match the current semantic criteria</p>
                  <p className="text-[11px] max-w-sm">Try broadening your text query or switching to 'All Sensors'.</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3.5">
                  {results.map((item) => (
                    <div
                      key={item.tile_id}
                      className={`group relative flex flex-col justify-between overflow-hidden rounded-xl border transition-all ${
                        isSelected(item.tile_id)
                          ? 'border-accent bg-accent/10 shadow-[0_0_12px_rgba(0,242,255,0.2)]'
                          : 'border-line bg-surface-2/70 hover:border-accent/50 hover:bg-surface-2 hover:shadow-lg'
                      }`}
                    >
                      {/* Selection Checkbox */}
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleSelectTile(item);
                        }}
                        className={`absolute top-2 left-2 z-20 flex h-5 w-5 items-center justify-center rounded border transition-all cursor-pointer ${
                          isSelected(item.tile_id)
                            ? 'bg-accent border-accent text-space-black font-bold'
                            : 'bg-space-black/70 border-white/40 text-transparent hover:border-white hover:bg-space-black/90'
                        }`}
                        title={isSelected(item.tile_id) ? 'Deselect scene' : 'Select scene for comparison / timeline'}
                      >
                        <Check size={12} strokeWidth={3} className={isSelected(item.tile_id) ? 'block' : 'hidden'} />
                      </button>

                      {/* Image Thumbnail Preview */}
                      <div
                        onClick={() => handleDirectSelectForMap(item)}
                        className="relative aspect-video w-full overflow-hidden bg-space-black/60 cursor-pointer"
                        title="Click to overlay on map"
                      >
                        {item.thumbnail_url ? (
                          <img
                            src={resolveAssetUrl(item.thumbnail_url) || item.thumbnail_url}
                            alt={item.filename}
                            className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                            loading="lazy"
                          />
                        ) : (
                          <div className="flex h-full w-full items-center justify-center text-ink-faint">
                            <ImageIcon size={24} />
                          </div>
                        )}

                        {/* Similarity Score Pill */}
                        <div className="absolute top-2 right-2 rounded-full bg-space-black/80 px-2 py-0.5 font-mono text-[10px] text-accent backdrop-blur-md border border-accent/30 shadow">
                          {Math.round(item.similarity_score * 100)}% match
                        </div>

                        {/* Sensor Badge */}
                        <div className="absolute bottom-2 right-2 rounded-md bg-space-black/80 px-1.5 py-0.5 font-mono text-[9px] backdrop-blur-md border border-line text-ink">
                          {item.sensor === 'Sentinel-1' ? 'S1 SAR' : 'S2 Optical'}
                        </div>

                        {/* Anomaly Badge */}
                        {item.is_anomaly && (
                          <div className="absolute bottom-2 left-2 rounded bg-amber-500/90 text-space-black font-semibold px-1.5 py-0.5 font-mono text-[9px] shadow flex items-center gap-1 backdrop-blur-sm">
                            <AlertTriangle size={10} />
                            <span>Anomaly</span>
                          </div>
                        )}
                      </div>

                      {/* Content Card Body */}
                      <div className="flex flex-1 flex-col p-3">
                        <div className="mb-2">
                          <span className="block truncate font-mono text-[11px] font-medium text-ink" title={item.filename}>
                            {item.filename}
                          </span>
                          <span className="text-[10.5px] text-accent font-medium line-clamp-1" title={item.archetype_title || item.cluster_label}>
                            {item.archetype_title || item.cluster_label}
                          </span>
                        </div>

                        {/* Relational Co-occurrence Info (§2.2.1) */}
                        {item.relational_match && (
                          <div className="mb-2 rounded-lg border border-cyan-500/30 bg-cyan-950/25 p-2 font-mono text-[9.5px]">
                            <div className="flex items-center justify-between text-cyan-400 font-semibold mb-0.5">
                              <span>🔗 {item.relational_match.operator}</span>
                              <span>Co-occur: {Math.round(item.relational_match.harmonic_cooccurrence * 100)}%</span>
                            </div>
                            <div className="flex items-center justify-between text-ink-muted text-[9px]">
                              <span>Dist: {item.relational_match.spatial_quadrant_distance.toFixed(1)} quads</span>
                              <span className="text-cyan-300/80 truncate max-w-[110px]">{item.relational_match.match_status}</span>
                            </div>
                          </div>
                        )}

                        <div className="mt-auto space-y-1 border-t border-line/50 pt-2 font-mono text-[10.5px] text-ink-muted">
                          <div className="flex items-center justify-between">
                            <span className="text-ink-faint">Acquisition:</span>
                            <span className="text-ink">{item.date}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-ink-faint">Local Stack:</span>
                            <span className="text-accent">{item.temporal_stack_count} scenes</span>
                          </div>
                        </div>

                        {/* Action Buttons: Map | Timeline | Similar */}
                        <div className="mt-3 grid grid-cols-3 gap-1 pt-1">
                          <button
                            type="button"
                            onClick={() => handleDirectSelectForMap(item)}
                            title="Overlay this satellite scene on the map"
                            className="flex items-center justify-center gap-1 rounded-lg border border-accent/40 bg-accent/15 px-1.5 py-1.5 font-mono text-[9.5px] text-accent transition-colors hover:bg-accent/25 cursor-pointer font-medium"
                          >
                            <MapPin size={11} />
                            <span>Map</span>
                          </button>
                          <button
                            type="button"
                            onClick={() => handleLoadTimeline(item)}
                            title="Open multi-temporal timeline slider for this location"
                            className="flex items-center justify-center gap-1 rounded-lg border border-line bg-surface-3 px-1.5 py-1.5 font-mono text-[9.5px] text-ink-muted transition-colors hover:text-ink hover:border-line-strong cursor-pointer"
                          >
                            <History size={11} />
                            <span>Timeline</span>
                          </button>
                          <button
                            type="button"
                            onClick={() => handleImageSimilarity(item)}
                            title="Find other tiles visually similar to this one"
                            className="flex items-center justify-center gap-1 rounded-lg border border-line bg-surface-3 px-1.5 py-1.5 font-mono text-[9.5px] text-ink-muted transition-colors hover:text-ink hover:border-line-strong cursor-pointer"
                          >
                            <Sparkles size={11} />
                            <span>Similar</span>
                          </button>
                        </div>

                        {/* Analyst Relevance Feedback Controls (§2.2.5) */}
                        <div className="mt-2.5 flex items-center justify-between border-t border-line/50 pt-2 font-mono text-[10px]" onClick={(e) => e.stopPropagation()}>
                          <span className="text-ink-faint text-[9px] uppercase tracking-wider">Feedback:</span>
                          <div className="flex items-center gap-1">
                            <button
                              type="button"
                              onClick={() => toggleConfirmTile(item.tile_id)}
                              title="Mark as Relevant (Rocchio boost)"
                              className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[9.5px] font-semibold border transition-all cursor-pointer ${
                                confirmedTileIds.includes(item.tile_id)
                                  ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300 font-bold shadow-[0_0_6px_rgba(16,185,129,0.3)]'
                                  : 'border-line bg-surface-3 text-ink-muted hover:border-emerald-500/50 hover:text-emerald-400'
                              }`}
                            >
                              <ThumbsUp size={10} />
                              <span>Relevant</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => toggleRejectTile(item.tile_id)}
                              title="Mark as Irrelevant (Rocchio penalty)"
                              className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[9.5px] font-semibold border transition-all cursor-pointer ${
                                rejectedTileIds.includes(item.tile_id)
                                  ? 'border-rose-500 bg-rose-500/20 text-rose-300 font-bold shadow-[0_0_6px_rgba(244,63,94,0.3)]'
                                  : 'border-line bg-surface-3 text-ink-muted hover:border-rose-500/50 hover:text-rose-400'
                              }`}
                            >
                              <ThumbsDown size={10} />
                              <span>Irrelevant</span>
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Archive Clusters View */}
        {activeTab === 'clusters' && (
          <div className="flex-1 overflow-y-auto p-5">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-muted font-mono">
                    Unsupervised Semantic Archetypes (Spherical k-Means)
                  </h3>
                  <span className="inline-flex items-center gap-1 rounded-full bg-accent/15 border border-accent/30 px-2 py-0.5 text-[9.5px] font-mono text-accent">
                    <Sparkles size={10} />
                    <span>Qwen3-VL Audited</span>
                  </span>
                </div>
                <p className="text-[11.5px] text-ink-faint mt-0.5">
                  Scenes automatically grouped by foundation visual embeddings and interpreted into domain-rich Earth Observation archetypes.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {clustersData?.clusters.map((c) => (
                <div
                  key={c.cluster_id}
                  className="flex flex-col justify-between overflow-hidden rounded-xl border border-line bg-surface-2 p-3.5 transition-all hover:border-accent/40 hover:shadow-lg"
                >
                  <div>
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="min-w-0 flex-1">
                        <h4 className="text-xs font-semibold text-ink leading-snug line-clamp-2" title={c.archetype_title || c.label}>
                          {c.archetype_title || c.label}
                        </h4>
                        <p className="font-mono text-[10px] text-ink-faint mt-0.5">
                          {c.count} scenes ({c.percentage}%) · Cluster #{c.cluster_id}
                        </p>
                      </div>
                      {c.is_audited && (
                        <span className="shrink-0 rounded bg-accent/15 px-1.5 py-0.5 font-mono text-[9px] text-accent flex items-center gap-0.5">
                          <ShieldCheck size={11} />
                          <span>Audited</span>
                        </span>
                      )}
                    </div>

                    {c.exemplar_thumbnail && (
                      <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-space-black/50 mb-2.5">
                        <img
                          src={resolveAssetUrl(c.exemplar_thumbnail) || c.exemplar_thumbnail}
                          alt={c.label}
                          className="h-full w-full object-cover"
                          loading="lazy"
                        />
                        <div className="absolute bottom-1 right-1.5 rounded bg-space-black/80 px-1.5 py-0.5 font-mono text-[9px] text-ink-muted">
                          Exemplar: {c.exemplar_sensor}
                        </div>
                      </div>
                    )}

                    {/* Archetype Description */}
                    {c.archetype_description && (
                      <div className="mb-2.5 rounded-lg border-l-2 border-accent bg-ground/50 p-2 text-[10.5px] text-ink-muted leading-relaxed">
                        {c.archetype_description}
                      </div>
                    )}

                    {/* Dominant Feature Chips */}
                    {c.dominant_features && c.dominant_features.length > 0 && (
                      <div className="mb-3 flex flex-wrap gap-1">
                        {c.dominant_features.map((feat, fIdx) => (
                          <span
                            key={fIdx}
                            className="rounded-md bg-surface-3 border border-line/60 px-1.5 py-0.5 font-mono text-[9.5px] text-ink-muted"
                          >
                            #{feat}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Inline quick preview drawer if expanded */}
                  {expandedClusterId === c.cluster_id && (
                    <div className="my-2 rounded-lg border border-line bg-surface-3 p-2 animate-in fade-in">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-mono text-[10px] text-ink-faint uppercase">Cluster Members:</span>
                        <span className="font-mono text-[10px] text-accent">{clusterTilesCache[c.cluster_id]?.length || 0} loaded</span>
                      </div>
                      {loadingClusterTiles === c.cluster_id ? (
                        <div className="flex h-16 items-center justify-center">
                          <Loader2 size={16} className="animate-spin text-accent" />
                        </div>
                      ) : (
                        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-none">
                          {(clusterTilesCache[c.cluster_id] || []).map((t) => (
                            <div key={t.tile_id} className="w-24 shrink-0 rounded border border-line/70 bg-surface-2 p-1 text-[9px]">
                              <div
                                onClick={() => handleDirectSelectForMap(t)}
                                className="relative aspect-video w-full overflow-hidden rounded bg-black cursor-pointer"
                                title="Overlay on Map"
                              >
                                {t.thumbnail_url && (
                                  <img src={resolveAssetUrl(t.thumbnail_url) || t.thumbnail_url} alt={t.filename} className="h-full w-full object-cover" />
                                )}
                              </div>
                              <span className="truncate block font-mono text-ink mt-0.5" title={t.filename}>{t.filename}</span>
                              <div className="flex gap-1 mt-1">
                                <button
                                  type="button"
                                  onClick={() => handleDirectSelectForMap(t)}
                                  className="flex-1 rounded bg-accent/20 text-accent text-center font-mono py-0.5 hover:bg-accent/30 cursor-pointer"
                                  title="Overlay on map"
                                >
                                  Map
                                </button>
                                <button
                                  type="button"
                                  onClick={() => handleLoadTimeline(t)}
                                  className="flex-1 rounded bg-surface text-ink-muted text-center font-mono py-0.5 hover:text-ink cursor-pointer"
                                  title="Load timeline"
                                >
                                  TL
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Dual Action Buttons for Cluster */}
                  <div className="mt-auto flex items-center gap-1.5 pt-2">
                    <button
                      type="button"
                      onClick={() => handleToggleClusterQuickPreview(c.cluster_id)}
                      className="flex items-center justify-center gap-1 rounded-lg border border-line bg-surface-3 px-2 py-1.5 text-xs text-ink-muted transition-colors hover:text-ink cursor-pointer"
                      title="Quick preview member scenes"
                    >
                      {expandedClusterId === c.cluster_id ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                      <span className="font-mono text-[10px]">Preview</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedClusterId(c.cluster_id);
                        setQuery('');
                        setReferenceTile(null);
                        setActiveTab('search');
                        triggerSearch('', null, c.cluster_id);
                      }}
                      className="flex-1 flex items-center justify-center gap-1.5 rounded-lg border border-accent/40 bg-accent/15 py-1.5 text-xs text-accent transition-colors hover:bg-accent/25 font-medium cursor-pointer"
                    >
                      <span>View All {c.count} Scenes</span>
                      <ChevronRight size={13} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 3: Physical Anomalies View */}
        {activeTab === 'anomalies' && (
          <div className="flex-1 overflow-y-auto p-5">
            <div className="mb-4">
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-400 font-mono flex items-center gap-1.5">
                  <AlertTriangle size={14} />
                  Detected Radiometric & Surface Anomalies
                </h3>
                <span className="rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2 py-0.5 text-[9.5px] font-mono">
                  Qwen3-VL Diagnosed
                </span>
              </div>
              <p className="text-[11.5px] text-ink-faint mt-0.5">
                Outlier scenes exhibiting significant cosine distance from cluster centroids, diagnosed with physical root-cause attributions.
              </p>
            </div>

            {clustersData?.anomalies && clustersData.anomalies.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {clustersData.anomalies.map((anom) => (
                  <div
                    key={anom.tile_id}
                    className={`relative flex flex-col justify-between overflow-hidden rounded-xl border p-3.5 transition-all shadow-sm hover:shadow-lg ${
                      isSelected(anom.tile_id)
                        ? 'border-accent bg-accent/10 shadow-[0_0_12px_rgba(0,242,255,0.2)]'
                        : 'border-amber-500/30 bg-surface-2 hover:border-amber-500/60'
                    }`}
                  >
                    {/* Checkbox for Multi-Selection */}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleSelectTile(anom);
                      }}
                      className={`absolute top-2 left-2 z-20 flex h-5 w-5 items-center justify-center rounded border transition-all cursor-pointer ${
                        isSelected(anom.tile_id)
                          ? 'bg-accent border-accent text-space-black font-bold'
                          : 'bg-space-black/70 border-white/40 text-transparent hover:border-white hover:bg-space-black/90'
                      }`}
                      title={isSelected(anom.tile_id) ? 'Deselect anomaly' : 'Select anomaly for comparison / timeline'}
                    >
                      <Check size={12} strokeWidth={3} className={isSelected(anom.tile_id) ? 'block' : 'hidden'} />
                    </button>

                    <div className="mb-2">
                      <div className="flex items-start justify-between gap-2">
                        <h4 className="text-xs font-semibold text-ink leading-snug line-clamp-1 pl-6" title={anom.anomaly_title || anom.filename}>
                          {anom.anomaly_title || anom.filename}
                        </h4>
                        <span className="shrink-0 rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-[9px] text-amber-300">
                          dist {anom.anomaly_score.toFixed(3)}
                        </span>
                      </div>
                      <p className="truncate font-mono text-[10px] text-ink-faint mt-0.5" title={anom.filename}>
                        {anom.filename} · {anom.date}
                      </p>
                    </div>

                    {/* Diagnostic Badges Row */}
                    <div className="mb-2.5 flex flex-wrap items-center gap-1.5">
                      <span className={`rounded-md border px-2 py-0.5 font-mono text-[9.5px] font-medium ${getCategoryStyle(anom.anomaly_category)}`}>
                        {anom.anomaly_category || 'Statistical Outlier'}
                      </span>
                      {anom.anomaly_severity && (
                        <span className={`rounded-md px-1.5 py-0.5 font-mono text-[9px] ${
                          anom.anomaly_severity.toLowerCase() === 'critical'
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30 font-semibold'
                            : 'bg-amber-500/15 text-amber-300 border border-amber-500/20'
                        }`}>
                          {anom.anomaly_severity}
                        </span>
                      )}
                      {anom.anomaly_confidence !== undefined && (
                        <span className="rounded-md bg-surface-3 border border-line px-1.5 py-0.5 font-mono text-[9px] text-ink-muted">
                          {Math.round(anom.anomaly_confidence * 100)}% Conf.
                        </span>
                      )}
                    </div>

                    {anom.thumbnail_url && (
                      <div
                        onClick={() => handleDirectSelectForMap(anom)}
                        className="relative aspect-video w-full overflow-hidden rounded-lg bg-space-black/50 mb-2.5 cursor-pointer"
                        title="Click to overlay anomaly on map"
                      >
                        <img
                          src={resolveAssetUrl(anom.thumbnail_url) || anom.thumbnail_url}
                          alt={anom.filename}
                          className="h-full w-full object-cover"
                        />
                        <div className="absolute bottom-1 right-1.5 rounded bg-space-black/80 px-1.5 py-0.5 font-mono text-[9px] text-ink-muted">
                          {anom.sensor}
                        </div>
                      </div>
                    )}

                    {/* Physical Explanation Box */}
                    <div className="mb-3 rounded-lg border border-amber-500/25 bg-amber-950/20 p-2.5 text-[10.5px] text-amber-200/90 leading-relaxed font-sans">
                      {anom.anomaly_reason || 'Spectral distance outlier exceeding cluster baseline threshold.'}
                    </div>

                    <div className="flex items-center gap-2 mt-auto">
                      <button
                        type="button"
                        onClick={() => handleDirectSelectForMap(anom)}
                        className="flex-1 flex items-center justify-center gap-1 rounded-lg border border-accent bg-accent/20 py-1.5 font-mono text-xs font-semibold text-accent hover:bg-accent/30 transition-colors cursor-pointer"
                        title="Overlay satellite anomaly image on map canvas"
                      >
                        <MapPin size={12} />
                        <span>Map Overlay</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleLoadTimeline(anom)}
                        className="flex items-center justify-center gap-1 rounded-lg border border-amber-500/40 bg-amber-500/15 px-3 py-1.5 font-mono text-xs text-amber-300 hover:bg-amber-500/25 transition-colors cursor-pointer"
                      >
                        <History size={12} />
                        <span>Timeline</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex h-48 items-center justify-center text-ink-faint text-xs">
                No extreme anomalies detected in the current archive.
              </div>
            )}
          </div>
        )}

        {/* Floating Multi-Selection Action Dock */}
        {selectedTiles.length > 0 && (
          <div className="shrink-0 flex items-center justify-between border-t border-accent/40 bg-surface-2/95 px-5 py-3 backdrop-blur-md shadow-2xl animate-in slide-in-from-bottom-2">
            <div className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent text-space-black font-mono text-xs font-bold">
                {selectedTiles.length}
              </span>
              <span className="text-xs font-medium text-ink">
                {selectedTiles.length === 1 ? '1 scene selected' : `${selectedTiles.length} scenes selected`}
              </span>
              <button
                type="button"
                onClick={() => setSelectedTiles([])}
                className="ml-2 text-[11px] font-mono text-ink-faint hover:text-ink underline cursor-pointer"
              >
                Clear
              </button>
            </div>

            <div className="flex items-center gap-2">
              {selectedTiles.length === 1 && (
                <button
                  type="button"
                  onClick={() => handleDirectSelectForMap(selectedTiles[0])}
                  className="flex items-center gap-1.5 rounded-lg border border-accent bg-accent/20 px-3.5 py-1.5 font-mono text-xs font-semibold text-accent hover:bg-accent/30 transition-colors shadow-sm cursor-pointer"
                >
                  <MapPin size={13} />
                  <span>Overlay on Map</span>
                </button>
              )}

              {selectedTiles.length === 2 && onLoadComparisonPair && (
                <button
                  type="button"
                  onClick={() => {
                    onLoadComparisonPair(selectedTiles[0], selectedTiles[1]);
                    onClose();
                  }}
                  className="flex items-center gap-1.5 rounded-lg border border-cyan-400 bg-cyan-500/20 px-3.5 py-1.5 font-mono text-xs font-semibold text-cyan-300 hover:bg-cyan-500/30 transition-colors shadow-sm cursor-pointer"
                >
                  <SplitSquareVertical size={13} />
                  <span>Compare as Bi-Temporal Pair (Swipe)</span>
                </button>
              )}

              {selectedTiles.length >= 2 && (
                <button
                  type="button"
                  onClick={() => {
                    if (onLoadTimelineStack) {
                      onLoadTimelineStack(selectedTiles, selectedTiles[0]);
                    }
                    onClose();
                  }}
                  className="flex items-center gap-1.5 rounded-lg border border-accent bg-accent px-3.5 py-1.5 font-mono text-xs font-semibold text-space-black hover:bg-accent/90 transition-colors shadow-[0_0_12px_rgba(0,242,255,0.3)] cursor-pointer"
                >
                  <History size={13} />
                  <span>Load Multi-Temporal Timeline ({selectedTiles.length})</span>
                </button>
              )}
            </div>
          </div>
        )}

        {/* Modal: Scope to Local Folder / Pen-Drive */}
        {showFolderModal && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="w-full max-w-lg rounded-2xl border border-line bg-surface-1 p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
              <div className="flex items-center justify-between border-b border-line pb-3">
                <div className="flex items-center gap-2.5">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    <HardDrive size={18} />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-ink">Scope to Local Folder / Pen-Drive</h3>
                    <p className="text-[11px] text-ink-muted">Offline or online ingestion of any satellite folder</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setShowFolderModal(false);
                    setFolderIngestError(null);
                    setFolderIngestSuccess(null);
                  }}
                  className="rounded-lg p-1 text-ink-muted hover:text-ink hover:bg-surface-2 transition-colors cursor-pointer"
                >
                  <X size={16} />
                </button>
              </div>

              <div className="space-y-3">
                <p className="text-xs text-ink-muted leading-relaxed">
                  Provide the absolute path to a local directory or connected USB pen-drive containing satellite imagery (<span className="text-ink font-mono">.tif, .tiff, .png, .jpg</span>). SatQuery will dynamically index these scenes for semantic search, clustering, and change detection.
                </p>

                {clustersData?.is_scoped_to_folder && clustersData?.active_folder_path && (
                  <div className="flex items-center justify-between p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs">
                    <div className="space-y-0.5 truncate pr-2">
                      <span className="text-[10px] uppercase font-mono text-emerald-400 font-semibold">Active Folder Scope:</span>
                      <p className="font-mono text-[11px] text-ink truncate" title={clustersData.active_folder_path}>
                        {clustersData.active_folder_path}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={async () => {
                        await handleResetScope();
                        setShowFolderModal(false);
                      }}
                      className="shrink-0 px-2.5 py-1 text-[11px] font-medium rounded-md bg-surface-2 border border-line text-ink-muted hover:text-ink hover:border-line-strong transition-colors cursor-pointer"
                    >
                      Reset to All
                    </button>
                  </div>
                )}

                <div className="space-y-1.5">
                  <label className="text-[11px] font-medium text-ink-muted uppercase tracking-wider">
                    Folder or Pen-Drive Path
                  </label>
                  <input
                    type="text"
                    value={folderPathInput}
                    onChange={(e) => setFolderPathInput(e.target.value)}
                    placeholder="e.g. E:\Judges_Scenes or D:\SatelliteData\DemoAOI"
                    className="w-full rounded-lg border border-line bg-surface-2 px-3 py-2 text-xs font-mono text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleIngestFolder();
                    }}
                  />
                  <p className="text-[10px] text-ink-faint">
                    Tip: For USB drives, use the drive letter (e.g. <span className="font-mono text-ink-muted">E:\</span> or <span className="font-mono text-ink-muted">F:\ISRO_EVAL</span>)
                  </p>
                </div>

                {folderIngestError && (
                  <div className="flex items-center gap-2 p-2.5 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-xs">
                    <AlertTriangle size={14} className="shrink-0 text-red-400" />
                    <span>{folderIngestError}</span>
                  </div>
                )}

                {folderIngestSuccess && (
                  <div className="flex items-center gap-2 p-2.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs">
                    <Check size={14} className="shrink-0 text-emerald-400" />
                    <span>{folderIngestSuccess}</span>
                  </div>
                )}
              </div>

              <div className="flex items-center justify-end gap-2 border-t border-line pt-3">
                <button
                  type="button"
                  onClick={() => {
                    setShowFolderModal(false);
                    setFolderIngestError(null);
                    setFolderIngestSuccess(null);
                  }}
                  className="px-3 py-1.5 text-xs text-ink-muted hover:text-ink rounded-lg hover:bg-surface-2 transition-colors cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleIngestFolder}
                  disabled={ingestingFolder || !folderPathInput.trim()}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 px-4 py-1.5 text-xs font-semibold text-white transition-colors disabled:opacity-50 cursor-pointer shadow-sm"
                >
                  {ingestingFolder ? (
                    <>
                      <Loader2 size={13} className="animate-spin" />
                      <span>Scanning & Indexing...</span>
                    </>
                  ) : (
                    <>
                      <FolderPlus size={13} />
                      <span>Ingest Folder</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
