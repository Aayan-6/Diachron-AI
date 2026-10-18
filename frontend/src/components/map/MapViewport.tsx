import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import {
  ArrowLeftRight,
  Building2,
  Calendar,
  Check,
  ChevronLeft,
  ChevronRight,
  Compass,
  Crop,
  Crosshair,
  Eye,
  EyeOff,
  Flag,
  Focus,
  Globe,
  Layers,
  Loader2,
  Locate,
  MapPin,
  Maximize2,
  Minimize2,
  Palette,
  Pause,
  Play,
  Search,
  Sparkles,
  X,
  ZoomIn,
  ZoomOut
} from 'lucide-react';
import { IconButton } from '../ui/Button';
import { useApp } from '../../context/AppState';
import { resolveAssetUrl } from '../../lib/api';
import type { ImageOverlayEvidence, SceneOverlay, SceneDataset, TimelineObservation } from '../../lib/types';
import {
  searchLocations,
  parseCoordinates,
  type GeocodingResult
} from '../../lib/geocoding';
import { AOIBoxSelector } from './aoi/AOIBoxSelector';
import { AOIFetchModal } from './aoi/AOIFetchModal';
import { AOIChatbotHandoff } from './aoi/AOIChatbotHandoff';
import { ArchiveSearchModal } from '../archive/ArchiveSearchModal';
import {
  type FetchAOIResponse,
  type FetchBiTemporalAOIResponse,
  aoiResponseToSceneOverlay,
  aoiPairResponseToSceneOverlay,
  aoiPairResponseToSceneDataset,
} from '../../lib/interactiveMapApi';

// Feature Flag: Interactive Map AOI Streamer (Phase 2)
const ENABLE_AOI_FETCHER = true;

// Production-grade global basemaps (Google Maps style, 0 API keys required)
const SATELLITE_STYLE: any = {
  version: 8,
  sources: {
    'esri-satellite': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri, Maxar',
      maxzoom: 19
    }
  },
  layers: [
    {
      id: 'esri-satellite-layer',
      type: 'raster',
      source: 'esri-satellite',
      minzoom: 0,
      maxzoom: 19
    }
  ]
};

const HYBRID_STYLE: any = {
  version: 8,
  sources: {
    'esri-satellite': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri, Maxar',
      maxzoom: 19
    },
    'esri-boundaries': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri',
      maxzoom: 19
    }
  },
  layers: [
    {
      id: 'esri-satellite-layer',
      type: 'raster',
      source: 'esri-satellite',
      minzoom: 0,
      maxzoom: 19
    },
    {
      id: 'esri-boundaries-layer',
      type: 'raster',
      source: 'esri-boundaries',
      minzoom: 0,
      maxzoom: 19
    }
  ]
};

const STREETS_STYLE: any = {
  version: 8,
  sources: {
    'osm-streets': {
      type: 'raster',
      tiles: [
        'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
      ],
      tileSize: 256,
      attribution: '&copy; OpenStreetMap contributors',
      maxzoom: 19
    }
  },
  layers: [
    {
      id: 'osm-streets-layer',
      type: 'raster',
      source: 'osm-streets',
      minzoom: 0,
      maxzoom: 19
    }
  ]
};

const DARK_TACTICAL_STYLE: any = {
  version: 8,
  sources: {
    'esri-dark-base': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri, HERE, Garmin',
      maxzoom: 16
    },
    'esri-dark-ref': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}'
      ],
      tileSize: 256,
      attribution: '&copy; Esri',
      maxzoom: 16
    }
  },
  layers: [
    {
      id: 'esri-dark-base-layer',
      type: 'raster',
      source: 'esri-dark-base',
      minzoom: 0,
      maxzoom: 16
    },
    {
      id: 'esri-dark-ref-layer',
      type: 'raster',
      source: 'esri-dark-ref',
      minzoom: 0,
      maxzoom: 16
    }
  ]
};

// 100% Zero-Network Air-Gapped Tactical Basemap (Graticule Grid & Dark Blueprint Canvas)
const TACTICAL_OFFLINE_STYLE: any = {
  version: 8,
  sources: {
    'tactical-graticule': {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: [
          ...Array.from({ length: 37 }, (_, i) => -180 + i * 10).map((lng) => ({
            type: 'Feature',
            geometry: { type: 'LineString', coordinates: [[lng, -85], [lng, 85]] },
            properties: { type: 'meridian' }
          })),
          ...Array.from({ length: 17 }, (_, i) => -80 + i * 10).map((lat) => ({
            type: 'Feature',
            geometry: { type: 'LineString', coordinates: [[-180, lat], [180, lat]] },
            properties: { type: 'parallel' }
          }))
        ]
      }
    }
  },
  layers: [
    {
      id: 'tactical-bg',
      type: 'background',
      paint: { 'background-color': '#060B13' }
    },
    {
      id: 'tactical-graticule-lines',
      type: 'line',
      source: 'tactical-graticule',
      paint: {
        'line-color': '#112233',
        'line-width': 1,
        'line-dasharray': [2, 3]
      }
    }
  ]
};

// Preset bookmarks for instant exploration
const PRESET_BOOKMARKS = [
  { name: 'India (Subcontinent)', coords: [78.9629, 20.5937] as [number, number], zoom: 4.8 },
  { name: 'Bengaluru Urban', coords: [77.5946, 12.9716] as [number, number], zoom: 13.5 },
  { name: 'Delhi NCR Region', coords: [77.2090, 28.6139] as [number, number], zoom: 12.5 },
  { name: 'Mumbai Coastline', coords: [72.8777, 19.0760] as [number, number], zoom: 12.5 },
  { name: 'Himalayas / Siachen', coords: [77.1000, 35.3000] as [number, number], zoom: 9.5 },
  { name: 'Dubai Palm Jumeirah', coords: [55.1388, 25.1124] as [number, number], zoom: 13.0 },
  { name: 'Suez Canal (Egypt)', coords: [32.3436, 30.7050] as [number, number], zoom: 11.5 },
  { name: 'Global Earth View', coords: [15.0, 25.0] as [number, number], zoom: 2.2 },
];

export type BaseLayerType = 'satellite' | 'hybrid' | 'streets' | 'sar' | 'tactical-offline';
export type BandFilterType = 'true-color' | 'false-color-nir' | 'sar-contrast' | 'edge-boost';
export type TemporalModeType = 'swipe' | 'fade' | 't1' | 't2';

/** How much chrome the map shows. `compact` is for the side dock. */
export type MapDensity = 'full' | 'compact';

/**
 * Analysis rasters (change masks) are drawn through a single image source that
 * is created once at map load and only ever updated afterwards.
 *
 * This map never reports a settled style on maplibre-gl 6 — `getStyle()` stays
 * undefined and `addSource` throws "Style is not done loading" indefinitely —
 * so anything added lazily never lands. Creating the source up front with a
 * transparent pixel sidesteps that entirely: `updateImage` and the paint/layout
 * setters work regardless of style state.
 */
const RESULT_RASTER_SOURCE = 'satquery-result-raster';
const RESULT_RASTER_LAYER = 'satquery-result-raster-layer';
const TRANSPARENT_PIXEL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
const PLACEHOLDER_COORDS: [[number, number], [number, number], [number, number], [number, number]] = [
  [-0.001, 0.001],
  [0.001, 0.001],
  [0.001, -0.001],
  [-0.001, -0.001]
];

const BASE_LAYERS: Array<{ value: BaseLayerType; label: string }> = [
  { value: 'satellite', label: 'Satellite' },
  { value: 'hybrid', label: 'Hybrid' },
  { value: 'streets', label: 'Streets' },
  { value: 'sar', label: 'SAR dark' },
  { value: 'tactical-offline', label: 'Tactical (Air-Gapped)' }
];

const BAND_PRESETS: Array<{ value: BandFilterType; label: string }> = [
  { value: 'true-color', label: 'True colour' },
  { value: 'false-color-nir', label: 'False colour (NIR)' },
  { value: 'sar-contrast', label: 'SAR contrast' },
  { value: 'edge-boost', label: 'Edge boost' }
];

const TEMPORAL_MODES: Array<{ value: TemporalModeType; label: string; title: string }> = [
  { value: 't1', label: 'T1', title: 'Show the earlier acquisition only' },
  { value: 'swipe', label: 'Swipe', title: 'Split the viewport between T1 and T2' },
  { value: 'fade', label: 'Fade', title: 'Cross-fade between T1 and T2' },
  { value: 't2', label: 'T2', title: 'Show the later acquisition only' }
];

const LEGEND_ENTRIES: Array<{ label: string; swatch: string; meaning: string }> = [
  {
    label: 'Deep blue / black',
    swatch: '#0b2545',
    meaning: 'Water, wet soil, or radar shadow — very low reflectance.'
  },
  {
    label: 'Green',
    swatch: '#2f8f4e',
    meaning: 'Healthy vegetation. Brighter means denser canopy.'
  },
  {
    label: 'Grey / white',
    swatch: '#b8c0cc',
    meaning: 'Built-up surfaces, roads, bare rock, or cloud.'
  },
  {
    label: 'Red outline',
    swatch: '#ff3b30',
    meaning: 'A region the change detector flagged between T1 and T2.'
  }
];

export interface MapViewportProps {
  /** Detections to draw, straight from the backend's visual_evidence. */
  geoJsonData?: any;
  datasetName?: string;
  sensor?: string;
  /** Explicit scene; falls back to whatever the workspace has loaded. */
  overlay?: SceneOverlay | null;
  /**
   * A georeferenced raster produced by an analysis (e.g. the change mask),
   * already warped to Web Mercator by the backend so its corners place it
   * exactly.
   */
  resultOverlay?: ImageOverlayEvidence | null;
  density?: MapDensity;
}

export const MapViewport: React.FC<MapViewportProps> = ({
  geoJsonData,
  datasetName,
  sensor,
  overlay: propOverlay,
  resultOverlay = null,
  density = 'full'
}) => {
  const {
    overlay: contextOverlay,
    setScene,
    sendQuery,
    openDock,
    activeTimelineStep,
    setActiveTimelineStep,
    focusedTimeline,
    activeSiteId,
    setActiveSiteId,
    analystReviews,
    focusedCandidateId,
    setFocusedCandidateId,
  } = useApp();
  const activeOverlay = propOverlay !== undefined ? propOverlay : contextOverlay;

  const analystReviewsRef = useRef(analystReviews);
  analystReviewsRef.current = analystReviews;
  const setFocusedCandidateIdRef = useRef(setFocusedCandidateId);
  setFocusedCandidateIdRef.current = setFocusedCandidateId;
  const prevGeoJsonDataRef = useRef<any>(null);
  const activeDetectionFeaturesRef = useRef<any[]>([]);

  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  // Fullscreen targets the whole component, not just the WebGL canvas, so the
  // controls come along with it.
  const rootRef = useRef<HTMLDivElement>(null);

  // Basemap & Viewport State
  const [activeBaseLayer, setActiveBaseLayer] = useState<BaseLayerType>('satellite');
  const [zoomLevel, setZoomLevel] = useState<number>(4.8);
  const [bearing, setBearing] = useState<number>(0);
  const [pitch, setPitch] = useState<number>(0);
  const [centerCoords, setCenterCoords] = useState<{ lat: number; lng: number }>({
    lat: 20.5937,
    lng: 78.9629
  });
  const [cursorCoords, setCursorCoords] = useState<{ lat: number; lng: number } | null>(null);

  // Sentinel Overlay Controls
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.92);
  const [temporalMode, setTemporalMode] = useState<TemporalModeType>('fade');
  const [temporalSlider, setTemporalSlider] = useState<number>(50); // 0 (100% T1) to 100 (100% T2)
  const [bandPreset, setBandPreset] = useState<BandFilterType>('true-color');
  const [isBlinking, setIsBlinking] = useState<boolean>(false);
  const [showOverlayControls, setShowOverlayControls] = useState<boolean>(true);
  const [isOverlayVisible, setIsOverlayVisible] = useState<boolean>(true);
  const [showColorLegend, setShowColorLegend] = useState<boolean>(false);

  // Authoritative NIR capability assessment based on current temporalMode and loaded imagery
  const nirCapability = useMemo(() => {
    if (!activeOverlay) {
      return {
        available: false,
        reason: 'No imagery loaded',
        t1Available: false,
        t2Available: false,
        jointAvailable: false,
        asymmetricNote: undefined as string | undefined,
      };
    }

    const t1HasNir = Boolean(
      activeOverlay.bandContract?.t1?.can_false_color_nir ||
      activeOverlay.bandContract?.capabilities?.t1_false_color_nir ||
      activeOverlay.t1NirImageUrl
    );
    const t2HasNir = Boolean(
      activeOverlay.bandContract?.t2?.can_false_color_nir ||
      activeOverlay.bandContract?.capabilities?.t2_false_color_nir ||
      activeOverlay.t2NirImageUrl
    );
    const jointAvailable = Boolean(
      activeOverlay.bandContract?.joint?.common_nir ||
      activeOverlay.bandContract?.capabilities?.joint_false_color_nir
    );

    if (activeOverlay.mode === 'multi-temporal') {
      const activeIdx = Math.min(Math.max(0, activeTimelineStep), (activeOverlay.scenes?.length || 1) - 1);
      const activeScene = activeOverlay.scenes?.[activeIdx];
      const hasNir = Boolean(activeScene?.nirImageUrl || activeOverlay.t1NirImageUrl);
      return {
        available: hasNir,
        reason: hasNir ? 'Available (NIR band detected for active observation)' : 'Unavailable: NIR band absent in active observation',
        t1Available: hasNir,
        t2Available: false,
        jointAvailable: false,
        asymmetricNote: undefined as string | undefined,
      };
    }

    if (activeOverlay.mode === 'single') {
      return {
        available: t1HasNir,
        reason: t1HasNir ? 'Available (NIR band detected)' : 'Unavailable: NIR band absent',
        t1Available: t1HasNir,
        t2Available: false,
        jointAvailable: false,
        asymmetricNote: undefined as string | undefined,
      };
    }

    if (temporalMode === 't1') {
      return {
        available: t1HasNir,
        reason: t1HasNir ? 'Available (T1 NIR band B8)' : 'Unavailable: NIR absent in T1',
        t1Available: t1HasNir,
        t2Available: t2HasNir,
        jointAvailable,
        asymmetricNote: undefined as string | undefined,
      };
    }

    if (temporalMode === 't2') {
      return {
        available: t2HasNir,
        reason: t2HasNir ? 'Available (T2 NIR band)' : 'Unavailable: NIR absent in T2',
        t1Available: t1HasNir,
        t2Available: t2HasNir,
        jointAvailable,
        asymmetricNote: undefined as string | undefined,
      };
    }

    // swipe or fade:
    const eitherHasNir = t1HasNir || t2HasNir;
    return {
      available: eitherHasNir,
      reason: jointAvailable
        ? 'Available across both dates'
        : t1HasNir && !t2HasNir
        ? 'T1 NIR available (T2 rendered in True Colour)'
        : !t1HasNir && t2HasNir
        ? 'T2 NIR available (T1 rendered in True Colour)'
        : 'Unavailable: NIR absent in both dates',
      t1Available: t1HasNir,
      t2Available: t2HasNir,
      jointAvailable,
      asymmetricNote: t1HasNir && !t2HasNir ? 'Bi-temporal comparison note: T2 lacks NIR band' : undefined,
    };
  }, [activeOverlay, temporalMode, activeTimelineStep]);

  // Derived multi-temporal timeline for timeline slider widget
  const timelineData: TimelineObservation[] | null = useMemo(() => {
    if (focusedTimeline && focusedTimeline.length > 1) {
      return focusedTimeline;
    }
    if (activeOverlay?.scenes && activeOverlay.scenes.length > 1) {
      return activeOverlay.scenes.map((s, idx) => ({
        step: idx,
        step_index: idx,
        date: s.date || s.acquisition_date || s.filename || `T${idx + 1}`,
        filename: s.filename,
        image_url: s.image_url || (s as any).imageUrl || null,
        imageUrl: s.image_url || (s as any).imageUrl || null,
        is_baseline: idx === 0,
        change_area_ha: 0,
        change_fraction: 0,
        n_polygons: 0,
      }));
    }
    return null;
  }, [focusedTimeline, activeOverlay]);

  // Multi-temporal identification & auto-playback controls
  const isMultiTemporal = Boolean(
    activeOverlay?.mode === 'multi-temporal' ||
    (timelineData && timelineData.length > 1 && activeOverlay?.mode !== 'bi-temporal')
  );

  const [isPlayingTimeline, setIsPlayingTimeline] = useState<boolean>(false);
  const activeTimelineStepRef = useRef(activeTimelineStep);
  useEffect(() => {
    activeTimelineStepRef.current = activeTimelineStep;
  }, [activeTimelineStep]);

  // Auto-advance timeline during playback
  useEffect(() => {
    if (!isPlayingTimeline || !timelineData || timelineData.length <= 1) return;

    const interval = setInterval(() => {
      setActiveTimelineStep((activeTimelineStepRef.current + 1) % timelineData.length);
    }, 1600);

    return () => clearInterval(interval);
  }, [isPlayingTimeline, timelineData, setActiveTimelineStep]);

  // Stop playback when overlay dataset changes
  useEffect(() => {
    setIsPlayingTimeline(false);
  }, [activeOverlay?.datasetId]);

  const currentStep = Math.min(Math.max(0, activeTimelineStep), (timelineData?.length || 1) - 1);
  const activeObs = timelineData?.[currentStep] || timelineData?.[0];
  const isBaseline = Boolean(activeObs?.is_baseline || currentStep === 0);

  const handleTimelineStepChange = (newStep: number) => {
    setIsPlayingTimeline(false);
    setActiveTimelineStep(newStep);
  };

  // When temporal display mode changes to T2, if T2 lacks NIR, safely fallback to true-colour
  useEffect(() => {
    if (temporalMode === 't2' && bandPreset === 'false-color-nir' && !nirCapability.available) {
      setBandPreset('true-color');
    }
  }, [temporalMode, bandPreset, nirCapability.available]);

  // HUD & UI Tools
  const [showReticle, setShowReticle] = useState<boolean>(false);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [detectionLabel, setDetectionLabel] = useState<string>('Global Interactive Slippy Map');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<GeocodingResult[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [showSearchResults, setShowSearchResults] = useState<boolean>(false);
  const searchAbortControllerRef = useRef<AbortController | null>(null);
  const searchContainerRef = useRef<HTMLDivElement>(null);
  const [showBookmarks, setShowBookmarks] = useState<boolean>(false);
  const [showLayers, setShowLayers] = useState<boolean>(false);
  const [bhuvanThematicLayer, setBhuvanThematicLayer] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);

  // Interactive Map STAC/COG AOI Streamer State (Phase 2)
  const [isAOIActive, setIsAOIActive] = useState<boolean>(false);
  const [selectedBbox, setSelectedBbox] = useState<[number, number, number, number] | null>(null);
  const [selectedAreaKm2, setSelectedAreaKm2] = useState<number>(0);
  const [showFetchModal, setShowFetchModal] = useState<boolean>(false);
  const [showArchiveModal, setShowArchiveModal] = useState<boolean>(false);
  const [fetchedScene, setFetchedScene] = useState<FetchAOIResponse | FetchBiTemporalAOIResponse | null>(null);

  // Auto-switch basemap if SAR sensor is provided
  useEffect(() => {
    const isSar = (sensor && sensor.toLowerCase().includes('sar')) ||
                  (activeOverlay && activeOverlay.sensor.toLowerCase().includes('sar'));
    if (isSar && activeBaseLayer !== 'sar') {
      handleBaseLayerChange('sar');
    }
  }, [sensor, activeOverlay, activeBaseLayer]);

  // Blink comparison timer (rapidly toggles T1 and T2 at 1.2 Hz)
  useEffect(() => {
    if (!isBlinking) return;
    const interval = setInterval(() => {
      setTemporalSlider((prev) => (prev > 50 ? 0 : 100));
    }, 750);
    return () => clearInterval(interval);
  }, [isBlinking]);

  // 4. Update the MapLibre layers when visibility toggles change
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    
    // Original bounding box logic
    const vis = showBoundingBoxes ? 'visible' : 'none';
    if (map.getLayer('satquery-fill')) map.setLayoutProperty('satquery-fill', 'visibility', vis);
    if (map.getLayer('satquery-glow')) map.setLayoutProperty('satquery-glow', 'visibility', vis);
    if (map.getLayer('satquery-outline')) map.setLayoutProperty('satquery-outline', 'visibility', vis);
    
    // New heatmap logic
    const heatVis = showHeatmap ? 'visible' : 'none';
    if (map.getLayer('satquery-heatmap')) map.setLayoutProperty('satquery-heatmap', 'visibility', heatVis);
    
  }, [showBoundingBoxes, showHeatmap]);

  // Sync Bhuvan Thematic Map Overlays
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const SOURCE_ID = 'bhuvan-thematic-source';
    const LAYER_ID = 'bhuvan-thematic-layer';

    // Always remove existing first to ensure tile URLs update
    if (map.getLayer(LAYER_ID)) map.removeLayer(LAYER_ID);
    if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID);

    if (bhuvanThematicLayer) {
      map.addSource(SOURCE_ID, {
        type: 'raster',
        tiles: [
          `/bhuvan-wms?service=WMS&version=1.1.1&request=GetMap&layers=${bhuvanThematicLayer}&styles=&format=image/png&transparent=true&srs=EPSG:3857&bbox={bbox-epsg-3857}&width=256&height=256`
        ],
        tileSize: 256
      });
      map.addLayer({
        id: LAYER_ID,
        type: 'raster',
        source: SOURCE_ID,
        paint: {
          'raster-opacity': 0.6
        }
      });
    }
  }, [bhuvanThematicLayer, activeBaseLayer]);


  // Helper to extract bounding box from arbitrary GeoJSON
  const getBounds = useCallback((geojson: any): [[number, number], [number, number]] | null => {
    let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;

    const processCoords = (coords: any) => {
      if (Array.isArray(coords) && typeof coords[0] === 'number' && typeof coords[1] === 'number') {
        const [lng, lat] = coords;
        if (lng < minLng) minLng = lng;
        if (lat < minLat) minLat = lat;
        if (lng > maxLng) maxLng = lng;
        if (lat > maxLat) maxLat = lat;
      } else if (Array.isArray(coords)) {
        coords.forEach(processCoords);
      }
    };

    if (geojson?.type === 'FeatureCollection' && Array.isArray(geojson.features)) {
      geojson.features.forEach((f: any) => processCoords(f?.geometry?.coordinates));
    } else if (geojson?.type === 'Feature') {
      processCoords(geojson?.geometry?.coordinates);
    }

    if (minLng !== Infinity && maxLng !== -Infinity && !isNaN(minLng) && !isNaN(maxLng)) {
      return [[minLng, minLat], [maxLng, maxLat]];
    }
    return null;
  }, []);

  // Update or mount Sentinel raster imagery layers onto MapLibre WebGL canvas
  const updateOverlayLayers = useCallback((
    map: maplibregl.Map,
    ov: SceneOverlay | null,
    opacity: number,
    mode: TemporalModeType,
    sliderVal: number,
    visible: boolean = true,
    filterPreset: BandFilterType = 'true-color'
  ) => {
    if (!ov || !ov.bounds || ov.bounds.length < 4) {
      // Clean up all overlay layers and sources safely
      ['sentinel-raster-layer-t2', 'sentinel-raster-layer-t1', 'sentinel-bbox-glow', 'sentinel-bbox-line', 'spatial-sites-active-line', 'spatial-sites-line', 'spatial-sites-fill'].forEach((id) => {
        if (map.getLayer(id)) map.removeLayer(id);
      });
      ['sentinel-raster-t2', 'sentinel-raster-t1', 'sentinel-bbox-source', 'spatial-sites-source'].forEach((id) => {
        if (map.getSource(id)) map.removeSource(id);
      });
      return;
    }

    const isVisible = visible && opacity > 0.01;
    const visibilityVal = isVisible ? 'visible' : 'none';

    let [minLng, minLat, maxLng, maxLat] = ov.bounds;
    if (ov.mode === 'multi-temporal' && ov.scenes && ov.scenes.length > 0) {
      const activeIdx = Math.min(Math.max(0, activeTimelineStep), ov.scenes.length - 1);
      const activeScene = ov.scenes[activeIdx];
      const scBounds = (activeScene as any)?.wgs84_bounds;
      if (scBounds && Array.isArray(scBounds) && scBounds.length === 4) {
        [minLng, minLat, maxLng, maxLat] = scBounds;
      }
    }
    const imageCoordinates: [[number, number], [number, number], [number, number], [number, number]] = [
      [minLng, maxLat], // Top-Left
      [maxLng, maxLat], // Top-Right
      [maxLng, minLat], // Bottom-Right
      [minLng, minLat], // Bottom-Left
    ];

    // Calculate opacities based on temporal mode
    let t1Opacity = isVisible ? opacity : 0;
    let t2Opacity = 0;

    if (ov.mode === 'bi-temporal' && ov.t2ImageUrl) {
      if (mode === 't1') {
        t1Opacity = isVisible ? opacity : 0;
        t2Opacity = 0;
      } else if (mode === 't2') {
        t1Opacity = 0;
        t2Opacity = isVisible ? opacity : 0;
      } else {
        // fade or swipe blend
        const ratio = sliderVal / 100;
        t1Opacity = isVisible ? opacity * (1 - ratio) : 0;
        t2Opacity = isVisible ? opacity * ratio : 0;
      }
    }

    // Determine insertion position (below AI detection polygons so boxes stay on top)
    const beforeLayerId = map.getLayer('satquery-fill') ? 'satquery-fill' : undefined;

    // Check NIR capabilities for genuine raster selection
    const isT1Nir = filterPreset === 'false-color-nir' && Boolean(ov.bandContract?.t1?.can_false_color_nir || ov.t1NirImageUrl);
    let targetT1Url = (isT1Nir && ov.t1NirImageUrl) ? ov.t1NirImageUrl : ov.t1ImageUrl;
    if (ov.mode === 'multi-temporal' && ov.scenes && ov.scenes.length > 0) {
      const activeIdx = Math.min(Math.max(0, activeTimelineStep), ov.scenes.length - 1);
      const activeScene = ov.scenes[activeIdx];
      if (activeScene) {
        const sceneUrl = activeScene.imageUrl || activeScene.image_url || ov.t1ImageUrl;
        const nirUrl = activeScene.nirImageUrl || activeScene.nir_image_url || ov.t1NirImageUrl;
        targetT1Url = (filterPreset === 'false-color-nir' && nirUrl) ? nirUrl : sceneUrl;
      }
    }

    const isT2Nir = filterPreset === 'false-color-nir' && Boolean(ov.bandContract?.t2?.can_false_color_nir || ov.t2NirImageUrl);
    const targetT2Url = (isT2Nir && ov.t2NirImageUrl) ? ov.t2NirImageUrl : (ov.t2ImageUrl || '');

    // 1. Add or Update T1 Raster Source & Layer
    if (targetT1Url) {
      try {
        const existingT1 = map.getSource('sentinel-raster-t1') as maplibregl.ImageSource | undefined;
        if (existingT1 && typeof existingT1.updateImage === 'function') {
          existingT1.updateImage({
            url: targetT1Url,
            coordinates: imageCoordinates
          });
          if (map.getLayer('sentinel-raster-layer-t1')) {
            map.setLayoutProperty('sentinel-raster-layer-t1', 'visibility', visibilityVal);
            map.setPaintProperty('sentinel-raster-layer-t1', 'raster-opacity', t1Opacity);
          }
        } else {
          if (map.getLayer('sentinel-raster-layer-t1')) map.removeLayer('sentinel-raster-layer-t1');
          if (map.getSource('sentinel-raster-t1')) map.removeSource('sentinel-raster-t1');

          map.addSource('sentinel-raster-t1', {
            type: 'image',
            url: targetT1Url,
            coordinates: imageCoordinates
          });
          map.addLayer(
            {
              id: 'sentinel-raster-layer-t1',
              type: 'raster',
              source: 'sentinel-raster-t1',
              layout: {
                visibility: visibilityVal
              },
              paint: {
                'raster-opacity': t1Opacity,
                'raster-fade-duration': 100
              }
            },
            beforeLayerId
          );
        }
      } catch (err) {
        console.warn('Sentinel T1 overlay update notice:', err);
      }
    } else {
      if (map.getLayer('sentinel-raster-layer-t1')) map.removeLayer('sentinel-raster-layer-t1');
      if (map.getSource('sentinel-raster-t1')) map.removeSource('sentinel-raster-t1');
    }

    // 2. Add or Update T2 Raster Source & Layer if bi-temporal
    if (ov.mode === 'bi-temporal' && targetT2Url) {
      try {
        const existingT2 = map.getSource('sentinel-raster-t2') as maplibregl.ImageSource | undefined;
        if (existingT2 && typeof existingT2.updateImage === 'function') {
          existingT2.updateImage({
            url: targetT2Url,
            coordinates: imageCoordinates
          });
          if (map.getLayer('sentinel-raster-layer-t2')) {
            map.setLayoutProperty('sentinel-raster-layer-t2', 'visibility', visibilityVal);
            map.setPaintProperty('sentinel-raster-layer-t2', 'raster-opacity', t2Opacity);
          }
        } else {
          if (map.getLayer('sentinel-raster-layer-t2')) map.removeLayer('sentinel-raster-layer-t2');
          if (map.getSource('sentinel-raster-t2')) map.removeSource('sentinel-raster-t2');

          map.addSource('sentinel-raster-t2', {
            type: 'image',
            url: targetT2Url,
            coordinates: imageCoordinates
          });
          map.addLayer(
            {
              id: 'sentinel-raster-layer-t2',
              type: 'raster',
              source: 'sentinel-raster-t2',
              layout: {
                visibility: visibilityVal
              },
              paint: {
                'raster-opacity': t2Opacity,
                'raster-fade-duration': 100
              }
            },
            beforeLayerId
          );
        }
      } catch (err) {
        console.warn('Sentinel T2 overlay update notice:', err);
      }
    } else {
      if (map.getLayer('sentinel-raster-layer-t2')) map.removeLayer('sentinel-raster-layer-t2');
      if (map.getSource('sentinel-raster-t2')) map.removeSource('sentinel-raster-t2');
    }

    // 3. Add or Update Georeferenced Bounding Box Perimeter
    const bboxGeoJson: any = {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [minLng, maxLat],
          [maxLng, maxLat],
          [maxLng, minLat],
          [minLng, minLat],
          [minLng, maxLat]
        ]]
      },
      properties: { name: ov.name }
    };

    try {
      const existingBBox = map.getSource('sentinel-bbox-source') as maplibregl.GeoJSONSource | undefined;
      if (existingBBox && typeof existingBBox.setData === 'function') {
        existingBBox.setData(bboxGeoJson);
        if (map.getLayer('sentinel-bbox-glow')) {
          map.setLayoutProperty('sentinel-bbox-glow', 'visibility', visibilityVal);
          map.setPaintProperty('sentinel-bbox-glow', 'line-opacity', isVisible ? 0.6 * opacity : 0);
        }
        if (map.getLayer('sentinel-bbox-line')) {
          map.setLayoutProperty('sentinel-bbox-line', 'visibility', visibilityVal);
          map.setPaintProperty('sentinel-bbox-line', 'line-opacity', isVisible ? 0.95 * opacity : 0);
        }
      } else {
        if (map.getLayer('sentinel-bbox-line')) map.removeLayer('sentinel-bbox-line');
        if (map.getLayer('sentinel-bbox-glow')) map.removeLayer('sentinel-bbox-glow');
        if (map.getSource('sentinel-bbox-source')) map.removeSource('sentinel-bbox-source');

        map.addSource('sentinel-bbox-source', {
          type: 'geojson',
          data: bboxGeoJson
        });

        // Neon cyan boundary glow halo
        map.addLayer({
          id: 'sentinel-bbox-glow',
          type: 'line',
          source: 'sentinel-bbox-source',
          layout: {
            visibility: visibilityVal
          },
          paint: {
            'line-color': '#00f2ff',
            'line-width': 5,
            'line-blur': 3,
            'line-opacity': isVisible ? 0.6 * opacity : 0
          }
        });

        // Crisp dashed boundary line
        map.addLayer({
          id: 'sentinel-bbox-line',
          type: 'line',
          source: 'sentinel-bbox-source',
          layout: {
            visibility: visibilityVal
          },
          paint: {
            'line-color': '#00f2ff',
            'line-width': 2,
            'line-dasharray': [4, 2],
            'line-opacity': isVisible ? 0.95 * opacity : 0
          }
        });
      }

      // Draw all discovered spatial site footprints if multiple exist
      if (ov.sites && ov.sites.length > 1) {
        const sitesGeoJson = {
          type: 'FeatureCollection',
          features: ov.sites.map((st) => {
            const [sMinLng, sMinLat, sMaxLng, sMaxLat] = st.bounds;
            const isCurrentActive = activeSiteId ? st.site_id === activeSiteId : st.site_id === ov.sites![0].site_id;
            return {
              type: 'Feature',
              properties: {
                site_id: st.site_id,
                name: st.name,
                scene_count: st.scene_count,
                date_range: st.date_range,
                isActive: isCurrentActive,
              },
              geometry: {
                type: 'Polygon',
                coordinates: [
                  [
                    [sMinLng, sMinLat],
                    [sMaxLng, sMinLat],
                    [sMaxLng, sMaxLat],
                    [sMinLng, sMaxLat],
                    [sMinLng, sMinLat]
                  ]
                ]
              }
            };
          })
        };

        const existingSitesSource = map.getSource('spatial-sites-source') as maplibregl.GeoJSONSource | undefined;
        if (existingSitesSource && typeof existingSitesSource.setData === 'function') {
          existingSitesSource.setData(sitesGeoJson as any);
        } else {
          ['spatial-sites-active-line', 'spatial-sites-line', 'spatial-sites-fill'].forEach((id) => {
            if (map.getLayer(id)) map.removeLayer(id);
          });
          if (map.getSource('spatial-sites-source')) map.removeSource('spatial-sites-source');

          map.addSource('spatial-sites-source', {
            type: 'geojson',
            data: sitesGeoJson as any
          });

          // Interactive fill for click detection
          map.addLayer({
            id: 'spatial-sites-fill',
            type: 'fill',
            source: 'spatial-sites-source',
            paint: {
              'fill-color': '#00f2ff',
              'fill-opacity': ['case', ['get', 'isActive'], 0.04, 0.09]
            }
          });

          // Inactive dashed border
          map.addLayer({
            id: 'spatial-sites-line',
            type: 'line',
            source: 'spatial-sites-source',
            filter: ['!', ['get', 'isActive']],
            paint: {
              'line-color': '#00f2ff',
              'line-width': 1.8,
              'line-dasharray': [3, 2],
              'line-opacity': 0.75
            }
          });

          // Active glowing border
          map.addLayer({
            id: 'spatial-sites-active-line',
            type: 'line',
            source: 'spatial-sites-source',
            filter: ['get', 'isActive'],
            paint: {
              'line-color': '#00f2ff',
              'line-width': 3,
              'line-opacity': 0.95
            }
          });
        }
      } else {
        ['spatial-sites-active-line', 'spatial-sites-line', 'spatial-sites-fill'].forEach((id) => {
          if (map.getLayer(id)) map.removeLayer(id);
        });
        if (map.getSource('spatial-sites-source')) map.removeSource('spatial-sites-source');
      }
    } catch (err) {
      console.warn('Sentinel bbox add notice:', err);
    }
  }, [activeTimelineStep, activeSiteId]);

  // Ensure AI visual grounding detection source & layers exist
  const ensureDetectionLayers = useCallback((map: maplibregl.Map, data: any) => {
    const emptyCollection = { type: 'FeatureCollection', features: [] };
    const sourceData = data || emptyCollection;

    // Generate centroids for heatmap
    const pointsData: any = {
      type: 'FeatureCollection',
      features: (sourceData.features || []).map((f: any) => {
        let coords = f.geometry.coordinates;
        if (f.geometry.type === 'Polygon') {
          const ring = coords[0];
          let sumLng = 0, sumLat = 0;
          ring.forEach((c: any) => { sumLng += c[0]; sumLat += c[1]; });
          coords = [sumLng / ring.length, sumLat / ring.length];
        } else if (f.geometry.type === 'MultiPolygon') {
          const ring = coords[0][0];
          let sumLng = 0, sumLat = 0;
          ring.forEach((c: any) => { sumLng += c[0]; sumLat += c[1]; });
          coords = [sumLng / ring.length, sumLat / ring.length];
        }
        return {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: coords },
          properties: f.properties || {}
        };
      })
    };

    if (!map.getSource('satquery-detections')) {
      map.addSource('satquery-detections', {
        type: 'geojson',
        data: sourceData
      });
      
      map.addSource('satquery-centroids', {
        type: 'geojson',
        data: pointsData
      });

      // Heatmap layer
      map.addLayer({
        id: 'satquery-heatmap',
        type: 'heatmap',
        source: 'satquery-centroids',
        layout: { visibility: 'none' }, // Toggled elsewhere
        paint: {
          'heatmap-weight': 1,
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 15, 3],
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0, 'rgba(33,102,172,0)',
            0.2, 'rgb(103,169,207)',
            0.4, 'rgb(209,229,240)',
            0.6, 'rgb(253,219,199)',
            0.8, 'rgb(239,138,98)',
            1, 'rgb(178,24,43)'
          ],
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 2, 15, 20],
          'heatmap-opacity': 0.8
        }
      });

      // Semi-transparent fill supporting feature properties (dynamic or fallback to cyan)
      map.addLayer({
        id: 'satquery-fill',
        type: 'fill',
        source: 'satquery-detections',
        paint: {
          'fill-color': [
            'case',
            ['boolean', ['get', 'query_matched'], false],
            '#00f2ff',
            [
              'coalesce',
              ['get', 'fillColor'],
              ['get', 'color'],
              '#f59e0b'
            ]
          ],
          'fill-opacity': [
            'case',
            ['boolean', ['get', 'query_matched'], false],
            0.38,
            [
              'coalesce',
              ['get', 'fillOpacity'],
              0.22
            ]
          ]
        }
      });

      // Neon glow halo
      map.addLayer({
        id: 'satquery-glow',
        type: 'line',
        source: 'satquery-detections',
        paint: {
          'line-color': [
            'case',
            ['boolean', ['get', 'query_matched'], false],
            '#00f2ff',
            [
              'coalesce',
              ['get', 'strokeColor'],
              ['get', 'color'],
              '#f59e0b'
            ]
          ],
          'line-width': [
            'case',
            ['boolean', ['get', 'query_matched'], false],
            8,
            4
          ],
          'line-blur': 4,
          'line-opacity': [
            'case',
            ['boolean', ['get', 'query_matched'], false],
            0.85,
            0.45
          ]
        }
      });

      // Crisp boundary outline
      map.addLayer({
        id: 'satquery-outline',
        type: 'line',
        source: 'satquery-detections',
        paint: {
          'line-color': [
            'case',
            ['boolean', ['get', 'query_matched'], false],
            '#00f2ff',
            [
              'coalesce',
              ['get', 'strokeColor'],
              ['get', 'color'],
              '#d97706'
            ]
          ],
          'line-width': [
            'case',
            ['boolean', ['get', 'query_matched'], false],
            3.0,
            1.8
          ],
          'line-opacity': 0.95
        }
      });

      // Interactive popup on clicking detected polygon
      map.on('click', 'satquery-fill', (e) => {
        if (!e.features || e.features.length === 0) return;
        const feat = e.features[0];
        const props = (feat.properties as any) || {};
        const featId = props.id || (feat.id !== undefined ? String(feat.id) : '');
        if (featId && setFocusedCandidateIdRef.current) {
          setFocusedCandidateIdRef.current(featId);
        }

        const review = analystReviewsRef.current[featId];
        const verdict = review?.verdict || props.analystVerdict || 'pending';
        const label = props.label || `Candidate #${featId || '1'}`;
        const areaHa = props.area_ha ? `${parseFloat(props.area_ha).toFixed(2)} ha` : null;
        const areaM2 = props.area_m2 ? `${parseFloat(props.area_m2).toLocaleString()} m²` : null;
        const typeStr = props.change_type || props.type || 'Detection';
        const confidence = props.confidence ? `${Math.round(props.confidence * 100)}%` : null;
        const isQueryMatched = Boolean(props.query_matched);
        const relPercent = props.semantic_relevance ? `${Math.round(props.semantic_relevance * 100)}%` : null;
        const groundingReason = props.grounding_reason || '';
        const color = isQueryMatched ? '#00f2ff' : (props.strokeColor || props.color || '#f59e0b');

        const verdictBadgeColor =
          verdict === 'confirmed' ? '#10b981' :
          verdict === 'rejected' ? '#64748b' :
          verdict === 'flagged' ? '#f59e0b' : '#00f2ff';

        const targetBadgeHtml = isQueryMatched
          ? `<span style="background: rgba(0,242,255,0.15); color: #00f2ff; border: 1px solid rgba(0,242,255,0.5); padding: 1px 6px; border-radius: 4px; font-size: 9px; font-weight: 700; text-transform: uppercase;">🎯 TARGET ${relPercent ? `(${relPercent})` : ''}</span>`
          : '';

        const groundingReasonHtml = groundingReason
          ? `<div style="font-size: 9.5px; color: #38bdf8; margin-top: 5px; padding: 4px 6px; background: rgba(0,242,255,0.06); border-radius: 4px; border-left: 2px solid #00f2ff;"><strong>Grounding:</strong> <em>${groundingReason}</em></div>`
          : '';

        new maplibregl.Popup({ closeButton: true, closeOnClick: true, className: 'satquery-detection-popup' })
          .setLngLat(e.lngLat)
          .setHTML(`
            <div style="padding: 10px; font-family: monospace; font-size: 11px; color: #f1f5f9; background: #0b0f19; border-radius: 8px; border: 1px solid ${isQueryMatched ? '#00f2ff' : verdictBadgeColor}; box-shadow: 0 0 20px rgba(0,0,0,0.9); min-width: 200px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; gap: 8px;">
                <span style="font-weight: 700; color: ${color}; text-transform: uppercase; letter-spacing: 0.5px;">${label}</span>
                <div style="display: flex; gap: 4px; align-items: center;">
                  ${targetBadgeHtml}
                  <span style="background: ${verdictBadgeColor}22; color: ${verdictBadgeColor}; border: 1px solid ${verdictBadgeColor}88; padding: 1px 6px; border-radius: 4px; font-size: 9px; font-weight: 700; text-transform: uppercase;">${verdict}</span>
                </div>
              </div>
              <div style="font-size: 10px; color: #94a3b8; margin-bottom: 3px;">Classification: <strong style="color:#e2e8f0;">${typeStr}</strong></div>
              ${confidence ? `<div style="font-size: 10px; color: #94a3b8; margin-bottom: 3px;">Confidence: <strong style="color:#38bdf8;">${confidence}</strong></div>` : ''}
              ${areaHa ? `<div style="font-size: 10px; color: #cbd5e1; margin-bottom: 3px;">Area: <strong>${areaHa}</strong> (${areaM2})</div>` : ''}
              ${groundingReasonHtml}
              ${review?.notes ? `<div style="font-size: 10px; color: #f59e0b; margin-top: 5px; padding-top: 4px; border-top: 1px solid #334155;">Note: <em>${review.notes}</em></div>` : ''}
              <div style="font-size: 9px; color: #64748b; margin-top: 6px;">Position: ${e.lngLat.lat.toFixed(4)}°N, ${e.lngLat.lng.toFixed(4)}°E</div>
            </div>
          `)
          .addTo(map);
      });

      map.on('mouseenter', 'satquery-fill', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', 'satquery-fill', () => {
        map.getCanvas().style.cursor = '';
      });
    } else {
      const source = map.getSource('satquery-detections') as maplibregl.GeoJSONSource;
      source.setData(sourceData);
      
      const centroidsSource = map.getSource('satquery-centroids') as maplibregl.GeoJSONSource;
      if (centroidsSource) centroidsSource.setData(pointsData);
    }
  }, []);

  // 1. Initialize MapLibre GL instance
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const initialCenter: [number, number] = activeOverlay?.center
      ? activeOverlay.center
      : [78.9629, 20.5937];
    const initialZoom = activeOverlay ? 13 : 4.8;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: SATELLITE_STYLE,
      center: initialCenter,
      zoom: initialZoom,
      pitch: 0,
      bearing: 0,
      attributionControl: false,
      dragRotate: true,
      touchPitch: true,
      pitchWithRotate: true,
      maxPitch: 70
    });

    mapRef.current = map;
    // Standard Scale Bar at bottom left
    const scale = new maplibregl.ScaleControl({
      maxWidth: 100,
      unit: 'metric'
    });
    map.addControl(scale, 'bottom-left');

    map.on('move', () => {
      const center = map.getCenter();
      setCenterCoords({ lat: parseFloat(center.lat.toFixed(4)), lng: parseFloat(center.lng.toFixed(4)) });
      setZoomLevel(parseFloat(map.getZoom().toFixed(1)));
      setBearing(parseFloat(map.getBearing().toFixed(0)));
      setPitch(parseFloat(map.getPitch().toFixed(0)));
    });

    map.on('mousemove', (e) => {
      setCursorCoords({
        lat: parseFloat(e.lngLat.lat.toFixed(5)),
        lng: parseFloat(e.lngLat.lng.toFixed(5))
      });
    });

    map.on('mouseout', () => {
      setCursorCoords(null);
    });

    map.on('load', () => {
      // Created first so the detection outlines added next sit above it.
      if (!map.getSource(RESULT_RASTER_SOURCE)) {
        map.addSource(RESULT_RASTER_SOURCE, {
          type: 'image',
          url: TRANSPARENT_PIXEL,
          coordinates: PLACEHOLDER_COORDS
        });
        map.addLayer({
          id: RESULT_RASTER_LAYER,
          type: 'raster',
          source: RESULT_RASTER_SOURCE,
          layout: { visibility: 'none' },
          paint: { 'raster-opacity': 0.7, 'raster-fade-duration': 200 }
        });
      }

      ensureDetectionLayers(map, null);

      // Register spatial site footprint click and hover listeners
      map.on('click', 'spatial-sites-fill', (e) => {
        if (e.features && e.features.length > 0) {
          const clickedSiteId = e.features[0].properties?.site_id;
          if (clickedSiteId) {
            setActiveSiteId(clickedSiteId);
          }
        }
      });
      map.on('mouseenter', 'spatial-sites-fill', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', 'spatial-sites-fill', () => {
        map.getCanvas().style.cursor = '';
      });

      if (activeOverlay) {
        updateOverlayLayers(map, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible, bandPreset);
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // 1b. Keep the WebGL canvas in step with its container. The map is embedded
  // in a resizable, collapsible dock, so relying on window resize alone leaves
  // it rendering at a stale size after a drag or a tab switch.
  useEffect(() => {
    const container = mapContainerRef.current;
    if (!container || typeof ResizeObserver === 'undefined') return;

    const observer = new ResizeObserver(() => {
      mapRef.current?.resize();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // 2. React to activeOverlay updates (uploaded Sentinel images or selected presets)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !activeOverlay || !activeOverlay.bounds) return;

    const applyOverlay = () => {
      updateOverlayLayers(map, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible, bandPreset);

      // Smooth camera glide to georeferenced bounding box
      const [minLng, minLat, maxLng, maxLat] = activeOverlay.bounds;
      map.fitBounds(
        [
          [minLng, minLat],
          [maxLng, maxLat]
        ],
        {
          padding: 85,
          maxZoom: 16.5,
          duration: 2200
        }
      );

      setDetectionLabel(
        `${activeOverlay.name} [${activeOverlay.mode === 'multi-temporal' ? 'Multi-Temporal Stack' : activeOverlay.mode === 'bi-temporal' ? 'Bi-Temporal Pair' : 'Single Scene'}]`
      );
    };

    if (map.isStyleLoaded()) {
      applyOverlay();
    } else {
      map.once('load', applyOverlay);
    }
  }, [activeOverlay, updateOverlayLayers, isOverlayVisible, bandPreset]);

  // 3. React to opacity, visibility, or temporal slider adjustments
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !activeOverlay || !map.isStyleLoaded()) return;

    const isVisible = isOverlayVisible && overlayOpacity > 0.01;
    const visibilityVal = isVisible ? 'visible' : 'none';

    let t1Op = isVisible ? overlayOpacity : 0;
    let t2Op = 0;

    if (activeOverlay.mode === 'bi-temporal' && activeOverlay.t2ImageUrl) {
      if (temporalMode === 't1') {
        t1Op = isVisible ? overlayOpacity : 0;
        t2Op = 0;
      } else if (temporalMode === 't2') {
        t1Op = 0;
        t2Op = isVisible ? overlayOpacity : 0;
      } else {
        const ratio = temporalSlider / 100;
        t1Op = isVisible ? overlayOpacity * (1 - ratio) : 0;
        t2Op = isVisible ? overlayOpacity * ratio : 0;
      }
    }

    if (map.getLayer('sentinel-raster-layer-t1')) {
      map.setLayoutProperty('sentinel-raster-layer-t1', 'visibility', visibilityVal);
      map.setPaintProperty('sentinel-raster-layer-t1', 'raster-opacity', t1Op);
    }
    if (map.getLayer('sentinel-raster-layer-t2')) {
      map.setLayoutProperty('sentinel-raster-layer-t2', 'visibility', visibilityVal);
      map.setPaintProperty('sentinel-raster-layer-t2', 'raster-opacity', t2Op);
    }
    if (map.getLayer('sentinel-bbox-glow')) {
      map.setLayoutProperty('sentinel-bbox-glow', 'visibility', visibilityVal);
      map.setPaintProperty('sentinel-bbox-glow', 'line-opacity', isVisible ? 0.6 * overlayOpacity : 0);
    }
    if (map.getLayer('sentinel-bbox-line')) {
      map.setLayoutProperty('sentinel-bbox-line', 'visibility', visibilityVal);
      map.setPaintProperty('sentinel-bbox-line', 'line-opacity', isVisible ? 0.95 * overlayOpacity : 0);
    }
  }, [overlayOpacity, isOverlayVisible, temporalSlider, temporalMode, activeOverlay]);

  // 3b. React to band preset changes (swapping genuine NIR raster overlay vs True Colour)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !activeOverlay || !map.isStyleLoaded()) return;
    updateOverlayLayers(map, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible, bandPreset);
  }, [bandPreset, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible, updateOverlayLayers]);

  // 3c. React to multi-temporal step scrubber adjustments
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !activeOverlay || !map.isStyleLoaded()) return;
    if (activeOverlay.mode === 'multi-temporal' || focusedTimeline) {
      updateOverlayLayers(map, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible, bandPreset);
    }
  }, [activeTimelineStep, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible, bandPreset, focusedTimeline, updateOverlayLayers]);

  // 4. React to dynamic GeoJSON detections from AI inference queries
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    try {
      let targetGeoJson: any = null;
      let parsedInput = geoJsonData;
      if (typeof parsedInput === 'string') {
        try {
          const trimmed = parsedInput.trim();
          if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
            parsedInput = JSON.parse(trimmed);
          }
        } catch (e) {
          // Not a JSON string
        }
      }

      if (parsedInput) {
        if (parsedInput.type === 'FeatureCollection') {
          targetGeoJson = JSON.parse(JSON.stringify(parsedInput));
        } else if (parsedInput.type === 'Feature') {
          targetGeoJson = { type: 'FeatureCollection', features: [JSON.parse(JSON.stringify(parsedInput))] };
        } else if (Array.isArray(parsedInput) && parsedInput.length > 0) {
          targetGeoJson = { type: 'FeatureCollection', features: JSON.parse(JSON.stringify(parsedInput)) };
        }
      }

      if (!targetGeoJson || !targetGeoJson.features || targetGeoJson.features.length === 0) {
        const source = map.getSource('satquery-detections') as maplibregl.GeoJSONSource;
        if (source) {
          source.setData({ type: 'FeatureCollection', features: [] });
        }
        setDetectionCount(0);
        activeDetectionFeaturesRef.current = [];
        return;
      }

      // Check if GeoJSON coordinates are normalized [0, 1] relative coordinates
      const rawBounds = getBounds(targetGeoJson);
      if (rawBounds) {
        const [[rMinLng, rMinLat], [rMaxLng, rMaxLat]] = rawBounds;
        const isRelative = rMinLng >= -0.05 && rMaxLng <= 1.05 && rMinLat >= -0.05 && rMaxLat <= 1.05;

        if (isRelative) {
          const sceneBounds = activeOverlay?.bounds || [77.618, 13.022, 77.652, 13.048];
          const [minLng, minLat, maxLng, maxLat] = sceneBounds;

          targetGeoJson.features.forEach((feat: any) => {
            if (feat.geometry?.type === 'Polygon' && Array.isArray(feat.geometry.coordinates)) {
              feat.geometry.coordinates = feat.geometry.coordinates.map((ring: any[]) =>
                ring.map((pt: any) => [
                  minLng + pt[0] * (maxLng - minLng),
                  maxLat - pt[1] * (maxLat - minLat)
                ])
              );
            }
          });
        }
      }

      // Decorate features with analyst triage verdict styling, candidate IDs, and selection focus
      targetGeoJson.features.forEach((feat: any, idx: number) => {
        if (!feat.properties) feat.properties = {};
        const featId = feat.properties.id || (feat.id !== undefined ? String(feat.id) : `cand-${idx + 1}`);
        feat.properties.id = featId;

        const review = analystReviews[featId];
        const verdict = review?.verdict || feat.properties.analyst_verdict || 'pending';
        const isFocused = focusedCandidateId === featId;

        let strokeColor = '#00f2ff';
        let fillColor = '#00f2ff';
        let fillOpacity = 0.28;
        let strokeWidth = 2.5;

        switch (verdict) {
          case 'confirmed':
            strokeColor = '#10b981'; // Emerald Green
            fillColor = '#10b981';
            fillOpacity = isFocused ? 0.55 : 0.35;
            strokeWidth = isFocused ? 4.5 : 3.0;
            break;
          case 'rejected':
            strokeColor = '#64748b'; // Muted Slate Gray
            fillColor = '#64748b';
            fillOpacity = isFocused ? 0.22 : 0.10;
            strokeWidth = isFocused ? 3.0 : 1.5;
            break;
          case 'flagged':
            strokeColor = '#f59e0b'; // Amber Warning
            fillColor = '#f59e0b';
            fillOpacity = isFocused ? 0.55 : 0.38;
            strokeWidth = isFocused ? 4.5 : 3.0;
            break;
          case 'pending':
          default:
            strokeColor = isFocused ? '#ffffff' : '#00f2ff';
            fillColor = '#00f2ff';
            fillOpacity = isFocused ? 0.50 : 0.28;
            strokeWidth = isFocused ? 4.5 : 2.5;
            break;
        }

        feat.properties.strokeColor = strokeColor;
        feat.properties.fillColor = fillColor;
        feat.properties.fillOpacity = fillOpacity;
        feat.properties.strokeWidth = strokeWidth;
        feat.properties.analystVerdict = verdict;
        feat.properties.analystNotes = review?.notes || '';
        feat.properties.isFocused = isFocused;
      });

      activeDetectionFeaturesRef.current = targetGeoJson.features;

      const source = map.getSource('satquery-detections') as maplibregl.GeoJSONSource;
      if (source) {
        source.setData(targetGeoJson);
        const centroidsSource = map.getSource('satquery-centroids') as maplibregl.GeoJSONSource;
        if (centroidsSource) {
          const pointsData: any = {
            type: 'FeatureCollection',
            features: (targetGeoJson.features || []).map((f: any) => {
              let coords = f.geometry.coordinates;
              if (f.geometry.type === 'Polygon') {
                const ring = coords[0];
                let sumLng = 0, sumLat = 0;
                ring.forEach((c: any) => { sumLng += c[0]; sumLat += c[1]; });
                coords = [sumLng / ring.length, sumLat / ring.length];
              } else if (f.geometry.type === 'MultiPolygon') {
                const ring = coords[0][0];
                let sumLng = 0, sumLat = 0;
                ring.forEach((c: any) => { sumLng += c[0]; sumLat += c[1]; });
                coords = [sumLng / ring.length, sumLat / ring.length];
              }
              return {
                type: 'Feature',
                geometry: { type: 'Point', coordinates: coords },
                properties: f.properties || {}
              };
            })
          };
          centroidsSource.setData(pointsData);
        }
      } else {
        ensureDetectionLayers(map, targetGeoJson);
      }

      const count = targetGeoJson.features ? targetGeoJson.features.length : 0;
      setDetectionCount(count);

      const isNewGeoJson = prevGeoJsonDataRef.current !== geoJsonData;
      prevGeoJsonDataRef.current = geoJsonData;

      const bounds = getBounds(targetGeoJson);
      if (bounds) {
        const [[minLng, minLat], [maxLng, maxLat]] = bounds;

        // Validate that bounds are within MapLibre's valid coordinate range.
        // Raw projected coordinates (e.g. UTM metres) will fail this check.
        const isValidLngLat =
          minLng >= -180 && maxLng <= 180 &&
          minLat >= -90 && maxLat <= 90 &&
          isFinite(minLng) && isFinite(maxLng) &&
          isFinite(minLat) && isFinite(maxLat);

        if (isValidLngLat) {
          const midLng = (minLng + maxLng) / 2;
          const midLat = (minLat + maxLat) / 2;
          setCenterCoords({ lat: parseFloat(midLat.toFixed(4)), lng: parseFloat(midLng.toFixed(4)) });

          if (isNewGeoJson) {
            try {
              map.fitBounds(bounds, {
                padding: 85,
                maxZoom: 16.5,
                duration: 1800
              });
            } catch (fitErr) {
              console.warn('[Diachron] fitBounds failed, using fallback:', fitErr);
            }
          }

          const featureLabel = targetGeoJson.features?.[0]?.properties?.label;
          setDetectionLabel(
            featureLabel || (
              datasetName
                ? `${datasetName} [Active AI Detection]`
                : `Detected Feature [${midLat.toFixed(4)}°N, ${midLng.toFixed(4)}°E]`
            )
          );
        } else {
          // Coordinates are out of range (projected CRS); use overlay bounds as fallback
          console.warn('[Diachron] GeoJSON bounds out of WGS84 range, using overlay fallback:', { minLng, minLat, maxLng, maxLat });
          if (activeOverlay?.bounds) {
            const [oMinLng, oMinLat, oMaxLng, oMaxLat] = activeOverlay.bounds;
            const midLng = (oMinLng + oMaxLng) / 2;
            const midLat = (oMinLat + oMaxLat) / 2;
            setCenterCoords({ lat: parseFloat(midLat.toFixed(4)), lng: parseFloat(midLng.toFixed(4)) });
            if (isNewGeoJson) {
              try {
                map.fitBounds([[oMinLng, oMinLat], [oMaxLng, oMaxLat]], {
                  padding: 85,
                  maxZoom: 16.5,
                  duration: 1800
                });
              } catch (fitErr) {
                console.warn('[Diachron] Overlay fallback fitBounds also failed:', fitErr);
              }
            }
          }
        }
      }
    } catch (err) {
      console.error('[Diachron] GeoJSON detection layer error caught by safety guard:', err);
    }
  }, [geoJsonData, datasetName, getBounds, ensureDetectionLayers, activeOverlay, analystReviews, focusedCandidateId]);

  // 4b. Smooth camera glide to focused candidate when selected in Review Queue
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !focusedCandidateId) return;

    const features = activeDetectionFeaturesRef.current;
    if (!features || features.length === 0) return;

    const matchedFeat = features.find(
      (f: any) => f.properties?.id === focusedCandidateId || String(f.id) === focusedCandidateId
    );
    if (!matchedFeat) return;

    const isValidLngLat = (lng: number, lat: number) =>
      isFinite(lng) && isFinite(lat) && lng >= -180 && lng <= 180 && lat >= -90 && lat <= 90;

    const props = matchedFeat.properties || {};
    const geom = matchedFeat.geometry || {};
    let featBounds: [[number, number], [number, number]] | null = null;
    let center: [number, number] | null = null;

    // 1. Try extracting explicit WGS84 centroid & bounding box from properties if present
    if (Array.isArray(props.centroid_wgs84) && props.centroid_wgs84.length >= 2) {
      const [cLng, cLat] = props.centroid_wgs84;
      if (isValidLngLat(cLng, cLat)) {
        center = [cLng, cLat];
      }
    } else if (Array.isArray(props.center) && props.center.length >= 2) {
      const [cLng, cLat] = props.center;
      if (isValidLngLat(cLng, cLat)) {
        center = [cLng, cLat];
      }
    }

    if (Array.isArray(props.bounding_box_wgs84) && props.bounding_box_wgs84.length === 4) {
      const [minX, minY, maxX, maxY] = props.bounding_box_wgs84;
      if (isValidLngLat(minX, minY) && isValidLngLat(maxX, maxY)) {
        featBounds = [[minX, minY], [maxX, maxY]];
      }
    }

    // 2. Extract from geometry coordinates if valid WGS84
    if (geom.type === 'Polygon' && Array.isArray(geom.coordinates) && geom.coordinates[0]?.length) {
      const ring = geom.coordinates[0];
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      let sumX = 0, sumY = 0;
      let allValid = true;

      ring.forEach(([x, y]: [number, number]) => {
        if (!isValidLngLat(x, y)) {
          allValid = false;
          return;
        }
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        sumX += x;
        sumY += y;
      });

      if (allValid && ring.length > 0) {
        featBounds = [[minX, minY], [maxX, maxY]];
        center = [sumX / ring.length, sumY / ring.length];
      }
    } else if (geom.type === 'Point' && Array.isArray(geom.coordinates)) {
      const [pLng, pLat] = geom.coordinates;
      if (isValidLngLat(pLng, pLat)) {
        center = [pLng, pLat];
      }
    }

    // 3. Perform safe map navigation without throwing errors
    try {
      if (
        featBounds &&
        isValidLngLat(featBounds[0][0], featBounds[0][1]) &&
        isValidLngLat(featBounds[1][0], featBounds[1][1]) &&
        (featBounds[1][0] - featBounds[0][0] > 0.0002 || featBounds[1][1] - featBounds[0][1] > 0.0002)
      ) {
        map.fitBounds(featBounds, {
          padding: 140,
          maxZoom: 17,
          duration: 1200,
        });
      } else if (center && isValidLngLat(center[0], center[1])) {
        map.flyTo({ center, zoom: 16.2, duration: 1200 });
      }
    } catch (navErr) {
      console.warn('[Diachron] Safe navigation caught bounds error:', navErr);
      if (center && isValidLngLat(center[0], center[1])) {
        try {
          map.flyTo({ center, zoom: 15, duration: 1000 });
        } catch (_) {}
      }
    }
  }, [focusedCandidateId]);

  // 4b. Analysis raster overlay (change mask and similar).
  //
  // Update-only: the source and layer are created at map load (see above), so
  // this never has to touch the style, which is what made every lazy attempt
  // fail. The overlay also survives basemap switches for the same reason.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const apply = (tries: number) => {
      if (cancelled) return;

      const source = map.getSource(RESULT_RASTER_SOURCE) as maplibregl.ImageSource | undefined;
      if (!source || !map.getLayer(RESULT_RASTER_LAYER)) {
        // Map is still initialising; the load handler will create them.
        if (tries < 40) timer = setTimeout(() => apply(tries + 1), 150);
        return;
      }

      if (!resultOverlay) {
        map.setLayoutProperty(RESULT_RASTER_LAYER, 'visibility', 'none');
        return;
      }

      const [minLng, minLat, maxLng, maxLat] = resultOverlay.bounds;
      source.updateImage({
        url: resultOverlay.url,
        coordinates: [
          [minLng, maxLat],
          [maxLng, maxLat],
          [maxLng, minLat],
          [minLng, minLat]
        ]
      });

      map.setPaintProperty(RESULT_RASTER_LAYER, 'raster-opacity', resultOverlay.opacity);
      map.setLayoutProperty(RESULT_RASTER_LAYER, 'visibility', 'visible');
    };

    apply(0);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [resultOverlay]);

  // 5. Basemap switching (Satellite, Hybrid, Streets, SAR)
  const handleBaseLayerChange = (layer: BaseLayerType) => {
    setActiveBaseLayer(layer);
    const map = mapRef.current;
    if (!map) return;

    let targetStyle = SATELLITE_STYLE;
    if (layer === 'hybrid') targetStyle = HYBRID_STYLE;
    else if (layer === 'streets') targetStyle = STREETS_STYLE;
    else if (layer === 'sar') targetStyle = DARK_TACTICAL_STYLE;
    else if (layer === 'tactical-offline') targetStyle = TACTICAL_OFFLINE_STYLE;

    map.setStyle(targetStyle);

    // Re-attach all dynamic Sentinel raster and vector layers after basemap reload
    map.once('style.load', () => {
      try {
        if (!map.getSource(RESULT_RASTER_SOURCE)) {
          map.addSource(RESULT_RASTER_SOURCE, {
            type: 'image',
            url: TRANSPARENT_PIXEL,
            coordinates: PLACEHOLDER_COORDS
          });
          map.addLayer({
            id: RESULT_RASTER_LAYER,
            type: 'raster',
            source: RESULT_RASTER_SOURCE,
            layout: { visibility: 'none' },
            paint: { 'raster-opacity': 0.7, 'raster-fade-duration': 200 }
          });
        }
        ensureDetectionLayers(map, geoJsonData);
        if (activeOverlay) {
          updateOverlayLayers(map, activeOverlay, overlayOpacity, temporalMode, temporalSlider, isOverlayVisible, bandPreset);
        }
      } catch (err) {
        console.warn('Post-style.load reattachment notice:', err);
      }
    });
  };

  // 6. Layer Visibility Toggle
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    ['satquery-fill', 'satquery-glow', 'satquery-outline'].forEach((layerId) => {
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, 'visibility', showBoundingBoxes ? 'visible' : 'none');
      }
    });
  }, [showBoundingBoxes]);

  // 7. Navigation Actions
  const handleZoomIn = () => mapRef.current?.zoomIn({ duration: 300 });
  const handleZoomOut = () => mapRef.current?.zoomOut({ duration: 300 });

  const handleResetNorthPitch = () => {
    mapRef.current?.easeTo({
      bearing: 0,
      pitch: 0,
      duration: 600
    });
  };

  const handleFlyToScene = () => {
    if (!mapRef.current || !activeOverlay?.bounds) return;
    const [minLng, minLat, maxLng, maxLat] = activeOverlay.bounds;
    mapRef.current.fitBounds(
      [
        [minLng, minLat],
        [maxLng, maxLat]
      ],
      {
        padding: 85,
        maxZoom: 16.5,
        duration: 2000
      }
    );
  };

  const handleLocateMe = () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser.');
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        mapRef.current?.flyTo({
          center: [longitude, latitude],
          zoom: 13,
          duration: 2000,
          essential: true
        });
        setDetectionLabel(`User Location [${latitude.toFixed(4)}°N, ${longitude.toFixed(4)}°E]`);
      },
      (err) => console.warn('Geolocation notice:', err.message)
    );
  };

  const handleFlyToBookmark = (coords: [number, number], zoom: number, name: string) => {
    setShowBookmarks(false);
    mapRef.current?.flyTo({
      center: coords,
      zoom: zoom,
      pitch: zoom > 10 ? 25 : 0,
      duration: 2200,
      essential: true
    });
    setDetectionLabel(`Location: ${name}`);
  };

  // 8. Geocoding Search: Debounced query to OpenStreetMap Nominatim with local fallback
  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (!trimmed || trimmed.length < 2) {
      setSearchResults([]);
      setIsSearching(false);
      setShowSearchResults(false);
      return;
    }

    if (searchAbortControllerRef.current) {
      searchAbortControllerRef.current.abort();
    }
    const controller = new AbortController();
    searchAbortControllerRef.current = controller;

    setIsSearching(true);
    const timer = setTimeout(async () => {
      try {
        const results = await searchLocations(trimmed, controller.signal);
        setSearchResults(results);
        setShowSearchResults(results.length > 0);
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.warn('Geocoding search notice:', err);
        }
      } finally {
        setIsSearching(false);
      }
    }, 280);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [searchQuery]);

  // Close search suggestions on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(e.target as Node)) {
        setShowSearchResults(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle selection of any geocoded location (City, State, Country, or POI)
  const selectSearchResult = (item: GeocodingResult) => {
    setShowSearchResults(false);
    setSearchQuery('');
    const map = mapRef.current;
    if (!map) return;

    // Type-sensitive navigation:
    // Country -> Broad framing (zoom ~4.5)
    // State   -> Regional framing (zoom ~7.5)
    // City    -> High-detail urban street grid (zoom ~12.5)
    // Landmark-> Extreme detail (zoom ~15.0)
    if (item.bounds) {
      const maxZoom =
        item.category === 'country'
          ? 5.5
          : item.category === 'state'
          ? 8.5
          : item.category === 'city'
          ? 14.0
          : 16.5;

      map.fitBounds(item.bounds, {
        padding: 65,
        maxZoom,
        duration: 2200
      });
    } else {
      map.flyTo({
        center: item.center,
        zoom: item.zoom,
        pitch: item.category === 'city' || item.category === 'poi' ? 25 : 0,
        duration: 2200,
        essential: true
      });
    }

    setDetectionLabel(`Location: ${item.name} [${item.badgeLabel}]`);
  };

  const handleSearchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = searchQuery.trim();
    if (!query) return;

    // 1. Raw Coordinates check (e.g. "28.6139, 77.2090")
    const coordMatch = parseCoordinates(query);
    if (coordMatch) {
      selectSearchResult(coordMatch);
      return;
    }

    // 2. If suggestions already loaded in state, pick top item
    if (searchResults.length > 0) {
      selectSearchResult(searchResults[0]);
      return;
    }

    // 3. Otherwise execute immediate search
    setIsSearching(true);
    try {
      const results = await searchLocations(query);
      if (results.length > 0) {
        selectSearchResult(results[0]);
      } else {
        const found = PRESET_BOOKMARKS.find((b) => b.name.toLowerCase().includes(query.toLowerCase()));
        if (found) {
          handleFlyToBookmark(found.coords, found.zoom, found.name);
          setSearchQuery('');
        } else {
          setDetectionLabel(`No location found for "${query}"`);
        }
      }
    } finally {
      setIsSearching(false);
    }
  };

  const toggleFullscreen = () => {
    if (!rootRef.current) return;
    if (!document.fullscreenElement) {
      rootRef.current.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  };

  // Track fullscreen from the browser rather than from the click, so pressing
  // Escape keeps the button label honest.
  useEffect(() => {
    const sync = () => setIsFullscreen(document.fullscreenElement === rootRef.current);
    document.addEventListener('fullscreenchange', sync);
    return () => document.removeEventListener('fullscreenchange', sync);
  }, []);

  // Compute CSS filter style for Band Visualization Presets
  const getBandFilterStyle = (): React.CSSProperties => {
    switch (bandPreset) {
      case 'false-color-nir':
        // Genuine NIR raster overlay is swapped directly on the MapLibre layer. No fake CSS hue rotation.
        return {};
      case 'sar-contrast':
        return { filter: 'contrast(190%) brightness(115%) grayscale(70%)' };
      case 'edge-boost':
        return { filter: 'contrast(240%) saturate(160%)' };
      default:
        return {};
    }
  };

  const isCompact = density === 'compact';
  const overlayOn = Boolean(activeOverlay || (timelineData && timelineData.length > 0)) && isOverlayVisible && overlayOpacity > 0.01;
  const readout = cursorCoords ?? centerCoords;

  const closeMenus = () => {
    setShowLayers(false);
    setShowBookmarks(false);
  };

  return (
    <div
      ref={rootRef}
      className="relative flex h-full min-h-0 w-full select-none flex-col overflow-hidden bg-surface-2"
    >
      {/* ------------------------------------------------------------------ */}
      {/* Toolbar — search, layers, bookmarks. Everything else lives on the   */}
      {/* canvas so the map keeps as much height as possible.                 */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex h-10 shrink-0 items-center gap-1.5 border-b border-line bg-surface px-2">
        <div ref={searchContainerRef} className="relative min-w-0 flex-1">
          <form onSubmit={handleSearchSubmit} className="relative flex items-center">
            {isSearching ? (
              <Loader2
                size={12}
                className="pointer-events-none absolute left-2.5 animate-spin text-accent"
                aria-hidden
              />
            ) : (
              <Search
                size={12}
                className="pointer-events-none absolute left-2.5 text-ink-faint"
                aria-hidden
              />
            )}
            <input
              type="text"
              value={searchQuery}
              onFocus={() => {
                if (searchResults.length > 0) setShowSearchResults(true);
              }}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={isCompact ? 'Search place or lat, lon' : 'Search a city, region, country, or "12.97, 77.59"'}
              aria-label="Search for a location"
              className="h-7 w-full rounded-md border border-line bg-surface-2 pl-7 pr-7 font-mono text-[11px] text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/50"
            />
            {searchQuery && (
              <button
                type="button"
                aria-label="Clear search"
                onClick={() => {
                  setSearchQuery('');
                  setSearchResults([]);
                  setShowSearchResults(false);
                }}
                className="absolute right-1.5 cursor-pointer rounded p-0.5 text-ink-faint transition-colors hover:text-ink"
              >
                <X size={11} />
              </button>
            )}
          </form>

          {showSearchResults && searchResults.length > 0 && (
            <ul className="scrollbar-slim absolute left-0 top-[calc(100%+6px)] z-50 max-h-72 w-full min-w-[16rem] overflow-y-auto rounded-xl border border-line-strong bg-surface-2 py-1 shadow-2xl">
              {searchResults.map((item) => {
                const tone =
                  item.category === 'country'
                    ? 'text-ok'
                    : item.category === 'state'
                      ? 'text-accent'
                      : item.category === 'city'
                        ? 'text-violet'
                        : 'text-ink-muted';
                const Icon =
                  item.category === 'country' ? Flag : item.category === 'city' ? Building2 : MapPin;
                return (
                  <li key={item.id}>
                    <button
                      type="button"
                      onClick={() => selectSearchResult(item)}
                      className="flex w-full cursor-pointer items-start gap-2.5 px-2.5 py-1.5 text-left transition-colors hover:bg-surface-3"
                    >
                      <Icon size={12} className={`mt-0.5 shrink-0 ${tone}`} aria-hidden />
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center justify-between gap-2">
                          <span className="truncate text-xs text-ink">{item.name}</span>
                          <span className={`shrink-0 font-mono text-[9px] uppercase ${tone}`}>
                            {item.badgeLabel}
                          </span>
                        </span>
                        <span className="mt-0.5 block truncate text-[10.5px] text-ink-faint">
                          {item.displayName}
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* Interactive Map STAC/COG AOI Streamer Tool Button */}
        {ENABLE_AOI_FETCHER && (
          <button
            type="button"
            onClick={() => setIsAOIActive(!isAOIActive)}
            className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-mono transition-all cursor-pointer border ${
              isAOIActive
                ? 'bg-accent text-space-black font-semibold border-accent shadow-[0_0_10px_rgba(0,242,255,0.3)]'
                : 'bg-surface-2 text-ink-muted border-line hover:border-accent/40 hover:text-accent'
            }`}
            title="Select an Area of Interest on the map to stream real satellite imagery"
          >
            <Crop size={12} />
            <span className="hidden sm:inline">Select AOI</span>
          </button>
        )}

        {/* Archive Semantic Retrieval & Clustering Button */}
        <button
          type="button"
          onClick={() => setShowArchiveModal(true)}
          className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-mono transition-all cursor-pointer border bg-surface-2 text-ink-muted border-line hover:border-accent/40 hover:text-accent shadow-sm"
          title="Search 75+ archive scenes by semantic meaning, explore clusters, or inspect anomalies"
        >
          <Sparkles size={12} className="text-accent" />
          <span className="hidden sm:inline">Archive Search</span>
        </button>

        {/* Layers & rendering */}
        <div className="relative">
          <IconButton
            label="Layers and rendering"
            size="sm"
            active={showLayers}
            onClick={() => {
              setShowLayers(!showLayers);
              setShowBookmarks(false);
            }}
          >
            <Layers size={13} />
          </IconButton>

          {showLayers && (
            <div className="absolute right-0 top-[calc(100%+6px)] z-50 w-60 rounded-xl border border-line-strong bg-surface-2 p-3 shadow-2xl">
              <p className="label-caps mb-2 text-ink-faint">Basemap</p>
              <div className="grid grid-cols-2 gap-1">
                {BASE_LAYERS.map((layer) => (
                  <button
                    key={layer.value}
                    type="button"
                    onClick={() => handleBaseLayerChange(layer.value)}
                    className={`cursor-pointer rounded-md px-2 py-1.5 text-left text-[11.5px] transition-colors ${
                      activeBaseLayer === layer.value
                        ? 'bg-accent/15 text-accent'
                        : 'text-ink-muted hover:bg-surface-3 hover:text-ink'
                    }`}
                  >
                    {layer.label}
                  </button>
                ))}
              </div>

              <p className="label-caps mb-2 mt-3.5 text-ink-faint">Band rendering</p>
              <div className="space-y-0.5">
                {BAND_PRESETS.map((preset) => {
                  const isNir = preset.value === 'false-color-nir';
                  const isDisabled = isNir && !nirCapability.available;
                  const isSelected = bandPreset === preset.value;

                  return (
                    <button
                      key={preset.value}
                      type="button"
                      disabled={isDisabled}
                      title={isNir && isDisabled ? nirCapability.reason : undefined}
                      onClick={() => !isDisabled && setBandPreset(preset.value)}
                      className={`flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-[11.5px] transition-colors ${
                        isDisabled
                          ? 'cursor-not-allowed opacity-45 text-ink-faint'
                          : 'cursor-pointer'
                      } ${
                        isSelected
                          ? 'bg-accent/15 text-accent font-medium'
                          : !isDisabled
                          ? 'text-ink-muted hover:bg-surface-3 hover:text-ink'
                          : ''
                      }`}
                    >
                      <div className="flex flex-col min-w-0 pr-1">
                        <span>{preset.label}</span>
                        {isNir && (
                          <span
                            className={`text-[9.5px] font-mono leading-tight ${
                              isDisabled ? 'text-amber-500/90' : 'text-emerald-500/90'
                            }`}
                          >
                            {nirCapability.reason}
                          </span>
                        )}
                      </div>
                      {isSelected && <Check size={11} aria-hidden className="shrink-0" />}
                    </button>
                  );
                })}
              </div>
              
              <p className="label-caps mb-2 mt-3.5 text-ink-faint">Bhuvan Thematic Overlays</p>
              <div className="space-y-0.5">
                {[
                  { value: 'none', label: 'None' },
                  { value: 'lulc:IN_LULC250K_1516', label: 'LULC (Land Use)' },
                  { value: 'waterbody:IN_Waterbody', label: 'Water Bodies' }
                ].map((layer) => (
                  <button
                    key={layer.value}
                    type="button"
                    onClick={() => setBhuvanThematicLayer(layer.value === 'none' ? null : layer.value)}
                    className={`flex w-full cursor-pointer items-center justify-between rounded-md px-2 py-1.5 text-left text-[11.5px] transition-colors ${
                      (bhuvanThematicLayer === layer.value || (layer.value === 'none' && !bhuvanThematicLayer))
                        ? 'bg-accent/15 text-accent'
                        : 'text-ink-muted hover:bg-surface-3 hover:text-ink'
                    }`}
                  >
                    {layer.label}
                    {(bhuvanThematicLayer === layer.value || (layer.value === 'none' && !bhuvanThematicLayer)) && <Check size={11} aria-hidden />}
                  </button>
                ))}
              </div>

              <div className="mt-3.5 space-y-1 border-t border-line pt-2.5">
                <label className="flex cursor-pointer items-center justify-between text-[11.5px] text-ink-muted">
                  <span>Detection outlines</span>
                  <input
                    type="checkbox"
                    checked={showBoundingBoxes}
                    onChange={(e) => setShowBoundingBoxes(e.target.checked)}
                    className="h-3.5 w-3.5 accent-[var(--color-accent)]"
                  />
                </label>
                <label className="flex cursor-pointer items-center justify-between text-[11.5px] text-ink-muted">
                  <span>Detection density heatmap</span>
                  <input
                    type="checkbox"
                    checked={showHeatmap}
                    onChange={(e) => setShowHeatmap(e.target.checked)}
                    className="h-3.5 w-3.5 accent-[var(--color-accent)]"
                  />
                </label>
                <label className="flex cursor-pointer items-center justify-between text-[11.5px] text-ink-muted">
                  <span>Centre reticle</span>
                  <input
                    type="checkbox"
                    checked={showReticle}
                    onChange={(e) => setShowReticle(e.target.checked)}
                    className="h-3.5 w-3.5 accent-[var(--color-accent)]"
                  />
                </label>
                {activeOverlay && (
                  <label className="flex cursor-pointer items-center justify-between text-[11.5px] text-ink-muted">
                    <span>Temporal controls</span>
                    <input
                      type="checkbox"
                      checked={showOverlayControls}
                      onChange={(e) => setShowOverlayControls(e.target.checked)}
                      className="h-3.5 w-3.5 accent-[var(--color-accent)]"
                    />
                  </label>
                )}
              </div>

              {activeOverlay && (
                <button
                  type="button"
                  onClick={() => {
                    setShowColorLegend(true);
                    setShowLayers(false);
                  }}
                  className="mt-3 flex w-full cursor-pointer items-center gap-1.5 rounded-md border border-line px-2 py-1.5 text-[11.5px] text-ink-muted transition-colors hover:border-amber/40 hover:text-amber"
                >
                  <Palette size={12} aria-hidden />
                  What do the colours mean?
                </button>
              )}
            </div>
          )}
        </div>

        {/* Location bookmarks */}
        <div className="relative">
          <IconButton
            label="Jump to a location"
            size="sm"
            active={showBookmarks}
            onClick={() => {
              setShowBookmarks(!showBookmarks);
              setShowLayers(false);
            }}
          >
            <MapPin size={13} />
          </IconButton>

          {showBookmarks && (
            <ul className="absolute right-0 top-[calc(100%+6px)] z-50 w-56 rounded-xl border border-line-strong bg-surface-2 py-1 shadow-2xl">
              {PRESET_BOOKMARKS.map((bookmark) => (
                <li key={bookmark.name}>
                  <button
                    type="button"
                    onClick={() => {
                      handleFlyToBookmark(bookmark.coords, bookmark.zoom, bookmark.name);
                      setShowBookmarks(false);
                    }}
                    className="flex w-full cursor-pointer items-center justify-between px-3 py-1.5 text-left text-[11.5px] text-ink-muted transition-colors hover:bg-surface-3 hover:text-ink"
                  >
                    <span className="truncate">{bookmark.name}</span>
                    <span className="ml-2 shrink-0 font-mono text-[10px] text-ink-faint">
                      z{bookmark.zoom}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Canvas                                                              */}
      {/* ------------------------------------------------------------------ */}
      <div className="relative min-h-0 flex-1 overflow-hidden bg-surface-2" onClick={closeMenus}>
        <div
          ref={mapContainerRef}
          style={getBandFilterStyle()}
          className="absolute inset-0 h-full w-full cursor-grab transition-[filter] duration-300 active:cursor-grabbing"
        />

        {showReticle && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-30">
            <Crosshair size={80} strokeWidth={0.7} className="text-accent" aria-hidden />
          </div>
        )}

        {/* Scene identity + detection count, top-left. */}
        {(activeOverlay || detectionCount > 0) && (
          <div className="pointer-events-none absolute left-2.5 top-2.5 z-10 max-w-[calc(100%-5.5rem)]">
            <div className="pointer-events-auto inline-flex max-w-full items-center gap-2 rounded-lg border border-line bg-ground/85 px-2.5 py-1.5 backdrop-blur-md">
              <Globe size={12} className="shrink-0 text-accent" aria-hidden />
              <span
                className="truncate font-mono text-[10.5px] text-ink"
                title={activeOverlay?.name ?? detectionLabel}
              >
                {activeOverlay?.name ?? detectionLabel}
              </span>
              {detectionCount > 0 && (
                <span className="shrink-0 rounded bg-accent/15 px-1.5 py-0.5 font-mono text-[9.5px] text-accent">
                  {detectionCount} region{detectionCount === 1 ? '' : 's'}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Navigation stack, top-right. */}
        <div className="absolute right-2.5 top-2.5 z-10 flex flex-col gap-1">
          <IconButton label="Zoom in" size="sm" onClick={handleZoomIn}>
            <ZoomIn size={13} />
          </IconButton>
          <IconButton label="Zoom out" size="sm" onClick={handleZoomOut}>
            <ZoomOut size={13} />
          </IconButton>
          <IconButton
            label={`Reset bearing (${Math.round(bearing)}°) and tilt`}
            size="sm"
            onClick={handleResetNorthPitch}
          >
            <Compass
              size={13}
              style={{ transform: `rotate(${-bearing}deg)` }}
              className="transition-transform"
            />
          </IconButton>
          {activeOverlay && (
            <IconButton label="Fit to loaded scene" size="sm" onClick={handleFlyToScene}>
              <Focus size={13} />
            </IconButton>
          )}
          {!isCompact && (
            <IconButton label="Centre on my location" size="sm" onClick={handleLocateMe}>
              <Locate size={13} />
            </IconButton>
          )}
          <IconButton
            label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen map'}
            size="sm"
            active={isFullscreen}
            onClick={toggleFullscreen}
          >
            {isFullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
          </IconButton>
        </div>

        {/* Coordinate readout, bottom-left. */}
        <div className="pointer-events-none absolute bottom-2.5 left-2.5 z-10">
          <div className="rounded-md border border-line bg-ground/85 px-2 py-1 font-mono text-[10px] text-ink-muted backdrop-blur-md">
            <span className="mr-1.5 font-semibold text-accent/80">NavIC / WGS84:</span>
            <span className={cursorCoords ? 'text-accent' : undefined}>
              {readout.lat}°, {readout.lng}°
            </span>
            <span className="mx-1.5 text-ink-faint" aria-hidden>
              |
            </span>
            <span>z{zoomLevel}</span>
            {pitch > 0 && (
              <>
                <span className="mx-1.5 text-ink-faint" aria-hidden>
                  |
                </span>
                <span>{pitch}° tilt</span>
              </>
            )}
          </div>
        </div>

        {/* --------------------------------------------------------------- */}
        {/* Unified Temporal / Multi-Temporal / Opacity Dock                */}
        {/* --------------------------------------------------------------- */}
        {(activeOverlay || (timelineData && timelineData.length > 1)) && showOverlayControls && (
          <div
            className={`absolute bottom-2.5 left-1/2 z-10 -translate-x-1/2 transition-all duration-200 ${
              isMultiTemporal && timelineData && timelineData.length > 1
                ? 'w-[min(36rem,calc(100%-1.25rem))]'
                : 'w-[min(30rem,calc(100%-1.25rem))]'
            }`}
          >
            <div className="rounded-xl border border-line bg-ground/90 p-2.5 shadow-2xl backdrop-blur-md">
              {/* Multi-Site Spatial Footprint Switcher Bar */}
              {activeOverlay?.sites && activeOverlay.sites.length > 1 && (
                <div className="mb-2 flex items-center gap-1.5 overflow-x-auto pb-1.5 border-b border-line/60 scrollbar-none">
                  <span className="shrink-0 text-[10px] font-mono text-ink-faint uppercase tracking-wider flex items-center gap-1">
                    <MapPin size={11} className="text-accent" />
                    Sites ({activeOverlay.sites.length}):
                  </span>
                  {activeOverlay.sites.map((st) => {
                    const isSelected = activeSiteId ? st.site_id === activeSiteId : st.site_id === activeOverlay.sites![0].site_id;
                    return (
                      <button
                        key={st.site_id}
                        type="button"
                        onClick={() => {
                          setActiveSiteId(st.site_id);
                          if (mapRef.current && st.bounds) {
                            mapRef.current.fitBounds(
                              [
                                [st.bounds[0], st.bounds[1]],
                                [st.bounds[2], st.bounds[3]]
                              ],
                              { padding: 85, maxZoom: 16.5, duration: 1800 }
                            );
                          }
                        }}
                        className={`shrink-0 cursor-pointer rounded-full px-2.5 py-0.5 font-mono text-[10px] transition-all flex items-center gap-1.5 border ${
                          isSelected
                            ? 'bg-accent/20 border-accent text-accent font-semibold shadow-[0_0_10px_rgba(0,242,255,0.25)]'
                            : 'bg-surface-2/80 border-line text-ink-muted hover:border-accent/40 hover:text-ink'
                        }`}
                        title={`Focus ${st.name} (${st.scene_count} Epochs, ${st.date_range})`}
                      >
                        <span className={`h-1.5 w-1.5 rounded-full ${isSelected ? 'bg-accent animate-pulse' : 'bg-ink-muted'}`} />
                        <span className="truncate max-w-[170px]">{st.name}</span>
                        <span className="text-[9px] opacity-75">({st.scene_count} Epochs)</span>
                      </button>
                    );
                  })}
                </div>
              )}

              {isMultiTemporal && timelineData && timelineData.length > 1 ? (
                <div className="flex flex-col gap-2">
                  {/* Top Bar: Playback Controls, Active Observation Chip, Change Badges, Opacity */}
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      {/* Play / Pause Toggle Button */}
                      <IconButton
                        label={isPlayingTimeline ? 'Pause timeline playback' : 'Play timeline animation'}
                        size="sm"
                        active={isPlayingTimeline}
                        onClick={() => {
                          setIsBlinking(false);
                          setIsPlayingTimeline(!isPlayingTimeline);
                        }}
                      >
                        {isPlayingTimeline ? <Pause size={12} className="text-amber-400" /> : <Play size={12} className="text-cyan-400" />}
                      </IconButton>

                      {/* Step Navigation Prev / Next */}
                      <div className="flex items-center">
                        <button
                          type="button"
                          title="Previous observation"
                          onClick={() => handleTimelineStepChange(Math.max(0, currentStep - 1))}
                          disabled={currentStep === 0}
                          className="flex h-6 w-6 items-center justify-center rounded-l-md border border-line bg-surface/80 text-ink-muted hover:bg-surface hover:text-ink disabled:opacity-40 transition-colors cursor-pointer"
                        >
                          <ChevronLeft size={13} />
                        </button>
                        <button
                          type="button"
                          title="Next observation"
                          onClick={() => handleTimelineStepChange(Math.min(timelineData.length - 1, currentStep + 1))}
                          disabled={currentStep >= timelineData.length - 1}
                          className="flex h-6 w-6 items-center justify-center rounded-r-md border-y border-r border-line bg-surface/80 text-ink-muted hover:bg-surface hover:text-ink disabled:opacity-40 transition-colors cursor-pointer"
                        >
                          <ChevronRight size={13} />
                        </button>
                      </div>

                      {/* Active Observation Status Chip */}
                      <div className="flex items-center gap-1.5 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 font-mono text-[10px] text-cyan-300">
                        <Calendar size={11} className="text-cyan-400 shrink-0" />
                        <span className="font-semibold">
                          {isBaseline ? 'T0 (Baseline)' : `T${activeObs?.step ?? currentStep}`}
                        </span>
                        <span className="text-cyan-400/90 font-medium">
                          {activeObs?.date ? (activeObs.date.length > 10 ? activeObs.date.slice(0, 10) : activeObs.date) : ''}
                        </span>
                        <span className="text-[9px] text-ink-faint">
                          ({currentStep + 1}/{timelineData.length})
                        </span>
                      </div>

                      {/* Dynamic Change Detection Metrics on Active Step */}
                      {!isBaseline && ((activeObs?.change_area_ha ?? 0) > 0) && (
                        <span className="hidden sm:inline-flex items-center rounded-full bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 font-mono text-[9.5px] font-medium text-amber-300">
                          +{((activeObs?.change_area_ha ?? 0)).toFixed(1)} ha ({((((activeObs?.change_fraction ?? 0))) * 100).toFixed(1)}%)
                        </span>
                      )}
                      {activeObs?.new_breakpoints && activeObs.new_breakpoints > 0 ? (
                        <span className="hidden sm:inline-flex items-center gap-1 rounded-full bg-rose-500/20 border border-rose-500/30 px-2 py-0.5 font-mono text-[9.5px] font-semibold text-rose-300">
                          <Sparkles size={9} />
                          {activeObs.new_breakpoints} new
                        </span>
                      ) : null}
                    </div>

                    {/* Right: Overlay Visibility & Opacity Slider */}
                    <div className="ml-auto flex min-w-0 flex-1 max-w-[12rem] items-center gap-1.5">
                      <IconButton
                        label={overlayOn ? 'Hide imagery overlay' : 'Show imagery overlay'}
                        size="sm"
                        active={overlayOn}
                        onClick={() => setIsOverlayVisible(!isOverlayVisible)}
                      >
                        {overlayOn ? <Eye size={12} /> : <EyeOff size={12} />}
                      </IconButton>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={Math.round(overlayOpacity * 100)}
                        aria-label="Overlay opacity"
                        onChange={(e) => setOverlayOpacity(parseFloat(e.target.value) / 100)}
                        className="min-w-0 flex-1"
                      />
                      <span className="w-8 shrink-0 text-right font-mono text-[10px] text-ink-muted">
                        {Math.round(overlayOpacity * 100)}%
                      </span>
                    </div>
                  </div>

                  {/* Scrubber & Timeline Date Marks */}
                  <div className="border-t border-line/50 pt-2">
                    <input
                      type="range"
                      min={0}
                      max={timelineData.length - 1}
                      value={currentStep}
                      aria-label="Multi-temporal step scrubber"
                      onChange={(e) => handleTimelineStepChange(parseInt(e.target.value, 10))}
                      className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-surface-2 accent-cyan-400 focus:outline-none"
                    />

                    {/* Date ticks under scrubber */}
                    <div className="mt-1.5 flex justify-between px-1">
                      {timelineData.map((obs, idx) => {
                        const active = idx === currentStep;
                        const past = idx <= currentStep;
                        const label = obs.date ? (obs.date.length >= 10 ? obs.date.slice(5, 10) : obs.date) : `T${idx}`;
                        return (
                          <button
                            key={idx}
                            type="button"
                            onClick={() => handleTimelineStepChange(idx)}
                            title={`Step T${idx}: ${obs.date} ${obs.is_baseline ? '(Baseline)' : ''}`}
                            className="group flex flex-col items-center gap-0.5 text-center focus:outline-none cursor-pointer"
                          >
                            <span
                              className={`h-1.5 w-1.5 rounded-full transition-transform ${
                                active
                                  ? 'scale-150 bg-cyan-400 ring-2 ring-cyan-400/40 shadow-sm shadow-cyan-400/50'
                                  : past
                                  ? 'bg-ink-muted group-hover:bg-ink'
                                  : 'bg-surface-3 group-hover:bg-ink-faint'
                              }`}
                            />
                            <span
                              className={`font-mono text-[9px] transition-colors ${
                                active
                                  ? 'font-bold text-cyan-300'
                                  : 'text-ink-faint group-hover:text-ink-muted'
                              }`}
                            >
                              {label}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Spectral capabilities & stack extent */}
                  <div className="flex flex-wrap items-center justify-between gap-1.5 border-t border-line/60 pt-1.5 text-[10px] font-mono">
                    <div className="flex items-center gap-1.5 text-ink-muted">
                      <span
                        className={`rounded px-1.5 py-0.5 ${
                          nirCapability.available
                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                            : 'bg-surface-2 text-ink-faint border border-line'
                        }`}
                        title={nirCapability.reason}
                      >
                        Obs T{currentStep}: RGB ✓ | NIR {nirCapability.available ? '✓' : '✗'} | NDVI {nirCapability.available ? '✓' : '✗'}
                      </span>
                    </div>
                    <div className="text-ink-faint text-[9.5px]">
                      {timelineData.length} observations ({timelineData[0]?.date.slice(0, 10)} → {timelineData[timelineData.length - 1]?.date.slice(0, 10)})
                    </div>
                  </div>
                </div>
              ) : (
                /* Bi-Temporal Pair & Single-Scene controls */
                <div>
                  <div className="flex items-center gap-2">
                    {activeOverlay?.mode === 'bi-temporal' ? (
                      <div className="flex items-center gap-0.5 rounded-lg border border-line bg-surface/80 p-0.5">
                        {TEMPORAL_MODES.map((mode) => (
                          <button
                            key={mode.value}
                            type="button"
                            title={mode.title}
                            onClick={() => setTemporalMode(mode.value)}
                            className={`cursor-pointer rounded-[6px] px-2 py-1 font-mono text-[10px] uppercase tracking-wide transition-colors ${
                              temporalMode === mode.value
                                ? 'bg-accent/15 text-accent'
                                : 'text-ink-faint hover:text-ink'
                            }`}
                          >
                            {mode.label}
                          </button>
                        ))}
                      </div>
                    ) : (
                      <span className="label-caps text-ink-faint">Overlay</span>
                    )}

                    {activeOverlay?.mode === 'bi-temporal' && (
                      <IconButton
                        label={isBlinking ? 'Stop blink comparison' : 'Blink between T1 and T2'}
                        size="sm"
                        active={isBlinking}
                        onClick={() => {
                          setIsPlayingTimeline(false);
                          setIsBlinking(!isBlinking);
                        }}
                      >
                        {isBlinking ? <Pause size={12} /> : <Play size={12} />}
                      </IconButton>
                    )}

                    <div className="ml-auto flex min-w-0 flex-1 items-center gap-1.5">
                      <IconButton
                        label={overlayOn ? 'Hide imagery overlay' : 'Show imagery overlay'}
                        size="sm"
                        active={overlayOn}
                        onClick={() => setIsOverlayVisible(!isOverlayVisible)}
                      >
                        {overlayOn ? <Eye size={12} /> : <EyeOff size={12} />}
                      </IconButton>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={Math.round(overlayOpacity * 100)}
                        aria-label="Overlay opacity"
                        onChange={(e) => setOverlayOpacity(parseFloat(e.target.value) / 100)}
                        className="min-w-0 flex-1"
                      />
                      <span className="w-8 shrink-0 text-right font-mono text-[10px] text-ink-muted">
                        {Math.round(overlayOpacity * 100)}%
                      </span>
                    </div>
                  </div>

                  {activeOverlay?.mode === 'bi-temporal' &&
                    (temporalMode === 'swipe' || temporalMode === 'fade') && (
                      <div className="mt-2 flex items-center gap-2 border-t border-line pt-2">
                        <span className="shrink-0 font-mono text-[10px] text-ink-faint">T1</span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={temporalSlider}
                          aria-label={temporalMode === 'swipe' ? 'Swipe position' : 'Blend between T1 and T2'}
                          onChange={(e) => {
                            setIsBlinking(false);
                            setTemporalSlider(parseInt(e.target.value, 10));
                          }}
                          className="min-w-0 flex-1"
                        />
                        <span className="shrink-0 font-mono text-[10px] text-ink-faint">T2</span>
                        <ArrowLeftRight size={11} className="shrink-0 text-accent" aria-hidden />
                      </div>
                    )}

                  {/* Authoritative Spectral Band & Index Capabilities */}
                  <div className="mt-2 flex flex-wrap items-center justify-between gap-1.5 border-t border-line/60 pt-1.5 text-[10px] font-mono">
                    <div className="flex items-center gap-1.5 text-ink-muted">
                      <span
                        className={`rounded px-1.5 py-0.5 ${
                          nirCapability.t1Available
                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                            : 'bg-surface-2 text-ink-faint border border-line'
                        }`}
                        title={nirCapability.t1Available ? 'T1 has Red, Green, Blue, NIR (B8)' : 'T1 lacks NIR band'}
                      >
                        T1: RGB ✓ | NIR {nirCapability.t1Available ? '✓' : '✗'} | NDVI {nirCapability.t1Available ? '✓' : '✗'}
                      </span>
                      {activeOverlay?.mode === 'bi-temporal' && (
                        <span
                          className={`rounded px-1.5 py-0.5 ${
                            nirCapability.t2Available
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              : 'bg-amber-500/10 text-amber-400/90 border border-amber-500/20'
                          }`}
                          title={nirCapability.t2Available ? 'T2 contains RGB only; NIR band is absent' : 'T2 contains RGB only; NIR band is absent'}
                        >
                          T2: RGB ✓ | NIR {nirCapability.t2Available ? '✓' : '✗'} | NDVI {nirCapability.t2Available ? '✓' : '✗'}
                        </span>
                      )}
                    </div>
                    {activeOverlay?.mode === 'bi-temporal' && !nirCapability.jointAvailable && (
                      <span className="text-amber-400/90 text-[9.5px]">
                        Bi-temporal ΔNDVI unavailable (T2 lacks NIR)
                      </span>
                    )}
                    {activeOverlay?.mode === 'bi-temporal' && nirCapability.asymmetricNote && (
                      <span className="text-amber-400/80 text-[9.5px]">
                        {nirCapability.asymmetricNote}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Spectral legend */}
        {showColorLegend && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-ground/70 p-4 backdrop-blur-sm">
            <div className="w-full max-w-sm rounded-panel border border-line-strong bg-surface-2 p-4 shadow-2xl">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div>
                  <h4 className="text-sm font-medium text-ink">Reading the imagery</h4>
                  <p className="mt-0.5 text-[11.5px] text-ink-faint">
                    {bandPreset === 'true-color' ? 'True colour' : 'False colour / enhanced'} rendering
                  </p>
                </div>
                <IconButton
                  label="Close legend"
                  size="sm"
                  onClick={() => setShowColorLegend(false)}
                >
                  <X size={13} />
                </IconButton>
              </div>

              <ul className="space-y-1.5">
                {LEGEND_ENTRIES.map((entry) => (
                  <li key={entry.label} className="flex items-start gap-2.5">
                    <span
                      className="mt-0.5 h-3 w-3 shrink-0 rounded-sm border border-line"
                      style={{ background: entry.swatch }}
                      aria-hidden
                    />
                    <span className="min-w-0">
                      <span className="block text-[12px] text-ink">{entry.label}</span>
                      <span className="block text-[11px] leading-snug text-ink-faint">
                        {entry.meaning}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>

              <p className="mt-3 border-t border-line pt-2.5 font-mono text-[10px] leading-relaxed text-ink-faint">
                Change polygons drawn by the detector are outlined in red and
                carry their area in hectares — click one on the map to inspect it.
              </p>
            </div>
          </div>
        )}

        {/* Interactive Map AOI Selector, Modal, and Chatbot Handoff (Phase 2) */}
        {ENABLE_AOI_FETCHER && (
          <>
            <AOIBoxSelector
              map={mapRef.current}
              isActive={isAOIActive}
              onCancel={() => setIsAOIActive(false)}
              onSelectBbox={(bbox, area) => {
                setSelectedBbox(bbox);
                setSelectedAreaKm2(area);
                setIsAOIActive(false);
                setShowFetchModal(true);
              }}
            />

            <AOIFetchModal
              bbox={selectedBbox}
              areaKm2={selectedAreaKm2}
              isOpen={showFetchModal}
              onClose={() => setShowFetchModal(false)}
              onSuccess={(response) => {
                setShowFetchModal(false);
                setFetchedScene(response);
                // Also fly to and align map with the fetched scene overlay
                const newOverlay = aoiResponseToSceneOverlay(response);
                if (mapRef.current && newOverlay.bounds && newOverlay.bounds.length === 4) {
                  const [minLng, minLat, maxLng, maxLat] = newOverlay.bounds;
                  if (
                    typeof minLng === 'number' && !isNaN(minLng) &&
                    typeof minLat === 'number' && !isNaN(minLat) &&
                    typeof maxLng === 'number' && !isNaN(maxLng) &&
                    typeof maxLat === 'number' && !isNaN(maxLat) &&
                    Math.abs(minLat) <= 90 && Math.abs(maxLat) <= 90
                  ) {
                    try {
                      mapRef.current.fitBounds(
                        [
                          [minLng, minLat],
                          [maxLng, maxLat],
                        ],
                        { padding: 80, duration: 1800 }
                      );
                    } catch (fitErr) {
                      console.warn('Could not fitBounds to fetched scene overlay:', fitErr);
                    }
                  }
                }
              }}
              onSuccessBiTemporal={(response) => {
                setShowFetchModal(false);
                setFetchedScene(response);
                const newDataset = aoiPairResponseToSceneDataset(response);
                const newOverlay = aoiPairResponseToSceneOverlay(response);
                setScene(newDataset, newOverlay);
                setTemporalMode('swipe');
                setTemporalSlider(50);
                if (mapRef.current && newOverlay.bounds && newOverlay.bounds.length === 4) {
                  const [minLng, minLat, maxLng, maxLat] = newOverlay.bounds;
                  if (
                    typeof minLng === 'number' && !isNaN(minLng) &&
                    typeof minLat === 'number' && !isNaN(minLat) &&
                    typeof maxLng === 'number' && !isNaN(maxLng) &&
                    typeof maxLat === 'number' && !isNaN(maxLat) &&
                    Math.abs(minLat) <= 90 && Math.abs(maxLat) <= 90
                  ) {
                    try {
                      mapRef.current.fitBounds(
                        [
                          [minLng, minLat],
                          [maxLng, maxLat],
                        ],
                        { padding: 80, duration: 1800 }
                      );
                    } catch (fitErr) {
                      console.warn('Could not fitBounds to bi-temporal overlay:', fitErr);
                    }
                  }
                }
              }}
              onRunChangeFormer={(response) => {
                setShowFetchModal(false);
                setFetchedScene(response);
                const newDataset = aoiPairResponseToSceneDataset(response);
                const newOverlay = aoiPairResponseToSceneOverlay(response);
                setScene(newDataset, newOverlay);
                setTemporalMode('swipe');
                setTemporalSlider(50);
                openDock('map');
                sendQuery(
                  `Run ChangeFormer change detection on this bi-temporal pair (${response.name}) to identify significant surface changes between ${response.t1?.acquisition_date || 'T1'} and ${response.t2?.acquisition_date || 'T2'}.`
                );
              }}
            />

            <AOIChatbotHandoff
              scene={fetchedScene}
              onDismiss={() => setFetchedScene(null)}
              onRunChangeFormer={(response) => {
                setFetchedScene(response);
                const newDataset = aoiPairResponseToSceneDataset(response);
                const newOverlay = aoiPairResponseToSceneOverlay(response);
                setScene(newDataset, newOverlay);
                setTemporalMode('swipe');
                setTemporalSlider(50);
                openDock('map');
                sendQuery(
                  `Run ChangeFormer change detection on this bi-temporal pair (${response.name}) to identify significant surface changes between ${response.t1?.acquisition_date || 'T1'} and ${response.t2?.acquisition_date || 'T2'}.`
                );
              }}
            />

            <ArchiveSearchModal
              isOpen={showArchiveModal}
              onClose={() => setShowArchiveModal(false)}
              currentMapBbox={
                mapRef.current
                  ? (() => {
                      const b = mapRef.current.getBounds();
                      return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
                    })()
                  : null
              }
              onSelectTileForMap={(tile) => {
                setShowArchiveModal(false);
                const thumbUrl = resolveAssetUrl(tile.thumbnail_url) || tile.thumbnail_url || '';
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
                  t1ImageUrl: thumbUrl,
                  t1Filename: tile.filename,
                };
                setScene(newDataset, newOverlay);
                openDock('map');
                if (mapRef.current && tile.bbox && tile.bbox.length === 4) {
                  mapRef.current.fitBounds(
                    [
                      [tile.bbox[0], tile.bbox[1]],
                      [tile.bbox[2], tile.bbox[3]],
                    ],
                    { padding: 80, duration: 1800 }
                  );
                }
              }}
              onLoadComparisonPair={(t1, t2) => {
                setShowArchiveModal(false);
                const sorted = [t1, t2].sort((a, b) => (a.date || '').localeCompare(b.date || ''));
                const older = sorted[0];
                const newer = sorted[1];
                const t1Url = resolveAssetUrl(older.thumbnail_url) || older.thumbnail_url || '';
                const t2Url = resolveAssetUrl(newer.thumbnail_url) || newer.thumbnail_url || '';

                const newDataset: SceneDataset = {
                  name: `${older.sensor} Pair (${older.date} vs ${newer.date})`,
                  sizeLabel: 'Bi-Temporal Pair',
                  sensor: older.sensor,
                  mode: 'bi-temporal',
                  crs: 'EPSG:4326 (WGS84)',
                  resolution: '10.0m',
                  syncedWithBackend: true,
                  georeferenced: true,
                  datasetId: `pair-${older.tile_id}-${newer.tile_id}`,
                  t1Filename: older.filename,
                  t2Filename: newer.filename,
                };

                const newOverlay: SceneOverlay = {
                  datasetId: `pair-${older.tile_id}-${newer.tile_id}`,
                  name: `Bi-Temporal Pair (${older.date} vs ${newer.date})`,
                  sensor: older.sensor,
                  mode: 'bi-temporal',
                  bounds: older.bbox || newer.bbox,
                  center: older.center || newer.center,
                  crs: 'EPSG:4326 (WGS84)',
                  resolution: '10.0m',
                  t1ImageUrl: t1Url,
                  t2ImageUrl: t2Url,
                  t1Filename: older.filename,
                  t2Filename: newer.filename,
                  scenes: [
                    {
                      step: 0,
                      step_index: 0,
                      filename: older.filename,
                      date: older.date,
                      acquisition_date: older.date,
                      sensor: older.sensor,
                      imageUrl: t1Url,
                      image_url: t1Url,
                      wgs84_bounds: older.bbox,
                    },
                    {
                      step: 1,
                      step_index: 1,
                      filename: newer.filename,
                      date: newer.date,
                      acquisition_date: newer.date,
                      sensor: newer.sensor,
                      imageUrl: t2Url,
                      image_url: t2Url,
                      wgs84_bounds: newer.bbox,
                    },
                  ],
                };

                setScene(newDataset, newOverlay);
                setTemporalMode('swipe');
                setTemporalSlider(50);
                openDock('map');
                const fitBbox = older.bbox || newer.bbox;
                if (mapRef.current && fitBbox && fitBbox.length === 4) {
                  mapRef.current.fitBounds(
                    [
                      [fitBbox[0], fitBbox[1]],
                      [fitBbox[2], fitBbox[3]],
                    ],
                    { padding: 80, duration: 1800 }
                  );
                }
              }}
              onLoadTimelineStack={(tiles, initialTile) => {
                setShowArchiveModal(false);
                if (tiles && tiles.length > 0) {
                  const sorted = [...tiles].sort((a, b) => (a.date || '').localeCompare(b.date || ''));
                  const first = sorted[0];
                  const latest = sorted[sorted.length - 1];
                  const isPair = sorted.length === 2;

                  const firstThumb = resolveAssetUrl(first.thumbnail_url) || first.thumbnail_url || '';
                  const latestThumb = resolveAssetUrl(latest.thumbnail_url) || latest.thumbnail_url || '';

                  const newDataset: SceneDataset = {
                    name: `${initialTile.sensor} Archive Stack (${sorted.length} scenes)`,
                    sizeLabel: 'Archive Stack',
                    sensor: initialTile.sensor,
                    mode: sorted.length >= 2 ? 'multi-temporal' : 'single',
                    crs: 'EPSG:4326 (WGS84)',
                    resolution: '10.0m',
                    syncedWithBackend: true,
                    georeferenced: true,
                    datasetId: `archive-timeline-${initialTile.tile_id}`,
                    t1Filename: first.filename,
                    t2Filename: latest.filename,
                  };

                  const newOverlay: SceneOverlay = {
                    datasetId: `archive-timeline-${initialTile.tile_id}`,
                    name: `${initialTile.sensor} Archive Timeline (${sorted.length} Epochs)`,
                    sensor: initialTile.sensor,
                    mode: sorted.length >= 2 ? 'multi-temporal' : 'single',
                    bounds: initialTile.bbox || latest.bbox,
                    center: initialTile.center || latest.center,
                    crs: 'EPSG:4326 (WGS84)',
                    resolution: '10.0m',
                    t1ImageUrl: latestThumb,
                    t2ImageUrl: isPair ? firstThumb : undefined,
                    t1Filename: latest.filename,
                    t2Filename: isPair ? first.filename : undefined,
                    scenes: sorted.map((s, idx) => ({
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
                  setActiveTimelineStep(sorted.length - 1);
                  openDock('map');

                  const targetBbox = initialTile.bbox || latest.bbox;
                  if (mapRef.current && targetBbox && targetBbox.length === 4) {
                    mapRef.current.fitBounds(
                      [
                        [targetBbox[0], targetBbox[1]],
                        [targetBbox[2], targetBbox[3]],
                      ],
                      { padding: 80, duration: 1800 }
                    );
                  }
                }
              }}
            />
          </>
        )}
      </div>
    </div>
  );
};

export default MapViewport;
