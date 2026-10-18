from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SatQueryResponse(BaseModel):
    text_answer: str = Field(
        ...,
        description="Natural-language answer produced by the selected specialist.",
    )

    visual_evidence: List[Union[str, List[float], Dict[str, Any]]] = Field(
        default_factory=list,
        description="Paths, URLs, coordinate bounding boxes, or GeoJSON features/FeatureCollections.",
    )

    execution_trace: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent execution steps, diagnostics and summary.",
    )

    trace_log: Optional[List[str]] = Field(
        default=None,
        description="Flat chronological list of execution steps for UI rendering.",
    )

    conversation_id: Optional[str] = Field(
        default=None,
        description="Persisted conversation ID for this turn.",
    )

    band_contract: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Authoritative band capability contract for loaded imagery.",
    )

    analysis_mode: Optional[str] = Field(
        default=None,
        description="'bi-temporal' or 'multi-temporal' or 'single'.",
    )

    timeline: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Chronological change observations per scene date.",
    )

    candidates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Ranked change candidates with earliest observation and spectral change type.",
    )

    scene_overlay: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full SceneOverlay payload (mode, bounds, scenes with image_urls) to auto-mount onto the MapLibre viewport.",
    )

    provenance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured analytical provenance record (sensors, scene IDs, processing steps, algorithms, timestamps).",
    )


