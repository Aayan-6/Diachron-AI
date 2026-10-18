"""
Generalized Complex & Relational Query Engine (ISRO PS 26167 §2.2.1)
====================================================================
Provides open-vocabulary syntactic decomposition, dual-concept embedding projection,
harmonic co-occurrence filtering, and 4x4 spatial quadrant distance evaluation for any
complex natural language Earth Observation query.

Handles 6 universal relational patterns:
  1. PROXIMITY: "near", "adjacent to", "close to", "next to", "beside", "in the vicinity of"
  2. CONTAINMENT: "on", "in", "inside", "within", "at", "upon"
  3. LINEAR: "along", "bordering", "lining", "skirting", "flanking"
  4. CONJUNCTION: "and", "with", "featuring", "plus", "alongside"
  5. EXCLUSION: "without", "excluding", "not near", "away from"
  6. ATOMIC: standard single-concept queries (fallback to standard cosine)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Universal Spatial Prepositions and Relational Operators
# Ordered by token length descending to match multi-word connectors first
RELATIONAL_CONNECTORS = [
    # Proximity
    ("in the vicinity of", "PROXIMITY"),
    ("adjacent to", "PROXIMITY"),
    ("close to", "PROXIMITY"),
    ("next to", "PROXIMITY"),
    ("nearby", "PROXIMITY"),
    ("near", "PROXIMITY"),
    ("beside", "PROXIMITY"),
    ("around", "PROXIMITY"),
    ("surrounding", "PROXIMITY"),
    ("neighboring", "PROXIMITY"),
    # Containment / Coincidence
    ("situated on", "CONTAINMENT"),
    ("located on", "CONTAINMENT"),
    ("inside", "CONTAINMENT"),
    ("within", "CONTAINMENT"),
    ("upon", "CONTAINMENT"),
    ("on", "CONTAINMENT"),
    ("in", "CONTAINMENT"),
    ("at", "CONTAINMENT"),
    # Linear / Boundary Following
    ("bordering on", "LINEAR"),
    ("bordering", "LINEAR"),
    ("flanking", "LINEAR"),
    ("skirting", "LINEAR"),
    ("lining", "LINEAR"),
    ("along", "LINEAR"),
    # Exclusion / Negation
    ("not near", "EXCLUSION"),
    ("away from", "EXCLUSION"),
    ("far from", "EXCLUSION"),
    ("excluding", "EXCLUSION"),
    ("without", "EXCLUSION"),
    # Conjunction
    ("alongside", "CONJUNCTION"),
    ("featuring", "CONJUNCTION"),
    ("with", "CONJUNCTION"),
    ("and", "CONJUNCTION"),
]

# Words that should not be treated as spatial anchors if they appear alone
STOPWORDS = {"a", "an", "the", "some", "any", "all", "of", "to", "for", "with"}


@dataclass
class RelationalQueryPlan:
    """Represents the decomposed semantic execution plan for any complex query."""
    raw_query: str
    is_compound: bool = False
    operator: str = "ATOMIC"  # PROXIMITY, CONTAINMENT, LINEAR, CONJUNCTION, EXCLUSION, ATOMIC
    connector_phrase: str = ""
    target_subject: str = ""
    context_anchor: str = ""
    spatial_proximity_threshold_km: float = 0.5  # Nominal ~500m proximity
    explanation: str = ""


def parse_complex_query(query: str) -> RelationalQueryPlan:
    """
    Syntactically decomposes any arbitrary free-text satellite query into Target,
    Relational Operator, and Context Anchor without hardcoding domain phrases.
    """
    clean = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not clean:
        return RelationalQueryPlan(raw_query=query, is_compound=False, operator="ATOMIC")

    # Search for spatial connectors
    for connector, op_type in RELATIONAL_CONNECTORS:
        # Match connector bounded by word boundaries or spaces
        pattern = r"\b" + re.escape(connector) + r"\b"
        match = re.search(pattern, clean)
        if match:
            start_idx, end_idx = match.span()
            subject_part = clean[:start_idx].strip()
            anchor_part = clean[end_idx:].strip()

            # Clean stopwords
            subject_clean = re.sub(r"^(find|search|show|locate|detect|identify)\s+", "", subject_part).strip()
            anchor_clean = re.sub(r"^(a|an|the)\s+", "", anchor_part).strip()

            # Validate that both subject and anchor contain meaningful content (at least 2 letters)
            if len(subject_clean) >= 2 and len(anchor_clean) >= 2 and subject_clean not in STOPWORDS and anchor_clean not in STOPWORDS:
                return RelationalQueryPlan(
                    raw_query=query,
                    is_compound=True,
                    operator=op_type,
                    connector_phrase=connector,
                    target_subject=subject_clean,
                    context_anchor=anchor_clean,
                    explanation=f"Parsed relational query: [{subject_clean}] --({op_type}: '{connector}')--> [{anchor_clean}]"
                )

    # Standard atomic single-concept query
    return RelationalQueryPlan(
        raw_query=query,
        is_compound=False,
        operator="ATOMIC",
        target_subject=clean,
        explanation="Standard atomic single-concept query."
    )


def extract_quadrant_activation_grid(embedding_vector: np.ndarray) -> np.ndarray:
    """
    Extracts the 4x4 spatial quadrant activation grid from a 512-dim embedding vector.
    Dimensions 64..191 in embedding_service contain 16 quadrant poolings x 8 features.
    Returns a (4, 4) float32 matrix representing localized spatial energy.
    """
    if len(embedding_vector) < 192:
        return np.ones((4, 4), dtype=np.float32)

    quad_slice = embedding_vector[64:192]
    # Reshape into 16 quadrants x 8 features
    try:
        quad_blocks = quad_slice.reshape((16, 8))
        # Compute spatial energy magnitude per quadrant
        quad_energies = np.linalg.norm(quad_blocks, axis=1)
        grid = quad_energies.reshape((4, 4))
        # Normalize grid to [0, 1]
        g_max = float(np.max(grid))
        g_min = float(np.min(grid))
        if g_max - g_min > 1e-6:
            grid = (grid - g_min) / (g_max - g_min)
        else:
            grid = np.ones((4, 4), dtype=np.float32) * 0.5
        return grid.astype(np.float32)
    except Exception:
        return np.ones((4, 4), dtype=np.float32) * 0.5


def compute_quadrant_spatial_distance(
    scene_vector: np.ndarray,
    target_vector: np.ndarray,
    anchor_vector: np.ndarray,
) -> float:
    """
    Calculates the minimum physical grid distance between where the Target Concept
    and Context Anchor manifest across the scene's 16 spatial quadrants (4x4 grid).
    
    Returns:
      - 0.0: Target and Anchor overlap in the same quadrant (coincidence)
      - 1.0 to 1.414: Adjacent quadrants (~300m - 500m proximity)
      - > 2.0: Distant quadrants (> 2km apart, not near)
    """
    scene_grid = extract_quadrant_activation_grid(scene_vector)
    target_grid = extract_quadrant_activation_grid(target_vector)
    anchor_grid = extract_quadrant_activation_grid(anchor_vector)

    # Compute localized spatial alignment across each quadrant (r, c)
    target_alignment = scene_grid * target_grid
    anchor_alignment = scene_grid * anchor_grid

    # Find quadrants with significant activation (> 50th percentile)
    t_thresh = float(np.percentile(target_alignment, 50))
    a_thresh = float(np.percentile(anchor_alignment, 50))

    t_coords = []
    a_coords = []

    for r in range(4):
        for c in range(4):
            if target_alignment[r, c] >= t_thresh:
                t_coords.append((r, c))
            if anchor_alignment[r, c] >= a_thresh:
                a_coords.append((r, c))

    if not t_coords or not a_coords:
        return 1.414  # Default nominal adjacent distance

    min_dist = 99.0
    for r1, c1 in t_coords:
        for r2, c2 in a_coords:
            d = math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
            if d < min_dist:
                min_dist = d

    return min_dist


def score_relational_scene(
    scene_vector: np.ndarray,
    target_vector: np.ndarray,
    anchor_vector: np.ndarray,
    plan: RelationalQueryPlan,
) -> Tuple[float, Dict[str, Any]]:
    """
    Computes a generalized relational score for a scene against any complex query plan.
    Applies harmonic co-occurrence, spatial quadrant proximity weighting, and exclusion penalties.
    """
    # 1. Individual Cosine Similarities
    s_target = float(np.dot(scene_vector, target_vector))
    s_anchor = float(np.dot(scene_vector, anchor_vector))

    # Clamp to non-negative
    s_t_safe = max(0.01, s_target)
    s_a_safe = max(0.01, s_anchor)

    # 2. Generalized Harmonic Co-occurrence Formulation
    harmonic_cooccur = (2.0 * s_t_safe * s_a_safe) / (s_t_safe + s_a_safe + 1e-6)

    # 3. Spatial Quadrant Grid Distance
    d_quad = compute_quadrant_spatial_distance(scene_vector, target_vector, anchor_vector)

    relational_multiplier = 1.0
    operator = plan.operator
    match_status = "COOCCURRING"

    if operator == "PROXIMITY":
        # Target must be near Anchor (d <= 1.414)
        if d_quad <= 1.414:
            relational_multiplier = 1.40  # Proximity boost
            match_status = f"PROXIMITY_MATCH (d={d_quad:.1f} quads)"
        elif d_quad <= 2.0:
            relational_multiplier = 1.15
            match_status = f"MODERATE_PROXIMITY (d={d_quad:.1f} quads)"
        else:
            relational_multiplier = 0.70  # Penalty: too far apart
            match_status = f"DISTANT_SEPARATION (d={d_quad:.1f} quads)"

    elif operator == "CONTAINMENT":
        # Target must be directly on/in Anchor (d == 0)
        if d_quad == 0.0:
            relational_multiplier = 1.55  # Direct spatial coincidence boost
            match_status = "CONTAINMENT_COINCIDENCE (Same Quadrant)"
        elif d_quad <= 1.0:
            relational_multiplier = 1.10
            match_status = f"CONTAINMENT_PERIPHERY (d={d_quad:.1f} quads)"
        else:
            relational_multiplier = 0.60  # Penalty: not contained
            match_status = f"NOT_CONTAINED (d={d_quad:.1f} quads)"

    elif operator == "LINEAR":
        # Linear feature following
        if d_quad <= 1.414:
            relational_multiplier = 1.45  # Boundary following boost
            match_status = f"LINEAR_BOUNDARY_ALIGNMENT (d={d_quad:.1f} quads)"
        else:
            relational_multiplier = 0.75

    elif operator == "CONJUNCTION":
        # Joint presence of both concepts
        relational_multiplier = 1.25
        match_status = "JOINT_CONJUNCTION"

    elif operator == "EXCLUSION":
        # Target must be present, Anchor must be absent
        # Severe penalty if anchor is present
        anchor_penalty = max(0.05, 1.0 - 2.5 * max(0.0, s_anchor))
        final_score = s_target * anchor_penalty
        return round(float(final_score), 4), {
            "target_similarity": round(s_target, 3),
            "excluded_anchor_similarity": round(s_anchor, 3),
            "harmonic_cooccurrence": round(s_target, 3),
            "spatial_quadrant_distance": d_quad,
            "relational_multiplier": round(anchor_penalty, 3),
            "match_status": "EXCLUSION_APPLIED" if s_anchor < 0.25 else "EXCLUDED_FEATURE_PRESENT",
            "is_relational_match": s_anchor < 0.25 and s_target > 0.45,
        }

    # Combined Balanced Score: 60% Harmonic Co-occurrence * Spatial Multiplier + 40% Target Focus
    final_score = (0.60 * harmonic_cooccur * relational_multiplier) + (0.40 * s_target)
    final_score = min(0.99, max(0.0, final_score))

    is_relational_match = (
        s_target >= 0.35
        and s_anchor >= 0.35
        and relational_multiplier >= 1.0
    )

    metadata = {
        "target_similarity": round(s_target, 3),
        "anchor_similarity": round(s_anchor, 3),
        "harmonic_cooccurrence": round(harmonic_cooccur, 3),
        "spatial_quadrant_distance": round(d_quad, 2),
        "relational_multiplier": round(relational_multiplier, 2),
        "match_status": match_status,
        "is_relational_match": is_relational_match,
        "target_subject": plan.target_subject,
        "context_anchor": plan.context_anchor,
        "operator": plan.operator,
    }

    return round(float(final_score), 4), metadata
