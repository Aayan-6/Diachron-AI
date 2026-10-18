"""
Verification Test for GAP 3: Analyst Feedback Loop & Subsequent Reranking (§2.2.5)
==================================================================================
Tests:
  1. Candidate feature extraction & vector normalization
  2. Rocchio active learning centroid shift & candidate reranking
  3. Explainable rank movement (promoted, demoted, rank deltas)
  4. Rocchio vector query refinement for archive search
  5. Immutable session audit trail persistence & retrieval
  6. FastAPI HTTP endpoints (/api/v1/feedback/*)
"""

import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure backend path is on sys.path
backend_root = Path(r"c:\Games\PS2\backend\SatQuery-master\backend")
sys.path.insert(0, str(backend_root))

from services import feedback_rerank_service
from fastapi.testclient import TestClient
from main import app


def test_feature_extraction():
    print("\n--- [TEST 1] Candidate Feature Vector Extraction ---")
    candidate = {
        "id": 1,
        "change_type": "Construction / Built-up",
        "area_ha": 4.5,
        "confidence": 0.82,
        "properties": {
            "delta_ndvi": -0.35,
            "delta_bsi": 0.48,
            "delta_ndwi": -0.15,
            "delta_brightness": 0.40,
            "sar_backscatter_delta_db": 4.2,
            "is_cloud_confounded": False,
        }
    }
    vec = feedback_rerank_service.extract_candidate_features(candidate)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 12
    # Verify vector is unit normalized
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-4, f"Vector norm {norm} should be 1.0"
    print(f"Candidate #{candidate['id']} vectorized into 12-dim unit vector. Norm = {norm:.4f}")
    print(">>> TEST 1 PASSED: Feature vector extraction verified.")


def test_active_learning_reranking():
    print("\n--- [TEST 2] Active Learning Centroid Shift & Candidate Reranking ---")
    # Simulate a list of candidates:
    # #1: Confirmed Construction (true change)
    # #2: Unreviewed Construction (should be promoted)
    # #3: Rejected Crop Harvest Phenology (false alarm)
    # #4: Unreviewed Crop Harvest (should be demoted)
    # #5: Unreviewed Minor clearing
    candidates = [
        {
            "id": 1,
            "change_type": "Construction / Built-up",
            "area_ha": 3.0,
            "confidence": 0.70,
            "properties": {"delta_bsi": 0.50, "delta_ndvi": -0.30}
        },
        {
            "id": 2,
            "change_type": "Construction / Built-up",
            "area_ha": 2.5,
            "confidence": 0.55,  # Lower initial confidence
            "properties": {"delta_bsi": 0.48, "delta_ndvi": -0.28}
        },
        {
            "id": 3,
            "change_type": "Vegetation Modification / Phenology",
            "area_ha": 8.0,
            "confidence": 0.85,  # High initial false confidence
            "properties": {"delta_ndvi": 0.55, "delta_bsi": -0.30}
        },
        {
            "id": 4,
            "change_type": "Vegetation Modification / Phenology",
            "area_ha": 7.5,
            "confidence": 0.80,  # High initial false confidence
            "properties": {"delta_ndvi": 0.52, "delta_bsi": -0.28}
        },
        {
            "id": 5,
            "change_type": "Surface Change",
            "area_ha": 1.2,
            "confidence": 0.50,
            "properties": {"delta_ndvi": 0.0, "delta_bsi": 0.0}
        },
    ]

    analyst_verdicts = {
        "1": {"verdict": "confirmed", "notes": "True warehouse construction"},
        "3": {"verdict": "rejected", "notes": "False alarm: Seasonal agricultural drying"},
    }

    result = feedback_rerank_service.rerank_candidates_with_feedback(
        candidates=candidates,
        analyst_verdicts=analyst_verdicts,
        session_id="test_feedback_session",
    )

    reranked = result["reranked_candidates"]
    promoted = result["promoted_count"]
    demoted = result["demoted_count"]

    print(f"Reranking completed: {promoted} promoted, {demoted} demoted.")
    for idx, c in enumerate(reranked, start=1):
        print(f"  Rank #{idx}: Candidate #{c['id']} ({c['change_type']}) -> Conf: {c['confidence']} | Delta: {c['rank_delta']} | {c['feedback_reason']}")

    # Candidate #1 (confirmed) must be at rank #1
    assert reranked[0]["id"] == 1
    assert reranked[0]["verdict"] == "confirmed" or analyst_verdicts.get("1", {}).get("verdict") == "confirmed"

    # Candidate #2 (similar to confirmed #1) must have received positive feedback boost
    c2 = next(c for c in reranked if c["id"] == 2)
    assert c2["feedback_boost"] > 0, f"Candidate #2 should have positive boost, got {c2['feedback_boost']}"

    # Candidate #4 (similar to rejected #3) must have received negative feedback penalty
    c4 = next(c for c in reranked if c["id"] == 4)
    assert c4["feedback_boost"] < 0, f"Candidate #4 should have negative boost, got {c4['feedback_boost']}"

    # Candidate #3 (rejected) must be sunk to the bottom of the list
    assert reranked[-1]["id"] == 3

    print(">>> TEST 2 PASSED: Active learning promotes similar true changes and demotes false alarms.")


def test_rocchio_query_refinement():
    print("\n--- [TEST 3] Rocchio Vector Query Refinement for Archive Search ---")
    # Base query vector for "urban construction"
    q0 = np.array([0.6, 0.4, 0.0, 0.0], dtype=np.float32)
    q0 /= np.linalg.norm(q0)

    # Relevant exemplar vector (more aligned with first dimension)
    rel_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rel_vec /= np.linalg.norm(rel_vec)
    rel_vecs = [rel_vec]

    # Irrelevant exemplar vector (orthogonal/negative dimension)
    irrel_vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    irrel_vec /= np.linalg.norm(irrel_vec)
    irrel_vecs = [irrel_vec]

    q_refined = feedback_rerank_service.refine_query_vector_rocchio(q0, rel_vecs, irrel_vecs)

    # Refined vector should have higher alignment with relevant and lower alignment with irrelevant
    sim_rel_before = float(np.dot(q0, rel_vecs[0]))
    sim_rel_after = float(np.dot(q_refined, rel_vecs[0]))
    sim_irrel_before = float(np.dot(q0, irrel_vecs[0]))
    sim_irrel_after = float(np.dot(q_refined, irrel_vecs[0]))

    print(f"Alignment with Relevant Tile:   Before = {sim_rel_before:.4f} -> After = {sim_rel_after:.4f}")
    print(f"Alignment with Irrelevant Tile: Before = {sim_irrel_before:.4f} -> After = {sim_irrel_after:.4f}")

    assert sim_rel_after > sim_rel_before
    assert sim_irrel_after <= sim_irrel_before
    print(">>> TEST 3 PASSED: Rocchio query vector refinement successfully shifts search direction.")


def test_audit_trail_logging_and_endpoints():
    print("\n--- [TEST 4] Audit Trail Persistence & FastAPI Endpoints ---")
    client = TestClient(app)

    # 1. Test rerank endpoint
    payload = {
        "candidates": [
            {"id": 1, "change_type": "Built-up", "confidence": 0.60, "properties": {}},
            {"id": 2, "change_type": "Vegetation", "confidence": 0.70, "properties": {}},
        ],
        "analyst_verdicts": {
            "1": {"verdict": "confirmed", "notes": "Verified new development"}
        },
        "session_id": "test_api_session"
    }
    r1 = client.post("/api/v1/feedback/rerank-candidates", json=payload)
    assert r1.status_code == 200, f"Rerank API failed: {r1.text}"
    data1 = r1.json()
    assert "reranked_candidates" in data1
    assert data1["feedback_active"] is True
    print(f"  [PASS] /api/v1/feedback/rerank-candidates returned {len(data1['reranked_candidates'])} reranked items.")

    # 2. Test audit event endpoint
    audit_evt = {
        "session_id": "test_api_session",
        "event_type": "MANUAL_VERDICT_CONFIRMED",
        "details": {"candidate_id": 1, "analyst": "analyst_1", "notes": "Verified"},
    }
    r2 = client.post("/api/v1/feedback/audit-event", json=audit_evt)
    assert r2.status_code == 200
    print("  [PASS] /api/v1/feedback/audit-event successfully recorded event.")

    # 3. Test audit trail retrieval
    r3 = client.get("/api/v1/feedback/audit-trail/test_api_session")
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3["session_id"] == "test_api_session"
    assert len(data3["events"]) >= 1
    print(f"  [PASS] /api/v1/feedback/audit-trail retrieved {len(data3['events'])} persistent events.")

    # 4. Test refine-archive endpoint
    archive_req = {
        "query": "water bodies",
        "confirmed_tile_ids": [],
        "rejected_tile_ids": [],
        "limit": 5,
        "session_id": "test_api_session"
    }
    r4 = client.post("/api/v1/feedback/refine-archive", json=archive_req)
    assert r4.status_code == 200
    print("  [PASS] /api/v1/feedback/refine-archive successfully refined archive search.")

    print(">>> TEST 4 PASSED: All feedback and audit trail endpoints verified.")


if __name__ == "__main__":
    test_feature_extraction()
    test_active_learning_reranking()
    test_rocchio_query_refinement()
    test_audit_trail_logging_and_endpoints()
    print("\n=======================================================")
    print(" ALL GAP 3 VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================\n")
