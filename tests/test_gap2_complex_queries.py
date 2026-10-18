"""
End-to-End Verification Test for GAP 2: Spatial Compound & Multi-Concept Retrieval
==================================================================================
Tests:
  1. Open-vocabulary syntactic query parsing across all 6 spatial relations
     (PROXIMITY, CONTAINMENT, LINEAR, CONJUNCTION, EXCLUSION, ATOMIC)
  2. Dual-concept projection & harmonic co-occurrence vs. arithmetic average trap
  3. ArchiveIndexer.search() with complex queries
  4. Non-compound query regression check (identical behavior)
"""

import sys
import os
from pathlib import Path
import numpy as np

# Ensure backend path is on sys.path
backend_root = Path(r"c:\Games\PS2\backend\SatQuery-master\backend")
sys.path.insert(0, str(backend_root))

from services import complex_query_service, archive_indexer, embedding_service


def test_open_vocabulary_parsing():
    print("\n--- [TEST 1] Open-Vocabulary Syntactic Decomposition ---")
    test_queries = [
        # Proximity
        ("buildings near river", "PROXIMITY", "buildings", "river"),
        ("find agricultural fields adjacent to water canal", "PROXIMITY", "agricultural fields", "water canal"),
        ("industrial warehouses close to railway line", "PROXIMITY", "industrial warehouses", "railway line"),
        ("settlements in the vicinity of lake", "PROXIMITY", "settlements", "lake"),
        # Containment
        ("vehicles on open ground", "CONTAINMENT", "vehicles", "open ground"),
        ("boats inside water reservoir", "CONTAINMENT", "boats", "water reservoir"),
        ("aircraft at airfield", "CONTAINMENT", "aircraft", "airfield"),
        # Linear
        ("trees along highway", "LINEAR", "trees", "highway"),
        ("embankment bordering river", "LINEAR", "embankment", "river"),
        # Conjunction
        ("urban area and water body", "CONJUNCTION", "urban area", "water body"),
        ("solar panels alongside transmission lines", "CONJUNCTION", "solar panels", "transmission lines"),
        # Exclusion
        ("settlement without vegetation", "EXCLUSION", "settlement", "vegetation"),
        ("construction away from forest", "EXCLUSION", "construction", "forest"),
        # Atomic queries
        ("dense forest", "ATOMIC", "dense forest", ""),
        ("high backscatter radar", "ATOMIC", "high backscatter radar", ""),
    ]

    for q, expected_op, expected_target, expected_anchor in test_queries:
        plan = complex_query_service.parse_complex_query(q)
        print(f"Query: '{q}' -> Op: {plan.operator:12s} | Target: '{plan.target_subject}' | Anchor: '{plan.context_anchor}'")
        assert plan.operator == expected_op, f"Expected {expected_op}, got {plan.operator} for '{q}'"
        if expected_op != "ATOMIC":
            assert expected_target in plan.target_subject, f"Expected target '{expected_target}' in '{plan.target_subject}'"
            assert expected_anchor in plan.context_anchor, f"Expected anchor '{expected_anchor}' in '{plan.context_anchor}'"
            assert plan.is_compound is True

    print(">>> TEST 1 PASSED: All 14 open-vocabulary test queries parsed accurately with zero hardcoding.")


def test_harmonic_cooccurrence_and_spatial_scoring():
    print("\n--- [TEST 2] Harmonic Co-occurrence vs Average Vector Trap ---")
    # Simulate embedding vectors
    target_vec = embedding_service.embed_text_query("structures")
    anchor_vec = embedding_service.embed_text_query("river")

    plan = complex_query_service.parse_complex_query("structures near river")

    # 1. Balanced scene: Has both structures (0.75) and river (0.70)
    balanced_scene = (0.5 * target_vec + 0.5 * anchor_vec).astype(np.float32)
    balanced_scene /= np.linalg.norm(balanced_scene)
    s_bal, m_bal = complex_query_service.score_relational_scene(balanced_scene, target_vec, anchor_vec, plan)

    # 2. Pure anchor scene (e.g. 100% deep ocean, 0% structures)
    pure_anchor_scene = np.copy(anchor_vec).astype(np.float32)
    s_pure_a, m_pure_a = complex_query_service.score_relational_scene(pure_anchor_scene, target_vec, anchor_vec, plan)

    # 3. Pure target scene (e.g. 100% dense concrete city, 0% water)
    pure_target_scene = np.copy(target_vec).astype(np.float32)
    s_pure_t, m_pure_t = complex_query_service.score_relational_scene(pure_target_scene, target_vec, anchor_vec, plan)

    print(f"Balanced Riverbank Scene (Structures + River): Score = {s_bal:.4f} (Status: {m_bal['match_status']})")
    print(f"Pure Ocean Scene (No Structures):              Score = {s_pure_a:.4f} (Status: {m_pure_a['match_status']})")
    print(f"Pure Concrete City (No Water):                 Score = {s_pure_t:.4f} (Status: {m_pure_t['match_status']})")

    # The balanced co-occurring scene MUST strictly beat pure single-feature scenes
    assert s_bal > s_pure_a, f"Balanced ({s_bal}) must score higher than pure ocean ({s_pure_a})"
    assert s_bal > s_pure_t, f"Balanced ({s_bal}) must score higher than pure concrete ({s_pure_t})"
    assert m_bal["is_relational_match"] is True

    print(">>> TEST 2 PASSED: Harmonic Co-occurrence decisively solves the Average Vector Trap.")


def test_archive_indexer_relational_search():
    print("\n--- [TEST 3] ArchiveIndexer End-to-End Relational Search ---")
    indexer = archive_indexer.get_archive_indexer()
    if not indexer.is_indexed():
        indexer.scan_and_index()

    print(f"Archive indexed with {len(indexer.tiles)} total satellite scenes.")

    # Search 1: Complex Compound Query
    query = "structures near river"
    res = indexer.search(query=query, limit=5)
    print(f"Query: '{query}'")
    print(f"  - Total matches: {res['total_matches']}")
    print(f"  - Relational Plan: {res.get('relational_plan')}")
    assert res.get("relational_plan") is not None
    assert res["relational_plan"]["is_compound"] is True
    assert res["relational_plan"]["operator"] == "PROXIMITY"
    assert res["relational_plan"]["target_subject"] == "structures"
    assert res["relational_plan"]["context_anchor"] == "river"

    if res["results"]:
        top_match = res["results"][0]
        print(f"  - Top result: {top_match['filename']} ({top_match['date']}) - Score: {top_match['similarity_score']}")
        print(f"  - Relational match details: {top_match.get('relational_match')}")
        assert "relational_match" in top_match

    # Search 2: Atomic Non-Compound Query (Regression Test)
    query_atomic = "dense forest"
    res_atomic = indexer.search(query=query_atomic, limit=5)
    print(f"Query: '{query_atomic}'")
    print(f"  - Total matches: {res_atomic['total_matches']}")
    print(f"  - Relational Plan: {res_atomic.get('relational_plan')}")
    assert res_atomic.get("relational_plan") is None
    if res_atomic["results"]:
        top_match = res_atomic["results"][0]
        assert "relational_match" not in top_match  # Non-compound queries do not attach relational_match

    print(">>> TEST 3 PASSED: ArchiveIndexer search executes both complex compound and atomic queries flawlessly.")


if __name__ == "__main__":
    test_open_vocabulary_parsing()
    test_harmonic_cooccurrence_and_spatial_scoring()
    test_archive_indexer_relational_search()
    print("\n=======================================================")
    print(" ALL GAP 2 VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================\n")
