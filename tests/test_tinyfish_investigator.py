"""
Test Suite: TinyFish Autonomous Spatio-Temporal Ground-Truth Investigator
========================================================================
Verifies:
1. Health endpoint & capability registration
2. Spatial Anchor Resolution (Reverse Geocoding & Context Enrichment)
3. Targeted Query Generation for diverse remote-sensing phenomena
4. Causal Attribution Engine scoring & timeline chronology
5. End-to-End FastAPI /api/v1/tinyfish/investigate endpoint integration
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parents[1] / "backend" / "SatQuery-master" / "backend"
project_root = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from main import app
from hackathon_addons.services.spatial_anchor import SpatialAnchorResolver
from hackathon_addons.services.causal_attribution import CausalAttributionEngine
from hackathon_addons.services.tinyfish_client import tinyfish_service


def test_tinyfish_suite():
    print("=" * 70)
    print("   TINYFISH SPATIO-TEMPORAL INVESTIGATOR VERIFICATION SUITE")
    print("=" * 70)

    client = TestClient(app)

    # 1. Test Health Endpoint
    print("\n[1/5] Testing /api/v1/tinyfish/health endpoint...")
    resp = client.get("/api/v1/tinyfish/health")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    data = resp.json()
    assert data["status"] == "online"
    assert data["configured"] is True, "TinyFish API key not configured"
    assert len(data["capabilities"]) >= 5
    print(f"  [PASS] Health check online. Configured: {data['configured']}")
    print(f"  [PASS] Capabilities: {len(data['capabilities'])} verified")

    # 2. Test Spatial Anchor Reverse Geocode
    print("\n[2/5] Testing Spatial Anchor Resolver (Centroid -> Parcel)...")
    anchor = SpatialAnchorResolver.reverse_geocode(18.922, 72.834)
    assert "country" in anchor and anchor["country"] == "India"
    assert anchor.get("state") == "Maharashtra" or anchor.get("display_name")
    print(f"  [PASS] Spatial Anchor Resolved: {anchor.get('display_name')}")

    # 3. Test Query Formulation for Different Phenomena
    print("\n[3/5] Testing Multi-Phenomenon Query Generation...")
    hydro_queries = SpatialAnchorResolver.generate_investigative_queries(
        spatial_context=anchor,
        change_type="Reservoir water volume drop",
    )
    assert len(hydro_queries) >= 2
    assert any("reservoir" in q["query"].lower() or "dam" in q["query"].lower() for q in hydro_queries)

    veg_queries = SpatialAnchorResolver.generate_investigative_queries(
        spatial_context=anchor,
        change_type="Forest canopy loss / clearing",
    )
    assert any("forest" in q["query"].lower() or "tree" in q["query"].lower() for q in veg_queries)
    print(f"  [PASS] Hydro query formulated: '{hydro_queries[0]['query']}'")
    print(f"  [PASS] Vegetation query formulated: '{veg_queries[0]['query']}'")

    # 4. Test Causal Attribution Engine
    print("\n[4/5] Testing Cross-Modal Causal Attribution & Chronology...")
    sample_sat = {
        "change_type": "Water level decline",
        "area_hectares": 30.0,
        "t1_date": "2025-01-01",
        "t2_date": "2025-08-01",
    }
    sample_web = [
        {
            "title": "State Water Department Orders Floodgate Spillway Release",
            "snippet": "Due to heavy catchment inflow, 4 gates were opened for planned discharge.",
            "url": "https://example.gov/water-bulletin",
            "domain": "example.gov",
            "published_date": "2025-07-20",
        }
    ]
    attr = CausalAttributionEngine.synthesize_attribution(sample_sat, sample_web, anchor)
    assert attr["causality_score"] >= 70.0
    assert attr["classification"] == "CLIMATIC_OR_DISASTER_EVENT"
    assert len(attr["timeline"]) == 3
    print(f"  [PASS] Causality Confidence: {attr['causality_score']}% ({attr['confidence_grade']})")
    print(f"  [PASS] Classification: {attr['classification_label']}")
    print(f"  [PASS] Chronology: {len(attr['timeline'])} milestones verified")

    # 5. Test Live Investigation Endpoint
    print("\n[5/5] Testing FastAPI /api/v1/tinyfish/investigate POST endpoint...")
    inv_resp = client.post(
        "/api/v1/tinyfish/investigate",
        json={
            "lat": 18.922,
            "lon": 72.834,
            "change_type": "Water level decline",
            "area_hectares": 24.5,
            "t1_date": "2025-01-01",
            "t2_date": "2025-08-01",
        },
    )
    assert inv_resp.status_code == 200, f"Investigation failed: {inv_resp.status_code} - {inv_resp.text}"
    inv_data = inv_resp.json()
    assert inv_data["status"] == "success"
    assert "attribution" in inv_data
    assert "corroborating_sources" in inv_data["attribution"]
    print(f"  [PASS] Investigation successful! Scanned {inv_data['total_sources_scanned']} web sources.")
    print(f"  [PASS] Discovered {len(inv_data['attribution']['corroborating_sources'])} corroborating reports.")

    print("\n" + "=" * 70)
    print("   ALL 5 TINYFISH INVESTIGATOR TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    test_tinyfish_suite()
