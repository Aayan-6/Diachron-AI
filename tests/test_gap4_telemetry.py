"""
Unit & Integration Test Suite: ISRO PS 26167 §2.3 Reproducible Telemetry & Deliverables
======================================================================================
Verifies:
1. System Telemetry API (/api/v1/system/telemetry)
2. Live Hardware Diagnostics (GPU, VRAM, RAM, CPU, CUDA)
3. Spatio-Temporal Archive Coverage (Scene counts, bounding boxes, indexed area in km2)
4. Storage Footprint Breakdown (Rasters, Vectors, Embeddings, Checkpoints)
5. Algorithmic Latency Benchmarks (§2.2.1 - §2.2.5)
6. Model & Dataset Provenance
7. Official Reproducible Evaluation Report (/api/v1/system/evaluation-report)
8. Live Benchmark Trigger (/api/v1/system/benchmark)
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend directory is on sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend" / "SatQuery-master" / "backend"
sys.path.insert(0, str(backend_dir))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from main import app

client = TestClient(app)


def test_gap4_system_telemetry():
    print("=" * 75)
    print("   TEST 1: LIVE SYSTEM TELEMETRY ENDPOINT (§2.3)")
    print("=" * 75)

    resp = client.get("/api/v1/system/telemetry")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    # Core high-level keys
    for key in ("timestamp_utc", "system_status", "hardware", "archive_coverage", "storage_footprint", "latency_benchmarks", "provenance_and_compliance"):
        assert key in data, f"Missing top-level key: {key}"

    assert data["system_status"] == "ONLINE_HEALTHY"
    print("  [PASS] Status: ONLINE_HEALTHY")

    # 1. Hardware Verification
    hw = data["hardware"]
    assert "operating_system" in hw
    assert "python_version" in hw
    assert "ram_total_gb" in hw and hw["ram_total_gb"] > 0
    assert "cuda_available" in hw
    print(f"  [PASS] Hardware: {hw.get('operating_system')} | Python {hw.get('python_version')}")
    print(f"         GPU: {hw.get('gpu_name')} | VRAM: {hw.get('vram_total_gb')} GB")
    print(f"         Host RAM: {hw.get('ram_total_gb')} GB | CPU Logical: {hw.get('cpu_cores_logical')}")

    # 2. Archive Coverage Verification
    cov = data["archive_coverage"]
    assert cov["total_scenes"] >= 70, f"Expected >= 70 scenes, got {cov['total_scenes']}"
    assert cov["optical_sentinel_2_count"] > 0
    assert cov["radar_sentinel_1_count"] > 0
    assert len(cov["bounding_box_wgs84"]) == 4
    assert cov["cumulative_tile_footprint_km2"] > 0
    print(f"  [PASS] Archive: {cov['total_scenes']} scenes ({cov['optical_sentinel_2_count']} S2 Optical, {cov['radar_sentinel_1_count']} S1 SAR)")
    print(f"         Bounding Box: {cov['bounding_box_wgs84']}")
    print(f"         Cumulative Footprint: {cov['cumulative_tile_footprint_km2']} km²")

    # 3. Storage Footprint Verification
    storage = data["storage_footprint"]
    assert storage["index_metadata_json_kb"] > 0
    assert storage["vector_embeddings_matrix_kb"] > 0
    assert storage["raster_imagery_cache_mb"] > 0
    assert storage["total_system_storage_gb"] > 0
    print(f"  [PASS] Storage Footprint:")
    print(f"         - JSON Index: {storage['index_metadata_json_kb']} KB")
    print(f"         - Vector Matrix: {storage['vector_embeddings_matrix_kb']} KB")
    print(f"         - Raster Cache: {storage['raster_imagery_cache_mb']} MB")
    print(f"         - Total Footprint: {storage['total_system_storage_gb']} GB")

    # 4. Latency Benchmarks
    lat = data["latency_benchmarks"]
    assert lat["atomic_vector_retrieval_ms"] < 20.0, f"Vector search too slow: {lat['atomic_vector_retrieval_ms']}"
    assert lat["complex_relational_query_ms"] < 100.0, f"Complex query too slow: {lat['complex_relational_query_ms']}"
    assert lat["rocchio_active_feedback_rerank_ms"] < 10.0, f"Rocchio rerank too slow: {lat['rocchio_active_feedback_rerank_ms']}"
    assert lat["index_build_cold_seconds"] > 0
    assert lat["incremental_ingest_per_scene_seconds"] > 0
    print(f"  [PASS] Latency Benchmarks:")
    print(f"         - Atomic Vector Search: {lat['atomic_vector_retrieval_ms']:.3f} ms")
    print(f"         - Complex Relational Query (§2.2.1): {lat['complex_relational_query_ms']:.3f} ms")
    print(f"         - SAR Cross-Sensor Arbitration (§2.2.3): {lat['sentinel1_sar_arbitration_ms']:.3f} ms")
    print(f"         - Rocchio Feedback Reranker (§2.2.5): {lat['rocchio_active_feedback_rerank_ms']:.3f} ms")
    print(f"         - Bi-Temporal ChangeFormer: {lat['changeformer_siamese_inference_ms']} ms")
    print(f"         - Qwen3-VL Multimodal GPU: {lat['qwen3_vl_reasoning_gpu_seconds']} s")

    # 5. Provenance & ISRO Rubric Compliance
    prov = data["provenance_and_compliance"]
    assert "Qwen/Qwen3-VL-2B-Instruct" in prov["vision_language_backbone"]["model_name"]
    assert prov["vision_language_backbone"]["lora_rank"] == 16
    rubric = prov["rubric_compliance"]
    assert rubric["section_2_2_1_complex_spatial_compound_query"]["status"] == "COMPLIANT"
    assert rubric["section_2_2_2_multi_temporal_change_and_onset"]["status"] == "COMPLIANT"
    assert rubric["section_2_2_3_false_alarm_suppression_and_sar_arbiter"]["status"] == "COMPLIANT"
    assert rubric["section_2_2_4_conversational_vqa_and_multi_turn"]["status"] == "COMPLIANT"
    assert rubric["section_2_2_5_analyst_feedback_loop_and_reranking"]["status"] == "COMPLIANT"
    assert rubric["section_2_3_reproducible_evaluation_report_and_telemetry"]["status"] == "COMPLIANT"
    print("  [PASS] ISRO PS 26167 Rubric Compliance: ALL 6 SECTIONS COMPLIANT")


def test_gap4_evaluation_report_endpoint():
    print("\n" + "=" * 75)
    print("   TEST 2: OFFICIAL REPRODUCIBLE EVALUATION REPORT (§2.3)")
    print("=" * 75)

    resp = client.get("/api/v1/system/evaluation-report")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    text = resp.text

    assert len(text) > 5000, f"Report too short: {len(text)} bytes"
    assert "ISRO PS 26167: Reproducible Evaluation & System Telemetry Report" in text

    # Verify all mandated sections
    mandated_sections = [
        "Executive Summary",
        "Hardware Environment",
        "Indexed Archive & Spatial Coverage",
        "Index-Build & Incremental-Ingestion Procedure",
        "Storage Footprint Breakdown",
        "End-to-End Query Latency Benchmarks",
        "Model & Dataset Provenance",
        "Comprehensive Verification of Gaps Closed",
        "Step-by-Step Reproduction Guide",
    ]
    for section in mandated_sections:
        assert section in text, f"Missing mandated section in report: '{section}'"
        print(f"  [PASS] Section verified: '{section}'")

    print(f"  [PASS] Full Reproducible Evaluation Report Verified ({len(text)} bytes, markdown)")


def test_gap4_benchmark_trigger():
    print("\n" + "=" * 75)
    print("   TEST 3: ON-DEMAND LIVE BENCHMARK TRIGGER")
    print("=" * 75)

    resp = client.post("/api/v1/system/benchmark")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    bench = resp.json()
    assert bench["status"] == "success"
    assert "latency_benchmarks" in bench
    print(f"  [PASS] Benchmark execution verified on device: {bench.get('device')}")


if __name__ == "__main__":
    test_gap4_system_telemetry()
    test_gap4_evaluation_report_endpoint()
    test_gap4_benchmark_trigger()
    print("\n" + "=" * 75)
    print("   ALL GAP-4 EVALUATION & TELEMETRY TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
