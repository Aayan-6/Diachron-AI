"""
System Telemetry & ISRO PS 26167 §2.3 Benchmarks API Routes
============================================================
Exposes REST endpoints for judges, analysts, and frontend telemetry panels
to inspect live hardware performance, archive coverage, storage footprints,
query latencies, and official reproducible evaluation reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from services import system_telemetry_service

logger = logging.getLogger(__name__)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_PATH = PROJECT_ROOT / "REPRODUCIBLE_EVALUATION_REPORT.md"


@router.get("/telemetry", response_model=Dict[str, Any])
async def get_system_telemetry() -> Dict[str, Any]:
    """
    Returns live system telemetry, hardware specifications, archive coverage,
    storage footprint, and empirical latency benchmarks compliant with ISRO PS 26167 §2.3.
    """
    try:
        return system_telemetry_service.get_full_system_telemetry()
    except Exception as e:
        logger.error(f"Failed to generate system telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"System telemetry error: {str(e)}")


@router.get("/evaluation-report", response_class=PlainTextResponse)
async def get_evaluation_report_markdown() -> str:
    """
    Returns the full, reproducible evaluation report markdown for ISRO judges.
    """
    if not REPORT_PATH.exists():
        fallback_path = PROJECT_ROOT / "EVALUATION_REPORT.md"
        if fallback_path.exists():
            return fallback_path.read_text(encoding="utf-8")
        raise HTTPException(status_code=404, detail="Reproducible evaluation report not found.")

    try:
        return REPORT_PATH.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read evaluation report: {e}")


@router.post("/benchmark", response_model=Dict[str, Any])
async def run_live_benchmark() -> Dict[str, Any]:
    """
    Triggers a live empirical latency benchmark run across all core pipeline components:
    Atomic Vector Search, Complex Compound Relational Search (§2.2.1), and Rocchio Active Reranking (§2.2.5).
    """
    try:
        latencies = system_telemetry_service.get_latency_benchmarks()
        hardware = system_telemetry_service.get_hardware_telemetry()
        return {
            "status": "success",
            "message": "Live empirical benchmarks completed successfully.",
            "latency_benchmarks": latencies,
            "device": hardware.get("gpu_name"),
            "vram_gb": hardware.get("vram_total_gb"),
        }
    except Exception as e:
        logger.error(f"Live benchmark failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")
