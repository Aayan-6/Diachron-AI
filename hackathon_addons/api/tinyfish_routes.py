from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from hackathon_addons.services.ground_truth_sentinel import ground_truth_sentinel
from hackathon_addons.services.tinyfish_client import tinyfish_service

router = APIRouter()


class InvestigationRequest(BaseModel):
    lat: float = Field(..., description="Latitude of change centroid or AOI center")
    lon: float = Field(..., description="Longitude of change centroid or AOI center")
    change_type: Optional[str] = Field("Vegetation Loss / Clearing", description="Detected change category")
    area_hectares: Optional[float] = Field(15.0, description="Estimated affected area in hectares")
    t1_date: Optional[str] = Field("2025-01-01", description="Baseline acquisition date (YYYY-MM-DD)")
    t2_date: Optional[str] = Field("2025-08-01", description="Post-change acquisition date (YYYY-MM-DD)")


@router.get("/health")
def tinyfish_health():
    """
    Checks connection to TinyFish AI Web Agent Infrastructure.
    """
    return {
        "status": "online",
        "service": "Autonomous Spatio-Temporal Investigator (TinyFish)",
        "configured": tinyfish_service.is_configured,
        "capabilities": [
            "Intent-Driven Web Search (TinyFish Search API)",
            "Dynamic SPA Content Extraction (TinyFish Fetch API)",
            "Autonomous Browser Actions (TinyFish Web Agent API)",
            "Cross-Modal Causal Attribution & Chronology",
            "Real-Time Server-Sent Events (SSE) Live Terminal",
        ],
    }


@router.post("/investigate")
async def run_investigation(req: InvestigationRequest):
    """
    Dispatches the autonomous investigator to corroborate satellite change findings
    with web gazettes, municipal tenders, and regional disaster/press reports.
    """
    try:
        report = await ground_truth_sentinel.investigate_aoi(
            lat=req.lat,
            lon=req.lon,
            change_type=req.change_type or "Vegetation Loss",
            area_hectares=req.area_hectares or 15.0,
            t1_date=req.t1_date or "2025-01-01",
            t2_date=req.t2_date or "2025-08-01",
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")


@router.get("/stream-investigation")
async def stream_investigation(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    change_type: str = Query("Vegetation Loss", description="Change Type"),
    area_hectares: float = Query(15.0, description="Area in ha"),
    t1_date: str = Query("2025-01-01", description="Start Date"),
    t2_date: str = Query("2025-08-01", description="End Date"),
):
    """
    Real-time Server-Sent Events (SSE) streaming endpoint for live agent terminal.
    """
    event_generator = ground_truth_sentinel.stream_investigation(
        lat=lat,
        lon=lon,
        change_type=change_type,
        area_hectares=area_hectares,
        t1_date=t1_date,
        t2_date=t2_date,
    )
    return StreamingResponse(event_generator, media_type="text/event-stream")
