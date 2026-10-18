import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure current directory is on sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from core.config import HOST, PORT, MODEL_SERVICE_URL, get_lan_ip, CORS_ORIGINS
from api.routes import router as satquery_router
from api.auth_routes import router as auth_router
from api.conversation_routes import router as conversation_router
from api.archive_routes import router as archive_router
from api.export_routes import router as export_router
from api.feedback_routes import router as feedback_router
from api.telemetry_routes import router as telemetry_router

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create SQLite / relational tables if not present
    try:
        from db.base import Base
        from db.session import engine
        import models  # noqa: F401
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[Warning] Could not initialize database schema: {e}")
    yield

app = FastAPI(
    title="Diachron AI Backend",
    description="Central Agentic Controller for Satellite Remote Sensing (SIH 2026 / ISRO PS 26167)",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS so local Vite dev server and remote hosts can make credentialed requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(conversation_router, prefix="/api/v1")
app.include_router(satquery_router, prefix="/api/v1")
app.include_router(archive_router, prefix="/api/v1/archive", tags=["Archive & Semantic Search"])
app.include_router(export_router, prefix="/api/v1/export", tags=["Analyst Export & Audit"])
app.include_router(feedback_router, prefix="/api/v1/feedback", tags=["Analyst Feedback & Subsequent Reranking"])
app.include_router(telemetry_router, prefix="/api/v1/system", tags=["System Telemetry & ISRO §2.3 Benchmarks"])

# Feature: Interactive Map STAC/COG AOI Streamer (Phase 2)
ENABLE_INTERACTIVE_MAP = True
if ENABLE_INTERACTIVE_MAP:
    try:
        from api.interactive_map_routes import router as interactive_map_router
        app.include_router(interactive_map_router, prefix="/api/v1/interactive-map", tags=["Interactive Map"])
    except Exception as e:
        print(f"[Warning] Interactive Map router could not be loaded: {e}")

# Feature: Autonomous Spatio-Temporal Ground-Truth Investigator (TinyFish Extension)
try:
    project_root = current_dir.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from hackathon_addons.api.tinyfish_routes import router as tinyfish_router
    app.include_router(tinyfish_router, prefix="/api/v1/tinyfish", tags=["TinyFish Ground-Truth Investigator"])
    print("[TinyFish] Autonomous Spatio-Temporal Investigator module successfully mounted at /api/v1/tinyfish")
except Exception as e:
    print(f"[Info] TinyFish module not mounted: {e}")


# Mount output directory for ChangeFormer / Change Detective masks, rasters, and GeoJSON files
# Note: Mounted BEFORE /static so /static does not shadow /static/outputs
candidate_output_dirs = [
    current_dir / "outputs",
    current_dir.parents[2] / "Bi-Temporal ChangeFormer" / "SatqueryAI" / "backend" / "outputs",
    current_dir.parents[1] / "Bi-Temporal ChangeFormer" / "SatqueryAI" / "backend" / "outputs",
    Path("/Users/divyatewari/Desktop/SatqueryAI/backend/outputs"),
]
outputs_dir = candidate_output_dirs[0]
for cand in candidate_output_dirs:
    if cand.exists():
        outputs_dir = cand
        break
os.makedirs(outputs_dir, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# Mount static directory for preview imagery & uploaded rasters
static_dir = os.path.join(current_dir, "temp_uploads")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    lan_ip = get_lan_ip()
    return {
        "service": "Diachron AI Backend",
        "status": "online",
        "local_url": f"http://localhost:{PORT}/api/v1",
        "documentation": f"http://localhost:{PORT}/docs",
        "remote_gpu_url": MODEL_SERVICE_URL if MODEL_SERVICE_URL else "Running on local GPU",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.get("/api/v1/status")
def system_status():
    """System status check reporting GPU, remote server connection, and active AI specialists."""
    lan_ip = get_lan_ip()
    gpu_available = False
    gpu_name = "None (CPU)"
    vram_gb = 0.0

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_name = "Apple Silicon (MPS)"
    except Exception:
        pass

    remote_status = "not_configured"
    remote_vram = None
    if MODEL_SERVICE_URL:
        import urllib.request
        import json
        try:
            health_url = f"{MODEL_SERVICE_URL.rstrip('/')}/health"
            req = urllib.request.Request(health_url, headers={"User-Agent": "SatQueryStatus/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                remote_status = data.get("status", "online")
                remote_vram = f"{data.get('vram_allocated_gb', 0)} / {data.get('vram_total_gb', 0)} GB ({data.get('device_name', 'GPU')})"
        except Exception as e:
            remote_status = f"offline or unreachable ({type(e).__name__})"

    return {
        "service": "Diachron AI",
        "status": "online",
        "execution_mode": "remote_gpu_api" if MODEL_SERVICE_URL else ("local_gpu_cuda" if gpu_available else "offline_fallback"),
        "model_engine": {
            "model_name": "Qwen3-VL-2B-Diachron (Multimodal S1/S2)",
            "model_checkpoint": "qwen3_vl_satquery_merged",
            "modalities_supported": ["Sentinel-1 SAR", "Sentinel-2 Optical", "Dual-Modal S1+S2"],
            "remote_service_url": MODEL_SERVICE_URL if MODEL_SERVICE_URL else None,
            "remote_host_status": remote_status if MODEL_SERVICE_URL else "N/A (running locally)",
            "remote_vram": remote_vram,
            "local_cuda_available": gpu_available,
            "local_device": gpu_name,
            "local_vram_gb": vram_gb,
        },
        "network": {
            "lan_ip": lan_ip,
            "port": PORT,
            "api_endpoint": f"http://localhost:{PORT}/api/v1/satquery",
        },
        "specialists": {
            "semantic_retrieval": "ready (RemoteCLIP Vision-Language Vector Search & Hybrid Ranking)",
            "multi_temporal_change": "ready (ChangeFormerV6 + IR-MAD + CVA + Onset Date Pinpointing)",
            "quality_false_alarm": "ready (SCL QA Masking + Radiometric Normalization)",
            "discovery_clustering": "ready (Spherical K-Means + Spatial AOI Expansion)",
            "analyst_provenance": "ready (Review Queue + RFC 7946 & W3C PROV Export)",
        },
    }


if __name__ == "__main__":
    import uvicorn
    lan_ip = get_lan_ip()
    print("\n" + "=" * 60)
    print("Diachron AI Central Backend Starting...")
    print(f"Local Access:  http://localhost:{PORT}")
    print(f"LAN Access:    http://{lan_ip}:{PORT}  <-- Use this from other laptops")
    print(f"API Docs:      http://localhost:{PORT}/docs")
    print("=" * 60 + "\n")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
