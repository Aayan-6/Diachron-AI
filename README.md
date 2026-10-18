# Diachron AI 🛰️ × TinyFish AI 🌐

> **Autonomous Multimodal Earth Observation & Spatio-Temporal Ground-Truth Intelligence Platform**  
> *Fusing Bi-Temporal Satellite Vision with Autonomous AI Web Agents for Real-World Causal Verification*

---

## 🌟 Overview

**Diachron AI** is an advanced multimodal geospatial analysis platform that bridges the gap between **Earth Observation (EO) satellite remote sensing** and **live real-world ground truth**. 

Traditional satellite systems detect *that* a change occurred (e.g., 20 hectares of canopy loss or reservoir volume decline), but **cannot explain why**. Diachron AI pairs deep learning bi-temporal transformers with **TinyFish AI Web Agents** to autonomously investigate, corroborate, and attribute physical ground changes against live municipal gazettes, infrastructure tenders, and disaster bulletins.

```
       SATELLITE INTELLIGENCE                       AUTONOMOUS WEB AGENT
 ┌─────────────────────────────────┐         ┌─────────────────────────────────┐
 │ Sentinel-1 SAR / Sentinel-2 MSI │         │  TinyFish Search & Fetch APIs   │
 │   Qwen3-VL-2B Remote Sensing    │         │  Autonomous Chromium Agent Pool │
 │    ChangeFormerV6 Attention     │         │   Municipal & Disaster Mining   │
 └────────────────┬────────────────┘         └────────────────┬────────────────┘
                  │                                           │
                  │ [Spatial Change Mask]                     │ [Verified Ground Events]
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │    CROSS-MODAL CAUSAL ATTRIBUTION ENGINE  │
                  │  • Causality Confidence Score (0-100%)    │
                  │  • Risk & Compliance Classification       │
                  │  • Spatio-Temporal Event Chronology       │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │      MAPLIBRE CONSOLE & LIVE TERMINAL     │
                  │  • Real-time SSE Agent Terminal Stream    │
                  │  • Interactive MapLibre & 3D Globe        │
                  │  • Clickable Corroborating Web Evidence   │
                  └───────────────────────────────────────────┘
```

---

## ✨ Key Capabilities

### 1. 🌐 Autonomous Ground-Truth Sentinel (TinyFish AI Integration)
* **Spatial-Semantic Anchor Discovery:** Automatically converts change centroids $[lat, lon]$ into administrative boundaries (village, taluka, district, state) and infrastructure landmarks (highways, rivers, industrial corridors).
* **Intent-Driven News & Gazette Mining (TinyFish Search):** Formulates targeted spatiotemporal queries tailored to the observed change phenomenon and acquisition dates ($T_1 \to T_2$).
* **Deep Dynamic SPA Scraping (TinyFish Fetch):** Renders complex JavaScript single-page news outlets and government portals via warm Chromium instances to extract clean markdown articles.
* **Cross-Modal Causal Attribution:** Evaluates temporal correlation and keyword semantics to classify events:
  * `AUTHORIZED_INFRASTRUCTURE` (Planned highways, industrial corridors, tenders)
  * `CLIMATIC_OR_DISASTER_EVENT` (Reservoir spillway discharges, flood alerts, wildfires)
  * `SUSPECTED_UNAUTHORIZED_ACTIVITY` (Clearing without matching tender/clearance)
* **Real-time SSE Terminal:** Streams the autonomous web agent's thoughts and actions directly to the frontend dock via Server-Sent Events.

### 2. 🛰️ Deep Learning Satellite Vision & VQA
* **Bi-Temporal Change Detection:** Powered by **ChangeFormerV6** to compute high-resolution change probability heatmaps, IR-MAD multi-spectral differencing, and vector masks.
* **Visual Question Answering (VQA):** Fine-tuned **Qwen3-VL-2B** remote-sensing specialist model answering complex natural-language queries over multi-spectral bands.
* **Automated GIS Coregistration:** End-to-end preprocessing pipeline for Sentinel-1 GRD and Sentinel-2 MSI data with radiometric calibration, reprojection to EPSG:4326, and dynamic tiling.

### 3. 🗺️ Analyst Operations Console
* **MapLibre GL & 3D Globe:** High-performance raster and vector overlays with side-by-side epoch swipe comparisons.
* **Review Queue & Multi-Temporal Timeline:** Analyst triage workflow for flagging and validating multi-epoch change candidates.
* **Automated Export & Audit:** Standards-compliant GeoJSON, GeoTIFF, and comprehensive analytical PDF reports.

---

## 🏗️ Monorepo Structure

```
Diachron-AI/
├── hackathon_addons/                 # 🌐 TinyFish Autonomous Investigator Extension
│   ├── api/
│   │   └── tinyfish_routes.py        # Endpoints: /investigate, /stream-investigation, /health
│   └── services/
│       ├── tinyfish_client.py        # TinyFish Search, Fetch & Web Agent client
│       ├── spatial_anchor.py         # Reverse geocoder & query synthesizer
│       ├── causal_attribution.py     # Cross-modal attribution & confidence scoring
│       └── ground_truth_sentinel.py  # Master orchestrator & SSE event generator
│
├── backend/                          # ⚡ Central FastAPI Platform Backend
│   └── SatQuery-master/backend/
│       ├── api/                      # Routes: auth, interactive-map, archive, telemetry
│       ├── core/                     # Agentic loop, query planner & system lifecycle
│       ├── services/                 # Model client, change service, spectral indices
│       └── main.py                   # Central server entry point (port 8000)
│
├── frontend/                         # 💻 React 19 + TypeScript + Vite Console
│   ├── src/
│   │   ├── components/
│   │   │   ├── map/                  # MapLibre viewport & timeline slider
│   │   │   ├── scene/                # SceneDock, inspector, and review queue
│   │   │   └── tinyfish/             # GroundTruthSentinelPanel & Live Terminal
│   │   ├── context/                  # AppState unified context
│   │   └── lib/                      # tinyfishApi, interactiveMapApi, telemetry
│   └── package.json
│
├── gis_extraction_logic/             # 🌍 Satellite Remote Sensing Engine
│   ├── scripts/                      # S1 & S2 download and processing scripts
│   └── src/                          # SAR calibration and optical normalization
│
├── Model Training/                   # 🧠 VLM Training & GPU Server
│   ├── satquery_model.py             # Qwen3-VL local PyTorch inference pipeline
│   └── run_gpu_host.py               # Dedicated GPU server runner
│
├── Bi-Temporal ChangeFormer/         # 🔬 ChangeFormerV6 Inference Service
├── requirements.txt                  # Python dependencies (TinyFish, PyTorch, FastAPI)
├── run_local.bat                     # One-click Windows local full-stack launcher
└── .env.example                      # Configuration template
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Operating System:** Windows 10/11 or Linux
- **Python:** Python 3.11+ (Python 3.13 recommended)
- **Node.js:** Node.js 18+ & npm
- **TinyFish API Key:** Obtain from [tinyfish.ai](https://tinyfish.ai)

### 2. Environment Configuration
Create a `.env` file in the repository root:
```bash
cp .env.example .env
```

Configure your `.env` variables:
```env
# Central Backend Network Settings
HOST=0.0.0.0
PORT=8000

# TinyFish Autonomous Web Agent Infrastructure
TINYFISH_API_KEY=sk-tinyfish-your-api-key-here

# Leave empty to run on local GPU:
MODEL_SERVICE_URL=
```

### 3. Installation

**Python Dependencies:**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install tinyfish python-dotenv
```

**Frontend Dependencies:**
```bash
cd frontend
npm install
cd ..
```

---

## ⚡ Launching the Application

### Option A: One-Click Launcher (Windows)
Double-click `run_local.bat` or run:
```powershell
.\run_local.bat
```
This automatically boots:
* **FastAPI Central Backend:** `http://localhost:8000` (API Docs: `http://localhost:8000/docs`)
* **React Vite Console:** `http://localhost:5173/console`

### Option B: Manual Execution

```bash
# Terminal 1: Backend
cd backend/SatQuery-master/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend Console
cd frontend
npm run dev
```

---

## 📡 TinyFish API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/tinyfish/health` | Verifies TinyFish API key and active capabilities |
| `POST` | `/api/v1/tinyfish/investigate` | Full autonomous multi-vector ground-truth investigation |
| `GET` | `/api/v1/tinyfish/stream-investigation` | Real-time **Server-Sent Events (SSE)** streaming live agent terminal logs |

**Sample Investigation Request:**
```json
POST /api/v1/tinyfish/investigate
{
  "lat": 18.922,
  "lon": 72.834,
  "change_type": "Water level decline",
  "area_hectares": 24.5,
  "t1_date": "2025-01-01",
  "t2_date": "2025-08-01"
}
```

---

## 🧪 Testing & Verification

1. **Verify TinyFish End-to-End Pipeline:**
   ```bash
   python -c "import asyncio; from hackathon_addons.services.ground_truth_sentinel import ground_truth_sentinel; res = asyncio.run(ground_truth_sentinel.investigate_aoi(lat=18.922, lon=72.834, change_type='Water drop')); print('Status:', res['status'], '| Score:', res['attribution']['causality_score'])"
   ```

2. **Verify Frontend Production Build:**
   ```bash
   cd frontend
   npm run build
   ```

---

## 📄 License & Acknowledgements

* **TinyFish AI:** Powered by the [TinyFish AI Web Agent Infrastructure](https://tinyfish.ai).
* **Model Architectures:** [ChangeFormer](https://github.com/wgcban/ChangeFormer) & [Qwen-VL Team](https://github.com/QwenLM/Qwen-VL).
* **Earth Observation:** Copernicus Sentinel-1 and Sentinel-2 data courtesy of the European Space Agency (ESA) and ISRO.
