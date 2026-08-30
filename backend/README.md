# SIH 2026 Problem Statement 26143 - Backend API Layer (Member 6)

Backend integration layer for **Oil Spill Detection & Vessel Attribution System**.

Built with **Python**, **FastAPI**, **Pydantic v2**, **Uvicorn**, and **pytest**.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ installed

### 2. Installation
Navigate into the `backend/` directory and install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

### 3. Running Unit Tests
Execute pytest from the root of `backend/`:

```bash
pytest
```

Or with verbose output:

```bash
pytest -v
```

### 4. Running the Backend Server
Start the FastAPI application with Uvicorn:

```bash
uvicorn backend.main:app --reload --port 8000
```
*(Or if running directly inside `backend/` directory: `uvicorn main:app --reload --port 8000`)*

### 5. Interactive API Documentation (Swagger & ReDoc)
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI Schema**: [http://127.0.0.1:8000/api/openapi.json](http://127.0.0.1:8000/api/openapi.json)

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check & service metadata |
| `POST` | `/api/spill/detect` | Satellite oil spill detection (Mock / ML model wrapper) |
| `POST` | `/api/spill/backtrack` | Ocean drift backtracking (Mock / Physics model wrapper) |
| `POST` | `/api/ais/candidates` | Query AIS candidate vessels near spill source |
| `POST` | `/api/vessels/rank` | Rank suspect vessels responsible for oil spill |

---

## 🔌 How Future Team Members Should Plug In Real Modules

This backend uses a **Service Layer Architecture** with **FastAPI Dependency Injection**. All endpoints depend on abstract service base classes located in `backend/services/`.

To replace mock implementations with real models/APIs without breaking API routes or frontend contracts:

### Step 1: Implement the Abstract Interface

1. **Satellite ML Team Member**:
   Create `backend/services/real/spill_ml_service.py`:
   ```python
   from backend.services.spill_service import BaseSpillService
   from backend.schemas.spill import SpillDetectionRequest, SpillDetectionResponse

   class RealSatelliteMLService(BaseSpillService):
       def __init__(self, model_path: str):
           # Load your trained PyTorch / TensorFlow / ONNX model here
           pass

       async def detect_spill(self, request: SpillDetectionRequest) -> SpillDetectionResponse:
           # 1. Download image from request.image_url or load using request.image_id
           # 2. Perform ML segmentation / inference
           # 3. Extract bounding polygon & confidence
           # 4. Return SpillDetectionResponse(...)
           pass
   ```

2. **Ocean Drift Physics Team Member**:
   Create `backend/services/real/drift_physics_service.py`:
   ```python
   from backend.services.drift_service import BaseDriftService
   from backend.schemas.drift import BacktrackRequest, BacktrackResponse

   class RealDriftService(BaseDriftService):
       async def backtrack(self, request: BacktrackRequest) -> BacktrackResponse:
           # 1. Query ocean current and wind vectors
           # 2. Run OpenDrift / hydrodynamic backtrack simulation
           # 3. Return BacktrackResponse(...)
           pass
   ```

3. **AIS Data Integration Team Member**:
   Create `backend/services/real/ais_api_service.py`:
   ```python
   from backend.services.ais_service import BaseAISService
   from backend.schemas.ais import AISCandidatesRequest, AISCandidatesResponse

   class RealAISService(BaseAISService):
       async def get_candidates(self, request: AISCandidatesRequest) -> AISCandidatesResponse:
           # 1. Connect to live AIS database or provider API (e.g. Spire, AISHub, MarineTraffic)
           # 2. Filter vessels inside search_radius_km & time_window_hours
           # 3. Return AISCandidatesResponse(...)
           pass
   ```

4. **Vessel Attribution Algorithm Team Member**:
   Create `backend/services/real/vessel_attribution_service.py`:
   ```python
   from backend.services.vessel_service import BaseVesselService
   from backend.schemas.vessel import VesselRankRequest, VesselRankResponse

   class RealVesselAttributionService(BaseVesselService):
       async def rank_vessels(self, request: VesselRankRequest) -> VesselRankResponse:
           # 1. Apply spatio-temporal trajectory correlation algorithm
           # 2. Compute risk scores and attribution factor breakdown
           # 3. Return VesselRankResponse(...)
           pass
   ```

### Step 2: Swap the Dependency Factory

Open `backend/dependencies.py` and update the return instance in the corresponding factory function:

```python
# backend/dependencies.py

# Import your real service implementation:
from backend.services.real.spill_ml_service import RealSatelliteMLService

_spill_service_instance: BaseSpillService = RealSatelliteMLService(model_path="weights/best.pt")

def get_spill_service() -> BaseSpillService:
    return _spill_service_instance
```

That's it! **No API endpoints (`backend/api/*.py`) or frontend request/response schemas need to change.**

---

## 📁 Directory Structure

```
backend/
├── main.py                # FastAPI entrypoint & middleware configuration
├── config.py              # Application settings (CORS, versions, environment)
├── dependencies.py        # Dependency injection factories for services
├── requirements.txt       # Python package dependencies
├── README.md              # Project documentation
├── api/                   # FastAPI route handlers
│   ├── health.py
│   ├── spill.py
│   ├── ais.py
│   └── vessels.py
├── schemas/               # Pydantic request/response data contracts
│   ├── spill.py
│   ├── drift.py
│   ├── ais.py
│   └── vessel.py
├── services/              # Abstract service interfaces & real/mock modules
│   ├── spill_service.py   # Abstract BaseSpillService
│   ├── drift_service.py   # Abstract BaseDriftService
│   ├── ais_service.py     # Abstract BaseAISService
│   ├── vessel_service.py  # Abstract BaseVesselService
│   └── mock/              # Clean mock service implementations
│       ├── spill.py
│       ├── drift.py
│       ├── ais.py
│       └── vessel.py
└── tests/                 # pytest test suite
    ├── conftest.py
    ├── test_health.py
    ├── test_spill.py
    ├── test_ais.py
    └── test_vessels.py
```
