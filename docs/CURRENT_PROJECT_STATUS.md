# SIH 2026 — Current Project Status

**Project Title**: Leveraging Satellite Imagery to Determine Oil Spills at Sea along with AIS Data Correlations to Identify the Vessel Responsible  
**Audit Timestamp**: 2026-08-29  
**Audit Author**: Member 6 (Backend / Integration Layer)  
**Audit Target**: Complete Git Repository Audit (including `origin/integration-m1-m2-m3` branch)

---

## 1. Audit Date & Time

- **Date**: August 29, 2026
- **Scope**: Comprehensive read-only empirical repository audit across all local and remote branches (`main`, `member1-satellite`, `member2-drift`, `member3-ais`, `member4-attribution`, `member5-frontend`, `member6-backend`, `integration-m1-m2-m3`).

---

## 2. Git Branch Status

| Branch Name | Remote Target | Latest Commit Hash | Latest Commit Message | Status |
| :--- | :--- | :--- | :--- | :--- |
| `main` | `origin/main` | `5f036d0` | Initial commit | Clean |
| `member1-satellite` | `origin/member1-satellite` | `0e882ee` | Integrate drift backtracking with oil spill detection | Active |
| `member2-drift` | `origin/member2-drift` | `ef9063e` | Complete Member 2 drift origin pipeline | Active |
| `member3-ais` | `origin/member3-ais` | `23c160d` | Finalize AIS vessel tracking and candidate filtering | Active |
| `member4-attribution` | `origin/member4-attribution` | `31fe71a` | Implement vessel attribution module and demo | Active |
| `member5-frontend` | `origin/member5-frontend` | `1885f3c` | feat: complete interactive frontend | Active |
| `member6-backend` *(current)* | `origin/member6-backend` | `5f036d0` (+ local) | Backend integration layer & API contracts | Active |
| **`integration-m1-m2-m3`** | `origin/integration-m1-m2-m3` | **`90bd488`** | **Integrate M1 M2 and M3 end-to-end** | **NEW Combined** |

---

## 3. Combined M1/M2/M3 Branch Status

- **Exact Branch Name**: `origin/integration-m1-m2-m3`
- **Latest Commit**: `90bd488` (*"Integrate M1 M2 and M3 end-to-end"*)
- **Pushed to Origin**: Yes (`origin/integration-m1-m2-m3`)
- **Contents**: Combines Member 1 (`ml/`), Member 2 (`src/drift/`), and Member 3 (`ais/`) along with end-to-end adapters (`drift_adapter.py`, `ais_adapter.py`), Streamlit app (`streamlit_app.py`), and test suites (`drift_tests/`, `ais/tests/`, `tests/`).

---

## 4. Member 1 Status: Satellite Oil Spill Detection ML

- **Classification**: **IMPLEMENTED BUT NOT TESTED WITH REAL WEIGHTS** (`50%`)
- **Location**: `ml/` (and `streamlit_app.py`, `colab_sih26143_pipeline.py`)
- **Implementation**:
  - `ml/training/unet.py` — PyTorch U-Net architecture (1-channel VV SAR input).
  - `ml/training/train.py` & `dataset.py` — Training loop & dataset loader.
  - `ml/training/inference.py` — `OilSpillPredictor` with `predict()` and `get_spill_location()`.
- **Model Checkpoint**: **MISSING**. `ml/training/checkpoints/best_unet.pth` is **NOT COMMITTED** to Git.
- **Dataset**: `ml/dataset/prepare_prototype_dataset.py` script exists, raw image files uncommitted.
- **Input**: 1-channel Sentinel-1 SAR VV GeoTIFF (`.tif`).
- **Output**: `spill_info` dictionary (`detected`, `latitude`, `longitude`, `area_km2`, `area_percent`, `confidence`, `date`, `timestamp`, `timestamp_type="synthetic"`). Extracted via `rasterio.warp.transform`.
- **Tests**: Jupyter notebooks (`ml/notebooks/*.ipynb`), no unit tests (`pytest`).
- **Blockers**: Missing `.pth` model checkpoint file; relies on synthetic timestamp string.

---

## 5. Member 2 Status: Ocean Drift & Backtracking Model

- **Classification**: **READY FOR INTEGRATION** (`85%`)
- **Location**: `src/drift/` and `drift_adapter.py`
- **Implementation**:
  - `src/drift/physics.py` — Vector drift physics (windage + ocean current vectors).
  - `src/drift/simulator.py` — Forward/backward trajectory simulation.
  - `src/drift/backtracker.py` — Backward drift particle simulation.
  - `src/drift/origin.py` — Probable release origin estimation.
  - `src/drift/environment.py` — `SyntheticEnvironmentalProvider` (`data/environment_demo.csv`).
  - `drift_adapter.py` — `run_drift_analysis(spill_info)` adapter bridging M1 → M2.
- **Input**: `DetectedSpillInput` (`spill_id`, `latitude`, `longitude`, `detection_timestamp`, `area_km2`, `confidence`).
- **Output**: `DriftOriginOutput` / `BacktrackingResult` (`detected_location`, `detected_time`, `estimated_origin`, `estimated_release_time`, `trajectory`, `uncertainty_radius_km`, `confidence`).
- **Environmental Data**: Uses procedural / synthetic CSV vectors (`data/environment_demo.csv`).
- **Tests**: 10 test files in `drift_tests/` (`test_backtracker.py`, `test_physics.py`, etc.).
- **Blockers**: Real MetOcean live API missing (uses synthetic environmental vectors).

---

## 6. Member 3 Status: AIS Processing & Candidate Search

- **Classification**: **READY FOR INTEGRATION WITH SCHEMA ADAPTER** (`80%`)
- **Location**: `ais/` and `ais_adapter.py`
- **Implementation**:
  - `ais/cleaner.py` & `loader.py` — AIS CSV loading and data validation.
  - `ais/filters.py` & `src/filtering.py` — Spatial distance & temporal window filtering.
  - `ais/trajectory.py` — Vessel grouping & trajectory reconstruction.
  - `ais/src/ranking.py` — Candidate vessel scoring and ranking.
  - `ais_adapter.py` — `run_ais_analysis(probable_lat, probable_lon, release_time)` adapter bridging M2 → M3.
- **AIS Dataset**: Uses **SYNTHETIC AIS CSV DATA** (`data/ais/synthetic/sih_demo_ais.csv`).
- **Input**: `probable_latitude`, `probable_longitude`, `estimated_release_time`, `search_radius_km`, `time_window_minutes`.
- **Output**: `CandidateOutput` (`spill_id`, `candidates` list with `vessel_id`, `closest_distance_km`, `closest_timestamp`, `latitude`, `longitude`, `speed_knots`, `heading_deg`).
- **Tests**: 11 test files in `ais/tests/`.
- **Blockers**: Uses synthetic AIS CSV; output field names (`vessel_id`, `closest_distance_km`) require translation adapter to match M4/M6 schemas (`mmsi`, `vessel_name`, `vessel_type`, `distance_to_source_km`).

---

## 7. Member 4 Status: Vessel Attribution Engine

- **Classification**: **IMPLEMENTED AND INTEGRATED** (`90%`)
- **Location**: `attribution/`
- **Implementation**:
  - `attribution/schemas.py`, `features.py`, `scorer.py`, `service.py`, `mock_data.py`, `api.py`.
  - 4-factor weighted scoring formula (Distance 30%, Time 25%, Dwell 25%, Speed/Heading 20%).
- **Backend Integration**: Member 6 built `RealVesselAttributionService` in `backend/services/real/vessel_attribution_service.py` wrapping Member 4's engine in-process for `POST /api/vessels/rank`.
- **Tests**: `tests/test_attribution.py` and `backend/tests/test_vessels.py` all **PASS**.

---

## 8. Member 5 Status: Frontend UI

- **Classification**: **PARTIALLY IMPLEMENTED** (`40%`)
- **Location**: `frontend/`
- **Implementation**: React + Vite dashboard (`frontend/src/App.jsx`) with Leaflet map, CartoDB dark tiles, vessel cards, environmental widgets.
- **Backend Integration**: **NOT INTEGRATED**. Map markers and vessel cards use static hardcoded React state with zero `fetch()` API calls.

---

## 9. Member 6 Status: Backend API & Integration Layer

- **Classification**: **IMPLEMENTED AND TESTED** (`95%`)
- **Location**: `backend/`
- **Implementation**: FastAPI app (`main.py`, `config.py`, `dependencies.py`, `schemas/`, `services/`, `api/`). Published contract in `docs/API_CONTRACT.md`.
- **Tests**: **12/12 unit and integration tests PASS**.

---

## 10. Dataset Status

| Dataset Name | Type | Location | Status |
| :--- | :--- | :--- | :--- |
| **Sentinel-1 SAR Imagery** | Prototype | `ml/dataset/` | Preparation script exists; raw images uncommitted. |
| **Ocean Environment** | Synthetic | `data/environment_demo.csv` | Synthetic wind/current vectors for drift simulation. |
| **AIS Vessel Records** | Synthetic | `data/ais/synthetic/sih_demo_ais.csv` | Synthetic CSV records for AIS candidate filtering. |
| **Attribution Test Data** | Synthetic | `attribution/mock_data.py` | Synthetic vessel trajectories for attribution scoring. |

---

## 11. Model & Checkpoint Status

| Checkpoint Path | Status | Verification Note |
| :--- | :--- | :--- |
| `ml/training/checkpoints/best_unet.pth` | **MISSING** | File is not committed to Git repository. `OilSpillPredictor` throws `FileNotFoundError` if executed without weights. |

---

## 12. Test Results

- **Backend Tests (`backend/tests/`)**: **12 passed / 0 failed** (`100% pass`).
- **Drift Model Tests (`drift_tests/`)**: 10 test files written on `integration-m1-m2-m3`.
- **AIS Processing Tests (`ais/tests/`)**: 11 test files written on `integration-m1-m2-m3`.
- **Attribution Tests (`tests/test_attribution.py`)**: Unit tests written and passing.

---

## 13. API Status

| Endpoint | Method | Status | Provider | Tested? |
| :--- | :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | **Active** | Backend | Yes |
| `/api/spill/detect` | `POST` | **Active** | `MockSpillService` | Yes |
| `/api/spill/backtrack` | `POST` | **Active** | `MockDriftService` | Yes |
| `/api/spill/predict` | `POST` | *Planned* | Future M2 | No |
| `/api/ais/candidates` | `POST` | **Active** | `MockAISService` | Yes |
| `/api/ais/tracks` | `POST` | *Planned* | Future M3 | No |
| `/api/vessels/rank` | `POST` | **Active** | `RealVesselAttributionService` (M4) | Yes |

---

## 14. M1 → M2 Compatibility

- **Status**: **COMPATIBLE** (via `drift_adapter.py`).
- **Details**: `drift_adapter.run_drift_analysis(spill_info)` accepts Member 1's `spill_info` dictionary (`latitude`, `longitude`, `timestamp`, `area_km2`, `confidence`) produced by `OilSpillPredictor.get_spill_location()` and converts it into `DetectedSpillInput` for Member 2's `process_detected_spill()`.

---

## 15. M2 → M3 Compatibility

- **Status**: **COMPATIBLE** (via `ais_adapter.py`).
- **Details**: `ais_adapter.run_ais_analysis(...)` accepts `probable_latitude`, `probable_longitude`, and `estimated_release_time` from Member 2's `DriftOriginOutput`.

---

## 16. M3 → M4 Compatibility

- **Status**: **PARTIALLY COMPATIBLE / REQUIRES SCHEMA ADAPTER**.
- **Details**: Member 3 outputs candidate vessel records with fields `vessel_id`, `closest_distance_km`, `closest_timestamp`, `latitude`, `longitude`, `speed_knots`, `heading_deg`. Member 4 expects `mmsi`, `vessel_name`, `vessel_type`, and a `positions` time-series list (`AISPosition`). An adapter is required to map M3 output records to M4's `AISTrajectoryRecord` format.

---

## 17. M4 → M6 Compatibility

- **Status**: **COMPATIBLE**.
- **Details**: Implemented via `RealVesselAttributionService` in `backend/services/real/vessel_attribution_service.py`. Maps backend request schemas to Member 4's `VesselAttributionService.analyze_attribution` and formats response with risk scores and factor breakdowns.

---

## 18. M6 → M5 Compatibility

- **Status**: **INCOMPATIBLE**.
- **Details**: Backend API endpoints and Pydantic schemas are published in [`docs/API_CONTRACT.md`](file:///c:/Users/prapt/sih26143-oil-spill/docs/API_CONTRACT.md), but React frontend (`frontend/src/App.jsx`) contains static hardcoded arrays and zero HTTP `fetch()` API calls.

---

## 19. Full Pipeline Readiness

```
[ M1: Satellite ML ] ──(drift_adapter)──> [ M2: Ocean Drift ] ──(ais_adapter)──> [ M3: AIS Processing ] ──(Adapter Needed)──> [ M4: Attribution ] ──(RealVesselService)──> [ M6: Backend ] ──(Needs Fetch)──> [ M5: Frontend ]
```

- **Pipeline Summary**: The analytical core from M1 to M4 is linked on `integration-m1-m2-m3` via Python adapters using synthetic environmental & AIS data. The final connection to M5 Frontend remains blocked by static frontend code.

---

## 20. Integration Blockers

1. **Missing Model Checkpoint**: `ml/training/checkpoints/best_unet.pth` is missing on Git. Real ML inference will fail if invoked without weights.
2. **M3 → M4 Schema Translation Gap**: Field name and structure mismatch between M3 candidate output (`closest_distance_km`, `vessel_id`) and M4 trajectory input schema (`mmsi`, `vessel_name`, `positions`).
3. **Frontend API Integration Gap**: `frontend/src/App.jsx` relies on hardcoded static state.

---

## 21. Missing Components

- Real PyTorch `.pth` checkpoint file for Sentinel-1 UNet model.
- Real MetOcean ocean current and wind live API integration.
- Real AIS live broadcast API stream.
- Frontend API fetch client integration.

---

## 22. Components Ready for Integration

- **Member 2 Drift Engine & Member 3 AIS Search**: Ready to be connected to Member 6 backend dependency injection providers (`get_drift_service` and `get_ais_service`) via adapter wrappers.

---

## 23. Components NOT Ready for Integration

- **Member 1 ML Inference**: Blocked until `best_unet.pth` model checkpoint is committed.
- **Member 5 Frontend UI**: Blocked until `App.jsx` adds API client integration calls.

---

## 24. Recommended Integration Order

1. **Frontend Integration**: Connect Member 5 React UI to Member 6 FastAPI backend endpoints (`/api/spill/detect`, `/api/spill/backtrack`, `/api/ais/candidates`, `/api/vessels/rank`).
2. **M3 AIS Service Adapter**: Wrap Member 3's `ais_adapter.py` to feed candidate vessels into Member 6 `get_ais_service()` and Member 4 Attribution Engine.
3. **M2 Drift Service Adapter**: Wrap Member 2's `drift_adapter.py` to feed backtrack drift results into Member 6 `get_drift_service()`.
4. **M1 ML Model Checkpoint**: Commit `best_unet.pth` and wrap `ml/training/inference.py` into Member 6 `get_spill_service()`.

---

## 25. System Risks

- **Model Weight Absence**: Invoking live M1 inference without `best_unet.pth` will raise `FileNotFoundError`.
- **Synthetic Data Reliance**: M2 and M3 rely on synthetic CSV files (`data/environment_demo.csv`, `data/ais/synthetic/sih_demo_ais.csv`) for demo evaluation.
- **Frontend Hardcoding**: Demo presentation will fail to show dynamic pipeline results until React frontend is wired to backend API.

---

## 26. Overall Project Readiness Calculation

### Component Readiness Scores:
- **M1 (Satellite ML)**: `50%` (Code, geospatial centroid extraction, `drift_adapter` written; checkpoint `.pth` missing) → **`10.0%`** (20% weight)
- **M2 (Ocean Drift)**: `85%` (Complete physics, simulator, backtracker, `drift_adapter` written; synthetic env) → **`17.0%`** (20% weight)
- **M3 (AIS Processing)**: `80%` (Complete filtering, ranking, `ais_adapter` written; synthetic CSV) → **`16.0%`** (20% weight)
- **M4 (Vessel Attribution)**: `90%` (Complete scoring engine, tested, connected to backend) → **`13.5%`** (15% weight)
- **M5 (Frontend UI)**: `40%` (UI built; no API fetch calls) → **`4.0%`** (10% weight)
- **M6 (Backend API)**: `95%` (FastAPI layer, adapters, contracts, 100% tests pass) → **`14.25%`** (15% weight)

**Total Effective System Readiness**: **74.75%**  
*(Effective operational readiness when accounting for missing `.pth` checkpoint and static frontend is **~65%**).*
