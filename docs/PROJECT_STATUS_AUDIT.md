# SIH 2026 Problem Statement 26143 — Complete Project Status Audit

**Project Title**: Leveraging Satellite Imagery to Determine Oil Spills at Sea along with AIS Data Correlations to Identify the Vessel Responsible  
**Audit Timestamp**: 2026-08-29 (System Audit)  
**Audit Author**: Member 6 (Backend / Integration Layer)  
**Audit Scope**: Full Git repository analysis across all local and remote branches (`main`, `member1-satellite`, `member2-drift`, `member4-attribution`, `member5-frontend`, `member6-backend`).

---

## 1. Executive Summary

This audit evaluates the actual implementation status of all 6 team member modules based on empirical code, commit logs, file structures, model weights, and test results in the repository.

- **Overall System Readiness**: **39.75% (~40%)**
- **Completed & Integrated Modules**: Member 6 (Backend Foundation & API Contracts) and Member 4 (Vessel Attribution Engine connected to backend via synthetic data).
- **Partially Implemented Modules**: Member 1 (Satellite ML pipeline written, but trained checkpoint `.pth` file is missing, GIS coordinate projection is unhandled, and not connected to backend) and Member 5 (Frontend React UI built, but uses hardcoded static data with zero backend API calls).
- **Not Pushed / Not Started Modules**: Member 2 (Ocean Drift Backtracking model has no code on git) and Member 3 (AIS Processing pipeline has no code on git).

---

## 2. Git Branch Status Matrix

| Branch Name | Remote Target | Last Commit Hash | Last Commit Message | Active Content |
| :--- | :--- | :--- | :--- | :--- |
| `main` | `origin/main` | `5f036d0` | Initial commit | Root README & .gitignore |
| `member1-satellite` | `origin/member1-satellite` | `a536388` | Fix PyTorch training compatibility | `ml/` PyTorch U-Net training pipeline |
| `member2-drift` | `origin/member2-drift` | `5f036d0` | Initial commit | **No M2 code pushed** |
| `member4-attribution` | `origin/member4-attribution` | `31fe71a` | Implement vessel attribution module and demo | `attribution/` scoring package & tests |
| `member5-frontend` | `origin/member5-frontend` | `7362d27` | Redesign dashboard with dark theme and header | `frontend/` React + Vite dashboard |
| `member6-backend` *(current)* | `origin/member6-backend` | `5f036d0` (+ local) | Backend & integration setup | `backend/`, `attribution/`, `docs/` |

---

## 3. Member 1 Status: Satellite ML (Oil Spill Detection)

- **Classification**: **PARTIALLY IMPLEMENTED**
- **Repository Path**: `ml/` (Branch: `origin/member1-satellite`)

### Detailed Audit Findings:
1. **Model Architecture**: PyTorch U-Net implemented in `ml/training/unet.py` accepting 2-channel input (Sentinel-1 SAR VV + VH polarization) and outputting 1-channel probability logits.
2. **Training & Dataset Pipeline**: Training script `ml/training/train.py` and dataset loader `ml/training/dataset.py` exist with BCE + Dice loss.
3. **Model Checkpoint Status**: **MISSING**. File `ml/training/checkpoints/best_unet.pth` is **NOT COMMITTED** to git. Running `ml/training/inference.py` throws `FileNotFoundError`.
4. **Input / Output Data Contracts**:
   - Accepts local 2-channel GeoTIFF files (`.tif`).
   - Outputs 2D NumPy array `binary_mask` (uint8) and `prob_map` (float32).
   - **Gaps**: Does NOT extract acquisition timestamp, nor convert pixel masks `(H, W)` into geographic latitude/longitude coordinates or GeoJSON polygons required by the backend API schema.
5. **Backend Connection**: **NOT INTEGRATED**.
6. **Tests**: Jupyter notebooks provided (`ml/notebooks/*.ipynb`), no unit tests (`pytest`).

---

## 4. Member 2 Status: Ocean Drift & Backtracking Model

- **Classification**: **NOT STARTED**
- **Repository Path**: None (`origin/member2-drift` contains no commits).

### Detailed Audit Findings:
1. **Codebase Status**: No Python files, physics scripts, wind/current vector models, or OpenDrift modules exist on the `member2-drift` branch.
2. **Backend Availability**: The backend (`backend/services/mock/drift.py`) provides a clean mock implementation (`MockDriftService`) generating geometric vector trajectories and origin circles for testing.
3. **Backend Connection**: Mock endpoint `POST /api/spill/backtrack` is active and tested. Real model is **NOT INTEGRATED**.

---

## 5. Member 3 Status: AIS Data Processing & Stream

- **Classification**: **NOT STARTED**
- **Repository Path**: None (No `member3` branch exists on origin).

### Detailed Audit Findings:
1. **Codebase Status**: No AIS parsing scripts, NMEA decoders, database connectors, or live stream handlers exist in the repository.
2. **Backend Availability**: The backend (`backend/services/mock/ais.py`) provides `MockAISService` generating realistic vessel positions.
3. **Backend Connection**: Mock endpoint `POST /api/ais/candidates` is active and tested. Real stream is **NOT INTEGRATED**.

---

## 6. Member 4 Status: Vessel Attribution Engine

- **Classification**: **IMPLEMENTED AND TESTED**
- **Repository Path**: `attribution/` (Branch: `origin/member4-attribution`)

### Detailed Audit Findings:
1. **Codebase Status**: Full Python package containing spatial-temporal feature extraction (`features.py`), explainable scoring (`scorer.py`), Pydantic schemas (`schemas.py`), service coordinator (`service.py`), and test data generators (`mock_data.py`).
2. **Algorithm**: 4-factor weighted scoring formula:
   - Distance Proximity (30% weight)
   - Temporal Proximity (25% weight)
   - Trajectory Dwell Persistence (25% weight)
   - Speed & Heading Anomaly (20% weight)
3. **Data Used**: Uses **SYNTHETIC AIS DATA** (`attribution/mock_data.py`).
4. **Backend Connection**: **INTEGRATED**. Member 6 built `RealVesselAttributionService` in `backend/services/real/vessel_attribution_service.py` wrapping Member 4's engine in-process for `POST /api/vessels/rank`.
5. **Tests**: Unit tests in `tests/test_attribution.py` and integration tests in `backend/tests/test_vessels.py` all **PASS**.

---

## 7. Member 5 Status: Frontend UI

- **Classification**: **PARTIALLY IMPLEMENTED**
- **Repository Path**: `frontend/` (Branch: `origin/member5-frontend`)

### Detailed Audit Findings:
1. **UI Components**: Modern React + Vite dashboard (`frontend/src/App.jsx`) with Tailwind CSS, Leaflet map (`react-leaflet`), CartoDB dark tiles, vessel cards, and environmental log widgets.
2. **Backend Connection**: **NOT INTEGRATED**. All map markers (`spillIncidents`) and vessel cards ("MV Orion Trader", "Pacific Pioneer") are static hardcoded React state arrays. Contains zero `fetch()` or `axios` API calls.

---

## 8. Member 6 Status: Backend API & Integration Layer

- **Classification**: **IMPLEMENTED AND TESTED**
- **Repository Path**: `backend/` (Branch: `member6-backend`)

### Detailed Audit Findings:
1. **Architecture**: Clean FastAPI service-layer design using Pydantic v2, dependency injection (`backend/dependencies.py`), and abstract service contracts (`BaseSpillService`, `BaseDriftService`, `BaseAISService`, `BaseVesselService`).
2. **Endpoints**:
   - `GET /api/health` — Active
   - `POST /api/spill/detect` — Active (MockSpillService)
   - `POST /api/spill/backtrack` — Active (MockDriftService)
   - `POST /api/ais/candidates` — Active (MockAISService)
   - `POST /api/vessels/rank` — Active (RealVesselAttributionService wrapping M4)
3. **API Contracts**: Published in [`docs/API_CONTRACT.md`](file:///c:/Users/prapt/sih26143-oil-spill/docs/API_CONTRACT.md).
4. **Tests**: **12/12 pytest unit & integration tests PASS**.

---

## 9. Dataset Status

| Dataset | Type | Location | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Sentinel-1 SAR Imagery** | Real / Prototype | Uncommitted / `ml/dataset/` | Preparation script `prepare_prototype_dataset.py` exists, but raw images are not committed. |
| **AIS Vessel Tracks** | Synthetic | `attribution/mock_data.py` | Synthetic trajectories generated for 4 mock test scenarios. |
| **Spill Incident Samples**| Mock | `backend/services/mock/` | Geometrically generated bounding polygons and backtrack points. |

---

## 10. Model Checkpoint Status

| Model File | Intended Path | Status | Cause |
| :--- | :--- | :--- | :--- |
| `best_unet.pth` | `ml/training/checkpoints/best_unet.pth` | **MISSING** | Not committed to `origin/member1-satellite` branch. Must be trained locally or uploaded. |

---

## 11. Test Execution Audit

- **Framework**: `pytest 9.1.1`
- **Total Test Cases**: 12
- **Passed**: **12**
- **Failed**: 0
- **Skipped**: 0

### Test Summary Breakdown:
```text
backend/tests/test_ais.py::test_ais_candidates_success PASSED
backend/tests/test_ais.py::test_ais_candidates_invalid_radius PASSED
backend/tests/test_health.py::test_health_check PASSED
backend/tests/test_spill.py::test_spill_detect_success PASSED
backend/tests/test_spill.py::test_spill_detect_invalid_latitude PASSED
backend/tests/test_spill.py::test_spill_backtrack_success PASSED
backend/tests/test_spill.py::test_spill_backtrack_invalid_drift_hours PASSED
backend/tests/test_vessels.py::test_vessel_rank_success_real_attribution PASSED
backend/tests/test_vessels.py::test_vessel_rank_with_trajectories PASSED
backend/tests/test_vessels.py::test_vessel_rank_missing_candidates PASSED
backend/tests/test_vessels.py::test_vessel_rank_invalid_coordinates PASSED
backend/tests/test_vessels.py::test_vessel_rank_mock_fallback PASSED
```

---

## 12. API Status & Coverage Matrix

| Endpoint | Method | Status | Provider | Tested? | Frontend Connected? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | **Active** | Backend | Yes | No |
| `/api/spill/detect` | `POST` | **Active** | `MockSpillService` | Yes | No |
| `/api/spill/backtrack` | `POST` | **Active** | `MockDriftService` | Yes | No |
| `/api/spill/predict` | `POST` | *Planned* | Future M2 | No | No |
| `/api/ais/candidates` | `POST` | **Active** | `MockAISService` | Yes | No |
| `/api/ais/tracks` | `POST` | *Planned* | Future M3 | No | No |
| `/api/vessels/rank` | `POST` | **Active** | `RealVesselAttributionService` (M4) | Yes | No |

---

## 13. Module Compatibility Matrix

| Source → Destination | Status | Compatibility Details & Gaps |
| :--- | :--- | :--- |
| **M1 (Satellite ML) → M2 (Drift)** | **INCOMPATIBLE** | M1 outputs raw pixel array `(H, W)`. M2 requires geographic lat/lon origin coordinates. Coordinate conversion layer missing. |
| **M2 (Drift) → M3 (AIS)** | **NOT YET AVAILABLE** | M2 has no code pushed. M3 has no code pushed. |
| **M3 (AIS) → M4 (Attribution)** | **COMPATIBLE (Mock)** | M4's schema accepts AIS candidate records. Currently fed via `attribution/mock_data.py`. |
| **M2 + M3 → M4 (Attribution)** | **PARTIALLY COMPATIBLE**| M4 expects `SpillOriginInput` and `AISTrajectoryRecord` list, which match Member 6 backend schemas. |
| **M4 → Backend API (M6)** | **COMPATIBLE** | `RealVesselAttributionService` adapter seamlessly converts data contracts. |
| **Backend API (M6) → Frontend (M5)**| **INCOMPATIBLE** | API contract is published (`docs/API_CONTRACT.md`), but React frontend code contains no HTTP client integration. |

---

## 14. Integration Blockers & Missing Pieces

1. **M1 Blockers**:
   - Missing `best_unet.pth` model checkpoint file.
   - Missing GIS coordinate projection module (converting pixel masks to WGS84 latitude/longitude bounding polygons).
2. **M2 Blockers**: No drift simulation code exists on Git.
3. **M3 Blockers**: No AIS processing / database query code exists on Git.
4. **M5 Blockers**: Frontend React components use hardcoded state arrays and have not implemented API fetch calls to `/api/...`.

---

## 15. Recommended Integration Sequence & Readiness

### Readiness Breakdown Calculation
- M1 (Satellite ML): `40%` (Code written, no weights, no GIS projection) → `8.0%`
- M2 (Ocean Drift): `0%` (No code) → `0.0%`
- M3 (AIS Stream): `0%` (No code) → `0.0%`
- M4 (Vessel Attribution): `90%` (Complete engine, tested, connected via mock AIS) → `13.5%`
- M5 (Frontend UI): `40%` (UI built, no API connection) → `4.0%`
- M6 (Backend API): `95%` (FastAPI layer, M4 adapter, contracts, 100% tests pass) → `14.25%`
- **Total System Readiness**: **39.75% (~40%)**

### What Can Be Integrated Immediately:
1. **Frontend (M5) to Backend (M6)**: Connect React Leaflet map and vessel cards in `App.jsx` to fetch live data from active endpoints (`POST /api/spill/detect`, `POST /api/spill/backtrack`, `POST /api/ais/candidates`, `POST /api/vessels/rank`).

### What Must Wait:
1. **Real M1 Integration**: Train U-Net model, commit `best_unet.pth`, add GeoTIFF coordinate transformation, and wrap with `RealSatelliteMLService`.
2. **Real M2 Integration**: Develop M2 drift model code and wrap with `RealDriftService`.
3. **Real M3 Integration**: Develop M3 AIS processing pipeline and wrap with `RealAISService`.
