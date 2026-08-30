# Full Analytical Pipeline Integration Flow

This document details the exact end-to-end data transformations across all sub-modules:
`M1 (Satellite ML) -> M2 (Drift Model) -> M3 (AIS Engine) -> M4 (Attribution) -> M6 (Backend) -> M5 (Frontend)`.

---

## Complete Data Flow Architecture

```
[ Satellite SAR GeoTIFF / Upload ]
               │
               ▼
[ M1: ml/training/inference.py ]
   Returns: spill_info { latitude, longitude, area_km2, confidence, timestamp }
               │
               ▼  (drift_adapter.py)
[ M2: src/drift/integration.py ]
   Returns: DriftOriginOutput { estimated_origin { latitude, longitude, radius_km }, trajectory }
               │
               ▼  (ais_adapter.py)
[ M3: ais/src/filtering.py & ranking.py ]
   Returns: candidate_vessels [ { vessel_id, closest_distance_km, latitude, longitude, ... } ]
               │
               ▼  (RealAISServiceAdapter: M3 -> M4 Schema Translation)
[ M4: attribution/service.py ]
   Returns: AttributionResponse { ranked_vessels [ { rank, risk_score, attribution_factors } ] }
               │
               ▼
[ M6: backend/api/vessels.py ]
   Returns: VesselRankResponse (FastAPI JSON)
               │
               ▼
[ M5: frontend/src/App.jsx (React UI) ]
```

---

## Detailed Stage-by-Stage Field Mapping

### Stage 1: M1 Satellite Detection -> M2 Drift Model
- **Adapter**: `drift_adapter.run_drift_analysis(spill_info)`
- **M1 Output Fields**:
  - `latitude`: `float`
  - `longitude`: `float`
  - `timestamp`: `ISO string`
  - `area_km2`: `float`
  - `confidence`: `float`
- **M2 Input Field**: `DetectedSpillInput(spill_id, latitude, longitude, detection_timestamp, area_km2, confidence)`

---

### Stage 2: M2 Drift Model -> M3 AIS Processing
- **Adapter**: `ais_adapter.run_ais_analysis(...)`
- **M2 Output Fields**:
  - `estimated_origin.latitude`: `float`
  - `estimated_origin.longitude`: `float`
  - `release_start_timestamp`: `datetime`
- **M3 Input Parameters**: `probable_latitude`, `probable_longitude`, `estimated_release_time`, `search_radius_km`, `time_window_minutes`

---

### Stage 3: M3 AIS Candidate Output -> M4 Attribution Engine (Critical Adapter)
- **Adapter**: `RealAISServiceAdapter` in `backend/services/real/ais_stream_service.py`
- **M3 Candidate Record Fields**:
  - `vessel_id`: `"V001"`
  - `closest_distance_km`: `1.2`
  - `closest_timestamp`: `"2026-08-27T05:45:00Z"`
  - `latitude`: `19.418`
  - `longitude`: `71.335`
  - `speed_knots`: `4.5`
  - `heading_deg`: `135.0`
- **M4 Expected Trajectory Record Schema (`AISTrajectoryRecord`)**:
  - `mmsi`: `"419000101"` (Derived from `vessel_id`)
  - `vessel_name`: `"Vessel V001"`
  - `vessel_type`: `"Cargo/Tanker"`
  - `distance_to_source_km`: `1.2`
  - `positions`: `[ { timestamp, latitude, longitude, speed_knots, heading_deg } ]`

---

### Stage 4: M4 Attribution Engine -> M6 Backend API -> M5 Frontend UI
- **M4 Output**: `AttributionResponse` (`ranked_vessels` list with raw score `0-100`, normalized score `0-1.0`, factor breakdown list).
- **M6 API Output**: `VesselRankResponse` (HTTP 200 JSON delivered over CORS to `http://localhost:5173`).
- **M5 UI Consumption**:
  - `ranked_vessels[0]` -> Top suspect highlighted in red on map with detail card.
  - `attribution_factors` -> Progress bars on right-hand suspect detail drawer.
