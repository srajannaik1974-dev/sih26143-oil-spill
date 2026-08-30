# Frontend Data Availability Matrix

Audited against the actual running backend application code on branch `integration-backend`.

| # | Data Requirement | Status | Provider / Endpoint | Implementation Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Oil spill detection** | **AVAILABLE** | `POST /api/spill/detect` | Returns `spill_detected` boolean. |
| 2 | **Spill coordinates** | **AVAILABLE** | `POST /api/spill/detect` | Returns center `latitude` & `longitude`. |
| 3 | **Spill polygon** | **AVAILABLE** | `POST /api/spill/detect` | `spill_polygon` array of `GeoCoordinate` vertices. |
| 4 | **Spill mask** | **PARTIALLY AVAILABLE** | `POST /api/spill/detect` | Mask converted to boundary polygon vertices array. |
| 5 | **Spill area** | **AVAILABLE** | `POST /api/spill/detect` | `estimated_area_sq_km` float value. |
| 6 | **Detection confidence** | **AVAILABLE** | `POST /api/spill/detect` | `confidence` score `0.0` to `1.0`. |
| 7 | **Detection timestamp** | **AVAILABLE** | `POST /api/spill/detect` | ISO-8601 UTC timestamp string. |
| 8 | **Probable origin** | **AVAILABLE** | `POST /api/spill/backtrack` | `estimated_source_area.center` coordinate. |
| 9 | **Origin uncertainty** | **AVAILABLE** | `POST /api/spill/backtrack` | `radius_km` & `boundary_polygon`. |
| 10 | **Backtracked trajectory** | **AVAILABLE** | `POST /api/spill/backtrack` | `trajectory` array of `TrajectoryPoint` objects. |
| 11 | **Predicted trajectory** | **NOT AVAILABLE** | *Planned Endpoint* | Drift forward simulation exists in M2 but no API route. |
| 12 | **Environmental info** | **PARTIALLY AVAILABLE** | `POST /api/spill/backtrack` | Wind/current vectors embedded in trajectory points. |
| 13 | **AIS candidate vessels** | **AVAILABLE** | `POST /api/ais/candidates` | List of `VesselAISData` objects. |
| 14 | **AIS vessel positions** | **AVAILABLE** | `POST /api/ais/candidates` | Position coordinates & `positions` array. |
| 15 | **Historical vessel tracks** | **PARTIALLY AVAILABLE** | `POST /api/ais/candidates` | Historical points included inside candidate positions. |
| 16 | **Vessel MMSI** | **AVAILABLE** | `POST /api/ais/candidates` | `mmsi` string identifier. |
| 17 | **Vessel name** | **AVAILABLE** | `POST /api/ais/candidates` | `vessel_name` string. |
| 18 | **Vessel type** | **AVAILABLE** | `POST /api/ais/candidates` | `vessel_type` string (e.g. "Crude Oil Tanker"). |
| 19 | **Vessel speed** | **AVAILABLE** | `POST /api/ais/candidates` | `speed_knots` float value. |
| 20 | **Vessel heading** | **AVAILABLE** | `POST /api/ais/candidates` | `heading_degrees` float value. |
| 21 | **Vessel timestamps** | **AVAILABLE** | `POST /api/ais/candidates` | `timestamp` ISO string. |
| 22 | **Distance from origin** | **AVAILABLE** | `POST /api/ais/candidates` | `distance_to_source_km` float value. |
| 23 | **Candidate ranking** | **AVAILABLE** | `POST /api/vessels/rank` | `rank` integer (1-indexed). |
| 24 | **Risk/correlation score** | **AVAILABLE** | `POST /api/vessels/rank` | `risk_score` (0-1.0) & `final_score` (0-100.0). |
| 25 | **Attribution factors** | **AVAILABLE** | `POST /api/vessels/rank` | `attribution_factors` breakdown array. |
| 26 | **Explanation/reason** | **AVAILABLE** | `POST /api/vessels/rank` | `classification` & `explanation` string summaries. |
| 27 | **Timeline information** | **PARTIALLY AVAILABLE** | All Endpoints | Datetime stamps present across all payload stages. |
