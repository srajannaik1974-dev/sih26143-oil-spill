# SIH 2026 Problem Statement 26143 — Frontend API Contract

**Service Name**: Oil Spill Detection & Vessel Attribution API  
**Integration Layer**: Member 6 FastAPI Backend (`member6-backend`)  
**Target Audience**: Member 5 (Frontend Team) & Team Integration  
**Document Version**: `1.0.0`  
**Last Updated**: 2026-08-27  

---

## 1. System Architecture & Overview

The system architecture follows a decoupled, service-oriented workflow where the **Member 6 FastAPI Backend** acts as the single central integration point for all data processing and model outputs:

```
[ M1: Satellite Detection ]
            ↓
[ M2: Drift / Backtrack ]
            ↓
[ M3: AIS Processing ]
            ↓
[ M6: Backend Integration Layer ]
            ↓
[ M4: Vessel Attribution Engine ]
            ↓
[ M6: FastAPI Backend API ]  ──────(REST / JSON)──────>  [ M5: Frontend UI ]
```

### Key Integration Rules for Frontend (Member 5):
1. **Single Entrypoint**: The frontend MUST communicate ONLY with the M6 FastAPI backend (`http://localhost:8000/api`).
2. **No Direct External Calls**: The frontend must NOT make direct HTTP/gRPC requests to M1 (Satellite ML), M2 (Drift physics), M3 (AIS processing), M4 (Attribution script), or external AIS provider APIs.
3. **Contract Stability Guarantee**: The request and response JSON schemas defined in this document are **STABLE**. When M1, M2, or M3 transition from mock services to real models/data, backend adapters translate internal model structures into these exact Pydantic schemas. Frontend code will NOT require modifications.

---

## 2. Global Conventions & Standards

- **Base URL**: `http://localhost:8000/api` (or environment-configured domain)
- **Content-Type**: `application/json`
- **Timestamp Format**: ISO-8601 UTC string (e.g., `2026-08-27T12:00:00Z`)
- **Geographic Coordinates**: Decimal degrees as floating point numbers.
  - Latitude: `-90.0` to `90.0`
  - Longitude: `-180.0` to `180.0`
- **Units**:
  - Distance: kilometers (`km`)
  - Area: square kilometers (`km²` / `sq_km`)
  - Speed: knots (`knots`)
  - Heading / Angle: degrees (`0.0` to `360.0`)
- **Error Format**:
  Standard FastAPI error format:
  ```json
  {
    "detail": "Error description or Pydantic validation error list"
  }
  ```

---

## 3. Endpoints Specification

### Summary Table

| Status | Method | Endpoint Path | Purpose |
| :--- | :--- | :--- | :--- |
| **ACTIVE** | `GET` | `/api/health` | Backend status & version check |
| **ACTIVE** | `POST` | `/api/spill/detect` | Satellite oil spill detection, confidence & polygon boundary |
| **ACTIVE** | `POST` | `/api/spill/backtrack` | Hydro-dynamic drift backtrack, origin estimation & trajectory |
| **PLANNED** | `POST` | `/api/spill/predict` | Forward ocean drift simulation & predicted spill trajectory |
| **ACTIVE** | `POST` | `/api/ais/candidates` | Candidate vessel query around spill origin within spatio-temporal radius |
| **PLANNED** | `POST` | `/api/ais/tracks` | Full historical AIS trajectory time-series track for a specific vessel |
| **ACTIVE** | `POST` | `/api/vessels/rank` | Multi-factor vessel attribution ranking & explainable suspect scoring |

---

### Endpoint 1: Health Check

- **Endpoint Name**: Backend Health Check
- **HTTP Method**: `GET`
- **URL Path**: `/api/health`
- **Status**: **ACTIVE**
- **Purpose**: Verify FastAPI backend service operational status and version.

#### Request
- **Headers**: None
- **Body**: None

#### Response
- **Status Code**: `200 OK`
- **Fields**:
  - `status` (`string`, required): `"ok"`
  - `service` (`string`, required): Service name
  - `version` (`string`, required): Semantic version string

#### Example Request
```http
GET /api/health HTTP/1.1
Host: localhost:8000
```

#### Example Response
```json
{
  "status": "ok",
  "service": "SIH 2026 - Oil Spill & Vessel Attribution API",
  "version": "1.0.0"
}
```

---

### Endpoint 2: Satellite Oil Spill Detection

- **Endpoint Name**: Detect Satellite Oil Spill
- **HTTP Method**: `POST`
- **URL Path**: `/api/spill/detect`
- **Status**: **ACTIVE**
- **Purpose**: Accepts satellite imagery metadata (ID, URL, acquisition timestamp, center coordinates) and returns oil spill detection presence, confidence score, bounding polygon vertices, and estimated surface area.

#### Request JSON Schema
| Field Name | Data Type | Required | Validation Rules | Description |
| :--- | :--- | :--- | :--- | :--- |
| `image_id` | `string` | **Yes** | Non-empty | Unique identifier for satellite image |
| `image_url` | `string` | No | Valid URI format | Optional direct URL to satellite image asset |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Satellite image acquisition timestamp (UTC) |
| `latitude` | `float` | **Yes** | `-90.0 <= lat <= 90.0` | Target center latitude |
| `longitude` | `float` | **Yes** | `-180.0 <= lon <= 180.0` | Target center longitude |

#### Response JSON Schema (`200 OK`)
| Field Name | Data Type | Required | Validation Rules | Description |
| :--- | :--- | :--- | :--- | :--- |
| `image_id` | `string` | **Yes** | — | Echoes input image ID |
| `spill_detected` | `boolean` | **Yes** | — | `true` if oil spill detected in imagery |
| `confidence` | `float` | **Yes** | `0.0 <= conf <= 1.0` | Detection confidence score (0.0 to 1.0) |
| `spill_polygon` | `array[GeoCoordinate]` | **Yes** | Ordered vertices | Polygon bounding vertices enclosing spill mask |
| `spill_polygon[].latitude` | `float` | **Yes** | `-90.0 <= lat <= 90.0` | Vertex latitude |
| `spill_polygon[].longitude` | `float` | **Yes** | `-180.0 <= lon <= 180.0` | Vertex longitude |
| `estimated_area_sq_km` | `float` | No | `>= 0.0` | Surface area of spill in square kilometers |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Detection processing completion timestamp |
| `disclaimer` | `string` | **Yes** | — | Operational/mock disclaimer notice |

#### Possible HTTP Status Codes
- `200 OK`: Successful detection analysis.
- `422 Unprocessable Entity`: Request body validation error (e.g. latitude out of range).
- `500 Internal Server Error`: Model inference or backend processing failure.

#### Example Request
```json
{
  "image_id": "SENTINEL2-20260827-IND-01",
  "image_url": "https://imagery.provider.org/sentinel/20260827_ind_01.tif",
  "timestamp": "2026-08-27T10:30:00Z",
  "latitude": 19.4167,
  "longitude": 71.3333
}
```

#### Example Response
```json
{
  "image_id": "SENTINEL2-20260827-IND-01",
  "spill_detected": true,
  "confidence": 0.91,
  "spill_polygon": [
    { "latitude": 19.4267, "longitude": 71.3233 },
    { "latitude": 19.4317, "longitude": 71.3433 },
    { "latitude": 19.4117, "longitude": 71.3533 },
    { "latitude": 19.4017, "longitude": 71.3283 },
    { "latitude": 19.4267, "longitude": 71.3233 }
  ],
  "estimated_area_sq_km": 4.85,
  "timestamp": "2026-08-27T10:30:05.123456Z",
  "disclaimer": "MOCK DETECTION RESULT: Confidence and polygon boundaries are simulated and must not be treated as real satellite ML inference."
}
```

---

### Endpoint 3: Spill Backtrack Trajectory & Origin

- **Endpoint Name**: Backtrack Ocean Drift
- **HTTP Method**: `POST`
- **URL Path**: `/api/spill/backtrack`
- **Status**: **ACTIVE**
- **Purpose**: Accepts detected oil spill location and timestamp to calculate backward ocean drift trajectory points and estimate probable discharge origin area.

#### Request JSON Schema
| Field Name | Data Type | Required | Validation Rules | Description |
| :--- | :--- | :--- | :--- | :--- |
| `spill_location` | `object` | **Yes** | GeoCoordinate object | Detected spill coordinates |
| `spill_location.latitude` | `float` | **Yes** | `-90.0 <= lat <= 90.0` | Detected latitude |
| `spill_location.longitude` | `float` | **Yes** | `-180.0 <= lon <= 180.0` | Detected longitude |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Spill detection timestamp (UTC) |
| `drift_hours` | `float` | No | `0.0 < hours <= 168.0` | Hours to backtrack drift (default: 24.0) |
| `wind_vector_deg` | `float` | No | `0.0 <= deg < 360.0` | Wind direction angle in degrees |
| `current_vector_deg` | `float` | No | `0.0 <= deg < 360.0` | Ocean current direction angle in degrees |

#### Response JSON Schema (`200 OK`)
| Field Name | Data Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `spill_location` | `object` | **Yes** | Echoes input detection coordinate |
| `detection_timestamp` | `string` | **Yes** | Echoes input detection timestamp |
| `estimated_source_area` | `object` | **Yes** | Estimated release origin zone details |
| `estimated_source_area.center` | `object` | **Yes** | Estimated origin center coordinate (`latitude`, `longitude`) |
| `estimated_source_area.radius_km` | `float` | **Yes** | Uncertainty radius of origin in km |
| `estimated_source_area.boundary_polygon` | `array[GeoCoordinate]` | **Yes** | Polygon enclosing origin region |
| `trajectory` | `array[TrajectoryPoint]` | **Yes** | Step-by-step backtrack positions ordered backwards in time |
| `trajectory[].timestamp` | `string` | **Yes** | Timestamp at backtrack step |
| `trajectory[].latitude` | `float` | **Yes** | Backtrack position latitude |
| `trajectory[].longitude` | `float` | **Yes** | Backtrack position longitude |
| `trajectory[].uncertainty_radius_km` | `float` | **Yes** | Positional uncertainty radius in km |
| `timestamp` | `string` | **Yes** | Calculation execution timestamp |
| `disclaimer` | `string` | **Yes** | Drift model disclaimer notice |

#### Example Request
```json
{
  "spill_location": {
    "latitude": 19.4167,
    "longitude": 71.3333
  },
  "timestamp": "2026-08-27T12:00:00Z",
  "drift_hours": 24.0
}
```

#### Example Response
```json
{
  "spill_location": {
    "latitude": 19.4167,
    "longitude": 71.3333
  },
  "detection_timestamp": "2026-08-27T12:00:00Z",
  "estimated_source_area": {
    "center": {
      "latitude": 19.3927,
      "longitude": 71.3477
    },
    "radius_km": 5.5,
    "boundary_polygon": [
      { "latitude": 19.4127, "longitude": 71.3477 },
      { "latitude": 19.3927, "longitude": 71.3677 },
      { "latitude": 19.3727, "longitude": 71.3477 },
      { "latitude": 19.3927, "longitude": 71.3277 },
      { "latitude": 19.4127, "longitude": 71.3477 }
    ]
  },
  "trajectory": [
    {
      "timestamp": "2026-08-27T12:00:00Z",
      "latitude": 19.4167,
      "longitude": 71.3333,
      "uncertainty_radius_km": 1.5
    },
    {
      "timestamp": "2026-08-27T07:12:00Z",
      "latitude": 19.4047,
      "longitude": 71.3405,
      "uncertainty_radius_km": 2.3
    },
    {
      "timestamp": "2026-08-27T00:00:00Z",
      "latitude": 19.3927,
      "longitude": 71.3477,
      "uncertainty_radius_km": 5.5
    }
  ],
  "timestamp": "2026-08-27T12:00:02.100000Z",
  "disclaimer": "MOCK DRIFT RESULT: Backtrack trajectory and source area are generated using simple geometric approximation and must not be treated as hydro-dynamic model output."
}
```

---

### Endpoint 4: Forward Predicted Drift Trajectory *(PLANNED / FUTURE)*

- **Endpoint Name**: Predict Forward Ocean Drift
- **HTTP Method**: `POST`
- **URL Path**: `/api/spill/predict`
- **Status**: **PLANNED / FUTURE ENDPOINT**
- **Purpose**: Computes forward ocean drift projection to predict future oil slick spread and coastal impact trajectories over the next 12 to 72 hours.

#### Request JSON Schema
```json
{
  "spill_location": {
    "latitude": 19.4167,
    "longitude": 71.3333
  },
  "current_timestamp": "2026-08-27T12:00:00Z",
  "forecast_hours": 48.0
}
```

#### Response JSON Schema (`200 OK`)
```json
{
  "spill_location": {
    "latitude": 19.4167,
    "longitude": 71.3333
  },
  "forecast_hours": 48.0,
  "predicted_trajectory": [
    {
      "timestamp": "2026-08-27T12:00:00Z",
      "latitude": 19.4167,
      "longitude": 71.3333,
      "uncertainty_radius_km": 1.5
    },
    {
      "timestamp": "2026-08-28T00:00:00Z",
      "latitude": 19.4350,
      "longitude": 71.3210,
      "uncertainty_radius_km": 3.8
    },
    {
      "timestamp": "2026-08-29T12:00:00Z",
      "latitude": 19.4600,
      "longitude": 71.3050,
      "uncertainty_radius_km": 7.2
    }
  ],
  "coastal_impact_warning": false,
  "timestamp": "2026-08-27T12:00:00Z"
}
```

---

### Endpoint 5: AIS Candidate Vessels Query

- **Endpoint Name**: Query AIS Candidate Vessels
- **HTTP Method**: `POST`
- **URL Path**: `/api/ais/candidates`
- **Status**: **ACTIVE**
- **Purpose**: Queries candidate vessels in the vicinity of an estimated spill origin location and time window from historical/real-time AIS stream processing.

#### Request JSON Schema
| Field Name | Data Type | Required | Validation Rules | Description |
| :--- | :--- | :--- | :--- | :--- |
| `source_latitude` | `float` | **Yes** | `-90.0 <= lat <= 90.0` | Estimated spill source latitude |
| `source_longitude` | `float` | **Yes** | `-180.0 <= lon <= 180.0` | Estimated spill source longitude |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Estimated spill discharge time |
| `search_radius_km` | `float` | No | `0.0 < radius <= 500.0` | Search radius in km (default: 50.0) |
| `time_window_hours` | `float` | No | `0.0 < hours <= 72.0` | Temporal window (+/- hours, default: 12.0) |

#### Response JSON Schema (`200 OK`)
| Field Name | Data Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `candidates` | `array[VesselAISData]` | **Yes** | List of candidate vessel positions |
| `candidates[].mmsi` | `string` | **Yes** | 9-digit MMSI identifier |
| `candidates[].vessel_name` | `string` | **Yes** | Name of vessel |
| `candidates[].vessel_type` | `string` | **Yes** | Classification (Tanker, Cargo, Tug, etc.) |
| `candidates[].callsign` | `string` | No | Radio callsign |
| `candidates[].flag` | `string` | No | Flag country |
| `candidates[].latitude` | `float` | **Yes** | Vessel latitude |
| `candidates[].longitude` | `float` | **Yes** | Vessel longitude |
| `candidates[].timestamp` | `string` | **Yes** | AIS broadcast timestamp (UTC) |
| `candidates[].speed_knots` | `float` | **Yes** | Speed Over Ground (SOG) in knots |
| `candidates[].heading_degrees` | `float` | **Yes** | Course Over Ground (COG) / Heading in deg |
| `candidates[].distance_to_source_km` | `float` | **Yes** | Distance from origin in km |
| `total_count` | `integer` | **Yes** | Total candidates matching query |
| `search_radius_km` | `float` | **Yes** | Query radius used |
| `timestamp` | `string` | **Yes** | Query execution timestamp |
| `disclaimer` | `string` | **Yes** | AIS data notice |

#### Example Request
```json
{
  "source_latitude": 19.4167,
  "source_longitude": 71.3333,
  "timestamp": "2026-08-27T12:00:00Z",
  "search_radius_km": 50.0,
  "time_window_hours": 12.0
}
```

#### Example Response
```json
{
  "candidates": [
    {
      "mmsi": "419000101",
      "vessel_name": "Ocean Titan",
      "vessel_type": "Crude Oil Tanker",
      "callsign": "VRHK8",
      "flag": "Panama",
      "latitude": 19.418,
      "longitude": 71.335,
      "timestamp": "2026-08-27T11:45:00Z",
      "speed_knots": 4.5,
      "heading_degrees": 135.0,
      "distance_to_source_km": 1.2
    },
    {
      "mmsi": "419000102",
      "vessel_name": "Pacific Voyager",
      "vessel_type": "Container Ship",
      "callsign": "2GBC4",
      "flag": "Liberia",
      "latitude": 19.486,
      "longitude": 71.383,
      "timestamp": "2026-08-27T10:00:00Z",
      "speed_knots": 13.2,
      "heading_degrees": 210.0,
      "distance_to_source_km": 8.0
    }
  ],
  "total_count": 2,
  "search_radius_km": 50.0,
  "timestamp": "2026-08-27T12:00:01.000000Z",
  "disclaimer": "MOCK AIS RESULT: Vessel positions and metadata are simulated candidate samples for pipeline integration testing."
}
```

---

### Endpoint 6: Vessel Historical Trajectory Track *(PLANNED / FUTURE)*

- **Endpoint Name**: Get Vessel Historical Track
- **HTTP Method**: `POST`
- **URL Path**: `/api/ais/tracks`
- **Status**: **PLANNED / FUTURE ENDPOINT**
- **Purpose**: Retrieves time-series AIS position history (vessel track) for a specific vessel MMSI over a specified time window.

#### Request JSON Schema
```json
{
  "mmsi": "419000101",
  "start_time": "2026-08-27T00:00:00Z",
  "end_time": "2026-08-27T18:00:00Z"
}
```

#### Response JSON Schema (`200 OK`)
```json
{
  "mmsi": "419000101",
  "vessel_name": "Ocean Titan",
  "vessel_type": "Crude Oil Tanker",
  "flag": "Panama",
  "track_points": [
    {
      "timestamp": "2026-08-27T10:45:00Z",
      "latitude": 19.4050,
      "longitude": 71.3200,
      "speed_knots": 11.0,
      "heading_degrees": 135.0
    },
    {
      "timestamp": "2026-08-27T11:45:00Z",
      "latitude": 19.4180,
      "longitude": 71.3350,
      "speed_knots": 4.5,
      "heading_degrees": 135.0
    },
    {
      "timestamp": "2026-08-27T12:45:00Z",
      "latitude": 19.4280,
      "longitude": 71.3450,
      "speed_knots": 10.5,
      "heading_degrees": 135.0
    }
  ],
  "total_points": 3
}
```

---

### Endpoint 7: Vessel Attribution & Suspect Ranking

- **Endpoint Name**: Rank Candidate Suspect Vessels
- **HTTP Method**: `POST`
- **URL Path**: `/api/vessels/rank`
- **Status**: **ACTIVE**
- **Purpose**: Evaluates candidate vessels against estimated oil spill origin location/time, calculates spatial-temporal feature correlations via Member 4's Vessel Attribution Engine, and returns ranked suspect vessels with explainable score breakdowns.

#### Request JSON Schema
| Field Name | Data Type | Required | Validation Rules | Description |
| :--- | :--- | :--- | :--- | :--- |
| `spill_source` | `object` | **Yes** | GeoCoordinate | Estimated spill origin coordinate |
| `spill_source.latitude` | `float` | **Yes** | `-90.0 <= lat <= 90.0` | Origin latitude |
| `spill_source.longitude` | `float` | **Yes** | `-180.0 <= lon <= 180.0` | Origin longitude |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Detection / release timestamp |
| `estimated_release_time` | `string` | No | ISO-8601 Datetime | Optional release time override |
| `search_radius_km` | `float` | No | `> 0.0` | Spatial filter radius (default: 50.0) |
| `time_window_hours` | `float` | No | `> 0.0` | Temporal filter window (default: 24.0) |
| `candidate_vessels` | `array[VesselAISData]` | **Yes** | Array of vessels | AIS candidates retrieved from `/api/ais/candidates` |

#### Response JSON Schema (`200 OK`)
| Field Name | Data Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `ranked_vessels` | `array[RankedVessel]` | **Yes** | List of suspect vessels sorted by rank |
| `ranked_vessels[].vessel` | `object` | **Yes** | Vessel AIS metadata (`mmsi`, `vessel_name`, etc.) |
| `ranked_vessels[].rank` | `integer` | **Yes** | 1-indexed suspect rank (`1` = highest correlation) |
| `ranked_vessels[].risk_score` | `float` | **Yes** | Normalized risk score (`0.0` to `1.0`) |
| `ranked_vessels[].final_score` | `float` | No | Raw weighted score (`0.0` to `100.0`) |
| `ranked_vessels[].classification` | `string` | No | Classification label (e.g., `"Highest-correlated candidate vessel"`) |
| `ranked_vessels[].explanation` | `string` | No | Explainable textual summary of correlation features |
| `ranked_vessels[].attribution_factors` | `array[AttributionFactor]` | **Yes** | Sub-score feature breakdown |
| `attribution_factors[].factor_name` | `string` | **Yes** | Feature name (Distance, Time, Trajectory, Speed/Heading) |
| `attribution_factors[].score` | `float` | **Yes** | Sub-score contribution (`0.0` to `1.0`) |
| `attribution_factors[].description` | `string` | **Yes** | Human-readable explanation of sub-score |
| `total_ranked` | `integer` | **Yes** | Total vessels ranked |
| `ranked_at` | `string` | **Yes** | Calculation execution timestamp |
| `disclaimer` | `string` | **Yes** | Attribution legal/scientific disclaimer |

#### Example Request
```json
{
  "spill_source": {
    "latitude": 19.4167,
    "longitude": 71.3333
  },
  "timestamp": "2026-08-27T12:00:00Z",
  "search_radius_km": 50.0,
  "time_window_hours": 24.0,
  "candidate_vessels": [
    {
      "mmsi": "419000101",
      "vessel_name": "Ocean Titan",
      "vessel_type": "Crude Oil Tanker",
      "latitude": 19.418,
      "longitude": 71.335,
      "timestamp": "2026-08-27T11:45:00Z",
      "speed_knots": 4.5,
      "heading_degrees": 135.0,
      "distance_to_source_km": 1.2
    },
    {
      "mmsi": "419000102",
      "vessel_name": "Pacific Voyager",
      "vessel_type": "Container Ship",
      "latitude": 19.486,
      "longitude": 71.383,
      "timestamp": "2026-08-27T10:00:00Z",
      "speed_knots": 13.2,
      "heading_degrees": 210.0,
      "distance_to_source_km": 8.0
    }
  ]
}
```

#### Example Response
```json
{
  "ranked_vessels": [
    {
      "vessel": {
        "mmsi": "419000101",
        "vessel_name": "Ocean Titan",
        "vessel_type": "Crude Oil Tanker",
        "callsign": "VRHK8",
        "flag": "Panama",
        "latitude": 19.418,
        "longitude": 71.335,
        "timestamp": "2026-08-27T11:45:00Z",
        "speed_knots": 4.5,
        "heading_degrees": 135.0,
        "distance_to_source_km": 0.23
      },
      "rank": 1,
      "risk_score": 0.871,
      "final_score": 87.1,
      "classification": "Highest-correlated candidate vessel",
      "explanation": "Classified as 'Highest-correlated candidate vessel' (Rank 1, Score 87.10/100). Passed within 0.23 km of spill origin with closest approach 15.0 minutes from estimated release time. Speed at CPA was 4.5 knots.",
      "attribution_factors": [
        {
          "factor_name": "Distance Proximity Score",
          "score": 0.9773,
          "description": "Distance weight (30%): Closest approach is 0.23 km from spill origin."
        },
        {
          "factor_name": "Time Proximity Score",
          "score": 0.92,
          "description": "Time weight (25%): Time difference is 15.0 minutes from release time."
        },
        {
          "factor_name": "Trajectory Dwell Score",
          "score": 0.7617,
          "description": "Trajectory weight (25%): Geometry and dwell persistence score is 76.2/100."
        },
        {
          "factor_name": "Speed & Heading Anomaly Score",
          "score": 0.7867,
          "description": "Behavior weight (20%): Speed changes and heading alignment score is 78.7/100."
        }
      ]
    }
  ],
  "total_ranked": 1,
  "ranked_at": "2026-08-27T12:00:03.456789Z",
  "disclaimer": "ATTRIBUTION RESULT: Vessel rankings and correlation scores calculated using Member 4 Vessel Attribution Engine."
}
```

---

## 4. Future Compatibility Guarantee for M1, M2 & M3

The backend uses **Abstract Service Base Classes** and **FastAPI Dependency Providers** (`backend/dependencies.py`). 

```python
# backend/dependencies.py

def get_spill_service() -> BaseSpillService:
    return _spill_service_instance  # MockSpillService -> RealSatelliteMLService

def get_drift_service() -> BaseDriftService:
    return _drift_service_instance  # MockDriftService -> RealDriftService

def get_ais_service() -> BaseAISService:
    return _ais_service_instance    # MockAISService -> RealAISService

def get_vessel_service() -> BaseVesselService:
    return _vessel_service_instance # RealVesselAttributionService (Member 4)
```

When Member 1 (Satellite ML), Member 2 (Drift Model), or Member 3 (AIS Processing) deliver their real implementations:
1. They create adapter classes inheriting from `BaseSpillService`, `BaseDriftService`, or `BaseAISService`.
2. The adapter translates internal model tensors/arrays into the Pydantic schemas documented above.
3. The dependency factory in `backend/dependencies.py` is updated to instantiate the real service.
4. **Result**: Zero changes required to API endpoints (`backend/api/*.py`) or frontend components (Member 5).
