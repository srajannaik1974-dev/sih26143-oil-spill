# Definitive Frontend API Contract

Derived directly from FastAPI application route definitions (`backend/api/`) and Pydantic schemas (`backend/schemas/`).

---

## Global Conventions

- **Base URL**: `http://127.0.0.1:8000/api`
- **Headers**: `Content-Type: application/json`
- **Coordinates**: Latitude `[-90.0, 90.0]`, Longitude `[-180.0, 180.0]` (Decimal degrees)
- **Timestamps**: ISO-8601 UTC strings (e.g. `2026-08-27T12:00:00Z`)

---

## 1. GET /api/health

- **HTTP Method**: `GET`
- **Path**: `/api/health`
- **Purpose**: Check backend operational status.

### Response (`200 OK`)
```json
{
  "status": "ok",
  "service": "SIH 2026 - Oil Spill & Vessel Attribution API",
  "version": "1.0.0"
}
```

---

## 2. POST /api/spill/detect/upload

- **HTTP Method**: `POST`
- **Path**: `/api/spill/detect/upload`
- **Content-Type**: `multipart/form-data`
- **Form Field**: `file` (Sentinel-1 `.tif` or `.tiff` binary file)
- **Purpose**: Upload a raw 1-channel Sentinel-1 SAR GeoTIFF image file and run PyTorch U-Net segmentation model inference.

### Request Form Data
| Field Name | Type | Required | Accepted Extension | Description |
| :--- | :--- | :--- | :--- | :--- |
| `file` | `UploadFile` | **Yes** | `.tif`, `.tiff` | Raw 1-channel Sentinel-1 SAR GeoTIFF image file |

### Response Body Schema (`SpillDetectionResponse` - `200 OK`)
| Field | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `image_id` | `string` | No | Original uploaded filename |
| `spill_detected` | `boolean` | No | `true` if oil spill detected in SAR imagery |
| `confidence` | `float` | No | Mean confidence score between `0.0` and `1.0` |
| `spill_polygon` | `array[GeoCoordinate]` | No | Array of `{latitude, longitude}` polygon vertices |
| `estimated_area_sq_km` | `float` | Yes | Calculated oil spill surface area in km² |
| `timestamp` | `string` | No | Execution ISO-8601 timestamp |
| `disclaimer` | `string` | No | Model inference disclaimer string |

### Example JavaScript Upload Request
```javascript
const formData = new FormData();
formData.append("file", tiffFileObject);

const response = await fetch("http://127.0.0.1:8000/api/spill/detect/upload", {
  method: "POST",
  body: formData
});
const data = await response.json();
```

### Example Response (`200 OK`)
```json
{
  "image_id": "test_input_gdrive.tif",
  "spill_detected": true,
  "confidence": 0.7548,
  "spill_polygon": [
    { "latitude": 28.912903, "longitude": -89.028632 },
    { "latitude": 28.917903, "longitude": -89.008632 },
    { "latitude": 28.897903, "longitude": -88.998632 },
    { "latitude": 28.887903, "longitude": -89.023632 },
    { "latitude": 28.912903, "longitude": -89.028632 }
  ],
  "estimated_area_sq_km": 54.71,
  "timestamp": "2026-08-30T00:35:00Z",
  "disclaimer": "M1 ML REAL INFERENCE: Detection evaluated on uploaded Sentinel-1 GeoTIFF using PyTorch UNet model."
}
```

---

## 3. POST /api/spill/detect

- **HTTP Method**: `POST`
- **Path**: `/api/spill/detect`
- **Purpose**: Detect satellite oil spill presence, polygon boundary, and surface area.

### Request Body Schema (`SpillDetectionRequest`)
| Field | Type | Required | Range / Constraint | Description |
| :--- | :--- | :--- | :--- | :--- |
| `image_id` | `string` | **Yes** | Non-empty | Satellite image identifier |
| `image_url` | `string` | No | Valid URI string | Optional URL to image file |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Image acquisition timestamp |
| `latitude` | `float` | **Yes** | `-90.0 <= lat <= 90.0` | Target center latitude |
| `longitude` | `float` | **Yes** | `-180.0 <= lon <= 180.0` | Target center longitude |

### Response Body Schema (`SpillDetectionResponse` - `200 OK`)
| Field | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `image_id` | `string` | No | Echoes request image ID |
| `spill_detected` | `boolean` | No | `true` if spill detected |
| `confidence` | `float` | No | Score between `0.0` and `1.0` |
| `spill_polygon` | `array[GeoCoordinate]` | No | Array of `{latitude, longitude}` vertices |
| `estimated_area_sq_km` | `float` | Yes | Spill area in km² |
| `timestamp` | `string` | No | Execution timestamp |
| `disclaimer` | `string` | No | Model/mock disclaimer string |

### Example Request
```json
{
  "image_id": "SAT-IMG-2026-001",
  "image_url": "https://example.com/satellite/image.tif",
  "timestamp": "2026-08-27T12:00:00Z",
  "latitude": 19.4167,
  "longitude": 71.3333
}
```

### Example Response
```json
{
  "image_id": "SAT-IMG-2026-001",
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
  "timestamp": "2026-08-27T12:00:05Z",
  "disclaimer": "M1 ML INFERENCE: Detection evaluated using Member 1 PyTorch UNet model."
}
```

---

## 3. POST /api/spill/backtrack

- **HTTP Method**: `POST`
- **Path**: `/api/spill/backtrack`
- **Purpose**: Calculate ocean drift backtrack trajectory and release origin zone.

### Request Body Schema (`BacktrackRequest`)
| Field | Type | Required | Range / Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `spill_location` | `object` | **Yes** | `{latitude, longitude}` | Spill coordinates |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Detection timestamp |
| `drift_hours` | `float` | No | `0.0 < hours <= 168.0` (Default: `24.0`) | Hours to backtrack |
| `wind_vector_deg` | `float` | No | `0.0 <= deg < 360.0` | Optional wind angle |
| `current_vector_deg` | `float` | No | `0.0 <= deg < 360.0` | Optional ocean current angle |

### Response Body Schema (`BacktrackResponse` - `200 OK`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `spill_location` | `object` | `{latitude, longitude}` detection point |
| `detection_timestamp` | `string` | Original detection time |
| `estimated_source_area` | `object` | `{center, radius_km, boundary_polygon}` |
| `trajectory` | `array` | List of `{timestamp, latitude, longitude, uncertainty_radius_km}` |
| `timestamp` | `string` | Execution timestamp |
| `disclaimer` | `string` | Drift model disclaimer notice |

### Example Request
```json
{
  "spill_location": {
    "latitude": 19.4167,
    "longitude": 71.3333
  },
  "timestamp": "2026-08-27T12:00:00Z",
  "drift_hours": 6.0
}
```

### Example Response
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
    "radius_km": 5.0,
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
      "timestamp": "2026-08-27T06:00:00Z",
      "latitude": 19.3927,
      "longitude": 71.3477,
      "uncertainty_radius_km": 5.0
    }
  ],
  "timestamp": "2026-08-27T12:00:01Z",
  "disclaimer": "M2 DRIFT MODEL: Backtrack trajectory calculated using Member 2 Hydrodynamic Ocean Drift Engine."
}
```

---

## 4. POST /api/ais/candidates

- **HTTP Method**: `POST`
- **Path**: `/api/ais/candidates`
- **Purpose**: Query candidate vessels operating near estimated origin point.

### Request Body Schema (`AISCandidatesRequest`)
| Field | Type | Required | Range / Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `source_latitude` | `float` | **Yes** | `-90.0 <= lat <= 90.0` | Origin latitude |
| `source_longitude` | `float` | **Yes** | `-180.0 <= lon <= 180.0` | Origin longitude |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Origin release time |
| `search_radius_km` | `float` | No | `0.0 < radius <= 500.0` (Default: `50.0`) | Query radius |
| `time_window_hours` | `float` | No | `0.0 < hours <= 72.0` (Default: `12.0`) | Temporal window |

### Response Body Schema (`AISCandidatesResponse` - `200 OK`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `candidates` | `array[VesselAISData]` | List of candidate vessel objects |
| `total_count` | `integer` | Candidate count found |
| `search_radius_km` | `float` | Search radius used |
| `timestamp` | `string` | Execution timestamp |
| `disclaimer` | `string` | AIS data notice |

### Example Request
```json
{
  "source_latitude": 19.3927,
  "source_longitude": 71.3477,
  "timestamp": "2026-08-27T06:00:00Z",
  "search_radius_km": 50.0,
  "time_window_hours": 12.0
}
```

### Example Response
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
      "timestamp": "2026-08-27T05:45:00Z",
      "speed_knots": 4.5,
      "heading_degrees": 135.0,
      "distance_to_source_km": 1.2,
      "vessel_id": "V001",
      "positions": [
        {
          "timestamp": "2026-08-27T05:45:00Z",
          "latitude": 19.418,
          "longitude": 71.335,
          "speed_knots": 4.5,
          "heading_deg": 135.0
        }
      ]
    }
  ],
  "total_count": 1,
  "search_radius_km": 50.0,
  "timestamp": "2026-08-27T12:00:02Z",
  "disclaimer": "M3 AIS SERVICE: Candidates processed using Member 3 AIS Trajectory Engine."
}
```

---

## 5. POST /api/vessels/rank

- **HTTP Method**: `POST`
- **Path**: `/api/vessels/rank`
- **Purpose**: Rank candidate suspect vessels using Member 4's Attribution Engine.

### Request Body Schema (`VesselRankRequest`)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `spill_source` | `object` | **Yes** | `{latitude, longitude}` origin coordinate |
| `timestamp` | `string` | **Yes** | Release timestamp |
| `candidate_vessels` | `array[VesselAISData]` | **Yes** | Candidates array from `/api/ais/candidates` |
| `estimated_release_time` | `string` | No | Optional release time override |
| `search_radius_km` | `float` | No | Spatial radius (Default: `50.0`) |
| `time_window_hours` | `float` | No | Temporal window (Default: `24.0`) |

### Response Body Schema (`VesselRankResponse` - `200 OK`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `ranked_vessels` | `array[RankedVessel]` | Ordered suspect vessels (`rank=1` is top suspect) |
| `total_ranked` | `integer` | Count of ranked vessels |
| `ranked_at` | `string` | Execution timestamp |
| `disclaimer` | `string` | Attribution notice |

#### `RankedVessel` Object Fields:
- `vessel` (`object`): Vessel AIS metadata
- `rank` (`int`): 1-indexed suspect position
- `risk_score` (`float`): Normalized score `0.0` to `1.0`
- `final_score` (`float`): Raw score `0.0` to `100.0`
- `classification` (`string`): e.g. `"Highest-correlated candidate vessel"`
- `explanation` (`string`): Textual explanation summary
- `attribution_factors` (`array`): Feature score breakdown array (`factor_name`, `score`, `description`)

### Example Request
```json
{
  "spill_source": {
    "latitude": 19.3927,
    "longitude": 71.3477
  },
  "timestamp": "2026-08-27T06:00:00Z",
  "candidate_vessels": [
    {
      "mmsi": "419000101",
      "vessel_name": "Ocean Titan",
      "vessel_type": "Crude Oil Tanker",
      "latitude": 19.418,
      "longitude": 71.335,
      "timestamp": "2026-08-27T05:45:00Z",
      "speed_knots": 4.5,
      "heading_degrees": 135.0,
      "distance_to_source_km": 1.2
    }
  ]
}
```

### Example Response
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
        "timestamp": "2026-08-27T05:45:00Z",
        "speed_knots": 4.5,
        "heading_degrees": 135.0,
        "distance_to_source_km": 0.23
      },
      "rank": 1,
      "risk_score": 0.871,
      "final_score": 87.1,
      "classification": "Highest-correlated candidate vessel",
      "explanation": "Classified as 'Highest-correlated candidate vessel' (Rank 1, Score 87.10/100)...",
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
        }
      ]
    }
  ],
  "total_ranked": 1,
  "ranked_at": "2026-08-27T12:00:03Z",
  "disclaimer": "ATTRIBUTION RESULT: Vessel rankings and correlation scores calculated using Member 4 Vessel Attribution Engine."
}
```
