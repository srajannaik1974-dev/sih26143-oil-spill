# Definitive Frontend Integration Guide — Single Source of Truth

**Project Title**: SIH 2026 Problem Statement 26143 — Leveraging Satellite Imagery to Determine Oil Spills at Sea along with AIS Data Correlations to Identify the Vessel Responsible  
**Author**: Member 6 (Backend & Integration Layer)  
**Target Audience**: Member 5 (Frontend UI Developer)  
**Target Branch**: `integration-backend`  
**Date**: August 30, 2026  

---

> [!IMPORTANT]
> **CRITICAL ARCHITECTURAL REQUIREMENT: THE .PTH MODEL FILE IS BACKEND/SERVER-ONLY**
> 
> The PyTorch model weights file (`best_unet.pth`) resides STRICTLY on the backend server (`ml/training/checkpoints/best_unet.pth`).
> 
> **The Frontend MUST NEVER:**
> 1. Import, download, or reference `best_unet.pth`.
> 2. Put `best_unet.pth` into `frontend/` or `public/`.
> 3. Bundle `best_unet.pth` with Vite/React.
> 4. Import PyTorch or attempt client-side model execution in browser.
> 5. Send `.pth` files from frontend to backend.
> 
> **The Browser Uploads the Sentinel-1 SAR GeoTIFF (`.tif`/`.tiff`) image via `FormData` to `POST /api/spill/detect/upload`, and FastAPI runs M1 inference on the server.**

---

## Table of Contents

1. [System Architecture & Sequence Overview](#1-system-architecture--sequence-overview)
2. [Backend Startup & Local Environment](#2-backend-startup--local-environment)
3. [CORS Configuration](#3-cors-configuration)
4. [M1 Satellite Oil Spill Detection Architecture](#4-m1-satellite-oil-spill-detection-architecture)
5. [Complete Step-by-Step Frontend Workflow](#5-complete-step-by-step-frontend-workflow)
6. [Definitive API Endpoint Specifications & Contracts](#6-definitive-api-endpoint-specifications--contracts)
   - [6.1 GET /api/health](#61-get-apihealth)
   - [6.2 POST /api/spill/detect/upload](#62-post-apispilldetectupload)
   - [6.3 POST /api/spill/detect](#63-post-apispilldetect)
   - [6.4 POST /api/spill/backtrack](#64-post-apispillbacktrack)
   - [6.5 POST /api/ais/candidates](#65-post-apiaiscandidates)
   - [6.6 POST /api/vessels/rank](#66-post-apivesselsrank)
7. [JavaScript Fetch Code Examples](#7-javascript-fetch-code-examples)
8. [cURL Terminal Test Commands](#8-curl-terminal-test-commands)
9. [Frontend UI Data & Map Display Mapping](#9-frontend-ui-data--map-display-mapping)
10. [Geospatial Coordinate & Map Layer Conventions](#10-geospatial-coordinate--map-layer-conventions)
11. [Recommended UI Loading States](#11-recommended-ui-loading-states)
12. [Error Handling Guidelines](#12-error-handling-guidelines)
13. [Frontend Testing & Verification Guide](#13-frontend-testing--verification-guide)
14. [Real Data vs. Synthetic Data Status](#14-real-data-vs-synthetic-data-status)
15. [Current Backend Test & Integration Status](#15-current-backend-test--integration-status)
16. [Member 5 Frontend Developer Checklist](#16-member-5-frontend-developer-checklist)

---

## 1. System Architecture & Sequence Overview

The system links five modular layers (M1 to M4 + M6 Backend) into a single continuous pipeline consumed by the Member 5 React frontend UI.

```
[ User Browser / React UI ]
           │
           │ 1. Select Sentinel-1 SAR GeoTIFF (.tif / .tiff)
           ▼
[ POST /api/spill/detect/upload ] ────► (Content-Type: multipart/form-data)
           │
           │ 2. Backend loads ml/training/checkpoints/best_unet.pth
           ▼
[ M1 PyTorch UNet Model ]
           │
           │ 3. Returns SpillDetectionResponse JSON
           ▼
[ React Component State ]
           │
           │ 4. Extract centroid lat/lon & timestamp ──► POST /api/spill/backtrack
           ▼
[ M2 Ocean Drift Engine ]
           │
           │ 5. Returns BacktrackResponse (Origin center, uncertainty radius & trajectory)
           ▼
[ React Component State ]
           │
           │ 6. Pass origin coordinates & time window ──► POST /api/ais/candidates
           ▼
[ M3 AIS Stream Search ]
           │
           │ 7. Returns AISCandidatesResponse (Candidate vessels within radius)
           ▼
[ React Component State ]
           │
           │ 8. Pass candidates & origin point ────────► POST /api/vessels/rank
           ▼
[ M4 Vessel Attribution Engine ]
           │
           │ 9. Returns VesselRankResponse (Ranked suspect vessels & 4-factor breakdown)
           ▼
[ Leaflet Map & Investigation UI ]
```

---

## 2. Backend Startup & Local Environment

### 2.1 Starting the Backend API Server

Run the backend server from the repository root on branch `integration-backend`:

```bash
# 1. Install required Python dependencies
pip install -r backend/requirements.txt

# 2. Launch FastAPI dev server via Uvicorn
python -m uvicorn backend.main:app --reload --port 8000
```

### 2.2 Server URLs & Interactive Documentation

- **Backend Base URL**: `http://127.0.0.1:8000` or `http://localhost:8000`
- **API Endpoints Base Path**: `http://127.0.0.1:8000/api`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc API Documentation**: `http://127.0.0.1:8000/redoc`
- **OpenAPI JSON Schema**: `http://127.0.0.1:8000/api/openapi.json`

### 2.3 Frontend Environment Configuration

In `frontend/.env.development`:
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

In your React source code:
```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
```

---

## 3. CORS Configuration

FastAPI CORS middleware is pre-configured in `backend/config.py` to support standard local development origins:

- `http://localhost:5173` (Default Vite React UI port)
- `http://127.0.0.1:5173`
- `http://localhost:3000` (Default Create-React-App port)
- `http://127.0.0.1:3000`
- `*` (Development wildcards enabled)

Allowed methods: `GET`, `POST`, `OPTIONS`  
Allowed headers: `*`  
Allow credentials: `true`

---

## 4. M1 Satellite Oil Spill Detection Architecture

- **Model File Location**: `ml/training/checkpoints/best_unet.pth` (89.91 MB single PyTorch weights zip file).
- **Architecture**: `UNet(in_channels=1, out_channels=1, base_features=32)` (7,849,025 parameters).
- **Supported Input**: 1-channel Sentinel-1 SAR GeoTIFF (`.tif` / `.tiff` format).
- **Inference Pipeline**:
  1. Frontend uploads `.tif`/`.tiff` file.
  2. Backend validates raster structure (1 band float32/int16 VV SAR backscatter).
  3. `OilSpillPredictor` extracts 256x256 tiles, evaluates U-Net sigmoid confidence, and constructs binary mask.
  4. Spatial transform calculates GeoTIFF centroid (`latitude`, `longitude`), `estimated_area_sq_km`, and boundary polygon vertices.
  5. JSON response returned to frontend.

---

## 5. Complete Step-by-Step Frontend Workflow

Follow these exact steps in your React component event flow:

- **STEP 1**: User selects a Sentinel-1 `.tif`/`.tiff` GeoTIFF image file via `<input type="file" accept=".tif,.tiff" />`.
- **STEP 2**: Frontend validates file extension (`.tif` or `.tiff`).
- **STEP 3**: Frontend creates `FormData` and appends file: `formData.append("file", file)`.
- **STEP 4**: Frontend sends `POST http://127.0.0.1:8000/api/spill/detect/upload` with body `formData`.
- **STEP 5**: Frontend receives `SpillDetectionResponse` containing `spill_detected`, `confidence`, `estimated_area_sq_km`, `spill_polygon`, `latitude`, `longitude`, `timestamp`.
- **STEP 6**: If `spill_detected === true`, extract centroid coordinates (`lat`, `lon`) and `timestamp`.
- **STEP 7**: Frontend sends `POST http://127.0.0.1:8000/api/spill/backtrack` with JSON `{ spill_location: { latitude, longitude }, timestamp, drift_hours: 6.0 }`.
- **STEP 8**: Frontend receives `BacktrackResponse` containing `estimated_source_area` (`center`, `radius_km`) and `trajectory` array.
- **STEP 9**: Extract estimated origin center coordinates (`originLat`, `originLon`).
- **STEP 10**: Frontend sends `POST http://127.0.0.1:8000/api/ais/candidates` with JSON `{ source_latitude: originLat, source_longitude: originLon, timestamp, search_radius_km: 50.0, time_window_hours: 12.0 }`.
- **STEP 11**: Frontend receives `AISCandidatesResponse` containing `candidates` array.
- **STEP 12**: Frontend sends `POST http://127.0.0.1:8000/api/vessels/rank` with JSON `{ spill_source: { latitude: originLat, longitude: originLon }, timestamp, candidate_vessels: candidates }`.
- **STEP 13**: Frontend receives `VesselRankResponse` containing `ranked_vessels` ordered by suspect probability.
- **STEP 14**: Render Leaflet map elements (Polygon, Circle, Trajectory Polyline, Vessel Markers) and attribution summary table.

---

## 6. Definitive API Endpoint Specifications & Contracts

### 6.1 GET /api/health

- **Method**: `GET`
- **Path**: `/api/health`
- **Content-Type**: `application/json`
- **Purpose**: Verify backend API health and version.

#### Request
No query parameters or request body.

#### Response (`200 OK`)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `status` | `string` | **Yes** | Server status (`"ok"`) |
| `service` | `string` | **Yes** | API Service Title |
| `version` | `string` | **Yes** | Service Version (`"1.0.0"`) |

```json
{
  "status": "ok",
  "service": "SIH 2026 - Oil Spill & Vessel Attribution API",
  "version": "1.0.0"
}
```

---

### 6.2 POST /api/spill/detect/upload

- **Method**: `POST`
- **Path**: `/api/spill/detect/upload`
- **Content-Type**: `multipart/form-data`
- **Form Field**: `file` (`UploadFile`)
- **Accepted Extensions**: `.tif`, `.tiff`
- **Purpose**: Upload a raw 1-channel Sentinel-1 SAR GeoTIFF image file and run server-side PyTorch U-Net inference.

#### Request Form Data
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | `File` (Binary) | **Yes** | 1-channel Sentinel-1 GeoTIFF image file (`.tif`/`.tiff`) |

#### Response (`200 OK`) — `SpillDetectionResponse`
| Field | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `image_id` | `string` | No | Original filename uploaded |
| `spill_detected` | `boolean` | No | `true` if oil spill detected by U-Net model |
| `confidence` | `float` | No | Model confidence score between `0.0` and `1.0` |
| `spill_polygon` | `array[GeoCoordinate]` | No | Array of `{latitude, longitude}` polygon vertices |
| `estimated_area_sq_km` | `float` | Yes | Calculated oil spill surface area in km² |
| `timestamp` | `string` | No | Acquisition / evaluation ISO-8601 UTC timestamp |
| `disclaimer` | `string` | No | Model execution disclaimer string |

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

### 6.3 POST /api/spill/detect

- **Method**: `POST`
- **Path**: `/api/spill/detect`
- **Content-Type**: `application/json`
- **Purpose**: JSON metadata detection endpoint (backward-compatible).

#### Request Body (`SpillDetectionRequest`)
| Field | Type | Required | Range / Constraint | Description |
| :--- | :--- | :--- | :--- | :--- |
| `image_id` | `string` | **Yes** | Non-empty | Image identifier |
| `image_url` | `string` | No | Valid URI | Optional image URL |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Image acquisition timestamp |
| `latitude` | `float` | **Yes** | `-90.0 <= lat <= 90.0` | Target centroid latitude |
| `longitude` | `float` | **Yes** | `-180.0 <= lon <= 180.0` | Target centroid longitude |

```json
{
  "image_id": "SAT-IMG-2026-001",
  "image_url": "https://example.com/satellite/image.tif",
  "timestamp": "2026-08-27T12:00:00Z",
  "latitude": 19.4167,
  "longitude": 71.3333
}
```

#### Response (`200 OK`) — `SpillDetectionResponse`
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

### 6.4 POST /api/spill/backtrack

- **Method**: `POST`
- **Path**: `/api/spill/backtrack`
- **Content-Type**: `application/json`
- **Purpose**: Calculate hydrodynamic ocean drift backtrack trajectory and estimated release origin.

#### Request Body (`BacktrackRequest`)
| Field | Type | Required | Default / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `spill_location` | `object` | **Yes** | `{latitude, longitude}` | Detected spill centroid |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Detection timestamp |
| `drift_hours` | `float` | No | `24.0` (`0.0 < h <= 168.0`) | Hours to backtrack |
| `wind_vector_deg` | `float` | No | `0.0 <= deg < 360.0` | Wind angle override |
| `current_vector_deg` | `float` | No | `0.0 <= deg < 360.0` | Ocean current angle override |

```json
{
  "spill_location": {
    "latitude": 28.912903,
    "longitude": -89.028632
  },
  "timestamp": "2026-08-30T00:35:00Z",
  "drift_hours": 6.0
}
```

#### Response (`200 OK`) — `BacktrackResponse`
| Field | Type | Description |
| :--- | :--- | :--- |
| `spill_location` | `object` | Echoes `{latitude, longitude}` input |
| `detection_timestamp` | `string` | Echoes detection timestamp |
| `estimated_source_area` | `object` | `{center: {latitude, longitude}, radius_km, boundary_polygon}` |
| `trajectory` | `array` | Backward drift path points `{timestamp, latitude, longitude, uncertainty_radius_km}` |
| `timestamp` | `string` | Execution timestamp |
| `disclaimer` | `string` | Drift engine disclaimer |

```json
{
  "spill_location": {
    "latitude": 28.912903,
    "longitude": -89.028632
  },
  "detection_timestamp": "2026-08-30T00:35:00Z",
  "estimated_source_area": {
    "center": {
      "latitude": 28.912903,
      "longitude": -89.028632
    },
    "radius_km": 5.0,
    "boundary_polygon": [
      { "latitude": 28.932903, "longitude": -89.028632 },
      { "latitude": 28.912903, "longitude": -89.008632 },
      { "latitude": 28.892903, "longitude": -89.028632 },
      { "latitude": 28.912903, "longitude": -89.048632 },
      { "latitude": 28.932903, "longitude": -89.028632 }
    ]
  },
  "trajectory": [
    {
      "timestamp": "2026-08-30T00:35:00Z",
      "latitude": 28.912903,
      "longitude": -89.028632,
      "uncertainty_radius_km": 1.5
    },
    {
      "timestamp": "2026-08-29T18:35:00Z",
      "latitude": 28.895000,
      "longitude": -89.010000,
      "uncertainty_radius_km": 5.0
    }
  ],
  "timestamp": "2026-08-30T00:35:01Z",
  "disclaimer": "M2 DRIFT MODEL: Backtrack trajectory calculated using Member 2 Hydrodynamic Ocean Drift Engine."
}
```

---

### 6.5 POST /api/ais/candidates

- **Method**: `POST`
- **Path**: `/api/ais/candidates`
- **Content-Type**: `application/json`
- **Purpose**: Query candidate vessels operating near estimated origin point within spatial/temporal window.

#### Request Body (`AISCandidatesRequest`)
| Field | Type | Required | Default / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `source_latitude` | `float` | **Yes** | `-90.0 <= lat <= 90.0` | Origin latitude |
| `source_longitude` | `float` | **Yes** | `-180.0 <= lon <= 180.0` | Origin longitude |
| `timestamp` | `string` | **Yes** | ISO-8601 Datetime | Estimated release timestamp |
| `search_radius_km` | `float` | No | `50.0` (`0.0 < r <= 500.0`) | Spatial search radius |
| `time_window_hours` | `float` | No | `12.0` (`0.0 < h <= 72.0`) | Temporal search window |

```json
{
  "source_latitude": 28.912903,
  "source_longitude": -89.028632,
  "timestamp": "2026-08-29T18:35:00Z",
  "search_radius_km": 50.0,
  "time_window_hours": 12.0
}
```

#### Response (`200 OK`) — `AISCandidatesResponse`
| Field | Type | Description |
| :--- | :--- | :--- |
| `candidates` | `array[VesselAISData]` | List of matching candidate vessel objects |
| `total_count` | `integer` | Count of candidate vessels found |
| `search_radius_km` | `float` | Search radius used |
| `timestamp` | `string` | Execution timestamp |
| `disclaimer` | `string` | AIS data notice |

```json
{
  "candidates": [
    {
      "mmsi": "419000101",
      "vessel_name": "OCEAN TITAN",
      "vessel_type": "Crude Oil Tanker",
      "callsign": "VRHK8",
      "flag": "Panama",
      "latitude": 28.918,
      "longitude": -89.025,
      "timestamp": "2026-08-29T18:15:00Z",
      "speed_knots": 4.5,
      "heading_degrees": 135.0,
      "distance_to_source_km": 0.8
    }
  ],
  "total_count": 1,
  "search_radius_km": 50.0,
  "timestamp": "2026-08-30T00:35:02Z",
  "disclaimer": "M3 AIS SERVICE: Candidates processed using Member 3 AIS Trajectory Engine."
}
```

---

### 6.6 POST /api/vessels/rank

- **Method**: `POST`
- **Path**: `/api/vessels/rank`
- **Content-Type**: `application/json`
- **Purpose**: Rank candidate vessels using Member 4's 4-Factor Vessel Attribution Engine.

#### Request Body (`VesselRankRequest`)
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `spill_source` | `object` | **Yes** | `{latitude, longitude}` origin coordinate |
| `timestamp` | `string` | **Yes** | ISO-8601 release timestamp |
| `candidate_vessels` | `array[VesselAISData]` | **Yes** | Array of candidates from `/api/ais/candidates` |
| `estimated_release_time` | `string` | No | Optional release time override |
| `search_radius_km` | `float` | No | Search radius (Default: `50.0`) |
| `time_window_hours` | `float` | No | Time window (Default: `24.0`) |

```json
{
  "spill_source": {
    "latitude": 28.912903,
    "longitude": -89.028632
  },
  "timestamp": "2026-08-29T18:35:00Z",
  "candidate_vessels": [
    {
      "mmsi": "419000101",
      "vessel_name": "OCEAN TITAN",
      "vessel_type": "Crude Oil Tanker",
      "latitude": 28.918,
      "longitude": -89.025,
      "timestamp": "2026-08-29T18:15:00Z",
      "speed_knots": 4.5,
      "heading_degrees": 135.0,
      "distance_to_source_km": 0.8
    }
  ]
}
```

#### Response (`200 OK`) — `VesselRankResponse`
| Field | Type | Description |
| :--- | :--- | :--- |
| `ranked_vessels` | `array[RankedVessel]` | Ranked list of suspect vessels (`rank=1` is top suspect) |
| `total_ranked` | `integer` | Count of ranked vessels |
| `ranked_at` | `string` | Execution timestamp |
| `disclaimer` | `string` | Attribution engine disclaimer |

##### `RankedVessel` Object Fields:
- `vessel` (`object`): Vessel AIS record (`mmsi`, `vessel_name`, `latitude`, `longitude`, etc.)
- `rank` (`integer`): 1-indexed suspect position
- `risk_score` (`float`): Normalized risk score between `0.0` and `1.0`
- `final_score` (`float`): Scaled score `0.0` to `100.0`
- `classification` (`string`): e.g. `"Highest-correlated candidate vessel"`
- `explanation` (`string`): Textual explanation summary
- `attribution_factors` (`array`): Feature score breakdown array (`factor_name`, `score`, `description`)

```json
{
  "ranked_vessels": [
    {
      "vessel": {
        "mmsi": "419000101",
        "vessel_name": "OCEAN TITAN",
        "vessel_type": "Crude Oil Tanker",
        "callsign": "VRHK8",
        "flag": "Panama",
        "latitude": 28.918,
        "longitude": -89.025,
        "timestamp": "2026-08-29T18:15:00Z",
        "speed_knots": 4.5,
        "heading_degrees": 135.0,
        "distance_to_source_km": 0.8
      },
      "rank": 1,
      "risk_score": 0.8076,
      "final_score": 80.76,
      "classification": "Highest-correlated candidate vessel",
      "explanation": "Classified as 'Highest-correlated candidate vessel' (Rank 1, Score 80.76/100)...",
      "attribution_factors": [
        {
          "factor_name": "Distance Proximity Score",
          "score": 0.9520,
          "description": "Closest approach is 0.8 km from spill origin."
        },
        {
          "factor_name": "Time Proximity Score",
          "score": 0.9000,
          "description": "Time difference is 20.0 minutes from release time."
        }
      ]
    }
  ],
  "total_ranked": 1,
  "ranked_at": "2026-08-30T00:35:03Z",
  "disclaimer": "ATTRIBUTION RESULT: Vessel rankings and correlation scores calculated using Member 4 Vessel Attribution Engine."
}
```

---

## 7. JavaScript Fetch Code Examples

Create `frontend/src/api/spillApi.js`:

```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";

// 1. Health Check
export async function checkBackendHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error("Backend service unavailable");
  return await res.json();
}

// 2. Upload Sentinel-1 SAR GeoTIFF (.tif/.tiff) for Real M1 U-Net Inference
export async function uploadSarGeoTiff(fileObject) {
  const formData = new FormData();
  formData.append("file", fileObject); // Field name MUST be "file"

  // CRITICAL: Do NOT manually set Content-Type header when using FormData.
  const res = await fetch(`${API_BASE_URL}/spill/detect/upload`, {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const errorBody = await res.json();
    throw new Error(errorBody.detail || "SAR image upload inference failed");
  }
  return await res.json();
}

// 3. Backtrack Ocean Drift (M2)
export async function backtrackDrift(lat, lon, timestamp, driftHours = 6.0) {
  const res = await fetch(`${API_BASE_URL}/spill/backtrack`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spill_location: { latitude: lat, longitude: lon },
      timestamp: timestamp || new Date().toISOString(),
      drift_hours: driftHours
    })
  });
  if (!res.ok) throw new Error("Ocean drift backtrack calculation failed");
  return await res.json();
}

// 4. Query AIS Candidate Vessels (M3)
export async function getAisCandidates(originLat, originLon, timestamp, radiusKm = 50.0) {
  const res = await fetch(`${API_BASE_URL}/ais/candidates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_latitude: originLat,
      source_longitude: originLon,
      timestamp: timestamp,
      search_radius_km: radiusKm,
      time_window_hours: 12.0
    })
  });
  if (!res.ok) throw new Error("AIS candidate vessel search failed");
  const data = await res.json();
  return data.candidates;
}

// 5. Rank Suspect Vessels (M4)
export async function rankSuspectVessels(originLat, originLon, timestamp, candidates) {
  const res = await fetch(`${API_BASE_URL}/vessels/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spill_source: { latitude: originLat, longitude: originLon },
      timestamp: timestamp,
      candidate_vessels: candidates
    })
  });
  if (!res.ok) throw new Error("Vessel attribution ranking failed");
  return await res.json();
}
```

---

## 8. cURL Terminal Test Commands

Use these terminal commands to verify the running backend API:

```bash
# 1. Health Check
curl -X GET http://127.0.0.1:8000/api/health

# 2. Upload GeoTIFF Image for Real M1 U-Net Inference
curl -X POST http://127.0.0.1:8000/api/spill/detect/upload \
  -F "file=@ml/dataset/test_input_gdrive.tif"

# 3. Backtrack Ocean Drift (M2)
curl -X POST http://127.0.0.1:8000/api/spill/backtrack \
  -H "Content-Type: application/json" \
  -d '{
    "spill_location": {"latitude": 28.912903, "longitude": -89.028632},
    "timestamp": "2026-08-30T00:35:00Z",
    "drift_hours": 6.0
  }'

# 4. Query AIS Candidate Vessels (M3)
curl -X POST http://127.0.0.1:8000/api/ais/candidates \
  -H "Content-Type: application/json" \
  -d '{
    "source_latitude": 28.912903,
    "source_longitude": -89.028632,
    "timestamp": "2026-08-29T18:35:00Z",
    "search_radius_km": 50.0
  }'

# 5. Rank Suspect Vessels (M4)
curl -X POST http://127.0.0.1:8000/api/vessels/rank \
  -H "Content-Type: application/json" \
  -d '{
    "spill_source": {"latitude": 28.912903, "longitude": -89.028632},
    "timestamp": "2026-08-29T18:35:00Z",
    "candidate_vessels": [
      {
        "mmsi": "419000101",
        "vessel_name": "OCEAN TITAN",
        "vessel_type": "Crude Oil Tanker",
        "latitude": 28.918,
        "longitude": -89.025,
        "timestamp": "2026-08-29T18:15:00Z",
        "speed_knots": 4.5,
        "heading_degrees": 135.0,
        "distance_to_source_km": 0.8
      }
    ]
  }'
```

---

## 9. Frontend UI Data & Map Display Mapping

| Backend Response JSON Field | UI Component / Layer | Format & Display Guideline |
| :--- | :--- | :--- |
| `spill_detected` | Status Header / Badge | `true` -> Green/Red "Oil Spill Detected" Banner |
| `confidence` | Meter / Percentage | `(confidence * 100).toFixed(1) + "%"` |
| `estimated_area_sq_km` | Metric Card | `estimated_area_sq_km + " km²"` |
| `spill_polygon` | Leaflet `<Polygon />` | Map vertices to Leaflet `[[lat, lon], ...]` |
| `estimated_source_area.center` | Leaflet `<Marker />` / `<Circle />` | Draw origin marker + circle (`radius_km * 1000` meters) |
| `trajectory` | Leaflet `<Polyline />` | Draw backward drift trajectory line |
| `ranked_vessels` | Table & Map Markers | Highlight Rank #1 vessel in RED. Render MMSI, Name, Speed |
| `risk_score` / `final_score` | Progress Bar | Display attribution percentage `(risk_score * 100).toFixed(1) + "%"` |
| `attribution_factors` | Breakdown Accordion | Display factor scores (`factor_name`, `score`, `description`) |

---

## 10. Geospatial Coordinate & Map Layer Conventions

> [!CAUTION]
> **COORDINATE ORDER CONVENTION**:
> All backend API schemas consistently use **Latitude FIRST, Longitude SECOND**:
> `{"latitude": 28.912903, "longitude": -89.028632}`
> 
> When mapping to **Leaflet** (`react-leaflet`):
> - Leaflet expects `[latitude, longitude]` array order: `[28.912903, -89.028632]`.
> - Map polygon array: `spill_polygon.map(pt => [pt.latitude, pt.longitude])`.

---

## 11. Recommended UI Loading States

Render step-by-step progress feedback during execution:

1. `Uploading Sentinel-1 SAR GeoTIFF image...`
2. `Executing PyTorch U-Net oil spill detection model...`
3. `Simulating hydrodynamic ocean drift backtracking...`
4. `Searching AIS vessel trajectory stream...`
5. `Evaluating 4-factor vessel attribution risk scores...`
6. `Pipeline complete.`

---

## 12. Error Handling Guidelines

| HTTP Status | Trigger Condition | Recommended UI Display Message |
| :--- | :--- | :--- |
| `400 Bad Request` | Invalid file extension or corrupt GeoTIFF | `"Invalid File: Only 1-channel Sentinel-1 GeoTIFF (.tif, .tiff) files are supported."` |
| `422 Unprocessable` | Out-of-bounds lat/lon or missing key | `"Invalid Input: Please verify input parameters and timestamp format."` |
| `500 Server Error` | Server-side execution exception | `"Server Error: Unable to complete analysis. Please try again."` |
| `TypeError: Failed to fetch` | Backend server offline | `"Connection Error: Backend server unreachable at http://127.0.0.1:8000."` |

---

## 13. Frontend Testing & Verification Guide

1. Start backend: `python -m uvicorn backend.main:app --reload --port 8000`.
2. Start frontend UI: `npm run dev`.
3. Open `http://localhost:5173`.
4. Upload a Sentinel-1 `.tif`/`.tiff` file.
5. Verify M1 detection results display on map.
6. Verify M2 drift trajectory and origin circle render.
7. Verify M3 AIS candidate vessels populate.
8. Verify M4 Rank 1 suspect vessel displays with attribution breakdown.

---

## 14. Real Data vs. Synthetic Data Status

- **M1 Satellite Detection**: **REAL** (PyTorch U-Net model trained on Sentinel-1 SAR imagery).
- **M2 Drift Simulation**: **REAL** (Hydrodynamic ocean drift velocity vector model).
- **M3 AIS Data**: **SYNTHETIC STREAM** (Realistic Gulf of Mexico vessel AIS trajectory generator).
- **M4 Attribution Engine**: **REAL** (4-factor spatial, temporal, course, and type correlation scorer).

---

## 15. Current Backend Test & Integration Status

All test suites on branch `integration-backend` pass cleanly:

```bash
python -m pytest -v
```

- **Total Backend Tests**: **175**
- **Passed**: **175** (100% pass rate)
- **Failed**: **0**
- **Skipped**: **0**

---

## 16. Member 5 Frontend Developer Checklist

- [ ] Backend server running on `http://127.0.0.1:8000`
- [ ] Frontend `.env.development` configured with `VITE_API_BASE_URL=http://127.0.0.1:8000`
- [ ] GeoTIFF File input accepts `.tif,.tiff`
- [ ] `FormData` upload appends file to field `"file"`
- [ ] `POST /api/spill/detect/upload` connected
- [ ] M1 detection polygon & centroid rendered on Leaflet map
- [ ] `POST /api/spill/backtrack` connected with M1 lat/lon
- [ ] M2 origin circle & drift polyline rendered on Leaflet map
- [ ] `POST /api/ais/candidates` connected with M2 origin lat/lon
- [ ] `POST /api/vessels/rank` connected with candidates array
- [ ] Rank #1 suspect vessel highlighted in RED on map & summary card
- [ ] 4-Factor attribution breakdown accordion rendered
- [ ] Loading progress states rendered
- [ ] Error boundary handles HTTP 400, 422, 500 cleanly
- [ ] End-to-end integration verified
