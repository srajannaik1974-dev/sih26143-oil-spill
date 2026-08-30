# SIH 2026 — Master Frontend Integration & Handover Package

**Problem Statement 26143**: Leveraging Satellite Imagery to Determine Oil Spills at Sea along with AIS Data Correlations to Identify the Vessel Responsible  
**Author**: Member 6 (Backend & Integration Layer)  
**Target**: Member 5 (Frontend UI Developer)  
**Branch**: `integration-backend`  
**Date**: August 30, 2026  

---

## 1. Executive Summary & System Architecture

This consolidated document provides the complete, self-contained frontend integration contract and handover package for **Member 5**.

### End-to-End System Data Flow

```
[ React / Vite Frontend UI ]
        │
        │ 1. User selects Sentinel-1 SAR GeoTIFF (.tif / .tiff)
        ▼
[ POST /api/spill/detect/upload ] (multipart/form-data, field: "file")
        │
        │ 2. Real M1 PyTorch U-Net Inference (best_unet.pth on server)
        ▼
[ M1 Spill Detection Result ] (latitude, longitude, area_km2, spill_polygon, confidence)
        │
        │ 3. Send detected coordinates & timestamp
        ▼
[ POST /api/spill/backtrack ]
        │
        │ 4. M2 Hydrodynamic Ocean Drift Simulation
        ▼
[ M2 Origin & Trajectory ] (estimated_source_area center/radius, trajectory array)
        │
        │ 5. Send estimated origin center & time window
        ▼
[ POST /api/ais/candidates ]
        │
        │ 6. M3 AIS Trajectory Candidate Search
        ▼
[ M3 Candidate Vessels List ] (nearby AIS records within spatial/temporal window)
        │
        │ 7. Send origin location & candidate vessel list
        ▼
[ POST /api/vessels/rank ]
        │
        │ 8. M4 Vessel Attribution 4-Factor Correlation Scorer
        ▼
[ Final Suspect Vessel Ranking ] (ranked_vessels, risk_score, attribution_factors)
        │
        │ 9. Render map layers (Polygon, Circles, Trajectory) & Suspect Rankings
        ▼
[ Interactive Investigation Dashboard ]
```

---

## 2. Critical Rule: Model Location & Security

> [!IMPORTANT]
> **The PyTorch model weights file (`best_unet.pth`) resides ONLY on the backend server (`ml/training/checkpoints/best_unet.pth`).**
> 
> The React frontend MUST NOT:
> - Import, download, or reference `best_unet.pth`
> - Import PyTorch or attempt client-side model execution
> - Expose server file paths in client code
> 
> The browser uploads the raw Sentinel-1 GeoTIFF image file (`.tif`/`.tiff`) to `POST /api/spill/detect/upload` via `FormData` and receives standard JSON responses.

---

## 3. Local Development Startup Instructions for Member 5

### 3.1 Start Backend API Server

From the repository root on branch `integration-backend`:

```bash
# 1. Install Python dependencies
pip install -r backend/requirements.txt

# 2. Launch FastAPI Uvicorn dev server
python -m uvicorn backend.main:app --reload --port 8000
```

- **API Base URL**: `http://127.0.0.1:8000/api`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

### 3.2 Frontend Environment Setup (`.env.development`)

In your `frontend/` directory, create/update `.env.development`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

In your React code:

```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
```

### 3.3 CORS Configuration

The backend CORS middleware is configured for local development origins:
- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `http://localhost:3000`
- `http://127.0.0.1:3000`

---

## 4. Complete API Endpoint Contracts & Schemas

### 4.1 GET /api/health
- **HTTP Method**: `GET`
- **Path**: `/api/health`
- **Purpose**: Verify backend server operational status.

**Response (`200 OK`)**:
```json
{
  "status": "ok",
  "service": "SIH 2026 - Oil Spill & Vessel Attribution API",
  "version": "1.0.0"
}
```

---

### 4.2 POST /api/spill/detect/upload
- **HTTP Method**: `POST`
- **Path**: `/api/spill/detect/upload`
- **Content-Type**: `multipart/form-data`
- **Form Field**: `file` (Sentinel-1 `.tif` or `.tiff` file)
- **Accepted Formats**: `.tif`, `.tiff` (rejects `.jpg`, `.png`, `.txt` with HTTP 400 Bad Request)

**JavaScript Fetch Example**:
```javascript
// CRITICAL: Do NOT manually set Content-Type header when using FormData;
// the browser automatically appends the multipart boundary.
const formData = new FormData();
formData.append("file", fileObject); // fileObject from <input type="file" />

const response = await fetch("http://127.0.0.1:8000/api/spill/detect/upload", {
  method: "POST",
  body: formData
});
const data = await response.json();
```

**Response (`200 OK`)**:
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

### 4.3 POST /api/spill/detect (JSON Metadata)
- **HTTP Method**: `POST`
- **Path**: `/api/spill/detect`
- **Content-Type**: `application/json`

**Request Body (`application/json`)**:
```json
{
  "image_id": "SAT-IMG-2026-001",
  "image_url": "https://example.com/satellite/image.tif",
  "timestamp": "2026-08-27T12:00:00Z",
  "latitude": 19.4167,
  "longitude": 71.3333
}
```

**Response (`200 OK`)**:
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

### 4.4 POST /api/spill/backtrack (M2 Ocean Drift)
- **HTTP Method**: `POST`
- **Path**: `/api/spill/backtrack`
- **Content-Type**: `application/json`

**Request Body**:
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

**Response (`200 OK`)**:
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

### 4.5 POST /api/ais/candidates (M3 AIS Search)
- **HTTP Method**: `POST`
- **Path**: `/api/ais/candidates`
- **Content-Type**: `application/json`

**Request Body**:
```json
{
  "source_latitude": 28.912903,
  "source_longitude": -89.028632,
  "timestamp": "2026-08-29T18:35:00Z",
  "search_radius_km": 50.0,
  "time_window_hours": 12.0
}
```

**Response (`200 OK`)**:
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

### 4.6 POST /api/vessels/rank (M4 Suspect Vessel Attribution)
- **HTTP Method**: `POST`
- **Path**: `/api/vessels/rank`
- **Content-Type**: `application/json`

**Request Body**:
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

**Response (`200 OK`)**:
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

## 5. Complete Standalone JavaScript Code Examples (`src/api/spillService.js`)

```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";

// 1. Health Check
export async function checkHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  return await res.json();
}

// 2. Upload Sentinel-1 GeoTIFF Image (.tif/.tiff) for Real M1 U-Net Inference
export async function uploadSentinelImage(file) {
  const formData = new FormData();
  formData.append("file", file); // Must use field name "file"

  const res = await fetch(`${API_BASE_URL}/spill/detect/upload`, {
    method: "POST",
    body: formData
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Image analysis failed");
  }
  return await res.json();
}

// 3. Backtrack Ocean Drift Trajectory (M2)
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
  if (!res.ok) throw new Error("Drift backtrack calculation failed");
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
  if (!res.ok) throw new Error("AIS candidate query failed");
  const data = await res.json();
  return data.candidates;
}

// 5. Rank Suspect Vessels using 4-Factor Attribution Engine (M4)
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

## 6. Leaflet Map & UI Visualization Mapping Contract

| Backend Response Field | Map Layer Component | React/Leaflet Formatting |
| :--- | :--- | :--- |
| `res.spill_polygon` | `<Polygon />` | `positions={res.spill_polygon.map(p => [p.latitude, p.longitude])}` |
| `res.estimated_source_area.center` | `<Circle />` | `center={[c.latitude, c.longitude]}` `radius={res.estimated_source_area.radius_km * 1000}` |
| `res.trajectory` | `<Polyline />` | `positions={res.trajectory.map(t => [t.latitude, t.longitude])}` |
| `res.ranked_vessels` | `<Marker />` list | Marker icons at `[v.latitude, v.longitude]`. Render Rank 1 in Red. |
| `v.risk_score` | Badge / Progress Bar | Format as percentage: `(v.risk_score * 100).toFixed(1) + "%"` |
| `v.attribution_factors` | Table / Accordion | Render 4-factor scoring breakdown (`factor_name`, `score`, `description`) |

---

## 7. Error Handling Guidelines for UI

| HTTP Status | Error Trigger | Recommended UI Display Message |
| :--- | :--- | :--- |
| `400 Bad Request` | Non-TIFF extension or corrupt raster | `"Invalid Image: Only 1-channel Sentinel-1 GeoTIFF (.tif, .tiff) files are accepted."` |
| `422 Unprocessable` | Invalid field range or missing key | `"Invalid Input: Please check coordinates and time parameters."` |
| `500 Server Error` | Server-side execution exception | `"Server Error: Unable to complete analysis. Please try again."` |
| `Failed to fetch` | Uvicorn server is offline | `"Connection Error: Unable to reach backend server at http://127.0.0.1:8000."` |

---

## 8. Verification & Integration Testing

To confirm backend readiness on `integration-backend`:

```bash
python -m pytest -v
```

**Baseline Pass Rate**: **175 / 175 tests PASSED (100%)**.
