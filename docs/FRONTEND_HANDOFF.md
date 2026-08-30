# SIH 2026 — Master Frontend Integration Package (Member 5 Handoff)

**Project Title**: Leveraging Satellite Imagery to Determine Oil Spills at Sea along with AIS Data Correlations to Identify the Vessel Responsible  
**Author**: Member 6 (Backend / Integration)  
**Target**: Member 5 (Frontend UI)  
**Target Branch**: `integration-backend`  
**Date**: August 29, 2026

---

## 1. Backend Server Overview

- **Base API URL**: `http://127.0.0.1:8000/api`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema**: `http://127.0.0.1:8000/api/openapi.json`

---

## 2. Backend Startup Procedure

From the repository root on branch `integration-backend`:

```bash
# 1. Ensure dependencies are installed
pip install -r backend/requirements.txt

# 2. Launch FastAPI backend
python -m uvicorn backend.main:app --reload --port 8000
```

Verify backend health in browser or terminal:
```bash
curl http://127.0.0.1:8000/api/health
```

---

## 3. CORS & Networking Configuration

### Allowed Development Origins
The backend permits requests from local React/Vite development servers:
- `http://localhost:5173`
- `http://localhost:3000`
- `http://127.0.0.1:5173`
- `http://127.0.0.1:3000`

### Multi-Machine / LAN Network Setup
> **Important Note**: If the React frontend and FastAPI backend run on different physical machines or virtual machines, `127.0.0.1` refers to the frontend machine itself.
>
> On the backend machine, run Uvicorn bound to `0.0.0.0`:
> ```bash
> python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
> ```
> Then replace `127.0.0.1` in the frontend configuration with the backend host's local IP address (e.g. `http://192.168.1.50:8000`).

---

## 4. Frontend Environment Setup (`.env`)

In the `frontend/` directory, configure `.env.development`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

In your React code (`frontend/src/App.jsx` or API helper utility):
```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
```

---

## 5. Active API Endpoints & Schemas

### 5.1 GET /api/health
- **Method**: `GET`
- **Content-Type**: N/A
- **Response `200 OK`**:
```json
{
  "status": "ok",
  "service": "SIH 2026 - Oil Spill & Vessel Attribution API",
  "version": "1.0.0"
}
```

---

### 5.2 POST /api/spill/detect
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "image_id": "SENTINEL-1-20260827-001",
  "image_url": "https://example.com/sat.tif",
  "timestamp": "2026-08-27T12:00:00Z",
  "latitude": 19.4167,
  "longitude": 71.3333
}
```
- **Response `200 OK`**:
```json
{
  "image_id": "SENTINEL-1-20260827-001",
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
- **Response `200 OK`**:
```json
{
  "image_id": "SENTINEL-1-20260827-001",
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

### 5.3 POST /api/spill/detect/upload
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Field Name**: `file` (Sentinel-1 `.tif` or `.tiff` file)
- **Accepted File Formats**: `.tif`, `.tiff` (rejects `.jpg`, `.png`, `.txt` with HTTP 400 Bad Request)
- **Validation**: File validated via `rasterio` for 1-channel numeric SAR raster band and CRS geospatial bounds.
- **JavaScript Fetch Example**:
```javascript
// Note: Do NOT manually set Content-Type header when using FormData;
// browser automatically appends the multipart boundary.
const formData = new FormData();
formData.append("file", selectedFile); // selectedFile from <input type="file" />

const response = await fetch("http://127.0.0.1:8000/api/spill/detect/upload", {
  method: "POST",
  body: formData
});
const data = await response.json();
```
- **Response `200 OK`**:
```json
{
  "image_id": "sentinel1_sar_sample.tif",
  "spill_detected": true,
  "confidence": 0.88,
  "spill_polygon": [
    { "latitude": 19.4267, "longitude": 71.3233 },
    { "latitude": 19.4317, "longitude": 71.3433 },
    { "latitude": 19.4117, "longitude": 71.3533 },
    { "latitude": 19.4017, "longitude": 71.3283 },
    { "latitude": 19.4267, "longitude": 71.3233 }
  ],
  "estimated_area_sq_km": 3.50,
  "timestamp": "2026-08-27T12:00:05Z",
  "disclaimer": "M1 ML REAL INFERENCE: Detection evaluated on uploaded Sentinel-1 GeoTIFF using PyTorch UNet model."
}
```

---

### 5.3 POST /api/spill/backtrack
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
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
- **Response `200 OK`**:
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

### 5.4 POST /api/ais/candidates
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "source_latitude": 19.3927,
  "source_longitude": 71.3477,
  "timestamp": "2026-08-27T06:00:00Z",
  "search_radius_km": 50.0,
  "time_window_hours": 12.0
}
```
- **Response `200 OK`**:
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
      "distance_to_source_km": 1.2
    }
  ],
  "total_count": 1,
  "search_radius_km": 50.0,
  "timestamp": "2026-08-27T12:00:02Z",
  "disclaimer": "M3 AIS SERVICE: Candidates processed using Member 3 AIS Trajectory Engine."
}
```

---

### 5.5 POST /api/vessels/rank
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
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
- **Response `200 OK`**:
```json
{
  "ranked_vessels": [
    {
      "vessel": {
        "mmsi": "419000101",
        "vessel_name": "Ocean Titan",
        "vessel_type": "Crude Oil Tanker",
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
        { "factor_name": "Distance Proximity Score", "score": 0.9773, "description": "Closest approach 0.23 km" },
        { "factor_name": "Time Proximity Score", "score": 0.9200, "description": "Time diff 15 mins" }
      ]
    }
  ],
  "total_ranked": 1,
  "ranked_at": "2026-08-27T12:00:03Z",
  "disclaimer": "ATTRIBUTION RESULT: Vessel rankings and correlation scores calculated using Member 4 Vessel Attribution Engine."
}
```

---

## 6. cURL Commands for Terminal Testing

### Health Check
```bash
curl -X GET http://127.0.0.1:8000/api/health
```

### Detect Spill
```bash
curl -X POST http://127.0.0.1:8000/api/spill/detect \
  -H "Content-Type: application/json" \
  -d '{"image_id":"SAT-001","timestamp":"2026-08-27T12:00:00Z","latitude":19.4167,"longitude":71.3333}'
```

### Backtrack Drift
```bash
curl -X POST http://127.0.0.1:8000/api/spill/backtrack \
  -H "Content-Type: application/json" \
  -d '{"spill_location":{"latitude":19.4167,"longitude":71.3333},"timestamp":"2026-08-27T12:00:00Z","drift_hours":6.0}'
```

### AIS Candidates
```bash
curl -X POST http://127.0.0.1:8000/api/ais/candidates \
  -H "Content-Type: application/json" \
  -d '{"source_latitude":19.3927,"source_longitude":71.3477,"timestamp":"2026-08-27T06:00:00Z","search_radius_km":50.0}'
```

### Rank Suspect Vessels
```bash
curl -X POST http://127.0.0.1:8000/api/vessels/rank \
  -H "Content-Type: application/json" \
  -d '{"spill_source":{"latitude":19.3927,"longitude":71.3477},"timestamp":"2026-08-27T06:00:00Z","candidate_vessels":[{"mmsi":"419000101","vessel_name":"Ocean Titan","vessel_type":"Crude Oil Tanker","latitude":19.418,"longitude":71.335,"timestamp":"2026-08-27T05:45:00Z","speed_knots":4.5,"heading_degrees":135.0,"distance_to_source_km":1.2}]}'
```

---

## 7. Standalone JavaScript Fetch Examples

```javascript
const API_BASE = "http://127.0.0.1:8000/api";

// 1. Detect Spill
export async function apiDetectSpill(lat, lon) {
  const res = await fetch(`${API_BASE}/spill/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_id: "SAT-FRONTEND-01",
      timestamp: new Date().toISOString(),
      latitude: lat,
      longitude: lon
    })
  });
  return await res.json();
}

// 2. Backtrack Origin
export async function apiBacktrackSpill(spillLat, spillLon, timestamp) {
  const res = await fetch(`${API_BASE}/spill/backtrack`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spill_location: { latitude: spillLat, longitude: spillLon },
      timestamp: timestamp,
      drift_hours: 6.0
    })
  });
  return await res.json();
}

// 3. Query AIS Candidates
export async function apiGetAISCandidates(originLat, originLon, timestamp) {
  const res = await fetch(`${API_BASE}/ais/candidates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_latitude: originLat,
      source_longitude: originLon,
      timestamp: timestamp,
      search_radius_km: 50.0
    })
  });
  const data = await res.json();
  return data.candidates;
}

// 4. Rank Suspect Vessels
export async function apiRankVessels(originLat, originLon, timestamp, candidates) {
  const res = await fetch(`${API_BASE}/vessels/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spill_source: { latitude: originLat, longitude: originLon },
      timestamp: timestamp,
      candidate_vessels: candidates
    })
  });
  return await res.json();
}
```

---

## 8. Leaflet Map Integration Contract

| Spatial Layer | Source API Response Field | Leaflet Component | Props / Formatting |
| :--- | :--- | :--- | :--- |
| **Spill Polygon** | `res.spill_polygon` | `<Polygon />` | `positions={res.spill_polygon.map(p => [p.latitude, p.longitude])}` |
| **Origin Zone** | `res.estimated_source_area` | `<Circle />` | `center={[res.center.latitude, res.center.longitude]}` `radius={res.radius_km * 1000}` |
| **Drift Trajectory** | `res.trajectory` | `<Polyline />` | `positions={res.trajectory.map(t => [t.latitude, t.longitude])}` |
| **Suspect Vessels** | `res.ranked_vessels` | `<CircleMarker />` | `center={[v.latitude, v.longitude]}` (Red if `rank === 1`) |

---

## 9. Real vs Synthetic Data Disclosure

- **M1 Satellite Detection**: U-Net model code active. Falls back to mock polygon if PyTorch `.pth` checkpoint weights are missing on local machine.
- **M2 Drift Simulation**: Vector drift physics active, calculated using synthetic wind/ocean current vectors (`data/environment_demo.csv`).
- **M3 AIS Stream**: Spatial/temporal filtering active, running on synthetic AIS CSV records (`data/ais/synthetic/sih_demo_ais.csv`).
- **M4 Attribution Engine**: Real 4-factor scoring engine calculating multi-factor proximity scores.

---

## 10. Troubleshooting Guide

- **`Failed to fetch` / NetworkError**: Backend server is not running. Run `python -m uvicorn backend.main:app --port 8000`.
- **CORS Error**: Check browser console. Ensure backend `CORS_ORIGINS` includes your dev URL (`http://localhost:5173`).
- **`422 Unprocessable Entity`**: Request validation error. Check JSON keys (e.g. `latitude` must be float, `timestamp` must be ISO string).
- **`500 Internal Server Error`**: Inspect Uvicorn console log for stack trace.
