# SIH 2026 — Member 5 Frontend Integration Guide

**Project Title**: Leveraging Satellite Imagery to Determine Oil Spills at Sea along with AIS Data Correlations to Identify the Vessel Responsible  
**Author**: Member 6 (Backend & Integration Layer)  
**Target Audience**: Member 5 (Frontend UI Developer)  
**Target Branch**: `integration-backend`  
**Date**: August 30, 2026

---

## 1. Executive Summary & System Architecture

This guide provides step-by-step instructions for Member 5 to connect the React/Vite frontend UI with the integrated FastAPI backend API.

### High-Level Architecture Flow

```
[ Browser / React UI ]
        │
        │ 1. Upload Sentinel-1 SAR GeoTIFF (.tif/.tiff) via FormData
        ▼
[ POST /api/spill/detect/upload ]
        │
        │ 2. Real M1 PyTorch U-Net Inference (best_unet.pth)
        ▼
[ M1 Spill Detection Response ] (latitude, longitude, area, polygon, confidence)
        │
        │ 3. Send detected spill location & timestamp
        ▼
[ POST /api/spill/backtrack ]
        │
        │ 4. M2 Hydrodynamic Ocean Drift Simulation
        ▼
[ M2 Origin Response ] (estimated origin center, radius_km, drift trajectory)
        │
        │ 5. Send estimated origin location & time window
        ▼
[ POST /api/ais/candidates ]
        │
        │ 6. M3 AIS Trajectory Candidate Filtering
        ▼
[ M3 Candidates Response ] (list of candidate vessels within spatial/temporal window)
        │
        │ 7. Send origin location & candidate vessels
        ▼
[ POST /api/vessels/rank ]
        │
        │ 8. M4 Vessel Attribution 4-Factor Scoring Engine
        ▼
[ Final Suspect Ranking Response ] (ranked vessels, risk scores, correlation factors)
```

---

## 2. Critical Rule: Model Location & Security

> [!IMPORTANT]
> **The PyTorch model weights file (`best_unet.pth`) belongs ONLY on the backend server (`ml/training/checkpoints/best_unet.pth`).**
> 
> The frontend MUST NEVER:
> - Store or import `best_unet.pth`
> - Import PyTorch or run client-side U-Net inference
> - Access filesystem paths on the server
> 
> The React frontend only uploads the Sentinel-1 SAR image file (`.tif`/`.tiff`) to `POST /api/spill/detect/upload` and receives standard JSON responses.

---

## 3. Local Development Setup for Member 5

### 3.1 Backend Server Startup

From the repository root on branch `integration-backend`:

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Start Uvicorn dev server
python -m uvicorn backend.main:app --reload --port 8000
```

- **Backend API Base URL**: `http://127.0.0.1:8000/api`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

### 3.2 Frontend Environment Configuration

In your `frontend/` directory, configure `.env.development`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

In your React code (`frontend/src/App.jsx` or API utility module):

```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
```

### 3.3 CORS Configuration

The backend is configured with FastAPI CORS middleware allowing local development origins:
- `http://localhost:5173`
- `http://localhost:3000`
- `http://127.0.0.1:5173`
- `http://127.0.0.1:3000`

---

## 4. Step-by-Step Frontend Integration Walkthrough

### STEP 1: User Selects & Uploads a Sentinel-1 SAR GeoTIFF Image

The user selects a raw 1-channel Sentinel-1 SAR GeoTIFF file (`.tif` or `.tiff`).

```javascript
// Step 1 & 2: Send file to POST /api/spill/detect/upload
export async function uploadSarImage(file) {
  const formData = new FormData();
  formData.append("file", file); // Must use field name "file"

  // CRITICAL: Do NOT manually set Content-Type header when using FormData.
  // The browser automatically sets Content-Type to multipart/form-data with boundary.
  const response = await fetch(`${API_BASE_URL}/api/spill/detect/upload`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "SAR image detection failed");
  }

  return await response.json();
}
```

**Expected Response Schema (`200 OK`)**:
```json
{
  "image_id": "sentinel1_sample.tif",
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

### STEP 2: Compute Ocean Drift Backtrack Trajectory

Send the detected spill centroid coordinates and timestamp to the ocean drift engine.

```javascript
// Step 4: Call POST /api/spill/backtrack
export async function backtrackSpill(latitude, longitude, timestamp, driftHours = 6.0) {
  const response = await fetch(`${API_BASE_URL}/api/spill/backtrack`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spill_location: { latitude, longitude },
      timestamp: timestamp || new Date().toISOString(),
      drift_hours: driftHours
    })
  });

  if (!response.ok) throw new Error("Backtrack calculation failed");
  return await response.json();
}
```

**Expected Response Schema (`200 OK`)**:
```json
{
  "spill_location": { "latitude": 28.912903, "longitude": -89.028632 },
  "detection_timestamp": "2026-08-30T00:35:00Z",
  "estimated_source_area": {
    "center": { "latitude": 28.912903, "longitude": -89.028632 },
    "radius_km": 5.0,
    "boundary_polygon": [ ... ]
  },
  "trajectory": [
    { "timestamp": "2026-08-30T00:35:00Z", "latitude": 28.912903, "longitude": -89.028632, "uncertainty_radius_km": 1.5 },
    { "timestamp": "2026-08-29T18:35:00Z", "latitude": 28.895000, "longitude": -89.010000, "uncertainty_radius_km": 5.0 }
  ],
  "timestamp": "2026-08-30T00:35:01Z",
  "disclaimer": "M2 DRIFT MODEL: Backtrack trajectory calculated using Member 2 Hydrodynamic Ocean Drift Engine."
}
```

---

### STEP 3: Query Candidate Suspect Vessels from AIS Stream

Pass the estimated origin center location to search for nearby AIS vessel tracks.

```javascript
// Step 6: Call POST /api/ais/candidates
export async function getAisCandidates(originLat, originLon, timestamp, radiusKm = 50.0) {
  const response = await fetch(`${API_BASE_URL}/api/ais/candidates`, {
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

  if (!response.ok) throw new Error("AIS candidate query failed");
  const data = await response.json();
  return data.candidates;
}
```

---

### STEP 4: Rank Suspect Vessels using 4-Factor Vessel Attribution Engine

Send candidate vessels and origin location to compute suspect risk scores and rankings.

```javascript
// Step 7: Call POST /api/vessels/rank
export async function rankSuspectVessels(originLat, originLon, timestamp, candidates) {
  const response = await fetch(`${API_BASE_URL}/api/vessels/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spill_source: { latitude: originLat, longitude: originLon },
      timestamp: timestamp,
      candidate_vessels: candidates
    })
  });

  if (!response.ok) throw new Error("Vessel attribution ranking failed");
  return await response.json();
}
```

**Expected Response Schema (`200 OK`)**:
```json
{
  "ranked_vessels": [
    {
      "vessel": {
        "mmsi": "419000101",
        "vessel_name": "OCEAN TITAN",
        "vessel_type": "Crude Oil Tanker",
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
        { "factor_name": "Distance Proximity Score", "score": 0.9520, "description": "Closest approach 0.8 km" },
        { "factor_name": "Time Proximity Score", "score": 0.9000, "description": "Time diff 20 mins" }
      ]
    }
  ],
  "total_ranked": 1,
  "ranked_at": "2026-08-30T00:35:03Z",
  "disclaimer": "ATTRIBUTION RESULT: Vessel rankings and correlation scores calculated using Member 4 Vessel Attribution Engine."
}
```

---

## 5. UI Data Mapping Table for Leaflet Maps & Charts

| Backend Response JSON Field | UI Element / Layer | Implementation Note |
| :--- | :--- | :--- |
| `res.spill_polygon` | `<Polygon />` | Map array of `{latitude, longitude}` to Leaflet `[[lat, lon], ...]` |
| `res.estimated_source_area.center` | `<Marker />` / `<Circle />` | Draw origin center marker and uncertainty circle (`radius_km * 1000` meters) |
| `res.trajectory` | `<Polyline />` | Draw backward drift path line connecting historical positions |
| `res.ranked_vessels` | `<Marker />` list | Render vessel icons on map. Highlight Rank 1 in Red |
| `v.risk_score` / `v.final_score` | Progress Bar / Badge | Display risk percentage `(v.risk_score * 100).toFixed(1) + "%"` |
| `v.attribution_factors` | Breakdown Accordion | Display 4-factor scoring breakdown table |

---

## 6. Frontend Error Handling Guidelines

| HTTP Code | Error Cause | Recommended UI Display Message |
| :--- | :--- | :--- |
| `400 Bad Request` | Invalid file extension (not `.tif`/`.tiff`) or corrupt GeoTIFF | `"Invalid File: Only 1-channel Sentinel-1 GeoTIFF (.tif, .tiff) files are supported."` |
| `422 Unprocessable Entity` | Missing form field or invalid lat/lon parameters | `"Invalid Parameters: Please check input coordinates and parameters."` |
| `500 Internal Server Error` | Server-side execution exception | `"Server Error: Unable to process imagery. Please try again later."` |
| `TypeError: Failed to fetch` | Uvicorn server is offline | `"Connection Error: Unable to reach backend server at http://127.0.0.1:8000."` |

---

## 7. Automated End-to-End Test Verification

Member 5 can verify that the entire backend pipeline is operational by running:

```bash
python -m pytest backend/tests drift_tests ais/tests -v
```

**Result Baseline**: **175 passed** (100% pass rate).
