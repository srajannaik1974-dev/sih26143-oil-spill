# JavaScript Fetch Examples for React Integration

Copy-pasteable standalone JavaScript `fetch()` helper functions designed for Member 5 (`frontend/src/App.jsx`).

---

## 1. Setup API Base URL Utility
Create `src/api.js` inside `frontend/src/`:

```javascript
// frontend/src/api.js
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function postJSON(endpoint, payload) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP Error ${response.status}: ${response.statusText}`);
  }

  return await response.json();
}
```

---

## 2. Detect Oil Spill (`POST /api/spill/detect`)

```javascript
export async function detectSpill(latitude, longitude, imageId = "SAT-001") {
  const payload = {
    image_id: imageId,
    timestamp: new Date().toISOString(),
    latitude: latitude,
    longitude: longitude,
  };

  const data = await postJSON("/api/spill/detect", payload);
  console.log("Spill Detection Result:", data);
  return data;
}
```

---

## 3. Backtrack Spill Trajectory & Origin (`POST /api/spill/backtrack`)

```javascript
export async function backtrackSpill(spillLat, spillLon, timestamp, driftHours = 6.0) {
  const payload = {
    spill_location: {
      latitude: spillLat,
      longitude: spillLon,
    },
    timestamp: timestamp,
    drift_hours: driftHours,
  };

  const data = await postJSON("/api/spill/backtrack", payload);
  console.log("Backtrack Result:", data);
  return data;
}
```

---

## 4. Query AIS Candidate Vessels (`POST /api/ais/candidates`)

```javascript
export async function fetchAISCandidates(originLat, originLon, timestamp, radiusKm = 50.0) {
  const payload = {
    source_latitude: originLat,
    source_longitude: originLon,
    timestamp: timestamp,
    search_radius_km: radiusKm,
    time_window_hours: 12.0,
  };

  const data = await postJSON("/api/ais/candidates", payload);
  console.log("AIS Candidates:", data.candidates);
  return data.candidates;
}
```

---

## 5. Rank Suspect Vessels (`POST /api/vessels/rank`)

```javascript
export async function rankVessels(originLat, originLon, timestamp, candidateVessels) {
  const payload = {
    spill_source: {
      latitude: originLat,
      longitude: originLon,
    },
    timestamp: timestamp,
    candidate_vessels: candidateVessels,
  };

  const data = await postJSON("/api/vessels/rank", payload);
  console.log("Ranked Suspect Vessels:", data.ranked_vessels);
  return data;
}
```

---

## 6. Complete Pipeline Orchestration Function for React Component

```javascript
export async function runFullAnalysisPipeline(detectionLat, detectionLon) {
  try {
    // 1. Detect Spill
    const detection = await detectSpill(detectionLat, detectionLon);
    if (!detection.spill_detected) {
      return { status: "NO_SPILL" };
    }

    // 2. Backtrack Drift Origin
    const drift = await backtrackSpill(detectionLat, detectionLon, detection.timestamp, 6.0);
    const originCenter = drift.estimated_source_area.center;
    const originTime = drift.detection_timestamp;

    // 3. Query AIS Candidate Vessels
    const candidates = await fetchAISCandidates(originCenter.latitude, originCenter.longitude, originTime, 50.0);

    // 4. Rank Suspect Vessels
    const ranking = await rankVessels(originCenter.latitude, originCenter.longitude, originTime, candidates);

    return {
      status: "SUCCESS",
      spillPolygon: detection.spill_polygon,
      spillAreaKm2: detection.estimated_area_sq_km,
      sourceAreaCenter: originCenter,
      sourceAreaPolygon: drift.estimated_source_area.boundary_polygon,
      backtrackTrajectory: drift.trajectory,
      candidates: candidates,
      rankedVessels: ranking.ranked_vessels,
    };
  } catch (error) {
    console.error("Pipeline Analysis Error:", error);
    throw error;
  }
}
```
