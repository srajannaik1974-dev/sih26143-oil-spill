# SIH 2026 — Frontend API Code Examples

This document provides standalone JavaScript code snippets for Member 5 (Frontend UI) to integrate all 5 active Member 6 API endpoints into React/Vite components.

---

## 1. Setup & Configuration

```javascript
// src/api/config.js
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";
```

---

## 2. API Helper Functions (`src/api/spillService.js`)

```javascript
import { API_BASE_URL } from "./config";

/**
 * 1. Health Check
 */
export async function checkBackendHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) throw new Error("Backend server is unreachable");
  return await response.json();
}

/**
 * 2. Upload Sentinel-1 GeoTIFF Image (.tif/.tiff) for Real M1 U-Net Inference
 * 
 * IMPORTANT: Do NOT manually set 'Content-Type' header when passing FormData;
 * the browser will automatically append the boundary parameter.
 */
export async function uploadSentinelImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/spill/detect/upload`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Failed to analyze satellite imagery");
  }
  return await response.json();
}

/**
 * 3. Backtrack Ocean Drift Trajectory (M2)
 */
export async function backtrackDrift(lat, lon, timestamp, driftHours = 6.0) {
  const response = await fetch(`${API_BASE_URL}/spill/backtrack`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spill_location: { latitude: lat, longitude: lon },
      timestamp: timestamp || new Date().toISOString(),
      drift_hours: driftHours
    })
  });

  if (!response.ok) throw new Error("Failed to calculate ocean backtrack");
  return await response.json();
}

/**
 * 4. Query AIS Candidate Vessels (M3)
 */
export async function getAisCandidates(originLat, originLon, timestamp, radiusKm = 50.0) {
  const response = await fetch(`${API_BASE_URL}/ais/candidates`, {
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

  if (!response.ok) throw new Error("Failed to query AIS candidates");
  const data = await response.json();
  return data.candidates;
}

/**
 * 5. Rank Suspect Vessels using 4-Factor Attribution Scorer (M4)
 */
export async function rankSuspectVessels(originLat, originLon, timestamp, candidates) {
  const response = await fetch(`${API_BASE_URL}/vessels/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spill_source: { latitude: originLat, longitude: originLon },
      timestamp: timestamp,
      candidate_vessels: candidates
    })
  });

  if (!response.ok) throw new Error("Failed to rank suspect vessels");
  return await response.json();
}
```

---

## 3. Complete End-to-End React Pipeline Flow (`src/components/PipelineRunner.jsx`)

```jsx
import React, { useState } from "react";
import {
  uploadSentinelImage,
  backtrackDrift,
  getAisCandidates,
  rankSuspectVessels
} from "../api/spillService";

export default function PipelineRunner() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleRunPipeline = async () => {
    if (!file) return alert("Please select a Sentinel-1 GeoTIFF (.tif) file first.");
    setLoading(true);
    setError(null);

    try {
      // Step 1: Upload & M1 Inference
      console.log("Running M1 U-Net Inference on uploaded file...");
      const spillRes = await uploadSentinelImage(file);
      
      const poly0 = spillRes.spill_polygon[0];
      const spillLat = poly0.latitude;
      const spillLon = poly0.longitude;
      const timestamp = spillRes.timestamp;

      // Step 2: M2 Hydrodynamic Drift Backtrack
      console.log("Running M2 Ocean Drift Backtracking...");
      const driftRes = await backtrackDrift(spillLat, spillLon, timestamp, 6.0);
      const originLat = driftRes.estimated_source_area.center.latitude;
      const originLon = driftRes.estimated_source_area.center.longitude;

      // Step 3: M3 AIS Candidate Query
      console.log("Running M3 AIS Candidate Search...");
      const candidates = await getAisCandidates(originLat, originLon, timestamp, 50.0);

      // Step 4: M4 Vessel Attribution Ranking
      console.log("Running M4 Vessel Attribution Ranking...");
      const rankRes = await rankSuspectVessels(originLat, originLon, timestamp, candidates);

      setResult({
        spill: spillRes,
        drift: driftRes,
        candidates: candidates,
        ranking: rankRes
      });
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>SIH 2026 — Oil Spill Detection & Vessel Attribution</h2>
      
      <input
        type="file"
        accept=".tif,.tiff"
        onChange={(e) => setFile(e.target.files[0])}
      />
      
      <button onClick={handleRunPipeline} disabled={loading || !file}>
        {loading ? "Processing Pipeline..." : "Run End-to-End Analysis"}
      </button>

      {error && <div style={{ color: "red", marginTop: "10px" }}>Error: {error}</div>}

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>Analysis Summary</h3>
          <p><strong>Spill Detected:</strong> {result.spill.spill_detected ? "YES" : "NO"}</p>
          <p><strong>Confidence:</strong> {(result.spill.confidence * 100).toFixed(2)}%</p>
          <p><strong>Estimated Area:</strong> {result.spill.estimated_area_sq_km} sq km</p>
          <p><strong>Centroid:</strong> {result.spill.spill_polygon[0].latitude}° N, {result.spill.spill_polygon[0].longitude}° E</p>

          <h3>Rank 1 Suspect Vessel</h3>
          {result.ranking.ranked_vessels.length > 0 ? (
            <div>
              <p><strong>Vessel Name:</strong> {result.ranking.ranked_vessels[0].vessel.vessel_name}</p>
              <p><strong>MMSI:</strong> {result.ranking.ranked_vessels[0].vessel.mmsi}</p>
              <p><strong>Risk Score:</strong> {(result.ranking.ranked_vessels[0].risk_score * 100).toFixed(1)}%</p>
            </div>
          ) : (
            <p>No suspect vessels identified in region.</p>
          )}
        </div>
      )}
    </div>
  );
}
```
