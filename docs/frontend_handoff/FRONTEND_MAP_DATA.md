# Leaflet Mapping Data Contracts

This document specifies how Member 5 can render spatial objects returned by the FastAPI backend on Leaflet / React-Leaflet maps (`<MapContainer>`, `<Polygon>`, `<Polyline>`, `<CircleMarker>`, `<Marker>`).

---

## 1. Oil Spill Bounding Polygon

- **Source API**: `POST /api/spill/detect` -> `res.spill_polygon`
- **Field Structure**: `Array<{ latitude: float, longitude: float }>`
- **React-Leaflet Format Conversion**:
  ```javascript
  const polygonPositions = res.spill_polygon.map(pt => [pt.latitude, pt.longitude]);
  ```
- **React-Leaflet Component**:
  ```jsx
  <Polygon
    positions={polygonPositions}
    pathOptions={{ color: '#FF0055', fillColor: '#FF0055', fillOpacity: 0.45, weight: 2 }}
  />
  ```

---

## 2. Estimated Release Origin Zone (Source Area)

- **Source API**: `POST /api/spill/backtrack` -> `res.estimated_source_area`
- **Fields**:
  - `center`: `{ latitude: float, longitude: float }`
  - `radius_km`: `float`
  - `boundary_polygon`: `Array<{ latitude: float, longitude: float }>`
- **React-Leaflet Circle Component**:
  ```jsx
  <Circle
    center={[res.estimated_source_area.center.latitude, res.estimated_source_area.center.longitude]}
    radius={res.estimated_source_area.radius_km * 1000} // Convert km to meters for Leaflet
    pathOptions={{ color: '#FFB800', fillColor: '#FFB800', fillOpacity: 0.25, dashArray: '5, 5' }}
  />
  ```

---

## 3. Backtrack Ocean Drift Trajectory Path

- **Source API**: `POST /api/spill/backtrack` -> `res.trajectory`
- **Field Structure**: `Array<{ timestamp: string, latitude: float, longitude: float, uncertainty_radius_km: float }>`
- **React-Leaflet Polyline Component**:
  ```jsx
  const trajectoryPositions = res.trajectory.map(pt => [pt.latitude, pt.longitude]);

  <Polyline
    positions={trajectoryPositions}
    pathOptions={{ color: '#00D9FF', weight: 3, dashArray: '8, 8' }}
  />
  ```

---

## 4. Candidate & Suspect Vessel Markers

- **Source API**: `POST /api/vessels/rank` -> `res.ranked_vessels`
- **Fields per Vessel**:
  - `vessel.latitude`: float
  - `vessel.longitude`: float
  - `vessel.heading_degrees`: float
  - `vessel.speed_knots`: float
  - `vessel.mmsi`: string
  - `vessel.vessel_name`: string
  - `rank`: int (1 = top suspect)
  - `risk_score`: float (0.0 to 1.0)
- **React-Leaflet CircleMarker Component**:
  ```jsx
  {res.ranked_vessels.map(item => (
    <CircleMarker
      key={item.vessel.mmsi}
      center={[item.vessel.latitude, item.vessel.longitude]}
      radius={item.rank === 1 ? 10 : 6}
      pathOptions={{
        color: item.rank === 1 ? '#FF0055' : '#00F5D4',
        fillColor: item.rank === 1 ? '#FF0055' : '#00F5D4',
        fillOpacity: 0.8
      }}
    >
      <Popup>
        <div>
          <strong>{item.vessel.vessel_name}</strong> (MMSI: {item.vessel.mmsi})<br />
          Rank: #{item.rank} | Risk: {(item.risk_score * 100).toFixed(1)}%<br />
          Type: {item.vessel.vessel_type} | Speed: {item.vessel.speed_knots} kn
        </div>
      </Popup>
    </CircleMarker>
  ))}
  ```
