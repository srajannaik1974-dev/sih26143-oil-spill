# AIS / Vessel Tracking Module — Member 3
**SIH 2026 — Problem Statement 26143: Oil Spill Investigation**

---

## 1. AIS Role in the SIH 26143 System

**Automatic Identification System (AIS)** is a maritime transponder tracking protocol broadcasting ship identification, GPS coordinates, UTC timestamps, Speed Over Ground (SOG), and Course/Heading telemetry.

This project uses a single canonical implementation under the `ais/` package as the source of truth. The duplicate legacy `ais/src/` tree has been removed so the AIS module consists of one authoritative implementation only.

In the **SIH 26143 Oil Spill Investigation System**, this module operates independently as **Member 3's core responsibility**. Given a probable spill origin (coordinates) and estimated release time window (from Member 2's drift & backtracking module), this module:
1. Ingests raw historical AIS data (CSV or JSON).
2. Cleans, validates, deduplicates, and normalizes timestamps to UTC.
3. Reconstructs each vessel's chronological trajectory.
4. Filters vessels within the estimated release window and search radius using Great-Circle Haversine calculations.
5. Emits a clean, standardized candidate output payload for Member 4's Attribution module.

### Canonical Architecture Boundary

The authoritative AIS implementation is:
- `ais/loader.py`
- `ais/cleaner.py`
- `ais/trajectory.py`
- `ais/filters.py`
- `ais/schemas.py`
- `ais/synthetic_generator.py`

The AIS module is strictly responsible for:
- data ingestion
- validation and normalization
- vessel trajectory construction
- time-window + distance-window filtering
- candidate vessel output for downstream attribution

The AIS module does not perform:
- spill detection
- drift/backtracking
- environmental modelling
- vessel guilt ranking
- attribution scoring
- final responsibility assignment

```text
       AIS Dataset (CSV / JSON)
                  ↓
           [ais/loader.py]
         Column normalization
                  ↓
          [ais/cleaner.py]
    Deduplication & UTC parsing
                  ↓
        [ais/trajectory.py]
   Vessel trajectory grouping
                  ↓
          [ais/filters.py]
    Time window & Radius search
                  ↓
          Candidate Vessels
                  ↓
         Attribution Module (Member 4)
```

> [!IMPORTANT]
> **Scope Boundary**: This module does **not** perform SAR oil-spill detection, ML segmentation, oil drift backtracking, final culpability attribution, or frontend rendering. It strictly focuses on AIS data validation, trajectory building, and spatio-temporal candidate vessel filtering.

---

## 2. Canonical Data Contract & Field Specifications

All internal data models and outputs strictly enforce the following canonical field names:

| Field | Type | Description | Mandatory |
| :--- | :--- | :--- | :--- |
| `vessel_id` | `str` | Unique vessel identifier (e.g. MMSI) | **Yes** |
| `timestamp` | `datetime` / `str` | ISO 8601 UTC timestamp (e.g. `"2026-08-20T14:30:00Z"`) | **Yes** |
| `latitude` | `float` | Geographic latitude in decimal degrees $[-90.0, 90.0]$ | **Yes** |
| `longitude` | `float` | Geographic longitude in decimal degrees $[-180.0, 180.0]$ | **Yes** |
| `speed_knots` | `float` / `None` | Speed Over Ground (knots) | Optional |
| `heading_deg` | `float` / `None` | True heading / course in degrees $[0.0, 360.0]$ | Optional |

### Automatic Alias Mapping
When loading raw files, the loader automatically maps common industry aliases to the canonical names:
* `mmsi`, `ship_id`, `imo` $\rightarrow$ `vessel_id`
* `datetime`, `time`, `base_datetime` $\rightarrow$ `timestamp`
* `lat`, `lat_deg`, `y` $\rightarrow$ `latitude`
* `lon`, `lng`, `lon_deg`, `x` $\rightarrow$ `longitude`
* `sog`, `speed`, `speed_kts` $\rightarrow$ `speed_knots`
* `cog`, `heading`, `course` $\rightarrow$ `heading_deg`

---

## 3. Public Functions Available to Other Modules

The module exports a modular Python API via `ais`:

```python
from ais import (
    # Ingestion & Cleaning
    load_ais_file,
    load_ais_csv,
    load_ais_json,
    clean_ais_records,
    parse_utc_timestamp,
    
    # Trajectory Building
    build_trajectories,
    get_vessel_trajectory,
    
    # Filtering & Candidate Search
    haversine_distance,
    filter_by_time_window,
    filter_by_distance,
    find_candidate_vessels,
    
    # Synthetic Generator
    generate_synthetic_ais,
    save_synthetic_ais_csv,
    
    # Dataclasses
    AISPoint,
    VesselTrajectory,
    CandidateVessel,
    CandidateOutput,
    SpillQuery,
)
```

### Key Function Signatures

#### `load_ais_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]`
Loads raw CSV or JSON files and maps headers to canonical field names.

#### `clean_ais_records(raw_records: List[Dict[str, Any]]) -> List[AISPoint]`
Validates coordinates, deduplicates identical `(vessel_id, timestamp)` records, normalizes timestamps to UTC, and sorts by vessel and time.

#### `build_trajectories(records: List[AISPoint]) -> Dict[str, VesselTrajectory]`
Groups points by `vessel_id` and ensures chronological point ordering.

#### `get_vessel_trajectory(trajectories: Dict[str, VesselTrajectory], vessel_id: str) -> Optional[VesselTrajectory]`
Retrieves a specific vessel's complete position sequence.

#### `find_candidate_vessels(...) -> CandidateOutput`
```python
find_candidate_vessels(
    trajectories: Union[Dict[str, VesselTrajectory], List[AISPoint]],
    origin_lat: float,
    origin_lon: float,
    release_start: Union[str, datetime],
    release_end: Union[str, datetime],
    search_radius_km: float = 10.0,
    spill_id: str = "spill_001",
) -> CandidateOutput
```

---

## 4. Input & Output Formats

### Spill Input Query (from Drift/Backtracking Module)
```json
{
  "spill_id": "SPILL_2026_01",
  "origin_lat": 12.8500,
  "origin_lon": 74.7200,
  "release_start": "2026-08-20T14:00:00Z",
  "release_end": "2026-08-20T15:00:00Z",
  "search_radius_km": 10.0
}
```

### Candidate Output Payload (to Attribution Module)
```json
{
  "spill_id": "SPILL_2026_01",
  "candidates": [
    {
      "vessel_id": "123456789",
      "closest_distance_km": 0.311,
      "closest_timestamp": "2026-08-20T14:29:00Z",
      "latitude": 12.852,
      "longitude": 74.718,
      "speed_knots": 12.4,
      "heading_deg": 142.0
    },
    {
      "vessel_id": "987654321",
      "closest_distance_km": 2.984,
      "closest_timestamp": "2026-08-20T14:25:00Z",
      "latitude": 12.875,
      "longitude": 74.73,
      "speed_knots": 14.1,
      "heading_deg": 208.0
    }
  ]
}
```

---

## 5. Synthetic AIS Generator for Demonstrations

To generate deterministic synthetic AIS data labeled as `DEMO/SYNTHETIC`:

```python
from ais import generate_synthetic_ais, save_synthetic_ais_csv

# Generate synthetic observations around a spill origin
points = generate_synthetic_ais(
    origin_lat=12.8500,
    origin_lon=74.7200,
    release_start="2026-08-20T14:00:00Z",
    release_end="2026-08-20T15:00:00Z",
    search_radius_km=10.0,
)

# Save to CSV
save_synthetic_ais_csv("ais/data/synthetic_demo.csv", points)
```

The generator creates 5 distinct test scenarios:
1. `VESSEL_001`: Direct close pass (~0.31 km) during the release window.
2. `VESSEL_002`: Moderate proximity (~3.0 km) during the release window.
3. `VESSEL_003`: Far transit (~15.5 km away, outside 10 km radius).
4. `VESSEL_004`: Close transit (~0.2 km away) but 4 hours late (outside time window).
5. `VESSEL_005`: Boundary vessel (~6.5 km away) with missing speed & heading data.

---

## 6. How to Run & Test

### Run All Unit and Integration Tests
```powershell
python -m unittest discover -s ais/tests -v
```

### Run End-to-End Pipeline in Python
```python
from ais import (
    load_ais_file,
    clean_ais_records,
    build_trajectories,
    find_candidate_vessels,
)
import json

# 1. Load & clean
raw = load_ais_file("ais/data/synthetic_ais.csv")
cleaned = clean_ais_records(raw)

# 2. Build trajectories
trajectories = build_trajectories(cleaned)

# 3. Find candidates near probable origin
candidates_result = find_candidate_vessels(
    trajectories=trajectories,
    origin_lat=12.8500,
    origin_lon=74.7200,
    release_start="2026-08-20T14:00:00Z",
    release_end="2026-08-20T15:00:00Z",
    search_radius_km=10.0,
    spill_id="SPILL_2026_01",
)

print(json.dumps(candidates_result.to_dict(), indent=2))
```

---

## 7. Current Status and Remaining Work

```text
AIS software module: COMPLETE
AIS filtering/ranking: COMPLETE
Tests: 41 passing
Mock/synthetic testing: COMPLETE

Real AIS dataset integration: PENDING
Member 1 real spill-output integration: PENDING
Real-world end-to-end validation: PENDING
```

---

## 8. Limitations & Operational Notes

1. **Missing Data Handling**: If `speed_knots` or `heading_deg` are missing in raw telemetry, they are stored as `None` and never invented.
2. **Dynamic Querying**: No coordinates, timestamps, or thresholds are hardcoded; all parameters are passed dynamically per incident query.
3. **Zero External Runtime Dependencies**: Implemented in pure Python 3.8+ standard library.
