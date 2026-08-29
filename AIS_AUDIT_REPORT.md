# AIS / Vessel-Tracking Module Audit Report

## Scope

This audit is inspection-only. No files were modified, created, renamed, or deleted beyond this report file, which is explicitly requested as a single consolidated document.

This audit covers only the AIS / vessel-tracking module responsibilities and does not include unrelated project modules such as satellite image processing, spill detection, drift/backtracking, attribution/ranking, frontend, or unrelated backend functionality.

---

# 1. CURRENT PROJECT STRUCTURE

The repository currently contains a project-level AIS module under the `ais` folder, not the assignment’s expected `backend/ais` layout.

Relevant repository structure:

- `README.md`
- `ais/`
  - `__init__.py`
  - `README.md`
  - `requirements.txt`
  - `cleaner.py`
  - `filters.py`
  - `loader.py`
  - `schemas.py`
  - `synthetic_generator.py`
  - `trajectory.py`
  - `src/`
    - `__init__.py`
    - `config.py`
    - `distance.py`
    - `filtering.py`
    - `loader.py`
    - `pipeline.py`
    - `ranking.py`
  - `data/`
    - `mock_spill.json`
    - `synthetic_ais.csv`
  - `output/`
    - `ais_result.json`
  - `tests/`
    - `test_cleaner.py`
    - `test_distance.py`
    - `test_filtering.py`
    - `test_filters.py`
    - `test_loader.py`
    - `test_pipeline.py`
    - `test_ranking.py`
    - `test_synthetic.py`
    - `test_trajectory.py`

Important observation:
- There is no `backend/ais` folder in the current repo.
- There is no top-level `data/ais/` with `raw/`, `processed/`, and `synthetic/` subfolders.
- The repo uses `ais/` plus `ais/src/` rather than the assignment-requested structure.

---

# 2. AIS FILES FOUND

File: `ais/loader.py`
Purpose: CSV/JSON AIS ingestion, alias mapping, required-column validation, raw normalization
Current status: Functional for CSV and JSON but mixed with older legacy naming conventions
Dependencies: csv, json, pathlib, typing
Used by: `ais/tests/test_loader.py`, `ais/src/pipeline.py`

File: `ais/cleaner.py`
Purpose: validation, deduplication, UTC timestamp normalization, coordinate checks
Current status: Mostly complete for AIS cleaning logic
Dependencies: datetime, typing, `ais/schemas.py`
Used by: `ais/tests/test_cleaner.py`

File: `ais/trajectory.py`
Purpose: grouping AIS observations per vessel and sorting by time
Current status: Complete for grouping and chronological ordering
Dependencies: defaultdict, `ais/schemas.py`
Used by: `ais/tests/test_trajectory.py`

File: `ais/filters.py`
Purpose: Haversine distance, time filtering, distance filtering, candidate search
Current status: Functional but not fully aligned with assignment output contract
Dependencies: math, datetime, `ais/cleaner.py`, `ais/schemas.py`, `ais/trajectory.py`
Used by: `ais/tests/test_filters.py`

File: `ais/synthetic_generator.py`
Purpose: synthetic demo data generation with vessel scenarios
Current status: Present and usable for demo generation
Dependencies: csv, datetime, pathlib, `ais/cleaner.py`, `ais/schemas.py`
Used by: `ais/tests/test_synthetic.py`

File: `ais/schemas.py`
Purpose: canonical dataclasses for `AISPoint`, `VesselTrajectory`, `CandidateVessel`, `CandidateOutput`
Current status: Implemented for a standard internal model
Dependencies: dataclasses, datetime, typing
Used by: `ais/cleaner.py`, `ais/filters.py`, `ais/trajectory.py`

File: `ais/src/distance.py`
Purpose: independent Haversine implementation
Current status: Correct distance math
Dependencies: math
Used by: `ais/src/filtering.py`, `ais/tests/test_distance.py`

File: `ais/src/filtering.py`
Purpose: filter AIS records by spill time and distance around spill coordinates
Current status: Implemented and works
Dependencies: `ais/src/config.py`, `ais/src/distance.py`, `ais/src/loader.py`
Used by: `ais/src/pipeline.py`

File: `ais/src/loader.py`
Purpose: load spill incident JSON + validate CSV AIS records with row rejection logs
Current status: Partial/legacy
Dependencies: csv, json, dataclass, datetime, typing
Used by: `ais/src/pipeline.py`, `ais/tests/test_pipeline.py`

File: `ais/src/pipeline.py`
Purpose: end-to-end candidate pipeline and CLI
Current status: Broken/incorrect relative to the agreed AIS-only contract
Dependencies: `ais/src/config.py`, `ais/src/filtering.py`, `ais/src/loader.py`, `ais/src/ranking.py`
Used by: CLI and tests

File: `ais/src/ranking.py`
Purpose: group-by-vessel and score/rank candidates
Current status: Broken/incorrect relative to AIS-only scope
Dependencies: `ais/src/config.py`, `ais/src/loader.py`
Used by: `ais/src/pipeline.py`, `ais/tests/test_ranking.py`

File: `ais/src/config.py`
Purpose: default search radius, time window, ranking weights, ship-type heuristics
Current status: Out of scope and not assignment-compliant
Dependencies: dataclass, pathlib, typing
Used by: `ais/src/pipeline.py`, `ais/src/filtering.py`, `ais/src/ranking.py`

File: `ais/tests/test_loader.py`
Purpose: loader tests for CSV and JSON alias mapping
Current status: Passes
Used by: unit suite

File: `ais/tests/test_cleaner.py`
Purpose: cleaning, dedup, invalid coordinate handling, timestamp helpers
Current status: Passes
Used by: unit suite

File: `ais/tests/test_distance.py`
Purpose: Haversine correctness tests
Current status: Passes
Used by: unit suite

File: `ais/tests/test_filters.py`
Purpose: raw module filtering and candidate generation tests
Current status: Passes
Used by: unit suite

File: `ais/tests/test_filtering.py`
Purpose: legacy module filtering tests
Current status: Passes
Used by: unit suite

File: `ais/tests/test_pipeline.py`
Purpose: end-to-end pipeline tests
Current status: Passes
Used by: unit suite

File: `ais/tests/test_ranking.py`
Purpose: candidate ranking and score-order tests
Current status: Aligned with older ranking heuristics, not with assignment contract
Used by: unit suite

File: `ais/tests/test_synthetic.py`
Purpose: synthetic generator validation and filtering scenario
Current status: Passes
Used by: unit suite

File: `ais/tests/test_trajectory.py`
Purpose: vessel grouping and chronological sorting tests
Current status: Passes
Used by: unit suite

File: `ais/README.md`
Purpose: module documentation
Current status: Partly useful but not aligned with the strict AIS-only scope
Dependencies: none
Used by: developers / readers

---

# 3. FEATURE STATUS

| Required Feature | Status | Existing File(s) | Evidence | Missing Work |
|---|---|---|---|---|
| 1. AIS data loader | ✅ COMPLETE | `ais/loader.py`, `ais/src/loader.py` | CSV and JSON loaders exist with required-column checks | None for basic loading |
| 2. AIS CSV support | ✅ COMPLETE | `ais/loader.py`, `ais/src/loader.py` | CSV parsing with `csv.DictReader` and validation | None |
| 3. AIS field mapping | ✅ COMPLETE | `ais/loader.py` | Maps `mmsi`, `datetime`, `lat`, `lon`, `sog`, `cog` to canonical internal names | Must remain canonical internally and consistent across repo |
| 4. Data validation | ✅ COMPLETE | `ais/cleaner.py`, `ais/src/loader.py` | Invalid coords, timestamps, missing vessel IDs are handled | Need clearer repo-wide enforcement |
| 5. Invalid coordinate handling | ✅ COMPLETE | `ais/cleaner.py`, `ais/src/loader.py`, `ais/src/distance.py` | range checks for lat/lon and exceptions | None |
| 6. Duplicate record handling | ✅ COMPLETE | `ais/cleaner.py` | dedup by `(vessel_id, timestamp)` | None |
| 7. Timestamp parsing / UTC normalization | ✅ COMPLETE | `ais/cleaner.py`, `ais/src/loader.py` | `parse_utc_timestamp` normalizes to UTC | None |
| 8. Vessel grouping | ✅ COMPLETE | `ais/trajectory.py`, `ais/src/ranking.py` | group-by-vessel exists | Need consistent field names |
| 9. Chronological sorting | ✅ COMPLETE | `ais/trajectory.py`, `ais/src/ranking.py` | sorted by timestamp before trajectory creation | None |
| 10. Vessel trajectory generation | ✅ COMPLETE | `ais/trajectory.py` | `VesselTrajectory` objects created and serialized | None |
| 11. Single-vessel trajectory retrieval | ✅ COMPLETE | `ais/trajectory.py` | `get_vessel_trajectory()` exists | None |
| 12. Geographic distance calculation | ✅ COMPLETE | `ais/filters.py`, `ais/src/distance.py` | Haversine formula with validation | None |
| 13. Time-window filtering | ✅ COMPLETE | `ais/filters.py`, `ais/src/filtering.py` | time-window checks around spill window | None |
| 14. Radius/distance filtering | ✅ COMPLETE | `ais/filters.py`, `ais/src/filtering.py` | radius checks by geographic distance | None |
| 15. Candidate-vessel generation | ⚠️ BROKEN / INCORRECT | `ais/filters.py`, `ais/src/pipeline.py`, `ais/src/ranking.py` | Works in a legacy pipeline but output is not the agreed project contract | Need exact payload: `spill_id`, `origin`, `release_window`, `search_radius_km`, `candidate_vessels` |
| 16. Synthetic AIS generator | ✅ COMPLETE | `ais/synthetic_generator.py` | demo generator exists | Must align with standard file layout and API |
| 17. Synthetic continuous vessel trajectories | ✅ COMPLETE | `ais/synthetic_generator.py` | continuous multi-point vessel tracks generated | None |
| 18. Synthetic scenario near spill origin | ✅ COMPLETE | `ais/synthetic_generator.py` | vessels near, moderate, far, close-but-outside-time, unrelated exist | None |
| 19. Unit tests | ✅ COMPLETE | `ais/tests` | many unit tests exist and pass | Need contract tests aligned with assignment |
| 20. Documentation | 🟡 PARTIAL | `ais/README.md`, `README.md` | docs exist but scope is broader than AIS-only responsibility | Needs cleanup to AIS-only specification |

---

# 4. WHAT WE HAVE ALREADY COMPLETED

The repository genuinely already does the following:

- Loads AIS CSV/JSON data with field alias mapping:
  - `ais/loader.py`
- Validates required fields and rejects malformed records:
  - `ais/cleaner.py`
  - `ais/src/loader.py`
- Converts timestamps to UTC-aware datetimes:
  - `ais/cleaner.py`
  - `ais/src/loader.py`
- Removes invalid lat/lon values and duplicates:
  - `ais/cleaner.py`
- Groups records by vessel and sorts by time:
  - `ais/trajectory.py`
- Builds chronological vessel trajectories:
  - `ais/trajectory.py`
- Computes geographic distance using Haversine:
  - `ais/filters.py`
  - `ais/src/distance.py`
- Filters AIS observations by time and radius:
  - `ais/filters.py`
  - `ais/src/filtering.py`
- Produces candidate vessel lists from spatio-temporal matches:
  - `ais/filters.py`
  - `ais/src/pipeline.py`
- Generates synthetic vessel trajectories:
  - `ais/synthetic_generator.py`
- Provides unit tests for AIS logic:
  - `ais/tests`

The test evidence is clear: running the suite with Python’s unittest runner produced a passing result:
- 44 tests ran
- result: OK

---

# 5. WHAT IS PARTIALLY COMPLETE

The system is partially complete in two main areas:

1. Common AIS pipeline is present, but the repo has duplicate implementations
- There are two parallel stacks:
  - `ais/` for simplified canonical AIS logic
  - `ais/src/` for a ranking-heavy pipeline
- This creates a split API and inconsistent semantics.

2. Candidate generation is implemented, but not to the assignment’s required output contract
- `ais/filters.py` returns a `CandidateOutput` dataclass with a `candidates` list.
- `ais/src/pipeline.py` returns a structured analysis payload with ranking metadata.
- Neither matches the assignment’s required output structure:
  - `spill_id`
  - `origin`
  - `release_window`
  - `search_radius_km`
  - `candidate_vessels`
- The assignment explicitly says the AIS module is not responsible for guilt scoring or attribution ranking. The repo’s scoring logic is therefore beyond the allowed scope.

3. Documentation exists, but the project scope is not cleanly enforced
- `ais/README.md` is useful but broader than the agreed AIS responsibility.
- It references the attribution/ranking workflow more than it should.

---

# 6. WHAT HAS NOT BEEN IMPLEMENTED

These are the required AIS features that are missing or not aligned with the agreed specification:

- No exact assignment layout:
  - no `backend/ais` directory
  - no top-level `data/ais/raw`, `data/ais/processed`, `data/ais/synthetic`
- No single canonical assignment API that matches:
  - `load_ais_data()`
  - `clean_ais_data()`
  - `get_vessel_trajectory()`
  - `calculate_distance_km()`
  - `filter_vessels_by_time()`
  - `filter_vessels_by_distance()`
  - `get_candidate_vessels()`
  - `generate_synthetic_ais()`
- Candidate output not in the exact required structure
- No explicit separation between:
  - AIS filtering / candidate generation
  - attribution / scoring module
- No strict enforcement that all internal fields use:
  - `vessel_id`
  - `timestamp`
  - `latitude`
  - `longitude`
  - `speed_knots`
  - `heading_deg`
- No explicit contract ensuring the module accepts:
  - `origin_lat`
  - `origin_lon`
  - `release_start`
  - `release_end`
  as direct function inputs in the candidate-generation stage
- No clean repository-wide public AIS facade that is free of duplicate legacy structures
- No final documentation blocking the ranking/attribution responsibilities from AIS

---

# 7. WHAT IS INCORRECT

These are the major issues found during inspection:

- Wrong field naming in the legacy pipeline
  - `ais/src/loader.py` uses `mmsi`, `sog`, `cog`, `ship_name`, `ship_type`, etc.
  - The agreed AIS contract says the internal standard is `vessel_id`, `speed_knots`, `heading_deg`, etc.
  - This is a direct mismatch.

- Ranking / scoring logic is outside AIS scope
  - `ais/src/config.py` defines score weights and ship-type heuristics.
  - `ais/src/ranking.py` computes a candidate score and ranks vessels.
  - This contradicts the requirement that AIS should only return candidate vessels and not decide guilt.

- The repo mixes two models of the same problem
  - canonical AIS layer in `ais/`
  - legacy ranking pipeline in `ais/src/`
  - this causes conflicting conventions and uncertain ownership.

- Hardcoded/legacy pipeline assumptions
  - `ais/src/pipeline.py` loads a spill JSON and computes ranked candidate output, which is not the assignment contract.
  - The task requires direct input parameters for origin and release window, not a spill JSON file.

- Path/layout mismatch
  - The repo uses `ais/data` instead of the required `data/ais` layout.
  - This does not match the requirement for raw/processed/synthetic data separation.

- Output contract mismatch
  - The required output structure is a dictionary with `origin` and `release_window`.
  - Existing output is closer to `{"spill_id": ..., "candidates": [...]}` or ranking payloads with extra score metadata.

- Missing strict AIS-only boundary
  - The repo’s docs and ranking pipeline imply responsibility assessment and vessel scoring, which is not allowed for this team member.

- Duplicate implementations create integration risk
  - The same concept exists in `ais/filters.py` and `ais/src/filtering.py`.
  - This can lead to inconsistent behavior depending on which API is used.

---

# 8. TEST STATUS

Existing tests:
- `ais/tests/test_loader.py`
- `ais/tests/test_cleaner.py`
- `ais/tests/test_distance.py`
- `ais/tests/test_filters.py`
- `ais/tests/test_filtering.py`
- `ais/tests/test_pipeline.py`
- `ais/tests/test_ranking.py`
- `ais/tests/test_synthetic.py`
- `ais/tests/test_trajectory.py`

What they test:
- CSV/JSON loading
- column mapping aliases
- missing fields/errors
- valid/invalid coordinates
- timestamps and UTC parsing
- duplicates
- filtering by time/distance
- trajectory grouping
- candidate generation
- synthetic dataset generation and filtering
- scoring/ranking logic

What is missing:
- Assignment-specific contract tests for exact output structure
- Tests for direct `origin_lat`, `origin_lon`, `release_start`, `release_end` function inputs
- Tests for required canonical names across the full repo
- Tests verifying the module remains AIS-only and does not produce attribution score outputs

Whether tests currently pass:
- From inspection and execution, the current suite passes.
- Command used: `python -m unittest discover -s ais/tests -v`
- Result: 44 tests ran and the result was `OK`.

---

# 9. SYNTHETIC AIS STATUS

Does a synthetic generator exist?
- Yes: `ais/synthetic_generator.py`

Does it generate continuous trajectories?
- Yes. It creates multiple time-ordered points per vessel.

Does it use the correct field names?
- Yes, for the synthetic generator itself it uses:
  - `vessel_id`
  - `timestamp`
  - `latitude`
  - `longitude`
  - `speed_knots`
  - `heading_deg`

Does it create vessels with different relationships to the spill origin?
- Yes. The generator includes scenarios analogous to:
  - near, moderate, far, close-but-outside-window, unrelated
- This is visible in the synthetic logic and tests.

Can it be used for our hackathon demo?
- Yes, as a demo dataset it is usable.
- But it is not yet fully aligned to the assignment’s exact directory/layout and function contract.

What is missing?
- Standardized location under `data/ais/synthetic`
- Direct assignment-compatible generation arguments
- Standardized output contract alongside the real AIS loader
- A public package-level function consistent with the agreed API

---

# 10. INTEGRATION STATUS

Can the current AIS module receive:
- `origin_lat`
- `origin_lon`
- `release_start`
- `release_end`

and produce candidate vessels in the required format?

Answer: partially, but not in the final agreed form.

What works:
- `ais/filters.py` has functions that accept origin and release window values and compute candidate vessels.
- `ais/src/filtering.py` does time-distance filtering around a spill incident.

What does not fully work:
- The repository does not expose one clean public function matching the assignment contract.
- The output is not exactly the required `candidate_vessels` dict format.
- It can produce candidate-like results, but not yet the downstream attribution-ready structure expected by the assignment.

Can the output be consumed by a future attribution module?
- Partially yes, but not exactly in the assignment format.
- The current output includes extra ranking metadata and legacy vessel metadata not expected by Member 4.
- It is therefore only partially compatible with attribution, not fully.

---

# 11. FINAL OUTCOME

Currently, the AIS module can do core AIS ingestion, cleaning, trajectory grouping, and distance/time filtering, but it cannot yet do the exact assignment-specified candidate-vessel workflow end-to-end in the required contract.

What currently works:
- Loading AIS CSV/JSON
- Alias normalization
- Cleaning and validation
- UTC timestamp parse/normalize
- valid coordinate checks
- duplicate removal
- grouping by vessel
- chronological sorting
- trajectory generation
- Haversine distance calculation
- time and distance filtering
- synthetic demonstration generation
- unit tests passing

What does not currently work in the assignment’s required form:
- exact public API
- exact candidate output structure
- no clean AIS-only boundary
- scoring/ranking remains in the module
- legacy duplicate architecture remains

---

# 12. REMAINING WORK

P0 = Must complete
- Unify the repo around a single AIS-only architecture
- Remove duplicate implementation between `ais/` and `ais/src/`
- Standardize the field contract to:
  - `vessel_id`
  - `timestamp`
  - `latitude`
  - `longitude`
  - `speed_knots`
  - `heading_deg`
- Replace legacy ranking output with the exact candidate-vessel payload required by the assignment
- Remove any guilt/attribution-scoring logic from AIS responsibilities
- align the directory structure to the expected data layout

P1 = Important
- Add direct function-entry contract for:
  - `origin_lat`
  - `origin_lon`
  - `release_start`
  - `release_end`
- Ensure the output is directly consumable by Member 4 without extra conversion
- Align documentation to the AIS-only scope
- Ensure all tests cover the exact assignment contract

P2 = Optional
- Improve synthetic generator ergonomics and demo config knobs
- Add richer real-AIS alias coverage
- Add stronger documentation for future JSON support and actual real-world usage

Summary: the repository already has a useful AIS foundation, but it is not yet fully aligned to the agreed SIH contract and is still carrying legacy ranking logic and duplicate architecture.
