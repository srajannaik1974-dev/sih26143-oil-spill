"""
ais_adapter.py
==============
Member 2 (Drift Origin) → Member 3 (AIS Vessel Attribution) integration adapter.

Exposes a clean API to attribution search by feeding estimated backtracking origin
and estimated release time directly into the canonical AIS candidate identification pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Ensure project root is in path for modules
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ais.src.config import AISConfig
from ais.src.loader import load_and_validate_ais_csv
from ais.src.filtering import filter_by_time, filter_by_distance
from ais.src.ranking import group_by_vessel, rank_candidates


def run_ais_analysis(
    probable_latitude: float,
    probable_longitude: float,
    estimated_release_time: datetime | str,
    ais_csv_path: Optional[str] = None,
    search_radius_km: float = 20.0,
    time_window_minutes: float = 60.0,
    top_n_candidates: int = 5,
) -> Dict[str, Any]:
    """
    Runs the AIS candidate vessel search using estimated release coordinates and timestamp.

    Parameters
    ----------
    probable_latitude : float
        The estimated origin latitude from Member 2.
    probable_longitude : float
        The estimated origin longitude from Member 2.
    estimated_release_time : datetime or str
        The estimated release time from Member 2.
    ais_csv_path : str, optional
        Path to the AIS CSV file. Defaults to data/ais/synthetic/sih_demo_ais.csv.
    search_radius_km : float
        The spatial search radius in kilometers. Defaults to 20.0.
    time_window_minutes : float
        The temporal search window in minutes. Defaults to 60.0.
    top_n_candidates : int
        The maximum number of candidate vessels to return. Defaults to 5.

    Returns
    -------
    Dict[str, Any]
        Dictionary formatted according to the contract, containing:
        - origin: latitude, longitude, estimated_release_time
        - search_parameters: radius_km, time_window_minutes
        - candidate_vessels: list of serialized candidate vessels
        - metadata: operational query metrics
    """
    # 1. Resolve AIS CSV path
    if ais_csv_path is None:
        ais_csv_path = str(_ROOT / "data" / "ais" / "synthetic" / "sih_demo_ais.csv")

    # 2. Parse and normalize timestamp to timezone-aware UTC
    if isinstance(estimated_release_time, str):
        # Normalize 'Z' representation
        cleaned_ts = estimated_release_time.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned_ts)
    else:
        dt = estimated_release_time

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    # 3. Create module configuration instance
    config = AISConfig(
        search_radius_km=search_radius_km,
        time_window_minutes=time_window_minutes,
        top_n_candidates=top_n_candidates,
    )

    # 4. Load and validate records from CSV
    records, errors = load_and_validate_ais_csv(ais_csv_path)

    # 5. Apply temporal filtering
    time_filtered = filter_by_time(records, dt, time_window_minutes)

    # 6. Apply spatial filtering
    spatially_filtered = filter_by_distance(
        time_filtered, probable_latitude, probable_longitude, search_radius_km
    )

    # 7. Group and score candidates
    grouped = group_by_vessel(spatially_filtered, config)

    # 8. Rank candidates
    ranked = rank_candidates(grouped, top_n=top_n_candidates)

    # 9. Format response payload
    return {
        "origin": {
            "latitude": probable_latitude,
            "longitude": probable_longitude,
            "estimated_release_time": dt.isoformat(),
        },
        "search_parameters": {
            "radius_km": search_radius_km,
            "time_window_minutes": time_window_minutes,
        },
        "candidate_vessels": [cand.to_dict() for cand in ranked],
        "metadata": {
            "records_loaded": len(records),
            "records_skipped": len(errors),
            "records_after_time_filter": len(time_filtered),
            "records_after_dist_filter": len(spatially_filtered),
            "unique_vessels_found": len(grouped),
        }
    }
