"""
drift_adapter.py
================
Member 1 → Member 2 integration adapter.

Converts Member 1's spill_info dict (produced by OilSpillPredictor.get_spill_location)
into Member 2's DetectedSpillInput and calls process_detected_spill().

This module is the ONLY coupling point between the two member pipelines.
Do NOT import from ml.training here; do NOT import from src.drift in inference.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Ensure src/ is on the path so src.drift can be imported
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def run_drift_analysis(
    spill_info: Dict[str, Any],
    duration_hours: float = 6.0,
    step_minutes: float = 30.0,
) -> Optional[Any]:
    """
    Adapter: takes Member 1's spill_info dict and returns Member 2's DriftOriginOutput.

    Parameters
    ----------
    spill_info : dict
        Dictionary produced by OilSpillPredictor.get_spill_location().
        Must contain: 'latitude', 'longitude', 'timestamp' (ISO 8601 UTC string).
        Optional: 'area_km2', 'confidence'.
    duration_hours : float
        How far back in time to simulate the backward trajectory (default 6 h).
    step_minutes : float
        Backward trajectory time step in minutes (default 30 min).

    Returns
    -------
    DriftOriginOutput or None
        None is returned only if the import itself fails (graceful degradation).

    Raises
    ------
    ValueError
        If spill_info is missing required keys or timestamp cannot be parsed.
    """
    # Late import so that missing src.drift never crashes the rest of the app
    try:
        from src.drift.integration import (
            DetectedSpillInput,
            DriftOriginOutput,
            process_detected_spill,
        )
        from src.drift.environment import SyntheticEnvironmentalProvider
    except ImportError as exc:
        raise ImportError(
            f"src.drift is not available — cannot run drift analysis: {exc}"
        ) from exc

    # ── Validate required keys ─────────────────────────────────────────────
    required = ("latitude", "longitude", "timestamp")
    missing  = [k for k in required if k not in spill_info or spill_info[k] is None]
    if missing:
        raise ValueError(
            f"spill_info is missing required keys for drift analysis: {missing}"
        )

    lat  = float(spill_info["latitude"])
    lon  = float(spill_info["longitude"])

    # ── Parse timestamp — accept ISO string, keep tzinfo ───────────────────
    ts_raw = spill_info["timestamp"]
    if isinstance(ts_raw, datetime):
        ts = ts_raw
    else:
        ts_raw_str = str(ts_raw).replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts_raw_str)

    # Ensure UTC-aware
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # ── Build a unique spill_id from the date and location ─────────────────
    date_part = ts.strftime("%Y%m%d")
    spill_id  = f"SPILL_{date_part}_{abs(lat):.2f}N_{abs(lon):.2f}W"

    # ── Build DetectedSpillInput (Member 2's upstream contract) ────────────
    detected_spill = DetectedSpillInput(
        spill_id           = spill_id,
        latitude           = lat,
        longitude          = lon,
        detection_timestamp= ts,
        area_km2           = spill_info.get("area_km2"),
        confidence         = spill_info.get("confidence"),
    )

    # ── Use SyntheticEnvironmentalProvider (no external data needed) ────────
    env_provider = SyntheticEnvironmentalProvider()

    # ── Run Member 2's pipeline ─────────────────────────────────────────────
    result: DriftOriginOutput = process_detected_spill(
        detected_spill  = detected_spill,
        duration_hours  = duration_hours,
        step_minutes    = step_minutes,
        windage_factor  = 0.03,
        max_gap_minutes = 60.0,
        env_provider    = env_provider,
    )

    return result
