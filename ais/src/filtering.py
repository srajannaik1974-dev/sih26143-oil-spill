"""Backward-compatible filter wrappers for legacy ais.src imports.

The authoritative implementation lives in the canonical ais.filters module, which is
responsible only for AIS temporal/spatial candidate filtering.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence

from ais.cleaner import parse_utc_timestamp
from ais.filters import filter_by_distance as canonical_filter_by_distance
from ais.filters import filter_by_time_window, haversine_distance
from ais.schemas import AISPoint


def _to_canonical_point(record: Any) -> AISPoint:
    """Convert a legacy AISRecord-like object into the canonical AISPoint shape."""
    vessel_id = getattr(record, "mmsi", None)
    if vessel_id is None:
        vessel_id = getattr(record, "vessel_id", None)
    if vessel_id is None:
        raise ValueError("Legacy AIS record is missing vessel identifier")

    return AISPoint(
        vessel_id=str(vessel_id),
        timestamp=parse_utc_timestamp(getattr(record, "timestamp")),
        latitude=float(record.latitude),
        longitude=float(record.longitude),
        speed_knots=getattr(record, "sog", None),
        heading_deg=getattr(record, "heading", getattr(record, "cog", None)),
    )


def _set_legacy_metrics(record: Any, distance_km: float = None, time_difference_minutes: float = None) -> Any:
    if distance_km is not None:
        record.distance_km = round(float(distance_km), 3)
    if time_difference_minutes is not None:
        record.time_difference_minutes = round(float(time_difference_minutes), 2)
    return record


def filter_by_time(records: Sequence[Any], spill_time: datetime, window_minutes: float) -> List[Any]:
    """Legacy compatibility wrapper around the canonical time-window filtering."""
    start = spill_time - timedelta(minutes=window_minutes)
    end = spill_time + timedelta(minutes=window_minutes)
    filtered = filter_by_time_window([_to_canonical_point(r) for r in records], start, end)
    result: List[Any] = []
    for item in records:
        point = _to_canonical_point(item)
        if point in filtered:
            result.append(_set_legacy_metrics(item, time_difference_minutes=abs((point.timestamp - spill_time).total_seconds()) / 60.0))
    return result


def filter_by_distance(records: Sequence[Any], spill_lat: float, spill_lon: float, radius_km: float) -> List[Any]:
    """Legacy compatibility wrapper around the canonical radius filtering."""
    canonical_points = [_to_canonical_point(r) for r in records]
    matched_points = canonical_filter_by_distance(canonical_points, spill_lat, spill_lon, radius_km)
    matched_by_vessel = {pt.vessel_id: pt for pt, _ in matched_points}
    result: List[Any] = []
    for record in records:
        point = _to_canonical_point(record)
        if point.vessel_id in matched_by_vessel:
            distance = haversine_distance(spill_lat, spill_lon, point.latitude, point.longitude)
            result.append(_set_legacy_metrics(record, distance_km=distance))
    return result


def apply_filters(records: Sequence[Any], spill: Any, config: Any) -> List[Any]:
    """Legacy compatibility wrapper that applies the canonical AIS filters."""
    time_filtered = filter_by_time(records, spill.timestamp, config.time_window_minutes)
    return filter_by_distance(time_filtered, spill.latitude, spill.longitude, config.search_radius_km)


__all__ = [
    "apply_filters",
    "filter_by_distance",
    "filter_by_time",
    "haversine_distance",
]

