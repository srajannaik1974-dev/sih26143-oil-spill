"""Spatio-temporal filtering of AIS records against spill parameters."""

from datetime import datetime
from typing import List

from .config import AISConfig
from .distance import haversine_distance
from .loader import AISRecord, SpillIncident


def filter_by_time(
    records: List[AISRecord],
    spill_time: datetime,
    window_minutes: float,
) -> List[AISRecord]:
    """Filter AIS records within the specified time window around the spill timestamp.

    Computes and attaches `time_difference_minutes` to each kept record.

    Args:
        records: List of AISRecord instances.
        spill_time: Estimated UTC timestamp of the oil spill incident.
        window_minutes: Maximum allowable time difference in minutes.

    Returns:
        List of AISRecord instances falling within [spill_time - window, spill_time + window].
    """
    valid_records: List[AISRecord] = []
    for rec in records:
        diff_seconds = abs((rec.timestamp - spill_time).total_seconds())
        diff_minutes = diff_seconds / 60.0
        if diff_minutes <= window_minutes:
            rec.time_difference_minutes = round(diff_minutes, 2)
            valid_records.append(rec)
    return valid_records


def filter_by_distance(
    records: List[AISRecord],
    spill_lat: float,
    spill_lon: float,
    radius_km: float,
) -> List[AISRecord]:
    """Filter AIS records within the specified geographic radius from the spill location.

    Computes and attaches `distance_km` to each kept record using the Haversine formula.

    Args:
        records: List of AISRecord instances.
        spill_lat: Latitude of the spill incident in decimal degrees.
        spill_lon: Longitude of the spill incident in decimal degrees.
        radius_km: Maximum allowable geographic distance in kilometers.

    Returns:
        List of AISRecord instances within the circular radius.
    """
    valid_records: List[AISRecord] = []
    for rec in records:
        dist = haversine_distance(spill_lat, spill_lon, rec.latitude, rec.longitude)
        if dist <= radius_km:
            rec.distance_km = round(dist, 3)
            valid_records.append(rec)
    return valid_records


def apply_filters(
    records: List[AISRecord],
    spill: SpillIncident,
    config: AISConfig,
) -> List[AISRecord]:
    """Apply both temporal and spatial filters to raw AIS records.

    Args:
        records: Raw validated AIS records.
        spill: SpillIncident containing location and time.
        config: AISConfig with search_radius_km and time_window_minutes.

    Returns:
        Filtered list of AISRecord objects with distance_km and time_difference_minutes populated.
    """
    time_filtered = filter_by_time(records, spill.timestamp, config.time_window_minutes)
    spatio_temporal_filtered = filter_by_distance(
        time_filtered, spill.latitude, spill.longitude, config.search_radius_km
    )
    return spatio_temporal_filtered

