"""Spatio-temporal filtering and candidate vessel identification using Haversine distance."""

import math
from datetime import datetime
from typing import Dict, List, Tuple, Union

from .cleaner import parse_utc_timestamp
from .schemas import AISPoint, CandidateOutput, CandidateVessel, VesselTrajectory

EARTH_RADIUS_KM: float = 6371.0088


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two geographic coordinates in kilometers.

    Uses the spherical Haversine formula on WGS84 coordinates.

    Args:
        lat1: Latitude of point 1 in decimal degrees [-90.0, 90.0].
        lon1: Longitude of point 1 in decimal degrees [-180.0, 180.0].
        lat2: Latitude of point 2 in decimal degrees [-90.0, 90.0].
        lon2: Longitude of point 2 in decimal degrees [-180.0, 180.0].

    Returns:
        float: Distance in kilometers.

    Raises:
        ValueError: If any coordinate falls outside valid geographic boundaries.
    """
    if not (-90.0 <= lat1 <= 90.0 and -90.0 <= lat2 <= 90.0):
        raise ValueError(f"Latitude must be in [-90, 90]. Got lat1={lat1}, lat2={lat2}")
    if not (-180.0 <= lon1 <= 180.0 and -180.0 <= lon2 <= 180.0):
        raise ValueError(f"Longitude must be in [-180, 180]. Got lon1={lon1}, lon2={lon2}")

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    sin_dphi_2 = math.sin(delta_phi / 2.0)
    sin_dlambda_2 = math.sin(delta_lambda / 2.0)

    a = sin_dphi_2 * sin_dphi_2 + math.cos(phi1) * math.cos(phi2) * sin_dlambda_2 * sin_dlambda_2
    a = min(1.0, max(0.0, a))

    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def filter_by_time_window(
    points: List[AISPoint],
    release_start: Union[str, datetime],
    release_end: Union[str, datetime],
) -> List[AISPoint]:
    """Filter AIS points that fall within the release time window [release_start, release_end].

    Args:
        points: List of AISPoint instances.
        release_start: Start timestamp (UTC).
        release_end: End timestamp (UTC).

    Returns:
        List of AISPoint instances within the time window.
    """
    t_start = parse_utc_timestamp(release_start)
    t_end = parse_utc_timestamp(release_end)

    if t_start > t_end:
        raise ValueError(f"release_start ({t_start}) cannot be after release_end ({t_end})")

    return [p for p in points if t_start <= p.timestamp <= t_end]


def filter_by_distance(
    points: List[AISPoint],
    origin_lat: float,
    origin_lon: float,
    search_radius_km: float,
) -> List[Tuple[AISPoint, float]]:
    """Filter AIS points within a circular radius from the probable spill origin.

    Args:
        points: List of AISPoint instances.
        origin_lat: Spill origin latitude in decimal degrees.
        origin_lon: Spill origin longitude in decimal degrees.
        search_radius_km: Maximum allowable search distance in km.

    Returns:
        List of tuples (AISPoint, distance_km) for points within search_radius_km.
    """
    matches: List[Tuple[AISPoint, float]] = []
    for pt in points:
        dist = haversine_distance(origin_lat, origin_lon, pt.latitude, pt.longitude)
        if dist <= search_radius_km:
            matches.append((pt, dist))
    return matches


def find_candidate_vessels(
    trajectories: Union[Dict[str, VesselTrajectory], List[AISPoint]],
    origin_lat: float,
    origin_lon: float,
    release_start: Union[str, datetime],
    release_end: Union[str, datetime],
    search_radius_km: float = 10.0,
    spill_id: str = "spill_001",
) -> CandidateOutput:
    """Identify candidate vessels near a probable spill origin during a release window.

    For each vessel, evaluates its observations inside the release time window,
    identifies the closest geographic approach to the origin, and retains vessels
    whose closest distance is within search_radius_km.

    Args:
        trajectories: Dict of vessel_id -> VesselTrajectory or a flat list of AISPoints.
        origin_lat: Latitude of probable spill origin.
        origin_lon: Longitude of probable spill origin.
        release_start: Start of release time window.
        release_end: End of release time window.
        search_radius_km: Search radius around origin in kilometers.
        spill_id: Identifier for the spill incident.

    Returns:
        CandidateOutput: Object containing spill_id and list of CandidateVessel objects.
    """
    t_start = parse_utc_timestamp(release_start)
    t_end = parse_utc_timestamp(release_end)

    if t_start > t_end:
        raise ValueError(f"release_start ({t_start}) cannot be after release_end ({t_end})")

    # Handle either Dict[str, VesselTrajectory] or flat List[AISPoint]
    if isinstance(trajectories, list):
        from .trajectory import build_trajectories
        traj_dict = build_trajectories(trajectories)
    elif isinstance(trajectories, dict):
        traj_dict = trajectories
    else:
        raise TypeError("trajectories must be a Dict[str, VesselTrajectory] or List[AISPoint]")

    candidates: List[CandidateVessel] = []

    for vessel_id, traj in traj_dict.items():
        # 1. Filter points in release window
        window_points = [p for p in traj.points if t_start <= p.timestamp <= t_end]
        if not window_points:
            continue

        # 2. Find closest point of approach during release window
        closest_point: AISPoint = window_points[0]
        min_dist: float = float("inf")

        for pt in window_points:
            dist = haversine_distance(origin_lat, origin_lon, pt.latitude, pt.longitude)
            if dist < min_dist:
                min_dist = dist
                closest_point = pt

        # 3. Spatial filter threshold
        if min_dist <= search_radius_km:
            candidates.append(
                CandidateVessel(
                    vessel_id=vessel_id,
                    closest_distance_km=round(min_dist, 3),
                    closest_timestamp=closest_point.timestamp_iso,
                    latitude=closest_point.latitude,
                    longitude=closest_point.longitude,
                    speed_knots=closest_point.speed_knots,
                    heading_deg=closest_point.heading_deg,
                )
            )

    # Sort candidates by closest distance ascending
    candidates.sort(key=lambda c: c.closest_distance_km)

    return CandidateOutput(spill_id=spill_id, candidates=candidates)

