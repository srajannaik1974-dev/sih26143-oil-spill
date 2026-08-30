"""
Spatial-temporal feature calculation and candidate filtering for Vessel Attribution.
Calculates Haversine distance, temporal difference, trajectory compatibility, and speed/heading metrics.
"""

import math
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from .schemas import SpillOriginInput, AISTrajectoryRecord, AISPosition


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth's surface in kilometers.
    Uses the Haversine formula.
    """
    r_km = 6371.0  # Earth's mean radius in kilometers

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return r_km * c


def calculate_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing (heading) in degrees from point 1 to point 2.
    Returns bearing in range [0, 360).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2) -
         math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda))

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def extract_vessel_features(
    spill: SpillOriginInput,
    vessel: AISTrajectoryRecord
) -> Optional[Dict[str, Any]]:
    """
    Extract spatial, temporal, trajectory, and motion features for a single vessel relative to spill origin.
    Returns None if vessel fails candidate spatial/temporal threshold filtering.
    """
    if not vessel.positions:
        return None

    spill_time = spill.estimated_release_time.replace(tzinfo=timezone.utc) if spill.estimated_release_time.tzinfo is None else spill.estimated_release_time

    # Evaluate all positions for spatial & temporal metrics
    pos_metrics = []
    for pos in vessel.positions:
        pos_time = pos.timestamp.replace(tzinfo=timezone.utc) if pos.timestamp.tzinfo is None else pos.timestamp
        dist_km = haversine_distance_km(spill.latitude, spill.longitude, pos.latitude, pos.longitude)
        dt_seconds = abs((pos_time - spill_time).total_seconds())
        pos_metrics.append({
            "position": pos,
            "distance_km": dist_km,
            "dt_seconds": dt_seconds,
            "dt_minutes": dt_seconds / 60.0,
            "timestamp": pos_time,
        })

    # Find point of closest spatial approach (CPA)
    cpa_point = min(pos_metrics, key=lambda x: x["distance_km"])
    min_dist_km = cpa_point["distance_km"]

    # Candidate Filtering: check spatial radius boundary
    if min_dist_km > spill.max_search_radius_km:
        return None

    # Candidate Filtering: check temporal window boundary (closest point within max_time_window_hours)
    min_dt_hours = cpa_point["dt_seconds"] / 3600.0
    if min_dt_hours > spill.max_time_window_hours:
        return None

    # Find position closest to estimated release time
    time_closest_point = min(pos_metrics, key=lambda x: x["dt_seconds"])

    # Trajectory compatibility features
    # Count positions within search radius & calculate average distance within zone
    zone_positions = [p for p in pos_metrics if p["distance_km"] <= spill.max_search_radius_km]
    dwell_count = len(zone_positions)
    avg_dist_in_zone = sum(p["distance_km"] for p in zone_positions) / dwell_count if zone_positions else min_dist_km

    # Speed & Heading compatibility features around CPA / release time
    target_pos = cpa_point["position"]
    speed_at_cpa = target_pos.speed_knots

    # Heading alignment: bearing from vessel position to spill origin
    bearing_to_spill = calculate_bearing_deg(
        target_pos.latitude, target_pos.longitude,
        spill.latitude, spill.longitude
    )
    vessel_heading = target_pos.heading_deg if target_pos.heading_deg is not None else target_pos.course_over_ground
    
    heading_diff_deg = 180.0
    if vessel_heading is not None:
        diff = abs(vessel_heading - bearing_to_spill) % 360.0
        heading_diff_deg = min(diff, 360.0 - diff)

    return {
        "vessel_id": vessel.vessel_id,
        "mmsi": vessel.mmsi,
        "vessel_name": vessel.vessel_name,
        "vessel_type": vessel.vessel_type,
        "min_distance_km": min_dist_km,
        "cpa_time_diff_minutes": cpa_point["dt_minutes"],
        "closest_time_diff_minutes": time_closest_point["dt_minutes"],
        "dwell_count": dwell_count,
        "avg_dist_in_zone_km": avg_dist_in_zone,
        "speed_at_cpa_knots": speed_at_cpa,
        "heading_diff_deg": heading_diff_deg,
        "cpa_timestamp": cpa_point["timestamp"],
    }
