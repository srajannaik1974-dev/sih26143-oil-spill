"""Synthetic AIS telemetry dataset generator for demonstrations and unit testing.

Clearly labeled: DEMO/SYNTHETIC TEST DATA — NOT REAL MARITIME TELEMETRY.
"""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Union

from .cleaner import parse_utc_timestamp
from .schemas import AISPoint


def generate_synthetic_ais(
    origin_lat: float = 12.8500,
    origin_lon: float = 74.7200,
    release_start: Union[str, datetime] = "2026-08-20T14:00:00Z",
    release_end: Union[str, datetime] = "2026-08-20T15:00:00Z",
    search_radius_km: float = 10.0,
) -> List[AISPoint]:
    """Generate deterministic synthetic AIS observation trajectories for demonstration.

    Constructs 5 distinct vessel scenarios around the given spill origin and release window:
      - VESSEL_001 (Passing close): Passes within ~0.3 km during window.
      - VESSEL_002 (Passing moderately close): Passes within ~3.0 km during window.
      - VESSEL_003 (Far away): Traverses ~15.5 km away (outside search radius).
      - VESSEL_004 (Outside time window): Operates near origin (~0.2 km) but 4 hours after release window.
      - VESSEL_005 (Boundary vessel): Operates ~6.5 km away with missing heading/speed data.

    Returns:
        List of AISPoint objects labeled as DEMO/SYNTHETIC data.
    """
    t_start = parse_utc_timestamp(release_start)
    t_end = parse_utc_timestamp(release_end)
    mid_time = t_start + (t_end - t_start) / 2

    points: List[AISPoint] = []

    # 1. VESSEL_001: Close approach (~0.31 km) during window
    for minute_offset in (-15, -10, -5, 0, 5, 10, 15):
        pt_time = mid_time + timedelta(minutes=minute_offset)
        # Trajectory passing at (12.8520, 74.7180) at minute_offset=0
        lat = origin_lat + 0.0020 + (minute_offset * 0.0003)
        lon = origin_lon - 0.0020 - (minute_offset * 0.0003)
        points.append(
            AISPoint(
                vessel_id="VESSEL_001",
                timestamp=pt_time,
                latitude=round(lat, 4),
                longitude=round(lon, 4),
                speed_knots=12.4,
                heading_deg=142.0,
            )
        )

    # 2. VESSEL_002: Moderate proximity (~3.0 km) during window
    for minute_offset in (-20, 0, 20):
        pt_time = mid_time + timedelta(minutes=minute_offset)
        lat = origin_lat + 0.0250 + (minute_offset * 0.0002)
        lon = origin_lon + 0.0100 + (minute_offset * 0.0002)
        points.append(
            AISPoint(
                vessel_id="VESSEL_002",
                timestamp=pt_time,
                latitude=round(lat, 4),
                longitude=round(lon, 4),
                speed_knots=14.1,
                heading_deg=210.0,
            )
        )

    # 3. VESSEL_003: Far outside search radius (~15.5 km)
    for minute_offset in (-10, 0, 10):
        pt_time = mid_time + timedelta(minutes=minute_offset)
        lat = origin_lat + 0.1300 + (minute_offset * 0.0001)
        lon = origin_lon + 0.0600 + (minute_offset * 0.0001)
        points.append(
            AISPoint(
                vessel_id="VESSEL_003",
                timestamp=pt_time,
                latitude=round(lat, 4),
                longitude=round(lon, 4),
                speed_knots=18.0,
                heading_deg=340.0,
            )
        )

    # 4. VESSEL_004: Close geographically (~0.2 km), but 4 hours late (outside time window)
    for minute_offset in (-5, 0, 5):
        pt_time = t_end + timedelta(hours=4, minutes=minute_offset)
        lat = origin_lat + 0.0010 + (minute_offset * 0.0002)
        lon = origin_lon + 0.0010 + (minute_offset * 0.0002)
        points.append(
            AISPoint(
                vessel_id="VESSEL_004",
                timestamp=pt_time,
                latitude=round(lat, 4),
                longitude=round(lon, 4),
                speed_knots=11.5,
                heading_deg=175.0,
            )
        )

    # 5. VESSEL_005: Within window and radius (~6.5 km), but missing speed & heading
    for minute_offset in (-10, 10):
        pt_time = mid_time + timedelta(minutes=minute_offset)
        lat = origin_lat - 0.0550 + (minute_offset * 0.0001)
        lon = origin_lon + 0.0250 + (minute_offset * 0.0001)
        points.append(
            AISPoint(
                vessel_id="VESSEL_005",
                timestamp=pt_time,
                latitude=round(lat, 4),
                longitude=round(lon, 4),
                speed_knots=None,
                heading_deg=None,
            )
        )

    return points


def save_synthetic_ais_csv(
    file_path: Union[str, Path],
    points: List[AISPoint],
) -> Path:
    """Save synthetic AIS points to a CSV file with clear demonstration headers.

    Args:
        file_path: Target path to write CSV.
        points: List of AISPoint instances.

    Returns:
        Path to written file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        # Header banner clearly identifying synthetic data
        f.write("# DEMO/SYNTHETIC TEST AIS DATA -- NOT REAL MARITIME TELEMETRY\n")
        f.write("# SIH 2026 Problem Statement 26143 -- AIS Vessel Tracking Module\n")

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vessel_id",
                "timestamp",
                "latitude",
                "longitude",
                "speed_knots",
                "heading_deg",
            ],
        )
        writer.writeheader()

        for pt in points:
            writer.writerow(
                {
                    "vessel_id": pt.vessel_id,
                    "timestamp": pt.timestamp_iso,
                    "latitude": pt.latitude,
                    "longitude": pt.longitude,
                    "speed_knots": pt.speed_knots if pt.speed_knots is not None else "",
                    "heading_deg": pt.heading_deg if pt.heading_deg is not None else "",
                }
            )

    return path

