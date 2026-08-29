"""Data cleaning, validation, deduplication, and UTC timestamp normalization for AIS records."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .schemas import AISPoint


def parse_utc_timestamp(ts_val: Union[str, datetime]) -> datetime:
    """Parse various timestamp representations into a timezone-aware UTC datetime.

    Supported formats include ISO 8601 ('2026-08-20T14:30:00Z', '2026-08-20T14:30:00+00:00'),
    space-separated ('2026-08-20 14:30:00'), and common date/time formats.

    Args:
        ts_val: Datetime object or timestamp string.

    Returns:
        datetime: Timezone-aware datetime in UTC.

    Raises:
        ValueError: If timestamp is empty, invalid, or cannot be parsed.
    """
    if isinstance(ts_val, datetime):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=timezone.utc)
        return ts_val.astimezone(timezone.utc)

    if not isinstance(ts_val, str) or not ts_val.strip():
        raise ValueError(f"Timestamp must be a non-empty string or datetime, got {type(ts_val)}")

    clean_str = ts_val.strip()

    # Normalize 'Z' to '+00:00' for ISO parsing
    if clean_str.endswith("Z") or clean_str.endswith("z"):
        clean_str = clean_str[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(clean_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    # Fallback common datetime format patterns
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(clean_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse timestamp '{ts_val}'. Expected ISO 8601 UTC format.")


def _safe_float(val: Any) -> Optional[float]:
    """Safely convert a value to float, returning None if empty or invalid."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan", ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def clean_ais_records(raw_records: List[Dict[str, Any]]) -> List[AISPoint]:
    """Validate, clean, normalize timestamps to UTC, deduplicate, and sort AIS records.

    Rules applied:
      - vessel_id must be non-empty string.
      - timestamp must be valid and normalized to UTC.
      - latitude must be numeric and in [-90.0, 90.0].
      - longitude must be numeric and in [-180.0, 180.0].
      - speed_knots and heading_deg are converted to float or None (not invented).
      - Deduplicates identical records matching (vessel_id, timestamp).
      - Sorts records by vessel_id and timestamp.

    Args:
        raw_records: List of normalized dictionary records.

    Returns:
        List of validated and cleaned AISPoint instances.
    """
    valid_points: List[AISPoint] = []
    seen_keys: Set[Tuple[str, str]] = set()

    for raw in raw_records:
        # 1. Vessel ID validation
        raw_id = raw.get("vessel_id")
        if raw_id is None or not str(raw_id).strip():
            continue
        clean_vessel_id = str(raw_id).strip()

        # 2. Timestamp validation
        raw_ts = raw.get("timestamp")
        try:
            dt = parse_utc_timestamp(raw_ts)
        except (ValueError, TypeError):
            continue

        # 3. Latitude validation
        lat_val = _safe_float(raw.get("latitude"))
        if lat_val is None or not (-90.0 <= lat_val <= 90.0):
            continue

        # 4. Longitude validation
        lon_val = _safe_float(raw.get("longitude"))
        if lon_val is None or not (-180.0 <= lon_val <= 180.0):
            continue

        # 5. Optional Speed and Heading (safe conversion, do not invent values)
        speed_val = _safe_float(raw.get("speed_knots"))
        if speed_val is not None and speed_val < 0:
            speed_val = None

        heading_val = _safe_float(raw.get("heading_deg"))
        if heading_val is not None and not (0.0 <= heading_val <= 360.0):
            heading_val = None

        # 6. Deduplication by (vessel_id, ISO timestamp)
        iso_ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        dedup_key = (clean_vessel_id, iso_ts)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        valid_points.append(
            AISPoint(
                vessel_id=clean_vessel_id,
                timestamp=dt,
                latitude=lat_val,
                longitude=lon_val,
                speed_knots=speed_val,
                heading_deg=heading_val,
            )
        )

    # 7. Sort records by vessel_id and timestamp
    valid_points.sort(key=lambda p: (p.vessel_id, p.timestamp))

    return valid_points


def clean_ais_data(raw_records: List[Dict[str, Any]]) -> List[AISPoint]:
    """Public alias for validation, deduplication, and UTC normalization of raw AIS records."""
    return clean_ais_records(raw_records)

