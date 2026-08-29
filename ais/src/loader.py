"""Data loading and validation for spill incident input and AIS records."""

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class SpillIncident:
    """Spill incident parameters received from upstream (e.g., Satellite + ML)."""

    latitude: float
    longitude: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Serialize spill incident to dictionary format with ISO-8601 UTC timestamp."""
        ts_str = self.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "estimated_time": ts_str,
        }


@dataclass
class AISRecord:
    """Validated AIS observation record for a vessel at a point in time."""

    mmsi: str
    timestamp: datetime
    latitude: float
    longitude: float
    imo: Optional[str] = None
    ship_name: Optional[str] = None
    ship_type: Optional[str] = None
    sog: Optional[float] = None
    cog: Optional[float] = None
    heading: Optional[float] = None
    distance_km: Optional[float] = None
    time_difference_minutes: Optional[float] = None

    @property
    def timestamp_iso(self) -> str:
        """Return ISO 8601 UTC formatted string."""
        return self.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_trajectory_dict(self) -> Dict[str, Any]:
        """Convert AIS record into a trajectory point dictionary for attribution."""
        data: Dict[str, Any] = {
            "timestamp": self.timestamp_iso,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "sog": self.sog,
            "cog": self.cog,
        }
        if self.heading is not None:
            data["heading"] = self.heading
        return data


def parse_utc_timestamp(ts_val: Union[str, datetime]) -> datetime:
    """Parse various timestamp representations into a timezone-aware UTC datetime.

    Supported formats include ISO 8601 ('2026-08-20T14:30:00Z', '2026-08-20T14:30:00+00:00'),
    space-separated ('2026-08-20 14:30:00'), and common date/time formats.

    Raises:
        ValueError: If timestamp string cannot be parsed into a valid datetime.
    """
    if isinstance(ts_val, datetime):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=timezone.utc)
        return ts_val.astimezone(timezone.utc)

    if not isinstance(ts_val, str) or not ts_val.strip():
        raise ValueError(f"Timestamp must be a non-empty string or datetime object, got {type(ts_val)}")

    clean_str = ts_val.strip()

    # Normalize 'Z' to '+00:00' for datetime.fromisoformat
    if clean_str.endswith("Z") or clean_str.endswith("z"):
        clean_str = clean_str[:-1] + "+00:00"

    # Attempt native ISO parsing first (Python 3.11+)
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

    raise ValueError(f"Unable to parse timestamp '{ts_val}'. Expected ISO 8601 format (e.g. '2026-08-20T14:30:00Z').")


def load_spill_input(file_path: Union[str, Path]) -> SpillIncident:
    """Load and validate the oil spill incident JSON file.

    Args:
        file_path: Path to the mock/real spill JSON file.

    Returns:
        SpillIncident: Validated spill incident object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON structure or coordinates/timestamp are invalid.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Spill input file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in spill input file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Spill input root must be a JSON object, got {type(data).__name__}")

    # Field extraction with backwards-compatible key mappings
    lat_val = data.get("spill_latitude", data.get("latitude"))
    lon_val = data.get("spill_longitude", data.get("longitude"))
    ts_val = data.get("spill_timestamp", data.get("timestamp"))

    if lat_val is None:
        raise ValueError("Spill input missing required field 'spill_latitude'")
    if lon_val is None:
        raise ValueError("Spill input missing required field 'spill_longitude'")
    if ts_val is None:
        raise ValueError("Spill input missing required field 'spill_timestamp'")

    try:
        lat = float(lat_val)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid spill_latitude: '{lat_val}'. Must be numeric.")

    try:
        lon = float(lon_val)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid spill_longitude: '{lon_val}'. Must be numeric.")

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"spill_latitude out of range [-90, 90]: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"spill_longitude out of range [-180, 180]: {lon}")

    parsed_dt = parse_utc_timestamp(ts_val)

    return SpillIncident(latitude=lat, longitude=lon, timestamp=parsed_dt)


def _safe_float(val: Any) -> Optional[float]:
    """Safely convert a value to float or return None."""
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_str(val: Any) -> Optional[str]:
    """Safely convert a value to stripped string or return None."""
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def load_and_validate_ais_csv(
    file_path: Union[str, Path]
) -> Tuple[List[AISRecord], List[str]]:
    """Load and validate AIS records from a CSV file.

    Skips invalid rows and logs clear validation messages for rejected records.

    Args:
        file_path: Path to AIS CSV dataset.

    Returns:
        Tuple containing:
            - List of valid AISRecord instances
            - List of string error/warning messages for rejected rows

    Raises:
        FileNotFoundError: If the CSV file is not found.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"AIS dataset file not found: {path.resolve()}")

    records: List[AISRecord] = []
    errors: List[str] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # Filter out comment lines (e.g. synthetic data headers)
        lines = [line for line in f if not line.strip().startswith("#")]

    if not lines:
        return records, ["AIS CSV file is empty or contains only comments."]

    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        return records, ["Could not read CSV header."]

    # Normalize header column lookup (case-insensitive)
    col_map = {name.strip().lower(): name for name in reader.fieldnames if name}

    # Required column aliases
    mmsi_col = col_map.get("mmsi")
    ts_col = col_map.get("timestamp") or col_map.get("datetime") or col_map.get("time") or col_map.get("base_datetime")
    lat_col = col_map.get("latitude") or col_map.get("lat")
    lon_col = col_map.get("longitude") or col_map.get("lon") or col_map.get("lng")

    # Optional column aliases
    imo_col = col_map.get("imo")
    name_col = col_map.get("ship_name") or col_map.get("shipname") or col_map.get("vessel_name") or col_map.get("name")
    type_col = col_map.get("ship_type") or col_map.get("shiptype") or col_map.get("vessel_type") or col_map.get("type")
    sog_col = col_map.get("sog") or col_map.get("speed")
    cog_col = col_map.get("cog") or col_map.get("course")
    heading_col = col_map.get("heading")

    missing_cols = []
    if not mmsi_col:
        missing_cols.append("MMSI")
    if not ts_col:
        missing_cols.append("timestamp")
    if not lat_col:
        missing_cols.append("latitude")
    if not lon_col:
        missing_cols.append("longitude")

    if missing_cols:
        return records, [f"CSV missing mandatory columns: {', '.join(missing_cols)}"]

    for row_idx, row in enumerate(reader, start=2):
        raw_mmsi = row.get(mmsi_col, "")
        raw_ts = row.get(ts_col, "")
        raw_lat = row.get(lat_col, "")
        raw_lon = row.get(lon_col, "")

        # 1. MMSI validation
        clean_mmsi = str(raw_mmsi).strip() if raw_mmsi is not None else ""
        if not clean_mmsi:
            errors.append(f"Row {row_idx}: Skipped — Missing or empty MMSI.")
            continue

        # 2. Timestamp validation
        try:
            parsed_dt = parse_utc_timestamp(raw_ts)
        except Exception as e:
            errors.append(f"Row {row_idx} (MMSI {clean_mmsi}): Skipped — Invalid timestamp '{raw_ts}': {e}")
            continue

        # 3. Latitude validation
        try:
            lat = float(raw_lat)
        except (ValueError, TypeError):
            errors.append(f"Row {row_idx} (MMSI {clean_mmsi}): Skipped — Non-numeric latitude '{raw_lat}'.")
            continue

        if not (-90.0 <= lat <= 90.0):
            errors.append(f"Row {row_idx} (MMSI {clean_mmsi}): Skipped — Latitude {lat} outside [-90, 90].")
            continue

        # 4. Longitude validation
        try:
            lon = float(raw_lon)
        except (ValueError, TypeError):
            errors.append(f"Row {row_idx} (MMSI {clean_mmsi}): Skipped — Non-numeric longitude '{raw_lon}'.")
            continue

        if not (-180.0 <= lon <= 180.0):
            errors.append(f"Row {row_idx} (MMSI {clean_mmsi}): Skipped — Longitude {lon} outside [-180, 180].")
            continue

        # Optional fields
        imo = _safe_str(row.get(imo_col)) if imo_col else None
        ship_name = _safe_str(row.get(name_col)) if name_col else None
        ship_type = _safe_str(row.get(type_col)) if type_col else None
        sog = _safe_float(row.get(sog_col)) if sog_col else None
        cog = _safe_float(row.get(cog_col)) if cog_col else None
        heading = _safe_float(row.get(heading_col)) if heading_col else None

        records.append(
            AISRecord(
                mmsi=clean_mmsi,
                timestamp=parsed_dt,
                latitude=lat,
                longitude=lon,
                imo=imo,
                ship_name=ship_name,
                ship_type=ship_type,
                sog=sog,
                cog=cog,
                heading=heading,
            )
        )

    return records, errors

