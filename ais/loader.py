"""Data loading module for AIS datasets supporting CSV and JSON formats."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Canonical column aliases in priority order (first matching alias takes precedence)
COLUMN_PRIORITY: Dict[str, List[str]] = {
    "vessel_id": [
        "vessel_id",
        "vesselid",
        "mmsi",
        "mmsi_id",
        "ship_id",
        "shipid",
    ],
    "timestamp": [
        "timestamp",
        "datetime",
        "time",
        "base_datetime",
        "date_time",
        "ts",
        "event_time",
    ],
    "latitude": [
        "latitude",
        "lat",
        "lat_deg",
        "lat_dd",
        "y",
    ],
    "longitude": [
        "longitude",
        "lon",
        "lng",
        "lon_deg",
        "lon_dd",
        "x",
    ],
    "speed_knots": [
        "speed_knots",
        "speed_kts",
        "sog",
        "speed",
        "speedoverground",
    ],
    "heading_deg": [
        "heading_deg",
        "heading",
        "cog",
        "course",
        "courseoverground",
        "true_heading",
    ],
}

REQUIRED_CANONICAL_COLUMNS = ["vessel_id", "timestamp", "latitude", "longitude"]


def _build_column_mapping(fieldnames: List[str]) -> Tuple[Dict[str, str], Set[str]]:
    """Build mapping from raw column headers to canonical field names with alias prioritization."""
    col_mapping: Dict[str, str] = {}
    found_canonical: Set[str] = set()

    # Clean headers
    cleaned_fields = {col.strip().lower(): col for col in fieldnames if col}

    for canonical_name, aliases in COLUMN_PRIORITY.items():
        for alias in aliases:
            if alias in cleaned_fields:
                raw_col = cleaned_fields[alias]
                col_mapping[raw_col] = canonical_name
                found_canonical.add(canonical_name)
                break  # Pick highest priority alias for this canonical field

    return col_mapping, found_canonical


def _normalize_raw_record(
    raw_dict: Dict[str, Any],
    col_mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Normalize a raw data row dictionary into standard canonical fields."""
    normalized: Dict[str, Any] = {
        "vessel_id": None,
        "timestamp": None,
        "latitude": None,
        "longitude": None,
        "speed_knots": None,
        "heading_deg": None,
    }

    for raw_key, raw_val in raw_dict.items():
        if raw_key in col_mapping:
            canonical_key = col_mapping[raw_key]
            normalized[canonical_key] = raw_val

    return normalized


def load_ais_csv(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load AIS records from a CSV file.

    Args:
        file_path: Path to the AIS CSV file.

    Returns:
        List of raw dictionaries normalized with canonical keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If CSV header is missing or required columns are absent.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"AIS CSV file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # Ignore comment lines (e.g. synthetic data header metadata)
        lines = [line for line in f if not line.strip().startswith("#")]

    if not lines:
        raise ValueError(f"AIS CSV file is empty or contains only comments: {path}")

    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        raise ValueError(f"Could not read CSV header from: {path}")

    col_mapping, found_canonical = _build_column_mapping(reader.fieldnames)

    # Check for missing required columns
    missing = [req for req in REQUIRED_CANONICAL_COLUMNS if req not in found_canonical]
    if missing:
        raise ValueError(
            f"Missing required columns in AIS file: {', '.join(missing)}. "
            f"Found columns: {list(reader.fieldnames)}"
        )

    records: List[Dict[str, Any]] = []
    for row in reader:
        records.append(_normalize_raw_record(row, col_mapping))

    return records


def load_ais_json(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load AIS records from a JSON file.

    Supports arrays of objects or root objects with 'records' or 'points' keys.

    Args:
        file_path: Path to the AIS JSON file.

    Returns:
        List of raw dictionaries normalized with canonical keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON format is invalid or required fields are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"AIS JSON file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in AIS file {path}: {exc}") from exc

    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            raw_list = data["records"]
        elif "points" in data and isinstance(data["points"], list):
            raw_list = data["points"]
        elif "ais_records" in data and isinstance(data["ais_records"], list):
            raw_list = data["ais_records"]
        else:
            raise ValueError(
                "JSON root object must be a list of records or contain a 'records'/'points' array."
            )
    elif isinstance(data, list):
        raw_list = data
    else:
        raise ValueError(f"Unexpected JSON root type: {type(data).__name__}")

    if not raw_list:
        return []

    # Map column headers based on first record keys
    first_item = raw_list[0]
    if not isinstance(first_item, dict):
        raise ValueError(f"JSON records must be objects, got {type(first_item).__name__}")

    col_mapping, found_canonical = _build_column_mapping(list(first_item.keys()))

    missing = [req for req in REQUIRED_CANONICAL_COLUMNS if req not in found_canonical]
    if missing:
        raise ValueError(
            f"Missing required fields in AIS JSON records: {', '.join(missing)}. "
            f"Available fields: {list(first_item.keys())}"
        )

    records: List[Dict[str, Any]] = []
    for item in raw_list:
        if isinstance(item, dict):
            records.append(_normalize_raw_record(item, col_mapping))

    return records


def load_ais_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load AIS records from a file, automatically dispatching based on file extension.

    Supported formats: CSV (.csv), JSON (.json).

    Args:
        file_path: Path to AIS file.

    Returns:
        List of normalized raw dictionary records.

    Raises:
        ValueError: If file format extension is unsupported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_ais_csv(path)
    elif suffix == ".json":
        return load_ais_json(path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: .csv, .json")
