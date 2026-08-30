import pytest
from fastapi.testclient import TestClient
from backend.main import app


def test_end_to_end_pipeline(client: TestClient):
    """
    Test complete end-to-end integration flow:
    M1 (Spill Detect) -> M2 (Drift Backtrack) -> M3 (AIS Candidates) -> M4 (Vessel Ranking) -> M6 Response.
    """
    # 1. M1 Satellite Spill Detection
    detect_payload = {
        "image_id": "SENTINEL-1-TEST-001",
        "timestamp": "2026-08-27T12:00:00Z",
        "latitude": 19.4167,
        "longitude": 71.3333
    }
    detect_res = client.post("/api/spill/detect", json=detect_payload)
    assert detect_res.status_code == 200
    spill_data = detect_res.json()
    assert spill_data["spill_detected"] is True
    assert spill_data["confidence"] > 0.0

    # 2. M2 Drift Backtrack
    backtrack_payload = {
        "spill_location": {
            "latitude": detect_payload["latitude"],
            "longitude": detect_payload["longitude"]
        },
        "timestamp": detect_payload["timestamp"],
        "drift_hours": 6.0
    }
    backtrack_res = client.post("/api/spill/backtrack", json=backtrack_payload)
    assert backtrack_res.status_code == 200
    drift_data = backtrack_res.json()
    assert "estimated_source_area" in drift_data
    assert "trajectory" in drift_data

    origin_center = drift_data["estimated_source_area"]["center"]
    origin_time = drift_data["detection_timestamp"]

    # 3. M3 AIS Candidate Query
    ais_payload = {
        "source_latitude": origin_center["latitude"],
        "source_longitude": origin_center["longitude"],
        "timestamp": origin_time,
        "search_radius_km": 50.0,
        "time_window_hours": 12.0
    }
    ais_res = client.post("/api/ais/candidates", json=ais_payload)
    assert ais_res.status_code == 200
    ais_data = ais_res.json()
    assert "candidates" in ais_data
    assert isinstance(ais_data["candidates"], list)

    # 4. M4 Vessel Attribution & Ranking
    rank_payload = {
        "spill_source": origin_center,
        "timestamp": origin_time,
        "candidate_vessels": ais_data["candidates"]
    }
    rank_res = client.post("/api/vessels/rank", json=rank_payload)
    assert rank_res.status_code == 200
    rank_data = rank_res.json()
    assert "ranked_vessels" in rank_data
    assert len(rank_data["ranked_vessels"]) > 0

    top_vessel = rank_data["ranked_vessels"][0]
    assert top_vessel["rank"] == 1
    assert 0.0 <= top_vessel["risk_score"] <= 1.0
    assert len(top_vessel["attribution_factors"]) > 0
