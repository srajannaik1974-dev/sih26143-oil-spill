from fastapi.testclient import TestClient


def test_ais_candidates_success(client: TestClient):
    payload = {
        "source_latitude": 15.35,
        "source_longitude": 73.80,
        "timestamp": "2026-08-26T12:00:00Z",
        "search_radius_km": 50.0,
        "time_window_hours": 12.0
    }
    response = client.post("/api/ais/candidates", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert isinstance(data["candidates"], list)
    assert data["total_count"] == len(data["candidates"])
    assert "MOCK AIS RESULT" in data["disclaimer"]
    if len(data["candidates"]) > 0:
        first = data["candidates"][0]
        assert "mmsi" in first
        assert "vessel_name" in first
        assert "speed_knots" in first


def test_ais_candidates_invalid_radius(client: TestClient):
    payload = {
        "source_latitude": 15.35,
        "source_longitude": 73.80,
        "timestamp": "2026-08-26T12:00:00Z",
        "search_radius_km": -10.0  # Invalid radius
    }
    response = client.post("/api/ais/candidates", json=payload)
    assert response.status_code == 422
