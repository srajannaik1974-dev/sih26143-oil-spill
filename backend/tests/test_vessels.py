import os
from fastapi.testclient import TestClient
from backend.main import app
from backend.dependencies import get_vessel_service, get_mock_vessel_service


def test_vessel_rank_success_real_attribution(client: TestClient):
    """
    Test POST /api/vessels/rank using Member 4's RealVesselAttributionService.
    """
    payload = {
        "spill_source": {
            "latitude": 19.4167,
            "longitude": 71.3333
        },
        "timestamp": "2026-08-26T12:00:00Z",
        "search_radius_km": 50.0,
        "time_window_hours": 24.0,
        "candidate_vessels": [
            {
                "mmsi": "419000101",
                "vessel_name": "Ocean Titan",
                "vessel_type": "Crude Oil Tanker",
                "latitude": 19.418,
                "longitude": 71.335,
                "timestamp": "2026-08-26T11:45:00Z",
                "speed_knots": 4.5,
                "heading_degrees": 135.0,
                "distance_to_source_km": 1.2
            },
            {
                "mmsi": "419000102",
                "vessel_name": "Pacific Voyager",
                "vessel_type": "Container Ship",
                "latitude": 19.486,
                "longitude": 71.383,
                "timestamp": "2026-08-26T10:00:00Z",
                "speed_knots": 13.2,
                "heading_degrees": 210.0,
                "distance_to_source_km": 8.0
            }
        ]
    }
    response = client.post("/api/vessels/rank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "ranked_vessels" in data
    assert len(data["ranked_vessels"]) == 2
    
    top = data["ranked_vessels"][0]
    assert top["rank"] == 1
    assert top["vessel"]["mmsi"] == "419000101"
    assert 0.0 <= top["risk_score"] <= 1.0
    assert len(top["attribution_factors"]) == 4
    assert top["classification"] is not None
    assert top["explanation"] is not None
    assert "Member 4" in data["disclaimer"]


def test_vessel_rank_with_trajectories(client: TestClient):
    """
    Test POST /api/vessels/rank with multi-point AIS trajectory position arrays.
    """
    payload = {
        "spill_source": {
            "latitude": 19.4167,
            "longitude": 71.3333
        },
        "timestamp": "2026-08-26T12:00:00Z",
        "candidate_vessels": [
            {
                "mmsi": "419000101",
                "vessel_name": "Ocean Titan",
                "vessel_type": "Crude Oil Tanker",
                "latitude": 19.4167,
                "longitude": 71.3333,
                "timestamp": "2026-08-26T12:00:00Z",
                "speed_knots": 4.5,
                "heading_degrees": 135.0,
                "positions": [
                    {
                        "timestamp": "2026-08-26T11:45:00Z",
                        "latitude": 19.418,
                        "longitude": 71.335,
                        "speed_knots": 4.5,
                        "heading_deg": 135.0,
                        "course_over_ground": 132.0
                    },
                    {
                        "timestamp": "2026-08-26T12:00:00Z",
                        "latitude": 19.4167,
                        "longitude": 71.3333,
                        "speed_knots": 4.0,
                        "heading_deg": 135.0,
                        "course_over_ground": 132.0
                    }
                ]
            }
        ]
    }
    response = client.post("/api/vessels/rank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["ranked_vessels"]) == 1
    assert data["ranked_vessels"][0]["vessel"]["mmsi"] == "419000101"


def test_vessel_rank_missing_candidates(client: TestClient):
    """
    Test validation failure when candidate_vessels field is missing.
    """
    payload = {
        "spill_source": {
            "latitude": 19.4167,
            "longitude": 71.3333
        },
        "timestamp": "2026-08-26T12:00:00Z"
    }
    response = client.post("/api/vessels/rank", json=payload)
    assert response.status_code == 422


def test_vessel_rank_invalid_coordinates(client: TestClient):
    """
    Test validation failure when invalid coordinates are provided.
    """
    payload = {
        "spill_source": {
            "latitude": 120.0,  # Invalid (>90)
            "longitude": 71.3333
        },
        "timestamp": "2026-08-26T12:00:00Z",
        "candidate_vessels": []
    }
    response = client.post("/api/vessels/rank", json=payload)
    assert response.status_code == 422


def test_vessel_rank_mock_fallback(client: TestClient):
    """
    Verify MockVesselService fallback functionality using FastAPI dependency override.
    """
    app.dependency_overrides[get_vessel_service] = get_mock_vessel_service
    try:
        payload = {
            "spill_source": {
                "latitude": 15.35,
                "longitude": 73.80
            },
            "timestamp": "2026-08-26T12:00:00Z",
            "candidate_vessels": [
                {
                    "mmsi": "413298410",
                    "vessel_name": "OCEAN TITAN",
                    "vessel_type": "Crude Oil Tanker",
                    "latitude": 15.362,
                    "longitude": 73.792,
                    "timestamp": "2026-08-26T12:00:00Z",
                    "speed_knots": 12.4,
                    "heading_degrees": 145.0,
                    "distance_to_source_km": 1.85
                }
            ]
        }
        response = client.post("/api/vessels/rank", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "MOCK ATTRIBUTION RESULT" in data["disclaimer"]
    finally:
        app.dependency_overrides.clear()
