import io
from fastapi.testclient import TestClient


def test_spill_detect_success(client: TestClient):
    payload = {
        "image_id": "SAT-IMG-2026-001",
        "image_url": "https://example.com/imagery/sat-img-001.tif",
        "timestamp": "2026-08-26T12:00:00Z",
        "latitude": 15.35,
        "longitude": 73.80
    }
    response = client.post("/api/spill/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["image_id"] == "SAT-IMG-2026-001"
    assert data["spill_detected"] is True
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["spill_polygon"], list)
    assert len(data["spill_polygon"]) >= 3
    assert "MOCK DETECTION RESULT" in data["disclaimer"]


def test_spill_detect_invalid_latitude(client: TestClient):
    payload = {
        "image_id": "SAT-IMG-2026-002",
        "timestamp": "2026-08-26T12:00:00Z",
        "latitude": 195.0,  # Invalid latitude (> 90)
        "longitude": 73.80
    }
    response = client.post("/api/spill/detect", json=payload)
    assert response.status_code == 422  # Unprocessable Entity validation error


def test_spill_detect_upload_invalid_extension(client: TestClient):
    """Test uploading non-TIFF file (e.g. .png) raises 400 Bad Request validation error."""
    files = {"file": ("test_image.png", io.BytesIO(b"fake png header"), "image/png")}
    response = client.post("/api/spill/detect/upload", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "Invalid file extension" in data["detail"]


def test_spill_detect_upload_missing_file(client: TestClient):
    """Test calling upload endpoint without file payload returns 422 error."""
    response = client.post("/api/spill/detect/upload")
    assert response.status_code == 422


def test_spill_detect_upload_corrupt_tiff(client: TestClient):
    """Test uploading corrupt .tif file raises 400 Bad Request from rasterio validator."""
    files = {"file": ("corrupt_sample.tif", io.BytesIO(b"NOT_A_REAL_TIFF_HEADER"), "image/tiff")}
    response = client.post("/api/spill/detect/upload", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "Corrupt or invalid" in data["detail"] or "Rasterio failed" in data["detail"]


def test_spill_backtrack_success(client: TestClient):
    payload = {
        "spill_location": {
            "latitude": 15.35,
            "longitude": 73.80
        },
        "timestamp": "2026-08-26T12:00:00Z",
        "drift_hours": 24.0
    }
    response = client.post("/api/spill/backtrack", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "estimated_source_area" in data
    assert "trajectory" in data
    assert len(data["trajectory"]) > 0
    assert "DRIFT" in data["disclaimer"]


def test_spill_backtrack_invalid_drift_hours(client: TestClient):
    payload = {
        "spill_location": {
            "latitude": 15.35,
            "longitude": 73.80
        },
        "timestamp": "2026-08-26T12:00:00Z",
        "drift_hours": -5.0  # Invalid negative drift hours
    }
    response = client.post("/api/spill/backtrack", json=payload)
    assert response.status_code == 422
