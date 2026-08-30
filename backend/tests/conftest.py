import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    """
    TestClient fixture for FastAPI application tests.
    """
    with TestClient(app) as test_client:
        yield test_client
