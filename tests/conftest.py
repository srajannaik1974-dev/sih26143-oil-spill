"""tests/conftest.py
====================
SIH 2026 — PS 26143: SAR Oil-Spill Detection API
Shared pytest fixtures.

All fixtures use mocks — no real model, no real dataset required.
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal valid TIFF bytes (1×1 pixel, 2 bands)
# Created entirely in-memory with rasterio — no file on disk needed.
# ---------------------------------------------------------------------------

def _make_minimal_tiff() -> bytes:
    """
    Build the smallest valid 2-band float32 GeoTIFF in memory.
    Returns raw bytes suitable for uploading in tests.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_bounds

        data = np.zeros((2, 4, 4), dtype=np.float32)  # 2-channel, 4×4
        data[0] = -10.0  # VV channel — typical SAR dB value
        data[1] = -15.0  # VH channel

        transform = from_bounds(0, 0, 1, 1, 4, 4)

        buf = io.BytesIO()
        with MemoryFile(buf) as memfile:
            with memfile.open(
                driver    = "GTiff",
                count     = 2,
                height    = 4,
                width     = 4,
                dtype     = "float32",
                transform = transform,
                crs       = None,
            ) as ds:
                ds.write(data)
            return memfile.read()

    except Exception:
        # If rasterio is not installed in the test env, return dummy bytes
        # The mock predictor will bypass actual TIFF parsing anyway.
        return b"FAKE_TIFF_BYTES_FOR_MOCK_TEST"


VALID_TIFF_BYTES: bytes = _make_minimal_tiff()


# ---------------------------------------------------------------------------
# Mock OilSpillPredictor
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_predictor():
    """
    A MagicMock that mimics OilSpillPredictor.predict().
    Returns a 4×4 binary mask (all zeros = no spill) and a probability map.
    Adjust the return value in individual tests to simulate spill detection.
    """
    predictor = MagicMock()
    predictor.threshold = 0.5

    binary_mask = np.zeros((4, 4), dtype=np.uint8)
    prob_map    = np.full((4, 4), 0.1, dtype=np.float32)

    predictor.predict.return_value = (binary_mask, prob_map)
    return predictor


@pytest.fixture()
def mock_predictor_with_spill():
    """Mock predictor that returns a mask with some oil-spill pixels."""
    predictor = MagicMock()
    predictor.threshold = 0.5

    binary_mask = np.zeros((4, 4), dtype=np.uint8)
    binary_mask[1:3, 1:3] = 1          # 4 spill pixels in the centre

    prob_map = np.full((4, 4), 0.1, dtype=np.float32)
    prob_map[1:3, 1:3] = 0.92          # high confidence in the spill zone

    predictor.predict.return_value = (binary_mask, prob_map)
    return predictor


# ---------------------------------------------------------------------------
# TestClient fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_no_model():
    """
    TestClient where the predictor singleton returns None (model not found).
    Simulates a freshly deployed server with no trained checkpoint.
    """
    from api.main import app
    import api.dependencies as deps

    # Reset singleton state so each test starts clean
    deps._reset_predictor()

    with patch.object(deps, "get_predictor", return_value=None), \
         patch.object(deps, "get_predictor_error",
                      return_value="Model checkpoint not found at 'ml/training/checkpoints/best_unet.pth'. "
                                   "Train the model first with: python ml/training/train.py"):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    deps._reset_predictor()


@pytest.fixture()
def client_with_mock_predictor(mock_predictor):
    """
    TestClient where the predictor singleton returns a mock predictor
    that reports no oil spill.
    """
    from api.main import app
    import api.dependencies as deps

    deps._reset_predictor()

    with patch.object(deps, "get_predictor", return_value=mock_predictor), \
         patch.object(deps, "get_predictor_error", return_value=None):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    deps._reset_predictor()


@pytest.fixture()
def client_with_spill_predictor(mock_predictor_with_spill):
    """
    TestClient where the mock predictor reports an oil spill.
    """
    from api.main import app
    import api.dependencies as deps

    deps._reset_predictor()

    with patch.object(deps, "get_predictor", return_value=mock_predictor_with_spill), \
         patch.object(deps, "get_predictor_error", return_value=None):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    deps._reset_predictor()
