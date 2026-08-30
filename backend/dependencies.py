import os
from backend.services.spill_service import BaseSpillService
from backend.services.drift_service import BaseDriftService
from backend.services.ais_service import BaseAISService
from backend.services.vessel_service import BaseVesselService

from backend.services.mock.spill import MockSpillService
from backend.services.mock.drift import MockDriftService
from backend.services.mock.ais import MockAISService
from backend.services.mock.vessel import MockVesselService

from backend.services.real.spill_ml_service import RealSpillServiceAdapter
from backend.services.real.drift_physics_service import RealDriftServiceAdapter
from backend.services.real.ais_stream_service import RealAISServiceAdapter
from backend.services.real.vessel_attribution_service import RealVesselAttributionService

# Mock Fallback Singletons
_mock_spill_service: BaseSpillService = MockSpillService()
_mock_drift_service: BaseDriftService = MockDriftService()
_mock_ais_service: BaseAISService = MockAISService()
_mock_vessel_service: BaseVesselService = MockVesselService()

# Real Adapter Singletons
_real_spill_service: BaseSpillService = RealSpillServiceAdapter()
_real_drift_service: BaseDriftService = RealDriftServiceAdapter()
_real_ais_service: BaseAISService = RealAISServiceAdapter()
_real_vessel_service: BaseVesselService = RealVesselAttributionService()


def get_spill_service() -> BaseSpillService:
    """
    Dependency provider for Satellite Spill Detection Service.
    Returns RealSpillServiceAdapter (Member 1) or MockSpillService if USE_MOCK_SPILL_SERVICE env var is set.
    """
    if os.getenv("USE_MOCK_SPILL_SERVICE", "false").lower() in ("true", "1", "yes"):
        return _mock_spill_service
    return _real_spill_service


def get_drift_service() -> BaseDriftService:
    """
    Dependency provider for Ocean Drift Service.
    Returns RealDriftServiceAdapter (Member 2) or MockDriftService if USE_MOCK_DRIFT_SERVICE env var is set.
    """
    if os.getenv("USE_MOCK_DRIFT_SERVICE", "false").lower() in ("true", "1", "yes"):
        return _mock_drift_service
    return _real_drift_service


def get_ais_service() -> BaseAISService:
    """
    Dependency provider for AIS Data Query Service.
    Returns RealAISServiceAdapter (Member 3) or MockAISService if USE_MOCK_AIS_SERVICE env var is set.
    """
    if os.getenv("USE_MOCK_AIS_SERVICE", "false").lower() in ("true", "1", "yes"):
        return _mock_ais_service
    return _real_ais_service


def get_vessel_service() -> BaseVesselService:
    """
    Dependency provider for Vessel Attribution Service.
    Returns RealVesselAttributionService (Member 4) or MockVesselService if USE_MOCK_VESSEL_SERVICE env var is set.
    """
    if os.getenv("USE_MOCK_VESSEL_SERVICE", "false").lower() in ("true", "1", "yes"):
        return _mock_vessel_service
    return _real_vessel_service


# Explicit Mock Provider Factories for Testing
def get_mock_spill_service() -> BaseSpillService:
    return _mock_spill_service

def get_mock_drift_service() -> BaseDriftService:
    return _mock_drift_service

def get_mock_ais_service() -> BaseAISService:
    return _mock_ais_service

def get_mock_vessel_service() -> BaseVesselService:
    return _mock_vessel_service
