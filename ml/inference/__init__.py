"""
ml.inference — public API for oil-spill prediction.

Import the predictor directly from here:
    from ml.inference import OilSpillPredictor
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.training.inference import OilSpillPredictor  # noqa: F401

__all__ = ["OilSpillPredictor"]
