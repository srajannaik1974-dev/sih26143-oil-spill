"""Backward-compatible wrapper around the canonical AIS Haversine implementation.

This module is kept only for compatibility with older imports. The authoritative
code lives in the canonical ais package.
"""

from ais.filters import haversine_distance

__all__ = ["haversine_distance"]

