"""Distance and travel-time matrices."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_MI = 3958.8


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance in miles between every pair of points."""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def road_miles(lat: np.ndarray, lon: np.ndarray, circuity: float = 1.3) -> np.ndarray:
    """Estimated road miles: straight-line distance times a circuity factor.

    A circuity factor of about 1.2-1.4 is a common planning approximation
    when no road-network API is available.
    """
    return haversine_matrix(lat, lon) * circuity
