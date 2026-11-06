"""Synthetic delivery-day data for a Central New Jersey last-mile operation.

All data is generated (no real customer or employer data). A fixed random
seed makes every run reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Depot: a delivery station in the Edison, NJ warehouse corridor.
DEPOT = {"stop_id": "DEPOT", "town": "Edison (station)", "lat": 40.5187, "lon": -74.4121}

# Towns served, with approximate centre coordinates.
TOWNS = {
    "Carteret": (40.5773, -74.2282),
    "Woodbridge": (40.5576, -74.2846),
    "Perth Amboy": (40.5068, -74.2654),
    "Metuchen": (40.5434, -74.3632),
    "Piscataway": (40.5551, -74.4610),
    "New Brunswick": (40.4862, -74.4518),
    "Highland Park": (40.4959, -74.4243),
    "East Brunswick": (40.4279, -74.4160),
    "Sayreville": (40.4590, -74.3610),
    "South Plainfield": (40.5793, -74.4115),
    "Rahway": (40.6082, -74.2776),
    "Old Bridge": (40.4148, -74.3654),
}


@dataclass(frozen=True)
class FleetParams:
    """Operating assumptions. Every number here is an editable assumption."""

    van_capacity: int = 150          # packages per van
    max_vans: int = 8                # vans available at the station
    avg_speed_mph: float = 25.0      # suburban stop-and-go average
    circuity: float = 1.3            # road miles per straight-line mile
    service_min_per_stop: float = 4.0
    service_min_per_pkg: float = 0.25
    cost_per_mile: float = 0.70      # fuel + maintenance
    driver_cost_per_hour: float = 28.0
    co2_kg_per_mile: float = 0.95    # diesel step van, approximate


def generate_stops(n_stops: int = 30, seed: int = 42, volume_multiplier: float = 1.0) -> pd.DataFrame:
    """Return a DataFrame of delivery stops; row 0 is the depot.

    Stops are scattered around the served towns. Package counts are drawn
    from a skewed distribution (most stops are small, a few are large
    business deliveries), then scaled by ``volume_multiplier`` for scenarios
    such as peak season.
    """
    rng = np.random.default_rng(seed)
    towns = list(TOWNS)
    rows = [dict(DEPOT, packages=0)]
    for i in range(n_stops):
        town = towns[rng.integers(len(towns))]
        lat0, lon0 = TOWNS[town]
        lat = lat0 + rng.normal(0, 0.010)
        lon = lon0 + rng.normal(0, 0.013)
        base = int(np.clip(rng.lognormal(mean=2.4, sigma=0.6), 2, 60))
        pkgs = max(1, int(round(base * volume_multiplier)))
        rows.append({"stop_id": f"S{i + 1:02d}", "town": town, "lat": round(lat, 5),
                     "lon": round(lon, 5), "packages": pkgs})
    return pd.DataFrame(rows)
