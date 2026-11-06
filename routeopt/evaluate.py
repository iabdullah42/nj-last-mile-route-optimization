"""Turn routes into the operating metrics a station manager cares about."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FleetParams
from .heuristics import Route, route_length


def check_routes(routes: list[Route], demand: np.ndarray, capacity: int) -> None:
    """Raise if any stop is missed, repeated, or any van is overloaded."""
    visited = [s for r in routes for s in r]
    expected = set(range(1, len(demand)))
    if sorted(visited) != sorted(expected):
        raise AssertionError("Every stop must be visited exactly once.")
    for r in routes:
        if demand[r].sum() > capacity:
            raise AssertionError(f"Route over capacity: {demand[r].sum()} > {capacity}")


def route_table(routes: list[Route], dist: np.ndarray, demand: np.ndarray, p: FleetParams) -> pd.DataFrame:
    rows = []
    for k, r in enumerate(routes, start=1):
        miles = route_length(r, dist)
        drive_h = miles / p.avg_speed_mph
        service_h = (len(r) * p.service_min_per_stop + demand[r].sum() * p.service_min_per_pkg) / 60
        rows.append({
            "van": k,
            "stops": len(r),
            "packages": int(demand[r].sum()),
            "utilization_pct": round(100 * demand[r].sum() / p.van_capacity, 1),
            "miles": round(miles, 2),
            "drive_hours": round(drive_h, 2),
            "route_hours": round(drive_h + service_h, 2),
        })
    return pd.DataFrame(rows)


def summarize(name: str, routes: list[Route], dist: np.ndarray, demand: np.ndarray, p: FleetParams) -> dict:
    check_routes(routes, demand, p.van_capacity)
    t = route_table(routes, dist, demand, p)
    miles = t["miles"].sum()
    hours = t["route_hours"].sum()
    return {
        "plan": name,
        "vans": len(routes),
        "miles": round(miles, 1),
        "driver_hours": round(hours, 2),
        "longest_route_hours": round(t["route_hours"].max(), 2),
        "avg_utilization_pct": round(t["utilization_pct"].mean(), 1),
        "daily_cost_usd": round(miles * p.cost_per_mile + hours * p.driver_cost_per_hour, 2),
        "co2_kg": round(miles * p.co2_kg_per_mile, 1),
    }
