"""Plan a delivery day three ways and compare the results.

    python run.py                 # normal day + peak-season scenario
    python run.py --skip-peak     # normal day only (fastest)
    python run.py --stops 25 --seed 7
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from routeopt.data import FleetParams, generate_stops
from routeopt.distance import road_miles
from routeopt.evaluate import route_table, summarize
from routeopt.exact import solve_cvrp
from routeopt.heuristics import clarke_wright, improve, nearest_neighbor
from routeopt.plots import comparison_bars, route_maps

ROOT = Path(__file__).parent
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "results" / "figures"


def plan_day(label: str, stops: int, seed: int, volume: float, p: FleetParams, time_limit: float):
    df = generate_stops(stops, seed, volume)
    dist = road_miles(df["lat"].to_numpy(), df["lon"].to_numpy(), p.circuity)
    q = df["packages"].to_numpy()

    plans, timings = {}, {}
    t = time.perf_counter(); plans["Baseline (nearest neighbor)"] = nearest_neighbor(dist, q, p.van_capacity)
    timings["Baseline (nearest neighbor)"] = time.perf_counter() - t
    t = time.perf_counter(); plans["Heuristic (Clarke-Wright + local search)"] = improve(
        clarke_wright(dist, q, p.van_capacity), dist, q, p.van_capacity)
    timings["Heuristic (Clarke-Wright + local search)"] = time.perf_counter() - t
    exact = solve_cvrp(dist, q, p.van_capacity, p.max_vans, time_limit=time_limit)
    if exact.routes is not None:
        plans["Optimized (exact MILP)"] = exact.routes
        timings["Optimized (exact MILP)"] = exact.seconds

    rows = []
    for name, routes in plans.items():
        s = summarize(name, routes, dist, q, p)
        s["solve_seconds"] = round(timings[name], 2)
        rows.append(s)
    summary = pd.DataFrame(rows)
    summary.insert(0, "scenario", label)
    print(f"\n=== {label}: {stops} stops, {q.sum()} packages ===")
    print(summary.drop(columns="scenario").to_string(index=False))
    print(f"Exact solver: proven optimal={exact.proven_optimal}, lower bound={exact.lower_bound:.2f} mi, "
          f"{exact.iterations} rounds, {exact.cuts_added} cuts, {exact.seconds:.1f}s")
    return df, dist, q, plans, summary, exact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stops", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--peak-volume", type=float, default=1.3, help="package multiplier for the peak scenario")
    ap.add_argument("--skip-peak", action="store_true")
    ap.add_argument("--time-limit", type=float, default=600, help="seconds for the exact solver")
    args = ap.parse_args()

    p = FleetParams()
    for d in (DATA, RESULTS, FIGS):
        d.mkdir(parents=True, exist_ok=True)

    df, dist, q, plans, summary, _ = plan_day("Normal day", args.stops, args.seed, 1.0, p, args.time_limit)
    df.to_csv(DATA / "stops.csv", index=False)
    summaries = [summary]

    base, opt = "Baseline (nearest neighbor)", "Optimized (exact MILP)"
    best = opt if opt in plans else "Heuristic (Clarke-Wright + local search)"
    route_maps(df, plans[base], plans[best], dist,
               summary.set_index("plan").loc[base, "miles"], summary.set_index("plan").loc[best, "miles"],
               FIGS / "route_maps.png")
    comparison_bars(summary, base, best, FIGS / "baseline_vs_optimized.png")
    route_table(plans[best], dist, q, p).to_csv(RESULTS / "optimized_routes.csv", index=False)

    if not args.skip_peak:
        *_, peak, _ = plan_day(f"Peak season (+{args.peak_volume - 1:.0%} volume)", args.stops, args.seed,
                               args.peak_volume, p, args.time_limit)
        summaries.append(peak)

    pd.concat(summaries).to_csv(RESULTS / "summary.csv", index=False)
    print(f"\nWrote {RESULTS / 'summary.csv'} and figures in {FIGS}")


if __name__ == "__main__":
    main()
