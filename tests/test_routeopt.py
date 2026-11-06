import itertools

import numpy as np
import pytest

from routeopt.data import FleetParams, generate_stops
from routeopt.distance import haversine_matrix, road_miles
from routeopt.evaluate import check_routes, summarize
from routeopt.exact import solve_cvrp
from routeopt.heuristics import clarke_wright, improve, nearest_neighbor, total_length, two_opt


@pytest.fixture
def small_day():
    df = generate_stops(n_stops=8, seed=1)
    dist = road_miles(df["lat"].to_numpy(), df["lon"].to_numpy())
    return dist, df["packages"].to_numpy(), 60


def brute_force_cvrp(dist, demand, capacity):
    """Exact answer by enumerating every split of every stop order (tiny inputs only)."""
    stops = list(range(1, len(demand)))
    best = np.inf
    for perm in itertools.permutations(stops):
        n = len(perm)
        # Dynamic programme over split points of this fixed order.
        cost = [0.0] + [np.inf] * n
        for i in range(n):
            load, path = 0, 0.0
            for j in range(i, n):
                load += demand[perm[j]]
                if load > capacity:
                    break
                path += dist[0, perm[i]] if j == i else dist[perm[j - 1], perm[j]]
                route = path + dist[perm[j], 0]
                cost[j + 1] = min(cost[j + 1], cost[i] + route)
        best = min(best, cost[n])
    return best


def test_haversine_known_distance():
    # Newark Penn Station to New York Penn Station is about 8.9 miles in a straight line.
    d = haversine_matrix(np.array([40.7347, 40.7506]), np.array([-74.1644, -73.9935]))
    assert d[0, 1] == pytest.approx(8.9, abs=0.3)
    assert np.allclose(d, d.T) and np.all(np.diag(d) == 0)


def test_data_is_reproducible():
    a, b = generate_stops(20, seed=3), generate_stops(20, seed=3)
    assert a.equals(b)
    assert a.loc[0, "stop_id"] == "DEPOT" and a.loc[0, "packages"] == 0


@pytest.mark.parametrize("builder", [nearest_neighbor, clarke_wright])
def test_heuristics_are_feasible(small_day, builder):
    dist, q, cap = small_day
    routes = builder(dist, q, cap)
    check_routes(routes, q, cap)
    check_routes(improve(routes, dist, q, cap), q, cap)


def test_local_search_never_makes_it_worse(small_day):
    dist, q, cap = small_day
    routes = nearest_neighbor(dist, q, cap)
    assert total_length(improve(routes, dist, q, cap), dist) <= total_length(routes, dist) + 1e-9
    for r in routes:
        assert sorted(two_opt(r, dist)) == sorted(r)


def test_exact_matches_brute_force(small_day):
    dist, q, cap = small_day
    res = solve_cvrp(dist, q, cap, max_vehicles=8, time_limit=60)
    assert res.proven_optimal
    check_routes(res.routes, q, cap)
    assert total_length(res.routes, dist) == pytest.approx(brute_force_cvrp(dist, q, cap), rel=1e-6)


def test_exact_is_no_worse_than_heuristics(small_day):
    dist, q, cap = small_day
    res = solve_cvrp(dist, q, cap, max_vehicles=8, time_limit=60)
    heur = improve(clarke_wright(dist, q, cap), dist, q, cap)
    assert total_length(res.routes, dist) <= total_length(heur, dist) + 1e-6


def test_check_routes_catches_overload():
    q = np.array([0, 50, 50])
    with pytest.raises(AssertionError):
        check_routes([[1, 2]], q, capacity=60)


def test_summary_cost_math():
    dist = np.array([[0, 10.0], [10.0, 0]])
    q = np.array([0, 10])
    p = FleetParams()
    s = summarize("t", [[1]], dist, q, p)
    hours = 20 / p.avg_speed_mph + (p.service_min_per_stop + 10 * p.service_min_per_pkg) / 60
    assert s["miles"] == 20
    assert s["daily_cost_usd"] == pytest.approx(20 * p.cost_per_mile + round(hours, 2) * p.driver_cost_per_hour, abs=0.01)
