"""Construction and improvement heuristics for the capacitated VRP.

A route is a list of stop indices that excludes the depot (index 0);
every route starts and ends at the depot.
"""
from __future__ import annotations

import numpy as np

Route = list[int]


def route_length(route: Route, dist: np.ndarray) -> float:
    if not route:
        return 0.0
    path = [0, *route, 0]
    return float(sum(dist[a, b] for a, b in zip(path[:-1], path[1:])))


def total_length(routes: list[Route], dist: np.ndarray) -> float:
    return sum(route_length(r, dist) for r in routes)


def nearest_neighbor(dist: np.ndarray, demand: np.ndarray, capacity: int) -> list[Route]:
    """Baseline dispatcher logic: always drive to the closest stop that still fits.

    When nothing fits, the van returns to the depot and a new van starts.
    """
    n = len(demand)
    unvisited = set(range(1, n))
    routes: list[Route] = []
    while unvisited:
        route, load, here = [], 0, 0
        while True:
            options = [j for j in unvisited if load + demand[j] <= capacity]
            if not options:
                break
            nxt = min(options, key=lambda j: dist[here, j])
            route.append(nxt)
            load += demand[nxt]
            unvisited.remove(nxt)
            here = nxt
        if not route:
            raise ValueError("A single stop exceeds van capacity.")
        routes.append(route)
    return routes


def clarke_wright(dist: np.ndarray, demand: np.ndarray, capacity: int) -> list[Route]:
    """Clarke-Wright savings algorithm (parallel version).

    Start with one route per stop, then merge route ends in order of the
    savings s(i, j) = d(0, i) + d(0, j) - d(i, j) while capacity allows.
    """
    n = len(demand)
    routes: dict[int, Route] = {i: [i] for i in range(1, n)}
    owner = {i: i for i in range(1, n)}
    load = {i: int(demand[i]) for i in range(1, n)}

    savings = [
        (dist[0, i] + dist[0, j] - dist[i, j], i, j)
        for i in range(1, n) for j in range(i + 1, n)
    ]
    savings.sort(reverse=True)

    for s, i, j in savings:
        if s <= 0:
            break
        ri, rj = owner[i], owner[j]
        if ri == rj or load[ri] + load[rj] > capacity:
            continue
        a, b = routes[ri], routes[rj]
        # i and j must be at the ends of their routes to be joined.
        if a[-1] == i and b[0] == j:
            merged = a + b
        elif a[0] == i and b[-1] == j:
            merged = b + a
        elif a[-1] == i and b[-1] == j:
            merged = a + b[::-1]
        elif a[0] == i and b[0] == j:
            merged = a[::-1] + b
        else:
            continue
        routes[ri] = merged
        load[ri] += load[rj]
        del routes[rj], load[rj]
        for k in merged:
            owner[k] = ri
    return list(routes.values())


def two_opt(route: Route, dist: np.ndarray) -> Route:
    """Improve one route by reversing segments until no reversal helps."""
    best = [0, *route, 0]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                a, b, c, d = best[i - 1], best[i], best[k], best[k + 1]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-9:
                    best[i:k + 1] = best[i:k + 1][::-1]
                    improved = True
    return best[1:-1]


def relocate(routes: list[Route], dist: np.ndarray, demand: np.ndarray, capacity: int) -> list[Route]:
    """Move single stops between routes while it shortens total distance."""
    routes = [list(r) for r in routes]
    improved = True
    while improved:
        improved = False
        for ri, r in enumerate(routes):
            for pos, stop in enumerate(r):
                prev_ = r[pos - 1] if pos > 0 else 0
                next_ = r[pos + 1] if pos < len(r) - 1 else 0
                removal_gain = dist[prev_, stop] + dist[stop, next_] - dist[prev_, next_]
                best = (0.0, None, None)
                for rj, t in enumerate(routes):
                    if rj == ri or sum(demand[x] for x in t) + demand[stop] > capacity:
                        continue
                    path = [0, *t, 0]
                    for ins in range(len(path) - 1):
                        u, v = path[ins], path[ins + 1]
                        gain = removal_gain - (dist[u, stop] + dist[stop, v] - dist[u, v])
                        if gain > best[0] + 1e-9:
                            best = (gain, rj, ins)
                if best[1] is not None:
                    r.pop(pos)
                    routes[best[1]].insert(best[2], stop)
                    improved = True
                    break
            if improved:
                break
        routes = [r for r in routes if r]
    return routes


def improve(routes: list[Route], dist: np.ndarray, demand: np.ndarray, capacity: int) -> list[Route]:
    """Inter-route relocate, then 2-opt inside every route."""
    routes = relocate(routes, dist, demand, capacity)
    return [two_opt(r, dist) for r in routes]
