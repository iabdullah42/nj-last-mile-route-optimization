"""Exact capacitated VRP solved as a mixed-integer linear program.

Formulation (two-index, directed):

    minimise    sum_ij  d_ij * x_ij
    subject to  sum_i x_ij = 1                 for every stop j   (one arrival)
                sum_j x_ij = 1                 for every stop i   (one departure)
                sum_j x_0j = sum_i x_i0                            (vans return)
                ceil(D / Q) <= sum_j x_0j <= K                     (fleet size)
                sum_{i,j in S} x_ij <= |S| - ceil(d(S) / Q)        (capacity cuts)
                x_ij in {0, 1}

There are exponentially many capacity cuts, so they are added lazily: solve,
look for subtours or overloaded routes, add the violated cuts, solve again.
Each MILP solved along the way is a relaxation of the full problem, so its
objective is a valid lower bound; the loop stops when the solution is a
feasible set of routes (proven optimal) or the time limit is reached.

Solved with SciPy's interface to the open-source HiGHS solver.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

Route = list[int]


@dataclass
class ExactResult:
    routes: list[Route] | None
    lower_bound: float
    proven_optimal: bool
    iterations: int
    cuts_added: int
    seconds: float
    log: list[str] = field(default_factory=list)


def _extract(x: np.ndarray, arcs: list[tuple[int, int]], n: int) -> tuple[list[Route], list[list[int]]]:
    """Split a 0/1 arc solution into depot routes and depot-free cycles."""
    succ: dict[int, list[int]] = {i: [] for i in range(n)}
    for val, (i, j) in zip(x, arcs):
        if val > 0.5:
            succ[i].append(j)
    routes, seen = [], set()
    for start in succ[0]:
        route, cur = [], start
        while cur != 0:
            route.append(cur)
            seen.add(cur)
            cur = succ[cur][0]
        routes.append(route)
    cycles = []
    for s in range(1, n):
        if s in seen:
            continue
        cyc, cur = [], s
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            cur = succ[cur][0]
        cycles.append(cyc)
    return routes, cycles


def solve_cvrp(dist: np.ndarray, demand: np.ndarray, capacity: int, max_vehicles: int,
               time_limit: float = 120.0, verbose: bool = False) -> ExactResult:
    n = len(demand)
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx = {a: k for k, a in enumerate(arcs)}
    m = len(arcs)
    c = np.array([dist[i, j] for i, j in arcs])
    min_vehicles = math.ceil(demand[1:].sum() / capacity)

    # Fixed degree / fleet constraints.
    rows = []
    A = lil_matrix((2 * (n - 1) + 2, m))
    lb, ub = [], []
    r = 0
    for j in range(1, n):                       # one arrival
        for i in range(n):
            if i != j:
                A[r, idx[(i, j)]] = 1
        lb.append(1); ub.append(1); r += 1
    for i in range(1, n):                       # one departure
        for j in range(n):
            if i != j:
                A[r, idx[(i, j)]] = 1
        lb.append(1); ub.append(1); r += 1
    for j in range(1, n):                       # depot out == depot in
        A[r, idx[(0, j)]] = 1
        A[r, idx[(j, 0)]] = -1
    lb.append(0); ub.append(0); r += 1
    for j in range(1, n):                       # fleet size
        A[r, idx[(0, j)]] = 1
    lb.append(min_vehicles); ub.append(max_vehicles); r += 1
    rows.append(LinearConstraint(A.tocsr(), lb, ub))

    # Lazy capacity cuts, seeded with all 2-stop cycles.
    cut_sets: list[frozenset[int]] = [frozenset((i, j)) for i in range(1, n) for j in range(i + 1, n)]
    seen_cuts = set(cut_sets)

    def cut_constraint(sets: list[frozenset[int]]) -> LinearConstraint:
        C = lil_matrix((len(sets), m))
        rhs = []
        for k, S in enumerate(sets):
            for i in S:
                for j in S:
                    if i != j:
                        C[k, idx[(i, j)]] = 1
            rhs.append(len(S) - math.ceil(demand[list(S)].sum() / capacity))
        return LinearConstraint(C.tocsr(), -np.inf, rhs)

    start = time.perf_counter()
    lower_bound, it, log = 0.0, 0, []
    while True:
        it += 1
        remaining = time_limit - (time.perf_counter() - start)
        if remaining <= 0:
            return ExactResult(None, lower_bound, False, it - 1, len(cut_sets), time.perf_counter() - start, log)
        res = milp(c, constraints=[*rows, cut_constraint(cut_sets)], integrality=np.ones(m),
                   bounds=Bounds(0, 1), options={"time_limit": remaining, "disp": False})
        if res.x is None:
            return ExactResult(None, lower_bound, False, it, len(cut_sets), time.perf_counter() - start, log)
        if res.status == 0:
            lower_bound = max(lower_bound, res.fun)
        routes, cycles = _extract(res.x, arcs, n)
        violated = [frozenset(cy) for cy in cycles]
        violated += [frozenset(rt) for rt in routes if demand[rt].sum() > capacity]
        new = [S for S in violated if S not in seen_cuts]
        msg = f"iter {it}: obj {res.fun:.2f}, subtours {len(cycles)}, overloaded {len(violated) - len(cycles)}"
        log.append(msg)
        if verbose:
            print(msg)
        if not violated and res.status == 0:
            return ExactResult(routes, res.fun, True, it, len(cut_sets), time.perf_counter() - start, log)
        if not new:  # time limit hit inside HiGHS; solution not proven
            return ExactResult(None, lower_bound, False, it, len(cut_sets), time.perf_counter() - start, log)
        cut_sets += new
        seen_cuts.update(new)
