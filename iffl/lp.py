"""The restricted LP shared by both papers.

Both formulations are the standard facility-location LP restricted to the admissible
edge set ``E_R = {(i, j) : d(i, j) <= R_j}``::

    min  sum_i f_i y_i + sum_{(i,j) in E_R} d(i,j) x_ij
    s.t. sum_{i : (i,j) in E_R} x_ij = 1       for every client j
         x_ij <= y_i                           for every (i,j) in E_R
         0 <= x_ij,  0 <= y_i <= 1

Paper A writes the covering constraint with ``=`` and Paper B with ``>=``; with metric
connection costs there is never a reason to over-assign, so equality is without loss of
generality and gives the solver a tighter system.

Restricting ``x`` to ``E_R`` is a correctness requirement, not an optimization: any
integral solution respecting strict fairness uses only those edges, which is exactly why
this LP's optimum lower-bounds ``OPT_fair``.  Variables for forbidden pairs would let
the LP cheat and destroy the bound both guarantees are measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_array

from iffl.instance import Instance


@dataclass
class RestrictedLP:
    """Optimal solution of the restricted LP.

    Attributes:
        obj: optimal value; the lower bound on ``OPT_fair``.
        y: ``(m,)`` fractional opening variables.
        x: ``(n_edges,)`` fractional assignment variables, aligned with
            ``edge_client`` / ``edge_facility``.
        edge_client: ``(n_edges,)`` client index of each edge.
        edge_facility: ``(n_edges,)`` facility index of each edge.
        edge_dist: ``(n_edges,)`` connection cost of each edge.
        F_star: fractional facility cost ``sum_i f_i y_i``.
        C_star: ``(n,)`` per-client fractional connection cost.
    """

    obj: float
    y: np.ndarray
    x: np.ndarray
    edge_client: np.ndarray
    edge_facility: np.ndarray
    edge_dist: np.ndarray
    F_star: float
    C_star: np.ndarray

    @property
    def n_edges(self) -> int:
        return self.x.size

    def edges_of_client(self) -> list[np.ndarray]:
        """For each client, the indices of its edges with positive ``x``."""
        n = self.C_star.size
        buckets: list[list[int]] = [[] for _ in range(n)]
        for e, j in enumerate(self.edge_client):
            buckets[int(j)].append(e)
        return [np.asarray(b, dtype=int) for b in buckets]


def solve_restricted_lp(inst: Instance, tol: float = 1e-9) -> RestrictedLP:
    """Solve the restricted LP with HiGHS.

    Raises:
        ValueError: if the instance has no radii, or the LP is infeasible (which for a
            feasible instance means a radius slipped below its nearest-facility floor).
    """
    if inst.R is None:
        raise ValueError(f"{inst.name}: instance has no radii")
    inst.check_feasible()

    js, is_ = inst.admissible_edges()
    n_edges = js.size
    n, m = inst.n, inst.m
    if n_edges == 0:
        raise ValueError(f"{inst.name}: E_R is empty")

    edge_dist = inst.d[js, is_]
    n_var = n_edges + m

    # Objective: connection costs on edges, opening costs on facilities.
    c = np.concatenate([edge_dist, inst.f])

    # Covering: one row per client, summing that client's edges to 1.
    A_eq = csr_array(
        (np.ones(n_edges), (js, np.arange(n_edges))),
        shape=(n, n_var),
    )
    b_eq = np.ones(n)

    # Linking: x_e - y_i <= 0, one row per edge.
    rows = np.repeat(np.arange(n_edges), 2)
    cols = np.empty(2 * n_edges, dtype=int)
    cols[0::2] = np.arange(n_edges)
    cols[1::2] = n_edges + is_
    vals = np.empty(2 * n_edges)
    vals[0::2] = 1.0
    vals[1::2] = -1.0
    A_ub = csr_array((vals, (rows, cols)), shape=(n_edges, n_var))
    b_ub = np.zeros(n_edges)

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not res.success:
        raise ValueError(f"{inst.name}: restricted LP failed -- {res.message}")

    x = np.clip(res.x[:n_edges], 0.0, 1.0)
    y = np.clip(res.x[n_edges:], 0.0, 1.0)
    x[x < tol] = 0.0
    y[y < tol] = 0.0

    C_star = np.zeros(n)
    np.add.at(C_star, js, edge_dist * x)

    return RestrictedLP(
        obj=float(res.fun),
        y=y,
        x=x,
        edge_client=js,
        edge_facility=is_,
        edge_dist=edge_dist,
        F_star=float(inst.f @ y),
        C_star=C_star,
    )
