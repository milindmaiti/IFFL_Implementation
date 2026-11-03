"""Shared metrics for both algorithms.

The one subtlety here is the radius ratio.  On the UFLP workbooks clients and facilities
are the same point set, so a client can legitimately have ``R_j = 0`` and be served at
distance 0.  The fairness constraint ``d <= 3 R_j`` is satisfied in that case, but the
ratio ``d / R_j`` is ``0/0``.  Reporting ``nan`` there would silently defeat every
``<= 3`` assertion, since ``nan <= 3`` is false but so is ``nan > 3``; reporting ``inf``
would flag a violation that did not happen.  :func:`radius_ratio` resolves it by the
constraint rather than the arithmetic.
"""

from __future__ import annotations

import numpy as np


def radius_ratio(dists: np.ndarray, R: np.ndarray) -> np.ndarray:
    """``d(j, sigma(j)) / R_j``, defined so that a satisfied constraint never reads nan.

    Where ``R_j == 0`` the ratio is 0 if the client is served at distance 0 (the
    constraint holds with equality) and ``inf`` otherwise (it genuinely fails).
    """
    dists = np.asarray(dists, dtype=float)
    R = np.asarray(R, dtype=float)
    out = np.empty_like(dists)
    zero = R <= 0.0
    np.divide(dists, R, out=out, where=~zero)
    out[zero & (dists <= 0.0)] = 0.0
    out[zero & (dists > 0.0)] = np.inf
    return out


def assignment_costs(inst, assignment: np.ndarray) -> tuple[float, float]:
    """Return ``(facility_cost, connection_cost)`` for an explicit assignment."""
    opened = np.unique(assignment)
    dists = inst.d[np.arange(inst.n), assignment]
    return float(inst.f[opened].sum()), float(dists.sum())


def threshold_round(inst, lp, eps: float = 0.01) -> np.ndarray:
    """Baseline: open every facility with ``y_i > 1 - eps``.

    This is the ``heuristic_round`` pattern from ``Fair-FL/fl-lp-pd-new.jl`` -- it has no
    approximation guarantee and returns nothing at all when the LP is fractional, which
    is exactly why the paper's rounding exists.  Included as a reference point.
    """
    return np.flatnonzero(lp.y > 1.0 - eps)


def greedy_nearest_feasible(inst) -> np.ndarray:
    """Baseline: greedily cover clients by the facility serving the most unserved ones.

    A strictly fair solution (every client inside its own ``R_j``, not ``3 R_j``), so its
    cost is an upper reference for what fairness costs without any bicriteria slack.
    """
    in_er = inst.d <= inst.R[:, None] + 1e-12
    uncovered = np.ones(inst.n, dtype=bool)
    opened: list[int] = []
    while uncovered.any():
        gain = (in_er & uncovered[:, None]).sum(axis=0)
        i = int(gain.argmax())
        if gain[i] == 0:
            raise ValueError(f"{inst.name}: instance is infeasible for greedy cover")
        opened.append(i)
        uncovered &= ~in_er[:, i]
    return np.asarray(sorted(opened), dtype=int)
