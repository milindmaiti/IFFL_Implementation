"""Fairness radius schemes.

Approach A (LP rounding) handles arbitrary nonuniform ``R_j``.  Approach B
(primal-dual) proves its (3,3) bound only for *uniform* ``R``: its witness argument
evaluates to ``R_j + R_{j*} + R_{j*}``, so a witness with a much larger radius destroys
any bound stated relative to ``R_j``.  Both regimes are provided so the
two algorithms can be compared on the same instances.

Every scheme is clamped to feasibility.  A client whose radius falls below its
nearest-facility distance has no admissible edge, which makes ``E_R`` empty for that
client and the restricted LP infeasible.  Each scheme reports how many clients it had
to clamp, since a high clamp count means the requested radius profile was not really
what got used.
"""

from __future__ import annotations

import math

import numpy as np

from iffl.instance import Instance, l2_matrix

SCHEMES = ("nearest_facility_multiple", "uniform", "jkl_client_knn")


def _clamp(R: np.ndarray, floor: np.ndarray) -> tuple[np.ndarray, int]:
    """Raise any radius below its feasibility floor; return the clamp count."""
    n_clamped = int(np.count_nonzero(R < floor - 1e-12))
    return np.maximum(R, floor), n_clamped


def facility_distance_basis(inst: Instance, rank: int = 1) -> np.ndarray:
    """``(n,)`` distance from each client to its ``rank``-th nearest facility.

    Both distance-based schemes build on this.  ``rank`` matters whenever clients sit on
    facility sites -- as they do in both UFLP workbooks, where clients and facilities are
    the same point set.  There the rank-1 distance is identically 0, which collapses
    every radius to 0 regardless of the scale parameter: ``E_R`` shrinks to the
    self-edges, each client is served by itself, and the instance becomes trivial.
    Rank 2 recovers a meaningful profile.
    """
    if rank < 1:
        raise ValueError("rank must be at least 1")
    r = min(rank, inst.m) - 1
    if inst.m == 1:
        return inst.d[:, 0]
    return np.partition(inst.d, r, axis=1)[:, r]


def nearest_facility_multiple(
    inst: Instance, alpha: float, rank: int = 1
) -> tuple[np.ndarray, dict]:
    """``R_j = alpha * d(j, rank-th nearest facility)``.

    Nonuniform, and feasible by construction for ``alpha >= 1`` -- the natural scheme
    for Approach A.  ``alpha`` controls how much slack each client gets: ``alpha = 1``
    with ``rank = 1`` pins every client to its single closest facility.

    A client sitting *exactly* on a facility site gets ``R_j = alpha * 0 = 0``, which
    means it must be served from that precise location.  That is a legitimate instance,
    not an error, and the reported ``radius_n_zero`` count makes it visible: one client
    of ``adult-429-65`` is such a case, and it is the sole reason Approach B reports an
    infinite radius ratio on that dataset.

    ``rank`` exists because of the UFLP workbooks, where clients and facilities are the
    *same* point set.  There every client sits on a facility site, so its
    nearest-facility distance is exactly 0 and ``alpha * 0 = 0`` collapses the radius
    for every ``alpha``: ``E_R`` degenerates to the self-edges, each client is served by
    itself, and the LP is forced to open all m facilities.  Basing the multiple on the
    ``rank``-th nearest facility (``rank = 2`` for those datasets) restores a meaningful
    radius profile.
    """
    floor = inst.nearest_facility_dist
    base = facility_distance_basis(inst, rank)
    R, n_clamped = _clamp(alpha * base, floor)
    return R, {
        "radius_scheme": "nearest_facility_multiple",
        "radius_param": alpha,
        "radius_rank": rank,
        "radius_clamped": n_clamped,
        "radius_degenerate": bool(np.all(R <= 1e-12)),
        "radius_n_zero": int(np.count_nonzero(R <= 1e-12)),
        "radius_uniform": bool(np.ptp(R) <= 1e-12),
    }


def uniform(
    inst: Instance, q: float, alpha: float = 1.0, rank: int = 1
) -> tuple[np.ndarray, dict]:
    """A single radius shared by every client -- what Approach B requires.

    ``R = max(alpha * quantile(basis, q), max_j nearest_j)`` where ``basis`` is the
    distance to each client's ``rank``-th nearest facility.

    The outer ``max`` is what keeps the result both uniform *and* feasible.  Taking a
    bare percentile and then clamping per client would leave the tail clients with
    larger radii than everyone else, quietly violating the uniformity the (3,3) proof
    depends on; lifting the shared value instead preserves it.  The effective quantile
    actually realised is reported back.

    ``rank`` plays the same role as in :func:`nearest_facility_multiple`: with clients
    sitting on facility sites the rank-1 basis is all zeros, so the shared radius would
    collapse to 0 and make the instance trivial.
    """
    floor = inst.nearest_facility_dist
    basis = facility_distance_basis(inst, rank)
    target = alpha * float(np.quantile(basis, q))
    value = max(target, float(floor.max()))
    R = np.full(inst.n, value, dtype=float)
    effective_q = float(np.mean(basis <= value))
    return R, {
        "radius_scheme": "uniform",
        "radius_param": q,
        "radius_alpha": alpha,
        "radius_rank": rank,
        "radius_value": value,
        "radius_requested_value": target,
        "radius_lifted_for_feasibility": bool(value > target + 1e-12),
        "radius_effective_quantile": effective_q,
        "radius_clamped": 0,
        "radius_degenerate": bool(value <= 1e-12),
        "radius_n_zero": int(inst.n) if value <= 1e-12 else 0,
        "radius_uniform": True,
    }


def jkl_client_knn(inst: Instance, k: int) -> tuple[np.ndarray, dict]:
    """Jung-Kannan-Lutz radii: distance from ``j`` to its ``ceil(n/k)``-th nearest client.

    The standard individual-fairness radius from the fair-clustering literature: if the
    clients were split into ``k`` equal groups, each client could expect a center within
    this distance.  It is defined purely on the client point set, so it carries no
    guarantee of being feasible against *this* facility set -- the clamp count tells you
    how often it had to be raised.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    n = inst.n
    rank = min(max(1, math.ceil(n / k)), n - 1) if n > 1 else 0
    dcc = l2_matrix(inst.client_pts, inst.client_pts)
    # Row r holds the distance to the r-th nearest *other* client (column 0 is self).
    raw = np.partition(dcc, rank, axis=1)[:, rank] if n > 1 else np.zeros(n)

    floor = inst.nearest_facility_dist
    R, n_clamped = _clamp(raw, floor)
    return R, {
        "radius_scheme": "jkl_client_knn",
        "radius_param": k,
        "radius_rank": rank,
        "radius_clamped": n_clamped,
        "radius_degenerate": bool(np.all(R <= 1e-12)),
        "radius_n_zero": int(np.count_nonzero(R <= 1e-12)),
        "radius_uniform": bool(np.ptp(R) <= 1e-12),
    }


def apply_scheme(inst: Instance, scheme: str, param: float, **kwargs) -> Instance:
    """Attach radii to ``inst`` using the named scheme."""
    if scheme == "nearest_facility_multiple":
        R, meta = nearest_facility_multiple(
            inst, float(param), rank=int(kwargs.get("rank", 1))
        )
    elif scheme == "uniform":
        R, meta = uniform(
            inst,
            float(param),
            alpha=float(kwargs.get("alpha", 1.0)),
            rank=int(kwargs.get("rank", 1)),
        )
    elif scheme == "jkl_client_knn":
        R, meta = jkl_client_knn(inst, int(param))
    else:
        raise KeyError(f"unknown radius scheme {scheme!r}; known: {', '.join(SCHEMES)}")
    out = inst.with_radii(R, **meta)
    out.check_feasible()
    return out
