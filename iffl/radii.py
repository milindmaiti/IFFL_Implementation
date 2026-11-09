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


def uniform(inst: Instance, mult: float, rank: int = 1) -> tuple[np.ndarray, dict]:
    """A single radius shared by every client -- what Approach B requires.

    ``R = mult * max_j basis_j``, where ``basis`` is each client's distance to its
    ``rank``-th nearest facility.  Since ``max_j basis_j`` is the smallest uniform radius
    that leaves every client with an admissible edge, ``mult >= 1`` is feasible by
    construction and ``mult`` is exactly "how much slack beyond the tightest feasible
    uniform radius".

    This replaced a quantile parameterisation, ``R = max(quantile(basis, q), max_j nn_j)``,
    which looked reasonable but was degenerate on most of these datasets: the
    nearest-facility distribution has a long right tail -- one client far from
    everything -- so ``max_j nn_j`` exceeded even the 90th percentile and the quantile
    never won.  On six of the eight bundled datasets every ``q`` produced an identical
    radius, making the sweep's uniform configurations 4-way duplicates.  Clamping per
    client instead would restore the variation but break the uniformity the (3,3) proof
    depends on, so the scale is applied to the shared value.
    """
    if mult < 1.0:
        raise ValueError("mult must be at least 1 for the radius to stay feasible")
    floor = inst.nearest_facility_dist
    basis = facility_distance_basis(inst, rank)
    value = float(mult * basis.max())
    value = max(value, float(floor.max()))      # rank > 1 can dip below the floor
    R = np.full(inst.n, value, dtype=float)
    return R, {
        "radius_scheme": "uniform",
        "radius_param": mult,
        "radius_rank": rank,
        "radius_value": value,
        "radius_min_feasible": float(max(basis.max(), floor.max())),
        "radius_fraction_inside": float(np.mean(basis <= value)),
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
        R, meta = uniform(inst, float(param), rank=int(kwargs.get("rank", 1)))
    elif scheme == "jkl_client_knn":
        R, meta = jkl_client_knn(inst, int(param))
    else:
        raise KeyError(f"unknown radius scheme {scheme!r}; known: {', '.join(SCHEMES)}")
    out = inst.with_radii(R, **meta)
    out.check_feasible()
    return out
