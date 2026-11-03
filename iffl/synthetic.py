"""Synthetic instances, including ones whose restricted LP is genuinely fractional.

Every one of the eight Fair-FL datasets yields an *integral* LP optimum under every
radius scheme tried.  That is common for geometric facility-location instances, and it
means Approach A degenerates there: with ``y* in {0,1}`` each client's close set lies
inside a single facility, so the rounding opens exactly the LP's facilities and returns
cost ratio 1.000 deterministically.  Correct, but it exercises none of the interesting
machinery -- facility splitting across shared supports, the global filtering refinement,
cluster formation, or the randomized rounding itself.

:func:`odd_cycle_instance` builds the classic fractional witness.  Facilities sit on the
vertices of an odd cycle and clients on its edges, with the shortest-path metric of the
incidence graph (metric by construction, as both papers require).  Restricting to
``R_j = 1`` makes each client admissible only to its two endpoints, so the LP reduces to
fractional vertex cover of an odd cycle: ``y_i = 1/2`` everywhere, costing ``k/2``, while
any integral cover needs ``ceil(k/2)`` facilities.  That half-integral solution forces
copies to be shared between clients and cut at different offsets, which is exactly the
path the real datasets never reach.
"""

from __future__ import annotations

import numpy as np

from iffl.instance import Instance


def odd_cycle_instance(k: int = 5, facility_cost: float = 10.0) -> Instance:
    """A fractional-LP instance on an odd cycle of length ``k``.

    Facilities are the ``k`` cycle vertices, clients the ``k`` cycle edges, and the
    distance is the shortest-path metric of the bipartite incidence graph (client ``j``
    joined to its two endpoints with unit length).  Deriving distances from shortest
    paths rather than a closed-form hop count is what guarantees the triangle inequality
    across the whole of ``F union C``, which both papers assume -- the 3R argument walks
    a client -> facility -> client -> facility path, so a merely plausible-looking
    distance table is not enough.

    Args:
        k: cycle length; must be odd and at least 3 for the LP to be fractional.
        facility_cost: uniform opening cost.  Larger values widen the integrality gap.

    Returns:
        An instance with ``R_j = 1`` for every client, whose LP optimum has
        ``y_i = 1/2`` for all ``i``.
    """
    if k < 3 or k % 2 == 0:
        raise ValueError("k must be odd and at least 3 for a fractional LP")

    from scipy.sparse import csr_array
    from scipy.sparse.csgraph import shortest_path

    # Nodes 0..k-1 are facilities (vertices); nodes k..2k-1 are clients (edges).
    rows, cols = [], []
    for j in range(k):
        for i in (j, (j + 1) % k):
            rows += [k + j, i]
            cols += [i, k + j]
    adj = csr_array(([1.0] * len(rows), (rows, cols)), shape=(2 * k, 2 * k))
    full = shortest_path(adj, method="D", directed=False)

    # Client rows, facility columns.
    d = full[k:, :k]

    ang = 2.0 * np.pi * np.arange(k) / k
    verts = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    edges = 0.5 * (verts + np.roll(verts, -1, axis=0))

    inst = Instance(
        name=f"oddcycle-{k}",
        client_pts=edges,
        facility_pts=verts,
        f=np.full(k, float(facility_cost)),
        d=d,
        groups={"c1": np.array(["edge"] * k)},
        meta={
            "source": "synthetic",
            "has_native_costs": True,
            "fractional_lp": True,
            "full_metric": full,
            "note": "LP optimum is half-integral (fractional vertex cover of an odd cycle)",
        },
    )
    return inst.with_radii(
        np.ones(k),
        radius_scheme="odd_cycle_unit",
        radius_param=1.0,
        radius_clamped=0,
        radius_uniform=True,
    )


def violates_triangle(full: np.ndarray, tol: float = 1e-9) -> float:
    """Largest triangle-inequality violation ``d(a,c) - d(a,b) - d(b,c)``; <= 0 if metric."""
    worst = -np.inf
    n = full.shape[0]
    for b in range(n):
        viol = full - (full[:, [b]] + full[[b], :])
        worst = max(worst, float(viol.max()))
    return worst


def line_instance() -> Instance:
    """A small hand-checkable instance: 3 facilities and 4 clients on a line.

    Clients sit at 0, 1, 9, 10 and facilities at 0, 5, 10.  With ``R = 2`` the middle
    facility is admissible to nobody, so the LP must open both ends and pay
    ``2 * f + (0 + 1 + 1 + 0)``.
    """
    client_pts = np.array([[0.0], [1.0], [9.0], [10.0]])
    facility_pts = np.array([[0.0], [5.0], [10.0]])
    d = np.abs(client_pts - facility_pts.T)
    inst = Instance(
        name="line-3x4",
        client_pts=client_pts,
        facility_pts=facility_pts,
        f=np.array([1.0, 1.0, 1.0]),
        d=d,
        meta={"source": "synthetic", "has_native_costs": True},
    )
    return inst.with_radii(
        np.full(4, 2.0),
        radius_scheme="manual",
        radius_param=2.0,
        radius_clamped=0,
        radius_uniform=True,
    )
