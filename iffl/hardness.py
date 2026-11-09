"""Instances encoding Set Cover, following the paper's hardness reduction.

Theorem 2 states that for any fixed radius violation ``1 <= beta < 3`` no deterministic
polynomial-time algorithm achieves a cost factor ``o(log n)`` unless P = NP.  The proof
encodes Set Cover in the admissible edges: incident facilities sit at distance one and
non-incident ones at distance three, so any violation strictly below three must still
respect the covering structure.

:func:`set_cover_instance` builds that reduction verbatim.  For a Set Cover instance with
universe ``U`` of size ``q`` and sets ``S_1..S_p``:

* two clients ``j_a0, j_a1`` per element (so ``N = |C| = 2q``), one facility per set;
* every facility has opening cost ``M = q^2``, and ``R_j = 1`` for every client;
* ``d(j_a0, j_a1) = 1``; ``d(j_ab, j_cb') = 2`` for ``a != c``; ``d(i_l, i_h) = 2``;
  ``d(j_ab, i_l) = 1`` if ``u_a`` is in ``S_l`` and ``3`` otherwise.

All distances between distinct points lie in ``{1, 2, 3}`` and form a metric, and
``OPT_FL = M k* + 2q`` where ``k*`` is the minimum cover size.  Because ``M = q^2``
dominates the ``2q`` connection term, the cost ratio of a *strictly fair* algorithm
tends to its set-cover ratio -- which is what makes the logarithmic growth visible rather
than drowned out by connection cost.

:func:`greedy_tight_set_cover` supplies the family where that growth actually shows.  On a
random Set Cover instance greedy is usually near-optimal; the classical tight construction
forces it to ``k/2`` times optimal on a universe of size ``2(2^k - 1)``, i.e. ``Theta(log q)``.
"""

from __future__ import annotations

import numpy as np

from iffl.instance import Instance


def greedy_tight_set_cover(k: int) -> tuple[int, list[set[int]], int]:
    """The classical instance on which greedy Set Cover is ``Theta(log q)`` times optimal.

    Elements form a 2 x (2^k - 1) grid.  The two optimal sets are the rows.  The greedy
    sets are column blocks of sizes ``2^(k-1), 2^(k-2), ..., 1`` taken over both rows, so
    block ``i`` has ``2^(k-i+1)`` elements while either row has only ``2^(k-i+1) - 1``
    left uncovered.  Greedy therefore takes every block -- ``k`` sets -- where two suffice.

    Returns ``(q, sets, k_opt)`` with ``q = 2(2^k - 1)`` elements and ``k_opt = 2``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    width = 2**k - 1
    q = 2 * width
    # Element id of (row, column).
    eid = lambda r, c: r * width + c  # noqa: E731

    rows = [{eid(0, c) for c in range(width)}, {eid(1, c) for c in range(width)}]

    blocks: list[set[int]] = []
    start = 0
    for i in range(1, k + 1):
        size = 2 ** (k - i)
        cols = range(start, start + size)
        blocks.append({eid(r, c) for r in (0, 1) for c in cols})
        start += size

    return q, rows + blocks, 2


def random_set_cover(
    q: int, p: int, set_size: int, seed: int = 0
) -> tuple[int, list[set[int]], int | None]:
    """A random Set Cover instance, with a planted cover guaranteeing feasibility.

    Greedy is typically near-optimal here, which is the point of the contrast with
    :func:`greedy_tight_set_cover`.  ``k_opt`` is returned as ``None`` because the true
    optimum is not known by construction -- only an upper bound from the planted cover.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(q)
    planted = [set(chunk.tolist()) for chunk in np.array_split(perm, max(1, p // 4))]
    extra = [
        set(rng.choice(q, size=min(set_size, q), replace=False).tolist())
        for _ in range(max(0, p - len(planted)))
    ]
    return q, planted + extra, None


def set_cover_instance(
    q: int, sets: list[set[int]], facility_cost: float | None = None
) -> Instance:
    """Build the paper's reduction from a Set Cover instance.

    Args:
        q: universe size; elements are ``0..q-1``.
        sets: the family ``S_1..S_p``; their union must be the universe.
        facility_cost: opening cost ``M``, defaulting to ``q^2`` as in the reduction.

    Returns:
        An instance with ``2q`` clients, ``len(sets)`` facilities and ``R_j = 1``.
    """
    covered = set().union(*sets) if sets else set()
    if covered != set(range(q)):
        raise ValueError("the sets must cover the universe for the instance to be feasible")

    M = float(q * q) if facility_cost is None else float(facility_cost)
    p = len(sets)
    n = 2 * q

    # Client index 2a + b is j_ab, the b-th copy of element a.
    elem = np.arange(n) // 2
    incident = np.zeros((n, p), dtype=bool)
    for l, S in enumerate(sets):
        for a in S:
            incident[2 * a, l] = True
            incident[2 * a + 1, l] = True

    d = np.where(incident, 1.0, 3.0)

    # Coordinates are not used by any algorithm; the metric is given explicitly.
    return Instance(
        name=f"setcover-q{q}-p{p}",
        client_pts=np.zeros((n, 1)),
        facility_pts=np.zeros((p, 1)),
        f=np.full(p, M),
        d=d,
        groups={"c1": np.array([f"u{a}" for a in elem])},
        meta={"source": "hardness", "has_native_costs": True, "universe": q,
              "n_sets": p, "facility_cost": M},
    ).with_radii(
        np.ones(n),
        radius_scheme="set_cover_unit", radius_param=1.0,
        radius_clamped=0, radius_n_zero=0, radius_uniform=True, radius_rank=1,
    )


def full_metric(q: int, sets: list[set[int]]) -> np.ndarray:
    """The full ``(2q + p)`` metric on clients and facilities, for triangle-inequality checks.

    Clients occupy the first ``2q`` indices, facilities the rest.
    """
    p, n = len(sets), 2 * q
    N = n + p
    D = np.zeros((N, N))
    elem = np.arange(n) // 2
    inc = np.zeros((n, p), dtype=bool)
    for l, S in enumerate(sets):
        for a in S:
            inc[2 * a, l] = inc[2 * a + 1, l] = True

    for x in range(n):
        for y in range(n):
            if x != y:
                D[x, y] = 1.0 if elem[x] == elem[y] else 2.0
    D[:n, n:] = np.where(inc, 1.0, 3.0)
    D[n:, :n] = D[:n, n:].T
    for x in range(n, N):
        for y in range(n, N):
            if x != y:
                D[x, y] = 2.0
    return D


def optimal_cost(q: int, k_opt: int, facility_cost: float | None = None) -> float:
    """``OPT_FL = M k* + 2q`` (Lemma 5 of the paper)."""
    M = float(q * q) if facility_cost is None else float(facility_cost)
    return M * k_opt + 2 * q
