"""Approach C: the greedy UFL algorithm (Williamson & Shmoys, Algorithm 9.2),
adapted to individual fairness.

The original algorithm, for uncapacitated facility location::

    S <- D;  X <- {}
    while S != {}:
        choose i in F and Y subset of S minimizing
            ( f_i - sum_{j not in S} (c(j,X) - c_ij)_+ + sum_{j in Y} c_ij ) / |Y|
        X <- X + {i};  f_i <- 0;  S <- S - Y
    open all facilities in X, assign each client to the closest facility in X

Each step buys the cheapest available *cost per newly connected client*.  The middle
term is the saving already-connected clients would get by switching to ``i``: it is what
lets a later facility pay for itself partly out of improvements to clients that are
already served.  Setting ``f_i <- 0`` makes a facility free once opened, so reopening it
costs only connections.

**The fair adaptation** is to forbid every client/facility pair outside the client's
radius, exactly as the two papers restrict their LP to
``E_R = {(i,j) : d(i,j) <= R_j}``:

* ``Y`` may only contain clients admissible to ``i``;
* a connected client contributes a saving only if it is admissible to ``i`` -- it cannot
  switch to a facility it is not allowed to use;
* the final assignment is to the closest *admissible* open facility.

That makes the output **strictly fair**: ``d(j, sigma(j)) <= R_j``, a ratio of 1 rather
than the 3 both papers settle for.  So this is a ``(rho, 1)`` algorithm against their
``(1.725, 3)`` and ``(3, 3)``, and the interesting question is what that stricter
fairness costs.

On the guarantee: the unrestricted algorithm is a 1.861-approximation by dual fitting.
The argument plausibly survives the restriction -- the dual constraint for ``i`` becomes
``sum_{j : (i,j) in E_R} (alpha_j - c_ij)_+ <= f_i``, which involves only admissible
pairs, and the chain ``alpha_j <= alpha_j' + c_ij' + c_ij`` it relies on is applied only
to clients that may use ``i``.  That is not a proof, so nothing here asserts 1.861; the
ratio is measured and reported like any other observable.
"""

from __future__ import annotations

import numpy as np

from iffl.evaluate import radius_ratio
from iffl.instance import Instance
from iffl.lp import RestrictedLP, solve_restricted_lp
from iffl.lp_rounding import Solution


def _best_move(
    d: np.ndarray,
    in_er: np.ndarray,
    f_work: np.ndarray,
    unconnected: np.ndarray,
    savings: np.ndarray,
    order: list[np.ndarray],
) -> tuple[float, int, np.ndarray]:
    """Find the ``(i, Y)`` minimizing the greedy ratio.

    Searching all subsets ``Y`` would be exponential, but it is unnecessary: for a fixed
    facility and a fixed ``|Y| = k``, the sum ``sum_{j in Y} c_ij`` is minimized by the
    ``k`` admissible unconnected clients closest to ``i``.  So scanning the prefixes of
    each facility's distance-sorted candidate list examines every subset that could
    possibly win.

    ``order`` holds that sorted candidate list per facility, computed once up front since
    distances never change, and ``savings`` is maintained incrementally by the caller.

    Returns ``(ratio, facility, clients)``; ``facility`` is -1 if no move exists.
    """
    best_ratio = np.inf
    best_i = -1
    best_Y: np.ndarray = np.empty(0, dtype=int)

    for i in range(d.shape[1]):
        cand = order[i][unconnected[order[i]]]
        if cand.size == 0:
            continue
        prefix = np.cumsum(d[cand, i])
        ratios = (f_work[i] - savings[i] + prefix) / np.arange(1, cand.size + 1)
        pos = int(np.argmin(ratios))
        if ratios[pos] < best_ratio:
            best_ratio = float(ratios[pos])
            best_i = i
            best_Y = cand[: pos + 1]

    return best_ratio, best_i, best_Y


def _initial_state(d: np.ndarray, in_er: np.ndarray) -> list[np.ndarray]:
    """Each facility's admissible clients, sorted by distance.  Distances are static, so
    this is computed once rather than re-sorted every round."""
    return [
        np.flatnonzero(in_er[:, i])[np.argsort(d[in_er[:, i], i], kind="stable")]
        for i in range(d.shape[1])
    ]


def greedy_iffl(
    inst: Instance,
    lp: RestrictedLP | None = None,
    check_fairness: bool = True,
) -> dict:
    """Run the fairness-restricted greedy algorithm.

    Args:
        inst: instance carrying radii and costs.
        lp: a pre-solved restricted LP, used only as the comparison bound.
        check_fairness: assert strict fairness, ``d(j, sigma(j)) <= R_j``.

    Returns:
        A dict shaped like the other algorithms' results.
    """
    if inst.R is None:
        raise ValueError(f"{inst.name}: instance has no radii")
    inst.check_feasible()
    if lp is None:
        lp = solve_restricted_lp(inst)

    d = inst.d
    in_er = d <= inst.R[:, None] + 1e-12
    f_work = np.asarray(inst.f, dtype=float).copy()

    unconnected = np.ones(inst.n, dtype=bool)
    cur_cost = np.full(inst.n, np.inf)
    order = _initial_state(d, in_er)
    # savings[i] = sum over connected, admissible j of (c(j,X) - c_ij)_+.  Rebuilding
    # this as an (n, m) array every round costs O(m n) per round and dominates the run
    # at scale, so it is updated in place instead: a client's contribution changes only
    # when its own connection cost does, which touches O(m) entries.
    savings = np.zeros(inst.m)
    opened: list[int] = []
    n_rounds = 0
    n_reopened = 0

    def _apply(j: int, new_cost: float) -> None:
        """Set client j's connection cost, adjusting savings for every facility it may use."""
        old = cur_cost[j]
        row = in_er[j]
        old_c = np.maximum(old - d[j], 0.0) if np.isfinite(old) else 0.0
        new_c = np.maximum(new_cost - d[j], 0.0)
        savings[row] += (new_c - old_c)[row] if np.isfinite(old) else new_c[row]
        cur_cost[j] = new_cost

    while unconnected.any():
        ratio, i, Y = _best_move(d, in_er, f_work, unconnected, savings, order)
        if i < 0:
            # Unreachable for a feasible instance: every client has an admissible
            # facility, so some move always exists.
            raise RuntimeError(
                f"{inst.name}: greedy stalled with {int(unconnected.sum())} "
                "client(s) unconnected"
            )
        if i in opened:
            n_reopened += 1
        else:
            opened.append(i)
        f_work[i] = 0.0
        unconnected[Y] = False
        # Opening i can also improve clients connected earlier; cur_cost tracks the
        # distance to the closest admissible open facility, which the final assignment
        # uses, and savings must follow it.
        improved = np.flatnonzero(in_er[:, i] & ~unconnected)
        for j in improved.tolist():
            if d[j, i] < cur_cost[j]:
                _apply(j, float(d[j, i]))
        n_rounds += 1

    open_facilities = np.asarray(sorted(opened), dtype=int)

    # Assign to the closest admissible open facility.  Every client has one: it was
    # removed from S by being connected to an admissible member of X.
    sub = np.where(in_er[:, open_facilities], d[:, open_facilities], np.inf)
    pos = sub.argmin(axis=1)
    assignment = open_facilities[pos]
    dists = sub[np.arange(inst.n), pos]
    if not np.all(np.isfinite(dists)):
        raise RuntimeError(f"{inst.name}: a client has no admissible open facility")

    sol = Solution(
        open_facilities=open_facilities,
        assignment=assignment,
        facility_cost=float(inst.f[open_facilities].sum()),
        connection_cost=float(dists.sum()),
        radius_ratio=radius_ratio(dists, inst.R),
    )

    if check_fairness and np.any(dists > inst.R + 1e-9):
        worst = int(np.argmax(dists - inst.R))
        raise AssertionError(
            f"{inst.name}: strict fairness violated -- client {worst} served at "
            f"{dists[worst]:.6g} with R_j = {inst.R[worst]:.6g}"
        )

    return {
        "algo": "greedy",
        "lp_obj": lp.obj,
        "cost": sol.cost,
        "facility_cost": sol.facility_cost,
        "connection_cost": sol.connection_cost,
        "cost_ratio": float(sol.cost / lp.obj) if lp.obj > 0 else float("nan"),
        "max_radius_ratio": sol.max_radius_ratio,
        "mean_radius_ratio": float(sol.radius_ratio.mean()),
        "n_open": int(open_facilities.size),
        "diagnostics": {
            "n_rounds": n_rounds,
            "n_reopened": n_reopened,
            "n_open": int(open_facilities.size),
        },
        "solution": sol,
    }
