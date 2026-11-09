"""Approach A: the (1.725, 3)-approximation by LP rounding.

Pipeline, following the paper section by section:

1. Solve the restricted LP over ``E_R`` (:mod:`iffl.lp`).
2. **Split** each facility into colocated copies so that ``x_ij in {0, y_i}`` and every
   client's support carries exactly one unit of opening mass.
3. **Scale** ``y`` by ``gamma = 69/40`` and **filter** each client's closest unit of mass
   into ``F^C_j`` (close), leaving ``gamma - 1`` in ``F^D_j`` (distant).
4. **Cluster**: repeatedly take the unassigned client of minimum ``d^max_j`` as a center
   and absorb every unassigned client whose close set intersects the center's.
5. **Round**: one copy per center group, sampled with probability ``y_c``; every other
   copy independently with probability ``y_c``.
6. Open the original facility behind each selected copy, paying ``f_i`` once, and connect
   every client to its nearest open facility.

The guarantee is bicriteria and asymmetric: ``E[cost] <= 1.725 * OPT_fair`` holds in
expectation, while ``d(j, S) <= 3 R_j`` holds in *every* outcome.  The second is checked
as a hard assertion on each trial, the first only on the mean over trials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from iffl.evaluate import radius_ratio
from iffl.instance import Instance
from iffl.lp import RestrictedLP, solve_restricted_lp

#: The scaling constant, fixed by the analysis at 69/40.
GAMMA = float(Fraction(69, 40))

#: Mass tolerance.  Copies below this are dropped and invariants are checked to it.
TOL = 1e-7


@dataclass
class CopyStructure:
    """The split, scaled and filtered copy structure feeding the rounding step.

    Attributes:
        facility: ``(n_copies,)`` original facility index behind each copy.
        mass: ``(n_copies,)`` *scaled* opening mass ``y_c = gamma * ybar_c``.
        close: for each client, the copy indices forming ``F^C_j`` (total mass 1).
        distant: for each client, the copy indices forming ``F^D_j`` (mass ``gamma-1``).
        d_max: ``(n,)`` ``d^max_j = max_{c in F^C_j} d(c, j)``, the clustering score.
    """

    facility: np.ndarray
    mass: np.ndarray
    close: list[np.ndarray]
    distant: list[np.ndarray]
    d_max: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_copies(self) -> int:
        return self.mass.size


def _split_facilities(
    inst: Instance, lp: RestrictedLP, tol: float = TOL
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Step 2: represent each facility as an interval and cut at every client's prefix.

    Facility ``i`` is the interval ``[0, y*_i]``; client ``j`` marks the prefix of length
    ``x*_ij``.  Cutting at all marked endpoints leaves segments on which every client's
    assignment is either 0 or the full segment mass, which is the ``x_ij in {0, y_i}``
    normal form the rest of the algorithm needs.

    Returns ``(facility_of_copy, unscaled_mass, support)`` where ``support[c]`` lists the
    clients assigned to copy ``c``.
    """
    edges_by_fac: list[list[tuple[int, float]]] = [[] for _ in range(inst.m)]
    for e in range(lp.n_edges):
        if lp.x[e] > tol:
            edges_by_fac[int(lp.edge_facility[e])].append(
                (int(lp.edge_client[e]), float(lp.x[e]))
            )

    fac_of_copy: list[int] = []
    mass: list[float] = []
    support: list[list[int]] = []

    for i in range(inst.m):
        y_i = float(lp.y[i])
        if y_i <= tol or not edges_by_fac[i]:
            continue
        marks = sorted({x for _, x in edges_by_fac[i]} | {y_i})
        prev = 0.0
        for t in marks:
            if t - prev <= tol:
                prev = t
                continue
            # Clients whose prefix covers this whole segment are assigned to it.
            members = [j for j, x in edges_by_fac[i] if x >= t - tol]
            if members:
                fac_of_copy.append(i)
                mass.append(t - prev)
                support.append(members)
            prev = t

    return np.asarray(fac_of_copy, dtype=int), np.asarray(mass, dtype=float), support


def _client_supports(n: int, support: list[list[int]]) -> list[list[int]]:
    """Invert copy -> clients into client -> copies."""
    out: list[list[int]] = [[] for _ in range(n)]
    for c, members in enumerate(support):
        for j in members:
            out[j].append(c)
    return out


def build_copy_structure(
    inst: Instance, lp: RestrictedLP, tol: float = TOL
) -> CopyStructure:
    """Steps 2-3: split, scale by ``gamma``, and filter the closest unit of mass."""
    fac_of_copy, ybar, support = _split_facilities(inst, lp, tol)
    if ybar.size == 0:
        raise ValueError(f"{inst.name}: LP solution produced no facility copies")

    F = _client_supports(inst.n, support)
    for j in range(inst.n):
        if not F[j]:
            raise ValueError(f"{inst.name}: client {j} has empty LP support")

    # Splitting must leave exactly one unit of unscaled mass in every client's support
    # (sum_i x*_ij = 1).  Anything beyond solver noise here means the split is wrong,
    # and every later invariant would inherit the error, so fail loudly now.
    support_mass = np.array([ybar[F[j]].sum() for j in range(inst.n)])
    max_dev = float(np.abs(support_mass - 1.0).max())
    if max_dev > 1e-6:
        j = int(np.argmax(np.abs(support_mass - 1.0)))
        raise ValueError(
            f"{inst.name}: split support mass for client {j} is "
            f"{support_mass[j]:.9g}, not 1 (max deviation {max_dev:.3g})"
        )

    # Distance from each client to each copy, via the copy's original facility.
    d_copy = inst.d[:, fac_of_copy]
    scaled = GAMMA * ybar

    # --- Pass 1: locate where each client's unit boundary falls inside a copy. ---
    cuts: dict[int, set[float]] = {}
    for j in range(inst.n):
        idx = np.asarray(F[j], dtype=int)
        # Sort by distance, breaking ties by copy index -- the paper's "breaking ties
        # consistently".  Copies of one facility are colocated, so an index-based
        # tie-break is what lets pass 2 reproduce this prefix exactly.
        order = idx[np.lexsort((idx, d_copy[j, idx]))]
        cum = np.concatenate([[0.0], np.cumsum(scaled[order])])
        k = int(np.searchsorted(cum, 1.0, side="left")) - 1
        if k < 0 or k >= order.size:
            continue
        frac = (1.0 - cum[k]) / scaled[order[k]]
        if tol < frac < 1.0 - tol:
            cuts.setdefault(int(order[k]), set()).add(float(frac))

    # Keep every scaled copy at mass <= 1 so the independent Bernoulli step is valid.
    # gamma < 2 and y*_i <= 1, so a single midpoint cut always suffices.
    for c in np.flatnonzero(scaled > 1.0).tolist():
        cuts.setdefault(int(c), set()).add(0.5)

    # --- Refine: subdivide cut copies; sub-copies inherit facility and support. ---
    new_fac: list[int] = []
    new_mass: list[float] = []
    new_support: list[list[int]] = []
    new_parent: list[int] = []
    for c in range(ybar.size):
        bounds = [0.0, *sorted(cuts.get(c, ())), 1.0]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a <= 1e-12:
                continue
            new_fac.append(int(fac_of_copy[c]))
            new_mass.append(ybar[c] * (b - a))
            new_support.append(support[c])
            new_parent.append(c)

    fac_of_copy = np.asarray(new_fac, dtype=int)
    ybar = np.asarray(new_mass, dtype=float)
    parent = np.asarray(new_parent, dtype=int)
    support = new_support
    F = _client_supports(inst.n, support)
    d_copy = inst.d[:, fac_of_copy]
    mass = GAMMA * ybar

    # --- Pass 2: the unit boundary now lands on a sub-copy edge for every client. ---
    close: list[np.ndarray] = []
    distant: list[np.ndarray] = []
    d_max = np.zeros(inst.n)
    worst_close = 0.0
    worst_distant = 0.0

    for j in range(inst.n):
        idx = np.asarray(F[j], dtype=int)
        # Same key as pass 1, with the parent index keeping a parent's sub-copies
        # contiguous and in subdivision order.
        order = idx[np.lexsort((idx, parent[idx], d_copy[j, idx]))]
        cum = np.cumsum(mass[order])
        n_close = min(int(np.searchsorted(cum, 1.0 - tol, side="left")) + 1, order.size)
        cj, dj = order[:n_close], order[n_close:]
        close.append(cj)
        distant.append(dj)
        worst_close = max(worst_close, abs(float(mass[cj].sum()) - 1.0))
        worst_distant = max(worst_distant, abs(float(mass[dj].sum()) - (GAMMA - 1.0)))
        d_max[j] = float(d_copy[j, cj].max())

    # Equation (3): close mass is exactly 1 and distant mass exactly gamma - 1.  These
    # are what the connection-cost analysis rests on, so they are checked on the real
    # masses used by the rounding, not on renormalized copies.
    if worst_close > 1e-6 or worst_distant > 1e-6:
        raise ValueError(
            f"{inst.name}: filtering invariant broken -- close-mass error "
            f"{worst_close:.3g}, distant-mass error {worst_distant:.3g}"
        )

    # d^max_j <= R_j (equation 4): the filtered close set stays inside the radius,
    # which is what makes the 3R_j guarantee deterministic.
    # Relative to the radius scale, so large-magnitude instances are not flagged for
    # ordinary floating-point noise.
    radius_slack = float(np.max(d_max - inst.R))
    if radius_slack > 1e-9 * float(max(1.0, inst.R.max())):
        raise ValueError(
            f"{inst.name}: d^max_j exceeds R_j by {radius_slack:.3g}"
        )

    return CopyStructure(
        facility=fac_of_copy,
        mass=mass,
        close=close,
        distant=distant,
        d_max=d_max,
        diagnostics={
            "n_copies": int(mass.size),
            "support_mass_max_dev": max_dev,
            "close_mass_max_dev": worst_close,
            "distant_mass_max_dev": worst_distant,
            "max_copy_mass": float(mass.max()),
            "d_max_vs_R_slack": radius_slack,
        },
    )


def cluster_centers(cs: CopyStructure, n: int) -> tuple[list[int], np.ndarray]:
    """Step 4: greedy clustering in increasing order of ``d^max``.

    This is the Chudak-Shmoys clustering step, scored by ``d^max_j`` and keyed on
    close-set intersection.

    Returns the center list and ``sigma``, mapping each client to its center.
    """
    # copy -> clients holding it in their close set, for fast intersection queries.
    holders: dict[int, list[int]] = {}
    for j in range(n):
        for c in cs.close[j].tolist():
            holders.setdefault(c, []).append(j)

    unassigned = np.ones(n, dtype=bool)
    order = np.argsort(cs.d_max, kind="stable")
    sigma = np.full(n, -1, dtype=int)
    centers: list[int] = []

    for s in order.tolist():
        if not unassigned[s]:
            continue
        centers.append(s)
        touched = {s}
        for c in cs.close[s].tolist():
            for j in holders.get(c, ()):
                if unassigned[j]:
                    touched.add(j)
        for j in touched:
            unassigned[j] = False
            sigma[j] = s

    if unassigned.any():
        raise ValueError(f"clustering left {int(unassigned.sum())} clients unassigned")

    # Lemma 2: close sets of distinct centers are pairwise disjoint.  Cheap to verify
    # and it catches filtering bugs immediately rather than as a cost anomaly later.
    seen: dict[int, int] = {}
    for s in centers:
        for c in cs.close[s].tolist():
            if c in seen:
                raise ValueError(
                    f"Lemma 2 violated: copy {c} shared by centers {seen[c]} and {s}"
                )
            seen[c] = s

    return centers, sigma


def round_once(
    cs: CopyStructure, centers: list[int], rng: np.random.Generator
) -> np.ndarray:
    """Step 5: sample the opening decisions; return the selected copy indices."""
    selected: list[int] = []
    in_center_group = np.zeros(cs.n_copies, dtype=bool)

    for s in centers:
        idx = cs.close[s]
        in_center_group[idx] = True
        p = cs.mass[idx]
        total = p.sum()
        # Exactly one copy per center group: the close mass is 1 by construction.
        pick = rng.choice(idx, p=p / total)
        selected.append(int(pick))

    free = np.flatnonzero(~in_center_group)
    if free.size:
        hits = rng.random(free.size) < cs.mass[free]
        selected.extend(free[hits].tolist())

    return np.asarray(sorted(set(selected)), dtype=int)


@dataclass
class Solution:
    """One rounded IFFL solution and its measured quantities."""

    open_facilities: np.ndarray
    assignment: np.ndarray
    facility_cost: float
    connection_cost: float
    radius_ratio: np.ndarray

    @property
    def cost(self) -> float:
        return self.facility_cost + self.connection_cost

    @property
    def max_radius_ratio(self) -> float:
        return float(self.radius_ratio.max())


def evaluate_open_set(inst: Instance, open_facilities: np.ndarray) -> Solution:
    """Connect every client to its nearest open facility and measure the result.

    Assignment ranges over *all* facilities, not just ``E_R``: the guarantee is
    ``d(j, S) <= 3 R_j``, so a client may legitimately be served outside its own radius.
    """
    if open_facilities.size == 0:
        raise ValueError(f"{inst.name}: no facility was opened")
    sub = inst.d[:, open_facilities]
    pos = sub.argmin(axis=1)
    assignment = open_facilities[pos]
    dists = sub[np.arange(inst.n), pos]
    return Solution(
        open_facilities=open_facilities,
        assignment=assignment,
        facility_cost=float(inst.f[open_facilities].sum()),
        connection_cost=float(dists.sum()),
        radius_ratio=radius_ratio(dists, inst.R),
    )


def lp_rounding_iffl(
    inst: Instance,
    trials: int = 50,
    seed: int = 0,
    lp: RestrictedLP | None = None,
    check_fairness: bool = True,
) -> dict:
    """Run Approach A end to end for ``trials`` independent roundings.

    Args:
        inst: instance carrying radii and costs.
        trials: number of independent roundings; the cost bound is in expectation, so
            the mean over trials is what should be compared against 1.725.
        seed: base seed for the trial RNG.
        lp: a pre-solved restricted LP, to avoid re-solving across configurations.
        check_fairness: assert ``d(j,S) <= 3 R_j`` on every trial.

    Returns:
        A dict with the LP bound, per-trial costs and radius ratios, and the aggregate
        cost ratio against the LP.
    """
    if lp is None:
        lp = solve_restricted_lp(inst)
    cs = build_copy_structure(inst, lp)
    centers, sigma = cluster_centers(cs, inst.n)

    rng = np.random.default_rng(seed)
    costs: list[float] = []
    ratios: list[float] = []
    n_open: list[int] = []
    solutions: list[Solution] = []

    for t in range(trials):
        selected = round_once(cs, centers, rng)
        opened = np.unique(cs.facility[selected])
        sol = evaluate_open_set(inst, opened)
        if check_fairness and sol.max_radius_ratio > 3.0 + 1e-6:
            raise AssertionError(
                f"{inst.name}: trial {t} violated the deterministic fairness bound -- "
                f"max d(j,S)/R_j = {sol.max_radius_ratio:.6g} > 3"
            )
        costs.append(sol.cost)
        ratios.append(sol.max_radius_ratio)
        n_open.append(int(opened.size))
        solutions.append(sol)

    costs_arr = np.asarray(costs)
    return {
        "algo": "lp_rounding",
        "lp_obj": lp.obj,
        "lp_F_star": lp.F_star,
        "lp_C_star": float(lp.C_star.sum()),
        "gamma": GAMMA,
        "n_centers": len(centers),
        "trials": trials,
        "cost_mean": float(costs_arr.mean()),
        "cost_min": float(costs_arr.min()),
        "cost_max": float(costs_arr.max()),
        "cost_ratio_mean": float(costs_arr.mean() / lp.obj) if lp.obj > 0 else float("nan"),
        "cost_ratio_max": float(costs_arr.max() / lp.obj) if lp.obj > 0 else float("nan"),
        "max_radius_ratio": float(np.max(ratios)),
        "mean_radius_ratio": float(np.mean([s.radius_ratio.mean() for s in solutions])),
        "n_open_mean": float(np.mean(n_open)),
        "costs": costs,
        "diagnostics": cs.diagnostics,
        "solutions": solutions,
    }
