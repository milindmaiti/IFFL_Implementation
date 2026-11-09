"""Approach B: the (3, 3)-bicriteria primal-dual algorithm for uniform radii.

A Jain-Vazirani style continuous dual growth, run entirely inside the admissible edge
set ``E_R``, followed by a pruning phase.

Phase 1 raises every active client's ``alpha_j`` at rate 1.  Once ``alpha_j`` reaches
``d(i, j)`` on an admissible edge the client starts paying into ``beta_ij``; when
``sum_j beta_ij`` covers ``f_i`` the facility goes tight, and every client with a tight
edge to it freezes.  Phase 2 takes the witness sets ``W(i) = {j : beta_ij > 0}``, builds
the conflict graph on overlapping witness sets, extracts a greedy maximal independent
set in ascending tight time, and assigns clients directly or through a shared witness.

Two implementation notes that matter:

* All active clients share the same ``alpha_j = t``, because they all start at 0 and
  grow at rate 1.  That makes the contributor test simply ``d(i, j) <= t``.
* ``beta_ij`` is never accumulated step by step.  A client contributes to ``i`` over
  exactly the window from ``d(i, j)`` until it freezes or ``i`` goes tight, so
  ``beta_ij = max(0, min(alpha_j, t_i) - d(i, j))`` in closed form once the simulation
  has produced the freeze and tight times.  The simulation only tracks contributor
  *counts*, which keeps each event O(1) instead of O(|E_R|).

The (3,3) guarantee holds for *uniform* radii only.  The witness argument bounds
``d(j, i)`` by ``R_j + R_{j*} + R_{j*}`` where ``j*`` is a shared witness; with
heterogeneous radii ``R_{j*}`` can be arbitrarily larger than ``R_j``, so nothing is
bounded relative to ``R_j``.  The code runs on nonuniform radii too and
reports the realised ratio, but only asserts the bound when the radii are uniform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iffl.evaluate import radius_ratio
from iffl.instance import Instance
from iffl.lp import RestrictedLP, solve_restricted_lp
from iffl.lp_rounding import Solution, evaluate_open_set

TOL = 1e-9


@dataclass
class DualGrowth:
    """Outcome of phase 1.

    Attributes:
        alpha: ``(n,)`` frozen dual values; also each client's freeze time.
        tight_time: ``(m,)`` time each facility went tight, ``inf`` if it never did.
        freeze_cause: ``(n,)`` facility whose tightening froze each client.
        beta: ``(n, m)`` witness payments, in closed form.
        n_events: number of simulation events, for diagnostics.
    """

    alpha: np.ndarray
    tight_time: np.ndarray
    freeze_cause: np.ndarray
    beta: np.ndarray
    n_events: int = 0
    diagnostics: dict = field(default_factory=dict)

    @property
    def f_temp(self) -> np.ndarray:
        """Temporarily opened facilities, those that went tight."""
        return np.flatnonzero(np.isfinite(self.tight_time))

    def witness_sets(self, tol: float = 1e-9) -> dict[int, np.ndarray]:
        """``W(i) = {j : beta_ij > 0}`` for every tight facility."""
        return {
            int(i): np.flatnonzero(self.beta[:, i] > tol)
            for i in self.f_temp.tolist()
        }


def grow_duals(inst: Instance, tol: float = TOL) -> DualGrowth:
    """Phase 1: continuous dual growth restricted to ``E_R``."""
    if inst.R is None:
        raise ValueError(f"{inst.name}: instance has no radii")
    inst.check_feasible()

    n, m = inst.n, inst.m
    in_er = inst.d <= inst.R[:, None] + 1e-12
    if not in_er.any(axis=1).all():
        raise ValueError(f"{inst.name}: some client has no admissible edge")

    # Admissible edges in ascending distance -- the order they activate in.
    e_j, e_i = np.nonzero(in_er)
    e_d = inst.d[e_j, e_i]
    order = np.lexsort((e_i, e_j, e_d))
    e_j, e_i, e_d = e_j[order], e_i[order], e_d[order]
    n_edges = e_d.size

    alpha = np.zeros(n)
    active = np.ones(n, dtype=bool)
    freeze_cause = np.full(n, -1, dtype=int)
    tight_time = np.full(m, np.inf)
    tight = np.zeros(m, dtype=bool)
    paid = np.zeros(m)
    n_contrib = np.zeros(m, dtype=int)
    contrib_facs: list[set[int]] = [set() for _ in range(n)]

    f = np.asarray(inst.f, dtype=float)
    # A zero-cost facility is tight from the start: it needs no dual to pay for it.
    for i in np.flatnonzero(f <= tol).tolist():
        tight[i] = True
        tight_time[i] = 0.0

    t = 0.0
    ptr = 0
    n_events = 0

    def freeze(j: int, cause: int) -> None:
        active[j] = False
        alpha[j] = t
        freeze_cause[j] = cause
        for i2 in contrib_facs[j]:
            if not tight[i2]:
                n_contrib[i2] -= 1

    # Facilities already tight at t=0 can freeze clients the moment an edge activates,
    # which the edge-activation branch below handles.
    while active.any():
        n_events += 1

        # Next edge activation, skipping edges whose client has already frozen.
        while ptr < n_edges and not active[e_j[ptr]]:
            ptr += 1
        t_edge = float(e_d[ptr]) if ptr < n_edges else np.inf

        # Next facility tightening.
        cand = (~tight) & (n_contrib > 0)
        if cand.any():
            remaining = (f[cand] - paid[cand]) / n_contrib[cand]
            t_fac = t + float(np.maximum(remaining, 0.0).min())
        else:
            t_fac = np.inf

        t_next = min(t_edge, t_fac)
        if not np.isfinite(t_next):
            raise RuntimeError(
                f"{inst.name}: dual growth stalled at t={t:.6g} with "
                f"{int(active.sum())} client(s) still active"
            )

        dt = t_next - t
        if dt > 0:
            grow = (~tight) & (n_contrib > 0)
            paid[grow] += n_contrib[grow] * dt
        t = t_next
        alpha[active] = t

        # Facility tightenings first, so a facility that goes tight at exactly this
        # instant also freezes the clients whose edges activate at the same instant.
        newly = np.flatnonzero((~tight) & (n_contrib > 0) & (paid >= f - 1e-9))
        for i in newly.tolist():
            tight[i] = True
            tight_time[i] = t
            # Rule 5(a): every active client with a tight edge to i freezes now.
            for j in np.flatnonzero(active & in_er[:, i] & (inst.d[:, i] <= t + 1e-12)).tolist():
                freeze(int(j), int(i))

        # Then the edge activations landing exactly at this time.
        while ptr < n_edges and e_d[ptr] <= t + 1e-12:
            j, i = int(e_j[ptr]), int(e_i[ptr])
            if active[j]:
                if tight[i]:
                    # Rule 5(b): alpha_j reached an already-tight facility.
                    freeze(j, i)
                else:
                    n_contrib[i] += 1
                    contrib_facs[j].add(i)
            ptr += 1

        if n_events > 20 * n_edges + 10 * m + 100:
            raise RuntimeError(f"{inst.name}: dual growth exceeded its event budget")

    # beta in closed form: j pays into i from the moment its edge activates until it
    # freezes or i goes tight, whichever comes first.
    beta = np.zeros((n, m))
    stop = np.minimum(alpha[e_j], tight_time[e_i])
    width = np.maximum(stop - e_d, 0.0)
    beta[e_j, e_i] = width

    # Dual feasibility, unconditional: no facility is overpaid and no edge constraint
    # alpha_j - beta_ij <= d(i,j) is violated.  The tolerances are *relative*: opening
    # costs here can run to 1e7 (the hardness reduction uses M = q^2), where an absolute
    # 1e-6 is below what double precision can deliver after summing thousands of terms.
    scale_f = float(max(1.0, f.max())) if m else 1.0
    overpay = float(np.max(beta.sum(axis=0) - f)) if m else 0.0
    if overpay > 1e-9 * scale_f:
        i = int(np.argmax(beta.sum(axis=0) - f))
        raise AssertionError(
            f"{inst.name}: facility {i} overpaid -- sum beta = "
            f"{beta[:, i].sum():.9g} > f_i = {f[i]:.9g}"
        )
    slack = alpha[e_j] - beta[e_j, e_i] - e_d
    scale_d = float(max(1.0, e_d.max())) if e_d.size else 1.0
    edge_violation = float(slack.max()) if slack.size else 0.0
    if edge_violation > 1e-9 * max(scale_d, scale_f):
        raise AssertionError(
            f"{inst.name}: dual edge constraint violated by {edge_violation:.3g}"
        )

    return DualGrowth(
        alpha=alpha,
        tight_time=tight_time,
        freeze_cause=freeze_cause,
        beta=beta,
        n_events=n_events,
        diagnostics={
            "n_edges": int(n_edges),
            "n_tight": int(tight.sum()),
            "max_overpay": overpay,
            "max_edge_violation": edge_violation,
            "alpha_sum": float(alpha.sum()),
            "final_time": t,
        },
    )


def prune_and_assign(
    inst: Instance, dg: DualGrowth, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Phase 2: greedy MIS on the conflict graph, then direct/indirect assignment.

    Returns ``(open_facilities, assignment, diagnostics)``.
    """
    W = dg.witness_sets(tol)
    f_temp = sorted(W, key=lambda i: (dg.tight_time[i], i))
    if not f_temp:
        raise RuntimeError(f"{inst.name}: no facility went tight")

    # Greedy maximal independent set in the conflict graph, processed in ascending tight
    # time.  Claiming a facility's witnesses is exactly the independence test: two
    # facilities conflict precisely when they share a witness.
    claimed = np.full(inst.n, -1, dtype=int)
    chosen: list[int] = []
    for i in f_temp:
        if np.any(claimed[W[i]] >= 0):
            continue
        chosen.append(i)
        claimed[W[i]] = i

    assignment = np.full(inst.n, -1, dtype=int)
    n_direct = 0
    n_via_cause = 0
    n_via_witness = 0
    n_fallback = 0

    # Witness -> owning open facility, and open facility -> its witness set, for the
    # indirect lookup below.
    chosen_set = set(chosen)
    for j in range(inst.n):
        if claimed[j] >= 0:
            assignment[j] = claimed[j]
            n_direct += 1
            continue

        cause = int(dg.freeze_cause[j])
        if cause in chosen_set:
            assignment[j] = cause
            n_via_cause += 1
            continue

        # The facility that froze j was pruned, so it conflicts with some open facility
        # that was processed no later than it.  Route j through their shared witness.
        target = -1
        if cause >= 0 and cause in W:
            for jstar in W[cause].tolist():
                owner = int(claimed[jstar])
                if owner >= 0 and dg.tight_time[owner] <= dg.tight_time[cause] + 1e-9:
                    target = owner
                    break
        if target >= 0:
            assignment[j] = target
            n_via_witness += 1
        else:
            # Should be unreachable; recorded rather than hidden so a real failure of
            # the witness argument shows up in the results instead of as a cost blip.
            opened = np.asarray(chosen, dtype=int)
            assignment[j] = int(opened[inst.d[j, opened].argmin()])
            n_fallback += 1

    return (
        np.asarray(sorted(chosen_set), dtype=int),
        assignment,
        {
            "n_f_temp": len(f_temp),
            "n_open": len(chosen),
            "n_direct": n_direct,
            "n_via_cause": n_via_cause,
            "n_via_witness": n_via_witness,
            "n_fallback": n_fallback,
        },
    )


def primal_dual_iffl(
    inst: Instance,
    lp: RestrictedLP | None = None,
    check_fairness: bool | None = None,
) -> dict:
    """Run Approach B end to end.

    Args:
        inst: instance carrying radii and costs.
        lp: a pre-solved restricted LP, used only as the comparison bound.
        check_fairness: assert ``d(j, sigma(j)) <= 3R`` and ``C + 3F <= 3 sum alpha``.
            Defaults to asserting only when the radii are uniform, which is the regime
            the proof covers.

    Returns:
        A dict with the realised cost, the structural bound check, and radius ratios.
    """
    if lp is None:
        lp = solve_restricted_lp(inst)

    uniform = bool(np.ptp(inst.R) <= 1e-12)
    if check_fairness is None:
        check_fairness = uniform

    dg = grow_duals(inst)
    opened, witness_assignment, pdiag = prune_and_assign(inst, dg)

    # The witness assignment exists to carry the proof: routing j through a shared
    # witness is what bounds d(j, i) by 3R and underpins C + 3F <= 3 sum(alpha).  It is
    # not what the algorithm should return.  Serving each client from the closest open
    # facility can only shorten its connection, so both bounds survive a fortiori --
    # d(j, S) <= d(j, witness) <= 3R, and a smaller C keeps C + 3F under the same
    # ceiling.  The witness cost is kept alongside for comparison.
    sol = evaluate_open_set(inst, opened)
    assignment = sol.assignment
    dists = inst.d[np.arange(inst.n), assignment]
    facility_cost = sol.facility_cost
    connection_cost = sol.connection_cost
    ratio = sol.radius_ratio

    witness_dists = inst.d[np.arange(inst.n), witness_assignment]
    witness_connection_cost = float(witness_dists.sum())
    alpha_sum = float(dg.alpha.sum())

    # Weak duality holds unconditionally, so a breach here means the freezing rules are
    # wrong rather than the bound being loose.
    if alpha_sum > lp.obj + 1e-5 * max(1.0, abs(lp.obj)):
        raise AssertionError(
            f"{inst.name}: weak duality violated -- sum alpha = {alpha_sum:.9g} > "
            f"OPT_LP = {lp.obj:.9g}"
        )

    structural_lhs = connection_cost + 3.0 * facility_cost
    structural_rhs = 3.0 * alpha_sum
    structural_ok = structural_lhs <= structural_rhs + 1e-5 * max(1.0, structural_rhs)

    if check_fairness:
        if not structural_ok:
            raise AssertionError(
                f"{inst.name}: structural bound violated -- C + 3F = "
                f"{structural_lhs:.9g} > 3 sum alpha = {structural_rhs:.9g}"
            )
        if sol.max_radius_ratio > 3.0 + 1e-6:
            raise AssertionError(
                f"{inst.name}: fairness bound violated -- max d(j,sigma(j))/R = "
                f"{sol.max_radius_ratio:.6g} > 3"
            )

    return {
        "algo": "primal_dual",
        "lp_obj": lp.obj,
        "radius_uniform": uniform,
        "cost": sol.cost,
        "facility_cost": facility_cost,
        "connection_cost": connection_cost,
        "cost_ratio": float(sol.cost / lp.obj) if lp.obj > 0 else float("nan"),
        "alpha_sum": alpha_sum,
        "structural_lhs": structural_lhs,
        "structural_rhs": structural_rhs,
        "structural_ok": structural_ok,
        "max_radius_ratio": sol.max_radius_ratio,
        "mean_radius_ratio": float(ratio.mean()),
        "n_open": int(opened.size),
        "n_events": dg.n_events,
        "witness_assignment": witness_assignment,
        "witness_connection_cost": witness_connection_cost,
        "witness_cost": facility_cost + witness_connection_cost,
        "witness_max_radius_ratio": float(
            radius_ratio(witness_dists, inst.R).max()
        ),
        "diagnostics": {**dg.diagnostics, **pdiag},
        "solution": sol,
    }
