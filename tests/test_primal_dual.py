"""Approach B: dual feasibility, the structural bound, and the 3R guarantee."""

from __future__ import annotations

import numpy as np
import pytest

from iffl import costs, radii
from iffl.loaders import load_dataset
from iffl.lp import solve_restricted_lp
from iffl.primal_dual import grow_duals, primal_dual_iffl, prune_and_assign
from iffl.synthetic import odd_cycle_instance

UNIFORM_CASES = [
    ("synth-225-25", 0.3),
    ("synth-500-50", 0.5),
    ("adult-500-20", 0.5),
    ("bank-500-20", 0.7),
    ("uflp-49", 0.5),
]


def _uniform_instance(name, q, scale=1.0):
    inst = radii.apply_scheme(load_dataset(name), "uniform", q)
    return costs.ensure_costs(inst, scale=scale, force=True)


@pytest.mark.parametrize("name,q", UNIFORM_CASES)
def test_dual_feasibility(name, q):
    """sum_j beta_ij <= f_i and alpha_j - beta_ij <= d(i,j) on every admissible edge."""
    inst = _uniform_instance(name, q)
    dg = grow_duals(inst)
    assert np.all(dg.beta.sum(axis=0) <= inst.f + 1e-6)
    js, is_ = inst.admissible_edges()
    assert np.all(dg.alpha[js] - dg.beta[js, is_] <= inst.d[js, is_] + 1e-6)
    assert np.all(dg.beta >= 0)


@pytest.mark.parametrize("name,q", UNIFORM_CASES)
def test_all_clients_freeze_with_a_cause(name, q):
    inst = _uniform_instance(name, q)
    dg = grow_duals(inst)
    assert np.all(dg.alpha > 0) or np.all(np.isfinite(dg.alpha))
    assert np.all(dg.freeze_cause >= 0)
    # The freezing facility must be admissible and must have gone tight by then.
    for j in range(inst.n):
        i = int(dg.freeze_cause[j])
        assert inst.d[j, i] <= inst.R[j] + 1e-9
        assert dg.tight_time[i] <= dg.alpha[j] + 1e-9
        assert inst.d[j, i] <= dg.alpha[j] + 1e-9


@pytest.mark.parametrize("name,q", UNIFORM_CASES)
def test_beta_is_zero_outside_admissible_edges(name, q):
    """Dual growth must stay inside E_R -- that restriction is the whole source of the
    fairness guarantee."""
    inst = _uniform_instance(name, q)
    dg = grow_duals(inst)
    forbidden = inst.d > inst.R[:, None] + 1e-12
    assert np.all(dg.beta[forbidden] == 0.0)


@pytest.mark.parametrize("name,q", UNIFORM_CASES)
def test_witness_sets_of_open_facilities_are_disjoint(name, q):
    """The MIS is taken on the conflict graph, so open facilities share no witness --
    which is what makes 3F exactly payable by the witnesses."""
    inst = _uniform_instance(name, q)
    dg = grow_duals(inst)
    opened, _, _ = prune_and_assign(inst, dg)
    W = dg.witness_sets()
    seen: set[int] = set()
    for i in opened.tolist():
        cur = set(W[i].tolist())
        assert not (cur & seen)
        seen |= cur


@pytest.mark.parametrize("name,q", UNIFORM_CASES)
def test_facility_cost_paid_exactly_by_witnesses(name, q):
    """f_i = sum_{j in W(i)} beta_ij for every open facility."""
    inst = _uniform_instance(name, q)
    dg = grow_duals(inst)
    opened, _, _ = prune_and_assign(inst, dg)
    for i in opened.tolist():
        assert dg.beta[:, i].sum() == pytest.approx(inst.f[i], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("name,q", UNIFORM_CASES)
def test_uniform_radii_give_3_approximation(name, q):
    """Theorem 2.2: d(j, sigma(j)) <= 3R and C + 3F <= 3 OPT_LP."""
    inst = _uniform_instance(name, q)
    res = primal_dual_iffl(inst)
    assert res["radius_uniform"] is True
    assert res["max_radius_ratio"] <= 3.0 + 1e-6
    assert res["structural_ok"]
    assert res["structural_lhs"] <= 3.0 * res["lp_obj"] + 1e-5 * abs(res["lp_obj"])


@pytest.mark.parametrize("name,q", UNIFORM_CASES)
def test_weak_duality(name, q):
    inst = _uniform_instance(name, q)
    res = primal_dual_iffl(inst)
    assert res["alpha_sum"] <= res["lp_obj"] + 1e-5 * max(1.0, abs(res["lp_obj"]))


def test_odd_cycle_uniform_bound_holds():
    for k in (5, 7, 9, 11):
        res = primal_dual_iffl(odd_cycle_instance(k, facility_cost=10.0))
        assert res["max_radius_ratio"] <= 3.0 + 1e-9
        assert res["structural_ok"]


def test_no_fallback_assignments_needed():
    """The witness argument should always find a route; a fallback means it failed."""
    for name, q in UNIFORM_CASES:
        res = primal_dual_iffl(_uniform_instance(name, q))
        assert res["diagnostics"]["n_fallback"] == 0


def test_indirect_assignment_path_is_exercised():
    """Guards against the tests passing only because every client assigns directly."""
    total_indirect = 0
    for name, q in UNIFORM_CASES:
        d = primal_dual_iffl(_uniform_instance(name, q))["diagnostics"]
        total_indirect += d["n_via_cause"] + d["n_via_witness"]
    assert total_indirect > 0


def test_nonuniform_radii_break_the_bound_as_remark_2_3_predicts():
    """Remark 2.3: with heterogeneous radii the witness's R_{j*} can dwarf R_j, so no
    bound relative to R_j survives.  The algorithm still runs; it just violates 3R."""
    inst = radii.apply_scheme(
        load_dataset("synth-225-25"), "nearest_facility_multiple", 2.0
    )
    inst = costs.ensure_costs(inst)
    res = primal_dual_iffl(inst, check_fairness=False)
    assert res["radius_uniform"] is False
    assert res["max_radius_ratio"] > 3.0


def test_zero_cost_facilities_are_tight_immediately():
    inst = _uniform_instance("synth-225-25", 0.5).with_costs(np.zeros(10))
    dg = grow_duals(inst)
    assert np.all(dg.tight_time == 0.0)


def test_missing_radii_rejected():
    inst = load_dataset("synth-225-25")
    with pytest.raises(ValueError, match="no radii"):
        grow_duals(inst)
