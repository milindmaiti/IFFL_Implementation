"""Approach A: structural invariants and the deterministic 3R_j guarantee."""

from __future__ import annotations

import numpy as np
import pytest

from iffl import costs, radii
from iffl.loaders import load_dataset
from iffl.lp import solve_restricted_lp
from iffl.lp_rounding import (
    GAMMA,
    build_copy_structure,
    cluster_centers,
    evaluate_open_set,
    lp_rounding_iffl,
    round_once,
)
from iffl.synthetic import odd_cycle_instance, violates_triangle

FRACTIONAL = [odd_cycle_instance(k, facility_cost=10.0) for k in (5, 7, 9)]


def _real_instance(name="synth-225-25", scheme="nearest_facility_multiple", param=2.0):
    inst = radii.apply_scheme(load_dataset(name), scheme, param)
    return costs.ensure_costs(inst, scale=1.0)


def test_gamma_is_69_over_40():
    assert GAMMA == pytest.approx(1.725)


@pytest.mark.parametrize("inst", FRACTIONAL, ids=lambda i: i.name)
def test_odd_cycle_is_a_true_metric(inst):
    """The 3R argument walks client -> facility -> client -> facility, so the whole of
    F union C must satisfy the triangle inequality, not just the client/facility block."""
    assert violates_triangle(inst.meta["full_metric"]) <= 1e-9


@pytest.mark.parametrize("inst", FRACTIONAL, ids=lambda i: i.name)
def test_splitting_gives_unit_support_mass(inst):
    """After splitting, every client's support carries exactly one unit of mass."""
    lp = solve_restricted_lp(inst)
    cs = build_copy_structure(inst, lp)
    assert cs.diagnostics["support_mass_max_dev"] < 1e-6
    for j in range(inst.n):
        total = cs.mass[cs.close[j]].sum() + cs.mass[cs.distant[j]].sum()
        assert total == pytest.approx(GAMMA, abs=1e-6)


@pytest.mark.parametrize("inst", FRACTIONAL, ids=lambda i: i.name)
def test_filtering_invariants(inst):
    """Equation (3): close mass is exactly 1, distant mass exactly gamma - 1."""
    lp = solve_restricted_lp(inst)
    cs = build_copy_structure(inst, lp)
    for j in range(inst.n):
        assert cs.mass[cs.close[j]].sum() == pytest.approx(1.0, abs=1e-6)
        assert cs.mass[cs.distant[j]].sum() == pytest.approx(GAMMA - 1.0, abs=1e-6)


@pytest.mark.parametrize("inst", FRACTIONAL, ids=lambda i: i.name)
def test_every_copy_mass_at_most_one(inst):
    """The independent Bernoulli step needs y_c <= 1 to be a valid probability."""
    lp = solve_restricted_lp(inst)
    cs = build_copy_structure(inst, lp)
    assert cs.mass.max() <= 1.0 + 1e-12
    assert cs.mass.min() > 0.0


@pytest.mark.parametrize("inst", FRACTIONAL, ids=lambda i: i.name)
def test_dmax_within_radius(inst):
    """Equation (4): d^max_j <= R_j, which is what makes the 3R bound deterministic."""
    lp = solve_restricted_lp(inst)
    cs = build_copy_structure(inst, lp)
    assert np.all(cs.d_max <= inst.R + 1e-9)


@pytest.mark.parametrize("inst", FRACTIONAL, ids=lambda i: i.name)
def test_lemma_2_close_sets_pairwise_disjoint(inst):
    """Lemma 2: distinct centers have disjoint close sets, which is what makes the
    one-copy-per-group rounding well defined."""
    lp = solve_restricted_lp(inst)
    cs = build_copy_structure(inst, lp)
    centers, sigma = cluster_centers(cs, inst.n)
    seen: set[int] = set()
    for s in centers:
        cur = set(cs.close[s].tolist())
        assert not (cur & seen)
        seen |= cur
    # Lemma 2 also requires each client's center to score no worse than the client.
    for j in range(inst.n):
        assert cs.d_max[sigma[j]] <= cs.d_max[j] + 1e-12
        assert set(cs.close[j].tolist()) & set(cs.close[sigma[j]].tolist())


@pytest.mark.parametrize("inst", FRACTIONAL, ids=lambda i: i.name)
def test_rounding_opens_exactly_one_copy_per_center_group(inst):
    lp = solve_restricted_lp(inst)
    cs = build_copy_structure(inst, lp)
    centers, _ = cluster_centers(cs, inst.n)
    rng = np.random.default_rng(0)
    for _ in range(25):
        selected = set(round_once(cs, centers, rng).tolist())
        for s in centers:
            assert len(selected & set(cs.close[s].tolist())) == 1


@pytest.mark.parametrize("inst", FRACTIONAL, ids=lambda i: i.name)
def test_deterministic_fairness_holds_in_every_outcome(inst):
    """d(j, S) <= 3 R_j must hold on every trial, not just in expectation."""
    res = lp_rounding_iffl(inst, trials=400, seed=11)
    assert res["max_radius_ratio"] <= 3.0 + 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_deterministic_fairness_on_real_data(seed):
    inst = _real_instance()
    res = lp_rounding_iffl(inst, trials=40, seed=seed)
    assert res["max_radius_ratio"] <= 3.0 + 1e-9


@pytest.mark.parametrize("inst", FRACTIONAL, ids=lambda i: i.name)
def test_expected_cost_within_1_725(inst):
    """The cost bound is in expectation, so it is the mean over trials that must hold."""
    res = lp_rounding_iffl(inst, trials=2000, seed=5)
    assert res["cost_ratio_mean"] <= 1.725 + 0.02


def test_rounding_is_actually_randomized_on_fractional_instances():
    """Guards against the degenerate case: on an integral LP the rounding is
    deterministic, which would make the fairness and cost tests vacuous."""
    res = lp_rounding_iffl(odd_cycle_instance(9, facility_cost=10.0), trials=300, seed=2)
    assert len(set(res["costs"])) > 1


def test_integral_lp_rounds_to_the_lp_solution():
    """Sanity check of the opposite regime: every Fair-FL dataset has an integral LP,
    where the rounding should reproduce the LP exactly."""
    inst = _real_instance()
    lp = solve_restricted_lp(inst)
    assert np.all((lp.y < 1e-6) | (lp.y > 1 - 1e-6))
    res = lp_rounding_iffl(inst, trials=10, seed=0, lp=lp)
    assert res["cost_ratio_mean"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_open_set_rejects_empty():
    inst = _real_instance()
    with pytest.raises(ValueError, match="no facility"):
        evaluate_open_set(inst, np.array([], dtype=int))
