"""Degenerate-radius handling, which the UFLP workbooks make unavoidable.

In ``data49UFLP.xlsx`` and ``data88UFLP.xlsx`` clients and facilities are the same point
set, so every client's rank-1 facility distance is exactly 0.  That makes the natural
radius schemes collapse, and makes ``d / R`` a 0/0 form, so both paths need explicit
handling rather than whatever the floating-point default happens to be.
"""

from __future__ import annotations

import numpy as np
import pytest

from iffl import radii
from iffl.evaluate import greedy_nearest_feasible, radius_ratio, threshold_round
from iffl.loaders import load_dataset
from iffl.lp import solve_restricted_lp
from iffl.primal_dual import primal_dual_iffl


def test_radius_ratio_resolves_zero_over_zero_by_the_constraint():
    """d <= 3R holds when both are 0, so the ratio must read 0, not nan."""
    out = radius_ratio(np.array([0.0, 0.0, 2.0, 1.0]), np.array([0.0, 1.0, 0.0, 2.0]))
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == np.inf  # served at positive distance with zero radius: a real failure
    assert out[3] == 0.5
    assert not np.isnan(out).any()


def test_radius_ratio_never_hides_a_violation_as_nan():
    """nan would defeat the <= 3 assertions silently, since nan compares false both ways."""
    out = radius_ratio(np.array([5.0]), np.array([0.0]))
    assert out[0] > 3.0


@pytest.mark.parametrize("name", ["uflp-49", "uflp-88"])
def test_rank_one_is_degenerate_on_uflp(name):
    """Documents the trap: clients sit on facility sites, so rank-1 radii are all 0."""
    inst = load_dataset(name)
    assert np.all(inst.nearest_facility_dist == 0.0)
    for scheme, param in (("nearest_facility_multiple", 4.0), ("uniform", 0.9)):
        out = radii.apply_scheme(inst, scheme, param, rank=1)
        assert out.meta["radius_degenerate"] is True
        assert np.all(out.R == 0.0)
        # Every client is then served only by itself, forcing all m facilities open.
        lp = solve_restricted_lp(out)
        assert int(np.count_nonzero(lp.y > 0.5)) == inst.m


@pytest.mark.parametrize("name", ["uflp-49", "uflp-88"])
@pytest.mark.parametrize("scheme,param", [("nearest_facility_multiple", 2.0), ("uniform", 0.5)])
def test_rank_two_restores_a_meaningful_instance(name, scheme, param):
    inst = radii.apply_scheme(load_dataset(name), scheme, param, rank=2)
    assert inst.meta["radius_degenerate"] is False
    assert np.all(inst.R > 0)
    lp = solve_restricted_lp(inst)
    assert int(np.count_nonzero(lp.y > 0.5)) < inst.m


@pytest.mark.parametrize("name", ["uflp-49", "uflp-88"])
def test_primal_dual_runs_on_degenerate_radii_without_nan(name):
    """Even the trivial rank-1 instance must produce a real, assertable ratio."""
    inst = radii.apply_scheme(load_dataset(name), "uniform", 0.5, rank=1)
    res = primal_dual_iffl(inst)
    assert not np.isnan(res["max_radius_ratio"])
    assert res["max_radius_ratio"] <= 3.0 + 1e-9


def test_threshold_baseline_returns_nothing_on_a_fractional_lp():
    """The heuristic_round pattern has no guarantee: it opens nothing when y is
    half-integral, which is precisely the gap the paper's rounding fills."""
    from iffl.synthetic import odd_cycle_instance

    inst = odd_cycle_instance(5, facility_cost=10.0)
    lp = solve_restricted_lp(inst)
    assert threshold_round(inst, lp).size == 0


def test_threshold_baseline_matches_lp_when_integral():
    inst = radii.apply_scheme(load_dataset("synth-225-25"), "uniform", 0.5)
    lp = solve_restricted_lp(inst)
    assert np.array_equal(threshold_round(inst, lp), np.flatnonzero(lp.y > 0.5))


def test_greedy_baseline_is_strictly_fair():
    inst = radii.apply_scheme(load_dataset("synth-225-25"), "nearest_facility_multiple", 2.0)
    opened = greedy_nearest_feasible(inst)
    dists = inst.d[:, opened].min(axis=1)
    assert np.all(dists <= inst.R + 1e-9)
