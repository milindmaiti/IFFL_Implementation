"""The restricted LP must be a correct lower bound on the fair optimum."""

from __future__ import annotations

import numpy as np
import pytest

from iffl import radii
from iffl.loaders import load_dataset
from iffl.lp import solve_restricted_lp
from iffl.synthetic import line_instance, odd_cycle_instance


def test_line_instance_matches_hand_computation():
    """Clients at 0,1,9,10; facilities at 0,5,10; R=2 rules the middle one out, so the
    LP must open both ends at cost 1 each and pay connections 0+1+1+0."""
    inst = line_instance()
    lp = solve_restricted_lp(inst)
    assert lp.obj == pytest.approx(4.0)
    assert lp.y[0] == pytest.approx(1.0)
    assert lp.y[2] == pytest.approx(1.0)
    assert lp.y[1] == pytest.approx(0.0)
    assert lp.F_star == pytest.approx(2.0)
    assert lp.C_star.sum() == pytest.approx(2.0)


def test_lp_only_uses_admissible_edges():
    """Variables exist only for E_R; a forbidden pair would break the lower bound."""
    inst = radii.apply_scheme(load_dataset("synth-225-25"), "nearest_facility_multiple", 2.0)
    lp = solve_restricted_lp(inst)
    assert np.all(inst.d[lp.edge_client, lp.edge_facility] <= inst.R[lp.edge_client] + 1e-9)


def test_lp_constraints_satisfied():
    inst = radii.apply_scheme(load_dataset("synth-225-25"), "uniform", 1.5)
    lp = solve_restricted_lp(inst)
    served = np.zeros(inst.n)
    np.add.at(served, lp.edge_client, lp.x)
    assert np.allclose(served, 1.0, atol=1e-6)
    assert np.all(lp.x <= lp.y[lp.edge_facility] + 1e-6)
    assert np.all(lp.y >= -1e-9) and np.all(lp.y <= 1 + 1e-9)


def test_objective_decomposes():
    inst = radii.apply_scheme(load_dataset("bank-500-20"), "uniform", 1.5)
    from iffl import costs

    inst = costs.ensure_costs(inst, scale=1.0)
    lp = solve_restricted_lp(inst)
    assert lp.obj == pytest.approx(lp.F_star + lp.C_star.sum(), rel=1e-7)


def test_odd_cycle_lp_is_half_integral():
    """The fractional witness: fractional vertex cover of an odd cycle gives y = 1/2."""
    for k in (5, 7, 9):
        lp = solve_restricted_lp(odd_cycle_instance(k, facility_cost=10.0))
        assert np.allclose(lp.y, 0.5, atol=1e-6)
        # k/2 facilities at cost 10, plus unit connection for each of k clients.
        assert lp.obj == pytest.approx(10.0 * k / 2 + k)


def test_lp_lower_bounds_integral_fair_optimum():
    """Brute force the best strictly-fair integral solution on the small line instance."""
    from itertools import chain, combinations

    inst = line_instance()
    lp = solve_restricted_lp(inst)
    best = np.inf
    idx = range(inst.m)
    for size in idx:
        for sub in combinations(idx, size + 1):
            sub = list(sub)
            dists = inst.d[:, sub].min(axis=1)
            if np.any(dists > inst.R + 1e-12):
                continue  # not strictly fair
            best = min(best, inst.f[sub].sum() + dists.sum())
    assert lp.obj <= best + 1e-9


def test_missing_radii_rejected():
    inst = load_dataset("synth-225-25")
    with pytest.raises(ValueError, match="no radii"):
        solve_restricted_lp(inst)
