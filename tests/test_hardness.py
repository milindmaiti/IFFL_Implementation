"""The Set Cover reduction behind Theorem 2, and the separation it produces."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iffl.greedy import greedy_iffl
from iffl.hardness import (
    full_metric,
    greedy_tight_set_cover,
    optimal_cost,
    random_set_cover,
    set_cover_instance,
)
from iffl.lp import solve_restricted_lp
from iffl.lp_rounding import lp_rounding_iffl
from iffl.primal_dual import primal_dual_iffl

KS = [2, 3, 4, 5, 6]


@pytest.mark.parametrize("k", KS)
def test_construction_matches_the_paper(k):
    """Two clients per element, one facility per set, M = q^2, R = 1."""
    q, sets, _ = greedy_tight_set_cover(k)
    inst = set_cover_instance(q, sets)
    assert inst.n == 2 * q
    assert inst.m == len(sets)
    assert np.all(inst.f == q * q)
    assert np.all(inst.R == 1.0)


@pytest.mark.parametrize("k", KS)
def test_distances_are_one_two_three_and_metric(k):
    """All nonzero distances lie in {1,2,3} and the whole of F u C is a metric --
    the 3R argument walks client -> facility -> client -> facility."""
    q, sets, _ = greedy_tight_set_cover(k)
    D = full_metric(q, sets)
    assert set(np.round(D[D > 0].ravel(), 9).tolist()) <= {1.0, 2.0, 3.0}
    assert np.allclose(D, D.T)
    worst = max(float((D - (D[:, [b]] + D[[b], :])).max()) for b in range(D.shape[0]))
    assert worst <= 1e-9


@pytest.mark.parametrize("k", KS)
def test_admissible_edges_are_exactly_the_incidences(k):
    """E_R under R = 1 is the set-cover incidence structure, which is what makes any
    radius violation below three preserve the covering constraint."""
    q, sets, _ = greedy_tight_set_cover(k)
    inst = set_cover_instance(q, sets)
    js, is_ = inst.admissible_edges()
    for j, i in zip(js.tolist(), is_.tolist()):
        assert (j // 2) in sets[i]
    assert js.size == 2 * sum(len(S) for S in sets)


@pytest.mark.parametrize("k", KS)
def test_lemma_5_optimal_cost(k):
    """OPT_FL = M k* + 2q, and the restricted LP attains it on this family."""
    q, sets, kopt = greedy_tight_set_cover(k)
    inst = set_cover_instance(q, sets)
    assert solve_restricted_lp(inst).obj == pytest.approx(optimal_cost(q, kopt), rel=1e-9)


def test_greedy_cost_grows_logarithmically():
    """Theorem 2 made concrete: a strictly fair algorithm cannot stay within a constant
    factor.  Greedy opens one facility per block -- k of them, against the optimum's two
    -- so its ratio tracks k/2 = (log2 N - 2)/2."""
    ratios, logs = [], []
    for k in range(3, 11):
        q, sets, kopt = greedy_tight_set_cover(k)
        inst = set_cover_instance(q, sets)
        res = greedy_iffl(inst)
        assert res["n_open"] == k            # greedy takes every block
        ratios.append(res["cost"] / optimal_cost(q, kopt))
        logs.append(math.log2(2 * q))
    assert ratios == sorted(ratios)          # monotonically worse
    assert ratios[-1] > 4.0                  # unbounded, not a constant
    A = np.vstack([logs, np.ones(len(logs))]).T
    slope, icpt = np.linalg.lstsq(A, np.asarray(ratios), rcond=None)[0]
    assert slope == pytest.approx(0.5, abs=0.05)   # theory: 0.5 * log2 N - 1
    assert icpt == pytest.approx(-1.0, abs=0.2)


@pytest.mark.parametrize("k", [3, 5, 7])
def test_bicriteria_algorithms_stay_bounded(k):
    """The same family that defeats strict fairness is easy once a factor-3 radius
    violation is allowed."""
    q, sets, kopt = greedy_tight_set_cover(k)
    inst = set_cover_instance(q, sets)
    OPT = optimal_cost(q, kopt)
    lp = solve_restricted_lp(inst)
    a = lp_rounding_iffl(inst, trials=10, seed=0, lp=lp)
    b = primal_dual_iffl(inst, lp=lp)
    assert a["cost_mean"] / OPT <= 1.725
    assert b["cost"] / OPT <= 3.0
    assert a["max_radius_ratio"] <= 3.0 + 1e-9
    assert b["max_radius_ratio"] <= 3.0 + 1e-9


def test_greedy_stays_strictly_fair_throughout():
    """Its cost blows up precisely because it refuses to violate the radius."""
    for k in (3, 6, 9):
        q, sets, _ = greedy_tight_set_cover(k)
        res = greedy_iffl(set_cover_instance(q, sets))
        assert res["max_radius_ratio"] <= 1.0 + 1e-9


def test_greedy_is_near_optimal_on_random_instances():
    """The blow-up is a worst-case phenomenon, not typical behaviour."""
    q, sets, _ = random_set_cover(400, 60, set_size=50, seed=2)
    inst = set_cover_instance(q, sets)
    res = greedy_iffl(inst)
    assert res["cost"] / optimal_cost(q, 15) < 3.0


def test_sets_must_cover_the_universe():
    with pytest.raises(ValueError, match="cover the universe"):
        set_cover_instance(5, [{0, 1}, {2}])
