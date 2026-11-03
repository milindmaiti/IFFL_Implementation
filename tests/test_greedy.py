"""Approach C: greedy UFL (Williamson & Shmoys Alg. 9.2) restricted to E_R."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from iffl import costs, radii
from iffl.greedy import _best_move, greedy_iffl
from iffl.loaders import load_dataset
from iffl.lp import solve_restricted_lp
from iffl.synthetic import line_instance, odd_cycle_instance

CASES = [
    ("synth-225-25", "nearest_facility_multiple", 2.0, 1.0),
    ("synth-225-25", "uniform", 0.5, 5.0),
    ("synth-500-50", "uniform", 0.6, 10.0),
    ("adult-500-20", "uniform", 0.5, 2.0),
    ("adult-429-65", "nearest_facility_multiple", 3.0, 5.0),
    ("bank-500-20", "uniform", 0.7, 1.0),
    ("uflp-49", "uniform", 0.5, 25.0),
]


def _inst(name, scheme, param, scale):
    rank = 2 if name.startswith("uflp") else 1
    out = radii.apply_scheme(load_dataset(name), scheme, param, rank=rank)
    return costs.ensure_costs(out, scale=scale, force=True)


@pytest.mark.parametrize("name,scheme,param,scale", CASES)
def test_output_is_strictly_fair(name, scheme, param, scale):
    """The whole point of the adaptation: every client inside its *own* radius, not 3R."""
    inst = _inst(name, scheme, param, scale)
    res = greedy_iffl(inst)
    sol = res["solution"]
    dists = inst.d[np.arange(inst.n), sol.assignment]
    assert np.all(dists <= inst.R + 1e-9)
    assert res["max_radius_ratio"] <= 1.0 + 1e-9


@pytest.mark.parametrize("name,scheme,param,scale", CASES)
def test_every_client_assigned_to_an_open_admissible_facility(name, scheme, param, scale):
    inst = _inst(name, scheme, param, scale)
    sol = greedy_iffl(inst)["solution"]
    assert np.all(np.isin(sol.assignment, sol.open_facilities))
    assert np.all(inst.d[np.arange(inst.n), sol.assignment] <= inst.R + 1e-9)


@pytest.mark.parametrize("name,scheme,param,scale", CASES)
def test_terminates_within_n_rounds(name, scheme, param, scale):
    """Each round connects at least one client, so the loop is bounded by n."""
    inst = _inst(name, scheme, param, scale)
    res = greedy_iffl(inst)
    assert 1 <= res["diagnostics"]["n_rounds"] <= inst.n


@pytest.mark.parametrize("name,scheme,param,scale", CASES)
def test_facility_cost_paid_once_per_facility(name, scheme, param, scale):
    """``f_i <- 0`` makes reopening free, so cost counts distinct facilities only."""
    inst = _inst(name, scheme, param, scale)
    res = greedy_iffl(inst)
    sol = res["solution"]
    assert sol.facility_cost == pytest.approx(inst.f[sol.open_facilities].sum())
    assert len(set(sol.open_facilities.tolist())) == len(sol.open_facilities)


def test_prefix_scan_matches_exhaustive_subset_search():
    """The one clever step: for a fixed facility, only the k nearest candidates can be
    the optimal Y of size k, so scanning prefixes examines every subset that could win.
    Verified here against a brute-force search over all subsets on a tiny instance."""
    rng = np.random.default_rng(0)
    for trial in range(25):
        n, m = 6, 3
        d = rng.uniform(0.1, 4.0, size=(n, m))
        R = np.full(n, 3.0)
        in_er = d <= R[:, None]
        # Guarantee feasibility so the brute force has something to find.
        in_er[np.arange(n), rng.integers(0, m, n)] = True
        f = rng.uniform(0.0, 5.0, size=m)
        unconnected = rng.random(n) < 0.7
        if not unconnected.any():
            continue
        cur = np.where(unconnected, np.inf, rng.uniform(0.1, 4.0, size=n))

        ratio, i, Y = _best_move(d, in_er, f, unconnected, cur)

        # Brute force: every facility, every nonempty admissible unconnected subset.
        connected = ~unconnected
        best = np.inf
        for ii in range(m):
            sav = float(
                np.where(in_er[:, ii] & connected, np.maximum(cur - d[:, ii], 0.0), 0.0).sum()
            )
            cand = [j for j in range(n) if unconnected[j] and in_er[j, ii]]
            for size in range(1, len(cand) + 1):
                for sub in itertools.combinations(cand, size):
                    val = (f[ii] - sav + sum(d[j, ii] for j in sub)) / len(sub)
                    best = min(best, val)
        assert ratio == pytest.approx(best, rel=1e-9, abs=1e-12), f"trial {trial}"


def test_savings_term_is_actually_exercised():
    """Guards against the middle term being dead code: without it the algorithm is a
    different (and weaker) one, so a run where it never fires proves nothing."""
    inst = _inst("synth-500-50", "uniform", 0.6, 5.0)
    d = inst.d
    in_er = d <= inst.R[:, None] + 1e-12
    f_work = inst.f.astype(float).copy()
    unconnected = np.ones(inst.n, bool)
    cur = np.full(inst.n, np.inf)
    fired = 0
    rounds = 0
    while unconnected.any():
        sav = np.where(
            in_er & (~unconnected)[:, None], np.maximum(cur[:, None] - d, 0.0), 0.0
        ).sum(axis=0)
        fired += int(np.any(sav > 1e-9))
        _, i, Y = _best_move(d, in_er, f_work, unconnected, cur)
        f_work[i] = 0.0
        unconnected[Y] = False
        imp = in_er[:, i] & ~unconnected
        cur[imp] = np.minimum(cur[imp], d[imp, i])
        rounds += 1
    assert fired > 0.5 * rounds


def test_never_uses_an_inadmissible_edge():
    """The fair adaptation forbids pairs outside E_R at every step, including as a
    source of savings."""
    inst = _inst("adult-429-65", "nearest_facility_multiple", 1.5, 2.0)
    sol = greedy_iffl(inst)["solution"]
    assert np.all(inst.d[np.arange(inst.n), sol.assignment] <= inst.R + 1e-9)


def test_line_instance_finds_the_optimum():
    """Clients at 0,1,9,10; facilities at 0,5,10; R=2 excludes the middle one, so the
    only fair solution opens both ends: cost 2*1 + (0+1+1+0) = 4."""
    res = greedy_iffl(line_instance())
    assert res["cost"] == pytest.approx(4.0)
    assert res["cost_ratio"] == pytest.approx(1.0)


def test_odd_cycle_beats_the_integral_cover_bound():
    """On the half-integral witness the greedy must still produce a real integral
    solution: ceil(k/2) facilities at cost 10, plus unit connections."""
    for k in (5, 9, 15):
        inst = odd_cycle_instance(k, facility_cost=10.0)
        res = greedy_iffl(inst)
        assert res["max_radius_ratio"] <= 1.0 + 1e-9
        assert res["n_open"] >= (k + 1) // 2
        assert res["cost"] == pytest.approx(10.0 * res["n_open"] + k)


@pytest.mark.parametrize("name,scheme,param,scale", CASES)
def test_cost_is_at_least_the_lp_bound(name, scheme, param, scale):
    """Unlike the bicriteria algorithms, this one is strictly fair, so the restricted LP
    really is a lower bound on it -- it cannot undercut the LP the way A and B can."""
    inst = _inst(name, scheme, param, scale)
    lp = solve_restricted_lp(inst)
    res = greedy_iffl(inst, lp=lp)
    assert res["cost"] >= lp.obj - 1e-6 * max(1.0, lp.obj)


def test_missing_radii_rejected():
    with pytest.raises(ValueError, match="no radii"):
        greedy_iffl(load_dataset("synth-225-25"))
