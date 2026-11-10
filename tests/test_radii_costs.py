"""Radius schemes must always be feasible; cost synthesis must be scale-comparable."""

from __future__ import annotations

import numpy as np
import pytest

from iffl import costs, radii
from iffl.loaders import ALL_DATASETS, load_dataset

SCHEMES = [
    ("nearest_facility_multiple", 1.0),
    ("nearest_facility_multiple", 2.0),
    ("nearest_facility_multiple", 4.0),
    ("uniform", 1.0),
    ("uniform", 2.0),
    ("uniform", 3.0),
    ("jkl_client_knn", 5),
    ("jkl_client_knn", 20),
]


@pytest.mark.parametrize("name", ALL_DATASETS)
@pytest.mark.parametrize("scheme,param", SCHEMES)
def test_every_scheme_is_feasible(name, scheme, param):
    """R_j below the nearest-facility distance would make E_R empty and the LP infeasible."""
    inst = radii.apply_scheme(load_dataset(name), scheme, param)
    assert np.all(inst.R >= inst.nearest_facility_dist - 1e-12)
    inst.check_feasible()
    js, _ = inst.admissible_edges()
    assert np.array_equal(np.unique(js), np.arange(inst.n))


@pytest.mark.parametrize("name", ALL_DATASETS)
@pytest.mark.parametrize("q", [1.0, 2.0, 4.0])
def test_uniform_scheme_stays_uniform(name, q):
    """Approach B's proof needs one shared R, so feasibility must not be bought by
    clamping individual clients upward."""
    inst = radii.apply_scheme(load_dataset(name), "uniform", q)
    assert np.ptp(inst.R) == 0.0
    assert inst.meta["radius_uniform"] is True
    assert inst.meta["radius_clamped"] == 0


def test_nearest_multiple_scales_linearly():
    inst = load_dataset("synth-225-25")
    r1, _ = radii.nearest_facility_multiple(inst, 1.0)
    r3, _ = radii.nearest_facility_multiple(inst, 3.0)
    assert np.allclose(r3, 3.0 * r1)
    assert np.allclose(r1, inst.nearest_facility_dist)


def test_uniform_multiplier_actually_varies_the_radius():
    """The previous quantile parameterisation was degenerate: a long right tail in the
    nearest-facility distribution meant the feasibility floor beat every quantile, so
    all four swept values produced an identical radius on six of eight datasets, making
    those configurations duplicates."""
    inst = load_dataset("synth-500-50")
    vals = [float(radii.uniform(inst, m)[0][0]) for m in (1.0, 1.5, 2.0, 3.0)]
    assert len(set(np.round(vals, 9))) == 4
    assert vals == sorted(vals)


def test_uniform_multiplier_one_is_the_tightest_feasible_radius():
    """mult = 1 is exactly the smallest uniform radius leaving every client an edge."""
    for name in ("synth-500-50", "adult-500-20", "bank-500-20"):
        inst = load_dataset(name)
        R, meta = radii.uniform(inst, 1.0)
        assert R[0] == pytest.approx(inst.nearest_facility_dist.max())
        assert R[0] == pytest.approx(meta["radius_min_feasible"])
        # Anything smaller would strand the client farthest from any facility.
        stranded = inst.nearest_facility_dist > R[0] * 0.999
        assert stranded.any()


def test_uniform_rejects_an_infeasible_multiplier():
    with pytest.raises(ValueError, match="at least 1"):
        radii.uniform(load_dataset("synth-225-25"), 0.5)


def test_jkl_reports_clamping():
    inst = load_dataset("synth-500-50")
    _, meta = radii.jkl_client_knn(inst, 50)
    assert meta["radius_clamped"] >= 0
    assert meta["radius_rank"] == 11


def test_unknown_scheme_rejected():
    with pytest.raises(KeyError):
        radii.apply_scheme(load_dataset("synth-225-25"), "nope", 1.0)


def test_cost_synthesis_is_scale_comparable_across_datasets():
    """One scale means the same thing on z-scored features and raw lat/lon."""
    for name in ("adult-500-20", "uflp-49"):
        inst = load_dataset(name)
        f, meta = costs.synthesize(inst, scale=2.0, mode="constant")
        assert np.allclose(f, 2.0 * costs.distance_unit(inst))
        assert meta["cost_scale"] == 2.0


def test_native_costs_kept_unless_forced():
    inst = load_dataset("synth-225-25")
    kept = costs.ensure_costs(inst)
    assert np.array_equal(kept.f, inst.f)
    assert kept.meta["cost_mode"] == "native"

    forced = costs.ensure_costs(inst, scale=1.0, force=True)
    assert not np.array_equal(forced.f, inst.f)
    assert forced.meta["cost_synthesized"] is True


def test_costless_datasets_get_synthesized_costs():
    inst = load_dataset("bank-500-20")
    out = costs.ensure_costs(inst, scale=1.0)
    assert np.all(out.f > 0)
    assert out.meta["cost_synthesized"] is True


def test_random_cost_mode_is_seeded():
    inst = load_dataset("synth-225-25")
    a, _ = costs.synthesize(inst, 1.0, mode="uniform_random", seed=3)
    b, _ = costs.synthesize(inst, 1.0, mode="uniform_random", seed=3)
    c, _ = costs.synthesize(inst, 1.0, mode="uniform_random", seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- inverse-attractiveness opening costs -------------------------------------------


def test_attractiveness_weights_average_to_one():
    """The normalisation is what keeps `scale` meaning the same thing across instances
    whose distance scales differ by several-fold despite z-scored features."""
    for name in ("synth-500-50", "adult-500-20", "bank-500-20"):
        w = costs.attractiveness_weights(load_dataset(name))
        assert w.mean() == pytest.approx(1.0, abs=1e-12)
        assert np.all(w > 0)


def test_mean_opening_cost_is_scale_times_distance_unit():
    """Mean cost matches the constant scheme exactly, so the two are comparable and a
    swept `scale` spans the same regimes under both."""
    for name in ("adult-500-20", "bank-500-20"):
        inst = load_dataset(name)
        unit = costs.distance_unit(inst)
        for c in (4.0, 7.0, 15.0):
            f, meta = costs.synthesize(inst, c, mode="inverse_attractiveness")
            assert f.mean() == pytest.approx(c * unit, rel=1e-12)
            assert meta["cost_mode"] == "inverse_attractiveness"


def test_well_placed_facilities_cost_more():
    """The whole point of the scheme: a site whose nearest clients are close is
    desirable, so it is priced above one that must reach further."""
    inst = load_dataset("synth-500-50")
    a = np.sort(inst.d, axis=0)[: costs.ATTRACTIVENESS_K, :].mean(axis=0)
    w = costs.attractiveness_weights(inst)
    best, worst = int(np.argmin(a)), int(np.argmax(a))
    assert w[best] > w[worst]
    # and the ordering is exactly reversed from the reach distance
    assert np.all(np.argsort(w) == np.argsort(-a))


def test_attractiveness_costs_are_deterministic():
    """No seed: the scheme is a function of the geometry alone."""
    inst = load_dataset("bank-500-20")
    a, _ = costs.synthesize(inst, 7.0, mode="inverse_attractiveness", seed=1)
    b, _ = costs.synthesize(inst, 7.0, mode="inverse_attractiveness", seed=99)
    assert np.array_equal(a, b)


def test_uniform_sweeps_a_prefix_of_the_nearest_facility_values():
    """The two multiplier schemes share a parameter meaning but not a range, and that is
    deliberate.  A uniform radius is forced by feasibility to be at least
    ``max_j d(j, F)``, fixed by the single worst-placed client; under randomly drawn
    facilities that client sits far enough out that the radius already admits most
    client-facility pairs at ``mult = 1`` and essentially all of them by ``mult = 2``,
    so larger values repeat the same unconstrained instance.  Uniform is therefore swept
    only over the range where it still restricts something."""
    from iffl.configs import RADIUS_GRID

    by_scheme = {}
    for scheme, param in RADIUS_GRID:
        by_scheme.setdefault(scheme, []).append(param)
    near = by_scheme["nearest_facility_multiple"]
    uni = by_scheme["uniform"]
    assert uni == near[: len(uni)], "uniform values should be a prefix of the nearest-facility ones"
    assert len(uni) < len(near)


def test_uniform_radii_go_slack_under_random_placement():
    """Documents the reason for the trim above: the radius is set by the worst-placed
    client, so a single outlier makes the constraint vacuous for everyone else."""
    inst = load_dataset("adult-429-65")
    nn = inst.nearest_facility_dist
    assert nn.max() / np.median(nn) > 3, "the worst-placed client is a long way out"
    R, _ = radii.uniform(inst, 1.0)
    js, _ = inst.with_radii(R).admissible_edges()
    density = js.size / (inst.n * inst.m)
    assert density > 0.1, "even the tightest feasible uniform radius is already loose"
