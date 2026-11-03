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
    ("uniform", 0.25),
    ("uniform", 0.5),
    ("uniform", 0.9),
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
@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
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


def test_uniform_reports_lifting_when_percentile_is_infeasible():
    """A low percentile is below some clients' floor, so the shared value is lifted."""
    inst = load_dataset("synth-225-25")
    R, meta = radii.uniform(inst, 0.01)
    assert meta["radius_lifted_for_feasibility"] is True
    assert R[0] == pytest.approx(inst.nearest_facility_dist.max())
    assert meta["radius_effective_quantile"] == pytest.approx(1.0)


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
