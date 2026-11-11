"""The upstream pipeline must reproduce the bundled instances exactly.

That equivalence is the whole point: it is what makes a freshly sampled 5,000-client
instance comparable with the bundled 500-client one, rather than merely similar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iffl.upstream import (
    FACILITY_MODES,
    load_adult_pool,
    load_bank_pool,
    make_facilities,
    sample_instance,
)

CACHE = Path(".cache")
pytestmark = pytest.mark.skipif(
    not (CACHE / "adult.zip").exists() or not (CACHE / "bank.zip").exists(),
    reason="upstream archives not cached; run scripts/run_experiments.py on an\n            upstream: dataset once to fetch",
)


def test_adult_pool_reproduces_bundled_coordinates():
    """adult.data + adult.test, drop rows with '?', z-score with population std."""
    pool = load_adult_pool(CACHE)
    assert len(pool) == 45222
    ours = pd.read_csv("data/adult-429-65.csv")
    got = ours[[f"d{i}" for i in range(1, 7)]].to_numpy()
    assert np.abs(pool.coords[ours["id"].to_numpy()] - got).max() < 1e-9


def test_adult_group_columns_align():
    pool = load_adult_pool(CACHE)
    ours = pd.read_csv("data/adult-429-65.csv")
    idx = ours["id"].to_numpy()
    assert (pool.groups["race"].to_numpy()[idx] == ours["c1"].to_numpy()).all()
    assert (pool.groups["sex"].to_numpy()[idx] == ours["c2"].to_numpy()).all()


def test_bank_pool_reproduces_bundled_coordinates():
    """The bundled bank instance comes from bank.csv (4,521 rows), not bank-full.csv."""
    pool = load_bank_pool(CACHE, full=False)
    assert len(pool) == 4521
    ours = pd.read_csv("data/bank-500-20.csv", sep=r"\s+", engine="python")
    got = ours[["d1", "d2", "d3"]].to_numpy()
    assert np.abs(pool.coords[ours["id"].to_numpy()] - got).max() < 1e-9


def test_bank_group_columns_align():
    """c4 is the target y, not 'default'."""
    pool = load_bank_pool(CACHE, full=False)
    ours = pd.read_csv("data/bank-500-20.csv", sep=r"\s+", engine="python")
    idx = ours["id"].to_numpy()
    for col, src in zip(("c1", "c2", "c3", "c4"), ("job", "marital", "education", "y")):
        assert (pool.groups[src].to_numpy()[idx] == ours[col].to_numpy()).all(), col


def test_bank_full_pool_is_ten_times_larger():
    assert len(load_bank_pool(CACHE, full=True)) == 45211


def test_sampling_is_seeded_and_reproducible():
    pool = load_adult_pool(CACHE)
    a, fa = sample_instance(pool, 300, 15, seed=7)
    b, fb = sample_instance(pool, 300, 15, seed=7)
    c, _ = sample_instance(pool, 300, 15, seed=8)
    assert a.equals(b)
    # Facility coordinates come from k-means, whose BLAS reductions are not
    # bit-reproducible across runs; the seeding guarantee is agreement to tolerance,
    # not identical floats.
    assert np.allclose(fa.to_numpy(), fb.to_numpy(), atol=1e-12)
    assert not a["id"].equals(c["id"])


def test_sample_has_the_bundled_schema():
    pool = load_adult_pool(CACHE)
    clients, facs = sample_instance(pool, 250, 12, seed=0)
    assert list(clients.columns) == ["id", *[f"d{i}" for i in range(1, 7)], "c1", "c2"]
    assert list(facs.columns) == [f"d{i}" for i in range(1, 7)]
    assert len(clients) == 250 and len(facs) == 12
    assert clients["id"].is_unique


def test_oversized_request_rejected():
    pool = load_bank_pool(CACHE, full=False)
    with pytest.raises(ValueError, match="pool has only"):
        sample_instance(pool, len(pool) + 1, 10)


@pytest.mark.parametrize("mode", FACILITY_MODES)
def test_facility_modes_produce_the_right_shape(mode):
    pool = load_adult_pool(CACHE)
    clients, _ = sample_instance(pool, 200, 10, seed=0)
    C = clients[[f"d{i}" for i in range(1, 7)]].to_numpy()
    F = make_facilities(pool, C, 10, mode=mode, seed=0)
    assert F.shape == (10, 6)
    assert np.all(np.isfinite(F))


def test_upstream_instance_spec_round_trip():
    from iffl.pipeline import upstream_instance

    inst = upstream_instance("upstream:adult:400:25:3")
    assert (inst.n, inst.m) == (400, 25)
    assert inst.meta["pool"] == "adult"
    assert inst.meta["sample_seed"] == 3
    assert inst.d.shape == (400, 25)


def test_sampled_facilities_never_coincide_with_clients():
    """A facility drawn onto a client gives that client a nearest-facility distance of
    zero, hence a zero radius under the distance-based schemes and an infinite radius
    ratio however it is served.  The draw excludes the sampled client rows."""
    from scipy.spatial.distance import cdist

    pool = load_adult_pool(CACHE)
    for seed in (0, 1, 2):
        clients, facs = sample_instance(pool, 1500, 300, seed=seed,
                                        facility_mode="sample_pool")
        C = clients[[c for c in clients.columns if c.startswith("d")]].to_numpy()
        assert cdist(C, facs.to_numpy()).min() > 0


def test_sample_pool_rejects_an_impossible_request():
    pool = load_bank_pool(CACHE, full=False)
    with pytest.raises(ValueError, match="rows left after exclusions"):
        sample_instance(pool, len(pool) - 5, 100, facility_mode="sample_pool")
