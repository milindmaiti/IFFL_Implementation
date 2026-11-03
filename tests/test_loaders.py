"""Loaders must reproduce the reference Julia pipeline's numbers exactly."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from iffl.loaders import ALL_DATASETS, default_data_dir, load_dataset

EXPECTED_SHAPES = {
    "adult-500-20": (500, 20, 6),
    "adult-429-65": (429, 65, 6),
    "bank-500-20": (500, 20, 3),
    "synth-225-25": (250, 10, 2),
    "synth-500-40": (550, 20, 2),
    "synth-500-50": (550, 20, 2),
    "uflp-49": (49, 49, 2),
    "uflp-88": (88, 88, 2),
}


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_shapes(name):
    inst = load_dataset(name)
    n, m, dim = EXPECTED_SHAPES[name]
    assert (inst.n, inst.m, inst.client_pts.shape[1]) == (n, m, dim)
    assert inst.d.shape == (n, m)
    assert inst.f.shape == (m,)


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_distances_finite_and_nonnegative(name):
    inst = load_dataset(name)
    assert np.all(np.isfinite(inst.d))
    assert np.all(inst.d >= 0)


def test_space_delimited_files_parse():
    """adult-500-20 and bank-500-20 are space-delimited, unlike the rest."""
    for name, dim in (("adult-500-20", 6), ("bank-500-20", 3)):
        inst = load_dataset(name)
        assert inst.client_pts.shape[1] == dim
        # A single-column parse would collapse the features into one object column.
        assert np.issubdtype(inst.client_pts.dtype, np.floating)


def test_l2_matches_julia_formula_on_known_row():
    """Spot-check d[0,0] against the explicit loop in Fair-FL/fl-lp-pd-new.jl's l2."""
    inst = load_dataset("adult-500-20")
    root = default_data_dir()
    clients = pd.read_csv(root / "adult-500-20.csv", sep=r"\s+", engine="python")
    facs = pd.read_csv(root / "adult-500-20-fac.csv", sep=r"\s+", engine="python")
    cols = [f"d{i}" for i in range(1, 7)]
    jpos = clients.loc[0, cols].to_numpy(dtype=float)
    ipos = facs.loc[0, cols].to_numpy(dtype=float)
    expected = math.sqrt(sum((a - b) ** 2 for a, b in zip(jpos, ipos)))
    assert inst.d[0, 0] == pytest.approx(expected, rel=1e-12)


def test_uflp_costs_are_fixed_cost_over_10000():
    """process49 in the Julia code divides the 'fixed cost' column by 10000."""
    inst = load_dataset("uflp-49")
    # Sacramento is the first row with fixed cost 115800.
    assert inst.f[0] == pytest.approx(115800 / 10000.0)
    assert inst.meta["clients_eq_facilities"] is True
    # Clients and facilities are the same point set, so the matrix is symmetric.
    assert np.allclose(inst.d, inst.d.T)
    assert np.allclose(np.diag(inst.d), 0.0)


def test_synth_costs_present_and_adult_bank_absent():
    for name in ("synth-225-25", "synth-500-40", "synth-500-50"):
        assert load_dataset(name).meta["has_native_costs"] is True
    for name in ("adult-500-20", "adult-429-65", "bank-500-20"):
        inst = load_dataset(name)
        assert inst.meta["has_native_costs"] is False
        assert np.all(inst.f == 0)


def test_synth_keeps_out_labelled_clients():
    """The 'out' rows are clients with a group tag, as processSynth treats them."""
    inst = load_dataset("synth-225-25")
    labels, counts = np.unique(inst.groups["c1"], return_counts=True)
    assert dict(zip(labels.tolist(), counts.tolist())) == {"in": 225, "out": 25}
