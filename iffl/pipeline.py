"""Build instances from a config and run every algorithm on them."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from iffl import costs as costs_mod
from iffl import radii as radii_mod
from iffl.configs import Config, radius_rank_for
from iffl.evaluate import greedy_nearest_feasible, threshold_round
from iffl.instance import Instance
from iffl.loaders import load_dataset
from iffl.lp import solve_restricted_lp
from iffl.lp_rounding import evaluate_open_set, lp_rounding_iffl
from iffl.primal_dual import primal_dual_iffl
from iffl.synthetic import odd_cycle_instance


def base_instance(dataset: str, data_dir: Path | str | None = None) -> Instance:
    """Load a dataset by name, including the synthetic odd-cycle instances."""
    if dataset.startswith("oddcycle-"):
        return odd_cycle_instance(int(dataset.split("-")[1]), facility_cost=10.0)
    return load_dataset(dataset, data_dir)


def build_instance(
    cfg: Config, data_dir: Path | str | None = None, base: Instance | None = None
) -> Instance:
    """Apply the config's radius scheme and cost scale to a base dataset."""
    inst = base if base is not None else base_instance(cfg.dataset, data_dir)
    if inst.R is not None and inst.meta.get("source") == "synthetic":
        # Odd-cycle instances carry their own unit radii; only costs are swept.  Their
        # integrality gap widens with the opening cost, so the scale is applied directly.
        scale = 1.0 if cfg.native_costs else cfg.cost_scale
        return inst.with_costs(
            np.full(inst.m, scale * 10.0),
            cost_mode="native" if cfg.native_costs else "synthetic",
            cost_scale=cfg.cost_scale,
        )
    inst = radii_mod.apply_scheme(
        inst, cfg.scheme, cfg.param, rank=radius_rank_for(cfg.dataset)
    )
    return costs_mod.ensure_costs(
        inst,
        scale=1.0 if cfg.native_costs else cfg.cost_scale,
        mode=cfg.cost_mode,
        force=not cfg.native_costs,
    )


def _row(cfg: Config, inst: Instance, lp, algo: str, **extra) -> dict:
    return {
        "dataset": cfg.dataset,
        "n": inst.n,
        "m": inst.m,
        "radius_scheme": inst.meta.get("radius_scheme", cfg.scheme),
        "radius_param": inst.meta.get("radius_param", cfg.param),
        "radius_rank": inst.meta.get("radius_rank", 1),
        "radius_uniform": inst.meta.get("radius_uniform"),
        "radius_degenerate": inst.meta.get("radius_degenerate", False),
        "radius_clamped": inst.meta.get("radius_clamped", 0),
        "radius_n_zero": inst.meta.get("radius_n_zero", 0),
        "R_min": float(inst.R.min()),
        "R_max": float(inst.R.max()),
        "cost_mode": inst.meta.get("cost_mode", cfg.cost_mode),
        "cost_scale": cfg.cost_scale,
        "n_edges": lp.n_edges,
        "lp_obj": lp.obj,
        "lp_integral": bool(np.all((lp.y < 1e-6) | (lp.y > 1 - 1e-6))),
        "algo": algo,
        **extra,
    }


def run_config(
    cfg: Config,
    trials: int = 50,
    seed: int = 0,
    data_dir: Path | str | None = None,
    base: Instance | None = None,
    strict: bool = True,
) -> list[dict]:
    """Run both algorithms plus both baselines on one config.

    Args:
        strict: propagate guarantee assertions.  With ``strict=False`` a failure is
            recorded in the row instead, so one bad config cannot abort a long sweep.

    Returns:
        One row per algorithm.
    """
    inst = build_instance(cfg, data_dir, base)
    t0 = time.perf_counter()
    lp = solve_restricted_lp(inst)
    lp_time = time.perf_counter() - t0
    rows: list[dict] = []

    uniform = bool(inst.meta.get("radius_uniform"))

    # --- Approach A: (1.725, 3) LP rounding. ---
    try:
        t0 = time.perf_counter()
        a = lp_rounding_iffl(inst, trials=trials, seed=seed, lp=lp, check_fairness=strict)
        rows.append(_row(cfg, inst, lp, "lp_rounding",
                         cost=a["cost_mean"],
                         cost_ratio=a["cost_ratio_mean"],
                         cost_ratio_max=a["cost_ratio_max"],
                         max_radius_ratio=a["max_radius_ratio"],
                         mean_radius_ratio=a["mean_radius_ratio"],
                         n_open=a["n_open_mean"],
                         trials=trials,
                         n_copies=a["diagnostics"]["n_copies"],
                         n_centers=a["n_centers"],
                         runtime_s=time.perf_counter() - t0 + lp_time,
                         seed=seed, error=""))
    except (AssertionError, ValueError, RuntimeError) as exc:
        if strict:
            raise
        rows.append(_row(cfg, inst, lp, "lp_rounding", runtime_s=lp_time, seed=seed,
                         error=f"{type(exc).__name__}: {exc}"))

    # --- Approach B: (3, 3) primal-dual.  Its bound is proved for uniform radii only,
    # so on nonuniform radii the ratio is measured and reported, never asserted. ---
    try:
        t0 = time.perf_counter()
        b = primal_dual_iffl(inst, lp=lp, check_fairness=strict and uniform)
        rows.append(_row(cfg, inst, lp, "primal_dual",
                         cost=b["cost"],
                         cost_ratio=b["cost_ratio"],
                         cost_ratio_max=b["cost_ratio"],
                         max_radius_ratio=b["max_radius_ratio"],
                         mean_radius_ratio=b["mean_radius_ratio"],
                         n_open=b["n_open"],
                         trials=1,
                         alpha_sum=b["alpha_sum"],
                         structural_lhs=b["structural_lhs"],
                         structural_rhs=b["structural_rhs"],
                         structural_ok=b["structural_ok"],
                         n_fallback=b["diagnostics"]["n_fallback"],
                         n_direct=b["diagnostics"]["n_direct"],
                         n_indirect=b["diagnostics"]["n_via_cause"]
                         + b["diagnostics"]["n_via_witness"],
                         runtime_s=time.perf_counter() - t0 + lp_time,
                         seed=seed, error=""))
    except (AssertionError, ValueError, RuntimeError) as exc:
        if strict:
            raise
        rows.append(_row(cfg, inst, lp, "primal_dual", runtime_s=lp_time, seed=seed,
                         error=f"{type(exc).__name__}: {exc}"))

    # --- Baselines. ---
    for algo, fn in (
        ("baseline_threshold", lambda: threshold_round(inst, lp)),
        ("baseline_greedy", lambda: greedy_nearest_feasible(inst)),
    ):
        try:
            t0 = time.perf_counter()
            opened = fn()
            sol = evaluate_open_set(inst, opened)
            rows.append(_row(cfg, inst, lp, algo,
                             cost=sol.cost,
                             cost_ratio=sol.cost / lp.obj if lp.obj > 0 else float("nan"),
                             cost_ratio_max=sol.cost / lp.obj if lp.obj > 0 else float("nan"),
                             max_radius_ratio=sol.max_radius_ratio,
                             mean_radius_ratio=float(sol.radius_ratio.mean()),
                             n_open=int(opened.size),
                             trials=1,
                             runtime_s=time.perf_counter() - t0 + lp_time,
                             seed=seed, error=""))
        except (ValueError, RuntimeError) as exc:
            rows.append(_row(cfg, inst, lp, algo, runtime_s=lp_time, seed=seed,
                             error=f"{type(exc).__name__}: {exc}"))

    return rows
