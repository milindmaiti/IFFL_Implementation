"""Build instances from a config and run every algorithm on them."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from iffl import costs as costs_mod
from iffl import radii as radii_mod
from iffl.configs import Config, radius_rank_for
from iffl.evaluate import greedy_nearest_feasible, threshold_round
from iffl.greedy import greedy_iffl
from iffl.instance import Instance, l2_matrix
from iffl.loaders import load_dataset
from iffl.lp import solve_restricted_lp
from iffl.lp_rounding import evaluate_open_set, lp_rounding_iffl
from iffl.primal_dual import primal_dual_iffl
from iffl.synthetic import odd_cycle_instance
from iffl.upstream import load_adult_pool, load_bank_pool, sample_instance


_POOL_CACHE: dict[str, object] = {}


def upstream_instance(
    spec: str,
    cache_dir: Path | str = ".cache",
    facility_mode: str = "kmeans_pool",
) -> Instance:
    """Build an instance sampled from a full upstream dataset.

    ``spec`` is ``"upstream:<pool>:<n>:<m>[:<seed>]"``, e.g. ``upstream:adult:2000:100``.
    Pools are ``adult`` (45,222 rows) and ``bank`` (45,211 rows from ``bank-full.csv``).
    The preprocessing matches the bundled instances exactly, so sizes are comparable;
    see :mod:`iffl.upstream`.
    """
    parts = spec.split(":")
    if len(parts) not in (4, 5) or parts[0] != "upstream":
        raise ValueError(f"bad upstream spec {spec!r}")
    pool_name, n, m = parts[1], int(parts[2]), int(parts[3])
    seed = int(parts[4]) if len(parts) == 5 else 0

    if pool_name not in _POOL_CACHE:
        loader = {"adult": load_adult_pool, "bank": load_bank_pool}[pool_name]
        _POOL_CACHE[pool_name] = loader(Path(cache_dir))
    pool = _POOL_CACHE[pool_name]

    clients, facs = sample_instance(pool, n, m, seed=seed, facility_mode=facility_mode)
    dims = [c for c in clients.columns if c.startswith("d")]
    C = clients[dims].to_numpy(dtype=float)
    F = facs[dims].to_numpy(dtype=float)
    groups = {c: clients[c].to_numpy() for c in clients.columns if c.startswith("c")}
    return Instance(
        name=spec,
        client_pts=C,
        facility_pts=F,
        f=np.zeros(m),
        d=l2_matrix(C, F),
        groups=groups,
        meta={"source": "upstream", "has_native_costs": False,
              "pool": pool.name, "pool_size": len(pool), "sample_seed": seed,
              "facility_mode": facility_mode},
    )


def base_instance(
    dataset: str,
    data_dir: Path | str | None = None,
    facility_mode: str = "kmeans_pool",
) -> Instance:
    """Load a dataset by name, including synthetic and upstream-sampled instances."""
    if dataset.startswith("oddcycle-"):
        return odd_cycle_instance(int(dataset.split("-")[1]), facility_cost=10.0)
    if dataset.startswith("upstream:"):
        return upstream_instance(dataset, facility_mode=facility_mode)
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


#: Quantile levels stored per (configuration, algorithm) when a radius distribution is
#: requested.  The quantile function is the inverse of the empirical CDF, so this grid
#: reconstructs the ECDF to 0.1% and costs 1001 floats instead of one per client.
DIST_LEVELS = np.linspace(0.0, 1.0, 1001)


def run_config(
    cfg: Config,
    trials: int = 50,
    seed: int = 0,
    data_dir: Path | str | None = None,
    base: Instance | None = None,
    strict: bool = True,
    dist_sink: dict | None = None,
) -> list[dict]:
    """Run both algorithms plus both baselines on one config.

    Args:
        strict: propagate guarantee assertions.  With ``strict=False`` a failure is
            recorded in the row instead, so one bad config cannot abort a long sweep.
        dist_sink: if given, receives ``{(config label, algo): quantiles}`` of the
            per-client radius ratio.  The CSV keeps only the mean and max, which cannot
            show what a typical client experiences; the randomized rounding pools every
            trial, so its distribution reflects the algorithm rather than one draw.

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
        if dist_sink is not None:
            pooled = np.concatenate([s_.radius_ratio for s_ in a["solutions"]])
            dist_sink[(cfg.label, "lp_rounding")] = np.quantile(
                pooled[np.isfinite(pooled)], DIST_LEVELS)
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
        if dist_sink is not None:
            rr = b["solution"].radius_ratio
            dist_sink[(cfg.label, "primal_dual")] = np.quantile(
                rr[np.isfinite(rr)], DIST_LEVELS)
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

    # --- Approach C: greedy (Williamson & Shmoys Alg. 9.2), restricted to E_R.
    # Strictly fair by construction, so its radius ratio is asserted against 1, not 3. ---
    try:
        t0 = time.perf_counter()
        c = greedy_iffl(inst, lp=lp, check_fairness=strict)
        if dist_sink is not None:
            rr = c["solution"].radius_ratio
            dist_sink[(cfg.label, "greedy")] = np.quantile(
                rr[np.isfinite(rr)], DIST_LEVELS)
        rows.append(_row(cfg, inst, lp, "greedy",
                         cost=c["cost"],
                         cost_ratio=c["cost_ratio"],
                         cost_ratio_max=c["cost_ratio"],
                         max_radius_ratio=c["max_radius_ratio"],
                         mean_radius_ratio=c["mean_radius_ratio"],
                         n_open=c["n_open"],
                         trials=1,
                         n_rounds=c["diagnostics"]["n_rounds"],
                         n_reopened=c["diagnostics"]["n_reopened"],
                         runtime_s=time.perf_counter() - t0 + lp_time,
                         seed=seed, error=""))
    except (AssertionError, ValueError, RuntimeError) as exc:
        if strict:
            raise
        rows.append(_row(cfg, inst, lp, "greedy", runtime_s=lp_time, seed=seed,
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
