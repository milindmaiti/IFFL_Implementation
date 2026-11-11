#!/usr/bin/env python3
"""Sweep both IFFL algorithms over the datasets and write a results CSV.

Each row is one (dataset, radius scheme, cost scale, algorithm) result, carrying the LP
lower bound, the realised cost and ratio, and the worst radius ratio observed.

The two guarantees are checked differently, because the papers state them differently:

* Approach A's ``d(j,S) <= 3 R_j`` is deterministic, so it is asserted on *every* trial,
  while its ``1.725`` cost bound holds only in expectation and is therefore compared
  against the mean over trials.
* Approach B's bound is proved for uniform radii only, so it is asserted
  there and merely *measured* on nonuniform radii -- where it is expected to fail, and
  by how much is one of the more interesting things the sweep reports.

Usage:
    python scripts/run_experiments.py --datasets upstream:adult:1000:200:0 --verbose
    python scripts/run_experiments.py --datasets upstream:adult:1000:200:0 \
        --facility-mode sample_pool --cost-mode inverse_attractiveness \
        --cost-scales 4 7 15 --trials 50 --strict \
        --dump-radius-dist results/radius_dist.npz \
        --out results/results_upstream.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iffl.configs import (  # noqa: E402
    COST_SCALES,
    DATASETS,
    RADIUS_GRID,
    Config,
    has_native_costs,
)
from iffl.pipeline import base_instance, run_config  # noqa: E402


def build_configs(args) -> list[Config]:
    out: list[Config] = []
    for dataset in args.datasets:
        if dataset.startswith("oddcycle-"):
            grid = [("odd_cycle_unit", 1.0)]
        elif args.radius:
            grid = [(args.radius, p) for p in (args.params or [2.0])]
        else:
            grid = list(RADIUS_GRID)
        scales = args.cost_scales if args.cost_scales is not None else list(COST_SCALES)
        for (scheme, param), scale in itertools.product(grid, scales):
            if scale is None and not has_native_costs(dataset):
                continue
            out.append(Config(dataset, scheme, param, scale, cost_mode=args.cost_mode))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="sweep every dataset")
    ap.add_argument("--radius", default=None, help="restrict to one radius scheme")
    ap.add_argument("--params", nargs="*", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None, help="alias for a single --params value")
    ap.add_argument("--cost-scales", nargs="*", type=float, default=None)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--facility-mode", default="kmeans_pool",
                    help="how upstream instances place candidate facilities")
    ap.add_argument("--cost-mode", default="constant",
                    help="opening-cost scheme (see iffl.costs.MODES)")
    ap.add_argument("--dump-radius-dist", default=None, type=Path,
                    help="write per-client radius-ratio quantiles to this .npz")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="abort on a guarantee failure instead of recording it in the row",
    )
    args = ap.parse_args(argv)

    if args.alpha is not None and args.params is None:
        args.params = [args.alpha]
    if args.datasets is None:
        args.datasets = list(DATASETS) if args.all else ["synth-225-25"]

    configs = build_configs(args)
    bases = {d: base_instance(d, args.data_dir, facility_mode=args.facility_mode)
             for d in {c.dataset for c in configs}}

    dists: dict = {} if args.dump_radius_dist else None
    rows: list[dict] = []
    t_start = time.perf_counter()
    for k, cfg in enumerate(configs, 1):
        if args.verbose:
            print(f"[{k}/{len(configs)}] {cfg.label}", flush=True)
        rows += run_config(
            cfg,
            trials=args.trials,
            seed=args.seed,
            data_dir=args.data_dir,
            base=bases[cfg.dataset],
            strict=args.strict,
            dist_sink=dists,
        )
    elapsed = time.perf_counter() - t_start

    if dists is not None:
        args.dump_radius_dist.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dump_radius_dist,
            **{f"{label}||{algo}": q for (label, algo), q in dists.items()},
        )
        print(f"wrote {len(dists)} radius distributions to {args.dump_radius_dist}")

    df = pd.DataFrame(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nwrote {len(df)} rows to {args.out}  ({elapsed:.1f}s)")

    summarize(df)
    return 0


def summarize(df: pd.DataFrame) -> None:
    """Print the checks the plan's success criteria are read off."""
    ok = df[df["error"] == ""]
    print(f"\n{'=' * 78}\nSUMMARY  ({len(ok)} successful rows, {len(df) - len(ok)} errored)\n{'=' * 78}")

    print("\nCost ratio vs LP bound, by algorithm:")
    g = ok.groupby("algo")["cost_ratio"].agg(["count", "mean", "max"])
    print(g.to_string())

    a = ok[ok["algo"] == "lp_rounding"]
    if len(a):
        worst = a["max_radius_ratio"].max()
        bad = a[a["max_radius_ratio"] > 3.0 + 1e-6]
        print(f"\n[A] lp_rounding   worst radius ratio = {worst:.4f}  "
              f"(bound 3, deterministic)  violations: {len(bad)}")
        over = a[a["cost_ratio"] > 1.725 + 1e-6]
        print(f"    mean cost ratio over 1.725 in {len(over)}/{len(a)} configs "
              f"(bound is in expectation)")
        frac = a[~a["lp_integral"]]
        print(f"    fractional-LP configs: {len(frac)}/{len(a)}"
              + (f"   mean ratio there = {frac['cost_ratio'].mean():.4f}" if len(frac) else ""))

    b = ok[ok["algo"] == "primal_dual"]
    if len(b):
        bu = b[b["radius_uniform"] == True]  # noqa: E712
        bn = b[b["radius_uniform"] == False]  # noqa: E712
        if len(bu):
            print(f"\n[B] primal_dual   UNIFORM radii: worst ratio = "
                  f"{bu['max_radius_ratio'].max():.4f} (bound 3)  "
                  f"violations: {len(bu[bu['max_radius_ratio'] > 3.0 + 1e-6])}")
            if "structural_ok" in bu:
                print(f"    structural bound C+3F <= 3*sum(alpha) holds: "
                      f"{int(bu['structural_ok'].sum())}/{len(bu)}")
        if len(bn):
            import numpy as np

            viol = bn[bn["max_radius_ratio"] > 3.0 + 1e-6]
            finite = viol[np.isfinite(viol["max_radius_ratio"])]
            infinite = viol[~np.isfinite(viol["max_radius_ratio"])]
            worst = finite["max_radius_ratio"].max() if len(finite) else float("nan")
            print(f"    NONUNIFORM radii: exceeds 3 in {len(viol)}/{len(bn)} configs "
                  f"(expected -- B's bound needs uniform radii); worst finite ratio = {worst:.4f}")
            if len(infinite):
                print(f"    {len(infinite)} of those are infinite: a client sits exactly on a "
                      f"facility site, so R_j = 0 and any positive distance violates it")

    c = ok[ok["algo"] == "greedy"]
    if len(c):
        strict_viol = c[c["max_radius_ratio"] > 1.0 + 1e-6]
        print(f"\n[C] greedy        strictly fair: worst radius ratio = "
              f"{c['max_radius_ratio'].max():.4f} (bound 1)  violations: {len(strict_viol)}")
        print(f"    cost ratio: mean {c['cost_ratio'].mean():.4f}, max {c['cost_ratio'].max():.4f}"
              f"   over 1.861 in {len(c[c['cost_ratio'] > 1.861])}/{len(c)} configs")
        print("    (1.861 is the unrestricted-UFL dual-fitting bound; whether it "
              "transfers to\n     the radius-restricted version is not proved here, "
              "only measured)")

    if (df["error"] != "").any():
        print("\nErrors:")
        print(df[df["error"] != ""].groupby(["algo", "error"]).size().to_string())


if __name__ == "__main__":
    raise SystemExit(main())
