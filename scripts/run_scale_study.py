#!/usr/bin/env python3
"""Scale study: how do the cost comparisons change on larger samples?

The bundled instances are 429-550 clients drawn from surveys of ~45,000.  This sweeps
instances resampled from the full upstream pools across a grid of sizes and, more
importantly, across the **facilities-per-client ratio** `m/n` -- the quantity that
governs whether the LP relaxation is integral, and therefore whether the three
algorithms differ at all.

Usage:
    python scripts/run_scale_study.py --out results/scale.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iffl.configs import Config  # noqa: E402
from iffl.pipeline import base_instance, run_config  # noqa: E402

DEFAULT_SIZES = [(500, 20), (500, 100), (500, 250),
                 (2000, 20), (2000, 100), (2000, 400),
                 (5000, 50), (5000, 250), (5000, 1000)]
DEFAULT_RADII = [("uniform", 0.5), ("nearest_facility_multiple", 2.0)]
DEFAULT_SCALES = [0.5, 2.0, 10.0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=["adult", "bank"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="results/scale.csv", type=Path)
    args = ap.parse_args(argv)

    rows: list[dict] = []
    specs = [
        f"upstream:{pool}:{n}:{m}:{seed}"
        for pool in args.pools
        for (n, m) in DEFAULT_SIZES
        for seed in args.seeds
    ]
    t0 = time.perf_counter()
    for k, spec in enumerate(specs, 1):
        base = base_instance(spec)
        print(f"[{k}/{len(specs)}] {spec}  (m/n = {base.m / base.n:.3f})", flush=True)
        for (scheme, param), scale in itertools.product(DEFAULT_RADII, DEFAULT_SCALES):
            cfg = Config(spec, scheme, param, scale)
            got = run_config(cfg, trials=args.trials, seed=0, base=base, strict=False)
            for r in got:
                r["pool"] = base.meta["pool"]
                r["m_over_n"] = base.m / base.n
            rows += got

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows to {args.out}  ({time.perf_counter() - t0:.0f}s)")

    ok = df[df["error"] == ""]
    algos = ["lp_rounding", "primal_dual", "greedy"]
    print("\n=== fractional-LP rate by m/n ===")
    a = ok[ok.algo == "lp_rounding"]
    print(a.groupby("m_over_n").agg(
        configs=("lp_integral", "size"),
        fractional=("lp_integral", lambda s: f"{100 * (~s).mean():.0f}%"),
    ).to_string())

    print("\n=== mean cost ratio by m/n ===")
    print(ok[ok.algo.isin(algos)].pivot_table(
        index="m_over_n", columns="algo", values="cost_ratio", aggfunc="mean"
    ).round(4).to_string())

    print("\n=== on fractional-LP configs only ===")
    fr = ok[~ok.lp_integral & ok.algo.isin(algos)]
    if len(fr):
        print(fr.pivot_table(index="pool", columns="algo",
                             values="cost_ratio", aggfunc=["mean", "max"]).round(4).to_string())
    else:
        print("  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
