#!/usr/bin/env python3
"""Theorem 2, empirically: strict fairness forces logarithmic cost growth.

Builds the paper's Set Cover reduction on the classical family where greedy Set Cover is
``Theta(log q)`` times optimal, and runs all three algorithms.  Because the optimum is
known by construction (``OPT_FL = M k* + 2q``), ratios here are against the *true*
strictly fair optimum rather than the LP bound.

Usage:
    python scripts/run_hardness_study.py --kmax 12 --out results/hardness.csv
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iffl.greedy import greedy_iffl  # noqa: E402
from iffl.hardness import (  # noqa: E402
    greedy_tight_set_cover,
    optimal_cost,
    set_cover_instance,
)
from iffl.lp import solve_restricted_lp  # noqa: E402
from iffl.lp_rounding import lp_rounding_iffl  # noqa: E402
from iffl.primal_dual import primal_dual_iffl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="results/hardness.csv", type=Path)
    args = ap.parse_args(argv)

    rows: list[dict] = []
    for k in range(args.kmin, args.kmax + 1):
        q, sets, kopt = greedy_tight_set_cover(k)
        inst = set_cover_instance(q, sets)
        OPT = optimal_cost(q, kopt)
        t0 = time.perf_counter()
        lp = solve_restricted_lp(inst)
        lp_s = time.perf_counter() - t0

        for algo, fn in (
            ("lp_rounding", lambda: lp_rounding_iffl(inst, trials=args.trials, seed=0, lp=lp)),
            ("primal_dual", lambda: primal_dual_iffl(inst, lp=lp)),
            ("greedy", lambda: greedy_iffl(inst, lp=lp)),
        ):
            t0 = time.perf_counter()
            r = fn()
            rows.append({
                "k": k, "q": q, "n": inst.n, "m": inst.m, "log2_n": math.log2(inst.n),
                "opt_fl": OPT, "lp_obj": lp.obj, "k_opt": kopt,
                "algo": algo,
                "cost": r.get("cost", r.get("cost_mean")),
                "cost_over_opt": r.get("cost", r.get("cost_mean")) / OPT,
                "max_radius_ratio": r["max_radius_ratio"],
                "n_open": r["n_open"] if "n_open" in r else r["n_open_mean"],
                "runtime_s": time.perf_counter() - t0 + lp_s,
            })
        print(f"k={k:2d}  N={inst.n:6d}  " + "  ".join(
            f"{x['algo']}={x['cost_over_opt']:.3f}" for x in rows[-3:]), flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows to {args.out}")

    g = df[df.algo == "greedy"]
    A = np.vstack([g.log2_n, np.ones(len(g))]).T
    slope, icpt = np.linalg.lstsq(A, g.cost_over_opt.to_numpy(), rcond=None)[0]
    pred = A @ [slope, icpt]
    r2 = 1 - ((g.cost_over_opt - pred) ** 2).sum() / ((g.cost_over_opt - g.cost_over_opt.mean()) ** 2).sum()
    print(f"\ngreedy (strictly fair): cost/OPT = {slope:.4f}*log2(N) {icpt:+.4f}   R^2 = {r2:.5f}")
    print(f"  theory: greedy opens k sets against the optimum's {int(g.k_opt.iloc[0])},")
    print("  so the ratio tends to k/2 = 0.5*log2(N) - 1")
    print("\nbicriteria algorithms over the same family:")
    print(df[df.algo != "greedy"].groupby("algo").agg(
        cost_over_opt_max=("cost_over_opt", "max"),
        max_radius_ratio=("max_radius_ratio", "max")).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
