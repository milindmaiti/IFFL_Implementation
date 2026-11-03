#!/usr/bin/env python3
"""Figures summarizing the IFFL sweep.

Four panels, each also written as a CSV table view beside the image:

1. Mean cost ratio against the LP bound, by algorithm.
2. Approach B's radius ratio under uniform vs nonuniform radii -- why its bound needs
   uniform radii, and the main reason Approach A exists.
3. Approach A's cost ratio on the fractional-LP configs, against its 1.725 bound.
4. Cost ratio against the opening-cost scale, for the two paper algorithms.

Usage:
    python scripts/plot_results.py --results results/results.csv --out results/figures
    python scripts/plot_results.py --dark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Categorical slots 1-4 of the reference palette, in its documented order.  Validated
# on the adjacent pairlist, which is the one that applies to bars: worst adjacent CVD
# dE 9.1 light / 8.4 dark.  Two of the light steps fall below 3:1 against the surface,
# so the relief rule applies -- every bar carries a direct value label and each figure
# ships a CSV table view.
LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#dcdbd6",
    "rule": "#8c8b85",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
}
DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#3a3a37",
    "rule": "#8c8b85",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}

ALGO_LABEL = {
    "lp_rounding": "A: LP rounding",
    "primal_dual": "B: primal-dual",
    "greedy": "C: greedy (strictly fair)",
    "baseline_threshold": "Baseline: threshold",
    "baseline_greedy": "Baseline: greedy fair",
}
ALGO_ORDER = ["lp_rounding", "primal_dual", "greedy",
              "baseline_threshold", "baseline_greedy"]


def style(pal: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": pal["surface"],
        "axes.facecolor": pal["surface"],
        "savefig.facecolor": pal["surface"],
        "text.color": pal["text"],
        "axes.labelcolor": pal["muted"],
        "xtick.color": pal["muted"],
        "ytick.color": pal["muted"],
        "axes.edgecolor": pal["grid"],
        "grid.color": pal["grid"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _finish(ax, pal, title, subtitle=None):
    ax.set_title(title, color=pal["text"], loc="left", pad=16 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=pal["muted"], fontsize=9)
    ax.grid(axis="x", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_cost_ratio_by_algo(df, pal, out: Path):
    """Horizontal bars: mean cost ratio per algorithm, with the two bounds marked."""
    g = (df.groupby("algo")["cost_ratio"]
           .agg(["mean", "median", "max", "count"])
           .reindex(ALGO_ORDER).dropna())
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ypos = np.arange(len(g))[::-1]
    for k, (algo, row) in enumerate(g.iterrows()):
        ax.barh(ypos[k], row["mean"], height=0.6, color=pal["series"][k],
                edgecolor=pal["surface"], linewidth=2)
        # Relief rule: a visible value label on every bar, not colour alone.
        ax.text(row["mean"] + 0.03, ypos[k], f"{row['mean']:.3f}",
                va="center", color=pal["text"], fontsize=10)
    ax.axvline(1.0, color=pal["rule"], lw=1.2, ls="-")
    ax.axvline(1.725, color=pal["rule"], lw=1.2, ls="--")
    ax.text(1.0, -0.85, "LP bound", color=pal["muted"], fontsize=9, ha="center")
    ax.text(1.725, -0.85, "A's bound 1.725", color=pal["muted"], fontsize=9, ha="center")
    ax.set_ylim(-1.2, len(g) - 0.3)
    ax.set_yticks(ypos, [ALGO_LABEL[a] for a in g.index], color=pal["text"])
    ax.set_xlabel("mean cost / LP lower bound")
    ax.set_xlim(0, max(2.0, g["mean"].max() * 1.25))
    _finish(ax, pal, "Cost against the LP lower bound",
            f"mean over {int(g['count'].iloc[0])} configurations per algorithm")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    g.to_csv(out.with_suffix(".csv"))


def fig_radius_uniform_vs_not(df, pal, out: Path):
    """ECDF of Approach B's worst radius ratio, split by whether radii are uniform."""
    b = df[df["algo"] == "primal_dual"]
    series = [
        ("uniform radii (bound proved)", b[b["radius_uniform"] == True], pal["series"][0]),  # noqa: E712
        ("nonuniform radii (no bound)", b[b["radius_uniform"] == False], pal["series"][1]),  # noqa: E712
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    cap = 12.0
    rows = []
    for label, sub, color in series:
        v = np.sort(sub["max_radius_ratio"].to_numpy(dtype=float))
        n_inf = int(np.isinf(v).sum())
        v = np.clip(v, 0, cap)
        y = np.arange(1, v.size + 1) / v.size
        ax.step(np.concatenate([[0], v]), np.concatenate([[0], y]), where="post",
                lw=2, color=color, label=label)
        rows.append({"series": label, "n": v.size,
                     "frac_within_3": float(np.mean(v <= 3.0 + 1e-9)),
                     "median": float(np.median(v)), "n_infinite": n_inf})
    ax.axvline(3.0, color=pal["rule"], lw=1.2, ls="--")
    ax.text(3.18, 0.32, "3R bound", color=pal["muted"], fontsize=9, ha="left")
    ax.set_xlim(0, cap)
    ax.set_xlabel(
        f"worst d(j, sigma(j)) / R_j   "
        f"(tail clipped at {cap:.0f}; infinite values are zero-radius clients)"
    )
    ax.set_ylabel("fraction of configurations")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", labelcolor=pal["text"],
              bbox_to_anchor=(1.0, 0.02))
    _finish(ax, pal, "Approach B keeps its 3R promise only for uniform radii",
            "with heterogeneous radii the witness's R can dwarf the client's, so no bound survives")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


def fig_fractional_configs(df, pal, out: Path):
    """Approach A's cost ratio where the LP is genuinely fractional."""
    a = df[(df["algo"] == "lp_rounding") & (~df["lp_integral"])]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    if len(a):
        ax.hist(a["cost_ratio"], bins=24, color=pal["series"][0],
                edgecolor=pal["surface"], linewidth=1.5)
    ax.axvline(1.725, color=pal["rule"], lw=1.2, ls="--")
    ax.set_xlim(right=1.80)
    ax.text(1.725, ax.get_ylim()[1] * 0.96, "bound 1.725", color=pal["muted"],
            fontsize=9, ha="right", va="top")
    ax.set_xlabel("mean cost / LP lower bound")
    ax.set_ylabel("configurations")
    _finish(ax, pal, "Approach A on fractional-LP instances",
            f"{len(a)} configurations where the LP optimum is not integral; "
            f"max observed {a['cost_ratio'].max():.3f}" if len(a) else "none")
    ax.grid(axis="y", lw=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    a[["dataset", "radius_scheme", "radius_param", "cost_scale",
       "cost_ratio", "max_radius_ratio"]].to_csv(out.with_suffix(".csv"), index=False)


def fig_cost_scale(df, pal, out: Path):
    """How the cost ratio moves with the opening-cost scale, for the three algorithms."""
    algos = ["lp_rounding", "primal_dual", "greedy"]
    sub = df[df["algo"].isin(algos) & df["cost_scale"].notna()]
    piv = sub.pivot_table(index="cost_scale", columns="algo",
                          values="cost_ratio", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(piv))
    width = 0.27
    for k, algo in enumerate(algos):
        if algo not in piv:
            continue
        ax.bar(x + (k - 1) * width, piv[algo], width * 0.9,
               color=pal["series"][k], edgecolor=pal["surface"], linewidth=2,
               label=ALGO_LABEL[algo])
        for xi, v in zip(x + (k - 1) * width, piv[algo]):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", color=pal["text"], fontsize=7)
    ax.axhline(1.0, color=pal["rule"], lw=1.2)
    ax.set_xticks(x, [f"{c:g}" for c in piv.index])
    ax.set_xlabel("opening-cost scale (multiples of mean nearest-facility distance)")
    ax.set_ylabel("mean cost / LP bound")
    ax.set_ylim(0.9, max(1.2, float(np.nanmax(piv.to_numpy())) * 1.08))
    ax.legend(frameon=False, labelcolor=pal["text"])
    _finish(ax, pal, "The greedy stays closest to the LP bound as opening costs rise",
            "all three track it closely; B drifts most at high cost scales")
    ax.grid(axis="y", lw=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    piv.to_csv(out.with_suffix(".csv"))


def fig_tradeoff(df, pal, out: Path):
    """What each algorithm buys and what it pays: fairness on x, cost on y.

    Restricted to **uniform radii**, the only regime where all three are comparable:
    Approach B has no proved bound otherwise, and its worst nonuniform ratio of 44 would
    push the other two off the axis without saying anything meaningful.  The nonuniform
    behaviour has its own figure.

    One point per algorithm, so the three sit within the all-pairs colour cap.  The axes
    are two different quantities, not two scales of one, so this is a plain scatter and
    not a dual-axis chart.
    """
    algos = ["lp_rounding", "primal_dual", "greedy"]
    uni = df[df["radius_uniform"] == True]  # noqa: E712
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    rows = []
    for k, algo in enumerate(algos):
        a = uni[uni["algo"] == algo]
        if not len(a):
            continue
        finite = a[np.isfinite(a["max_radius_ratio"])]
        x = float(finite["max_radius_ratio"].max())
        y = float(a["cost_ratio"].mean())
        ax.scatter([x], [y], s=190, color=pal["series"][k], zorder=3,
                   edgecolor=pal["surface"], linewidth=2, label=ALGO_LABEL[algo])
        rows.append({"algo": algo, "worst_radius_ratio": x, "mean_cost_ratio": y,
                     "n_configs": len(a)})
    # Offsets keep the two labels at x = 3 from landing on top of each other.
    offsets = {"lp_rounding": (0, 18), "primal_dual": (0, -26), "greedy": (0, 18)}
    for k, algo in enumerate(algos):
        r = next((r for r in rows if r["algo"] == algo), None)
        if r is None:
            continue
        ax.annotate(ALGO_LABEL[algo], (r["worst_radius_ratio"], r["mean_cost_ratio"]),
                    textcoords="offset points", xytext=offsets[algo], ha="center",
                    color=pal["text"], fontsize=9)
    ax.axvline(1.0, color=pal["rule"], lw=1.0, ls=":")
    ax.axvline(3.0, color=pal["rule"], lw=1.2, ls="--")
    lo = min(r["mean_cost_ratio"] for r in rows) if rows else 1.0
    hi = max(r["mean_cost_ratio"] for r in rows) if rows else 1.1
    ax.set_ylim(1.0, hi + (hi - lo) * 0.9 + 0.004)
    ax.text(1.04, 1.001, "strictly fair", color=pal["muted"], fontsize=9, va="bottom")
    ax.text(2.96, 1.001, "3R bicriteria bound", color=pal["muted"], fontsize=9,
            va="bottom", ha="right")
    ax.set_xlabel("worst d(j, sigma(j)) / R_j observed  (lower is fairer)")
    ax.set_ylabel("mean cost / LP bound  (lower is cheaper)")
    ax.set_xlim(0.6, 3.5)
    _finish(ax, pal, "The strictly fair greedy is both fairer and cheaper here",
            f"uniform-radius configurations only ({len(uni[uni.algo == 'greedy'])} of them); "
            "y starts at the LP bound")
    ax.grid(axis="y", lw=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/results.csv", type=Path)
    ap.add_argument("--out", default="results/figures", type=Path)
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.results)
    df = df[df["error"].isna() | (df["error"] == "")]
    pal = DARK if args.dark else LIGHT
    style(pal)
    args.out.mkdir(parents=True, exist_ok=True)
    sfx = "-dark" if args.dark else ""

    fig_cost_ratio_by_algo(df, pal, args.out / f"cost-ratio-by-algorithm{sfx}")
    fig_radius_uniform_vs_not(df, pal, args.out / f"radius-uniform-vs-nonuniform{sfx}")
    fig_fractional_configs(df, pal, args.out / f"approach-a-fractional{sfx}")
    fig_cost_scale(df, pal, args.out / f"cost-ratio-by-cost-scale{sfx}")
    fig_tradeoff(df, pal, args.out / f"fairness-cost-tradeoff{sfx}")
    print(f"wrote 5 figures (+ CSV table views) to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
