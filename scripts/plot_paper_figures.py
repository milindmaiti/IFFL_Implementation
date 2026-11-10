#!/usr/bin/env python3
"""The three figures for the paper, on UCI data.

1. Cost      -- cost / LP bound against radius tightness.
2. Cost       -- unnormalised cost against radius tightness, with the LP optimum, so
                 problem hardening is visible separately from algorithm error.
3. Fairness  -- worst radius violation, and the distribution over clients.

Styled for inclusion in a paper rather than for standalone reading: no titles, no
annotations, no explanatory text.  Everything a reader needs beyond the axes belongs in
the LaTeX caption and the surrounding discussion.  The only in-figure text besides axis
labels and the legend is a short tag on each reference line, without which a bare dashed
rule would be unreadable.

Palette: navy / vermillion / mint.  Chosen over the usual Okabe-Ito set because papers
get printed: its lightness values are 21 / 52 / 74, a minimum L* gap of 23 against
Okabe-Ito's 3.6, so the series stay distinguishable in greyscale.  Worst pairwise CVD
difference is dE 40 (protan/deutan/tritan simulated), and each series additionally
carries its own marker and dash pattern, so colour is never the sole encoding.

Usage:
    python scripts/plot_paper_figures.py [--format pdf]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PALETTE = dict(surface="#ffffff", text="#111111", muted="#555555", grid="#d8d8d8",
               rule="#777777", series=["#16335e", "#cf5420", "#6fc99a"])

ALGOS = ["lp_rounding", "primal_dual", "greedy"]
LABEL = {"lp_rounding": "LP rounding", "primal_dual": "primal-dual", "greedy": "greedy"}
MARKER = {"lp_rounding": "o", "primal_dual": "s", "greedy": "^"}
DASH = {"lp_rounding": (0, (1, 1.4)), "primal_dual": (0, (5, 1.8)), "greedy": "solid"}
WIDTH = {"lp_rounding": 1.9, "primal_dual": 1.4, "greedy": 1.2}


def style(pal):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.facecolor": pal["surface"], "axes.facecolor": pal["surface"],
        "savefig.facecolor": pal["surface"], "text.color": pal["text"],
        "axes.labelcolor": pal["text"], "xtick.color": pal["text"],
        "ytick.color": pal["text"], "axes.edgecolor": pal["muted"],
        "axes.linewidth": 0.7, "grid.color": pal["grid"], "grid.linewidth": 0.5,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _grid(ax, axis="y"):
    ax.grid(axis=axis, lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def uci(main: pd.DataFrame) -> pd.DataFrame:
    """Rows from Adult and Bank, bundled or resampled from the full upstream pools."""
    d = main[main.dataset.str.contains("adult") | main.dataset.str.contains("bank")]
    return d[d.algo.isin(ALGOS)]


def _series(ax, x, y, k, algo, pal):
    ax.plot(x, y, marker=MARKER[algo], ms=4.5, lw=WIDTH[algo], ls=DASH[algo],
            color=pal["series"][k], label=LABEL[algo], zorder=3,
            markeredgecolor=pal["surface"], markeredgewidth=0.7)


SCHEMES = [("nearest_facility_multiple", "nearest-facility"),
           ("jkl_client_knn", "Jung–Kannan–Lutz")]


def _eff_alpha(g):
    """Mean effective radius multiplier of a group, the scheme-independent x-coordinate."""
    return g.groupby("radius_param")["radius_eff_alpha"].mean()


def fig_cost(main, pal, out: Path, fmt: str):
    """Cost relative to the LP bound, by radius scheme, on a shared tightness axis.

    The two schemes parameterise tightness differently and in opposite directions -- a
    larger multiplier loosens the radius, a larger neighbourhood count tightens it -- so
    neither parameter can label a common axis.  Both are plotted against
    ``median_j(R_j / d(j, F))`` instead, which is the multiplier scheme's own parameter by
    definition and puts the two on one comparable scale: matched positions then differ in
    radius *heterogeneity* rather than in tightness.
    """
    d = uci(main)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5), sharey=True)
    for ax, (scheme, name) in zip(axes, SCHEMES):
        g = d[d.radius_scheme == scheme]
        x = _eff_alpha(g)
        piv = g.pivot_table(index="radius_param", columns="algo", values="cost_ratio",
                            aggfunc="mean")
        for k, a in enumerate(ALGOS):
            _series(ax, x.loc[piv.index].values, piv[a].values, k, a, pal)
        ax.axhline(1.0, color=pal["rule"], lw=0.8)
        ax.set_xlabel(r"effective multiplier $R_j / d(j,F)$")
        ax.set_title(name, color=pal["muted"], fontsize=8.5, loc="left", pad=3)
        ax.set_xlim(0.8, 4.2)
        _grid(ax)
    axes[0].set_ylabel("cost / LP bound")
    axes[0].legend(loc="lower right", handlelength=2.2, borderaxespad=0.3, labelspacing=0.25)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out.with_suffix(f".{fmt}"), dpi=400)
    plt.close(fig)
    d.pivot_table(index=["radius_scheme", "radius_param"], columns="algo",
                  values="cost_ratio", aggfunc="mean").to_csv(out.with_suffix(".csv"))


def fig_fairness(main, pal, out: Path, fmt: str, dist: dict | None = None):
    """What fraction of clients are served beyond x times their radius.

    A worst case over clients answers the question a guarantee asks but hides how rare a
    violation is, and a CDF compresses exactly the tail that matters.  The survival
    function ``P(ratio > x)`` on log axes shows both: where each curve leaves 1 is the
    typical client, and where it ends is the worst one.
    """
    if not dist:
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6), sharey=True)
    rows = []
    for ax, (scheme, name) in zip(axes, SCHEMES):
        for k, a in enumerate(ALGOS):
            vs = [v for (lbl, alg), v in dist.items() if alg == a and scheme in lbl]
            if not vs:
                continue
            v = np.sort(np.concatenate(vs))
            surv = 1.0 - np.arange(v.size) / v.size
            step = max(1, v.size // 4000)          # thin for a compact vector file
            ax.plot(v[::step], surv[::step], lw=WIDTH[a], ls=DASH[a],
                    color=pal["series"][k], label=LABEL[a])
            rows.append({"scheme": scheme, "algo": a, "n_clients": v.size,
                         "frac_over_1": float((v > 1).mean()),
                         "frac_over_3": float((v > 3).mean()), "max": float(v.max())})
        ax.axvline(1.0, color=pal["rule"], lw=0.6, ls=":")
        ax.axvline(3.0, color=pal["rule"], lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(2e-5, 1.6)
        ax.set_xlim(0.15, 300)
        ax.set_xlabel(r"$x = d(j,\sigma(j))\,/\,R_j$")
        ax.set_title(name, color=pal["muted"], fontsize=8.5, loc="left", pad=3)
        _grid(ax, axis="both")
    axes[0].set_ylabel(r"fraction of clients $> x$")
    axes[0].legend(loc="lower left", handlelength=2.2, borderaxespad=0.3, labelspacing=0.25)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out.with_suffix(f".{fmt}"), dpi=400)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


def fig_cost_absolute(main, pal, out: Path, fmt: str):
    """Unnormalised cost against radius tightness, one panel per instance.

    The ratio figure divides out the LP bound and so hides how much harder the problem
    itself becomes as radii tighten.  This plots the costs themselves, with the LP
    optimum as a fourth line: the gap between that line and the algorithms is their
    error, while its own slope is the problem hardening.  Raw costs are not comparable
    across instances of different size, hence one panel each rather than an average.
    """
    d = uci(main)
    d = d[d.radius_scheme == "nearest_facility_multiple"]
    specs = sorted(d.dataset.unique())

    fig, axes = plt.subplots(1, len(specs), figsize=(5.6, 2.5), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, spec in zip(axes, specs):
        g = d[d.dataset == spec]
        piv = g.pivot_table(index="radius_param", columns="algo", values="cost", aggfunc="mean")
        lp = g.groupby("radius_param")["lp_obj"].mean()
        ax.plot(lp.index, lp.values, lw=1.0, ls="-", color=pal["rule"], zorder=2,
                label="LP bound")
        for k, a in enumerate(ALGOS):
            _series(ax, piv.index, piv[a], k, a, pal)
        pool, n, m = spec.split(":")[1:4]
        ax.set_title(f"{pool}, $m={m}$", color=pal["text"], fontsize=8.5, loc="left", pad=3)
        ax.set_xlabel(r"radius multiplier $\alpha$")
        ax.set_xticks([1, 2, 3, 4])
        _grid(ax)
    axes[0].set_ylabel("cost")
    # Curves fall steeply from the left, so the legend goes under them on the right.
    axes[-1].legend(loc="upper right", handlelength=2.0, borderaxespad=0.3,
                    labelspacing=0.2, fontsize=7)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(out.with_suffix(f".{fmt}"), dpi=400)
    plt.close(fig)
    d.pivot_table(index=["dataset", "radius_param"], columns="algo",
                  values="cost", aggfunc="mean").to_csv(out.with_suffix(".csv"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"],
                    help="pdf for LaTeX inclusion (default); png to preview")
    ap.add_argument("--out", default="results/paper", type=Path)
    a = ap.parse_args(argv)

    # Figures 1 and 2 need the full radius sweep on the largest UCI samples available.
    # results_upstream.csv carries that grid on instances resampled from the full pools
    # (n = 2000, m = 100 and 400); results.csv is the fallback, but its UCI instances are
    # the small bundled ones at n <= 500.
    frames = []
    for f in ("results/results_upstream.csv", "results/results.csv"):
        if Path(f).exists():
            frames.append(pd.read_csv(f))
    m = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    m = m[m.error.isna() | (m.error == "")]
    if (m.dataset.str.startswith("upstream:")).any():
        m = m[m.dataset.str.startswith("upstream:")]   # prefer the larger samples
    pal = PALETTE
    style(pal)
    a.out.mkdir(parents=True, exist_ok=True)
    dist = None
    dp = Path("results/radius_dist.npz")
    if dp.exists():
        z = np.load(dp)
        dist = {tuple(k.split("||")): z[k] for k in z.files if k != "levels"}
    fig_cost(m, pal, a.out / "fig1-cost", a.format)
    fig_cost_absolute(m, pal, a.out / "fig2-cost-absolute", a.format)
    fig_fairness(m, pal, a.out / "fig3-fairness", a.format, dist=dist)
    print(f"wrote 3 figures (.{a.format} + .csv) to {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
