#!/usr/bin/env python3
"""The three figures for the paper, on UCI data.

1. Cost      -- cost / LP bound against radius tightness.
2. Fairness  -- worst radius violation against radius tightness, by radius scheme.
3. Runtime   -- seconds against |E_R|, the quantity that actually drives it.

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


def fig_cost(main, pal, out: Path, fmt: str):
    d = uci(main)
    d = d[d.radius_scheme == "nearest_facility_multiple"]
    piv = d.pivot_table(index="radius_param", columns="algo", values="cost_ratio", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for k, a in enumerate(ALGOS):
        _series(ax, piv.index, piv[a], k, a, pal)
    ax.axhline(1.0, color=pal["rule"], lw=0.8)
    ax.set_xlabel(r"radius multiplier $\alpha$")
    ax.set_ylabel("cost / LP bound")
    lo, hi = float(piv.to_numpy().min()), float(piv.to_numpy().max())
    pad = max(0.0015, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([1, 2, 3, 4])
    _grid(ax)
    ax.legend(loc="upper left", handlelength=2.6, borderaxespad=0.2)
    fig.savefig(out.with_suffix(f".{fmt}"), dpi=400)
    plt.close(fig)
    piv.to_csv(out.with_suffix(".csv"))


def fig_fairness(main, pal, out: Path, fmt: str):
    """Worst radius violation, by radius scheme.

    The y-axis is logarithmic.  The primal-dual's breach reaches 12x its own bound, and
    on a linear range of that size the LP rounding and the greedy collapse into two
    indistinguishable lines on the axis.  A log scale keeps the natural quantity
    ``d(j,sigma(j))/R_j`` and the ``3R`` rule where a reader expects them, while giving
    the neighbourhood of 1 enough room to separate the LP rounding's climb to 1.21 from
    the greedy's flat ``<= 1``.  Ticks are placed explicitly, since the default decade
    ticks would label nothing between 1 and 10.

    Columns split by radius scheme because the primal-dual's failure is a property of
    radius heterogeneity, not of the algorithm alone: under Jung-Kannan-Lutz radii, which
    are derived from client density and come out relatively homogeneous, it never breaches.
    """
    d = uci(main)
    f = lambda s: np.nanmax(s[np.isfinite(s)])  # noqa: E731
    panels = [("nearest_facility_multiple", r"radius multiplier $\alpha$", [1, 2, 3, 4]),
              ("jkl_client_knn", r"neighbourhood parameter $k$", [5, 10, 20])]
    yticks = [1, 1.25, 1.5, 2, 3, 5, 10]

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9), sharey=True)
    for ax, (scheme, xlab, ticks) in zip(axes, panels):
        piv = d[d.radius_scheme == scheme].pivot_table(
            index="radius_param", columns="algo", values="max_radius_ratio", aggfunc=f)
        for k, a in enumerate(ALGOS):
            _series(ax, piv.index, piv[a], k, a, pal)
        ax.axhline(3.0, color=pal["rule"], lw=0.8, ls="--")
        ax.axhline(1.0, color=pal["rule"], lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in yticks]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0.95, 15)
        ax.set_xlabel(xlab)
        ax.set_xticks(ticks)
        _grid(ax)

    axes[0].set_ylabel(r"worst $d(j,\sigma(j))\,/\,R_j$")
    axes[0].text(0.98, 3.0, "$3R$", transform=axes[0].get_yaxis_transform(),
                 color=pal["muted"], fontsize=8, va="bottom", ha="right")
    axes[1].legend(loc="upper left", handlelength=2.4, borderaxespad=0.4, labelspacing=0.3)
    fig.subplots_adjust(wspace=0.07)
    fig.savefig(out.with_suffix(f".{fmt}"), dpi=400)
    plt.close(fig)
    d.pivot_table(index=["radius_scheme", "radius_param"], columns="algo",
                  values="max_radius_ratio", aggfunc=f).to_csv(out.with_suffix(".csv"))


def fig_runtime(scale, pal, out: Path, fmt: str):
    d = scale[scale.algo.isin(ALGOS)].copy()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for k, a in enumerate(ALGOS):
        s = d[d.algo == a].sort_values("n_edges")
        bins = np.logspace(np.log10(s.n_edges.min()), np.log10(s.n_edges.max()), 11)
        g = s.groupby(pd.cut(s.n_edges, bins=bins), observed=True).agg(
            x=("n_edges", "median"), y=("runtime_s", "median"))
        _series(ax, g.x, g.y, k, a, pal)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"admissible edges $|E_R|$")
    ax.set_ylabel("runtime (s)")
    _grid(ax, axis="both")
    ax.legend(loc="upper left", handlelength=2.6, borderaxespad=0.2)
    fig.savefig(out.with_suffix(f".{fmt}"), dpi=400)
    plt.close(fig)
    d.groupby(["algo", "n", "m"])[["n_edges", "runtime_s"]].median().to_csv(out.with_suffix(".csv"))


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
    sc = pd.read_csv("results/scale.csv")
    sc = sc[sc.error.isna() | (sc.error == "")]

    pal = PALETTE
    style(pal)
    a.out.mkdir(parents=True, exist_ok=True)
    fig_cost(m, pal, a.out / "fig1-cost", a.format)
    fig_fairness(m, pal, a.out / "fig2-fairness", a.format)
    fig_runtime(sc, pal, a.out / "fig3-runtime", a.format)
    print(f"wrote 3 figures (.{a.format} + .csv) to {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
