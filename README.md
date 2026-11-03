# Individually Fair Facility Location

Python implementation of two approximation algorithms for **Individually Fair Facility
Location (IFFL)**, run over the datasets in the sibling [`Fair-FL`](../Fair-FL) repo.

In IFFL every client `j` carries a personal fairness radius `R_j`, and the solution must
serve `j` within a bounded multiple of it. That is *individual* fairness, as opposed to
the *group* fairness (per-group outlier budgets) the Fair-FL Julia notebooks implement.

## The two algorithms

| | Approach A | Approach B |
|---|---|---|
| source | `lp_fl_approach.pdf` | `Individually_Fair_FL.pdf` |
| method | LP rounding | primal-dual (Jain-Vazirani style) |
| guarantee | `E[cost] <= 1.725 OPT_fair`, `d(j,S) <= 3R_j` always | `C + 3F <= 3 OPT_LP`, `d(j,sigma(j)) <= 3R` |
| radii | arbitrary **nonuniform** | **uniform only** (Remark 2.3) |
| module | `iffl/lp_rounding.py` | `iffl/primal_dual.py` |

Both restrict to the admissible edge set `E_R = {(i,j) : d(i,j) <= R_j}`. That
restriction is what makes the LP optimum a valid lower bound on `OPT_fair`, and it is
the only source of Approach B's fairness guarantee.

The cost and fairness bounds are *not* checked the same way, because the papers do not
state them the same way. Approach A's `3R_j` is deterministic, so it is asserted on
every single trial; its `1.725` holds only in expectation, so it is compared against the
mean over trials. Approach B's bound is proved for uniform radii, so it is asserted
there and merely measured on nonuniform radii.

## Quick start

```bash
/opt/homebrew/bin/python3.14 -m venv .venv      # see "Environment" below
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q

.venv/bin/python scripts/run_experiments.py --all --trials 50 --out results/results.csv
.venv/bin/python scripts/plot_results.py
```

## Layout

```
iffl/
  instance.py     Instance dataclass; distances (cdist, see "Gotchas")
  loaders.py      the six CSV pairs + two UFLP workbooks
  radii.py        uniform / nearest-facility-multiple / Jung-Kannan-Lutz schemes
  costs.py        opening-cost synthesis for the datasets that lack costs
  lp.py           restricted LP over E_R, via scipy + HiGHS
  lp_rounding.py  Approach A
  primal_dual.py  Approach B
  synthetic.py    odd-cycle instances with a half-integral LP
  evaluate.py     shared metrics and two baselines
  configs.py      the experiment grid
  pipeline.py     build an instance from a config, run everything on it
scripts/
  make_iffl_instances.py   write (R, f) sidecars + manifest to instances/
  run_experiments.py       sweep -> results/results.csv
  plot_results.py          figures + CSV table views
```

## Datasets

`Fair-FL` is only ever read, never modified. Radii and synthesized costs go to sidecar
files under `instances/`.

| dataset | clients | facilities | dims | delimiter | native costs |
|---|---|---|---|---|---|
| `adult-500-20` | 500 | 20 | 6 | **space** | no |
| `adult-429-65` | 429 | 65 | 6 | comma | no |
| `bank-500-20` | 500 | 20 | 3 | **space** | no |
| `synth-225-25` | 250 (225 `in` + 25 `out`) | 10 | 2 | comma | yes (40/80) |
| `synth-500-40` | 550 (500 + 50) | 20 | 2 | comma | yes (40/80) |
| `synth-500-50` | 550 (500 + 50) | 20 | 2 | comma | yes (40/80) |
| `uflp-49` | 49 | 49 (**same points**) | lat/lon | xlsx | yes |
| `uflp-88` | 88 | 88 (**same points**) | lat/lon | xlsx | yes |
| `oddcycle-{5,9,15}` | k | k | synthetic | - | yes |

Distances and costs mirror the Julia originals (`l2` over the `d*` columns,
`fixed cost / 10000` for the workbooks) so numbers stay comparable.

## Gotchas these datasets actually hit

Four things that are easy to get wrong here, each covered by a test:

1. **Clients sitting on facility sites.** In both UFLP workbooks clients and facilities
   are the *same* point set, so every client's nearest-facility distance is exactly 0.
   A radius of `alpha * 0` is 0 for every `alpha`: `E_R` collapses to self-edges, each
   client is served by itself, and the LP must open all `m` facilities. The distance
   schemes therefore take a `rank` argument, and `configs.radius_rank_for` uses rank 2
   for those datasets. The same zero leaked into cost synthesis, where a unit of "mean
   nearest-facility distance" is also 0 - `costs.distance_unit` steps up ranks until it
   finds a positive scale.

2. **`d / R` when `R = 0`.** One client of `adult-429-65` coincides exactly with a
   facility, so `R_j = 0` legitimately. Reporting `nan` there would silently defeat
   every `<= 3` assertion, since `nan <= 3` and `nan > 3` are both false.
   `evaluate.radius_ratio` resolves it by the constraint instead: 0 when the client is
   served at distance 0, `inf` when it is not.

3. **Catastrophic cancellation in the distance matrix.** The usual
   `|a|^2 + |b|^2 - 2a.b` expansion returns a self-distance near `1e-6` on raw lat/lon,
   which then perturbs every radius keyed on the nearest-facility distance. `cdist` is
   used instead.

4. **Integral LPs hiding the interesting code path.** Most configurations of the real
   datasets have an integral LP optimum. Where that happens each client's close set sits
   inside one facility, so Approach A deterministically reproduces the LP and returns
   ratio 1.000 - correct, but it exercises neither the splitting, the filtering, nor the
   randomized rounding. 60 of 657 configurations are fractional (concentrated in
   `adult-429-65`, `uflp-49`, `uflp-88`), and `synthetic.odd_cycle_instance` supplies a
   controlled half-integral case on demand: facilities on an odd cycle's vertices,
   clients on its edges, so the LP reduces to fractional vertex cover and `y_i = 1/2`.
   Its distances come from the incidence graph's shortest paths, because the `3R`
   argument walks a client -> facility -> client -> facility path and so needs the
   triangle inequality across all of `F union C`.

## Results

From `results/results.csv` (2628 rows; 657 configurations x 4 algorithms):

| algorithm | mean cost / LP | worst radius ratio |
|---|---|---|
| A: LP rounding | 1.017 | 3.000 (bound 3, 0 violations) |
| B: primal-dual, uniform radii | - | 3.000 (bound 3, 0 violations) |
| B: primal-dual, nonuniform radii | - | exceeds 3 in 78/424 configs |
| Baseline: threshold rounding | 1.012 | opens nothing on a fractional LP |
| Baseline: strictly fair greedy | 1.781 | 1.000 by construction |

* Approach A never violated `3R_j`, on any trial of any configuration, and its mean cost
  ratio stayed under 1.725 in all 657 configurations (max observed 1.602, on a
  fractional instance).
* Approach B satisfied `C + 3F <= 3 sum(alpha)` in all 233 uniform-radius configurations.
* On nonuniform radii Approach B exceeds `3R_j` in 78 of 424 configurations, worst finite
  ratio 44.1 - the concrete form of Remark 2.3, and the gap Approach A exists to close.
* Both algorithms track the LP bound closely; the strictly fair greedy baseline, which
  never uses the bicriteria slack, costs 78% more on average.

Figures are in `results/figures/` (light and dark), each with a CSV table view.

## Environment

`python3` on this machine is Xcode's 3.9.6, whose `site-packages` is not writable.
Homebrew's `python@3.14` is marked `EXTERNALLY-MANAGED`, so a venv is the supported
route - which is what the quick start does. Commands use `.venv/bin/python` explicitly
rather than a bare `python3`.
