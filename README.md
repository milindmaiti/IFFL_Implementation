# Individually Fair Facility Location

Implementations and an empirical comparison of three approximation algorithms for
**Individually Fair Facility Location (IFFL)**.

In IFFL each client `j` carries a personal fairness radius `R_j`, and the solution must
serve `j` within a bounded multiple of it. This is *individual* fairness: the guarantee
attaches to each client separately, rather than to a protected group in aggregate.

Formally, given facilities `F` with opening costs `f_i`, clients `C` in a metric space,
and radii `R_j`, we want to open `S ⊆ F` minimising

```
cost(S) = Σ_{i∈S} f_i + Σ_{j∈C} d(j, S)
```

compared against `OPT_fair`, the cheapest `S` with `d(j,S) ≤ R_j` for every client. An
`(α, β)`-approximation costs at most `α·OPT_fair` and serves every client within `β·R_j`.

---

## The three algorithms

| | Approach A | Approach B | Approach C |
|---|---|---|---|
| **method** | LP rounding | primal-dual (Jain–Vazirani style) | greedy with savings |
| **fairness** | `d(j,S) ≤ 3R_j`, every outcome | `d(j,σ(j)) ≤ 3R` | `d(j,σ(j)) ≤ R_j` (**strict**) |
| **cost** | `E[cost] ≤ 1.725·OPT_fair` | `C + 3F ≤ 3·OPT_LP` | 1.861 unrestricted; see note |
| **radii** | arbitrary **nonuniform** | **uniform only** | arbitrary |
| **module** | [`iffl/lp_rounding.py`](iffl/lp_rounding.py) | [`iffl/primal_dual.py`](iffl/primal_dual.py) | [`iffl/greedy.py`](iffl/greedy.py) |

All three restrict to the admissible edge set `E_R = {(i,j) : d(i,j) ≤ R_j}`. That
restriction is what makes the LP optimum a valid lower bound on `OPT_fair`, it is the
sole source of Approach B's fairness guarantee, and it is the whole of Approach C's
adaptation to the fair setting.

**Approach A** scales a restricted LP solution by `γ = 69/40`, splits facilities into
colocated copies, filters each client's closest unit of opening mass, forms disjoint
clusters in increasing order of maximum connection distance, and rounds.

**Approach B** grows client duals continuously inside `E_R` until facilities go tight,
then prunes to a maximal independent set in the conflict graph of witness sets. Its
bound is proved for *uniform* radii only: the witness argument evaluates to
`R_j + R_{j*} + R_{j*}`, so a witness with a much larger radius destroys any bound
stated relative to `R_j`. We confirm this empirically below.

**Approach C** is the odd one out in a useful way. Forbidding inadmissible pairs at
every step makes it **strictly fair** — a radius ratio of 1, not 3 — so it is a
`(ρ, 1)` algorithm against the others' `(1.725, 3)` and `(3, 3)`, and the question it
answers is what that stricter fairness costs. Its 1.861 dual-fitting bound is for
*unrestricted* UFL; the argument plausibly transfers, since the restricted dual
constraint involves only `E_R` pairs, but nothing here asserts it — the ratio is
measured like any other observable.

---

## Install

Requires Python ≥ 3.12 (set by the numpy/scipy floors). Developed and tested on 3.14.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Reproduce the experiments

```bash
# 1. unit tests, including every invariant the three proofs rely on
.venv/bin/python -m pytest tests/ -q

# 2. full sweep: 657 configurations x 5 algorithms  (~1 min)
#    --strict aborts on any guarantee violation rather than recording it
.venv/bin/python scripts/run_experiments.py --all --trials 50 --strict \
    --out results/results.csv

# 3. figures, light and dark, each with a CSV table view
.venv/bin/python scripts/plot_results.py
.venv/bin/python scripts/plot_results.py --dark

# optional: materialize (radii, costs) sidecars + a provenance manifest
.venv/bin/python scripts/make_iffl_instances.py --out instances/
```

A single instance, for a quick look:

```bash
.venv/bin/python scripts/run_experiments.py --datasets synth-225-25 \
    --radius nearest_facility_multiple --alpha 2.0 --trials 20 --verbose
```

---

## Layout

```
iffl/
  instance.py     Instance dataclass; distance matrix
  loaders.py      the six CSV pairs and two workbooks under data/
  radii.py        uniform / nearest-facility-multiple / Jung–Kannan–Lutz schemes
  costs.py        opening-cost synthesis for the datasets that lack costs
  lp.py           restricted LP over E_R, via scipy + HiGHS
  lp_rounding.py  Approach A
  primal_dual.py  Approach B
  greedy.py       Approach C
  synthetic.py    odd-cycle instances with a half-integral LP
  evaluate.py     shared metrics and two baselines
  configs.py      the experiment grid
  pipeline.py     build an instance from a config, run everything on it
scripts/
  run_experiments.py       sweep → results/results.csv
  plot_results.py          figures + CSV table views
  make_iffl_instances.py   write (R, f) sidecars + manifest to instances/
data/                      benchmark instances; see data/SOURCES.md
tests/                     266 tests
```

---

## Datasets

Bundled under [`data/`](data/) and described in [`data/SOURCES.md`](data/SOURCES.md).

| dataset | n | m | n/m | dim | groups | native costs | LP fractional |
|---|---|---|---|---|---|---|---|
| `adult-500-20` | 500 | 20 | 25 | 6 | race(5), sex(2) | no | 0% |
| `adult-429-65` | 429 | 65 | 6.6 | 6 | race(5), sex(2) | no | 19.4% |
| `bank-500-20` | 500 | 20 | 25 | 3 | job(12), marital(3), educ(4), default(2) | no | 0% |
| `synth-225-25` | 250 | 10 | 25 | 2 | in/out | 40/80 | 0% |
| `synth-500-40` | 550 | 20 | 27.5 | 2 | in/out | 40/80 | 0% |
| `synth-500-50` | 550 | 20 | 27.5 | 2 | in/out | 40/80 | 0% |
| `uflp-49` | 49 | 49 | 1.0 | lat/lon | party(2) | 3.8–16.1 | 22.6% |
| `uflp-88` | 88 | 88 | 1.0 | lat/lon | region(3) | 2.6–29.9 | 7.1% |
| `oddcycle-{5,9,15}` | k | k | 1.0 | synthetic | — | uniform | 100% |

Categorical columns are carried for analysis only; no algorithm here uses them.

Two structural facts drive most of the behaviour below:

* **Fractionality tracks facilities-per-client.** Every dataset with `m/n ≈ 0.04` has an
  integral LP optimum in every configuration; the only fractional ones are
  `adult-429-65` (`m/n = 0.15`) and the two workbooks (`m/n = 1.0`). To get instances
  where the LP rounding does real work, you need more candidate facilities.
* **Radius heterogeneity drives Approach B's failures.** The synthetic sets plant 10%
  `out` clients that sit 1.6–3.5× farther from facilities than the rest, giving the
  widest `R_max/R_med` spread in the corpus — and they produce B's worst nonuniform
  violations.

### Sampling larger instances

The bundled files are small samples (429–550 clients) of surveys with ~45,000 rows.
[`iffl/upstream.py`](iffl/upstream.py) rebuilds those populations and draws arbitrarily
larger instances. The preprocessing was recovered by matching each bundled file's `id`
column against the upstream data, and reproduces the bundled coordinates to within
`1e-12` — so a freshly sampled 5,000-client instance is *comparable* with the bundled
one, not merely similar.

| pool | rows | preprocessing |
|---|---|---|
| `adult` | 45,222 | `adult.data` + `adult.test`, rows with missing values dropped; `age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week` z-scored with the **population** std |
| `bank` | 45,211 | `bank-full.csv`; `age, balance, duration`, same scaling |

The bundled `bank-500-20` comes from `bank.csv` (4,521 rows), not `bank-full.csv`; the
sampler uses the larger pool by default, which changes the normalisation base and is
recorded in each instance's provenance. Facilities in the bundled files are not pool
rows but synthetic points closest in geometry to k-means centroids of the population, so
`kmeans_pool` is the default generator — a reconstruction, not a claim to reproduce the
original generator exactly.

```bash
python scripts/run_scale_study.py --pools adult bank --out results/scale.csv
```

Instances are addressed as `upstream:<pool>:<n>:<m>[:<seed>]`, e.g.
`upstream:adult:5000:1000`, and work anywhere a dataset name does. The first call
downloads and caches the archives under `.cache/`.

### Radius schemes

Radii are not in the data; they are generated, and every scheme is clamped to
feasibility (a client whose radius falls below its nearest-facility distance has no
admissible edge and makes the LP infeasible).

* `nearest_facility_multiple(α, rank)` — `R_j = α · d(j, rank-th nearest facility)`.
  Nonuniform, feasible by construction for `α ≥ 1`.
* `uniform(mult, rank)` — one shared radius, `mult × max_j basis_j`. Since
  `max_j basis_j` is the tightest uniform radius leaving every client an admissible
  edge, `mult ≥ 1` is feasible by construction and the parameter means "slack beyond
  the tightest feasible uniform radius". Clamping individual clients instead would
  restore variation but break the uniformity Approach B's proof depends on, so the
  scale is applied to the shared value.
* `jkl_client_knn(k)` — Jung–Kannan–Lutz: distance to the `⌈n/k⌉`-th nearest *client*.

---

## Results

From `results/results.csv` — 657 configurations (dataset × radius scheme × cost scale)
× 5 algorithms. Approach A is averaged over 50 independent roundings per configuration.

| algorithm | mean cost / LP | max | worst radius ratio |
|---|---|---|---|
| A: LP rounding | 1.017 | 1.602 | 3.000 — bound 3, **0 violations** |
| B: primal-dual, uniform radii | 1.012 | — | 3.000 — bound 3, **0 violations** |
| B: primal-dual, nonuniform radii | 1.012 | 1.200 | exceeds 3 in **76/424** configs |
| **C: greedy (strictly fair)** | **1.006** | **1.196** | **1.000** — bound 1, 0 violations |
| baseline: threshold rounding | 1.012 | 2.648 | opens nothing on a fractional LP |
| baseline: set-cover greedy | 1.781 | 40.43 | 1.000 by construction |

* **Approach A** never violated `3R_j` on any trial of any configuration, and its mean
  cost ratio stayed under 1.725 everywhere (max 1.602, on a fractional instance).
* **Approach B** satisfied `C + 3F ≤ 3·Σα` in all 233 uniform-radius configurations.
  On nonuniform radii it exceeds `3R_j` in 76 of 424, worst finite ratio 44.1 — the
  concrete form of the uniform-radii requirement, and the gap Approach A closes.
* **Approach C** was both the fairest and the cheapest on this data: strictly fair in
  all 657 configurations, mean ratio 1.006, never above 1.861 (max 1.196). On the 54
  fractional-LP configurations, where the comparison is meaningful, it averaged 1.044
  against A's 1.181.

Keep that last point in proportion: these are *empirical* ratios on eight geometric
datasets, not worst cases. A and B carry proved worst-case guarantees that C, in the
restricted setting, does not.

### At larger scale

`results/scale.csv` resamples both pools across instance sizes up to 5,000 clients and
1,000 facilities (540 rows). Two things come out of it.

**Fractional LPs appear exactly where `m/n` predicts**, on real data:

| `m/n` | adult | bank |
|---|---|---|
| 0.01 | 0% | 0% |
| 0.04 | 0% | 0% |
| 0.05 | 0% | 0% |
| 0.20 | 22% | 17% |
| 0.50 | 33% | 17% |

Bank fractures less readily than Adult — its feature space is 3-D against Adult's 6-D,
so facilities are easier to place well and the LP has less room to hedge.

**The algorithms converge at scale.** On the fractional-LP configurations, which are the
only ones where the three differ at all:

| algorithm | bundled, all | bundled, real data only | upstream (n = 500–5000) | upstream max |
|---|---|---|---|---|
| A: LP rounding | 1.181 | 1.033 | **1.012** | 1.024 |
| B: primal-dual | 0.992 | 1.014 | 1.042 | 1.087 |
| C: greedy | 1.044 | 1.014 | **1.009** | 1.018 |

A's apparent weakness on the bundled data was an artifact of small instances: 21 of
those 60 fractional configurations are 5–15 node odd cycles, where its rounding pays its
worst price. Strip those out and the spread already narrows; resample at realistic size
and A and C are within 0.3% of each other. **The honest summary is that on fractional
instances at scale all three sit within ~4% of the LP bound, and Approach C's real
advantage is its fairness — strictly within `R_j` — rather than its cost.**

Every guarantee continues to hold at this scale: 0 violations in 54 uniform-radius
configurations per algorithm. Approach B still breaks on nonuniform radii, in 20 of 54,
worst ratio 28.8.

### Reading the cost ratios

The denominator is the **LP relaxation** optimum over `E_R`, never `OPT_fair`. No
optimum is computed anywhere and none is needed: since `LP ≤ OPT_fair`, we have
`cost/LP ≥ cost/OPT_fair`, so every guarantee verified against the LP holds a fortiori
against the true optimum. The check is conservative, never flattering. On the 597
integral-LP configurations `LP = OPT_fair` exactly, so those ratios *are* the true
approximation ratios.

This is also why A and B can score **below 1**. Being bicriteria, they may serve clients
out to `3R_j` and so undercut the best strictly fair solution; the LP does not lower
bound them. Only C, being strictly fair, is genuinely floored at 1.

### Approach B's assignment

The primal-dual pruning routes each client either directly to its witness facility or
indirectly through a shared witness. That routing is an analytical device: it is what
bounds `d(j,i)` by `3R` and underpins `C + 3F ≤ 3·Σα`. It is not what the algorithm
should return. Clients are therefore served from the **closest open facility**, which
can only shorten a connection, so both bounds survive a fortiori. The witness routing is
still computed and reported (`witness_cost`, `witness_max_radius_ratio`) for comparison;
serving from the nearest open facility is worth about 1% of cost and tightens the worst
radius ratio too.

Figures are in `results/figures/`, light and dark, each with a CSV table view. The
tradeoff panel is restricted to uniform radii, the only regime where all three
algorithms are comparable.

---

## Implementation notes

Four things these datasets make easy to get wrong, each covered by a test.

1. **Clients sitting on facility sites.** In both workbooks clients and facilities are
   the *same* point set, so every client's nearest-facility distance is exactly 0. A
   radius of `α · 0` is 0 for every `α`: `E_R` collapses to self-edges, each client is
   served by itself, and the LP must open all `m` facilities. Hence the `rank` argument
   on the distance-based schemes. The same zero leaked into cost synthesis, where the
   unit is a mean nearest-facility distance; `costs.distance_unit` steps up ranks until
   it finds a positive scale.

2. **`d / R` when `R = 0`.** One client of `adult-429-65` coincides exactly with a
   facility, so `R_j = 0` legitimately. Reporting `nan` would silently defeat every
   `≤ 3` assertion, since `nan ≤ 3` and `nan > 3` are both false. `evaluate.radius_ratio`
   resolves it by the constraint instead: 0 when the client is served at distance 0,
   `inf` when it is not.

3. **Catastrophic cancellation in the distance matrix.** The usual
   `|a|² + |b|² − 2a·b` expansion returns a self-distance near `1e-6` on raw lat/lon,
   which then perturbs every radius keyed on the nearest-facility distance. `cdist` is
   used instead.

4. **Integral LPs hiding the interesting code path.** Where the LP optimum is integral,
   each client's close set sits inside one facility, so Approach A deterministically
   reproduces the LP and returns ratio 1.000 — correct, but exercising neither the
   splitting, the filtering, nor the randomized rounding.
   `synthetic.odd_cycle_instance` supplies a controlled half-integral case: facilities
   on an odd cycle's vertices, clients on its edges, so the LP reduces to fractional
   vertex cover and `y_i = 1/2`. Its distances come from the incidence graph's shortest
   paths, because the `3R` argument walks a client → facility → client → facility path
   and so needs the triangle inequality across all of `F ∪ C`.

---

## References

The greedy algorithm is Algorithm 9.2 of:

> D. P. Williamson and D. B. Shmoys. *The Design of Approximation Algorithms*.
> Cambridge University Press, 2011.

The LP-rounding framework follows:

> F. A. Chudak and D. B. Shmoys. Improved approximation algorithms for the uncapacitated
> facility location problem. *SIAM Journal on Computing*, 33(1):1–25, 2003.
> [doi:10.1137/S0097539703405754](https://doi.org/10.1137/S0097539703405754)

> J. Byrka and K. Aardal. An optimal bifactor approximation algorithm for the metric
> uncapacitated facility location problem. *SIAM Journal on Computing*,
> 39(6):2212–2231, 2010. [arXiv:cs/0703010](https://arxiv.org/abs/cs/0703010)

Datasets are redistributed from the **Fair-FL** project and its upstream sources; see
[`data/SOURCES.md`](data/SOURCES.md) for full attribution.
