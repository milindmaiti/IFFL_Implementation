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

# 2. the paper's sweep: 132 configurations x 3 algorithms
#    --strict aborts on any guarantee violation rather than recording it
.venv/bin/python scripts/run_experiments.py \
    --datasets upstream:adult:1000:200:0 upstream:adult:1000:400:0 \
               upstream:bank:1000:200:0  upstream:bank:1000:400:0 \
    --facility-mode sample_pool --cost-mode inverse_attractiveness \
    --cost-scales 4 7 15 --trials 50 --strict \
    --dump-radius-dist results/radius_dist.npz \
    --out results/results_upstream.csv

# 3. the three paper figures as vector PDF, each with a CSV table view
.venv/bin/python scripts/plot_paper_figures.py --format pdf
```

Instances are drawn from the full UCI pools; the first call downloads and caches the
archives under `.cache/`.

A single configuration, for a quick look:

```bash
.venv/bin/python scripts/run_experiments.py --datasets upstream:adult:1000:200:0 \
    --facility-mode sample_pool --cost-mode inverse_attractiveness \
    --radius nearest_facility_multiple --alpha 2.0 --trials 20 --verbose
```

---

## Layout

```
iffl/
  instance.py     Instance dataclass; distance matrix
  loaders.py      the bundled test fixtures under data/
  radii.py        uniform / nearest-facility-multiple / Jung–Kannan–Lutz schemes
  costs.py        opening-cost synthesis for the datasets that lack costs
  lp.py           restricted LP over E_R, via scipy + HiGHS
  lp_rounding.py  Approach A
  primal_dual.py  Approach B
  greedy.py       Approach C
  synthetic.py    odd-cycle instances with a half-integral LP
  evaluate.py     shared metrics and two baselines
  configs.py      the experiment grid
  upstream.py     rebuild the full UCI pools; sample arbitrary instances
  pipeline.py     build an instance from a config, run everything on it
scripts/
  run_experiments.py       sweep → results/results_upstream.csv
  plot_paper_figures.py    the three paper figures + CSV table views
  run_hardness_study.py    Theorem 2 on the Set Cover reduction (not in the paper)
data/                      test fixtures; see data/SOURCES.md
tests/                     316 tests
```

---

## Datasets

The experiments draw their instances from the **full UCI pools**, rebuilt by
[`iffl/upstream.py`](iffl/upstream.py) and addressed as
`upstream:<pool>:<n>:<m>[:<seed>]`, e.g. `upstream:adult:1000:200:0`.

| pool | rows | preprocessing |
|---|---|---|
| `adult` | 45,222 | `adult.data` + `adult.test`, rows with missing values dropped; `age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week` z-scored with the **population** std |
| `bank` | 45,211 | `bank-full.csv`; `age, balance, duration`, same scaling |

The paper uses four instances: each pool at `n = 1000` with `m ∈ {200, 400}`. Candidate
facilities are drawn from the records themselves (`--facility-mode sample_pool`), so
sites are real locations; any record whose coordinates coincide with a sampled client is
excluded, keeping `d(j,F) > 0` for every client. Opening costs are synthesised by
`--cost-mode inverse_attractiveness`, which prices a site by how cheaply it can reach
nearby clients, so a facility amid many clients is a genuine trade-off rather than a free
choice. Under uniform costs the instance is a `k`-median problem for a suitable `k`,
which is exactly the regime where the relaxation tends to be integral.

### Bundled fixtures — tests only

[`data/`](data/) holds small instances (429–550 clients) sampled from the same two
surveys, two UFLP workbooks, and three synthetic sets; see
[`data/SOURCES.md`](data/SOURCES.md). **Nothing in the paper is computed from them.**
They are kept because they serve two purposes for the test suite:

* most of the 316 tests run against them — small enough to be fast, and checked in, so
  the suite runs without network access (the oracle tests below skip when the upstream
  archives are not yet cached);
* [`tests/test_upstream.py`](tests/test_upstream.py) uses them as an *oracle*. The
  preprocessing in `upstream.py` was recovered by matching each bundled file's `id`
  column against the upstream data, and reproduces the bundled coordinates to within
  `1e-12`. That check is what licenses treating a freshly sampled instance as
  comparable with the published one rather than merely similar.

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

From `results/results_upstream.csv` — 132 configurations (4 instances x 3 opening-cost
scales x 11 radius settings) x 3 algorithms, Approach A averaged over 50 independent
roundings. The LP relaxation is **fractional in 72 of the 132**, so the rounding does
real work rather than reproducing an integral optimum.

| algorithm | mean cost / LP | max | min | worst radius ratio |
|---|---|---|---|---|
| A: LP rounding | 1.018 | 1.136 | 1.000 | 1.474 — bound 3, **never exceeded** |
| B: primal-dual | 0.944 | 1.110 | **0.474** | **215.9** — bound 3, uniform radii only |
| **C: greedy (strictly fair)** | **1.014** | **1.082** | 1.000 | **1.000** — bound 1, never exceeded |

* **Approach A** stays far inside both guarantees: never past `1.474 R_j` against a
  bound of `3`, and exact on the 60 configurations where the LP optimum is integral.
  Pooled over clients it serves at least **99.86%** of them within their radius under
  either client-specific scheme — and 99.99% restricted to the fractional
  configurations, where the rounding is actually exercised.
* **Approach B** falls as low as `0.474`, below a bound that constrains only strictly
  fair solutions. It exceeds its radius for about a fifth of client assignments under
  either scheme, but the tails diverge sharply: never past `2.22` under
  Jung–Kannan–Lutz radii, against `2.7%` of assignments past `3R_j` and a worst case of
  `216` under the nearest-facility multiple, which inherits the dispersion of outlying
  clients so that a witness may carry a radius many times that of the client routed
  through it. Its whole distribution is shifted, not just its tail: the median client
  sits at `0.78` of its radius, against `0.57` for the other two.
* **Approach C** is strictly fair in every configuration at `1.4%` above the LP bound on
  average. Theorem 2 of the paper rules out an `o(log n)` cost factor for any strictly
  fair algorithm unless `P = NP`; that bound is realised by instances encoding Set Cover
  rather than by geometric data.

Two structural facts drive the rest:

* **Fractionality tracks the radius scheme.** 81% of the Jung–Kannan–Lutz configurations
  are fractional, against 15% of the nearest-facility ones. It is near-insensitive to
  the opening-cost scale over the range swept (57% / 55% / 52% at `c = 4 / 7 / 15`).
* **A feasible uniform radius constrains almost nothing here.** It must reach the
  worst-placed client, who is a far outlier, so the scheme admits 91–100% of all
  `(i,j)` pairs. It is swept and reported but omitted from the figures; the comparisons
  above are over all 132 configurations, the figures over the 108 client-specific ones.

### Reading the cost ratios

The denominator is the **LP relaxation** optimum over `E_R`, never `OPT_fair`. No
optimum is computed anywhere and none is needed: since `LP ≤ OPT_fair`, we have
`cost/LP ≥ cost/OPT_fair`, so every guarantee verified against the LP holds a fortiori
against the true optimum. The check is conservative, never flattering. On the 60
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

Figures are written to `results/paper/`, as vector PDF with a CSV table view beside
each.

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
