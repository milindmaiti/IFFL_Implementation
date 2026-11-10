"""Rebuild benchmark instances from the full upstream UCI datasets.

The bundled instances under ``data/`` are small samples (429-550 clients) of much larger
surveys.  This module reproduces the preprocessing they were built with, so larger
samples are directly comparable with them, and then draws new ones.

The preprocessing was recovered by matching the bundled files' ``id`` column against the
upstream data, and both pipelines reproduce the bundled coordinates to within 1e-12:

**Adult** -- concatenate ``adult.data`` and ``adult.test`` (48,842 rows), drop every row
containing a missing value (leaving 45,222), reset the index, and z-score the six
continuous columns ``age, fnlwgt, education-num, capital-gain, capital-loss,
hours-per-week`` using the **population** standard deviation.  ``id`` is the 0-based row
index into that frame; ``c1`` is ``race`` and ``c2`` is ``sex``.

**Bank** -- the bundled file is drawn from ``bank.csv``, the 4,521-row 10% sample, not
from ``bank-full.csv``.  Columns ``age, balance, duration`` are z-scored the same way;
``c1..c4`` are ``job, marital, education, y``.  Sampling from ``bank-full.csv`` instead
gives ten times the pool, at the cost of a different normalisation base -- which is why
the pool is recorded in every generated file's provenance.

Facilities are not rows of the pool in the bundled files; they are synthetic points whose
geometry is closest to k-means centroids of the full population.  ``kmeans_pool`` is
therefore the default here.  It is a reconstruction, not a claim to reproduce the
original generator exactly.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
ADULT_NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
                 "hours-per-week"]
ADULT_GROUPS = ["race", "sex"]

BANK_NUMERIC = ["age", "balance", "duration"]
BANK_GROUPS = ["job", "marital", "education", "y"]

FACILITY_MODES = ("kmeans_pool", "kmeans_clients", "sample_pool")


@dataclass
class Pool:
    """A preprocessed population to draw instances from."""

    name: str
    coords: np.ndarray           # (N, dim) z-scored numeric features
    groups: pd.DataFrame         # (N, k) categorical attributes
    source: str

    def __len__(self) -> int:
        return self.coords.shape[0]


def _download(url: str, cache: Path) -> bytes:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_bytes()
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310
        blob = r.read()
    cache.write_bytes(blob)
    return blob


def _zscore(df: pd.DataFrame) -> np.ndarray:
    """Z-score with the population standard deviation, matching the bundled files."""
    return ((df - df.mean()) / df.std(ddof=0)).to_numpy(dtype=float)


def load_adult_pool(cache_dir: Path = Path(".cache")) -> Pool:
    """The 45,222-row Adult population the bundled instances were drawn from."""
    blob = _download(ADULT_URL, cache_dir / "adult.zip")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        read = lambda n, skip: pd.read_csv(  # noqa: E731
            io.BytesIO(z.read(n)), header=None, names=ADULT_COLS,
            skipinitialspace=True, na_values="?", skiprows=skip,
        )
        full = pd.concat([read("adult.data", 0), read("adult.test", 1)], ignore_index=True)
    full = full.dropna().reset_index(drop=True)
    return Pool("adult", _zscore(full[ADULT_NUMERIC]), full[ADULT_GROUPS],
                "UCI Adult (adult.data + adult.test, rows with missing values dropped)")


def load_bank_pool(cache_dir: Path = Path(".cache"), full: bool = True) -> Pool:
    """The Bank Marketing population.

    Args:
        full: use ``bank-full.csv`` (45,211 rows).  ``False`` uses ``bank.csv``
            (4,521 rows), the pool the bundled ``bank-500-20`` was drawn from.
    """
    blob = _download(BANK_URL, cache_dir / "bank.zip")
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("bank.zip"))) as inner:
            name = "bank-full.csv" if full else "bank.csv"
            df = pd.read_csv(io.BytesIO(inner.read(name)), sep=";")
    return Pool("bank", _zscore(df[BANK_NUMERIC]), df[BANK_GROUPS],
                f"UCI Bank Marketing ({'bank-full.csv' if full else 'bank.csv'})")


def make_facilities(
    pool: Pool,
    clients: np.ndarray,
    m: int,
    mode: str = "kmeans_pool",
    seed: int = 0,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Generate ``m`` candidate facility sites.

    Args:
        exclude: pool rows the facilities may not be drawn from, used by ``sample_pool``
            to keep a facility from landing exactly on a client. A coincident pair gives
            that client a nearest-facility distance of 0, hence a radius of 0 under the
            distance-based schemes, and an infinite radius ratio however it is served.
            Exclusion is by *coordinate*, not index: both pools contain a few hundred
            duplicate rows (418 in Adult, 434 in Bank), so dropping a client's own index
            would still leave its twin available to be drawn as a facility.
    """
    if mode not in FACILITY_MODES:
        raise KeyError(f"unknown facility mode {mode!r}; known: {', '.join(FACILITY_MODES)}")
    if mode == "sample_pool":
        rng = np.random.default_rng(seed)
        avail = np.arange(len(pool))
        if exclude is not None:
            _, klass = np.unique(pool.coords, axis=0, return_inverse=True)
            taken = np.unique(klass[np.asarray(exclude)])
            avail = avail[~np.isin(klass, taken)]
        if avail.size < m:
            raise ValueError(
                f"pool has {avail.size} rows left after exclusions but {m} were requested"
            )
        return pool.coords[rng.choice(avail, m, replace=False)]

    from sklearn.cluster import KMeans

    src = pool.coords if mode == "kmeans_pool" else clients
    km = KMeans(n_clusters=m, n_init=10, random_state=seed).fit(src)
    return km.cluster_centers_


def sample_instance(
    pool: Pool, n: int, m: int, seed: int = 0, facility_mode: str = "kmeans_pool"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw ``n`` clients and build ``m`` facilities, in the bundled file schema.

    Returns ``(clients, facilities)`` as DataFrames with ``id, d1..dk, c1..cj`` and
    ``d1..dk`` respectively -- the same layout :mod:`iffl.loaders` already reads.
    """
    if n > len(pool):
        raise ValueError(f"requested {n} clients but pool has only {len(pool)}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), n, replace=False)
    idx.sort()

    coords = pool.coords[idx]
    dim = coords.shape[1]
    clients = pd.DataFrame({"id": idx})
    for k in range(dim):
        clients[f"d{k + 1}"] = coords[:, k]
    for k, col in enumerate(pool.groups.columns):
        clients[f"c{k + 1}"] = pool.groups[col].to_numpy()[idx]

    F = make_facilities(pool, coords, m, facility_mode, seed, exclude=idx)
    facilities = pd.DataFrame({f"d{k + 1}": F[:, k] for k in range(dim)})
    return clients, facilities
