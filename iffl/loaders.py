"""Dataset loaders for the eight Fair-FL datasets.

The six CSV pairs and two UFLP workbooks differ in ways that matter:

* ``adult-500-20`` and ``bank-500-20`` are *space*-delimited; the rest are comma
  delimited.  Delimiter sniffing handles both.
* Feature width varies (6/6/3/2 dims) and so does the number of protected attribute
  columns (2/2/4/1), so columns are matched by pattern rather than by position.
* Only ``synth-*-fac.csv`` carries a facility cost column (``c``).  The adult and bank
  facility files have none, so costs must be synthesized -- see :mod:`iffl.costs`.
* In the UFLP workbooks clients and facilities are the *same* point set (n == m) and
  the distance matrix is symmetric.

Formulas mirror the reference Julia implementation so numbers stay comparable:
plain L2 over the ``d*`` columns, and ``fixed cost / 10000`` for workbook costs.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from iffl.instance import Instance, l2_matrix

#: CSV datasets, as (client file stem, facility file stem).
CSV_DATASETS = (
    "adult-500-20",
    "adult-429-65",
    "bank-500-20",
    "synth-225-25",
    "synth-500-40",
    "synth-500-50",
)

#: UFLP workbooks, mapping dataset name to (filename, row count).
XLSX_DATASETS = {
    "uflp-49": ("data49UFLP.xlsx", 49),
    "uflp-88": ("data88UFLP.xlsx", 88),
}

ALL_DATASETS = CSV_DATASETS + tuple(XLSX_DATASETS)

_DIM_RE = re.compile(r"^d\d+$")
_GROUP_RE = re.compile(r"^c\d+$")


def default_data_dir() -> Path:
    """The sibling ``Fair-FL`` checkout, which is only ever read, never written."""
    return (Path(__file__).resolve().parent.parent.parent / "Fair-FL").resolve()


def _read_table(path: Path) -> pd.DataFrame:
    """Read a whitespace- or comma-delimited table, sniffing the separator."""
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] == 1:
        # Sniffing fell back to a single column; force whitespace splitting.
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    return df


def _dim_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted(
        (c for c in df.columns if _DIM_RE.match(str(c))),
        key=lambda c: int(str(c)[1:]),
    )
    if not cols:
        raise ValueError(f"no d1..dk feature columns found in {list(df.columns)}")
    return cols


def load_csv_dataset(name: str, data_dir: Path | str | None = None) -> Instance:
    """Load one ``<name>.csv`` / ``<name>-fac.csv`` pair.

    Every client row is kept, including the ``out``-labelled synthetic points: those
    are real clients whose ``c1`` value is just a group tag, which is how
    ``processSynth`` in the reference Julia code treats them.
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    clients = _read_table(root / f"{name}.csv")
    facs = _read_table(root / f"{name}-fac.csv")

    cdims = _dim_columns(clients)
    fdims = _dim_columns(facs)
    if cdims != fdims:
        raise ValueError(f"{name}: client dims {cdims} != facility dims {fdims}")

    client_pts = clients[cdims].to_numpy(dtype=float)
    facility_pts = facs[fdims].to_numpy(dtype=float)

    # Only the synth facility files carry a cost column; zeros are a placeholder that
    # iffl.costs replaces, and run_experiments refuses to run on an all-zero f.
    if "c" in facs.columns:
        f = facs["c"].to_numpy(dtype=float)
        has_costs = True
    else:
        f = np.zeros(len(facility_pts))
        has_costs = False

    groups = {
        str(c): clients[c].to_numpy()
        for c in clients.columns
        if _GROUP_RE.match(str(c))
    }

    return Instance(
        name=name,
        client_pts=client_pts,
        facility_pts=facility_pts,
        f=f,
        d=l2_matrix(client_pts, facility_pts),
        groups=groups,
        meta={"source": "csv", "has_native_costs": has_costs},
    )


def load_xlsx_dataset(name: str, data_dir: Path | str | None = None) -> Instance:
    """Load one UFLP workbook (``data49UFLP.xlsx`` / ``data88UFLP.xlsx``).

    Both workbooks put their data rows from row 3 onward, with columns
    ``index, demand, emerg cost, failable, fixed cost, lat, lon, ...``.  Clients and
    facilities are the same point set, distances are Euclidean on raw (lat, lon), and
    opening costs are ``fixed cost / 10000`` -- all matching ``process49``/``process88``.
    """
    from openpyxl import load_workbook

    if name not in XLSX_DATASETS:
        raise KeyError(f"unknown workbook dataset {name!r}")
    filename, n_rows = XLSX_DATASETS[name]
    root = Path(data_dir) if data_dir is not None else default_data_dir()

    wb = load_workbook(root / filename, data_only=True, read_only=True)
    ws = wb["data"] if "data" in wb.sheetnames else wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(min_row=3, max_row=2 + n_rows, values_only=True))
    wb.close()

    if len(rows) != n_rows:
        raise ValueError(f"{name}: expected {n_rows} data rows, read {len(rows)}")

    fixed_cost = np.array([r[4] for r in rows], dtype=float)
    pts = np.array([[r[5], r[6]] for r in rows], dtype=float)
    # Trailing label column: political lean for the 49, region code for the 88.
    label = np.array([("" if r[9] is None else r[9]) if len(r) > 9 else "" for r in rows])
    if not label.any():
        label = np.array([("" if r[7] is None else r[7]) if len(r) > 7 else "" for r in rows])

    return Instance(
        name=name,
        client_pts=pts,
        facility_pts=pts,
        f=fixed_cost / 10000.0,
        d=l2_matrix(pts, pts),
        groups={"c1": label},
        meta={"source": "xlsx", "has_native_costs": True, "clients_eq_facilities": True},
    )


def load_dataset(name: str, data_dir: Path | str | None = None) -> Instance:
    """Load any of the eight datasets by name."""
    if name in XLSX_DATASETS:
        return load_xlsx_dataset(name, data_dir)
    if name in CSV_DATASETS:
        return load_csv_dataset(name, data_dir)
    raise KeyError(f"unknown dataset {name!r}; known: {', '.join(ALL_DATASETS)}")


def load_all(data_dir: Path | str | None = None) -> dict[str, Instance]:
    """Load every dataset, keyed by name."""
    return {name: load_dataset(name, data_dir) for name in ALL_DATASETS}
