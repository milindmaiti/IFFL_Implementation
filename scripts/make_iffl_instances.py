#!/usr/bin/env python3
"""Materialize IFFL instances: radii and opening costs, plus a manifest.

Turns each dataset into concrete IFFL instances by attaching a radius vector and an
opening-cost vector, written as sidecar ``.npz`` files under ``instances/``.  The files
under ``data/`` are only read, never modified.

The manifest records the scheme, its parameters, the clamp count and the seed for every
instance, so a sweep can be reproduced -- or inspected for the degeneracies that bite on
these datasets -- without regenerating anything.

Usage:
    python scripts/make_iffl_instances.py --out instances/
    python scripts/make_iffl_instances.py --datasets uflp-49 synth-225-25
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iffl.configs import (  # noqa: E402
    COST_SCALES,
    DATASETS,
    RADIUS_GRID,
    Config,
    has_native_costs,
)
from iffl.pipeline import base_instance, build_instance  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--out", default="instances", type=Path)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for dataset in args.datasets:
        base = base_instance(dataset, args.data_dir)
        grid = (
            [("odd_cycle_unit", 1.0)]
            if dataset.startswith("oddcycle-")
            else list(RADIUS_GRID)
        )
        for (scheme, param), scale in itertools.product(grid, COST_SCALES):
            if scale is None and not has_native_costs(dataset):
                continue
            cfg = Config(dataset, scheme, param, scale)
            inst = build_instance(cfg, args.data_dir, base=base)
            stem = (
                f"{dataset}__{scheme}_{param}__"
                f"f_{'native' if scale is None else scale}"
            ).replace(" ", "")
            path = args.out / f"{stem}.npz"
            np.savez_compressed(path, R=inst.R, f=inst.f)
            manifest.append(
                {
                    "file": path.name,
                    "dataset": dataset,
                    "n": inst.n,
                    "m": inst.m,
                    "radius_scheme": inst.meta.get("radius_scheme", scheme),
                    "radius_param": inst.meta.get("radius_param", param),
                    "radius_rank": inst.meta.get("radius_rank", 1),
                    "radius_uniform": bool(inst.meta.get("radius_uniform", False)),
                    "radius_degenerate": bool(inst.meta.get("radius_degenerate", False)),
                    "radius_clamped": int(inst.meta.get("radius_clamped", 0)),
                    "R_min": float(inst.R.min()),
                    "R_max": float(inst.R.max()),
                    "cost_mode": inst.meta.get("cost_mode"),
                    "cost_scale": scale,
                    "cost_seed": inst.meta.get("cost_seed"),
                    "f_min": float(inst.f.min()),
                    "f_max": float(inst.f.max()),
                    "n_edges": int(inst.admissible_edges()[0].size),
                }
            )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    degenerate = sum(1 for r in manifest if r["radius_degenerate"])
    clamped = sum(1 for r in manifest if r["radius_clamped"] > 0)
    print(f"wrote {len(manifest)} instances to {args.out}/")
    print(f"  degenerate radii: {degenerate}   with clamped clients: {clamped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
