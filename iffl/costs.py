"""Facility opening-cost synthesis.

Three of the six CSV datasets (``adult-500-20``, ``adult-429-65``, ``bank-500-20``)
ship facility files with coordinates only -- they were built for k-median, which has no
opening costs.  Facility location needs ``f_i``, so costs are synthesized for those and
the originals are left untouched.

Costs are expressed as a multiple of the instance's mean nearest-facility distance, so
one ``scale`` is comparable across datasets whose coordinates live on very different
numeric ranges (z-scored features vs. raw lat/lon).  Sweeping ``scale`` is what makes
the facility/connection tradeoff visible: small scale opens many facilities and the
connection term dominates; large scale opens few and the opening term dominates.
"""

from __future__ import annotations

import numpy as np

from iffl.instance import Instance

MODES = ("constant", "uniform_random")

#: Cost multiples swept by the experiment driver.
DEFAULT_SCALES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


def distance_unit(inst: Instance) -> float:
    """Mean nearest-facility distance -- the yardstick costs are expressed in."""
    return float(inst.nearest_facility_dist.mean())


def synthesize(
    inst: Instance,
    scale: float,
    mode: str = "constant",
    seed: int | None = 0,
) -> tuple[np.ndarray, dict]:
    """Build an opening-cost vector of shape ``(m,)``."""
    unit = distance_unit(inst)
    if mode == "constant":
        f = np.full(inst.m, scale * unit, dtype=float)
    elif mode == "uniform_random":
        rng = np.random.default_rng(seed)
        f = rng.uniform(0.5, 1.5, size=inst.m) * scale * unit
    else:
        raise KeyError(f"unknown cost mode {mode!r}; known: {', '.join(MODES)}")
    return f, {
        "cost_mode": mode,
        "cost_scale": scale,
        "cost_seed": seed,
        "cost_unit": unit,
        "cost_synthesized": True,
    }


def ensure_costs(
    inst: Instance,
    scale: float = 1.0,
    mode: str = "constant",
    seed: int | None = 0,
    force: bool = False,
) -> Instance:
    """Attach costs, keeping native ones unless ``force`` is set.

    Datasets with real costs (the synth pairs and both workbooks) keep them by default
    so results stay tied to the published data; ``force=True`` overrides so every
    dataset can be swept on the same cost scale.
    """
    native = bool(inst.meta.get("has_native_costs")) and np.any(inst.f > 0)
    if native and not force:
        return inst.with_costs(
            inst.f,
            cost_mode="native",
            cost_scale=None,
            cost_synthesized=False,
            cost_unit=distance_unit(inst),
        )
    f, meta = synthesize(inst, scale=scale, mode=mode, seed=seed)
    return inst.with_costs(f, **meta)
