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
    """A positive length scale for the instance, used as the unit for opening costs.

    Normally the mean nearest-facility distance.  On the UFLP workbooks, though, clients
    and facilities are the same point set, so that mean is exactly 0 and every
    synthesized cost would come out 0 -- the LP would then open all m facilities for
    free and report an objective of 0, making every cost ratio a 0/0 form.  Falling back
    to the next facility rank keeps the unit positive and the sweep meaningful.
    """
    from iffl.radii import facility_distance_basis

    for rank in range(1, min(inst.m, 5) + 1):
        unit = float(facility_distance_basis(inst, rank).mean())
        if unit > 0:
            return unit
    # Degenerate fallback: all candidate facilities coincide with all clients.
    positive = inst.d[inst.d > 0]
    return float(positive.mean()) if positive.size else 1.0


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
