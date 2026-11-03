"""Instance representation for Individually Fair Facility Location.

Distance convention: ``d[j, i]`` is the distance from client ``j`` to facility ``i``
(client row, facility column).  This mirrors ``distMatr[j][i]`` in the reference Julia
code (``Fair-FL/testing.jl``) and is used consistently everywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial.distance import cdist


def l2_matrix(client_pts: np.ndarray, facility_pts: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances, shape ``(n_clients, n_facilities)``.

    Equivalent to the ``l2`` helper in ``Fair-FL/fl-lp-pd-new.jl`` applied to every
    client/facility pair.  ``cdist`` is used rather than the usual
    ``|a|^2 + |b|^2 - 2 a.b`` expansion because that expansion cancels catastrophically
    when coordinates are large relative to their spread: on the UFLP workbooks, whose
    raw lat/lon values sit near 38 and 121, it returns a self-distance around 1e-6
    instead of 0.  That matters beyond tidiness -- a nonzero self-distance perturbs
    every radius scheme keyed on the nearest-facility distance.
    """
    a = np.asarray(client_pts, dtype=float)
    b = np.asarray(facility_pts, dtype=float)
    return cdist(a, b, metric="euclidean")


@dataclass
class Instance:
    """One IFFL instance.

    Attributes:
        name: dataset identifier, e.g. ``"synth-225-25"``.
        client_pts: ``(n, dim)`` client coordinates.
        facility_pts: ``(m, dim)`` candidate facility coordinates.  For the UFLP
            workbooks this is the same point set as ``client_pts``.
        f: ``(m,)`` facility opening costs.
        d: ``(n, m)`` distance matrix, ``d[j, i]`` = client ``j`` to facility ``i``.
        groups: protected attribute columns, kept for analysis only -- the algorithms
            here enforce *individual*, not group, fairness.
        R: ``(n,)`` per-client fairness radii, or ``None`` before a radius scheme runs.
        meta: provenance (radius scheme, cost mode, seeds, clamp counts).
    """

    name: str
    client_pts: np.ndarray
    facility_pts: np.ndarray
    f: np.ndarray
    d: np.ndarray
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    R: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.d.shape[0]

    @property
    def m(self) -> int:
        return self.d.shape[1]

    @property
    def nearest_facility_dist(self) -> np.ndarray:
        """``(n,)`` distance from each client to its closest candidate facility.

        This is the feasibility floor for any radius assignment: a client with
        ``R_j`` below this value has no admissible edge and makes the LP infeasible.
        """
        return self.d.min(axis=1)

    def with_radii(self, R: np.ndarray, **meta) -> "Instance":
        """Return a copy carrying radii ``R`` and merged metadata."""
        R = np.asarray(R, dtype=float)
        if R.shape != (self.n,):
            raise ValueError(f"R must have shape ({self.n},), got {R.shape}")
        return replace(self, R=R, meta={**self.meta, **meta})

    def with_costs(self, f: np.ndarray, **meta) -> "Instance":
        """Return a copy carrying opening costs ``f`` and merged metadata."""
        f = np.asarray(f, dtype=float)
        if f.shape != (self.m,):
            raise ValueError(f"f must have shape ({self.m},), got {f.shape}")
        return replace(self, f=f, meta={**self.meta, **meta})

    def check_feasible(self) -> None:
        """Raise if any client has no facility within its radius."""
        if self.R is None:
            raise ValueError("instance has no radii; call with_radii() first")
        bad = np.flatnonzero(self.R < self.nearest_facility_dist - 1e-12)
        if bad.size:
            j = int(bad[0])
            raise ValueError(
                f"{self.name}: {bad.size} client(s) have no facility within radius; "
                f"e.g. client {j} has R={self.R[j]:.6g} but nearest facility is "
                f"{self.nearest_facility_dist[j]:.6g} away"
            )

    def admissible_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(clients, facilities)`` index arrays for ``E_R``.

        ``E_R = {(i, j) : d(i, j) <= R_j}`` -- the edge set both papers restrict to.
        """
        if self.R is None:
            raise ValueError("instance has no radii; call with_radii() first")
        js, is_ = np.nonzero(self.d <= self.R[:, None] + 1e-12)
        return js, is_

    def summary(self) -> str:
        parts = [f"{self.name}: n={self.n} m={self.m} dim={self.client_pts.shape[1]}"]
        parts.append(f"f in [{self.f.min():.4g}, {self.f.max():.4g}]")
        if self.R is not None:
            parts.append(f"R in [{self.R.min():.4g}, {self.R.max():.4g}]")
            js, _ = self.admissible_edges()
            parts.append(f"|E_R|={js.size}")
        if self.groups:
            parts.append("groups=" + ",".join(self.groups))
        return "  ".join(parts)
