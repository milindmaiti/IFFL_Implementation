"""The experiment grid, in one place so scripts and tests agree on it."""

from __future__ import annotations

from dataclasses import dataclass

from iffl.loaders import ALL_DATASETS, XLSX_DATASETS

#: Odd-cycle sizes added to the sweep.  Every real dataset has an integral LP, so these
#: are the only instances that exercise Approach A's fractional machinery.
SYNTHETIC_DATASETS = ("oddcycle-5", "oddcycle-9", "oddcycle-15")

DATASETS = tuple(ALL_DATASETS) + SYNTHETIC_DATASETS

#: (scheme, param) pairs.
#:
#: The two multiplier schemes do not sweep the same range, because they do not behave
#: the same way.  A uniform radius is forced by feasibility to be at least
#: ``max_j d(j, F)``, set by the single worst-placed client, and under randomly drawn
#: facilities that client sits about ten times further out than the median.  The radius
#: is then already loose enough at ``mult = 1`` to admit 84% of all client-facility
#: pairs, and by ``mult = 2`` it admits 99% -- the constraint stops binding, successive
#: values give the same instance, and the LPs reach 1.6M variables.  Uniform radii are
#: therefore swept only where they still restrict anything; the nonuniform schemes,
#: which key off each client's own geometry, are unaffected and keep the full range.  Nonuniform schemes are the regime Approach A is built for;
#: the uniform ones are the only regime where Approach B's bound is proved.
RADIUS_GRID = (
    ("nearest_facility_multiple", 1.0),
    ("nearest_facility_multiple", 1.5),
    ("nearest_facility_multiple", 2.0),
    ("nearest_facility_multiple", 3.0),
    ("nearest_facility_multiple", 4.0),
    ("uniform", 1.0),
    ("uniform", 1.5),
    # Chosen so each lands on roughly the same effective radius multiplier,
    # median_j(R_j / d(j, F)), as a value of the nearest-facility sweep: k = 150, 50, 20
    # and 10 measure about 1.5, 2.1, 3.0 and 3.8.  The two schemes then differ in radius
    # *heterogeneity* at matched tightness rather than in tightness itself.
    ("jkl_client_knn", 150),
    ("jkl_client_knn", 50),
    ("jkl_client_knn", 20),
    ("jkl_client_knn", 10),
)

#: Synthesized cost multiples.  ``None`` means "use the dataset's own costs", which
#: only the synth pairs and the two workbooks have; it keeps results tied to the
#: published data, while the numeric scales make all datasets directly comparable.
COST_SCALES = (None, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


def has_native_costs(dataset: str) -> bool:
    """True for datasets whose facility file carries real opening costs."""
    return not dataset.startswith(("adult-", "bank-"))


def radius_rank_for(dataset: str) -> int:
    """Which facility rank a distance-based radius scheme should key on.

    The UFLP workbooks use the same point set for clients and facilities, so a rank-1
    basis is identically zero and collapses every radius to 0.  Rank 2 is used there so
    the instances stay non-trivial; see :func:`iffl.radii.facility_distance_basis`.
    """
    return 2 if dataset in XLSX_DATASETS else 1


@dataclass(frozen=True)
class Config:
    """One point of the sweep."""

    dataset: str
    scheme: str
    param: float
    cost_scale: float | None
    cost_mode: str = "constant"

    @property
    def native_costs(self) -> bool:
        """``cost_scale is None`` selects the dataset's own opening costs."""
        return self.cost_scale is None

    @property
    def label(self) -> str:
        f = "native" if self.native_costs else self.cost_scale
        return f"{self.dataset}|{self.scheme}={self.param}|f={f}"
