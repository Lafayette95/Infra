"""Model versions: WHICH factors exist and WHICH release may load on which factor - the
only thing that differs between versions (a)-(d). The data preparation (panel.py) and
the estimation (dfm.py) are shared and never branch on the version name.

* ``a`` - one factor, every release loads on it.
* ``b`` - one factor per category (``block_level``), every release loads on EVERY factor
  (unrestricted); each factor starts from its category's first principal component, but
  nothing keeps it there - labels are a starting point, not a constraint (the loadings
  are only identified up to a rotation).
* ``c`` - same factors as (b), HARD blocks: a release's loadings on the factors of
  categories it is not in are fixed at exactly 0 (NY Fed-style block structure).
* ``d`` - same as (c) but SOFT: off-block loadings are free with a Gaussian prior centred
  on 0 (sd ``tau``, in units of the standardized series per unit-variance factor); the
  EM M-step becomes a ridge/MAP step. ``tau -> 0`` is (c), ``tau -> inf`` is (b).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from infra.config import MacroRelease

GLOBAL = "Global"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    blocks: bool  # False: one factor (a); True: one factor per category
    restriction: str = "none"  # "none" (b) | "hard" (c) | "soft" (d)
    tau: float = 0.1  # soft: prior sd of an off-block loading
    # Category source for the blocks: "sub" = the table's Activity_*/Price_* columns,
    # "cat1" = Activity/Price, "cat2" = Survey/Hard.
    block_level: str = "sub"
    global_factor: bool = False  # add a factor every release loads on (NY Fed: global + blocks)
    factor_lags: int = 1  # VAR order of the factors
    # "full": one VAR across all factors, correlated shocks. "independent": each factor its
    # own AR(p), uncorrelated shocks (Banbura & Modugno 2014 / the NY Fed's block-diagonal
    # transition) - with a global factor, blocks then carry only LOCAL co-movement,
    # orthogonal to the global cycle, which keeps per-block attribution interpretable.
    factor_dynamics: str = "full"
    idio: str = "ar1"  # idiosyncratic components: "ar1" (in the state, NY Fed) | "iid"
    gaussianize: bool = False  # append ``gauss`` to every ECDF transform
    # False: skip the table's transforms - releases enter in the table's UNITS (diff / pct
    # / yoy / level), standardized by the model, like the NY Fed's panel. For comparing.
    use_transforms: bool = True
    target: str = "GDP CQOQ Index"
    sample_start: str = "1990-01-01"  # first month of the estimation panel
    # Months the model never sees, every series (estimation AND nowcast/news filtering):
    # COVID's -28% / +35% SAAR GDP quarters otherwise dominate the EM fit - verified on
    # real data 2026-09-30 (unmasked: no convergence in 100 iterations, offsetting block
    # contributions of +-13pp). Inclusive month ranges.
    exclude: tuple[tuple[str, str], ...] = (("2020-03-01", "2021-06-01"),)
    max_iter: int = 100
    tol: float = 1e-5  # EM stops when the log-likelihood's relative change is below this

    def with_(self, **kw) -> "ModelSpec":
        return replace(self, **kw)


VERSIONS: dict[str, ModelSpec] = {
    "a": ModelSpec("a", blocks=False),
    "b": ModelSpec("b", blocks=True, restriction="none"),
    "c": ModelSpec("c", blocks=True, restriction="hard"),
    "d": ModelSpec("d", blocks=True, restriction="soft"),
}


def categories(release: MacroRelease, level: str) -> tuple[str, ...]:
    if level == "sub":
        return release.blocks
    if level == "cat1":
        return (release.cat1,)
    if level == "cat2":
        return (release.cat2,)
    raise ValueError(f"unknown block_level {level!r}")


@dataclass(frozen=True)
class FactorStructure:
    factors: tuple[str, ...]
    series: tuple[str, ...]  # release tickers, panel column order
    member: np.ndarray  # (n, r) bool: series i belongs to factor j's category
    free: np.ndarray  # (n, r) bool: loading estimated (False = fixed at 0)
    penalized: np.ndarray  # (n, r) bool: loading carries the soft prior

    def members(self, factor: str) -> list[str]:
        j = self.factors.index(factor)
        return [s for s, m in zip(self.series, self.member[:, j]) if m]


def factor_structure(spec: ModelSpec, releases: dict[str, MacroRelease], series: list[str]) -> FactorStructure:
    """The loading pattern for ``series`` (release tickers) under ``spec``."""
    if not spec.blocks:
        factors = [GLOBAL]
        member = np.ones((len(series), 1), bool)
    else:
        seen = [c for s in series for c in categories(releases[s], spec.block_level)]
        factors = ([GLOBAL] if spec.global_factor else []) + list(dict.fromkeys(seen))
        member = np.array([[f == GLOBAL or f in categories(releases[s], spec.block_level) for f in factors]
                           for s in series], bool)
    if spec.restriction == "hard":
        free, penalized = member.copy(), np.zeros_like(member)
    elif spec.restriction == "soft":
        free, penalized = np.ones_like(member), ~member
    elif spec.restriction == "none":
        free, penalized = np.ones_like(member), np.zeros_like(member)
    else:
        raise ValueError(f"unknown restriction {spec.restriction!r}")
    return FactorStructure(tuple(factors), tuple(series), member, free, penalized)
