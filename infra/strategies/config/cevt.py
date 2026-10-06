"""CEVT strategy configurations (``infra.strategies.cevt``): event-study views from one or more
event-family RUNS (``infra.jobs.family_runs``: the walk-forward record of every code)."""
from __future__ import annotations

from dataclasses import dataclass, replace

from infra.strategies.base import DAILY, INTRADAY, StrategySpec, strategy_registry


@dataclass(frozen=True)
class CEVTSpec(StrategySpec):
    families: tuple[str, ...] = ()              # family RUN names (Database/Derived/ModelRuns/<name>)
    # t-signal: tanh(a * clip(t, -cap, cap)) / tanh(a * cap) - full strength (|1|) at the cap
    t_cap: float = 3.0
    t_curve: float = 0.5
    # which codes count: "fdr" (the family's point-in-time BH verdict at the row's fit), "passed"
    # (the code's own tests), "none"; failing codes are excluded, or zeros with include_failing
    gate: str = "fdr"
    include_failing: bool = False
    # aggregation of the views active at a label: within a family (AGGREGATORS: "smooth",
    # "threshold:0.8", "mean", "absmax"), then the mean over the ACTIVE families; pool_families
    # = one stage over every code of every family
    within: str = "smooth"
    pool_families: bool = False
    position_signal: str = "t"                  # the view positions are built from: "t" | "ev"
    scaling: str = "full_strength"              # sparse strategy (StrategySpec docstring)


STRUCTURES_UST = ("DUR__TY", "CURVE__FV__WN", "FLY__FV__UXY__WN", "FRONT__TU__H", "MICRO__TY__FV__H",
                  "MICRO__US__WN__H")

CEVT_STRATEGIES_INTRADAY: dict[str, CEVTSpec] = strategy_registry("intraday", (
    CEVTSpec("cevt_nfp", "NFP intraday family on ZT / ZN", families=("nfp",), instruments=("ZT.v.0", "ZN.v.0"),
             target_vol_usd=1_000_000.0, **INTRADAY),
    CEVTSpec("cevt_nfp_layers", "NFP intraday family on the curve structures, layered", families=("nfps",),
             instruments=STRUCTURES_UST, layers="ust_layers", **INTRADAY),
))

# DAILY: one decision a day just before the 14:00 CT settlement, traded at it (root CLAUDE.md 29)
CEVT_STRATEGIES_DAILY: dict[str, CEVTSpec] = strategy_registry("daily", (
    CEVTSpec("cevt_nfp_days_layers", "NFP day-windows on the curve structures (daily family nfp_days_struct)",
             families=("nfp_days_s",), instruments=STRUCTURES_UST, layers="ust_layers", **DAILY),
))

CEVT_STRATEGIES: dict[str, CEVTSpec] = {**CEVT_STRATEGIES_INTRADAY, **CEVT_STRATEGIES_DAILY}
assert len(CEVT_STRATEGIES) == len(CEVT_STRATEGIES_INTRADAY) + len(CEVT_STRATEGIES_DAILY), "a name in both"


def get_cevt_spec(spec: CEVTSpec | str, **overrides) -> CEVTSpec:
    base = CEVT_STRATEGIES[spec] if isinstance(spec, str) else spec
    return replace(base, **overrides) if overrides else base
