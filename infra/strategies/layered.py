"""Layered sizing: views on curve STRUCTURES -> contracts per future (root CLAUDE.md 28;
``infra/strategies/CLAUDE.md`` 5). Used by any strategy whose ``StrategySpec.layers`` names a
``LAYER_SIZINGS`` entry: its views' columns are then structure names (``DUR__TY``,
``CURVE__FV__WN``, ...), not futures.

Per label, with the state of its CME trading day (known before the day: hedge betas, structure
vols, the legs' covariance, contract DV01s - ``infra.pipeline.structures``):
1. structure position ($/bp) = view x layer budget / sqrt(structures in the layer) / the
   structure's annual bp vol;
2. leg exposures = W' x positions, NETTED across structures and layers;
3. the netted book's vol with the full leg covariance; above ``portfolio_cap`` everything is
   scaled down (``scale``); standalone vol per layer and their sum are kept for diagnostics
   (sum > total = the layers diversify, sum < total = they add up);
4. contracts = exposure / contract DV01. A day without a complete state (hedge warm-up, a
   missing DV01) uses the latest complete one before it; plan days past the data use the last.
   The columns are relative futures, so ``to_absolute``
   and the accounting work unchanged.
"""
from __future__ import annotations

import pandas as pd

from infra.analytics.structures import size_layers
from infra.pipeline.structures import StructureState, structure_state
from infra.reference.structures import STRUCTURE_SETS
from infra.strategies.config.layers import get_layer_sizing
from infra.trading_calendar import trading_day


def decision_days(labels: pd.DatetimeIndex) -> pd.Series:
    """Each label's CME trading day (the state it uses)."""
    return pd.Series(pd.DatetimeIndex(trading_day(pd.DatetimeIndex(labels), "GLBX.MDP3")).normalize(), index=labels)


def layered_positions(views: pd.DataFrame, spec_name: str, *, state: StructureState | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(contracts per relative future, diagnostics) for structure views (labels x structures)."""
    spec = get_layer_sizing(spec_name)
    sset = STRUCTURE_SETS[spec.structure_set]
    unknown = [c for c in views.columns if c not in sset.structures]
    if unknown:
        raise KeyError(f"views on {unknown}: not structures of set {sset.name!r} ({list(sset.structures)})")
    days = decision_days(views.index)
    if state is None:
        state = structure_state(sset, days.min(), days.max(), vol_span=spec.vol_span, cov_span=spec.cov_span)
    # each decision day uses the latest day with a COMPLETE state at or before it: the day itself
    # in history; for plan days past the stored data, the last state (provisional)
    ready = pd.DatetimeIndex(sorted(d for d in state.weights if d in state.cov
                                    and state.dv01 is not None and d in state.dv01.index
                                    and state.dv01.loc[d].notna().all()))
    pos = ready.searchsorted(pd.DatetimeIndex(days.to_numpy()), side="right") - 1
    mapped = pd.Series(pd.NaT, index=days.index, dtype="datetime64[ns]")
    ok = pos >= 0
    mapped[ok] = ready.to_numpy()[pos[ok]]
    days = mapped
    layers = sset.layers()
    sizing = size_layers(views, days, layers, spec.budgets, state.sigma, state.weights, state.cov, state.dv01,
                         spec.portfolio_cap)
    diag = sizing.diagnostics.join(sizing.positions.add_prefix("usd_dv01:"))
    return sizing.contracts, diag
