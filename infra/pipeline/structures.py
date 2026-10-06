"""Curve structures' data, point in time per DECISION trading day (root CLAUDE.md 28). Disk only.

* ``daily_bp_moves(legs, start, end)``: each relative future's daily move in bp = the
  back-adjusted settlement change (on the contract held that day) x point value / that
  contract's DV01 on the PRIOR trading day (bmk risk store) - price-return per unit DV01,
  + = rallied. Known at the day's close.
* ``structure_state(set, start, end)``: for every decision day D (a CME trading day; intraday
  labels map to theirs by ``trading_day``), what is known BEFORE D starts:
  - ``weights[D]``: structures x legs DV01 weights; hedged structures use betas fitted on moves
    through D-1 (``hedge_window`` rows);
  - ``moves``: each structure's daily bp move on D, with D's weights (its realised P&L per unit);
  - ``sigma``: annual bp vol of each structure (EWMA of ``moves`` through D-1, ``vol_span``);
  - ``cov[D]``: the legs' annual bp covariance (EWMA through D-1);
  - ``dv01``: per leg, the DV01 ($/bp per contract) of the contract the leg maps to ON D, as of
    D-2 (D-1's DV01 needs FedInvest's END OF DAY, posted ~10:00 New York on D - after the
    evening session of D has started).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from infra.analytics import structures as an
from infra.config import BMK_ROOT, FUTURES_ROOTS
from infra.pipeline.event_pnl import contract_map, dv01_prior, root_of
from infra.reference.structures import STRUCTURE_SETS, STRUCTURES, StructureSet

ANNUAL = an.ANNUAL


def daily_bp_moves(legs: list[str], start, end, *, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trading day x leg daily bp moves (see the module docstring)."""
    if panel is None:
        from infra.pipeline.series_panel import read_panel
        panel = read_panel([f"fut:{leg}" for leg in legs], start, end)
    out = {}
    for leg in legs:
        lvl = panel[f"fut:{leg}"].dropna()
        days = pd.DatetimeIndex(lvl.index).normalize()
        con = contract_map(leg, days)
        dv = dv01_prior(con, pd.Series(days, index=days), risk_root=BMK_ROOT / "Risk")
        out[leg] = pd.Series(lvl.diff().to_numpy() * FUTURES_ROOTS[root_of(leg)].point_value / dv, index=days)
    return pd.DataFrame(out).sort_index()


def decision_dv01(legs: list[str], days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(contracts, dv01): per decision day D and leg, the contract mapped on D and its DV01 as of
    D-2 (two trading days back in ``days``)."""
    days = pd.DatetimeIndex(days)
    prev = pd.Series(days, index=days).shift(1)
    cons, dvs = {}, {}
    for leg in legs:
        con = contract_map(leg, days)
        cons[leg] = con
        dvs[leg] = pd.Series(dv01_prior(con.ffill(), prev.fillna(days[0] - pd.Timedelta(days=1)),
                                        risk_root=BMK_ROOT / "Risk"), index=days)
    return pd.DataFrame(cons), pd.DataFrame(dvs)


@dataclass
class StructureState:
    set: StructureSet
    days: pd.DatetimeIndex
    leg_moves: pd.DataFrame
    moves: pd.DataFrame
    sigma: pd.DataFrame
    weights: dict = field(default_factory=dict)
    cov: dict = field(default_factory=dict)
    dv01: pd.DataFrame | None = None
    contracts: pd.DataFrame | None = None
    betas: dict = field(default_factory=dict)       # hedged structure -> days x targets (as used on D)

    def W(self, day) -> pd.DataFrame:
        return self.weights[pd.Timestamp(day)]


def structure_state(structure_set: str | StructureSet, start, end, *, vol_span: int = 60, cov_span: int = 120,
                    history_days: int = 900, leg_moves: pd.DataFrame | None = None,
                    with_dv01: bool = True) -> StructureState:
    """See the module docstring. ``leg_moves`` (trading day x leg) may be passed in."""
    sset = STRUCTURE_SETS[structure_set] if isinstance(structure_set, str) else structure_set
    legs = sset.legs()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if leg_moves is None:
        leg_moves = daily_bp_moves(legs, start - pd.Timedelta(days=history_days), end)
    lm = leg_moves.reindex(columns=legs).dropna(how="all")
    days = pd.DatetimeIndex(lm.index)
    base = {s: pd.Series(STRUCTURES[s].base_weights()).reindex(legs).fillna(0.0) for s in sset.structures}
    plain = [s for s in sset.structures if STRUCTURES[s].hedge_layer is None]
    moves = pd.DataFrame(index=days)
    for s in plain:
        moves[s] = lm @ base[s]
    weights_by_s = {s: pd.DataFrame([base[s]] * len(days), index=days) for s in plain}
    betas = {}
    for s in sset.structures:
        if s in plain:
            continue
        targets = sset.hedge_targets(s)
        y = lm @ base[s]
        b = an.rolling_betas(y, moves[targets], sset.hedge_window, sset.hedge_min_obs).shift(1)  # known before D
        betas[s] = b
        T = pd.DataFrame({t: base[t] for t in targets}).T
        w = base[s].to_numpy()[None, :] - b.fillna(np.nan).to_numpy() @ T.to_numpy()
        weights_by_s[s] = pd.DataFrame(w, index=days, columns=legs)
        moves[s] = y - (b * moves[targets]).sum(axis=1, min_count=len(targets))
    sigma = moves.ewm(span=vol_span, min_periods=max(vol_span // 2, 20)).std().shift(1) * np.sqrt(ANNUAL)
    covs = (lm.ewm(span=cov_span, min_periods=max(cov_span // 2, 30)).cov() * ANNUAL)
    keep = days[(days >= start) & (days <= end)]
    cov, weights = {}, {}
    prev_day = pd.Series(days, index=days).shift(1)
    for d in keep:
        p = prev_day.get(d)
        if pd.notna(p) and p in covs.index.get_level_values(0):
            c = covs.loc[p]
            if not c.isna().any().any():
                cov[d] = c
        W = pd.DataFrame({s: weights_by_s[s].loc[d] for s in sset.structures}).T
        if not W.isna().any().any():
            weights[d] = W
    state = StructureState(sset, keep, lm, moves, sigma, weights, cov, betas=betas)
    if with_dv01:
        state.contracts, state.dv01 = decision_dv01(legs, keep)
    return state
