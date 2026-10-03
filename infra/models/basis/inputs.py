"""Reading stored data for the basis models - the only module here that touches storage,
through ``infra.pipeline.futures_basis`` (disk only, never fetches)."""
from __future__ import annotations

from typing import Iterator

import pandas as pd

from infra.models.basis.config import BasisSpec
from infra.pipeline.bond_yields import read_bond_yields
from infra.pipeline.futures_basis import BasisDay, BasisInputs
from infra.pipeline.specialness import successors
from infra.pipeline.treasury_prices import read_prices
from infra.pipeline.treasury_ref import read_securities


def basis_days(spec: BasisSpec, start, end, *, contracts: str = "quoted", days=None) -> Iterator[BasisDay]:
    """One ``BasisDay`` per business day in ``[start, end]`` with cash prices and baskets.
    Tiers above M0 also get ``meta["levels"]``: the CMT par yields (%) at each root's
    ``level_tenor``, published by the end of the day (point in time)."""
    src = BasisInputs(start, end, roots=spec.roots, cash_mid_frac=spec.cash_mid_frac,
                      funding_model=spec.funding_model, contracts=contracts)
    levels = level_history(spec, pd.Timestamp(start) - pd.Timedelta(days=spec.vol_history_days), end) \
        if spec.tier != "M0" else None
    panel = yield_panel(spec, start, end) if spec.tier not in ("M0", "M1") else None
    iv = iv_panel(spec, start, end) if spec.tier != "M0" and spec.level_vol_source == "iv" else None
    only = None if days is None else set(pd.DatetimeIndex(days).normalize())  # a sample: skip the rest
    for day in src.days:
        if only is not None and day not in only:
            continue
        if pd.Timestamp(start) <= day <= pd.Timestamp(end):
            raw = src.day(day)
            if levels is not None:
                raw.meta["levels"] = levels[levels.index <= day]
            if panel is not None:
                y = panel["yields"]
                raw.meta.update({"yields": y[y.index <= day].tail(spec.factor_window + 1),
                                 "predecessor": panel["predecessor"], "maturity": panel["maturity"]})
            if iv is not None:
                raw.meta["iv"] = iv_for_day(iv, day)
            yield raw


def iv_panel(spec: BasisSpec, start, end) -> dict:
    """Add-on IV: ``iv_root``'s at-the-money vols (per day x expiry) and the EWMA of its
    front contract's daily moves (points/day), over the range."""
    from infra.pipeline.futures_iv import atm_iv, front_ewma_points
    return {"atm": atm_iv(spec.iv_root, start, end),
            "ewma_pts": front_ewma_points(spec.iv_root, start, end, lam=spec.vol_lambda)}


def iv_for_day(iv: dict, day) -> dict:
    """The day's slice of ``iv_panel`` (point in time: that day's settlements)."""
    day = pd.Timestamp(day)
    a = iv["atm"]
    return {"atm": a[a["timestamp"] == day], "ewma_pts": iv["ewma_pts"].get(day, float("nan"))}


def yield_panel(spec: BasisSpec, start, end) -> dict:
    """M2+: END OF DAY yields (%) of every note and bond, wide by day, from ``factor_window``
    business days (~ x 1.5 calendar) before ``start``; each tracked issue's predecessor (the
    previous original issue of the same type and term) and every bond's maturity."""
    first = pd.Timestamp(start) - pd.Timedelta(days=int(spec.factor_window * 1.5) + 30)
    p = read_prices(first, pd.Timestamp(end) + pd.Timedelta(days=1))
    p = p[p["security_type"].isin(["Note", "Bond"])]
    yields = p.pivot_table(index="timestamp", columns="cusip", values="yield_eod", observed=True).sort_index()
    yields.columns = yields.columns.astype(str)
    sec = read_securities()
    succ = successors(sec).sort_values(["tenor", "issue_date"])
    predecessor = {}
    for _, g in succ.groupby("tenor"):
        cus = list(g["cusip"])
        predecessor.update({c: (cus[i - 1] if i > 0 else None) for i, c in enumerate(cus)})
    maturity = dict(zip(sec["cusip"].astype(str), pd.to_datetime(sec["maturity_date"])))
    return {"yields": yields, "predecessor": predecessor, "maturity": maturity}


def level_history(spec: BasisSpec, start, end) -> pd.DataFrame:
    """CMT par yields (%), one column per root (its ``level_tenor``), by day."""
    tenors = dict(spec.level_tenor)
    tickers = sorted({f"US_BOND_{t}y" for t in tenors.values()})
    y = read_bond_yields(tickers, pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), source="cmt")
    wide = y.pivot_table(index="timestamp", columns="ticker", values="yield")
    return pd.DataFrame({root: wide.get(f"US_BOND_{t}y") for root, t in tenors.items()}).sort_index()
