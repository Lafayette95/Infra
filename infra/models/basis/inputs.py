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
    ms = ms_panel(spec, start, end) if spec.tier != "M0" and spec.ms_calendar else None
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
            if ms is not None:
                raw.meta["ms"] = ms_for_day(ms, day)
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


def ms_panel(spec: BasisSpec, start, end) -> dict:
    """Add-on MS: what the per-day bond states need - original issues (tenor, issue and
    announcement dates, each one's successor), the on-the-run map, the median issue cycle
    per tenor - and the event profiles per model YEAR (point in time: path observations
    before 1 January of the year, net of specialness carry; ``infra.pipeline.event_study``)."""
    from infra.pipeline.specialness import median_cycle_days, successors
    from infra.pipeline.treasury_otr import read_otr
    from infra.pipeline.treasury_ref import read_securities
    sec = read_securities()
    succ = successors(sec)
    ann = sec.drop_duplicates("cusip").set_index(sec.drop_duplicates("cusip")["cusip"].astype(str))["announced_date"]
    succ["next_cusip"] = succ.groupby("tenor")["cusip"].shift(-1)
    succ["next_announced"] = pd.to_datetime(succ["next_cusip"].map(ann))
    otr = read_otr(pd.Timestamp(start) - pd.Timedelta(days=10), pd.Timestamp(end))
    otr = otr[(otr["convention"] == "issue") & (otr["rank"] == 0)][["timestamp", "tenor", "cusip"]]
    return {"succ": succ.set_index("cusip"), "cycle": median_cycle_days(succ), "otr": otr, "profiles": {}}


def ms_for_day(ms: dict, day) -> dict:
    """The day's bond states (``cusip`` -> tenor, issue_date, is_otr, successor_issue: the
    successor's issue date once ANNOUNCED by the day, else issue + the tenor's median cycle)
    and the year's profiles."""
    from infra.pipeline.event_study import aging_profiles_as_of, net_profiles_as_of
    day = pd.Timestamp(day)
    year = day.year
    if year not in ms["profiles"]:
        start = pd.Timestamp(year=year, month=1, day=1)
        ev, bdays = net_profiles_as_of(start)
        ms["profiles"][year] = (pd.concat([ev, aging_profiles_as_of(start)], ignore_index=True), bdays)
    succ = ms["succ"]
    otr_today = set(ms["otr"].loc[ms["otr"]["timestamp"] == ms["otr"]["timestamp"][ms["otr"]["timestamp"] <= day].max(), "cusip"].astype(str))
    states = {}
    for c in succ.index[succ["issue_date"] <= day]:  # every tracked issue (aging applies to all)
        if c not in succ.index:
            continue
        r = succ.loc[c]
        known = pd.notna(r["next_announced"]) and r["next_announced"] <= day and pd.notna(r["next_issue"])
        nxt = r["next_issue"] if known else r["issue_date"] + pd.Timedelta(days=ms["cycle"].get(r["tenor"], 91))
        states[c] = {"tenor": r["tenor"], "issue_date": r["issue_date"], "is_otr": c in otr_today, "successor_issue": nxt}
    profiles, bdays = ms["profiles"][year]
    return {"states": states, "profiles": profiles, "bdays": bdays}