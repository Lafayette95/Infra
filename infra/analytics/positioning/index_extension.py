"""Month-end duration extension of a Treasury index, computed from its rules (pure).

Index trackers hold the index's membership through the month and switch to the next
month's at the month-end close: bonds issued (settled) by then enter, bonds now inside
``min_years`` of maturity leave, reopenings add to an amount. The switch changes the
index's duration - the EXTENSION - and passive money must buy that much duration at the
close: a forced, rules-based flow (TOFIX "Positioning: roadmap").

Rules here are the Bloomberg US Treasury index's published ones as we read them: fixed-rate
nominal notes and bonds, >= 1 year to maturity, amount outstanding >= $300m (every coupon
issue qualifies), market-value weights, new issues entering once settled by month-end. NOT
modelled: TIPS / FRNs / bills (out of scope anyway), whether the index nets out the Fed's
holdings (we use the full issue, `total_accepted`), buybacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def modified_duration(settle: pd.Timestamp, maturity: pd.Series, coupon_pct: pd.Series, yield_pct: pd.Series,
                      freq: int = 2) -> pd.Series:
    """Modified duration (years) of fixed-coupon bonds settling on ``settle``: cash flows on
    the coupon dates counted back from maturity, a fractional first period (street
    convention, equal periods), discounted at the bond's own yield."""
    out = np.full(len(maturity), np.nan)
    for i, (m, c, y) in enumerate(zip(pd.to_datetime(maturity), coupon_pct.to_numpy(float), yield_pct.to_numpy(float))):
        if not (np.isfinite(c) and np.isfinite(y)) or m <= settle:
            continue
        dates, d = [], m
        while d > settle:
            dates.append(d)
            d = d - pd.DateOffset(months=12 // freq)
        dates = dates[::-1]
        prev = d
        frac = (dates[0] - settle).days / max((dates[0] - prev).days, 1)   # fraction of the first period left
        t = frac + np.arange(len(dates))                                     # in periods
        cf = np.full(len(dates), c / freq)
        cf[-1] += 100.0
        disc = (1 + y / 100 / freq) ** (-t)
        pv = cf * disc
        mac = (pv * t).sum() / pv.sum() / freq
        out[i] = mac / (1 + y / 100 / freq)
    return pd.Series(out, index=maturity.index)


def members(securities: pd.DataFrame, amounts: pd.DataFrame, as_of: pd.Timestamp, *, min_years: float = 1.0,
            min_amount: float = 300e6) -> pd.DataFrame:
    """The index membership set at the ``as_of`` rebalance: issued by then, >= ``min_years``
    to maturity, outstanding >= ``min_amount`` (amounts = auctions settled by ``as_of``)."""
    sec = securities[securities["security_type"].astype(str).isin(["Note", "Bond"])]
    sec = sec[(pd.to_datetime(sec["issue_date"]) <= as_of)
              & (pd.to_datetime(sec["maturity_date"]) >= as_of + pd.DateOffset(days=int(round(365.25 * min_years))))]
    amt = amounts[pd.to_datetime(amounts["issue_date"]) <= as_of].groupby("cusip")["amount"].sum()
    out = sec.set_index("cusip").join(amt.rename("amount"), how="inner")
    return out[out["amount"] >= min_amount]


def index_duration(memb: pd.DataFrame, prices: pd.DataFrame, settle: pd.Timestamp) -> tuple[float, float, int]:
    """Market-value-weighted modified duration of a membership at ``prices`` (one day:
    ``cusip``, ``price_eod``, ``accrued``, ``yield_eod``). Returns (duration, market value $bn,
    bonds priced)."""
    p = prices.set_index("cusip")[["price_eod", "accrued", "yield_eod"]]
    m = memb.join(p, how="inner").dropna(subset=["price_eod", "yield_eod"])
    if m.empty:
        return np.nan, np.nan, 0
    mv = m["amount"] * (m["price_eod"] + m["accrued"].fillna(0)) / 100
    dur = modified_duration(settle, m["maturity_date"], m["coupon"], m["yield_eod"])
    ok = dur.notna()
    return float((mv[ok] * dur[ok]).sum() / mv[ok].sum()), float(mv[ok].sum() / 1e9), int(ok.sum())


def extension(securities: pd.DataFrame, amounts: pd.DataFrame, prices: pd.DataFrame, month_end: pd.Timestamp,
              prior_month_end: pd.Timestamp, **rules) -> dict:
    """Duration of next month's membership minus the current one's, both at ``month_end``
    prices: the extension passive trackers buy at the close."""
    old = members(securities, amounts, prior_month_end, **rules)
    new = members(securities, amounts, month_end, **rules)
    d_old, mv_old, n_old = index_duration(old, prices, month_end)
    d_new, mv_new, n_new = index_duration(new, prices, month_end)
    return {"month_end": month_end, "duration_old": d_old, "duration_new": d_new, "extension": d_new - d_old,
            "n_old": n_old, "n_new": n_new, "mv_old_bn": mv_old, "mv_new_bn": mv_new,
            "entering": len(set(new.index) - set(old.index)), "leaving": len(set(old.index) - set(new.index))}
