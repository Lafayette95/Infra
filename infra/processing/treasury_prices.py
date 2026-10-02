"""Treasury prices per CUSIP (CLAUDE.md 18): a FedInvest page -> typed prices, accrued
interest and yields. Pure functions, no I/O.

* Prices are CLEAN, per 100 face. END OF DAY is the close (matches the 3:30pm CMT); BUY /
  SELL are the Treasury's ~1pm prices. 0 on the page means "no price" -> NaN.
* Settlement is T+1 business day (the Treasury cash standard).
* Coupon securities: accrued = coupon/f x (settle - previous coupon) / (period); yield =
  US STREET convention, compounded ``f`` times a year (semi-annual), fractional first
  period. Before the first coupon, accrual runs from the dated date over the regular
  period ending at the first coupon (an odd first period is approximated as regular).
* Bills: bond-equivalent yield (365-day basis; the Treasury's quadratic form beyond 182
  days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from infra.config import TREASURY_TYPES

PRICE_COLUMNS = ["timestamp", "cusip", "security_type", "price_buy", "price_sell", "price_eod", "accrued",
                 "yield_eod"]
PRICE_KEYS = ["timestamp", "cusip"]
_TYPES = {"MARKET BASED NOTE": "Note", "MARKET BASED BOND": "Bond", "MARKET BASED BILL": "Bill"}  # TIPS/FRN out
_SCALE = 10_000


def settlement_day(day) -> pd.Timestamp:
    return pd.Timestamp(day).normalize() + pd.offsets.BDay(1)


def normalize(page: pd.DataFrame, day) -> pd.DataFrame:
    """Page rows (text) -> in-scope securities with float prices (NaN where 0)."""
    df = page.copy()
    df["security_type"] = df["security_type"].map(_TYPES)
    df = df[df["security_type"].isin(TREASURY_TYPES)]
    out = pd.DataFrame({"timestamp": pd.Timestamp(day).normalize(), "cusip": df["cusip"].str.strip(),
                        "security_type": df["security_type"]})
    for src, dst in (("buy", "price_buy"), ("sell", "price_sell"), ("end_of_day", "price_eod")):
        v = pd.to_numeric(df[src], errors="coerce")
        out[dst] = v.where(v > 0)
    return out.reset_index(drop=True)


def coupon_dates(maturity: pd.Timestamp, settle: pd.Timestamp, per_year: int) -> list[pd.Timestamp]:
    """Regular coupon dates from the last one at or before ``settle`` through maturity,
    stepping back from maturity (end-of-month maturities stay end of month)."""
    months = 12 // per_year
    eom = (maturity + pd.Timedelta(days=1)).day == 1
    dates, k = [], 0
    while True:
        d = maturity - pd.DateOffset(months=months * k)
        if eom:
            d = d + pd.offsets.MonthEnd(0)
        dates.append(d)
        if d <= settle:
            break
        k += 1
    return sorted(dates)


def accrued_and_yield(price: float, coupon: float, maturity, settle, per_year: int, dated_date=None,
                      first_coupon=None) -> tuple[float, float]:
    """(accrued per 100, street yield in %) for one coupon security. NaN yield if no price."""
    maturity, settle = pd.Timestamp(maturity), pd.Timestamp(settle)
    if settle >= maturity:
        return np.nan, np.nan
    dates = coupon_dates(maturity, settle, per_year)
    prev, nxt = dates[0], dates[1]
    period = (nxt - prev).days
    accrual_start = prev
    if first_coupon is not None and pd.notna(first_coupon) and settle < pd.Timestamp(first_coupon):
        accrual_start = pd.Timestamp(dated_date) if dated_date is not None and pd.notna(dated_date) else prev
    accrued = coupon / per_year * max((settle - accrual_start).days, 0) / period
    if price is None or not np.isfinite(price):
        return accrued, np.nan
    n = len(dates) - 1  # coupons still to come
    w = (nxt - settle).days / period
    full = price + accrued
    c = coupon / per_year
    k = np.arange(n)

    def pv(y):
        disc = (1 + y / per_year) ** -(k + w)
        return c * disc.sum() + 100 * disc[-1] - full

    try:
        return accrued, 100 * brentq(pv, -0.2, 1.0)
    except ValueError:
        return accrued, np.nan


def bill_yield(price: float, maturity, settle) -> float:
    """Bond-equivalent yield (%) of a bill from its price per 100."""
    if price is None or not np.isfinite(price) or price <= 0:
        return np.nan
    d = (pd.Timestamp(maturity) - pd.Timestamp(settle)).days
    if d <= 0:
        return np.nan
    if d <= 182:
        return 100 * (100 - price) / price * 365 / d
    # Treasury's form: [-2b + 2 sqrt(b^2 - (2b - 1)(1 - 100/P))] / (2b - 1), b = d/365
    b = d / 365
    a = b - 0.5
    cc = (price - 100) / price
    return 100 * (-b + np.sqrt(b * b - 2 * a * cc)) / a


def prices_with_yields(page: pd.DataFrame, day, securities: pd.DataFrame) -> pd.DataFrame:
    """One day's in-scope prices + accrued + END OF DAY yield, using the reference
    table (``infra.pipeline.treasury_ref``) for coupon, dates and frequency. A CUSIP
    missing from the reference keeps its prices with NaN accrued/yield."""
    df = normalize(page, day)
    if df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    ref = securities.drop_duplicates("cusip").set_index("cusip")
    settle = settlement_day(day)
    accrued, yields = [], []
    for _, r in df.iterrows():
        s = ref.loc[r["cusip"]] if r["cusip"] in ref.index else None
        if s is None:
            accrued.append(np.nan), yields.append(np.nan)
        elif r["security_type"] == "Bill":
            accrued.append(0.0), yields.append(bill_yield(r["price_eod"], s["maturity_date"], settle))
        else:
            a, y = accrued_and_yield(r["price_eod"], float(s["coupon"]), s["maturity_date"], settle,
                                     int(s["coupons_per_year"]) or 2, s["dated_date"], s["first_coupon_date"])
            accrued.append(a), yields.append(y)
    df["accrued"], df["yield_eod"] = accrued, yields
    return df[PRICE_COLUMNS]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Disk form (CLAUDE.md 6b): prices, accrued and yield x10000 nullable Int32."""
    out = df[PRICE_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    for c in ("price_buy", "price_sell", "price_eod", "accrued", "yield_eod"):
        out[c] = (out[c].astype("float64") * _SCALE).round().astype("Int32")
    return out


def decode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("price_buy", "price_sell", "price_eod", "accrued", "yield_eod"):
        out[c] = out[c].astype("float64") / _SCALE
    out["cusip"] = out["cusip"].astype("category")
    return out
