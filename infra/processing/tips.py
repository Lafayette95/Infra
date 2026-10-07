"""TIPS (Treasury Inflation-Protected Securities) - pure: the reference table, the
Treasury's reference CPI and index ratios, real accrued and real yields. No I/O.

* Reference CPI for a day d in month m (31 CFR 356 Appendix B): CPI-U NSA of month m-3
  plus (day-1)/(days in m) of the step to month m-2. NSA CPI-U isn't revised, so the
  stored latest values are the ones the Treasury used.
* Index ratio = reference CPI on the settlement date / reference CPI on the DATED date
  (Fiscal Data's ``ref_cpi_on_dated_date`` of the original issue).
* FedInvest prices TIPS in REAL terms (clean, per 100 of inflation-adjusted principal,
  verified 2026-10-07: 53 TIPS near par on its page). The real yield is the street yield
  of the real cash flows - coupon / 2 semi-annually and 100 at maturity (the deflation
  floor on principal is ignored) - the nominal pricer (``treasury_prices.
  accrued_and_yield``) applied to them.
"""
from __future__ import annotations

import calendar
import json

import numpy as np
import pandas as pd

from infra.processing.treasury_prices import accrued_and_yield, settlement_day

TIPS_TYPE = "TIPS"  # FedInvest's security type
REF_COLUMNS = ["cusip", "coupon", "dated_date", "issue_date", "maturity_date", "ref_cpi_dated", "term"]
PRICE_COLUMNS = ["timestamp", "cusip", "price_buy", "price_sell", "price_eod", "accrued", "real_yield", "index_ratio",
                 "ref_cpi"]
PRICE_KEYS = ["timestamp", "cusip"]
_SCALE = 10_000
_RATIO_SCALE = 1_000_000


def reference(auctions: pd.DataFrame) -> pd.DataFrame:
    """One row per TIPS CUSIP from its ORIGINAL issue (``reopening`` = No): coupon, dated /
    issue / maturity dates and the base reference CPI. ``auctions``: the raw auctions store
    (``infra.pipeline.tsy_auctions.read_auctions(nominal_only=False)``), every field in
    ``raw_json``."""
    rows = []
    for raw in auctions["raw_json"]:
        d = json.loads(raw)
        if str(d.get("inflation_index_security", "")).lower() not in ("yes", "y") or str(d.get("reopening")).lower() == "yes":
            continue
        try:
            rows.append({"cusip": d["cusip"].strip().upper(), "coupon": float(d["int_rate"]),
                         "dated_date": pd.Timestamp(d.get("dated_date") if d.get("dated_date") not in (None, "null") else d["issue_date"]),
                         "issue_date": pd.Timestamp(d["issue_date"]), "maturity_date": pd.Timestamp(d["maturity_date"]),
                         "ref_cpi_dated": float(d["ref_cpi_on_dated_date"]),
                         "term": d.get("original_security_term") or d.get("security_term")})
        except (KeyError, TypeError, ValueError):
            continue  # an announced auction not held yet (no coupon) - picked up once it is
    out = pd.DataFrame(rows, columns=REF_COLUMNS)
    return out.drop_duplicates("cusip", keep="last").reset_index(drop=True)


def fill_missing_months(cpi: pd.Series) -> pd.Series:
    """A month MISSING while a later month exists (never published - October 2025, the
    government shutdown) takes the Treasury's fallback (31 CFR 356 Appendix B): the last
    available CPI escalated by its own year-over-year change over one month,
    CPI(m) = CPI(m-1) x (CPI(m-1) / CPI(m-13))^(1/12). Verified 2026-10-07: reproduces the
    Treasury's published reference CPI on 2025-12-31 and 2026-01-30 issue dates. Months not
    published YET (after the last one) are never filled."""
    s = cpi.sort_index().copy()
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    for m in full:
        if m not in s.index or pd.isna(s.get(m)):
            prev, base = m - pd.DateOffset(months=1), m - pd.DateOffset(months=13)
            if prev in s.index and base in s.index:
                s.loc[m] = round(s[prev] * (s[prev] / s[base]) ** (1.0 / 12.0), 3)  # CPI precision
    return s.sort_index()


def reference_cpi(days, cpi: pd.Series) -> pd.Series:
    """Reference CPI per day. ``cpi``: CPI-U NSA indexed by month start (a never-published
    month filled by ``fill_missing_months``). NaN where a needed month isn't available."""
    cpi = fill_missing_months(cpi)
    days = pd.DatetimeIndex(pd.to_datetime(days)).normalize()
    month = days.to_period("M").to_timestamp()
    m3 = cpi.reindex(month - pd.DateOffset(months=3)).to_numpy()
    m2 = cpi.reindex(month - pd.DateOffset(months=2)).to_numpy()
    dim = np.array([calendar.monthrange(d.year, d.month)[1] for d in days])
    frac = (days.day.to_numpy() - 1) / dim
    return pd.Series(m3 + frac * (m2 - m3), index=days)


def prices(page: pd.DataFrame, day, ref: pd.DataFrame, cpi: pd.Series) -> pd.DataFrame:
    """One FedInvest page's TIPS rows -> real prices, real accrued and yield, the index
    ratio at the T+1 settlement and the reference CPI. A TIPS missing from ``ref`` keeps its
    prices with NaN derived fields."""
    df = page[page["security_type"].astype(str).str.strip().str.upper() == TIPS_TYPE]
    if df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    out = pd.DataFrame({"timestamp": pd.Timestamp(day).normalize(), "cusip": df["cusip"].str.strip().str.upper()})
    for src, dst in (("buy", "price_buy"), ("sell", "price_sell"), ("end_of_day", "price_eod")):
        v = pd.to_numeric(df[src], errors="coerce").to_numpy()
        out[dst] = np.where(v > 0, v, np.nan)
    settle = settlement_day(day)
    rc = float(reference_cpi([settle], cpi).iloc[0])
    r = ref.set_index("cusip")
    acc, yld, ratio = [], [], []
    for cusip, px in zip(out["cusip"], out["price_eod"]):
        if cusip not in r.index:
            acc.append(np.nan), yld.append(np.nan), ratio.append(np.nan)
            continue
        s = r.loc[cusip]
        a, y = accrued_and_yield(px, float(s["coupon"]), s["maturity_date"], settle, 2, s["dated_date"], None)
        acc.append(a), yld.append(y), ratio.append(rc / float(s["ref_cpi_dated"]))
    out["accrued"], out["real_yield"], out["index_ratio"], out["ref_cpi"] = acc, yld, ratio, rc
    return out[PRICE_COLUMNS].reset_index(drop=True)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Disk form (CLAUDE.md 6b): prices, accrued, yield and reference CPI x10000 nullable
    Int32; the index ratio x1,000,000 (6 decimals, as the Treasury publishes it)."""
    out = df[PRICE_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    for c in ("price_buy", "price_sell", "price_eod", "accrued", "real_yield", "ref_cpi"):
        out[c] = (out[c].astype("float64") * _SCALE).round().astype("Int32")
    out["index_ratio"] = (out["index_ratio"].astype("float64") * _RATIO_SCALE).round().astype("Int32")
    return out


def decode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("price_buy", "price_sell", "price_eod", "accrued", "real_yield", "ref_cpi"):
        out[c] = out[c].astype("float64") / _SCALE
    out["index_ratio"] = out["index_ratio"].astype("float64") / _RATIO_SCALE
    out["cusip"] = out["cusip"].astype(str)
    return out
