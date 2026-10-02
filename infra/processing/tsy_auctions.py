"""US Treasury auctions: Fiscal Data records -> typed rows, nominal-coupon selection,
point-in-time and release-calendar rows. Pure (no I/O).

* ``timestamp`` - the competitive CLOSE, tz-naive UTC (the record's own
  ``closing_time_comp``, New York wall clock -> ``infra.trading_calendar.snap_instants``):
  when the auction's results become knowable. Results are published minutes after it.
* Reopenings carry their REMAINING term (``security_term`` "9-Year 10-Month" for a 10y
  reopening), so the tenor is read from ``original_security_term``.
* Yields/prices are float percent/price as published; amounts in dollars (float - they
  run to 10^11, beyond int32, and are not prices: CLAUDE.md 6b doesn't apply).
* ``raw_json`` keeps the WHOLE record (114 fields) as delivered.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from infra.trading_calendar import snap_instants

KEYS = ["timestamp", "cusip"]
NUMERIC = ["offering_amt", "total_tendered", "total_accepted", "high_yield", "low_yield", "avg_med_yield",
           "high_price", "bid_to_cover_ratio", "direct_bidder_accepted", "indirect_bidder_accepted",
           "primary_dealer_accepted", "soma_accepted", "comp_accepted", "int_rate"]
TEXT = ["cusip", "security_type", "security_term", "original_security_term", "reopening",
        "inflation_index_security", "floating_rate", "closing_time_comp"]
DATES = ["announcemt_date", "auction_date", "issue_date", "maturity_date"]
COLUMNS = ["timestamp", *TEXT, *DATES, "tenor_years", *NUMERIC, "raw_json"]
NOMINAL_COUPON_TENORS = (2, 3, 5, 7, 10, 20, 30)


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _close_time(text: str) -> str:
    """``"01:00 PM"`` -> ``"13:00"``; unknown -> 13:00 (every coupon closes then)."""
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*(AM|PM)\s*$", str(text), re.I)
    if not m:
        return "13:00"
    hour = int(m.group(1)) % 12 + (12 if m.group(3).upper() == "PM" else 0)
    return f"{hour:02d}:{m.group(2)}"


def parse(records: pd.DataFrame) -> pd.DataFrame:
    """Raw API records (strings) -> typed ``COLUMNS``."""
    if records.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame({c: records.get(c, pd.Series("null", index=records.index)).astype(str) for c in TEXT})
    for c in DATES:
        df[c] = pd.to_datetime(records[c].replace("null", None), errors="coerce").astype("datetime64[ms]")
    for c in NUMERIC:
        df[c] = records.get(c, pd.Series(np.nan, index=records.index)).map(_num).astype("float64")
    df["tenor_years"] = df["original_security_term"].str.extract(r"^(\d+)-Year")[0].astype("float64")
    df["timestamp"] = [snap_instants([d], _close_time(t), "America/New_York")[0]
                       for d, t in zip(df["auction_date"], df["closing_time_comp"])]
    df["timestamp"] = df["timestamp"].astype("datetime64[ms]")
    df["raw_json"] = [json.dumps(r, sort_keys=True) for r in records.to_dict("records")]
    return df[COLUMNS].sort_values(KEYS).reset_index(drop=True)


def nominal_coupons(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed-rate notes and bonds, 2y-30y original term - no TIPS, FRNs or bills."""
    mask = (df["security_type"].isin(["Note", "Bond"]) & (df["inflation_index_security"] == "No")
            & (df["floating_rate"] == "No") & df["tenor_years"].isin(NOMINAL_COUPON_TENORS))
    return df[mask].reset_index(drop=True)


def held(df: pd.DataFrame) -> pd.Series:
    """The auction has happened and its results are in."""
    return df["high_yield"].notna() | df["high_price"].notna()


def as_of(df: pd.DataFrame, instant) -> pd.DataFrame:
    """Auctions as known at ``instant`` (UTC): announced by then; RESULTS only for those
    whose competitive close is at or before it (blanked otherwise - no look-ahead)."""
    instant = pd.Timestamp(instant)
    out = df[df["announcemt_date"] <= instant.normalize()].copy()
    future = out["timestamp"] > instant
    out.loc[future, [c for c in NUMERIC if c not in ("offering_amt", "int_rate")]] = np.nan
    return out.reset_index(drop=True)


def calendar_rows(df: pd.DataFrame, observed: pd.Timestamp) -> pd.DataFrame:
    """Release-calendar rows (infra.processing.release_calendar.COLUMNS) for nominal coupon
    auctions: at the exact close, known from the announcement day."""
    nc = nominal_coupons(df)
    observed = pd.Timestamp(observed).normalize()
    return pd.DataFrame({
        "timestamp": nc["timestamp"].astype("datetime64[ms]"),
        "event": [f"US_TSY_AUCTION_{int(t)}Y" for t in nc["tenor_years"]],
        "source": "fiscal_data", "stage": "", "time_source": "source",
        "known_from": nc["announcemt_date"].fillna(nc["auction_date"]).astype("datetime64[ms]"),
        "last_seen": pd.DatetimeIndex([observed] * len(nc)).astype("datetime64[ms]"),
    })
