"""US Treasury reference data (CLAUDE.md 18): the static security table, from the raw
auctions (infra.pipeline.tsy_auctions, whose ``raw_json`` keeps every Fiscal Data field).
Pure functions, no I/O.

One row per CUSIP, from its ORIGINAL issue (``reopening == "No"``; verified 2026-10-01:
every coupon CUSIP has exactly one) - the characteristics a reopening never changes.
Amounts are deliberately not here: they grow with each reopening, so they stay, point in
time, in the auctions themselves.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from infra.config import TREASURY_TYPES

SECURITY_COLUMNS = ["timestamp", "cusip", "security_type", "original_term", "term_months", "coupon",
                    "announced_date", "auction_date", "issue_date", "dated_date", "maturity_date",
                    "first_coupon_date", "coupons_per_year", "cash_management_bill"]
SECURITY_KEYS = ["cusip"]
_FREQUENCY = {"Semi-Annual": 2, "Annual": 1, "Quarterly": 4, "Monthly": 12}
_TERM = re.compile(r"(\d+)-(Year|Month|Week|Day)")
_UNIT_MONTHS = {"Year": 12.0, "Month": 1.0, "Week": 12.0 / 52.0, "Day": 12.0 / 365.0}


def term_months(term: str) -> float:
    """``"10-Year"`` -> 120, ``"9-Year 10-Month"`` -> 118, ``"26-Week"`` -> 6.0, ``"42-Day"``
    -> 1.38 (bills in weeks/days are approximate months). NaN if unparseable."""
    parts = _TERM.findall(str(term))
    return float(sum(int(n) * _UNIT_MONTHS[u] for n, u in parts)) if parts else np.nan


def _date(v) -> pd.Timestamp:
    return pd.to_datetime(v, errors="coerce") if v not in (None, "null", "") else pd.NaT


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def in_scope(auctions: pd.DataFrame) -> pd.DataFrame:
    """Bills, notes and bonds (``TREASURY_TYPES``) - not TIPS, not floating-rate notes."""
    return auctions[auctions["security_type"].isin(TREASURY_TYPES)
                    & (auctions["inflation_index_security"] != "Yes") & (auctions["floating_rate"] != "Yes")]


def securities(auctions: pd.DataFrame) -> pd.DataFrame:
    """The static table: one row per in-scope CUSIP, from its original issue. ``timestamp``
    is the ANNOUNCEMENT day - when the CUSIP became known - so an ``as_of`` read sees only
    securities announced by then. Columns SECURITY_COLUMNS."""
    orig = in_scope(auctions)
    orig = orig[orig["reopening"] == "No"]
    rows = []
    for _, a in orig.iterrows():
        r = json.loads(a["raw_json"])
        bill = a["security_type"] == "Bill"
        rows.append({
            "cusip": a["cusip"],
            "security_type": a["security_type"],
            "original_term": r.get("original_security_term") or r.get("security_term"),
            "coupon": np.nan if bill else _num(r.get("int_rate")),
            "announced_date": _date(r.get("announcemt_date")),
            "auction_date": _date(r.get("auction_date")),
            "issue_date": _date(r.get("issue_date")),
            "dated_date": pd.NaT if bill else _date(r.get("dated_date")),
            "maturity_date": _date(r.get("maturity_date")),
            "first_coupon_date": pd.NaT if bill else _date(r.get("first_int_payment_date")),
            "coupons_per_year": 0 if bill else _FREQUENCY.get(r.get("int_payment_frequency"), 0),
            "cash_management_bill": r.get("cash_management_bill_cmb") == "Yes",
        })
    if not rows:
        return pd.DataFrame(columns=SECURITY_COLUMNS)
    out = pd.DataFrame(rows)
    out["term_months"] = out["original_term"].map(term_months)
    out["timestamp"] = out["announced_date"].fillna(out["auction_date"])
    for c in ("timestamp", "announced_date", "auction_date", "issue_date", "dated_date", "maturity_date",
              "first_coupon_date"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
    out["coupons_per_year"] = out["coupons_per_year"].astype("int8")
    return out[SECURITY_COLUMNS].sort_values(["timestamp", "cusip"]).reset_index(drop=True)
