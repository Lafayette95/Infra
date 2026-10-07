"""Real (TIPS) yield curve and breakevens (root CLAUDE.md 25a; the nominal curve's own
fitting, ``infra.pipeline.treasury_curves.fit_day``, on REAL cash flows and real prices).
Disk only: ``Daily/TipsPrices`` (``infra.pipeline.tips``), the TIPS reference, and the
nominal ``Derived/TreasuryCurves`` for the breakevens.

Stores ``Derived/TipsCurves`` (keys ``timestamp``, ``method``: the fit, real par/zero yields
on ``CURVE_GRID_YEARS`` from ``TIPS_FIT_MIN_YEARS`` on (shorter points would be extrapolations -
the 1y read 8.2% real / -3.6% breakeven before they were blanked) and the BREAKEVENS ``be_par_<m>y`` / ``be_zero_<m>y`` = nominal minus
real, same day and method) and ``Derived/TipsRV`` (keys ``timestamp``, ``cusip``,
``method``: per TIPS its real yield, z-spread to the real curve, carry and rolldown). TIPS
within ``TIPS_FIT_MIN_YEARS`` of maturity are out of the fit. Svensson warm-starts from the
STORED previous day, so a window equals a full sequential build (as the nominal curve).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.config import (CURVE_GRID_YEARS, DAILY_TIPS_PRICES_DIR, TIPS_CURVES_DIR, TIPS_FIT_MIN_YEARS, TIPS_RV_DIR,
                          TREASURY_CURVES_DIR)
from infra.pipeline.tips import read_tips, tips_reference
from infra.pipeline.treasury_curves import CURVE_KEYS, RV_KEYS, fit_day, read_curves
from infra.processing.treasury_prices import settlement_day
from infra.storage import parquet_store

_ONE_DAY = pd.Timedelta(days=1)


def stored_seed(before, *, root: Path = TIPS_CURVES_DIR):
    if not parquet_store.has_data(root):
        return None
    c = read_curves(pd.Timestamp(before) - pd.Timedelta(days=30), pd.Timestamp(before) - _ONE_DAY, method="svensson",
                    root=root)
    return None if c.empty else np.array([c.iloc[-1][f"p{k}"] for k in range(6)], dtype="float64")


def compute_tips_curves(start, end, *, prices_root: Path = DAILY_TIPS_PRICES_DIR, curves_root: Path = TIPS_CURVES_DIR,
                        nominal_root: Path = TREASURY_CURVES_DIR, auctions_root=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit every day with TIPS prices in ``[start, end]``; no writes."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    px = read_tips(start, end + _ONE_DAY, root=prices_root).rename(columns={"real_yield": "yield_eod"})
    ref = tips_reference(auctions_root=auctions_root).assign(coupons_per_year=2)[["cusip", "coupon", "maturity_date",
                                                                                   "coupons_per_year"]]
    nominal = read_curves(start, end, root=nominal_root)
    seed = stored_seed(start, root=curves_root)
    curves, rows, days, empty = [], [], [], {}
    for day, g in px.groupby("timestamp"):
        day = pd.Timestamp(day)
        if not g["price_eod"].gt(0).any():
            continue
        days.append(day)
        settle = settlement_day(day)
        short = set(ref.loc[pd.to_datetime(ref["maturity_date"]) < settle + pd.DateOffset(months=int(TIPS_FIT_MIN_YEARS * 12)),
                            "cusip"])
        c, r, seed2 = fit_day(day, g[["cusip", "price_eod", "accrued", "yield_eod"]], ref, short, svensson_start=seed)
        if not c:
            empty[day] = f"too few priced TIPS to fit ({g['price_eod'].gt(0).sum()} with a price, 20 needed)"
            continue
        seed = seed2
        n = nominal[nominal["timestamp"] == day].set_index("method")
        for row in c:
            for m in CURVE_GRID_YEARS:  # shorter than the fitted range: an extrapolation, not a yield
                if m < TIPS_FIT_MIN_YEARS:
                    row[f"par_{m}y"] = row[f"zero_{m}y"] = np.nan
            if row["method"] in n.index:
                for m in CURVE_GRID_YEARS:
                    row[f"be_par_{m}y"] = float(n.loc[row["method"], f"par_{m}y"]) - row[f"par_{m}y"]
                    row[f"be_zero_{m}y"] = float(n.loc[row["method"], f"zero_{m}y"]) - row[f"zero_{m}y"]
        curves += c
        rows += r
    cdf, rdf = pd.DataFrame(curves), pd.DataFrame(rows)
    for df in (cdf, rdf):
        if len(df):
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
    return cdf, rdf, {"days": days, "empty_days": empty}


def store_days(df: pd.DataFrame, start, end, *, root: Path, keys) -> int:
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    parquet_store.delete_where(root, lambda p: pd.to_datetime(p["timestamp"]).ge(lo) & pd.to_datetime(p["timestamp"]).lt(hi))
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, list(keys))
    return len(df)


def build_tips_curves(start, end, *, curves_root: Path = TIPS_CURVES_DIR, rv_root: Path = TIPS_RV_DIR) -> dict:
    """Year by year (each year's Svensson seeded from the stored end of the previous one)."""
    out = {"days": 0, "empty": {}}
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        a, b = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")), min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
        cdf, rdf, d = compute_tips_curves(a, b, curves_root=curves_root)
        store_days(cdf, a, b, root=curves_root, keys=CURVE_KEYS)
        store_days(rdf, a, b, root=rv_root, keys=RV_KEYS)
        out["days"] += len(d["days"])
        out["empty"].update(d["empty_days"])
    return out


def read_tips_curves(start=None, end=None, *, method: str | None = None, root: Path = TIPS_CURVES_DIR) -> pd.DataFrame:
    return read_curves(start, end, method=method, root=root)
