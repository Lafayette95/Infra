"""Treasury swap spreads per day and tenor, two sources (root CLAUDE.md 16; config
``SWAP_SPREAD_*``). Disk only: the OIS curve (Derived/OisCurves), CMT par yields
(Daily/Bonds), the on-the-run map, the securities table and FedInvest END OF DAY prices.

* ``cmt``: the curve's par OIS rate at the tenor minus the CMT par yield - the PLAIN
  difference (the market's quote convention); ``spread_bb_bp`` converts the swap rate to the
  Treasury's basis first (semi-annual, ACT/365: ``infra.analytics.swap_curve.bond_basis``).
* ``otr``: the on-the-run bond's (and the 1-old's) PAR-PAR asset-swap spread over the curve
  (``asw_bp``, + = cheap to swaps); ``spread_bp`` = -``asw_bp``, so both sources read as
  "swap minus Treasury" and move together.
All three marks are 15:30 New York (the NY1530 close, CMT, FedInvest END OF DAY).

Store ``Derived/SwapSpreads``, keys ``timestamp`` (the day), ``ticker`` (``US_SWSP_<t>y``),
``source``, ``cusip`` ("" for cmt); a recomputed day replaces its rows.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import swap_curve as sc
from infra.analytics import treasury_curve as tc
from infra.config import (
    DAILY_BONDS_DIR,
    DAILY_TREASURY_PRICES_DIR,
    OIS_CURVES,
    OIS_CURVES_DIR,
    SWAP_CURVES,
    SWAP_SPREAD_CURVE,
    SWAP_SPREAD_OTR_RANKS,
    SWAP_SPREAD_TENORS,
    SWAP_SPREADS_DIR,
    TREASURY_OTR_DIR,
    TREASURY_SECURITIES_DIR,
)
from infra.pipeline.ois_curves import read_ois_curves
from infra.processing.treasury_prices import settlement_day
from infra.storage import parquet_store

log = logging.getLogger(__name__)
KEYS = ["timestamp", "ticker", "source", "cusip"]
COLUMNS = KEYS + ["tenor", "rank", "spread_bp", "spread_bb_bp", "asw_bp", "swap_rate", "bond_yield", "maturity_date"]
_ONE_DAY = pd.Timedelta(days=1)


def ticker(tenor: int) -> str:
    return f"US_SWSP_{tenor}y"


def compute_swap_spreads(start, end, *, curve: str = SWAP_SPREAD_CURVE, tenors=SWAP_SPREAD_TENORS,
                         ois_root: Path = OIS_CURVES_DIR, bonds_root: Path = DAILY_BONDS_DIR,
                         otr_root: Path = TREASURY_OTR_DIR, securities_root: Path = TREASURY_SECURITIES_DIR,
                         prices_root: Path = DAILY_TREASURY_PRICES_DIR) -> tuple[pd.DataFrame, dict]:
    """Rows for every day in ``[start, end]`` with an OIS curve. Returns (rows,
    ``{"days": curve days, "empty_days": {day: reason}}``) - a day with a curve and its
    inputs but no spread is listed with the reason."""
    from infra.pipeline.bond_yields import read_bond_yields
    from infra.pipeline.treasury_otr import read_otr
    from infra.pipeline.treasury_prices import read_prices
    from infra.pipeline.treasury_ref import read_securities
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    hi = end + _ONE_DAY
    nodes = read_ois_curves(start, hi, curve=curve, root=ois_root)
    if nodes.empty:
        return pd.DataFrame(columns=COLUMNS), {"days": [], "empty_days": {}}
    lag = SWAP_CURVES[OIS_CURVES[curve].currency].spot_lag_days
    nodes["day"] = pd.to_datetime(nodes["timestamp"]).dt.normalize()
    curves = {d: sc.curve_from_nodes(g) for d, g in nodes.groupby("day")}
    cmt = read_bond_yields([f"US_BOND_{t}y" for t in tenors], start, hi, source="cmt", cmt_root=bonds_root)
    cmt = cmt.set_index(["timestamp", "ticker"])["yield"]
    otr = read_otr(start, hi, root=otr_root)
    otr = otr[otr["rank"].isin(SWAP_SPREAD_OTR_RANKS) & otr["tenor"].isin([f"{t}y" for t in tenors])]
    px = read_prices(start, hi, cusips=sorted(set(otr["cusip"].astype(str))), root=prices_root) if len(otr) else otr
    px = px[px["price_eod"] > 0].set_index(["timestamp", "cusip"]) if len(px) else None
    sec = read_securities(root=securities_root).drop_duplicates("cusip").set_index("cusip")
    rows, empty = [], {}
    for day, c in sorted(curves.items()):
        swap = {t: sc.par_rate(c, day, sc.swap_schedule(day, t, lag)) for t in tenors}
        n_cmt = 0
        for t in tenors:
            y = cmt.get((day, f"US_BOND_{t}y"))
            if y is None or pd.isna(y):
                continue
            rows.append({"timestamp": day, "ticker": ticker(t), "source": "cmt", "cusip": "", "tenor": t, "rank": -1,
                         "spread_bp": (swap[t] - y) * 100.0, "spread_bb_bp": (sc.bond_basis(swap[t]) - y) * 100.0,
                         "asw_bp": np.nan, "swap_rate": swap[t], "bond_yield": float(y), "maturity_date": pd.NaT})
            n_cmt += 1
        settle = settlement_day(day)
        n_otr = 0
        for r in otr[otr["timestamp"] == day].itertuples():
            if px is None or (day, r.cusip) not in px.index:
                continue
            p = px.loc[(day, r.cusip)]
            s = sec.loc[r.cusip]
            t_cf, a = tc.cash_flows(float(s["coupon"]), s["maturity_date"], settle, int(s["coupons_per_year"] or 2))
            asw = sc.asw_par_par(c, day, settle, float(p["price_eod"]) + float(p["accrued"]), t_cf, a, s["maturity_date"])
            tenor = int(str(r.tenor).rstrip("y"))
            rows.append({"timestamp": day, "ticker": ticker(tenor), "source": "otr", "cusip": str(r.cusip), "tenor": tenor,
                         "rank": int(r.rank), "spread_bp": -asw, "spread_bb_bp": np.nan, "asw_bp": asw,
                         "swap_rate": np.nan, "bond_yield": float(p["yield_eod"]), "maturity_date": s["maturity_date"]})
            n_otr += 1
        if n_cmt == 0 and n_otr == 0:
            empty[day] = "an OIS curve but neither CMT yields nor on-the-run END OF DAY prices"
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df = df.astype({"timestamp": "datetime64[ms]", "tenor": "int16", "rank": "int8"})
        df["maturity_date"] = pd.to_datetime(df["maturity_date"]).astype("datetime64[ms]")
    return df, {"days": sorted(curves), "empty_days": empty}


def store_swap_spreads(df: pd.DataFrame, start, end, *, root: Path = SWAP_SPREADS_DIR) -> int:
    """Replace the days ``[start, end]`` by ``df``."""
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    parquet_store.delete_where(root, lambda part: pd.to_datetime(part["timestamp"]).ge(lo)
                               & pd.to_datetime(part["timestamp"]).lt(hi))
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, KEYS)
    return len(df)


def build_swap_spreads(start, end, *, root: Path = SWAP_SPREADS_DIR, **kw) -> dict:
    df, diag = compute_swap_spreads(start, end, **kw)
    n = store_swap_spreads(df, start, end, root=root)
    log.info("swap spreads %s..%s: %d row(s)", pd.Timestamp(start).date(), pd.Timestamp(end).date(), n)
    return {"rows": n, **diag}


def read_swap_spreads(start=None, end=None, *, source: str | None = None, tickers=None,
                      root: Path = SWAP_SPREADS_DIR) -> pd.DataFrame:
    """Stored rows with ``timestamp`` in ``[start, end)``."""
    eq = {}
    if source:
        eq["source"] = [source]
    if tickers is not None:
        eq["ticker"] = list(tickers)
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end), equals_in=eq or None)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    for col in ("ticker", "source", "cusip"):
        df[col] = df[col].astype(str)
    return df[COLUMNS].sort_values(KEYS).reset_index(drop=True)
