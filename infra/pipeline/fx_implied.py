"""FX-implied 3-month rates from the DTCC FOREX archive (root CLAUDE.md 16; pure parts
``infra.processing.dtcc_fx``): per day and currency (EUR, GBP, JPY, CAD), the median over
spot-start ~3-month FX swaps of at least ``FX_IMPLIED_MIN_NOTIONAL_USD`` of the currency's
implied rate over the swap's period - USD's growth from the SOFR curve, the swap's forward
points for the rest - as a SEMI-ANNUAL rate. It is the rolling hedge's (r + b), basis and
turn-of-year premia included. Disk only; store ``Derived/FxImpliedRates`` (keys ``timestamp`` =
day, ``currency``), ``available_at`` = the posting of DTCC's file for that UTC day.

Checked 2026-10-07 against v1 (OIS 3m + the 3m cross-currency basis): GBP +4bp, stable;
EUR +2bp except windows crossing a year-end (+60bp - the year-end USD funding premium FX
swaps price and an OIS rate misses); JPY -17 to -33bp and CAD +49 to -48bp drifting with the
priced BoJ / BoC paths - their OIS curves are flat to the 1y, FX swaps are not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import hedged_yields as hy
from infra.analytics import swap_curve as sc
from infra.config import DTCC_DIR, FX_IMPLIED_DIR, FX_IMPLIED_MIN_NOTIONAL_USD, OIS_CURVES_DIR
from infra.processing import dtcc_fx as fx
from infra.storage import parquet_store

KEYS = ["timestamp", "currency"]
COLUMNS = KEYS + ["rate_pct", "n_swaps", "spread_bp", "crosses_year_end", "available_at"]
_ONE_DAY = pd.Timedelta(days=1)


def compute_fx_implied(start, end, *, dtcc_root: Path = DTCC_DIR, ois_root: Path = OIS_CURVES_DIR) -> pd.DataFrame:
    from infra.pipeline import dtcc
    from infra.pipeline.ois_curves import read_ois_curves
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = sorted(d for d in dtcc.archived_days("FOREX", root=dtcc_root) if start <= d <= end)
    usd = read_ois_curves(start - pd.Timedelta(days=10), end + _ONE_DAY, curve="USD_SOFR", root=ois_root)
    if usd.empty or not days:
        return pd.DataFrame(columns=COLUMNS)
    usd["day"] = pd.to_datetime(usd["timestamp"]).dt.normalize()
    curves = {d: sc.curve_from_nodes(g) for d, g in usd.groupby("day")}
    cdays = np.array(sorted(curves), dtype="datetime64[ns]")
    rows = []
    for d in days:
        sw = fx.spot_start_swaps(fx.legs(dtcc.read_dtcc_day("FOREX", d, root=dtcc_root), d), d)
        sw = sw[sw["orientation_ok"] & (sw["usd_notional"] >= FX_IMPLIED_MIN_NOTIONAL_USD)]
        i = np.searchsorted(cdays, np.datetime64(d), side="right") - 1
        if sw.empty or i < 0:
            continue
        c = curves[pd.Timestamp(cdays[i])]
        for ccy, g in sw.groupby("ccy"):
            vals, ye = [], []
            for r in g.itertuples():
                g_usd = float(c.discount([r.near_days / 365.25])[0] / c.discount([r.far_days / 365.25])[0])
                growth = fx.implied_growth(r.near_rate, r.far_rate, g_usd, ccy)
                vals.append(hy.df_to_semiannual(1.0 / growth, (r.far_days - r.near_days) / 365.0))
                far_date = d + pd.Timedelta(days=int(r.far_days))
                ye.append(far_date.year > d.year)
            v = np.array(vals, dtype=float)
            rows.append({"timestamp": d, "currency": ccy, "rate_pct": float(np.median(v)), "n_swaps": len(v),
                         "spread_bp": float((v.max() - v.min()) * 100), "crosses_year_end": float(np.mean(ye)),
                         "available_at": d + _ONE_DAY + pd.Timedelta(minutes=30)})
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
        df["available_at"] = pd.to_datetime(df["available_at"]).astype("datetime64[ms]")
        df["currency"] = df["currency"].astype("string")
        df["n_swaps"] = df["n_swaps"].astype("int32")
    return df


def store_fx_implied(df: pd.DataFrame, start, end, *, root: Path = FX_IMPLIED_DIR) -> int:
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    if parquet_store.has_data(root):
        parquet_store.delete_where(root, lambda part: pd.to_datetime(part["timestamp"]).ge(lo)
                                   & pd.to_datetime(part["timestamp"]).lt(hi))
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, KEYS)
    return len(df)


def read_fx_implied(start=None, end=None, *, currency: str | None = None, root: Path = FX_IMPLIED_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end) + _ONE_DAY,
                                        equals_in={"currency": [currency]} if currency else None) \
        if parquet_store.has_data(root) else None
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["currency"] = df["currency"].astype(str)
    return df.sort_values(KEYS).reset_index(drop=True)
