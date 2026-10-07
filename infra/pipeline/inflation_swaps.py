"""Zero-coupon inflation swaps from the DTCC archive (root CLAUDE.md 16; spec
``infra.config.INFLATION_SWAPS``): daily closes and a ZC curve with forward breakevens and
real rates. Disk only - the archived RATES files and the OIS curve.

* Closes (``Derived/InflationSwapCloses``, keys ``timestamp`` = the snap instant,
  ``close``, ``curve``, ``tenor``, ``method``): the same trade handling and PURE estimator
  as the OIS closes (``infra.pipeline.swap_closes``: corrections/cancellations linked by
  ``trade_key``, spot-starting whole-year swaps, no upfront, off-market filter, weighted
  median in the close's window), package legs kept per spec. Day D reads corrections in
  files D..D+``SWAP_CORRECTION_DAYS``; a recomputed day replaces its rows.
* Curve (``Derived/InflationCurves``, keys ``timestamp``, ``curve``, ``tenor``): per close
  day the ZC rates (``zc_pct``, annual compounding: index(T) / index(0) = (1 + z)^T, the
  CPI lagged 3 months as the product is quoted), the forward breakeven from the previous
  tenor (``fwd_pct``, e.g. 5y5y on the 10y row), and the REAL zero rate against the OIS
  curve at the same instant (``real_pct``: (1 + nominal)^T / (1 + z)^T - 1, nominal the
  OIS zero rate converted to annual compounding). Missing interior tenors are filled like
  the OIS curve's (``fill_missing_tenors``), flagged ``filled``.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import swap_curve as sc
from infra.config import (
    DTCC_DIR,
    INFLATION_CURVES_DIR,
    INFLATION_SWAP_CLOSES_DIR,
    INFLATION_SWAPS,
    OIS_CURVES_DIR,
    SWAP_CLOSE_WEIGHTING,
    SWAP_CLOSES,
    InflationSwapSpec,
)
from infra.pipeline import dtcc
from infra.pipeline.ois_curves import read_ois_curves
from infra.pipeline.swap_closes import REPORT, day_events
from infra.processing import dtcc_trades as dt
from infra.processing.swap_closes import CLOSE_COLUMNS, pure_closes
from infra.storage import parquet_store
from infra.trading_calendar import snap_instants

log = logging.getLogger(__name__)
CLOSE_KEYS = ["timestamp", "close", "curve", "tenor", "method"]
CURVE_KEYS = ["timestamp", "curve", "tenor"]
_ONE_DAY = pd.Timedelta(days=1)


def trades_on(day, spec: InflationSwapSpec, events: pd.DataFrame, *, as_of=None) -> pd.DataFrame:
    """The spec's clean ZC trades EXECUTED on UTC day ``day``."""
    if events.empty:
        return pd.DataFrame(columns=dt.TRADE_COLUMNS)
    current = dt.current_trades(dt.trade_events(dt.product_rows(events, spec.curve)), as_of)
    tr = dt.par_trades(current, spec.curve, spec.currency, allow_packages=spec.allow_packages)
    day = pd.Timestamp(day).normalize()
    return tr[(tr["executed"] >= day) & (tr["executed"] < day + _ONE_DAY)].reset_index(drop=True)


def compute_closes(day, *, curves=None, as_of=None, dtcc_root: Path = DTCC_DIR, cache: dict | None = None) -> pd.DataFrame:
    """Every spec's closes on ``day`` (pure method). No writes."""
    day = pd.Timestamp(day).normalize()
    events = day_events(day, as_of=as_of, dtcc_root=dtcc_root, cache=cache)
    out = []
    for name in curves or tuple(INFLATION_SWAPS):
        spec = INFLATION_SWAPS[name]
        tr = trades_on(day, spec, events, as_of=as_of)
        for close in spec.closes:
            cs = dataclasses.replace(SWAP_CLOSES[close], pure_fallback_half_window_min=spec.fallback_half_window_min)
            instant = snap_instants([day], cs.local_time, cs.timezone)[0]
            out.append(pure_closes(tr, instant, cs, SWAP_CLOSE_WEIGHTING, close_name=close, currency=name))
    out = [o for o in out if not o.empty]
    if not out:
        return pd.DataFrame(columns=CLOSE_KEYS)
    df = pd.concat(out, ignore_index=True)[CLOSE_COLUMNS].rename(columns={"currency": "curve"})
    return df.astype({"timestamp": "datetime64[ms]", "tenor": "int32", "n_trades": "int32", "half_window_min": "int32"})


def backfill_closes(start, end, *, root: Path = INFLATION_SWAP_CLOSES_DIR, dtcc_root: Path = DTCC_DIR) -> dict:
    """Compute and store every archived day in ``[start, end]`` (a day's rows replaced)."""
    have = dtcc.archived_days(REPORT, root=dtcc_root)
    frames, cache, days = [], {}, []
    for day in pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()):
        for k in [k for k in cache if k < day]:
            del cache[k]
        if day in have:
            frames.append(compute_closes(day, dtcc_root=dtcc_root, cache=cache))
            days.append(day)
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    n = store_days(df, start, end, root=root, keys=CLOSE_KEYS)
    return {"days": len(days), "rows": n}


def store_days(df: pd.DataFrame, start, end, *, root: Path, keys) -> int:
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    parquet_store.delete_where(root, lambda p: pd.to_datetime(p["timestamp"]).ge(lo) & pd.to_datetime(p["timestamp"]).lt(hi))
    if df is None or df.empty:
        return 0
    parquet_store.write_partitioned(df, root, list(keys))
    return len(df)


def read_closes(start=None, end=None, *, curve: str | None = None, close: str | None = None,
                root: Path = INFLATION_SWAP_CLOSES_DIR) -> pd.DataFrame:
    eq = {k: [v] for k, v in (("curve", curve), ("close", close)) if v is not None}
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end), equals_in=eq or None)
    if df is None or df.empty:
        return pd.DataFrame(columns=CLOSE_KEYS)
    for c in ("close", "curve", "method"):
        df[c] = df[c].astype(str)
    return df.sort_values(CLOSE_KEYS).reset_index(drop=True)


# --------------------------------------------------------------------------- curve
def compute_curves(start, end, *, curves=None, closes_root: Path = INFLATION_SWAP_CLOSES_DIR,
                   ois_root: Path = OIS_CURVES_DIR, fill_max_age_days: int = 14) -> tuple[pd.DataFrame, dict]:
    """Per close day: ZC rates by tenor (interior gaps filled), forward breakevens and
    real zero rates. Reads ``fill_max_age_days`` + slack before ``start`` for the fills."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    rows, days = [], []
    for name in curves or tuple(INFLATION_SWAPS):
        spec = INFLATION_SWAPS[name]
        for close in spec.closes:
            c = read_closes(start - pd.Timedelta(days=fill_max_age_days + 17), end + _ONE_DAY, curve=name, close=close,
                            root=closes_root)
            if c.empty:
                continue
            c["day"] = pd.to_datetime(c["timestamp"]).dt.normalize()
            rates = c.pivot(index="day", columns="tenor", values="rate").sort_index()
            filled_rates, filled = sc.fill_missing_tenors(rates, max_age_days=fill_max_age_days)
            ois = read_ois_curves(start, end + _ONE_DAY, curve=spec.ois_curve, root=ois_root)
            ois["day"] = pd.to_datetime(ois["timestamp"]).dt.normalize() if len(ois) else ois.get("timestamp")
            oc = {d: sc.curve_from_nodes(g) for d, g in ois.groupby("day")} if len(ois) else {}
            insts = c.drop_duplicates("day").set_index("day")["timestamp"]
            for day in [d for d in filled_rates.index if start <= d <= end]:
                z = filled_rates.loc[day].dropna()
                if z.empty:
                    continue
                days.append(day)
                prev_t, prev_g = 0, 1.0
                for t, zc in z.items():
                    growth = (1 + zc / 100.0) ** t
                    fwd = ((growth / prev_g) ** (1.0 / (t - prev_t)) - 1) * 100.0
                    real = np.nan
                    if day in oc:
                        n_cc = float(oc[day].zero([t])[0]) / 100.0  # continuously compounded, ACT/365.25
                        real = (np.exp(n_cc * t) / growth) ** (1.0 / t) * 100.0 - 100.0
                    rows.append({"timestamp": insts[day], "curve": name, "close": close, "tenor": int(t), "zc_pct": float(zc),
                                 "fwd_pct": float(fwd), "fwd_from": int(prev_t), "real_pct": float(real),
                                 "filled": bool(filled.loc[day, t])})
                    prev_t, prev_g = t, growth
    df = pd.DataFrame(rows)
    if len(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
        df["tenor"] = df["tenor"].astype("int16")
        df["fwd_from"] = df["fwd_from"].astype("int16")
    return df, {"days": sorted(set(days)), "empty_days": {}}


def read_curves(start=None, end=None, *, curve: str | None = None, root: Path = INFLATION_CURVES_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end),
                                        equals_in={"curve": [curve]} if curve else None)
    if df is None or df.empty:
        return pd.DataFrame(columns=CURVE_KEYS)
    df["curve"] = df["curve"].astype(str)
    return df.sort_values(CURVE_KEYS).reset_index(drop=True)
