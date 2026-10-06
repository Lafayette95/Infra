"""Benchmark YIELD P&L, part of the ``bmk_pnl`` step (root CLAUDE.md 12; config
``BMK_YIELD_TICKERS`` / ``BMK_YIELD_SOURCES``): per ``US_BOND_<t>y`` ticker and source, the
day's price-action P&L in bp of a long position (``infra.processing.yield_pnl``), stored in
``Bmk/Pnl`` under bmk ``yield_cmt`` / ``yield_otr`` / ``yield_curve`` next to the futures'
``futures_price`` rows - ``pnl_per_dv01`` is bp for every bmk, so one column serves every
instrument. Local computation over stored data, no API:

* ``cmt``: the Treasury par curve (Daily/Bonds, the px step);
* ``otr``: the on-the-run bond's END OF DAY yield (the OTR map, ref step; FedInvest prices, px
  step), on the bond held the previous day;
* ``curve``: our fitted spline's par yield (Derived/TreasuryCurves) - NOT built by the cycle yet,
  so its rows reach only as far as the last hand build (``yield_pnl_present`` warns).

Upsert by ``timestamp, ticker, bmk``: a recomputed day replaces its row, a revised source
yield shows up in ``pnl_no_revisions``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.config import BMK_YIELD_AGREE_BP, BMK_YIELD_MAX_GAP_DAYS, BMK_YIELD_SOURCES, BMK_YIELD_TICKERS
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.processing import yield_pnl as yp
from infra.processing.otr_yields import ticker_for
from infra.storage import parquet_store

_LOOKBACK = pd.Timedelta(days=14)         # a prior observation for the window's first day
_ONE_DAY = pd.Timedelta(days=1)
PNL_KEYS = ["timestamp", "ticker", "bmk"]


def _pnl_dir(paths: CyclePaths):
    return paths.bmk_root / "Pnl"


def _yields(source: str, tickers, start, end, paths: CyclePaths) -> pd.DataFrame:
    from infra.pipeline.bond_yields import read_bond_yields
    return read_bond_yields(tickers, start, end, source=source, cmt_root=paths.daily_bonds_dir,
                            curve_root=paths.treasury_curves_dir)[["timestamp", "ticker", "yield"]]


def _otr_pnl(tickers, start, end, paths: CyclePaths) -> pd.DataFrame:
    from infra.pipeline.treasury_otr import read_otr
    from infra.pipeline.treasury_prices import read_prices
    m = read_otr(start, end, rank=0, root=paths.treasury_otr_dir)
    if m.empty:
        return pd.DataFrame(columns=yp.PNL_COLUMNS)
    m = m.assign(ticker=m["tenor"].map(ticker_for))
    m = m[m["ticker"].isin(list(tickers))][["timestamp", "ticker", "cusip"]]
    px = read_prices(start, end, cusips=sorted(set(m["cusip"].astype(str))), root=paths.treasury_prices_dir)
    by = px[["timestamp", "cusip"]].assign(**{"yield": px["yield_eod"]})
    return yp.held_bond_pnl(m, by, "yield_otr", max_gap_days=BMK_YIELD_MAX_GAP_DAYS)


def compute_yield_pnl(start, end, *, paths: CyclePaths | None = None, tickers=BMK_YIELD_TICKERS,
                      sources=BMK_YIELD_SOURCES) -> pd.DataFrame:
    """Rows for days in ``[start, end]`` (inclusive), every source."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    lo, hi = start - _LOOKBACK, end + _ONE_DAY
    parts = []
    for src in sources:
        if src == "otr":
            parts.append(_otr_pnl(tickers, lo, hi, paths))
        else:
            parts.append(yp.level_change_pnl(_yields(src, tickers, lo, hi, paths), f"yield_{src}",
                                             max_gap_days=BMK_YIELD_MAX_GAP_DAYS))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=yp.PNL_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def backfill_daily_yield_pnl(start, end, *, paths: CyclePaths | None = None, **kw) -> dict:
    paths = paths or CyclePaths.default()
    df = compute_yield_pnl(start, end, paths=paths, **kw)
    if len(df):
        out = df.copy()
        for col in ("timestamp", "prev_timestamp"):
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ms]")
        parquet_store.write_partitioned(out, _pnl_dir(paths), PNL_KEYS)
    by = df.groupby("bmk").size().to_dict() if len(df) else {}
    last = df.groupby("bmk")["timestamp"].max().dt.date.astype(str).to_dict() if len(df) else {}
    return {"rows": len(df), "by_bmk": by, "last_day": last}


# ------------------------------------------------------------------------ checks
def _read(ctx: StepContext) -> pd.DataFrame:
    df = parquet_store.read_partitioned(_pnl_dir(ctx.paths), start=ctx.start, end=ctx.end + _ONE_DAY)
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "bmk", "pnl_per_dv01"])
    df["bmk"], df["ticker"] = df["bmk"].astype(str), df["ticker"].astype(str)
    return df[df["bmk"].str.startswith("yield_")]


def check_yield_present(ctx: StepContext):
    """Every source has a row for every ticker on the window's latest day it published (warn):
    lists a source whose rows stop early - the curve until it joins the cycle."""
    out = ctx.output.get("yields", {})
    last = out.get("last_day", {})
    missing = [f"yield_{s}" for s in BMK_YIELD_SOURCES if f"yield_{s}" not in last]
    if not last:
        return False, "no yield P&L in the window", None
    newest = max(last.values())
    stale = {k: v for k, v in last.items() if v < newest}
    if not missing and not stale:
        return True, f"yield P&L for every source through {newest}", None
    return False, f"missing {missing}, behind {stale} (latest {newest})", None


def check_yield_sane(ctx: StepContext):
    df = _read(ctx)
    bad = df[df["pnl_per_dv01"].abs() > 75]
    return (bad.empty, "every daily yield move within 75bp" if bad.empty else f"{len(bad)} move(s) beyond 75bp",
            None if bad.empty else bad[["timestamp", "ticker", "bmk", "pnl_per_dv01"]])


def check_yield_sources_agree(ctx: StepContext):
    """The sources measure the same rate: a day where they disagree by more than
    ``BMK_YIELD_AGREE_BP`` (an on-the-run switch done wrong, a bad print, a bad curve fit) is
    listed (warn)."""
    df = _read(ctx)
    if df.empty:
        return True, "no yield rows", None
    w = df.pivot_table(index=["timestamp", "ticker"], columns="bmk", values="pnl_per_dv01")
    if w.shape[1] < 2:
        return True, "fewer than two sources in the window", None
    spread = w.max(axis=1) - w.min(axis=1)
    bad = spread[spread > BMK_YIELD_AGREE_BP]
    if bad.empty:
        return True, f"sources agree within {BMK_YIELD_AGREE_BP}bp on {len(w)} ticker-days", None
    return False, f"{len(bad)} ticker-day(s) where sources disagree by > {BMK_YIELD_AGREE_BP}bp", \
        w.loc[bad.index].assign(spread=bad).reset_index()
