"""Benchmark YIELD P&L, part of the ``bmk_pnl`` step (root CLAUDE.md 12; config
``BMK_YIELD_TICKERS`` / ``BMK_YIELD_SOURCES``): per ``US_BOND_<t>y`` ticker and source, the
day's price-action P&L in bp of a long position (``infra.processing.yield_pnl``), stored in
``Bmk/Pnl`` under bmk ``yield_cmt`` / ``yield_otr`` / ``yield_curve`` next to the futures'
``futures_price`` rows - ``pnl_per_dv01`` is bp for every bmk, so one column serves every
instrument. Local computation over stored data, no API:

* ``cmt``: the Treasury par curve (Daily/Bonds, the px step);
* ``otr``: the on-the-run bond's END OF DAY yield (the OTR map, ref step; FedInvest prices, px
  step), on the bond held the previous day;
* ``curve``: our fitted spline's par yield (Derived/TreasuryCurves).

Non-US (``BMK_YIELD_OFFICIAL``): ``UK_BOND_<t>y`` / ``DE_BOND_<t>y`` from each country's official
par curve in Daily/Bonds (the BoE's, the Bundesbank's), bmk ``yield_boe`` / ``yield_bundesbank``,
in the curve's currency. And ``DE_BOND_<t>y`` under ``yield_otr`` too: the German on-the-run
bond's Bundesbank yield (``infra.pipeline.bunds.otr_map``; an 11:15 Frankfurt snapshot), on the
bond held the previous day - its ISIN in the ``cusip`` / ``prev_cusip`` columns.

Upsert by ``timestamp, ticker, bmk``: a recomputed day replaces its row, a revised source
yield shows up in ``pnl_no_revisions``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.config import (BMK_YIELD_AGREE_BP, BMK_YIELD_MAX_GAP_DAYS, BMK_YIELD_OFFICIAL, BMK_YIELD_OFFICIAL_TENORS,
                          BMK_YIELD_SOURCES, BMK_YIELD_TICKERS, BOND_CURVES)
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


def _de_otr_pnl(start, end, paths: CyclePaths) -> pd.DataFrame:
    from infra.pipeline import bunds
    m = bunds.otr_map(start, end, depth=0, root=paths.de_auctions_dir)
    if m.empty:
        return pd.DataFrame(columns=yp.PNL_COLUMNS)
    m = m.assign(ticker="DE_BOND_" + m["tenor"], cusip=m["isin"])[["timestamp", "ticker", "cusip"]]
    px = bunds.read_bund_prices(start, end, isins=sorted(set(m["cusip"])), root=paths.bund_prices_dir)
    return yp.held_bond_pnl(m, px.rename(columns={"isin": "cusip"})[["timestamp", "cusip", "yield"]], "yield_otr",
                            max_gap_days=BMK_YIELD_MAX_GAP_DAYS, currency="EUR")


def official_tickers(country: str) -> list[str]:
    return [f"{country}_BOND_{t}y" for t in BMK_YIELD_OFFICIAL_TENORS]


def _official_pnl(lo, hi, paths: CyclePaths, official) -> list[pd.DataFrame]:
    from infra.pipeline.bonds import read_bonds_from_disk
    parts = []
    for country, src in official.items():
        df = read_bonds_from_disk(official_tickers(country), lo, hi, root=paths.daily_bonds_dir)
        if len(df):
            parts.append(yp.level_change_pnl(df.rename(columns={"par_yield": "yield"})[["timestamp", "ticker", "yield"]],
                                             f"yield_{src}", max_gap_days=BMK_YIELD_MAX_GAP_DAYS,
                                             currency=BOND_CURVES[country].currency))
    return parts


def compute_yield_pnl(start, end, *, paths: CyclePaths | None = None, tickers=BMK_YIELD_TICKERS,
                      sources=BMK_YIELD_SOURCES, official=BMK_YIELD_OFFICIAL) -> pd.DataFrame:
    """Rows for days in ``[start, end]`` (inclusive), every source - the US ones and each
    ``official`` country curve."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    lo, hi = start - _LOOKBACK, end + _ONE_DAY
    parts = []
    for src in sources:
        if src == "otr":
            parts.append(_otr_pnl(tickers, lo, hi, paths))
            parts.append(_de_otr_pnl(lo, hi, paths))
        else:
            parts.append(yp.level_change_pnl(_yields(src, tickers, lo, hi, paths), f"yield_{src}",
                                             max_gap_days=BMK_YIELD_MAX_GAP_DAYS))
    parts += _official_pnl(lo, hi, paths, official)
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=yp.PNL_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def input_last_days(start, end, *, paths: CyclePaths, tickers=BMK_YIELD_TICKERS) -> dict:
    """Per bmk, the latest day in ``[start, end]`` its INPUT published - what its P&L should
    reach: CMT (Daily/Bonds); FedInvest END OF DAY yields (``otr``); and for ``curve`` the same
    FedInvest day (the curve is fitted on those prices - a curve built short of it is stale)."""
    from infra.pipeline.treasury_prices import read_prices
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    out = {}
    cmt = _yields("cmt", tickers, lo, hi, paths)
    if len(cmt):
        out["yield_cmt"] = str(pd.Timestamp(cmt["timestamp"].max()).date())
    px = read_prices(lo, hi, root=paths.treasury_prices_dir)
    px = px.dropna(subset=["yield_eod"]) if len(px) else px
    if len(px):
        out["yield_otr"] = out["yield_curve"] = str(pd.Timestamp(px["timestamp"].max()).date())
    from infra.pipeline.bonds import read_bonds_from_disk
    for country, src in BMK_YIELD_OFFICIAL.items():
        b = read_bonds_from_disk(official_tickers(country), lo, hi, root=paths.daily_bonds_dir)
        b = b.dropna(subset=["par_yield"]) if len(b) else b
        if len(b):
            out[f"yield_{src}"] = str(pd.Timestamp(b["timestamp"].max()).date())
    return out


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
    return {"rows": len(df), "by_bmk": by, "last_day": last,
            "expected_last": input_last_days(start, end, paths=paths)}


# ------------------------------------------------------------------------ checks
def _read(ctx: StepContext) -> pd.DataFrame:
    df = parquet_store.read_partitioned(_pnl_dir(ctx.paths), start=ctx.start, end=ctx.end + _ONE_DAY)
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "bmk", "pnl_per_dv01"])
    df["bmk"], df["ticker"] = df["bmk"].astype(str), df["ticker"].astype(str)
    return df[df["bmk"].str.startswith("yield_")]


def check_yield_present(ctx: StepContext):
    """Each source's P&L reaches the latest day ITS OWN input published in the window (warn):
    the on-the-run and curve P&L legitimately trail CMT by a day (FedInvest posts day D's END OF
    DAY ~10:00 New York on D+1, after the 06:00 run), so sources are never compared with each
    other's dates. A curve built short of the prices it is fitted on shows here."""
    out = ctx.output.get("yields", {})
    last, expected = out.get("last_day", {}), out.get("expected_last", {})
    if not expected:
        return True, "no yield inputs in the window", None
    behind = {k: f"{last.get(k, 'none')} < {v}" for k, v in expected.items() if last.get(k, "") < v}
    if not behind:
        return True, "every source's yield P&L reaches its input's latest day: " + \
            ", ".join(f"{k} {v}" for k, v in sorted(expected.items())), None
    return False, "behind their input: " + "; ".join(f"{k} {v}" for k, v in sorted(behind.items())), None


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
