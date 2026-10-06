"""Benchmark yield P&L in bp (root CLAUDE.md 12, bmk): pure. Sign: + = a LONG position (long the
bond / duration) made money, i.e. pnl_bp = -(yield change) x 100 with yields in percent - so a
position's P&L is position x pnl_bp, whatever the instrument (user decision 2026-10-05).

* ``level_change_pnl``: a constant-maturity series (CMT, our curve's par point): the day's
  change of the series itself. Price action only - no carry, no rolldown.
* ``held_bond_pnl``: an on-the-run series: the day's change of the bond held the PREVIOUS day
  (``prev_cusip``), so the switch to a new issue is never a move (the new bond's yield differs
  from the old one's by a few bp - a fake jump at every new issue otherwise).
A change spanning more than ``max_gap_days`` (missing data) gets no row: not one day's P&L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PNL_COLUMNS = ["timestamp", "ticker", "bmk", "currency", "yield", "prev_timestamp", "prev_yield", "cusip",
               "prev_cusip", "pnl", "pnl_per_dv01"]


def _finish(df: pd.DataFrame, bmk: str, max_gap_days: int) -> pd.DataFrame:
    gap = (pd.to_datetime(df["timestamp"]) - pd.to_datetime(df["prev_timestamp"])).dt.days
    df = df[df["prev_yield"].notna() & df["yield"].notna() & (gap <= max_gap_days)].copy()
    df["pnl_per_dv01"] = -(df["yield"] - df["prev_yield"]) * 100.0
    df["pnl"] = df["pnl_per_dv01"]          # per $1 of DV01 the $ P&L is the bp number
    df["bmk"], df["currency"] = bmk, "USD"
    for c in ("cusip", "prev_cusip"):
        if c not in df:
            df[c] = None
    return df[PNL_COLUMNS].sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def level_change_pnl(yields: pd.DataFrame, bmk: str, *, max_gap_days: int = 7) -> pd.DataFrame:
    """``yields``: ``timestamp, ticker, yield`` (percent)."""
    if yields.empty:
        return pd.DataFrame(columns=PNL_COLUMNS)
    y = yields.dropna(subset=["yield"]).sort_values(["ticker", "timestamp"]).copy()
    g = y.groupby("ticker")
    y["prev_timestamp"], y["prev_yield"] = g["timestamp"].shift(1), g["yield"].shift(1)
    return _finish(y, bmk, max_gap_days)


def held_bond_pnl(otr: pd.DataFrame, bond_yields: pd.DataFrame, bmk: str, *, max_gap_days: int = 7) -> pd.DataFrame:
    """``otr``: ``timestamp, ticker, cusip`` (the on-the-run bond per day); ``bond_yields``:
    ``timestamp, cusip, yield`` for every CUSIP. Day D's P&L = the move from D-1 to D of the
    bond on the run on D-1 (``prev_cusip``); its own yield on D must exist."""
    if otr.empty or bond_yields.empty:
        return pd.DataFrame(columns=PNL_COLUMNS)
    o = otr.sort_values(["ticker", "timestamp"]).copy()
    o["cusip"] = o["cusip"].astype(str)
    g = o.groupby("ticker")
    o["prev_timestamp"], o["prev_cusip"] = g["timestamp"].shift(1), g["cusip"].shift(1)
    by = bond_yields.assign(cusip=bond_yields["cusip"].astype(str)).dropna(subset=["yield"])
    by = by.drop_duplicates(["timestamp", "cusip"]).set_index(["timestamp", "cusip"])["yield"]
    o["yield"] = by.reindex(pd.MultiIndex.from_arrays([o["timestamp"], o["prev_cusip"].fillna("")])).to_numpy()
    o["prev_yield"] = by.reindex(pd.MultiIndex.from_arrays([o["prev_timestamp"], o["prev_cusip"].fillna("")])).to_numpy()
    return _finish(o, bmk, max_gap_days)
