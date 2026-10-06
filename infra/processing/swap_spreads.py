"""Benchmark swap-spread P&L in bp - pure (root CLAUDE.md 12/16).

Sign: + = LONG THE TREASURY AGAINST SWAPS (long the bond, pay fixed), DV01-matched - the
project's "+ = long the bond" convention (section 3) applied to the spread trade. Its
price-action P&L per unit of DV01 is the change in ``spread_bp`` = swap rate minus Treasury
yield (bp): the bond's yield falling and the swap rate rising both pay. For the on-the-run
par-par ASW series ``spread_bp`` = -ASW, so the same sign holds (a long asset-swap package
gains when its spread tightens).

* ``level_pnl``: a constant-maturity series (CMT swap spread) - the day's change of the
  series itself.
* ``held_bond_pnl``: the on-the-run series - the day's change on the bond held the PREVIOUS
  day (``prev_cusip``): a switch to a new issue is never a move.
A change spanning more than ``max_gap_days`` gets no row. Price action only, no carry.
"""
from __future__ import annotations

import pandas as pd

PNL_COLUMNS = ["timestamp", "ticker", "bmk", "currency", "level_bp", "prev_timestamp", "prev_level_bp", "cusip",
               "prev_cusip", "pnl", "pnl_per_dv01"]


def _finish(df: pd.DataFrame, bmk: str, max_gap_days: int) -> pd.DataFrame:
    gap = (pd.to_datetime(df["timestamp"]) - pd.to_datetime(df["prev_timestamp"])).dt.days
    df = df[df["prev_level_bp"].notna() & df["level_bp"].notna() & (gap <= max_gap_days)].copy()
    df["pnl_per_dv01"] = df["level_bp"] - df["prev_level_bp"]
    df["pnl"] = df["pnl_per_dv01"]  # per $1 of DV01 the $ P&L is the bp number
    df["bmk"], df["currency"] = bmk, "USD"
    for c in ("cusip", "prev_cusip"):
        if c not in df:
            df[c] = None
    return df[PNL_COLUMNS].sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def level_pnl(levels: pd.DataFrame, bmk: str, *, max_gap_days: int = 7) -> pd.DataFrame:
    """``levels``: ``timestamp, ticker, spread_bp``."""
    if levels.empty:
        return pd.DataFrame(columns=PNL_COLUMNS)
    y = levels.dropna(subset=["spread_bp"]).sort_values(["ticker", "timestamp"]).copy()
    y["level_bp"] = y["spread_bp"]
    g = y.groupby("ticker")
    y["prev_timestamp"], y["prev_level_bp"] = g["timestamp"].shift(1), g["level_bp"].shift(1)
    return _finish(y, bmk, max_gap_days)


def held_bond_pnl(bonds: pd.DataFrame, bmk: str, *, max_gap_days: int = 7) -> pd.DataFrame:
    """``bonds``: ``timestamp, ticker, cusip, rank, spread_bp`` - each day's on-the-run
    (rank 0) and other ranks' bonds. Day D's P&L = the move from the previous day to D of the
    bond on the run the previous day; its own spread on D must exist."""
    if bonds.empty:
        return pd.DataFrame(columns=PNL_COLUMNS)
    b = bonds.dropna(subset=["spread_bp"]).assign(cusip=lambda x: x["cusip"].astype(str))
    level = b.drop_duplicates(["timestamp", "cusip"]).set_index(["timestamp", "cusip"])["spread_bp"]
    o = b[b["rank"] == 0].sort_values(["ticker", "timestamp"]).copy()
    g = o.groupby("ticker")
    o["prev_timestamp"], o["prev_cusip"] = g["timestamp"].shift(1), g["cusip"].shift(1)
    o["level_bp"] = level.reindex(pd.MultiIndex.from_arrays([o["timestamp"], o["prev_cusip"].fillna("")])).to_numpy()
    o["prev_level_bp"] = level.reindex(pd.MultiIndex.from_arrays([o["prev_timestamp"], o["prev_cusip"].fillna("")])).to_numpy()
    return _finish(o, bmk, max_gap_days)
