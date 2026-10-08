"""Lag-aware joins on the benchmark P&L (root CLAUDE.md 12): WHEN each row's interval starts
and ends, and which decision it belongs to - so a cross-market backtest on series marked at
different times (MoF 15:00 Tokyo, Bundesbank 11:15 Frankfurt, CMT 15:30 New York, ...) never
books an interval that started before the decision. Disk only.

* ``mark_spec(bmk, ticker)``: the market instant a series is marked at (``BMK_MARK_TIMES``,
  futures by their settlement time, ``...@LDN1615`` at the snap).
* ``timed_pnl(bmk, ticker, start, end)``: the stored rows with ``prev_mark_at`` / ``mark_at`` (UTC)
  - day D's row covers the move from the previous row's mark to D's mark.
* ``earned(rows, positions)``: positions are labelled by DECISION instant (root CLAUDE.md 3,
  time conventions); each row goes to the latest decision at or before its interval START
  (``prev_mark_at``) - a row whose interval began before a decision belongs to the position
  held before it, even if it ends after. The cost of trading only at the marks: a decision
  between two marks misses the part of the move between it and the next mark, never sees it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.config import (BMK_MARK_TIMES, BMK_ROOT, BMK_SYNC_SNAP, FUTURES_ROOTS, SETTLEMENT_MARK_TIMES,
                          SWAP_CLOSES)
from infra.storage import parquet_store
from infra.trading_calendar import snap_instants


def _issuer(ticker: str) -> str:
    return ticker.split("_")[0]


def mark_spec(bmk: str, ticker: str) -> tuple[str, str]:
    """``(local time, IANA zone)`` of the instant ``bmk`` / ``ticker`` is marked at."""
    if bmk.endswith(f"@{BMK_SYNC_SNAP}"):
        s = SWAP_CLOSES[BMK_SYNC_SNAP]
        return s.local_time, s.timezone
    if bmk == "futures_price":
        root = next((r for r in sorted(FUTURES_ROOTS, key=len, reverse=True) if ticker.startswith(r)), None)
        if root is None:
            raise KeyError(f"no futures root for {ticker!r}")
        return SETTLEMENT_MARK_TIMES[FUTURES_ROOTS[root].dataset]
    by_issuer = BMK_MARK_TIMES.get(bmk)
    if by_issuer is None or _issuer(ticker) not in by_issuer:
        raise KeyError(f"no mark time for bmk {bmk!r} / {ticker!r} (BMK_MARK_TIMES)")
    return by_issuer[_issuer(ticker)]


def mark_instants(bmk: str, ticker: str, days) -> pd.DatetimeIndex:
    """UTC instants of the marks on the (local) days ``days``."""
    lt, tz = mark_spec(bmk, ticker)
    return pd.DatetimeIndex(snap_instants(pd.DatetimeIndex(days).normalize(), lt, tz))


def timed_pnl(bmk: str, ticker: str, start, end, *, root: Path = BMK_ROOT) -> pd.DataFrame:
    """``timestamp, prev_mark_at, mark_at, pnl_per_dv01, pnl`` for one stored series over days
    ``[start, end]``."""
    df = parquet_store.read_partitioned(root / "Pnl", start=pd.Timestamp(start), end=pd.Timestamp(end) + pd.Timedelta(days=1),
                                        equals_in={"bmk": [bmk], "ticker": [ticker]})
    cols = ["timestamp", "prev_mark_at", "mark_at", "pnl_per_dv01", "pnl"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    df = df.sort_values("timestamp")
    out = pd.DataFrame({"timestamp": pd.to_datetime(df["timestamp"]).to_numpy(),
                        "prev_mark_at": mark_instants(bmk, ticker, pd.to_datetime(df["prev_timestamp"])),
                        "mark_at": mark_instants(bmk, ticker, pd.to_datetime(df["timestamp"])),
                        "pnl_per_dv01": df["pnl_per_dv01"].astype(float).to_numpy(),
                        "pnl": df["pnl"].astype(float).to_numpy()})
    return out.reset_index(drop=True)


def earned(rows: pd.DataFrame, positions: pd.Series, value: str = "pnl_per_dv01") -> pd.DataFrame:
    """``rows`` (``timed_pnl``) with ``decision_at`` (the latest decision at or before each
    row's interval start), ``position`` and ``earned`` = position x ``value``. ``positions``:
    UTC decision instant -> position (+ = long duration). Rows before the first decision earn
    nothing (position 0)."""
    out = rows.copy()
    if out.empty:
        return out.assign(decision_at=pd.NaT, position=np.nan, earned=np.nan)
    pos = positions.sort_index()
    t = pd.DatetimeIndex(pos.index).to_numpy(dtype="datetime64[ns]")
    k = np.searchsorted(t, pd.DatetimeIndex(out["prev_mark_at"]).to_numpy(dtype="datetime64[ns]"), side="right") - 1
    picked = pd.Series(t[np.clip(k, 0, None)]) if len(t) else pd.Series(pd.NaT, index=range(len(out)))
    out["decision_at"] = pd.DatetimeIndex(picked.where(pd.Series(k >= 0)))
    out["position"] = np.where(k >= 0, pos.to_numpy(dtype=float)[np.clip(k, 0, None)], 0.0)
    out["earned"] = out["position"] * out[value].astype(float)
    return out
