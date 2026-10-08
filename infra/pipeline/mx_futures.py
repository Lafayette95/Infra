"""Montréal Exchange futures (Canada's bond futures: CGB 10y) - archive / store / read (root
CLAUDE.md 12; client ``infra.api.mx_client``, parsing ``infra.processing.mx_futures``).

* Raw archive ``MX_RAW_DIR/<root>/``: ``<year>Q<q>.csv`` for every COMPLETE past quarter (fetched
  once: the file is its coverage, Rule 2.1) and ``daily/<run day>.csv`` for the current year's
  window. The site returns at most ~500 rows a download, silently: a download at that cap is split
  in two (``ROW_CAP``), and a file holding several downloads separates them with a ``\x1e`` line.
* Store ``MX_FUTURES_DIR`` (keys ``timestamp`` = trading day, ``ticker``), upserted.
* ``front_series(root)``: per day the contract held (largest open interest the day before) with
  its settlement and its change since the previous day ON THE SAME CONTRACT - the continuous
  daily move (a roll is never a move).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import mx_client
from infra.config import MX_FUTURES_DIR, MX_FUTURES_ROOTS, MX_HISTORY_START, MX_RAW_DIR
from infra.processing import mx_futures as mf
from infra.storage import parquet_store

log = logging.getLogger(__name__)
KEYS = ["timestamp", "ticker"]
DAILY_LOOKBACK_DAYS = 10     # the current year's refresh re-reads this many days (late revisions)
FETCH = mx_client.fetch_historical   # network hook; tests stub it
_ONE_DAY = pd.Timedelta(days=1)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, "utf-8")
    tmp.replace(path)


def _store(df: pd.DataFrame, root: Path) -> int:
    if df.empty:
        return 0
    parquet_store.write_partitioned(mf.encode(df), root, KEYS)
    return len(df)


ROW_CAP = 495               # the site returns at most ~500 rows per download (verified 2026-10-08)


def _fetch(r: str, lo: pd.Timestamp, hi: pd.Timestamp) -> list[str]:
    """``[lo, hi]`` for root ``r``, split in two whenever a download comes back at the row cap
    (it would be silently truncated)."""
    text = FETCH(r, lo, hi)
    if text.count("\n") - 1 < ROW_CAP or hi <= lo:
        return [text]
    mid = (lo + (hi - lo) / 2).normalize()
    return _fetch(r, lo, mid) + _fetch(r, mid + _ONE_DAY, hi)


def backfill(roots=MX_FUTURES_ROOTS, *, start=MX_HISTORY_START, now=None, raw_root: Path = MX_RAW_DIR,
             root: Path = MX_FUTURES_DIR) -> dict:
    """Every complete past QUARTER since ``start`` (one request each, at the site's crawl delay;
    split further if it hits the row cap), each fetched once and archived as
    ``<root>/<year>Q<q>.csv``; the current year through ``update``."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    out = {}
    for r in roots:
        n = 0
        for q in pd.period_range(pd.Timestamp(start), pd.Timestamp(now.year, 1, 1) - _ONE_DAY, freq="Q"):
            path = Path(raw_root) / r / f"{q.year}Q{q.quarter}.csv"
            if path.exists():
                texts = path.read_text("utf-8").split("\n\x1e\n")
            else:
                lo = max(pd.Timestamp(start), q.start_time.normalize())
                texts = _fetch(r, lo, q.end_time.normalize())
                _write(path, "\n\x1e\n".join(texts))
            for t in texts:
                n += _store(mf.parse(t, r), root)
        out[r] = n
    out.update({f"{r} (current year)": v for r, v in update(roots, now=now, raw_root=raw_root, root=root)["rows"].items()})
    return out


def update(roots=MX_FUTURES_ROOTS, *, now=None, raw_root: Path = MX_RAW_DIR, root: Path = MX_FUTURES_DIR) -> dict:
    """The daily refresh: this year from the later of Jan 1 and the latest stored day minus
    ``DAILY_LOOKBACK_DAYS``, through yesterday; archived and upserted."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    end = now.normalize() - _ONE_DAY
    rows, days = {}, {}
    for r in roots:
        stored = read_mx_futures(pd.Timestamp(now.year, 1, 1), end, roots=[r], root=root)
        lo = pd.Timestamp(now.year, 1, 1)
        if len(stored):
            lo = max(lo, stored["timestamp"].max() - pd.Timedelta(days=DAILY_LOOKBACK_DAYS))
        texts = _fetch(r, lo, end)
        df = pd.concat([mf.parse(t, r) for t in texts], ignore_index=True).drop_duplicates(KEYS, keep="last")
        _write(Path(raw_root) / r / "daily" / f"{now:%Y-%m-%d}.csv", "\n\x1e\n".join(texts))
        rows[r] = _store(df, root)
        days[r] = None if df.empty else df["timestamp"].max()
    return {"rows": rows, "latest": days}


def read_mx_futures(start=None, end=None, *, roots=None, root: Path = MX_FUTURES_DIR) -> pd.DataFrame:
    if not parquet_store.has_data(root):
        return pd.DataFrame(columns=mf.COLUMNS)
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end) + _ONE_DAY,
                                        equals_in={"root": list(roots)} if roots else None)
    if df is None or df.empty:
        return pd.DataFrame(columns=mf.COLUMNS)
    return mf.decode(df)[mf.COLUMNS].sort_values(KEYS).reset_index(drop=True)


def front_series(root_symbol: str = "CGB", start=None, end=None, *, root: Path = MX_FUTURES_DIR) -> pd.DataFrame:
    """``timestamp, ticker, settlement, prev_settlement, change`` - the held contract each day
    and its settlement change on that same contract."""
    df = read_mx_futures(start, end, roots=[root_symbol], root=root)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "settlement", "prev_settlement", "change"])
    held = mf.front(df)
    s = df.pivot_table(index="timestamp", columns="ticker", values="settlement").sort_index()
    rows = []
    days = list(s.index)
    for prev, day in zip(days[:-1], days[1:]):
        tk = held.get(day)
        if tk is None or tk not in s:
            continue
        a, b = s.at[prev, tk], s.at[day, tk]
        rows.append((day, tk, b, a, b - a if pd.notna(a) and pd.notna(b) else float("nan")))
    return pd.DataFrame(rows, columns=["timestamp", "ticker", "settlement", "prev_settlement", "change"])
