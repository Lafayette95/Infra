"""The Bank of England's SONIA OIS spot curve (root CLAUDE.md 16; client ``infra.api.
boe_client.fetch_ois_curve``): store, read, and discount-factor nodes for the GBP OIS curve's
short end. Free, daily since 2009; spot rates SEMI-ANNUALLY compounded (verified against
DTCC SONIA closes). Store ``Daily/BoeOIS`` (keys ``timestamp``, ``maturity`` in years;
``spot_pct``); coverage ``Daily/_coverage/boe_ois.parquet`` (only what the source covers -
a not-yet-published day is never claimed). Point in time: the curve for day D is fitted on
D's close, so a same-day 16:15 London use takes D-1's (``boe_ois_nodes``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.api import boe_client
from infra.config import DAILY_BOE_OIS_COVERAGE_FILE, DAILY_BOE_OIS_DIR
from infra.coverage.intervals import find_missing_ranges
from infra.storage import coverage_store, parquet_store

KEY = "BOE_OIS"
FETCH = boe_client.fetch_ois_curve  # network hook; tests stub it
FIRST_DAY = "2009-01-02"
_ONE_DAY = pd.Timedelta(days=1)


def plan(start, end, *, coverage_file: Path = DAILY_BOE_OIS_COVERAGE_FILE) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = max(pd.Timestamp(start).normalize(), pd.Timestamp(FIRST_DAY))
    end = pd.Timestamp(end).normalize() + _ONE_DAY
    if end <= start:
        return []
    return find_missing_ranges((start, end), coverage_store.read_covered(coverage_file, KEY))


def fetch_and_store(ranges, *, root: Path = DAILY_BOE_OIS_DIR, coverage_file: Path = DAILY_BOE_OIS_COVERAGE_FILE) -> int:
    """One fetch over the union of ``ranges`` (the workbooks are whole years / months
    anyway); rows stored, coverage recorded for what the source says it covers."""
    if not ranges:
        return 0
    lo, hi = min(a for a, _ in ranges), max(b for _, b in ranges)
    df, covered = FETCH(lo, hi)
    n = 0
    if len(df):
        out = pd.DataFrame({"timestamp": pd.to_datetime(df["timestamp"]).astype("datetime64[ms]"),
                            "maturity": df["maturity"].astype("float64"),
                            "spot_pct": (df["value"].astype("float64") * 10000).round().astype("Int32")})
        parquet_store.write_partitioned(out, root, ["timestamp", "maturity"])
        n = len(out)
    if covered:
        coverage_store.record_covered(coverage_file, KEY, covered)
    return n


def read_boe_ois(start=None, end=None, *, root: Path = DAILY_BOE_OIS_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end))
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "maturity", "spot_pct"])
    df["spot_pct"] = df["spot_pct"].astype("float64") / 10000
    return df.sort_values(["timestamp", "maturity"]).reset_index(drop=True)


def boe_ois_nodes(day, *, max_years: float, root: Path = DAILY_BOE_OIS_DIR, lag_days: int = 1,
                  lookback_days: int = 10) -> list[tuple[float, float]]:
    """(t, DF) nodes up to ``max_years`` from the latest BoE curve dated at least
    ``lag_days`` before ``day`` (point in time), semi-annual compounding. [] if none."""
    d = pd.Timestamp(day).normalize()
    df = read_boe_ois(d - pd.Timedelta(days=lookback_days), d - pd.Timedelta(days=lag_days) + _ONE_DAY, root=root)
    if df.empty:
        return []
    last = df[df["timestamp"] == df["timestamp"].max()]
    last = last[(last["maturity"] <= max_years) & last["spot_pct"].notna()]
    return [(float(m), float((1 + z / 200.0) ** (-2 * m))) for m, z in zip(last["maturity"], last["spot_pct"])]
