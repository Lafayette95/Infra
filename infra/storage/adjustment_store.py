"""The adjustments log: a sidecar record of every value the pipeline TOUCHED - a bad print
set to NA or rolled forward, and whatever else comes later (intraday cleaning, manual
fixes). Store-agnostic; CLAUDE.md section 12.

Why a sidecar and not a flag column on each store: the data stores keep exactly what the
vendor delivered (so a re-fetch can never silently undo a treatment, and revision checks
keep comparing real vendor data); the log holds what a column couldn't - the ORIGINAL
value, what it became, why, which process did it, and when - identically for any store
(daily settlements today, options and intraday bars later), with no schema change to any
of them. Readers overlay it at read time (``apply``); raw stays one flag away.

One row per touched value, keyed ``(store, timestamp, key, column)``; hive-partitioned by
the data's ``timestamp`` like every other store.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.storage import parquet_store

COLUMNS = ["store", "timestamp", "key", "column", "action", "original", "adjusted",
           "source", "detail", "run_day"]
KEYS = ["store", "timestamp", "key", "column"]


def empty() -> pd.DataFrame:
    return pd.DataFrame({
        "store": pd.Series(dtype="str"), "timestamp": pd.Series(dtype="datetime64[ms]"),
        "key": pd.Series(dtype="str"), "column": pd.Series(dtype="str"),
        "action": pd.Series(dtype="str"), "original": pd.Series(dtype="float64"),
        "adjusted": pd.Series(dtype="float64"), "source": pd.Series(dtype="str"),
        "detail": pd.Series(dtype="str"), "run_day": pd.Series(dtype="datetime64[ms]"),
    })[COLUMNS]


def record(root: Path, rows: pd.DataFrame) -> None:
    """Upsert adjustment rows (a later adjustment of the same value replaces the earlier)."""
    if rows.empty:
        return
    out = rows[COLUMNS].copy()
    for col in ("timestamp", "run_day"):
        out[col] = pd.to_datetime(out[col]).astype("datetime64[ms]")
    for col in ("original", "adjusted"):
        out[col] = out[col].astype("float64")
    parquet_store.write_partitioned(out, root, KEYS)


def clear(root: Path, *, store: str, source: str, keys, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Remove ``source``'s adjustments of ``store`` for ``keys`` with timestamp in
    ``[start, end]`` (inclusive) - so a process that re-evaluates a window replaces its own
    earlier verdicts (e.g. a print later corrected upstream is no longer adjusted)."""
    keys = {str(k) for k in keys}
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)

    def match(part: pd.DataFrame) -> pd.Series:
        ts = pd.to_datetime(part["timestamp"])
        return ((part["store"] == store) & (part["source"] == source)
                & part["key"].astype(str).isin(keys) & (ts >= lo) & (ts <= hi))

    return parquet_store.delete_where(root, match)


def read(root: Path, *, store: str, start=None, end=None, keys=None) -> pd.DataFrame:
    """Adjustments of ``store`` (optionally within ``[start, end)`` / for ``keys``)."""
    equals_in = {"store": [store]}
    if keys is not None:
        equals_in["key"] = [str(k) for k in keys]
    df = parquet_store.read_partitioned(root, start=start, end=end, equals_in=equals_in)
    if df is None or df.empty:
        return empty()
    return df[COLUMNS].reset_index(drop=True)


def apply(df: pd.DataFrame, adjustments: pd.DataFrame, *, key_column: str,
          timestamp_column: str = "timestamp") -> pd.DataFrame:
    """``df`` with every adjusted value overlaid (NaN where the action was NA). Rows and
    columns are unchanged otherwise; a copy is returned, ``df`` itself is never mutated."""
    if adjustments.empty or df.empty:
        return df
    out = df.copy()
    keys = out[key_column].astype(str).to_numpy()
    stamps = pd.to_datetime(out[timestamp_column]).to_numpy()
    for column, adj in adjustments.groupby("column"):
        if column not in out.columns:
            continue
        lookup = dict(zip(zip(adj["key"].astype(str), pd.to_datetime(adj["timestamp"]).to_numpy()),
                          adj["adjusted"].astype("float64")))
        hit = np.array([(k, t) in lookup for k, t in zip(keys, stamps)])
        if not hit.any():
            continue
        values = out[column].astype("float64").to_numpy(copy=True)
        values[hit] = [lookup[(k, t)] for k, t, h in zip(keys, stamps, hit) if h]
        out[column] = values
    return out
