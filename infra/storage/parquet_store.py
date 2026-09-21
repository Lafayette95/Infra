"""Hive-partitioned parquet read/write (year/quarter). No API or plotting code here.

Files are flat (``index=False``), ZSTD level 5, and never partitioned by ticker/day.
Reads push predicates down through a PyArrow dataset before touching pandas.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from infra.config import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_FILE_NAME,
)
from infra.processing.transforms import add_partition_columns

_PARTITION_COLS = ["year", "quarter"]


def has_data(root: Path) -> bool:
    """True if ``root`` contains at least one parquet file."""
    return root.exists() and any(root.rglob("*.parquet"))


def build_filter(
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    equals_in: dict[str, Iterable] | None = None,
) -> ds.Expression | None:
    """Build a PyArrow expression: ``start <= timestamp < end`` AND column IN values.

    Year predicates are added so whole partitions are pruned before any file is read.
    """
    expr: ds.Expression | None = None

    def _and(current, new):
        return new if current is None else current & new

    if start is not None:
        start = pd.Timestamp(start)
        expr = _and(expr, ds.field("year") >= start.year)
        expr = _and(expr, ds.field("timestamp") >= pa.scalar(start.to_pydatetime(), pa.timestamp("ms")))
    if end is not None:
        end = pd.Timestamp(end)
        expr = _and(expr, ds.field("year") <= end.year)
        expr = _and(expr, ds.field("timestamp") < pa.scalar(end.to_pydatetime(), pa.timestamp("ms")))
    for column, values in (equals_in or {}).items():
        expr = _and(expr, ds.field(column).isin(list(values)))
    return expr


def read_partitioned(
    root: Path,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    equals_in: dict[str, Iterable] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """Read the encoded (still fixed-point) rows matching the filters.

    Returns ``None`` when ``root`` holds no data so callers can distinguish
    "nothing on disk" from "nothing matched".
    """
    if not has_data(root):
        return None
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    table = dataset.to_table(
        columns=columns,
        filter=build_filter(start=start, end=end, equals_in=equals_in),
    )
    df = table.to_pandas()
    return df.drop(columns=[c for c in _PARTITION_COLS if c in df.columns])


def write_partitioned(df: pd.DataFrame, root: Path, key_columns: list[str]) -> list[Path]:
    """Merge encoded rows into their year/quarter files, de-duplicating on ``key_columns``.

    Existing rows with the same key are replaced by the new ones. Each file is
    written to a temp path and atomically moved into place.
    """
    if df.empty:
        return []
    written: list[Path] = []
    df = add_partition_columns(df)
    for (year, quarter), part in df.groupby(_PARTITION_COLS, sort=True):
        part = part.drop(columns=_PARTITION_COLS)
        path = root / f"year={year}" / f"quarter={quarter}" / PARQUET_FILE_NAME
        if path.exists():
            existing = pd.read_parquet(path, engine="pyarrow")
            part = pd.concat([existing, part], ignore_index=True)
        part = (
            part.drop_duplicates(subset=key_columns, keep="last")
            .sort_values(key_columns)
            .reset_index(drop=True)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        part.to_parquet(
            tmp,
            engine="pyarrow",
            index=False,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
        )
        os.replace(tmp, path)
        written.append(path)
    return written


def list_values(root: Path, column: str) -> list[str]:
    """Distinct values of one column (e.g. tickers on disk), reading only that column."""
    if not has_data(root):
        return []
    table = ds.dataset(root, format="parquet", partitioning="hive").to_table(columns=[column])
    return sorted(pd.unique(table.column(column).to_pandas().astype(str)))
