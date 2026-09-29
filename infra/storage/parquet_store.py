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


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dot-prefixed so pyarrow's dataset discovery (which skips "." / "_" names) never
    # reads an in-flight or orphaned temp file as data - a concurrent reader (the
    # dashboard during a cycle write) or a run killed mid-write would otherwise see it.
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(
        tmp,
        engine="pyarrow",
        index=False,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )
    os.replace(tmp, path)


def delete_where(root: Path, predicate) -> int:
    """Delete every row, across all year/quarter files under ``root``, for which
    ``predicate(frame) -> bool Series`` is True; empty files are removed. Returns rows
    removed. Each file is rewritten atomically."""
    if not has_data(root):
        return 0
    removed = 0
    for path in sorted(root.rglob(PARQUET_FILE_NAME)):
        part = pd.read_parquet(path, engine="pyarrow")
        drop = predicate(part).to_numpy(dtype=bool)
        if not drop.any():
            continue
        removed += int(drop.sum())
        if drop.all():
            path.unlink()
        else:
            _atomic_write(part[~drop].reset_index(drop=True), path)
    return removed


def prune_rows(root: Path, column: str, values: Iterable) -> int:
    """Delete EVERY row whose ``column`` is in ``values``, across all year/quarter files
    under ``root`` (not just the partitions a later write touches). Returns rows removed.

    Storage-level only: a caller that also tracks coverage must reset it for the pruned
    keys, or the manifest will still claim history that no longer exists on disk (see
    infra.pipeline.daily.fetch_and_store_daily).
    """
    values = list(values)

    def match(part: pd.DataFrame) -> pd.Series:
        if column not in part.columns:
            return pd.Series(False, index=part.index)
        col = part[column]
        if pd.api.types.is_datetime64_any_dtype(col):
            # typed, not string, comparison: astype(str) renders an all-midnight column
            # as "2026-09-23" while str(Timestamp) gives "2026-09-23 00:00:00" - a string
            # match silently prunes nothing
            return col.isin(pd.to_datetime(values))
        return col.astype(str).isin({str(v) for v in values})

    return delete_where(root, match)


def write_partitioned(
    df: pd.DataFrame,
    root: Path,
    key_columns: list[str],
    *,
    prune: bool = False,
    prune_column: str = "ticker",
    coalesce: bool = False,
) -> list[Path]:
    """Merge encoded rows into their year/quarter files, de-duplicating on ``key_columns``.

    ``prune=False`` (default): append - existing history is kept, and only rows sharing
    a key with an incoming row are replaced by it. ``coalesce=True``: a key conflict is
    merged COLUMN BY COLUMN instead - an incoming value wins, but a MISSING incoming value
    never erases an existing one. For stores whose rows are pivots of independently
    published fields (the daily statistics: settlement and open interest arrive in
    different sessions), where a partial row is a partial observation, not a retraction.
    ``prune=True``: first delete the ENTIRE
    stored history of every instrument (``prune_column`` value, a ticker by default)
    present in ``df``, then write ``df`` - the instrument's history becomes exactly the
    incoming rows. Each file is written to a temp path and atomically moved into place.
    """
    if df.empty:
        return []
    if prune:
        prune_rows(root, prune_column, df[prune_column].unique())
    written: list[Path] = []
    df = add_partition_columns(df)
    for (year, quarter), part in df.groupby(_PARTITION_COLS, sort=True):
        part = part.drop(columns=_PARTITION_COLS)
        path = root / f"year={year}" / f"quarter={quarter}" / PARQUET_FILE_NAME
        if path.exists():
            existing = pd.read_parquet(path, engine="pyarrow")
            part = pd.concat([existing, part], ignore_index=True)
        if coalesce:
            # groupby().last() takes, per column, the last NON-null value: the incoming
            # row's where it has one, the existing row's where it doesn't
            columns = list(part.columns)
            part = part.groupby(key_columns, sort=True).last().reset_index()[columns]
        part = (
            part.drop_duplicates(subset=key_columns, keep="last")
            .sort_values(key_columns)
            .reset_index(drop=True)
        )
        _atomic_write(part, path)
        written.append(path)
    return written


def list_values(root: Path, column: str) -> list[str]:
    """Distinct values of one column (e.g. tickers on disk), reading only that column."""
    if not has_data(root):
        return []
    table = ds.dataset(root, format="parquet", partitioning="hive").to_table(columns=[column])
    return sorted(pd.unique(table.column(column).to_pandas().astype(str)))
