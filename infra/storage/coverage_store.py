"""Persistent record of which (key, date range) were already queried from the API.

Why: a day with no data (holiday) leaves no rows on disk, so inspecting the data
alone would re-query it forever and re-charge the wallet (Rule 2.1). The manifest
records the *request*, not the rows. ``key`` is a ticker (futures) or an
instrument id (options).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.coverage.intervals import Interval, merge_intervals

_COLUMNS = ["key", "start", "end"]


def read_covered(path: Path, key: str) -> list[Interval]:
    """Already-queried ``[start, end)`` intervals for one key."""
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    rows = df[df["key"] == str(key)]
    return merge_intervals(
        (pd.Timestamp(s), pd.Timestamp(e)) for s, e in zip(rows["start"], rows["end"])
    )


def read_all(path: Path) -> pd.DataFrame:
    """The whole manifest (flat frame), empty if none exists yet."""
    if not path.exists():
        return pd.DataFrame({
            "key": pd.Series(dtype="str"),
            "start": pd.Series(dtype="datetime64[ms]"),
            "end": pd.Series(dtype="datetime64[ms]"),
        })
    return pd.read_parquet(path)


def record_covered(path: Path, key: str, intervals: list[Interval]) -> None:
    """Add intervals for ``key`` and compact the manifest by merging per key."""
    new = pd.DataFrame(
        [(str(key), s, e) for s, e in intervals], columns=_COLUMNS
    ).astype({"start": "datetime64[ms]", "end": "datetime64[ms]"})
    frames = [read_all(path), new]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    compact = [
        (k, s, e)
        for k, grp in df.groupby("key", sort=True)
        for s, e in merge_intervals(zip(grp["start"], grp["end"]))
    ]
    out = pd.DataFrame(compact, columns=_COLUMNS).astype(
        {"start": "datetime64[ms]", "end": "datetime64[ms]"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, engine="pyarrow", index=False, compression="zstd", compression_level=5)
    tmp.replace(path)
