"""Pure interval arithmetic used to decide which date ranges are missing on disk.

All intervals are half-open ``[start, end)`` pairs of tz-naive UTC ``pd.Timestamp``
values aligned to whole days.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

Interval = tuple[pd.Timestamp, pd.Timestamp]


def to_utc_day(value) -> pd.Timestamp:
    """Normalise any date-like to a tz-naive UTC midnight timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def clamp_to_past(interval: Interval, now: pd.Timestamp | None = None) -> Interval | None:
    """Clip ``interval`` so it never reaches into today's (incomplete) UTC day."""
    today = to_utc_day(now if now is not None else pd.Timestamp.now(tz="UTC"))
    start, end = interval
    end = min(end, today)
    return (start, end) if start < end else None


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Union overlapping or touching intervals into a sorted, disjoint list."""
    merged: list[Interval] = []
    for start, end in sorted(i for i in intervals if i[0] < i[1]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(target: Interval, covered: Iterable[Interval]) -> list[Interval]:
    """Return the parts of ``target`` not covered by any interval in ``covered``."""
    missing: list[Interval] = []
    cursor, target_end = target
    for start, end in merge_intervals(covered):
        if end <= cursor:
            continue
        if start >= target_end:
            break
        if start > cursor:
            missing.append((cursor, start))
        cursor = max(cursor, end)
        if cursor >= target_end:
            break
    if cursor < target_end:
        missing.append((cursor, target_end))
    return missing


def find_missing_ranges(
    requested: Interval,
    covered: Iterable[Interval],
    now: pd.Timestamp | None = None,
) -> list[Interval]:
    """Missing sub-ranges of ``requested`` given prior coverage (past days only)."""
    clamped = clamp_to_past(requested, now)
    if clamped is None:
        return []
    return subtract_intervals(clamped, covered)
