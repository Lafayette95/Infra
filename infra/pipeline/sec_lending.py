"""NY Fed Treasury securities lending (CLAUDE.md 19): plan / fetch / store / read +
``load_sec_lending`` over ``Daily/SecLending``.

Coverage key ``nyfed_seclending`` (``Daily/_coverage/sec_lending``), per business day:
uncovered days are requested as ranges of at most ``MAX_REQUEST_DAYS``, and a day is
claimed covered once ``REPO_SETTLE_DAYS`` old (extensions post the next morning). A day
without operations (a holiday) is just covered - the program has run every business day.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.api import nyfed_client
from infra.config import REPO_SETTLE_DAYS, SEC_LENDING_COVERAGE_FILE, SEC_LENDING_DIR, SEC_LENDING_START
from infra.pipeline.repo import _today, uncovered_span
from infra.processing import sec_lending as sl
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)
KEY = "nyfed_seclending"
FETCH: Callable = nyfed_client.fetch_sec_lending  # (start, end) -> operations; tests replace it
MAX_REQUEST_DAYS = 366
_ONE_DAY = pd.Timedelta(days=1)


def plan_sec_lending_update(start, end, *, now=None, coverage_file: Path = SEC_LENDING_COVERAGE_FILE,
                            force_refetch: bool = False) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Ranges ``(first, last)`` (inclusive) to request for ``[start, end]``, never past
    yesterday. No network."""
    today = _today(now)
    end = min(pd.Timestamp(end).normalize(), today - _ONE_DAY)
    span = uncovered_span(KEY, max(pd.Timestamp(start), pd.Timestamp(SEC_LENDING_START)), end,
                          coverage_file=coverage_file, force_refetch=force_refetch)
    if span is None:
        return []
    out, first = [], span[0]
    while first <= span[1]:
        last = min(first + pd.Timedelta(days=MAX_REQUEST_DAYS - 1), span[1])
        out.append((first, last))
        first = last + _ONE_DAY
    return out


def store_sec_lending(rows: pd.DataFrame, *, root: Path = SEC_LENDING_DIR) -> int:
    """FILES ONLY: upsert rows (keys ``LENDING_KEYS``)."""
    if rows.empty:
        return 0
    parquet_store.write_partitioned(sl.encode(rows), root, sl.LENDING_KEYS)
    return len(rows)


def fetch_and_store_sec_lending(ranges, *, now=None, root: Path = SEC_LENDING_DIR,
                                coverage_file: Path = SEC_LENDING_COVERAGE_FILE) -> dict:
    """Run the planned ranges; per-range failures collected, not raised."""
    settled = _today(now) - pd.Timedelta(days=REPO_SETTLE_DAYS)
    out = {"rows": 0, "days": 0, "errors": {}}
    for first, last in ranges:
        try:
            rows = sl.parse_operations(FETCH(first, last))
            out["rows"] += store_sec_lending(rows, root=root)
            out["days"] += rows["timestamp"].nunique() if len(rows) else 0
            cover_to = min(pd.Timestamp(last), settled)
            if cover_to >= first:
                coverage_store.record_covered(coverage_file, KEY, [(pd.Timestamp(first), cover_to + _ONE_DAY)])
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            out["errors"][f"{pd.Timestamp(first).date()}..{pd.Timestamp(last).date()}"] = msg
            log.warning("securities lending %s..%s failed: %s", pd.Timestamp(first).date(), pd.Timestamp(last).date(), msg)
    return out


def read_sec_lending(start, end, *, cusips=None, operation_type: str | None = "lending",
                     security_class: str | None = "treasury", root: Path = SEC_LENDING_DIR) -> pd.DataFrame:
    """Decoded rows with ``timestamp`` in ``[start, end)``; by default only the lending
    auctions (``operation_type=None`` for extensions too) of Treasuries
    (``security_class=None`` for agency debt too). No network."""
    eq = {}
    if security_class is not None:
        eq["security_class"] = [security_class]
    if cusips is not None:
        eq["cusip"] = list(cusips)
    if operation_type is not None:
        eq["operation_type"] = [operation_type]
    raw = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end), equals_in=eq or None)
    if raw is None or raw.empty:
        return sl._empty()
    return sl.decode(raw[sl.LENDING_COLUMNS]).sort_values(sl.LENDING_KEYS).reset_index(drop=True)


def load_sec_lending(start, end, *, cusips=None, operation_type: str | None = "lending",
                     security_class: str | None = "treasury", fetch_missing: bool = True,
                     root: Path = SEC_LENDING_DIR, coverage_file: Path = SEC_LENDING_COVERAGE_FILE) -> pd.DataFrame:
    """Parent: fetch whatever ``[start, end)`` lacks, then read."""
    if fetch_missing:
        ranges = plan_sec_lending_update(start, pd.Timestamp(end) - _ONE_DAY, coverage_file=coverage_file)
        if ranges:
            fetch_and_store_sec_lending(ranges, root=root, coverage_file=coverage_file)
    return read_sec_lending(start, end, cusips=cusips, operation_type=operation_type,
                            security_class=security_class, root=root)
