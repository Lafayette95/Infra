"""Repo rates (CLAUDE.md 19): plan / fetch / store / read + ``load_repo`` over
``Daily/Repo``. Three sources, each with its own coverage key in ``Daily/_coverage/repo``:

* ``nyfed`` (SOFR, TGCR, BGCR; key ``nyfed:<RATE>``): uncovered business days requested
  as one range per rate; a day is claimed covered once ``REPO_SETTLE_DAYS`` old (the NY
  Fed can revise it the afternoon after publication).
* ``ofr`` (key ``ofr``): ONE request for every configured series from the first day not
  yet FINAL. Preliminary rows are stored as they come and stay next to the finals; days
  are claimed covered only through the last final day, so the recent ~3 months are asked
  again each run until OFR finalises them.
* ``dtcc_gcf`` (key ``dtcc_gcf``): the frozen 2005-2024 workbook, fetched once.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.api import dtcc_client, nyfed_client, ofr_client
from infra.config import (
    DTCC_GCF_SPAN,
    NYFED_REPO_RATES,
    NYFED_REPO_START,
    OFR_FINAL_REFERENCE,
    OFR_REPO_BUCKETS,
    OFR_REPO_START,
    REPO_COVERAGE_FILE,
    REPO_DIR,
    REPO_SETTLE_DAYS,
)
from infra.coverage.intervals import find_missing_ranges, to_utc_day
from infra.processing import repo as rp
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)
# Network hooks (module-level so tests replace them; tests/conftest.py)
NYFED_FETCH: Callable = nyfed_client.fetch_reference_rates      # (rate, start, end) -> records
OFR_FETCH: Callable = ofr_client.fetch_series                   # (mnemonics, start) -> payload
GCF_FETCH: Callable = dtcc_client.fetch_gcf_index_workbook      # () -> xlsx bytes
SOURCES = ("nyfed", "ofr", "dtcc_gcf")


def _today(now) -> pd.Timestamp:
    return to_utc_day(pd.Timestamp.now(tz="UTC") if now is None else now)


def uncovered_span(key: str, start, end, *, coverage_file: Path, force_refetch: bool = False):
    """``(first, last)`` business days in ``[start, end]`` not covered under ``key`` (the
    whole span with ``force_refetch``), or None. No network."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end < start:
        return None
    days = pd.bdate_range(start, end)
    if not force_refetch:
        gaps = find_missing_ranges((start, end + _ONE_DAY), coverage_store.read_covered(coverage_file, key))
        days = [d for d in days if any(g0 <= d < g1 for g0, g1 in gaps)]
    return (days[0], days[-1]) if len(days) else None


def plan_repo_update(start, end, *, now=None, coverage_file: Path = REPO_COVERAGE_FILE,
                     force_refetch: bool = False, sources=SOURCES) -> dict[str, tuple]:
    """``{key: (first, last)}`` to request, per source key, for ``[start, end]``
    (inclusive, never past yesterday). No network - also the dry run's view."""
    today = _today(now)
    end = min(pd.Timestamp(end).normalize(), today - _ONE_DAY)
    plan = {}
    if "nyfed" in sources:
        for rate in NYFED_REPO_RATES:
            span = uncovered_span(f"nyfed:{rate}", max(pd.Timestamp(start), pd.Timestamp(NYFED_REPO_START)), end,
                                  coverage_file=coverage_file, force_refetch=force_refetch)
            if span:
                plan[f"nyfed:{rate}"] = span
    if "ofr" in sources:
        span = uncovered_span("ofr", max(pd.Timestamp(start), pd.Timestamp(OFR_REPO_START)), end,
                              coverage_file=coverage_file, force_refetch=force_refetch)
        if span:
            plan["ofr"] = span
    if "dtcc_gcf" in sources:
        g0, g1 = (pd.Timestamp(d) for d in DTCC_GCF_SPAN)
        span = uncovered_span("dtcc_gcf", max(pd.Timestamp(start), g0), min(end, g1), coverage_file=coverage_file)
        if span:  # the workbook is frozen: force_refetch never re-asks it
            plan["dtcc_gcf"] = span
    return plan


def store_repo(rows: pd.DataFrame, *, root: Path = REPO_DIR) -> int:
    """FILES ONLY: upsert rows (keys ``REPO_KEYS``)."""
    if rows.empty:
        return 0
    parquet_store.write_partitioned(rp.encode(rows), root, rp.REPO_KEYS)
    return len(rows)


def _cover(coverage_file: Path, key: str, first, last) -> None:
    if pd.Timestamp(last) >= pd.Timestamp(first):
        coverage_store.record_covered(coverage_file, key, [(pd.Timestamp(first), pd.Timestamp(last) + _ONE_DAY)])


def fetch_and_store_repo(plan: dict[str, tuple], *, now=None, root: Path = REPO_DIR,
                         coverage_file: Path = REPO_COVERAGE_FILE) -> dict:
    """Run a plan; per-key failures collected, not raised. Returns ``{"rows": {key: n},
    "errors": {key: msg}}``."""
    today = _today(now)
    settled = today - pd.Timedelta(days=REPO_SETTLE_DAYS)
    out = {"rows": {}, "errors": {}}
    for key, (first, last) in plan.items():
        try:
            if key.startswith("nyfed:"):
                rows = rp.nyfed_rates(NYFED_FETCH(key.split(":", 1)[1], first, last))
                out["rows"][key] = store_repo(rows, root=root)
                _cover(coverage_file, key, first, min(pd.Timestamp(last), settled))
            elif key == "ofr":
                rows = rp.ofr_rates(OFR_FETCH(rp.ofr_mnemonics(OFR_REPO_BUCKETS), first), OFR_REPO_BUCKETS)
                rows = rows[rows["timestamp"] <= pd.Timestamp(last)]
                out["rows"][key] = store_repo(rows, root=root)
                ref = rows[(rows["series"] == rp.ofr_series(*OFR_FINAL_REFERENCE)) & (rows["status"] == rp.FINAL)]
                if not ref.empty:
                    _cover(coverage_file, key, first, ref["timestamp"].max())
            elif key == "dtcc_gcf":
                rows = rp.gcf_index(GCF_FETCH())
                out["rows"][key] = store_repo(rows, root=root)
                _cover(coverage_file, key, *DTCC_GCF_SPAN)
        except Exception as exc:
            out["errors"][key] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            log.warning("repo %s failed: %s", key, out["errors"][key])
    return out


def read_repo(start, end, *, series=None, status: str = "best", root: Path = REPO_DIR) -> pd.DataFrame:
    """Decoded rows with ``timestamp`` in ``[start, end)``, optionally only ``series``.
    ``status``: ``"best"`` (final where it exists, else preliminary), ``"final"``,
    ``"preliminary"`` or ``"all"`` (both rows). No network."""
    raw = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end),
                                         equals_in={"series": list(series)} if series is not None else None)
    if raw is None or raw.empty:
        return rp._empty()
    df = rp.decode(raw[rp.REPO_COLUMNS]).sort_values(rp.REPO_KEYS).reset_index(drop=True)
    if status == "best":
        return rp.best(df)
    return df if status == "all" else df[df["status"] == status].reset_index(drop=True)


def latest_repo_rate(series: str, as_of=None, *, publication_lag_days: int = 1, status: str = "best",
                     root: Path = REPO_DIR) -> tuple[pd.Timestamp, float] | None:
    """``(day, rate)`` of the latest value of ``series`` PUBLISHED by the end of
    ``as_of`` (default today): a rate for day D is published the next business day, so
    only days up to ``as_of - publication_lag_days`` business days count (point in time,
    CLAUDE.md 3). Caveat: ``status="best"`` may return a final OFR value that was only
    published ~3 months later; pass ``"preliminary"`` for strictly point-in-time OFR."""
    as_of = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of).normalize()
    cutoff = as_of - pd.offsets.BDay(publication_lag_days) if publication_lag_days else as_of
    df = read_repo(cutoff - pd.Timedelta(days=31), cutoff + _ONE_DAY, series=[series], status=status, root=root)
    if df.empty:
        return None
    row = df.iloc[-1]
    return pd.Timestamp(row["timestamp"]), float(row["rate"])


def load_repo(start, end, *, series=None, status: str = "best", fetch_missing: bool = True,
              root: Path = REPO_DIR, coverage_file: Path = REPO_COVERAGE_FILE) -> pd.DataFrame:
    """Parent: fetch whatever ``[start, end)`` lacks, then read."""
    if fetch_missing:
        plan = plan_repo_update(start, pd.Timestamp(end) - _ONE_DAY, coverage_file=coverage_file)
        if plan:
            fetch_and_store_repo(plan, root=root, coverage_file=coverage_file)
    return read_repo(start, end, series=series, status=status, root=root)
