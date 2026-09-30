"""Daily cash-bond par yields (constant-maturity sovereign curves, ``US_BOND_10y`` ...).

Same 4-function shape as pipeline/daily.py (read / plan / fetch / store + a ``load``
parent), reusing the generic primitives unchanged (parquet_store, coverage_store,
adjustment_store) - pointed at ``Database/Daily/Bonds`` with its own coverage manifest.
Two differences from the futures pipeline, both from the sources themselves:

* the sources are free official publications, one per curve (``SOURCES``, config
  ``BondCurve.source``) - no Databento, no cost guardrail needed;
* one request always returns a WHOLE curve, so coverage is keyed by curve (``"US"``),
  not by ticker - there is no cheaper per-tenor request to plan for.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.api import boe_client, bundesbank_client, treasury_client
from infra.config import ADJUSTMENTS_DIR, BOND_CURVES, DAILY_BONDS_COVERAGE_FILE, DAILY_BONDS_DIR, BondCurve, bond_ticker
from infra.coverage.intervals import Interval, find_missing_ranges, to_utc_day
from infra.processing import bond_curves as bc
from infra.storage import adjustment_store, coverage_store, parquet_store

log = logging.getLogger(__name__)

STORE = "Daily/Bonds"  # this store's name in the adjustments log

def bundesbank_svensson_curve(start: pd.Timestamp, end: pd.Timestamp):
    """The Bundesbank's daily Svensson parameters, evaluated into the model's zero curve
    (``infra.processing.bond_curves.svensson_curve``)."""
    params, covered = bundesbank_client.fetch_svensson_params(start, end)
    return bc.svensson_curve(params), covered


# source name -> fn(start, end) -> (long frame timestamp/maturity/value, covered intervals)
SOURCES: dict[str, Callable[[pd.Timestamp, pd.Timestamp], tuple[pd.DataFrame, list[Interval]]]] = {
    "treasury": treasury_client.fetch_par_curve,
    "boe": boe_client.fetch_spot_curve,
    "bundesbank": bundesbank_svensson_curve,
}


def curve_tickers(spec: BondCurve) -> list[str]:
    return [bond_ticker(spec.country, t) for t in spec.tenors]


def read_bonds_from_disk(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    root: Path = DAILY_BONDS_DIR,
    adjusted: bool = True,
    adjustments_dir: Path | None = None,
) -> pd.DataFrame:
    """Decoded (float, percent) par yields in ``[start, end)``. No network. ``adjusted``
    overlays the adjustments log exactly like ``infra.pipeline.daily.read_daily_from_disk``."""
    raw = parquet_store.read_partitioned(root, start=start, end=end, equals_in={"ticker": tickers})
    if raw is None or raw.empty:
        return bc.decode_bonds(bc.empty_bonds())
    df = bc.decode_bonds(raw[bc.BOND_COLUMNS]).sort_values(bc.BOND_KEYS)
    if adjusted:
        adjustments_dir = ADJUSTMENTS_DIR if adjustments_dir is None else adjustments_dir
        adj = adjustment_store.read(adjustments_dir, store=STORE, start=start, end=end, keys=tickers)
        df = adjustment_store.apply(df, adj, key_column="ticker")
    return df.reset_index(drop=True)


def plan_bonds_update(
    spec: BondCurve,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    coverage_file: Path = DAILY_BONDS_COVERAGE_FILE,
    force_refetch: bool = False,
) -> list[Interval]:
    """Ranges of ``[start, end)`` (never before the curve's history start) not yet
    queried. ``force_refetch`` plans the whole range (the scheduled run's revision
    window), as in ``infra.pipeline.daily.plan_daily_update``."""
    start = max(to_utc_day(start), pd.Timestamp(spec.history_start))
    requested = (start, to_utc_day(end))
    if requested[0] >= requested[1]:
        return []
    if force_refetch:
        return [requested]
    return find_missing_ranges(requested, coverage_store.read_covered(coverage_file, spec.country))


# (range_start, range_end, source curve, covered intervals) per queried range
Fetched = list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, list[Interval]]]


def fetch_bonds_raw(spec: BondCurve, ranges: list[Interval], *, sources=None) -> Fetched:
    """NETWORK ONLY: the source's curve for each range. Touches no files."""
    fetch = (SOURCES if sources is None else sources)[spec.source]
    return [(s, e, *fetch(s, e)) for s, e in ranges]


def store_bonds_raw(
    spec: BondCurve,
    fetched: Fetched,
    *,
    root: Path = DAILY_BONDS_DIR,
    coverage_file: Path = DAILY_BONDS_COVERAGE_FILE,
) -> int:
    """FILES ONLY: derive par rows, save, and record ONLY what the source said it covers
    (a not-yet-published day is never claimed covered - retried next run)."""
    rows = 0
    for range_start, range_end, curve, covered in fetched:
        clean = bc.to_par_rows(curve, spec)
        parquet_store.write_partitioned(bc.encode_bonds(clean), root, bc.BOND_KEYS)
        if covered:
            coverage_store.record_covered(coverage_file, spec.country, covered)
        rows += len(clean)
        log.info("%s bonds %s->%s: %d rows saved", spec.country, range_start.date(), range_end.date(), len(clean))
    return rows


def load_bonds(
    countries: list[str],
    start,
    end,
    *,
    fetch_missing: bool = True,
    root: Path = DAILY_BONDS_DIR,
    coverage_file: Path = DAILY_BONDS_COVERAGE_FILE,
    force_refetch: bool = False,
    curves: dict[str, BondCurve] = BOND_CURVES,
    sources=None,
) -> pd.DataFrame:
    """Parent: ensure ``[start, end)`` is on disk for each curve (network only for gaps),
    then read every tenor of those curves."""
    start, end = to_utc_day(start), to_utc_day(end)
    tickers = []
    for country in countries:
        spec = curves[country]
        tickers += curve_tickers(spec)
        if fetch_missing:
            gaps = plan_bonds_update(spec, start, end, coverage_file=coverage_file, force_refetch=force_refetch)
            if gaps:
                store_bonds_raw(spec, fetch_bonds_raw(spec, gaps, sources=sources), root=root, coverage_file=coverage_file)
    return read_bonds_from_disk(tickers, start, end, root=root)
