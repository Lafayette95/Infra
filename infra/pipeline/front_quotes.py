"""Quote (``bbo-1m``) history for a root's FRONT contract, by the volume roll rule
(``<root>.v.0``, infra.pipeline.relative - for bond futures: the front two ranked on
the exchange's daily cleared volume, 2-day lookback), plus the OTHER front-two contract
only around each roll, so both legs exist where a backtest needs them and nowhere else.

Plan (cheap, fetched as needed - Rule 2.1): definition snapshots -> contracts table;
daily statistics (cleared volume) for the front two -> the ``.v.0`` mapping.
Execute (the real cost): ``bbo-1m`` for each planned window not yet covered.

Decade-safe: CME reuses raw symbols every ten years (``ZNH5`` = March 2015 AND March
2025), so the contracts used for a window are limited to expiries within a year of it
(a twin is never a candidate) and fetch windows are each contract's UNBROKEN runs
(``infra.relative.rolls.contract_runs``), never a per-ticker min..max.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import databento_client as api
from infra.config import (
    BBO_FUTURES_COVERAGE_FILE,
    BBO_FUTURES_DIR,
    DAILY_FUTURES_COVERAGE_FILE,
    DAILY_FUTURES_DIR,
    FUTURES_CONTRACTS_FILE,
    FUTURES_DEFS_COVERAGE_FILE,
    FUTURES_ROOTS,
    MAX_COST_USD,
    SCHEMA_BBO_1M,
)
from infra.coverage.intervals import Interval, merge_intervals, to_utc_day
from infra.pipeline import bbo
from infra.pipeline import contracts as contracts_pipe
from infra.pipeline.relative import volume_ranked_mapping
from infra.relative.rolls import around_switch_windows, contract_runs
from infra.relative.symbology import RelativeSpec

log = logging.getLogger(__name__)

ROLL_PAD = pd.offsets.BDay(5)  # the other front-two contract: 5 business days either side of a roll
_CONTRACT_HORIZON = pd.Timedelta(days=366)  # candidates: expiries within a year of the window


def front_mapping(
    root: str,
    start,
    end,
    *,
    fetch_missing: bool = True,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    defs_coverage: Path = FUTURES_DEFS_COVERAGE_FILE,
    daily_root: Path = DAILY_FUTURES_DIR,
    daily_coverage: Path = DAILY_FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """The ``<root>.v.0`` mapping (rank 0) over ``[start, end)``, fetching the cheap
    planning inputs (definitions, daily cleared volume) if missing."""
    cfg = FUTURES_ROOTS[root]
    start, end = to_utc_day(start), to_utc_day(end)
    contracts = contracts_pipe.ensure_contracts(
        cfg, start - pd.Timedelta(days=30), end, fetch_missing=fetch_missing, contracts_file=contracts_file,
        coverage_file=defs_coverage, max_cost_usd=max_cost_usd, client=client)
    contracts = contracts[(contracts["expiry"] >= start - _CONTRACT_HORIZON)
                          & (contracts["expiry"] <= end + _CONTRACT_HORIZON)]
    return volume_ranked_mapping(
        [RelativeSpec(root, "v", 0)], cfg, contracts, start, end, fetch_missing=fetch_missing,
        daily_root=daily_root, daily_coverage_file=daily_coverage, max_cost_usd=max_cost_usd, client=client)


def quote_windows(mapping: pd.DataFrame, *, pad=ROLL_PAD) -> dict[str, list[Interval]]:
    """``{ticker: [intervals]}`` to hold quotes for: each ``v.0`` run, plus the other
    front-two contract ``pad`` either side of every switch. Clipped to the mapping."""
    lo, hi = mapping.index.min(), mapping.index.max() + pd.Timedelta(days=1)
    windows: dict[str, list[Interval]] = {}
    for ticker, a, b in contract_runs(mapping, 0) + around_switch_windows(mapping, 0, pad=pad):
        a, b = max(pd.Timestamp(a), lo), min(pd.Timestamp(b), hi)
        if a < b:
            windows.setdefault(ticker, []).append((a, b))
    return {t: merge_intervals(w) for t, w in windows.items()}


def plan_front_quotes(
    root: str, start, end, *, pad=ROLL_PAD, bbo_coverage: Path = BBO_FUTURES_COVERAGE_FILE, **kw,
) -> tuple[pd.DataFrame, dict[str, list[Interval]]]:
    """``(mapping, {ticker: gaps})`` - the quote gaps not yet on disk. Only the cheap
    planning inputs are fetched (``kw`` -> ``front_mapping``)."""
    mapping = front_mapping(root, start, end, **kw)
    gaps = {}
    for ticker, intervals in quote_windows(mapping, pad=pad).items():
        g = [x for a, b in intervals for x in bbo.plan_bbo_update(ticker, a, b, coverage_file=bbo_coverage)]
        if g:
            gaps[ticker] = merge_intervals(g)
    return mapping, gaps


def price_front_quotes(root: str, gaps: dict[str, list[Interval]], *, client=None) -> float:
    """Estimated USD for ``gaps`` (Databento's free cost endpoint)."""
    dataset = FUTURES_ROOTS[root].dataset
    return sum(api.estimate_cost(dataset, SCHEMA_BBO_1M, [t], a, b, "raw_symbol", client)
               for t, intervals in gaps.items() for a, b in intervals)


def fetch_front_quotes(
    root: str, gaps: dict[str, list[Interval]], *, root_dir: Path = BBO_FUTURES_DIR,
    bbo_coverage: Path = BBO_FUTURES_COVERAGE_FILE, max_cost_usd: float = MAX_COST_USD, client=None,
) -> tuple[int, dict[str, str]]:
    """Fetch and store every gap. Returns ``(rows, {ticker: error})`` - one failure never
    stops the others."""
    dataset, rows, errors = FUTURES_ROOTS[root].dataset, 0, {}
    for ticker, intervals in sorted(gaps.items()):
        try:
            rows += bbo.fetch_and_store_bbo(ticker, intervals, dataset=dataset, root=root_dir,
                                            coverage_file=bbo_coverage, max_cost_usd=max_cost_usd, client=client)
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            log.warning("bbo-1m fetch failed for %s: %s", ticker, errors[ticker])
    return rows, errors
