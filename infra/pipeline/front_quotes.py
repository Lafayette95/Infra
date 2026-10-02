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


# ------------------------------------------------------------------- parallel stages
FETCH_WORKERS = 8  # Databento answers each request in ~1-2 min regardless of size (2026-10-02)


def _parallel(jobs: dict, fetch, store, workers: int) -> dict[str, str]:
    """Run ``fetch(key, job)`` for every job concurrently (NETWORK only) and
    ``store(key, job, result)`` for each on THIS thread as it lands (files are shared, so
    never written concurrently). Returns ``{key: error}`` - one failure never stops the rest."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    errors = {}
    if not jobs:
        return errors
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fq-fetch") as pool:
        futures = {pool.submit(fetch, k, j): k for k, j in jobs.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                store(key, jobs[key], fut.result())
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
                log.warning("%s failed: %s", key, errors[key])
    return errors


def backfill_front_quotes(
    roots: list[str],
    start,
    end,
    *,
    pad=ROLL_PAD,
    dry_run: bool = False,
    workers: int = FETCH_WORKERS,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    defs_coverage: Path = FUTURES_DEFS_COVERAGE_FILE,
    daily_root: Path = DAILY_FUTURES_DIR,
    daily_coverage: Path = DAILY_FUTURES_COVERAGE_FILE,
    bbo_root: Path = BBO_FUTURES_DIR,
    bbo_coverage: Path = BBO_FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
    on_root=None,
) -> dict:
    """Every root at once, in three phases - definitions, daily cleared volume, quotes -
    each fetched in parallel across ALL roots and stored sequentially. ``dry_run`` stops
    before buying quotes (the planning inputs are still fetched; they're what the roll
    dates come from). ``on_root(root, info)`` is called as each root's quotes complete.
    Returns ``{"roots": {root: {mapping, gaps, est_usd, rows}}, "errors": {...}}``."""
    from infra.pipeline import daily as dl
    from infra.pipeline.relative import volume_candidate_windows

    start, end = to_utc_day(start), to_utc_day(end)
    cfgs = {r: FUTURES_ROOTS[r] for r in roots}
    errors: dict[str, str] = {}

    # 1. definitions -> contracts table
    jobs = {f"defs {r} {day.date()}": (cfgs[r], day) for r in roots
            for day, _ in contracts_pipe.missing_snapshots(cfgs[r], start - pd.Timedelta(days=30), end,
                                                           coverage_file=defs_coverage)}
    errors |= _parallel(
        jobs, lambda k, j: api.fetch_definitions(j[0].dataset, [j[0].parent], j[1], max_cost_usd=max_cost_usd, client=client),
        lambda k, j, raw: contracts_pipe.store_definitions_snapshot(j[0], j[1], raw, contracts_file=contracts_file,
                                                                    coverage_file=defs_coverage), workers)

    def contracts_for(root):
        c = contracts_pipe.ensure_contracts(cfgs[root], start, end, fetch_missing=False, contracts_file=contracts_file)
        return c[(c["expiry"] >= start - _CONTRACT_HORIZON) & (c["expiry"] <= end + _CONTRACT_HORIZON)]

    # 2. daily cleared volume for each root's ranking pool
    spec = {r: [RelativeSpec(r, "v", 0)] for r in roots}
    jobs = {}
    for r in roots:
        for ticker, (w0, w1) in volume_candidate_windows(spec[r], cfgs[r], contracts_for(r), start, end).items():
            gaps = dl.plan_daily_update(ticker, w0, w1, coverage_file=daily_coverage)
            if gaps:
                jobs[f"stats {ticker}"] = (r, ticker, gaps)
    errors |= _parallel(
        jobs, lambda k, j: dl.fetch_daily_raw(j[1], j[2], dataset=cfgs[j[0]].dataset, max_cost_usd=max_cost_usd, client=client),
        lambda k, j, fetched: dl.store_daily_raw(j[1], fetched, dataset=cfgs[j[0]].dataset, root=daily_root,
                                                 coverage_file=daily_coverage), workers)

    # 3. mapping (local) -> quote gaps -> price -> (fetch)
    out = {}
    jobs = {}
    for r in roots:
        mapping = volume_ranked_mapping(spec[r], cfgs[r], contracts_for(r), start, end, fetch_missing=False,
                                        daily_root=daily_root, daily_coverage_file=daily_coverage)
        gaps = {}
        for ticker, intervals in quote_windows(mapping, pad=pad).items():
            g = [x for a, b in intervals for x in bbo.plan_bbo_update(ticker, a, b, coverage_file=bbo_coverage)]
            if g:
                gaps[ticker] = merge_intervals(g)
        out[r] = {"mapping": mapping, "gaps": gaps, "est_usd": price_front_quotes(r, gaps, client=client), "rows": 0}
        jobs |= {f"quotes {r} {t}": (r, t, g) for t, g in gaps.items()}
    if dry_run:
        return {"roots": out, "errors": errors}

    pending = {r: sum(1 for j in jobs.values() if j[0] == r) for r in roots}

    def store_quotes(key, job, fetched):
        r, ticker, _ = job
        out[r]["rows"] += bbo.store_bbo_raw(ticker, fetched, root=bbo_root, coverage_file=bbo_coverage)
        pending[r] -= 1
        if pending[r] == 0 and on_root:
            on_root(r, out[r])

    errors |= _parallel(
        jobs, lambda k, j: bbo.fetch_bbo_raw(j[1], j[2], dataset=cfgs[j[0]].dataset, max_cost_usd=max_cost_usd, client=client),
        store_quotes, workers)
    return {"roots": out, "errors": errors}
