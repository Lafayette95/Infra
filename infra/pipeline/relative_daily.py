"""Relative DAILY settlement price / open interest (``SR3.c.0``, ``SR3.v.0``), built
locally from ABSOLUTE contracts' daily statistics.

Mirrors infra.pipeline.relative's flow exactly, reusing its contract/mapping machinery
unchanged (``group_by_root``, ``needed_contract_windows``, ``calendar_mapping``,
``volume_mapping``, ``apply_mapping``) - only the data source differs: this reads/fetches
via infra.pipeline.daily (Database/Daily) instead of infra.pipeline.futures
(Database/ohlcv-1m). See CLAUDE.md section 8.

``apply_mapping`` needs no changes to work on daily rows: a daily row's ``timestamp`` IS
already the trading day, and ``infra.trading_calendar.trading_day`` is a verified no-op
on an already-resolved trading day (checked across a full year, both DST transitions,
every configured dataset - 2026-09-21) - so re-applying it inside ``apply_mapping``
still assigns each row to its own, correct trading day.

``.v.N`` ranking needs ``daily_volume()``, computed from 1-minute OHLCV bars (the daily
statistics data carries open interest, not volume) - so a volume-ranked daily series has
an implicit dependency on the 1-minute pipeline. When ``fetch_missing=True`` this
function fetches 1-minute bars too if a ``.v.N`` spec needs them (mirroring
``load_relative_futures``'s existing behaviour exactly, same cost guardrail) - but only
as a ranking input; the 1-minute bars themselves are never part of the return value.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.config import (
    DAILY_FUTURES_COVERAGE_FILE,
    DAILY_FUTURES_DIR,
    FUTURES_CONTRACTS_FILE,
    FUTURES_COVERAGE_FILE,
    FUTURES_DEFS_COVERAGE_FILE,
    FUTURES_DIR,
    FUTURES_ROOTS,
    MAX_COST_USD,
    VOLUME_LOOKBACK_DAYS,
)
from infra.coverage.intervals import Interval, to_utc_day
from infra.pipeline import contracts as contracts_pipe
from infra.pipeline import daily as dl
from infra.pipeline import futures as fut
from infra.pipeline.relative import group_by_root, needed_contract_windows
from infra.processing.statistics import decode_daily, empty_daily
from infra.relative.rolls import day_index, volume_mapping
from infra.relative.series import apply_mapping, daily_volume
from infra.relative.symbology import RelativeSpec


def plan_relative_daily_update(
    specs: list[RelativeSpec],
    start,
    end,
    *,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    defs_coverage_file: Path = FUTURES_DEFS_COVERAGE_FILE,
    daily_coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    futures_coverage_file: Path = FUTURES_COVERAGE_FILE,
) -> dict[str, list[Interval]]:
    """Everything still missing on disk for ``specs`` (no API).

    Keys: absolute tickers needing daily settlement data, ``1m:<ticker>`` for 1-minute
    bars a ``.v.N`` spec still needs for ranking, and ``definitions:<root>``.
    """
    from infra.storage import contract_store

    plan: dict[str, list[Interval]] = {}
    for root, root_specs in group_by_root(specs).items():
        cfg = FUTURES_ROOTS[root]
        needs_volume = any(s.kind == "v" for s in root_specs)
        snaps = contracts_pipe.missing_snapshots(cfg, start, end, coverage_file=defs_coverage_file)
        if snaps:
            plan[f"definitions:{root}"] = snaps
        contracts = contract_store.read_contracts(contracts_file, root)
        if contracts.empty:
            continue
        windows, _ = needed_contract_windows(root_specs, cfg, contracts, start, end)
        for ticker, (w0, w1) in windows.items():
            gaps = dl.plan_daily_update(ticker, w0, w1, coverage_file=daily_coverage_file)
            if gaps:
                plan[ticker] = gaps
            if needs_volume:
                gaps_1m = fut.plan_futures_update(ticker, w0, w1, coverage_file=futures_coverage_file)
                if gaps_1m:
                    plan[f"1m:{ticker}"] = gaps_1m
    return plan


def load_relative_daily(
    specs: list[RelativeSpec],
    start,
    end,
    *,
    fetch_missing: bool = True,
    daily_root: Path = DAILY_FUTURES_DIR,
    daily_coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    futures_root_dir: Path = FUTURES_DIR,
    futures_coverage_file: Path = FUTURES_COVERAGE_FILE,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    defs_coverage_file: Path = FUTURES_DEFS_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """Parent: relative DAILY settlement/OI series for ``specs`` over ``[start, end)``."""
    start, end = to_utc_day(start), to_utc_day(end)
    frames: list[pd.DataFrame] = []
    for root, root_specs in group_by_root(specs).items():
        cfg = FUTURES_ROOTS[root]
        contracts = contracts_pipe.ensure_contracts(
            cfg, start, end, fetch_missing=fetch_missing, contracts_file=contracts_file,
            coverage_file=defs_coverage_file, max_cost_usd=max_cost_usd, client=client,
        )
        if contracts.empty:
            continue
        windows, cal_map = needed_contract_windows(root_specs, cfg, contracts, start, end)
        needs_volume = any(s.kind == "v" for s in root_specs)

        if fetch_missing:
            for ticker, (w0, w1) in windows.items():
                gaps = dl.plan_daily_update(ticker, w0, w1, coverage_file=daily_coverage_file)
                if gaps:
                    dl.fetch_and_store_daily(
                        ticker, gaps, dataset=cfg.dataset, root=daily_root,
                        coverage_file=daily_coverage_file, max_cost_usd=max_cost_usd, client=client,
                    )
                if needs_volume:  # ranking signal only - mirrors load_relative_futures exactly
                    gaps_1m = fut.plan_futures_update(ticker, w0, w1, coverage_file=futures_coverage_file)
                    if gaps_1m:
                        fut.fetch_and_store_futures(
                            ticker, gaps_1m, dataset=cfg.dataset, root=futures_root_dir,
                            coverage_file=futures_coverage_file, max_cost_usd=max_cost_usd, client=client,
                        )

        daily_rows = dl.read_daily_from_disk(list(windows), start, end, root=daily_root)
        if daily_rows.empty:
            continue
        vol = None
        if needs_volume:
            min_bars = fut.read_futures_from_disk(list(windows), start, end, root=futures_root_dir)
            if not min_bars.empty:
                vol = daily_volume(min_bars, dataset=cfg.dataset)

        for spec in root_specs:
            if spec.kind == "c":
                mapping = cal_map
            elif vol is None:
                continue  # can't rank by volume without any 1-minute bars on disk
            else:
                mapping = volume_mapping(
                    vol, contracts, day_index(start, end),
                    max_rank=spec.rank, expiry_months=cfg.expiry_months,
                    roll_offset_days=cfg.roll_offset_days, lookback_days=VOLUME_LOOKBACK_DAYS,
                )
            frames.append(apply_mapping(daily_rows, mapping, spec.rank, spec.label, dataset=cfg.dataset))
    if not frames:
        return decode_daily(empty_daily()).assign(contract=pd.Series(dtype="str"))
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
