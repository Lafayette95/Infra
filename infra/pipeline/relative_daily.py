"""Relative DAILY settlement price / open interest (``SR3.c.0``, ``SR3.v.0``), built
locally from ABSOLUTE contracts' daily statistics.

Mirrors infra.pipeline.relative's flow exactly, reusing its contract/mapping machinery
unchanged (``group_by_root``, ``needed_contract_windows``, ``volume_ranked_mapping``,
``series_windows``, ``apply_mapping``) - only the data source differs: this reads/fetches
via infra.pipeline.daily (Database/Daily) instead of infra.pipeline.futures
(Database/ohlcv-1m). See CLAUDE.md section 8.

``apply_mapping`` needs no changes to work on daily rows: a daily row's ``timestamp`` IS
already the trading day, and ``infra.trading_calendar.trading_day`` is a verified no-op
on an already-resolved trading day (checked across a full year, both DST transitions,
every configured dataset - 2026-09-21) - so re-applying it inside ``apply_mapping``
still assigns each row to its own, correct trading day.

``.v.N`` ranks by the daily statistics' own cleared volume (the same store), so a
volume-ranked daily series needs no 1-minute data at all.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.config import (
    DAILY_FUTURES_COVERAGE_FILE,
    DAILY_FUTURES_DIR,
    FUTURES_CONTRACTS_FILE,
    FUTURES_DEFS_COVERAGE_FILE,
    FUTURES_ROOTS,
    MAX_COST_USD,
)
from infra.coverage.intervals import Interval, to_utc_day
from infra.pipeline import contracts as contracts_pipe
from infra.pipeline import daily as dl
from infra.pipeline.relative import (
    VOLUME_HISTORY,
    group_by_root,
    needed_contract_windows,
    series_windows,
    volume_candidate_windows,
    volume_ranked_mapping,
)
from infra.processing.statistics import decode_daily, empty_daily
from infra.relative.series import apply_mapping
from infra.relative.symbology import RelativeSpec


def plan_relative_daily_update(
    specs: list[RelativeSpec],
    start,
    end,
    *,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    defs_coverage_file: Path = FUTURES_DEFS_COVERAGE_FILE,
    daily_root: Path = DAILY_FUTURES_DIR,
    daily_coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
) -> dict[str, list[Interval]]:
    """Everything still missing on disk for ``specs`` (no API). Keys: absolute tickers
    needing daily statistics (incl. a ``.v.N`` ranking's candidates), and
    ``definitions:<root>``."""
    from infra.storage import contract_store

    plan: dict[str, list[Interval]] = {}
    for root, root_specs in group_by_root(specs).items():
        cfg = FUTURES_ROOTS[root]
        snaps = contracts_pipe.missing_snapshots(cfg, start, end, coverage_file=defs_coverage_file)
        if snaps:
            plan[f"definitions:{root}"] = snaps
        contracts = contract_store.read_contracts(contracts_file, root)
        if contracts.empty:
            continue
        _, cal_map = needed_contract_windows(root_specs, cfg, contracts, start, end)
        vol_map = volume_ranked_mapping(root_specs, cfg, contracts, start, end, fetch_missing=False,
                                        daily_root=daily_root, daily_coverage_file=daily_coverage_file)
        windows = series_windows(root_specs, cal_map, vol_map)
        for ticker, (w0, w1) in volume_candidate_windows(root_specs, cfg, contracts, start, end).items():
            lo, hi = windows.get(ticker, (w0, w1))
            windows[ticker] = (min(lo, w0), max(hi, w1))
        for ticker, (w0, w1) in windows.items():
            gaps = dl.plan_daily_update(ticker, w0, w1, coverage_file=daily_coverage_file)
            if gaps:
                plan[ticker] = gaps
    return plan


def load_relative_daily(
    specs: list[RelativeSpec],
    start,
    end,
    *,
    fetch_missing: bool = True,
    daily_root: Path = DAILY_FUTURES_DIR,
    daily_coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    defs_coverage_file: Path = FUTURES_DEFS_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """Parent: relative DAILY settlement/OI/volume series for ``specs`` over ``[start, end)``."""
    start, end = to_utc_day(start), to_utc_day(end)
    frames: list[pd.DataFrame] = []
    for root, root_specs in group_by_root(specs).items():
        cfg = FUTURES_ROOTS[root]
        has_v = any(s.kind == "v" for s in root_specs)
        contracts = contracts_pipe.ensure_contracts(
            cfg, start - VOLUME_HISTORY if has_v else start, end, fetch_missing=fetch_missing,
            contracts_file=contracts_file, coverage_file=defs_coverage_file, max_cost_usd=max_cost_usd, client=client,
        )
        if contracts.empty:
            continue
        _, cal_map = needed_contract_windows(root_specs, cfg, contracts, start, end)
        vol_map = volume_ranked_mapping(root_specs, cfg, contracts, start, end, fetch_missing=fetch_missing,
                                        daily_root=daily_root, daily_coverage_file=daily_coverage_file,
                                        max_cost_usd=max_cost_usd, client=client)
        windows = series_windows(root_specs, cal_map, vol_map)
        if fetch_missing:
            for ticker, (w0, w1) in windows.items():
                gaps = dl.plan_daily_update(ticker, w0, w1, coverage_file=daily_coverage_file)
                if gaps:
                    dl.fetch_and_store_daily(
                        ticker, gaps, dataset=cfg.dataset, root=daily_root,
                        coverage_file=daily_coverage_file, max_cost_usd=max_cost_usd, client=client,
                    )
        daily_rows = dl.read_daily_from_disk(list(windows), start, end, root=daily_root)
        if daily_rows.empty:
            continue
        for spec in root_specs:
            mapping = cal_map if spec.kind == "c" else vol_map
            frames.append(apply_mapping(daily_rows, mapping, spec.rank, spec.label, dataset=cfg.dataset))
    if not frames:
        return decode_daily(empty_daily()).assign(contract=pd.Series(dtype="str"))
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
