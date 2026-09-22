"""Relative futures series (``SR3.c.0``, ``SR3.v.0``) built locally from ABSOLUTE contracts.

Flow per root: contracts table -> roll mapping -> the absolute contracts actually needed
(and their date windows) -> fetch only missing gaps of those -> read -> apply mapping.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.config import (
    FUTURES_CONTRACTS_FILE,
    FUTURES_COVERAGE_FILE,
    FUTURES_DEFS_COVERAGE_FILE,
    FUTURES_DIR,
    FUTURES_ROOTS,
    MAX_COST_USD,
    VOLUME_EXTRA_CANDIDATES,
    VOLUME_LOOKBACK_DAYS,
    FuturesRoot,
)
from infra.coverage.intervals import Interval, to_utc_day
from infra.pipeline import contracts as contracts_pipe
from infra.pipeline import futures as fut
from infra.relative.rolls import calendar_mapping, day_index, volume_mapping
from infra.relative.series import apply_mapping, daily_volume
from infra.relative.symbology import RelativeSpec

_ONE_DAY = pd.Timedelta(days=1)


def group_by_root(specs: list[RelativeSpec]) -> dict[str, list[RelativeSpec]]:
    """Specs grouped by root; unknown roots fail early with a clear message."""
    grouped: dict[str, list[RelativeSpec]] = {}
    for spec in specs:
        if spec.root not in FUTURES_ROOTS:
            raise KeyError(f"Unknown root {spec.root!r}; add it to FUTURES_ROOTS in infra/config.py")
        grouped.setdefault(spec.root, []).append(spec)
    return grouped


def needed_contract_windows(
    specs: list[RelativeSpec],
    cfg: FuturesRoot,
    contracts: pd.DataFrame,
    start,
    end,
) -> tuple[dict[str, Interval], pd.DataFrame]:
    """Absolute contracts needed by ``specs`` and the ``[first_use, last_use+1d)`` window of each.

    Also returns the calendar mapping (ranks 0..max) that the windows were derived from.
    """
    dates = day_index(to_utc_day(start), to_utc_day(end))
    ranks = {s.rank for s in specs if s.kind == "c"}
    v_ranks = [s.rank for s in specs if s.kind == "v"]
    if v_ranks:  # volume ranking needs a pool of calendar candidates
        ranks |= set(range(max(v_ranks) + VOLUME_EXTRA_CANDIDATES + 1))
    cal_map = calendar_mapping(
        contracts, dates, max_rank=max(ranks),
        expiry_months=cfg.expiry_months, roll_offset_days=cfg.roll_offset_days,
    )
    windows: dict[str, Interval] = {}
    for rank in sorted(ranks):
        for ticker, days in cal_map[rank].dropna().groupby(cal_map[rank].dropna()).groups.items():
            first, last = min(days), max(days) + _ONE_DAY
            lo, hi = windows.get(ticker, (first, last))
            windows[ticker] = (min(lo, first), max(hi, last))
    return windows, cal_map


def plan_relative_update(
    specs: list[RelativeSpec],
    start,
    end,
    *,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    defs_coverage_file: Path = FUTURES_DEFS_COVERAGE_FILE,
    coverage_file: Path = FUTURES_COVERAGE_FILE,
) -> dict[str, list[Interval]]:
    """Everything still missing on disk for ``specs`` (no API). Keys: absolute tickers, plus
    ``definitions:<root>`` for definition snapshots not yet pulled."""
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
        windows, _ = needed_contract_windows(root_specs, cfg, contracts, start, end)
        for ticker, (w0, w1) in windows.items():
            gaps = fut.plan_futures_update(ticker, w0, w1, coverage_file=coverage_file)
            if gaps:
                plan[ticker] = gaps
    return plan


def load_relative_futures(
    specs: list[RelativeSpec],
    start,
    end,
    *,
    fetch_missing: bool = True,
    root_dir: Path = FUTURES_DIR,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    defs_coverage_file: Path = FUTURES_DEFS_COVERAGE_FILE,
    coverage_file: Path = FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """Parent: relative series for ``specs`` over ``[start, end)`` (unadjusted across rolls)."""
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
        if fetch_missing:
            for ticker, (w0, w1) in windows.items():
                gaps = fut.plan_futures_update(ticker, w0, w1, coverage_file=coverage_file)
                if gaps:
                    fut.fetch_and_store_futures(
                        ticker, gaps, dataset=cfg.dataset, root=root_dir,
                        coverage_file=coverage_file, max_cost_usd=max_cost_usd, client=client,
                    )
        bars = fut.read_futures_from_disk(list(windows), start, end, root=root_dir)
        if bars.empty:
            continue
        vol = daily_volume(bars) if any(s.kind == "v" for s in root_specs) else None
        for spec in root_specs:
            if spec.kind == "c":
                mapping = cal_map
            else:
                mapping = volume_mapping(
                    vol, contracts, day_index(start, end),
                    max_rank=spec.rank, expiry_months=cfg.expiry_months,
                    roll_offset_days=cfg.roll_offset_days, lookback_days=VOLUME_LOOKBACK_DAYS,
                )
            frames.append(apply_mapping(bars, mapping, spec.rank, spec.label))
    if not frames:
        empty = fut.read_futures_from_disk([], start, start)
        return empty.assign(contract=pd.Series(dtype="str"))
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
