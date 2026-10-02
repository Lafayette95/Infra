"""Relative futures series (``SR3.c.0``, ``SR3.v.0``) built locally from ABSOLUTE contracts.

Flow per root: contracts table -> roll mapping -> the absolute contracts actually needed
(and their date windows) -> fetch only missing gaps of those -> read -> apply mapping.

``.v.N`` ranks by each contract's official daily CLEARED VOLUME, from the daily statistics
store (infra.pipeline.daily, CLAUDE.md 8) - not by summing 1-minute bars: it's the
exchange's own figure (blocks, EFPs and spread legs included) and costs a fraction of
the bars. So the candidates' volume comes from daily statistics, and 1-minute bars are
fetched only for the contract each day's ranking actually picks.
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
    VOLUME_EXTRA_CANDIDATES,
    VOLUME_LOOKBACK_DAYS,
    FuturesRoot,
)
from infra.coverage.intervals import Interval, to_utc_day
from infra.pipeline import contracts as contracts_pipe
from infra.pipeline import daily as dl
from infra.pipeline import futures as fut
from infra.relative.rolls import calendar_mapping, day_index, volume_mapping
from infra.relative.series import apply_mapping, volume_pivot
from infra.relative.symbology import RelativeSpec

_ONE_DAY = pd.Timedelta(days=1)
# Volume history read before ``start``, so the first days rank on real prior volume
# (VOLUME_LOOKBACK_DAYS trading days, plus weekends and holidays) instead of falling back
# to calendar order.
VOLUME_HISTORY = pd.Timedelta(days=2 * VOLUME_LOOKBACK_DAYS + 7)


def group_by_root(specs: list[RelativeSpec]) -> dict[str, list[RelativeSpec]]:
    """Specs grouped by root; unknown roots fail early with a clear message."""
    grouped: dict[str, list[RelativeSpec]] = {}
    for spec in specs:
        if spec.root not in FUTURES_ROOTS:
            raise KeyError(f"Unknown root {spec.root!r}; add it to FUTURES_ROOTS in infra/config.py")
        grouped.setdefault(spec.root, []).append(spec)
    return grouped


def volume_extra_candidates(cfg: FuturesRoot) -> int:
    return VOLUME_EXTRA_CANDIDATES if cfg.volume_extra_candidates is None else cfg.volume_extra_candidates


def volume_lookback_days(cfg: FuturesRoot) -> int:
    return VOLUME_LOOKBACK_DAYS if cfg.volume_lookback_days is None else cfg.volume_lookback_days


def mapping_windows(mapping: pd.DataFrame, ranks) -> dict[str, Interval]:
    """``{ticker: [first_day, last_day+1d)}`` of every contract ``mapping`` assigns to ``ranks``."""
    windows: dict[str, Interval] = {}
    for rank in sorted(ranks):
        col = mapping[rank].dropna()
        for ticker, days in col.groupby(col).groups.items():
            first, last = min(days), max(days) + _ONE_DAY
            lo, hi = windows.get(ticker, (first, last))
            windows[ticker] = (min(lo, first), max(hi, last))
    return windows


def needed_contract_windows(
    specs: list[RelativeSpec],
    cfg: FuturesRoot,
    contracts: pd.DataFrame,
    start,
    end,
) -> tuple[dict[str, Interval], pd.DataFrame]:
    """Absolute contracts the CALENDAR ranks of ``specs`` cover, and the ``[first_use,
    last_use+1d)`` window of each - for ``.c.N`` specs their own rank, for ``.v.N`` specs
    the candidate pool (calendar ranks ``0..N + volume_extra_candidates``).

    Also returns the calendar mapping (ranks 0..max) that the windows were derived from.
    """
    dates = day_index(to_utc_day(start), to_utc_day(end))
    ranks = {s.rank for s in specs if s.kind == "c"}
    v_ranks = [s.rank for s in specs if s.kind == "v"]
    if v_ranks:  # volume ranking needs a pool of calendar candidates
        ranks |= set(range(max(v_ranks) + volume_extra_candidates(cfg) + 1))
    cal_map = calendar_mapping(
        contracts, dates, max_rank=max(ranks),
        expiry_months=cfg.expiry_months, roll_offset_days=cfg.roll_offset_days,
    )
    return mapping_windows(cal_map, ranks), cal_map


def _volume_specs(specs: list[RelativeSpec]) -> list[RelativeSpec]:
    return [s for s in specs if s.kind == "v"]


def volume_candidate_windows(specs, cfg, contracts, start, end) -> dict[str, Interval]:
    """Daily-statistics windows the ``.v.N`` ranking reads: the candidate pool, from
    VOLUME_HISTORY before ``start``. Empty if no spec is volume-ranked."""
    v_specs = _volume_specs(specs)
    if not v_specs:
        return {}
    windows, _ = needed_contract_windows(v_specs, cfg, contracts, to_utc_day(start) - VOLUME_HISTORY, end)
    return windows


def volume_ranked_mapping(
    specs: list[RelativeSpec],
    cfg: FuturesRoot,
    contracts: pd.DataFrame,
    start,
    end,
    *,
    fetch_missing: bool = True,
    daily_root: Path = DAILY_FUTURES_DIR,
    daily_coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame | None:
    """The ``.v`` mapping (ranks 0..max v rank) over ``[start, end)``, ranked by daily
    cleared volume. Fetches the candidates' missing daily statistics first if
    ``fetch_missing`` (cheap; Rule 2.1 as usual). None if no spec is volume-ranked."""
    v_specs = _volume_specs(specs)
    if not v_specs:
        return None
    start, end = to_utc_day(start), to_utc_day(end)
    pool = volume_candidate_windows(v_specs, cfg, contracts, start, end)
    if fetch_missing:
        for ticker, (w0, w1) in pool.items():
            gaps = dl.plan_daily_update(ticker, w0, w1, coverage_file=daily_coverage_file)
            if gaps:
                dl.fetch_and_store_daily(ticker, gaps, dataset=cfg.dataset, root=daily_root,
                                         coverage_file=daily_coverage_file, max_cost_usd=max_cost_usd, client=client)
    daily_rows = dl.read_daily_from_disk(list(pool), start - VOLUME_HISTORY, end, root=daily_root)
    return volume_mapping(
        volume_pivot(daily_rows), contracts, day_index(start, end),
        max_rank=max(s.rank for s in v_specs), expiry_months=cfg.expiry_months,
        roll_offset_days=cfg.roll_offset_days, lookback_days=volume_lookback_days(cfg),
    )


def series_windows(specs, cal_map: pd.DataFrame, vol_map: pd.DataFrame | None) -> dict[str, Interval]:
    """Windows of the contracts the returned series actually use: each ``.c.N`` spec's
    calendar rank, each ``.v.N`` spec's volume rank."""
    windows = mapping_windows(cal_map, {s.rank for s in specs if s.kind == "c"})
    if vol_map is not None:
        for ticker, (w0, w1) in mapping_windows(vol_map, {s.rank for s in specs if s.kind == "v"}).items():
            lo, hi = windows.get(ticker, (w0, w1))
            windows[ticker] = (min(lo, w0), max(hi, w1))
    return windows


def plan_relative_update(
    specs: list[RelativeSpec],
    start,
    end,
    *,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    defs_coverage_file: Path = FUTURES_DEFS_COVERAGE_FILE,
    coverage_file: Path = FUTURES_COVERAGE_FILE,
    daily_root: Path = DAILY_FUTURES_DIR,
    daily_coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
) -> dict[str, list[Interval]]:
    """Everything still missing on disk for ``specs`` (no API). Keys: absolute tickers
    (1-minute bars), ``daily:<ticker>`` (daily statistics a ``.v.N`` ranking needs) and
    ``definitions:<root>``. A ``.v.N`` spec's bars are planned from the volume on disk now;
    days whose volume isn't fetched yet rank in calendar order until it is."""
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
        for ticker, (w0, w1) in volume_candidate_windows(root_specs, cfg, contracts, start, end).items():
            gaps = dl.plan_daily_update(ticker, w0, w1, coverage_file=daily_coverage_file)
            if gaps:
                plan[f"daily:{ticker}"] = gaps
        _, cal_map = needed_contract_windows(root_specs, cfg, contracts, start, end)
        vol_map = volume_ranked_mapping(root_specs, cfg, contracts, start, end, fetch_missing=False,
                                        daily_root=daily_root, daily_coverage_file=daily_coverage_file)
        for ticker, (w0, w1) in series_windows(root_specs, cal_map, vol_map).items():
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
    daily_root: Path = DAILY_FUTURES_DIR,
    daily_coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """Parent: relative series for ``specs`` over ``[start, end)`` (unadjusted across rolls)."""
    start, end = to_utc_day(start), to_utc_day(end)
    frames: list[pd.DataFrame] = []
    for root, root_specs in group_by_root(specs).items():
        cfg = FUTURES_ROOTS[root]
        contracts = contracts_pipe.ensure_contracts(
            cfg, start - VOLUME_HISTORY if _volume_specs(root_specs) else start, end,
            fetch_missing=fetch_missing, contracts_file=contracts_file,
            coverage_file=defs_coverage_file, max_cost_usd=max_cost_usd, client=client,
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
                gaps = fut.plan_futures_update(ticker, w0, w1, coverage_file=coverage_file)
                if gaps:
                    fut.fetch_and_store_futures(
                        ticker, gaps, dataset=cfg.dataset, root=root_dir,
                        coverage_file=coverage_file, max_cost_usd=max_cost_usd, client=client,
                    )
        bars = fut.read_futures_from_disk(list(windows), start, end, root=root_dir)
        if bars.empty:
            continue
        for spec in root_specs:
            mapping = cal_map if spec.kind == "c" else vol_map
            frames.append(apply_mapping(bars, mapping, spec.rank, spec.label, dataset=cfg.dataset))
    if not frames:
        empty = fut.read_futures_from_disk([], start, start)
        return empty.assign(contract=pd.Series(dtype="str"))
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
