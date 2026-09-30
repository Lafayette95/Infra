"""The INTRADAY cycle's building blocks (not yet scheduled): intraday px for the ZQ strip
and the intraday WIRP derived from it. Plain Python, like the daily cycle (CLAUDE.md
12) - a scheduler will only wrap these.

* ``backfill_intraday_px``: ``bbo-1m`` quotes (what intraday WIRP prices from) and
  ``ohlcv-1m`` trade bars for every contract of the ``INTRADAY_UNIVERSE`` roots
  (all 12 nearest ZQ contracts, point in time - the daily cycle's own universe rule,
  ``infra.cycle.universe``), over the days each is in the universe. Rule 2.1 as usual:
  only never-queried days are fetched.
* ``backfill_intraday_wirp``: ``infra.pipeline.wirp.intraday_schedules`` on the
  ``WIRP_INTRADAY_GRID`` (15 min), stored in ``Derived/WIRP_intraday``; every
  recomputed grid time REPLACES its rows (as the daily WIRP replaces its day).

1-second data (``ohlcv-1s``, ``bbo-1s``) is deliberately NOT here - on demand only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from infra import config
from infra.config import DAILY_BACKFILL, MAX_COST_USD, SCHEMA_BBO_1M, SCHEMA_OHLCV, DailyBackfillSpec
from infra.cycle.paths import CyclePaths
from infra.cycle.universe import daily_universe
from infra.pipeline import bbo, futures
from infra.pipeline.wirp import intraday_schedules
from infra.storage import parquet_store

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)

# Roots the intraday cycle covers, with the SAME point-in-time universe rule as the daily
# cycle: ZQ's 12 nearest contracts (user decision 2026-09-30: all ZQ, not only the months
# WIRP happens to read on a given day).
INTRADAY_UNIVERSE: dict[str, DailyBackfillSpec] = {"ZQ": DAILY_BACKFILL["ZQ"]}
INTRADAY_SCHEMAS = (SCHEMA_BBO_1M, SCHEMA_OHLCV)
WIRP_INTRADAY_KEYS = ["timestamp", "meeting_date", "outcome_step"]


@dataclass(frozen=True)
class IntradayPaths:
    """Every store the intraday cycle touches (rebasable under a temp dir for tests)."""
    bbo_dir: Path
    bbo_coverage: Path
    ohlcv_dir: Path
    ohlcv_coverage: Path
    wirp_intraday_dir: Path

    @classmethod
    def default(cls) -> IntradayPaths:
        return cls(config.BBO_FUTURES_DIR, config.BBO_FUTURES_COVERAGE_FILE, config.FUTURES_DIR,
                   config.FUTURES_COVERAGE_FILE, config.WIRP_INTRADAY_DIR)

    @classmethod
    def under(cls, root: Path) -> IntradayPaths:
        base = cls.default()
        return cls(**{f.name: root / getattr(base, f.name).relative_to(config.DATABASE_ROOT) for f in fields(cls)})


def plan_intraday_px(members, start: pd.Timestamp, end: pd.Timestamp, *, ipaths: IntradayPaths,
                     schemas=INTRADAY_SCHEMAS) -> dict[tuple[str, str], list]:
    """``{(schema, ticker): gaps}`` never queried, over each member's days in the universe
    within ``[start, end]`` (inclusive). No API - also the dry run's view."""
    plan = {}
    for m in members.values():
        w0, w1 = max(m.first, start), min(m.last, end) + _ONE_DAY
        if w0 >= w1:
            continue
        for schema in schemas:
            if schema == SCHEMA_BBO_1M:
                gaps = bbo.plan_bbo_update(m.ticker, w0, w1, coverage_file=ipaths.bbo_coverage)
            else:
                gaps = futures.plan_futures_update(m.ticker, w0, w1, coverage_file=ipaths.ohlcv_coverage)
            if gaps:
                plan[(schema, m.ticker)] = gaps
    return plan


def backfill_intraday_px(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    ipaths: IntradayPaths | None = None,
    specs: dict[str, DailyBackfillSpec] = INTRADAY_UNIVERSE,
    schemas=INTRADAY_SCHEMAS,
    refresh_contracts: bool = False,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> dict:
    """Fetch and store every planned (schema, contract) gap. Per-request failures are
    collected, not raised, so one bad contract never hides the others."""
    paths, ipaths = paths or CyclePaths.default(), ipaths or IntradayPaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    members, universe_errors = daily_universe(start, end, paths=paths, specs=specs,
                                              refresh_contracts=refresh_contracts, client=client)
    plan = plan_intraday_px(members, start, end, ipaths=ipaths, schemas=schemas)
    rows, errors = {s: 0 for s in schemas}, {}
    for (schema, ticker), gaps in sorted(plan.items()):
        dataset = members[ticker].dataset
        try:
            if schema == SCHEMA_BBO_1M:
                rows[schema] += bbo.fetch_and_store_bbo(ticker, gaps, dataset=dataset, root=ipaths.bbo_dir,
                                                        coverage_file=ipaths.bbo_coverage,
                                                        max_cost_usd=max_cost_usd, client=client)
            else:
                rows[schema] += futures.fetch_and_store_futures(ticker, gaps, dataset=dataset, root=ipaths.ohlcv_dir,
                                                                coverage_file=ipaths.ohlcv_coverage,
                                                                max_cost_usd=max_cost_usd, client=client)
        except Exception as exc:
            errors[f"{schema} {ticker}"] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            log.warning("intraday fetch failed for %s %s: %s", schema, ticker, errors[f"{schema} {ticker}"])
    return {"members": members, "universe_errors": universe_errors, "planned": len(plan),
            "rows": rows, "fetch_errors": errors}


def backfill_intraday_wirp(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    ipaths: IntradayPaths | None = None,
    grid: str | None = None,
) -> dict:
    """Intraday WIRP on the grid for ``[start, end]`` (inclusive days), replacing every
    grid time of those days in ``Derived/WIRP_intraday``. No API."""
    paths, ipaths = paths or CyclePaths.default(), ipaths or IntradayPaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    df = intraday_schedules(start, end + _ONE_DAY, grid=grid, source="bbo-1m",
                            contracts_file=paths.contracts_file, close_root=paths.daily_futures_dir,
                            intraday_root=ipaths.bbo_dir, adjustments_dir=paths.adjustments_dir)
    parquet_store.delete_where(ipaths.wirp_intraday_dir, lambda part: pd.to_datetime(part["timestamp"]).between(
        start, end + _ONE_DAY, inclusive="left"))
    if not df.empty:
        parquet_store.write_partitioned(df, ipaths.wirp_intraday_dir, WIRP_INTRADAY_KEYS)
    return {"rows": len(df), "grid_times": df["timestamp"].nunique() if not df.empty else 0}


def check_wirp_intraday(ipaths: IntradayPaths, start, end) -> list[str]:
    """Problems in the stored intraday WIRP over ``[start, end]`` (empty list = clean):
    probabilities in [0, 1] summing to 1 per (grid time, meeting) - the daily WIRP's rule."""
    df = parquet_store.read_partitioned(ipaths.wirp_intraday_dir, start=pd.Timestamp(start),
                                        end=pd.Timestamp(end) + _ONE_DAY)
    if df is None or df.empty:
        return ["no intraday WIRP rows in window"]
    problems = []
    if not df["probability"].between(0.0, 1.0).all():
        problems.append(f"{(~df['probability'].between(0.0, 1.0)).sum()} probability(ies) outside [0, 1]")
    sums = df.groupby(["timestamp", "meeting_date"])["probability"].sum()
    if not np.isclose(sums, 1.0, atol=1e-9).all():
        problems.append(f"{(~np.isclose(sums, 1.0, atol=1e-9)).sum()} distribution(s) not summing to 1")
    return problems
