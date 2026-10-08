"""The INTRADAY building blocks. SCHEDULED (since 2026-10-06): the daily cycle's
``intraday`` step - ``bbo-1m`` quotes of the bond-futures HEDGE contracts (each
``SWAP_HEDGES`` root's ``.v.0``), which move swap prints and swaption forwards to a common
instant (``backfill_hedge_bbo``). Not yet scheduled: intraday px for the ZQ strip and the
intraday WIRP derived from it. Plain Python, like the daily cycle (CLAUDE.md 12) - a
scheduler only wraps these.

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
from infra.config import DAILY_BACKFILL, MAX_COST_USD, SCHEMA_BBO_1M, SCHEMA_OHLCV, SCHEMA_OHLCV_1D, DailyBackfillSpec
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
# Foreign government-bond futures (added 2026-10-07, user): 1-minute quotes of the nearest two
# contracts (the daily cycle's point-in-time universe rule) for a 16:15 London / basis snap and
# hedging, plus the Long Gilt's DAILY BARS (its ICE statistics - settlements - cost ~30x as
# much; the bar's close is the last trade). Priced 2026-10-07: history ~$17.80 (Eurex since
# 2025-03-10, the Long Gilt since 2019), ongoing ~$0.80 a month.
FOREIGN_BOND_FUTURES: dict[str, DailyBackfillSpec] = {r: DailyBackfillSpec(2) for r in ("FGBL", "FGBM", "FGBS", "FGBX", "FBTP", "R")}
FOREIGN_DAILY_BARS: dict[str, DailyBackfillSpec] = {"R": DailyBackfillSpec(2)}
WIRP_INTRADAY_KEYS = ["timestamp", "meeting_date", "outcome_step"]


@dataclass(frozen=True)
class IntradayPaths:
    """Every store the intraday cycle touches (rebasable under a temp dir for tests)."""
    bbo_dir: Path
    bbo_coverage: Path
    ohlcv_dir: Path
    ohlcv_coverage: Path
    wirp_intraday_dir: Path
    ohlcv_1d_dir: Path
    ohlcv_1d_coverage: Path

    @classmethod
    def default(cls) -> IntradayPaths:
        return cls(config.BBO_FUTURES_DIR, config.BBO_FUTURES_COVERAGE_FILE, config.FUTURES_DIR,
                   config.FUTURES_COVERAGE_FILE, config.WIRP_INTRADAY_DIR, config.OHLCV_1D_FUTURES_DIR,
                   config.OHLCV_1D_FUTURES_COVERAGE_FILE)

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
            elif schema == SCHEMA_OHLCV_1D:
                gaps = futures.plan_futures_update(m.ticker, w0, w1, coverage_file=ipaths.ohlcv_1d_coverage)
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
            elif schema == SCHEMA_OHLCV_1D:
                rows[schema] += futures.fetch_and_store_futures(ticker, gaps, dataset=dataset, root=ipaths.ohlcv_1d_dir,
                                                                coverage_file=ipaths.ohlcv_1d_coverage,
                                                                max_cost_usd=max_cost_usd, client=client,
                                                                schema=SCHEMA_OHLCV_1D)
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


# ------------------------------------------------- hedge quotes (scheduled, daily cycle)
HEDGE_GAP_DAYS = 30  # never-fetched hedge days this far back are picked up too (a missed run)


def hedge_members(start, end, *, paths: CyclePaths) -> dict[str, tuple[pd.Timestamp, pd.Timestamp, str]]:
    """``{ticker: (first day, last day, dataset)}`` - every hedge contract over ``[start,
    end]`` per ``infra.pipeline.swap_hedge.hedge_contracts`` (the book's own choice)."""
    from infra.config import FUTURES_ROOTS, SWAP_HEDGE_QUOTE_RATIO
    from infra.pipeline.swap_hedge import hedge_contracts, hedge_roots
    out = {}
    # quote-based hedge roots (Eurex, the Long Gilt) are chosen FROM stored quotes, which the
    # foreign-futures fetch (FOREIGN_BOND_FUTURES) already brings in - nothing to fetch here
    roots = [r for r in hedge_roots() if r not in SWAP_HEDGE_QUOTE_RATIO]
    for root, s in hedge_contracts(start, end, roots=roots, daily_root=paths.daily_futures_dir,
                                   contracts_file=paths.contracts_file).items():
        s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        for ticker, days in s.groupby(s.astype(str)).groups.items():
            out[ticker] = (min(days), max(days), FUTURES_ROOTS[root].dataset)
    return out


def plan_hedge_bbo(start, end, *, paths: CyclePaths, ipaths: IntradayPaths) -> dict[str, list]:
    """``{ticker: gaps}`` never queried (Rule 2.1), over each hedge contract's days."""
    plan = {}
    for ticker, (first, last, _) in hedge_members(start, end, paths=paths).items():
        gaps = bbo.plan_bbo_update(ticker, first, last + _ONE_DAY, coverage_file=ipaths.bbo_coverage)
        if gaps:
            plan[ticker] = gaps
    return plan


def backfill_hedge_bbo(start, end, *, paths: CyclePaths | None = None, ipaths: IntradayPaths | None = None,
                       max_cost_usd: float = MAX_COST_USD, client=None, dry_run: bool = False) -> dict:
    """Fetch (or, ``dry_run``, only price) every never-queried day of every hedge contract
    over ``[start - HEDGE_GAP_DAYS, end]``. Requests are clamped to Databento's available
    end by the bbo fetcher; a partial day is never claimed covered."""
    from infra.api import databento_client as api
    paths, ipaths = paths or CyclePaths.default(), ipaths or IntradayPaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    lo = start - pd.Timedelta(days=HEDGE_GAP_DAYS)
    members = hedge_members(lo, end, paths=paths)
    plan = plan_hedge_bbo(lo, end, paths=paths, ipaths=ipaths)
    rows, errors, cost = 0, {}, 0.0
    for ticker, gaps in sorted(plan.items()):
        dataset = members[ticker][2]
        try:
            if dry_run:
                client = client or api.get_client()
                avail = api.available_end(dataset, SCHEMA_BBO_1M, client)
                for g0, g1 in gaps:
                    if min(g1, avail) > g0:
                        cost += api.estimate_cost(dataset, SCHEMA_BBO_1M, [ticker], g0, min(g1, avail), "raw_symbol", client)
                continue
            rows += bbo.fetch_and_store_bbo(ticker, gaps, dataset=dataset, root=ipaths.bbo_dir,
                                            coverage_file=ipaths.bbo_coverage, max_cost_usd=max_cost_usd, client=client)
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            log.warning("hedge bbo fetch failed for %s: %s", ticker, errors[ticker])
    return {"members": {t: (str(a.date()), str(b.date())) for t, (a, b, _) in members.items()},
            "planned": {t: [(str(a), str(b)) for a, b in g] for t, g in plan.items()},
            "rows": rows, "fetch_errors": errors, "estimated_cost_usd": round(cost, 4) if dry_run else None}


# --------------------------------------------------------------------- the cycle step
def _ipaths(paths: CyclePaths) -> IntradayPaths:
    return IntradayPaths.default() if paths.database_root == config.DATABASE_ROOT else \
        IntradayPaths.under(paths.database_root)


def _run(ctx) -> dict:
    out = backfill_hedge_bbo(ctx.start, ctx.end, paths=ctx.paths, ipaths=_ipaths(ctx.paths),
                             client=ctx.options.get("client"))
    lo = ctx.start - pd.Timedelta(days=HEDGE_GAP_DAYS)
    ip = _ipaths(ctx.paths)
    fq = backfill_intraday_px(lo, ctx.end, paths=ctx.paths, ipaths=ip, specs=FOREIGN_BOND_FUTURES,
                              schemas=(SCHEMA_BBO_1M,), client=ctx.options.get("client"))
    fd = backfill_intraday_px(lo, ctx.end, paths=ctx.paths, ipaths=ip, specs=FOREIGN_DAILY_BARS,
                              schemas=(SCHEMA_OHLCV_1D,), client=ctx.options.get("client"))
    out["foreign"] = {"rows": {**fq["rows"], **fd["rows"]}, "fetch_errors": {**fq["fetch_errors"], **fd["fetch_errors"]},
                      "universe_errors": {**fq["universe_errors"], **fd["universe_errors"]}}
    return out


def _check_fetch_ok(ctx):
    f = ctx.output.get("foreign", {})
    errs = {**ctx.output.get("fetch_errors", {}), **f.get("fetch_errors", {}), **f.get("universe_errors", {})}
    if not errs:
        return True, f"{ctx.output.get('rows', 0)} hedge quote rows fetched, no errors", None
    return False, f"{len(errs)} hedge contract fetch error(s)", pd.DataFrame(
        {"ticker": list(errs), "error": list(errs.values())})


def _check_present(ctx):
    """Every hedge contract has quotes on each window day it SETTLED (a day without a
    settlement - a holiday, or not published yet - isn't judged)."""
    from infra.pipeline.daily import read_daily_from_disk
    members = hedge_members(ctx.start, ctx.end, paths=ctx.paths)
    if not members:
        return True, "no hedge contracts in the window", None
    ip = _ipaths(ctx.paths)
    st = read_daily_from_disk(list(members), ctx.start, ctx.end + _ONE_DAY, root=ctx.paths.daily_futures_dir)
    st = st.dropna(subset=["settlement_price"])
    q = bbo.read_bbo_from_disk(list(members), ctx.start, ctx.end + _ONE_DAY, root=ip.bbo_dir)
    have = set(zip(q["ticker"].astype(str), pd.to_datetime(q["timestamp"]).dt.normalize())) if len(q) else set()
    missing = [(t, d) for t, d in zip(st["ticker"].astype(str), pd.to_datetime(st["timestamp"]))
               if (t, d) not in have and members[t][0] <= d <= members[t][1]]
    if not missing:
        return True, f"quotes on every settled day of {len(members)} hedge contract(s)", None
    return False, f"{len(missing)} settled hedge contract-day(s) without quotes", pd.DataFrame(missing, columns=["ticker", "day"])


from infra.cycle.core import Check, Severity, Step  # noqa: E402

INTRADAY_STEP = Step("intraday", _run, depends_on=("px",), checks=(
    Check("hedge_bbo_fetch_ok", _check_fetch_ok, Severity.WARN),
    Check("hedge_bbo_present", _check_present, Severity.WARN),
))
