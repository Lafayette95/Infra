"""Step 1a - ``backfill_daily_px_data``: daily settlement prices (and open interest) for
the cycle's point-in-time contract universe, through the existing daily pipeline
(infra.pipeline.daily) - no new fetch/storage code, only which contracts and which days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.config import DAILY_BACKFILL, MAX_COST_USD, DailyBackfillSpec
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, Step, StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.universe import UniverseMember, daily_universe
from infra.pipeline import daily as dl

_ONE_DAY = pd.Timedelta(days=1)

# Custom outlier check: a day-over-day settlement move larger than OUTLIER_K times the
# contract's own median absolute daily move over its prior OUTLIER_LOOKBACK moves.
OUTLIER_K = 10.0
OUTLIER_LOOKBACK = 60
OUTLIER_MIN_HISTORY = 20
# A contract that rolls INTO the universe mid-window is also fetched this many calendar
# days before its first universe day, so that day still has a prior settlement to diff
# against (the bmk pnl step). Contracts in the universe from the window's start get
# their prior day from earlier runs instead - widening their fetch would silently turn
# the scheduled run's T-3 revision window into a much longer one.
PRIOR_SETTLEMENT_DAYS = 7


def backfill_daily_px_data(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    force_refetch: bool = False,
    fetch_missing: bool = True,
    refresh_contracts: bool = True,
    specs: dict[str, DailyBackfillSpec] = DAILY_BACKFILL,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> dict:
    """Settlement/OI for every universe contract over the days it is in the universe,
    within ``[start, end]`` (inclusive). Per-contract failures are collected, not raised,
    so one bad contract never hides the others - the ``px_fetch_ok`` check fails on them.
    """
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    members, universe_errors = daily_universe(
        start, end, paths=paths, specs=specs, refresh_contracts=refresh_contracts,
        max_cost_usd=max_cost_usd, client=client,
    )
    fetch_errors: dict[str, str] = {}
    rows = 0
    if fetch_missing:
        for m in members.values():
            w0, w1 = max(m.first, start), min(m.last, end) + _ONE_DAY
            if m.first > start:
                w0 -= pd.Timedelta(days=PRIOR_SETTLEMENT_DAYS)
            try:
                gaps = dl.plan_daily_update(
                    m.ticker, w0, w1, coverage_file=paths.daily_futures_coverage,
                    force_refetch=force_refetch,
                )
                if gaps:
                    rows += dl.fetch_and_store_daily(
                        m.ticker, gaps, dataset=m.dataset, root=paths.daily_futures_dir,
                        coverage_file=paths.daily_futures_coverage, max_cost_usd=max_cost_usd,
                        client=client,
                    )
            except Exception as exc:
                fetch_errors[m.ticker] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return {"members": members, "universe_errors": universe_errors,
            "fetch_errors": fetch_errors, "rows": rows}


# ------------------------------------------------------------------------ checks
def last_weekday(day: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(day).normalize()
    while day.weekday() >= 5:
        day -= _ONE_DAY
    return day


def _members(ctx: StepContext) -> dict[str, UniverseMember]:
    return ctx.output.get("members", {})


def _settlements(ctx: StepContext, lookback: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame:
    df = dl.read_daily_from_disk(list(_members(ctx)), ctx.start - lookback, ctx.end + _ONE_DAY,
                                 root=ctx.paths.daily_futures_dir)
    df["ticker"] = df["ticker"].astype(str)
    return df.dropna(subset=["settlement_price"])


def _check_fetch_ok(ctx: StepContext):
    problems = {**{f"root {r}": e for r, e in ctx.output.get("universe_errors", {}).items()},
                **ctx.output.get("fetch_errors", {})}
    if not problems:
        return True, f"{len(_members(ctx))} contracts, {ctx.output.get('rows', 0)} rows written", None
    details = pd.DataFrame({"what": list(problems), "error": list(problems.values())})
    return False, f"{len(problems)} fetch error(s)", details


def _expected_on(ctx: StepContext) -> tuple[pd.Timestamp, list[UniverseMember]]:
    day = last_weekday(ctx.end)
    return day, [m for m in _members(ctx).values() if day >= ctx.start and m.expected_on(day)]


def _check_present(ctx: StepContext):
    """Test (a). Only judged for datasets that published SOMETHING that day - a dataset
    with nothing at all is ambiguous (holiday vs not-yet-published) and is reported by
    the separate, warning-level px_dataset_present check instead."""
    day, expected = _expected_on(ctx)
    if not expected:
        return True, "no contracts expected (window has no weekday)", None
    have = _settlements(ctx)
    have = set(have.loc[have["timestamp"] == day, "ticker"])
    live_datasets = {m.dataset for m in expected if m.ticker in have}
    missing = [m for m in expected if m.dataset in live_datasets and m.ticker not in have]
    if not missing:
        return True, f"all {len(expected)} expected settlements present for {day.date()}", None
    details = pd.DataFrame([(m.root, m.ticker, m.dataset) for m in missing], columns=["root", "ticker", "dataset"])
    return False, f"{len(missing)} of {len(expected)} settlements missing for {day.date()}", details


def _check_dataset_present(ctx: StepContext):
    day, expected = _expected_on(ctx)
    have = _settlements(ctx)
    have = set(have.loc[have["timestamp"] == day, "ticker"])
    empty = sorted({m.dataset for m in expected} - {m.dataset for m in expected if m.ticker in have})
    if not empty:
        return True, f"every dataset has settlements for {day.date()}", None
    return False, (f"no settlements at all for {', '.join(empty)} on {day.date()} - exchange "
                   f"holiday, or not published yet (no exchange calendar to tell them apart)"), None


def _check_sane(ctx: StepContext):
    df = _settlements(ctx)
    bad = df[~np.isfinite(df["settlement_price"]) | (df["settlement_price"] <= 0)]
    if bad.empty:
        return True, f"{len(df)} settlements finite and positive", None
    return False, f"{len(bad)} non-positive / non-finite settlement(s)", bad


def _check_outliers(ctx: StepContext):
    """Test (c): flags a move > OUTLIER_K x the contract's own recent median |move|.
    Contracts with too little history, or a zero median (rarely-moving deep deferreds),
    are not judged rather than guessed at."""
    hist = _settlements(ctx, lookback=pd.Timedelta(days=180)).sort_values(["ticker", "timestamp"])
    flagged = []
    for ticker, g in hist.groupby("ticker"):
        moves = g.set_index("timestamp")["settlement_price"].diff().dropna()
        for day, move in moves[moves.index >= ctx.start].items():
            prior = moves[moves.index < day].tail(OUTLIER_LOOKBACK).abs()
            scale = prior.median() if len(prior) >= OUTLIER_MIN_HISTORY else 0.0
            if scale > 0 and abs(move) > OUTLIER_K * scale:
                flagged.append((ticker, day, move, scale))
    if not flagged:
        return True, f"no moves > {OUTLIER_K:g}x recent median", None
    details = pd.DataFrame(flagged, columns=["ticker", "timestamp", "move", "median_abs_move"])
    return False, f"{len(details)} outlier move(s)", details


PX_CHECKS = (
    Check("px_fetch_ok", _check_fetch_ok),
    Check("px_present", _check_present),
    Check("px_dataset_present", _check_dataset_present, severity=Severity.WARN),
    revision_check(lambda p: p.daily_futures_dir, ["timestamp", "ticker"], name="px_no_revisions",
                   equals_in=lambda ctx: {"ticker": list(_members(ctx))}),
    Check("px_sane", _check_sane),
    Check("px_outliers", _check_outliers),
)


def _run(ctx: StepContext) -> dict:
    opts = ctx.options
    return backfill_daily_px_data(
        ctx.start, ctx.end, paths=ctx.paths, force_refetch=ctx.force_refetch,
        **{k: opts[k] for k in ("fetch_missing", "refresh_contracts", "specs", "max_cost_usd", "client")
           if k in opts},
    )


PX_STEP = Step("px", _run, checks=PX_CHECKS)
