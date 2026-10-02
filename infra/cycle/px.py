"""Step 1a - ``backfill_daily_px_data``: every daily px input, as two halves:

* FUTURES (this module, ``backfill_daily_futures_px``): settlement prices (and open
  interest) for the cycle's point-in-time contract universe, through the existing daily
  pipeline (infra.pipeline.daily) - only which contracts and which days;
* CASH BONDS (``infra.cycle.px_bonds.backfill_daily_bond_px``): par yields per sovereign
  curve (``US_BOND_10y`` ...), through infra.pipeline.bonds.

Both share one bad-print rule (infra.cycle.bad_prints), the adjustments log and the
generic revision check; each half has its own checks.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from infra.api import databento_client as api
from infra.config import BOND_CURVES, DAILY_BACKFILL, MAX_COST_USD, SCHEMA_STATISTICS, BondCurve, DailyBackfillSpec
from infra.cycle.bad_prints import (  # noqa: F401 - re-exported, the px step's public API
    _NEXT_SESSION_LOOKAHEAD,
    BAD_PRINT_SOURCE,
    last_weekday,
    peer_outliers_frame,
    treatment_rows,
)
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, Step, StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.px_bonds import BOND_CHECKS, backfill_daily_bond_px
from infra.cycle.px_repo import REPO_CHECKS, backfill_daily_repo_px
from infra.cycle.px_treasuries import TREASURY_CHECKS, backfill_daily_treasury_px
from infra.cycle.universe import UniverseMember, daily_universe, rank_on
from infra.pipeline import daily as dl
from infra.storage import adjustment_store

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)

# A contract that rolls INTO the universe mid-window is also fetched this many calendar
# days before its first universe day, so that day still has a prior settlement to diff
# against (the bmk pnl step). Contracts in the universe from the window's start get
# their prior day from earlier runs instead - widening their fetch would silently turn
# the scheduled run's T-3 revision window into a much longer one.
PRIOR_SETTLEMENT_DAYS = 7
# Contracts fetched concurrently. Databento answers each request with a latency that can
# be tens of seconds (2026-09-28: ~85s per one-contract statistics request), so a
# strictly sequential loop over ~150 contracts took hours. Only the NETWORK half runs in
# parallel; every file write stays sequential on the calling thread.
PX_FETCH_WORKERS = 8


def fetch_window(m: UniverseMember, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``[w0, w1)`` to fetch for one member: its universe days within the window, plus
    PRIOR_SETTLEMENT_DAYS before its first day if it rolls in mid-window."""
    w0, w1 = max(m.first, start), min(m.last, end) + _ONE_DAY
    if m.first > start:
        w0 -= pd.Timedelta(days=PRIOR_SETTLEMENT_DAYS)
    return w0, w1


def plan_daily_px_data(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    force_refetch: bool = False,
    specs: dict[str, DailyBackfillSpec] = DAILY_BACKFILL,
) -> tuple[dict[str, tuple[UniverseMember, list]], dict[str, str]]:
    """What ``backfill_daily_px_data`` WOULD fetch - ``{ticker: (member, gaps)}`` for
    tickers with something to fetch, plus universe errors. No API, no definitions pulled
    (uses the contracts already on disk), so a dry run is free."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    members, errors = daily_universe(start, end, paths=paths, specs=specs, refresh_contracts=False)
    return plan_gaps(members, start, end, paths=paths, force_refetch=force_refetch), errors


def plan_gaps(
    members: dict[str, UniverseMember],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    paths: CyclePaths,
    force_refetch: bool = False,
) -> dict[str, tuple[UniverseMember, list]]:
    """``{ticker: (member, gaps)}`` for every member with days not yet covered on disk
    (Rule 2.1) - the ONE planning rule, shared by the real run and the free dry run so
    the dry run always prices exactly what the real run would fetch."""
    plan = {}
    for m in members.values():
        w0, w1 = fetch_window(m, start, end)
        gaps = dl.plan_daily_update(m.ticker, w0, w1, coverage_file=paths.daily_futures_coverage,
                                    force_refetch=force_refetch)
        if gaps:
            plan[m.ticker] = (m, gaps)
    return plan


def fetch_and_store_px(
    work: dict[str, tuple[UniverseMember, list]],
    *,
    paths: CyclePaths,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
    workers: int = PX_FETCH_WORKERS,
) -> tuple[int, dict[str, str]]:
    """Fetch every planned ticker's gaps concurrently and store each as it lands.
    Returns ``(rows stored, {ticker: error})`` - per-ticker failures are collected, not
    raised, so one bad contract never hides the others."""
    rows, errors = 0, {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="px-fetch") as pool:
        futures = {
            pool.submit(dl.fetch_daily_raw, t, gaps, dataset=m.dataset,
                        max_cost_usd=max_cost_usd, client=client): t
            for t, (m, gaps) in work.items()
        }
        for fut in as_completed(futures):  # store each as it lands, one at a time
            ticker = futures[fut]
            try:
                rows += dl.store_daily_raw(ticker, fut.result(), dataset=work[ticker][0].dataset,
                                           root=paths.daily_futures_dir,
                                           coverage_file=paths.daily_futures_coverage)
            except Exception as exc:
                errors[ticker] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                log.warning("px fetch failed for %s: %s", ticker, errors[ticker])
    return rows, errors


def backfill_daily_px_data(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    force_refetch: bool = False,
    fetch_missing: bool = True,
    run_day: pd.Timestamp | None = None,
    bonds: bool = True,
    bond_curves: dict[str, BondCurve] = BOND_CURVES,
    bond_sources=None,
    treasuries: bool = True,
    treasury_fetch=None,
    repo: bool = True,
    **futures_options,
) -> dict:
    """Every daily px input over ``[start, end]`` (inclusive): futures settlements
    (``backfill_daily_futures_px``, whose keys stay at the top level - downstream steps
    read them there), cash-bond par yields under ``"bonds"`` (``bonds=False`` skips
    them), Treasury prices per CUSIP under ``"treasuries"`` (``treasuries=False`` skips
    them) and repo rates + securities lending under ``"repo"`` (``repo=False`` skips them). ``futures_options``: ``refresh_contracts``, ``specs``, ``max_cost_usd``,
    ``client``, ``workers``."""
    out = backfill_daily_futures_px(start, end, paths=paths, force_refetch=force_refetch,
                                     fetch_missing=fetch_missing, run_day=run_day, **futures_options)
    if bonds:
        out["bonds"] = backfill_daily_bond_px(start, end, paths=paths, force_refetch=force_refetch,
                                              fetch_missing=fetch_missing, curves=bond_curves,
                                              sources=bond_sources, run_day=run_day)
    if treasuries:
        out["treasuries"] = backfill_daily_treasury_px(start, end, paths=paths, force_refetch=force_refetch,
                                                       fetch_missing=fetch_missing, fetch=treasury_fetch)
    if repo:
        out["repo"] = backfill_daily_repo_px(start, end, paths=paths, force_refetch=force_refetch,
                                             fetch_missing=fetch_missing)
    return out


def backfill_daily_futures_px(
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
    workers: int = PX_FETCH_WORKERS,
    run_day: pd.Timestamp | None = None,
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
    rows, fetch_errors = 0, {}
    if fetch_missing:
        work = plan_gaps(members, start, end, paths=paths, force_refetch=force_refetch)
        rows, fetch_errors = fetch_and_store_px(work, paths=paths, max_cost_usd=max_cost_usd,
                                                client=client, workers=workers)
    treated, pending = treat_bad_prints(start, end, members, specs, paths, run_day=run_day)
    return {"members": members, "universe_errors": universe_errors, "fetch_errors": fetch_errors,
            "rows": rows, "available_end": _availability(members, end, client) if fetch_missing else {},
            "bad_prints": treated, "pending_bad_prints": pending}


def _availability(members: dict[str, UniverseMember], end: pd.Timestamp, client) -> dict[str, pd.Timestamp]:
    """Each dataset's advertised available end, recorded for the presence check - only
    looked up when the window reaches the present (history backfills never need it)."""
    if end < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=7):
        return {}
    out = {}
    for ds in {m.dataset for m in members.values()}:
        try:
            out[ds] = api.available_end(ds, SCHEMA_STATISTICS, client)
        except Exception:  # unknown availability -> the check falls back to the window end
            pass
    return out


# ------------------------------------------------------------------------ checks
def _members(ctx: StepContext) -> dict[str, UniverseMember]:
    return ctx.output.get("members", {})


def _settlements(ctx: StepContext, lookback: pd.Timedelta = pd.Timedelta(0),
                 lookahead: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame:
    df = dl.read_daily_from_disk(list(_members(ctx)), ctx.start - lookback, ctx.end + _ONE_DAY + lookahead,
                                 root=ctx.paths.daily_futures_dir, adjusted=False)  # px judges RAW data
    df["ticker"] = df["ticker"].astype(str)
    return df.dropna(subset=["settlement_price"])


def _check_fetch_ok(ctx: StepContext):
    problems = {**{f"root {r}": e for r, e in ctx.output.get("universe_errors", {}).items()},
                **ctx.output.get("fetch_errors", {})}
    if not problems:
        return True, f"{len(_members(ctx))} contracts, {ctx.output.get('rows', 0)} rows written", None
    details = pd.DataFrame({"what": list(problems), "error": list(problems.values())})
    return False, f"{len(problems)} fetch error(s): {', '.join(list(problems)[:10])}", details


def complete_day(ctx: StepContext, dataset: str) -> pd.Timestamp:
    """The latest weekday ``dataset`` has FULLY published within the window: the day before
    its advertised available end (GLBX runs ~8h behind now, Eurex ~a day - see
    api.available_end), capped at the window's end. Without a recorded availability
    (history runs, offline tests), just the window's last weekday."""
    available = ctx.output.get("available_end", {}).get(dataset)
    cap = ctx.end if available is None else min(ctx.end, available.normalize() - _ONE_DAY)
    return last_weekday(cap)


def _expected(ctx: StepContext) -> dict[str, tuple[pd.Timestamp, list[UniverseMember]]]:
    """``{dataset: (its complete day, members expected to have settled that day)}``."""
    out = {}
    for ds in sorted({m.dataset for m in _members(ctx).values()}):
        day = complete_day(ctx, ds)
        members = [m for m in _members(ctx).values()
                   if m.dataset == ds and day >= ctx.start and m.expected_on(day)]
        if members:
            out[ds] = (day, members)
    return out


def _settled_on(ctx: StepContext) -> set[tuple[str, pd.Timestamp]]:
    df = _settlements(ctx)
    return set(zip(df["ticker"], df["timestamp"]))


def _check_present(ctx: StepContext):
    """Test (a): per dataset, every expected contract settled on the latest day that
    dataset has fully published. Only judged for datasets that published SOMETHING that
    day - nothing at all is ambiguous (exchange holiday) and is reported by the separate,
    warning-level px_dataset_present check instead."""
    expected, have = _expected(ctx), _settled_on(ctx)
    if not expected:
        return True, "no contracts expected (window has no complete weekday)", None
    missing, judged = [], []
    for ds, (day, members) in expected.items():
        if not any((m.ticker, day) in have for m in members):
            continue
        judged.append(f"{ds} {day.date()}")
        missing += [(m.root, m.ticker, ds, day) for m in members if (m.ticker, day) not in have]
    n = sum(len(ms) for _, ms in expected.values())
    if not missing:
        return True, f"all {n} expected settlements present ({'; '.join(judged) or 'no dataset judged'})", None
    details = pd.DataFrame(missing, columns=["root", "ticker", "dataset", "timestamp"])
    return False, f"{len(missing)} of {n} expected settlements missing", details


def _check_dataset_present(ctx: StepContext):
    expected, have = _expected(ctx), _settled_on(ctx)
    empty = [f"{ds} on {day.date()}" for ds, (day, members) in expected.items()
             if not any((m.ticker, day) in have for m in members)]
    if not empty:
        return True, "every dataset has settlements for its latest complete day", None
    return False, (f"no settlements at all for {', '.join(empty)} - exchange holiday, or a "
                   f"publication problem (no exchange calendar to tell them apart)"), None


def _check_sane(ctx: StepContext):
    df = _settlements(ctx)
    bad = df[~np.isfinite(df["settlement_price"]) | (df["settlement_price"] <= 0)]
    if bad.empty:
        return True, f"{len(df)} settlements finite and positive", None
    return False, f"{len(bad)} non-positive / non-finite settlement(s)", bad


def peer_outliers(
    settle: pd.DataFrame, members: dict[str, UniverseMember], start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Futures bad-print candidates (``peer_outliers_frame`` on settlements): peers are
    the nearest same-root contracts by expiry; a root with too few contracts that day
    (bond futures carry 2) falls back to every same-currency, same-category contract
    (e.g. the USD Treasury complex)."""
    from infra.config import FUTURES_ROOTS
    meta = pd.DataFrame(
        [(t, m.root, m.curve_order, (FUTURES_ROOTS[m.root].currency, FUTURES_ROOTS[m.root].category))
         for t, m in members.items()],
        columns=["ticker", "root", "order", "group"]).set_index("ticker")
    return peer_outliers_frame(settle.rename(columns={"settlement_price": "value"}), meta, start, end,
                               group_label="currency+category")


def treat_bad_prints(
    start: pd.Timestamp, end: pd.Timestamp, members: dict[str, UniverseMember],
    specs: dict[str, DailyBackfillSpec], paths: CyclePaths, *, run_day: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect bad prints in ``[start, end]`` on the RAW settlements (``peer_outliers``) and
    treat each by its contract's policy that day (``DailyBackfillSpec.bad_print_policy``
    of its rank): ``NA`` drops the value, ``roll`` carries the last good settlement
    forward. Treatments go to the adjustments log - the stored settlement is never
    changed - replacing this process's earlier verdicts for the window, so a print later
    corrected upstream stops being adjusted. Returns ``(treated, pending)``."""
    run_day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if run_day is None else pd.Timestamp(run_day)
    hist = dl.read_daily_from_disk(list(members), start - pd.Timedelta(days=180),
                                   end + _ONE_DAY + _NEXT_SESSION_LOOKAHEAD,
                                   root=paths.daily_futures_dir, adjusted=False)
    hist["ticker"] = hist["ticker"].astype(str)
    hist = hist.dropna(subset=["settlement_price"])
    flagged, pending = peer_outliers(hist, members, start, end)
    adjustment_store.clear(paths.adjustments_dir, store=dl.STORE, source=BAD_PRINT_SOURCE,
                           keys=list(members), start=start, end=end)
    def policy(r):
        rank = rank_on(r.ticker, r.timestamp, members)
        return specs[members[r.ticker].root].bad_print_policy(rank), f"rank {rank}: "

    rows = treatment_rows(flagged, hist.rename(columns={"settlement_price": "value"}), policy=policy,
                          store=dl.STORE, column="settlement_price", run_day=run_day)
    treated = pd.DataFrame(rows, columns=adjustment_store.COLUMNS)
    adjustment_store.record(paths.adjustments_dir, treated)
    return treated, pending


def _check_outliers(ctx: StepContext):
    """Test (c), an EXCEPTION not a stop: bad prints found and how each was treated
    (NA / roll, per the contract's policy) - the run continues on the treated data."""
    treated = ctx.output.get("bad_prints", pd.DataFrame())
    if treated.empty:
        n = len(ctx.output.get("pending_bad_prints", []))
        return True, f"no bad prints ({n} candidate(s) awaiting their next session)", None
    counts = ", ".join(f"{n} {a}" for a, n in treated["action"].value_counts().items())
    details = treated[["key", "timestamp", "action", "original", "adjusted", "detail"]].rename(columns={"key": "ticker"})
    return False, f"{len(treated)} bad print(s) treated ({counts})", details


def _check_outliers_pending(ctx: StepContext):
    """Warning: a print already out of line with its peers whose next session hasn't
    happened yet - judged (and treated, if it reverses) by the next run."""
    pending = ctx.output.get("pending_bad_prints", pd.DataFrame())
    if pending.empty:
        return True, "no off-peer prints awaiting judgement", None
    return False, f"{len(pending)} off-peer print(s) to be judged once their next session exists", pending


PX_CHECKS = (
    Check("px_fetch_ok", _check_fetch_ok),
    Check("px_present", _check_present),
    Check("px_dataset_present", _check_dataset_present, severity=Severity.WARN),
    revision_check(lambda p: p.daily_futures_dir, ["timestamp", "ticker"], name="px_no_revisions",
                   equals_in=lambda ctx: {"ticker": list(_members(ctx))}),
    Check("px_sane", _check_sane),
    Check("px_outliers", _check_outliers, severity=Severity.WARN),
    Check("px_outliers_pending", _check_outliers_pending, severity=Severity.WARN),
    *BOND_CHECKS,
    *TREASURY_CHECKS,
    *REPO_CHECKS,
)


def _run(ctx: StepContext) -> dict:
    opts = ctx.options
    return backfill_daily_px_data(
        ctx.start, ctx.end, paths=ctx.paths, force_refetch=ctx.force_refetch,
        run_day=ctx.run_day,
        **{k: opts[k] for k in ("fetch_missing", "refresh_contracts", "specs", "max_cost_usd", "client",
                                "workers", "bonds", "bond_curves", "bond_sources", "treasuries",
                                "treasury_fetch", "repo") if k in opts},
    )


PX_STEP = Step("px", _run, checks=PX_CHECKS)
