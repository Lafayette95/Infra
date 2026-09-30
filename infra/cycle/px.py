"""Step 1a - ``backfill_daily_px_data``: daily settlement prices (and open interest) for
the cycle's point-in-time contract universe, through the existing daily pipeline
(infra.pipeline.daily) - no new fetch/storage code, only which contracts and which days.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from infra.api import databento_client as api
from infra.config import DAILY_BACKFILL, MAX_COST_USD, SCHEMA_STATISTICS, DailyBackfillSpec
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, Step, StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.universe import UniverseMember, daily_universe, rank_on
from infra.pipeline import daily as dl
from infra.storage import adjustment_store

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)

# Custom outlier check (test c): a BAD PRINT, not a big market day. Each move is first put
# in units of its own contract's typical move (z = move / median |move| over the prior
# OUTLIER_LOOKBACK moves), then compared with its peers' z the same day. A contract is
# flagged when it (1) moved a lot by its own standard (|z| >= OUTLIER_Z_MIN), (2) was out
# of line with its peers (|z - median peer z| >= OUTLIER_DEV_MIN), and (3) that deviation
# REVERSED at its next session by at least OUTLIER_REVERSAL of its size. Calibrated
# 2026-09-28 on a year of real settlements (2025-07 .. 2026-09, 19,324 contract-days): it
# flags exactly three prints - ESRM7 and ESRU7 on 2026-09-11 (a ~40bp kink mid-strip,
# gone next session) and ESRZ6 on 2025-10-31 (-6bp while every neighbour rose, +5.75bp
# back) - and none of NFP 2025-08-01, the ECB-dated EUR days, 2026-04-08, or FOMC
# 2026-06-17/07-29. Each condition is load-bearing: without (1) a nearly-expired contract
# that barely moved on a shock day looks "out of line"; without (3) an FOMC meeting-month
# contract repricing alone looks like a bad print. Cost of (3): a print can only be judged
# once its next session exists - the latest day is reported as pending (warn), not judged.
OUTLIER_Z_MIN = 5.0
OUTLIER_DEV_MIN = 6.0
OUTLIER_REVERSAL = 0.5
OUTLIER_PEERS = 4  # nearest same-root contracts by expiry
OUTLIER_LOOKBACK = 60
OUTLIER_MIN_HISTORY = 20
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
def last_weekday(day: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(day).normalize()
    while day.weekday() >= 5:
        day -= _ONE_DAY
    return day


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
    """``(flagged, pending)`` bad-print candidates for days in ``[start, end]`` - see the
    OUTLIER_* constants for the rule and its calibration. ``settle`` needs history
    before ``start`` (for each contract's own scale). Peers: the OUTLIER_PEERS nearest
    same-root contracts by expiry; a root with too few contracts that day (bond futures
    carry 2) falls back to every same-currency, same-category contract (e.g. the USD
    Treasury complex) - a contract with fewer than 2 peers isn't judged."""
    from infra.config import FUTURES_ROOTS
    df = settle[settle["ticker"].isin(members)].sort_values(["ticker", "timestamp"]).copy()
    df["move"] = df.groupby("ticker")["settlement_price"].diff()
    df["scale"] = df.groupby("ticker")["move"].transform(
        lambda m: m.abs().shift(1).rolling(OUTLIER_LOOKBACK, min_periods=OUTLIER_MIN_HISTORY).median())
    df["z"] = df["move"] / df["scale"]
    df = df[np.isfinite(df["z"]) & (df["timestamp"] >= start)]
    cols = ["ticker", "timestamp", "move", "z", "dev", "peers"]
    if df.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    df["root"] = df["ticker"].map(lambda t: members[t].root)
    df["expiry"] = df["ticker"].map(lambda t: members[t].curve_order)
    df["group"] = df["root"].map(lambda r: (FUTURES_ROOTS[r].currency, FUTURES_ROOTS[r].category))

    devs, pools = [], []
    for _, day in df.groupby("timestamp"):
        for idx, r in day.iterrows():
            peers = day[(day["root"] == r["root"]) & (day.index != idx)]
            if len(peers) >= 2:
                peers = peers.iloc[(peers["expiry"] - r["expiry"]).abs().argsort()[:OUTLIER_PEERS]]
                pool = "root"
            else:
                peers = day[(day["group"] == r["group"]) & (day.index != idx)]
                pool = "currency+category"
            devs.append((idx, r["z"] - peers["z"].median() if len(peers) >= 2 else np.nan))
            pools.append((idx, pool))
    df["dev"] = pd.Series(dict(devs))
    df["peers"] = pd.Series(dict(pools))
    df = df.dropna(subset=["dev"])
    df["next_dev"] = df.groupby("ticker")["dev"].shift(-1)
    candidate = (df["z"].abs() >= OUTLIER_Z_MIN) & (df["dev"].abs() >= OUTLIER_DEV_MIN) & (df["timestamp"] <= end)
    reverted = (np.sign(df["next_dev"]) == -np.sign(df["dev"])) & (df["next_dev"].abs() >= OUTLIER_REVERSAL * df["dev"].abs())
    return (df[candidate & reverted][cols + ["next_dev"]].reset_index(drop=True),
            df[candidate & df["next_dev"].isna()][cols].reset_index(drop=True))


# The reversal test needs each judged day's NEXT session, which for the window's last day
# lies after the window. Read it if it's already on disk - only days inside the window are
# ever JUDGED, so this is a data-quality lookup, not look-ahead in any value computed.
# Without it, non-overlapping history windows (e.g. monthly) never judged a boundary day:
# ESRZ6's bad print on 2025-10-31 slipped through an Oct-2025 window this way.
_NEXT_SESSION_LOOKAHEAD = pd.Timedelta(days=10)


BAD_PRINT_SOURCE = "px_bad_print"  # this process's name in the adjustments log


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
    rows, done = [], {}  # done: (ticker, day) -> adjusted value, for consecutive bad prints
    for r in flagged.sort_values(["ticker", "timestamp"]).itertuples(index=False):
        m = members[r.ticker]
        rank = rank_on(r.ticker, r.timestamp, members)
        policy = specs[m.root].bad_print_policy(rank)
        series = hist.loc[hist["ticker"] == r.ticker].set_index("timestamp")["settlement_price"]
        original, adjusted, note = float(series[r.timestamp]), np.nan, ""
        if policy == "roll":
            for day in reversed(series.index[series.index < r.timestamp]):
                value = done.get((r.ticker, day), series[day])
                if np.isfinite(value):
                    adjusted = float(value)
                    break
            else:
                policy, note = "NA", " (roll had no earlier good value - NA instead)"
        done[(r.ticker, r.timestamp)] = adjusted
        rows.append({
            "store": dl.STORE, "timestamp": r.timestamp, "key": r.ticker, "column": "settlement_price",
            "action": policy, "original": original, "adjusted": adjusted, "source": BAD_PRINT_SOURCE,
            "detail": (f"rank {rank}: off-peer move (z {r.z:.1f}, peer deviation {r.dev:.1f}) "
                       f"reversed next session ({r.next_dev:.1f}){note}"),
            "run_day": run_day,
        })
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
)


def _run(ctx: StepContext) -> dict:
    opts = ctx.options
    return backfill_daily_px_data(
        ctx.start, ctx.end, paths=ctx.paths, force_refetch=ctx.force_refetch,
        run_day=ctx.run_day,
        **{k: opts[k] for k in ("fetch_missing", "refresh_contracts", "specs", "max_cost_usd", "client",
                                "workers") if k in opts},
    )


PX_STEP = Step("px", _run, checks=PX_CHECKS)
