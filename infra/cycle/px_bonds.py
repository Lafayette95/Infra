"""Step 1a, cash bonds - ``backfill_daily_bond_px``: daily par yields for every enabled
curve in ``BOND_CURVES`` (``US_BOND_2y`` .. ``UK_BOND_30y``), through
``infra.pipeline.bonds`` - no new fetch/storage code here, only which curves and days,
plus the same regression checks as the futures half of the px step: fetch ok, presence,
no revisions, sanity, and bad prints (the shared rule, peers = neighbouring tenors).

Called by ``infra.cycle.px.backfill_daily_px_data``, which returns this under ``"bonds"``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from infra.config import BOND_CURVES, BondCurve, bond_ticker
from infra.cycle.bad_prints import (
    _NEXT_SESSION_LOOKAHEAD,
    BAD_PRINT_SOURCE,
    last_weekday,
    peer_outliers_frame,
    treatment_rows,
)
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, StepContext
from infra.coverage.intervals import merge_intervals
from infra.cycle.paths import CyclePaths
from infra.pipeline import bonds as pb
from infra.storage import adjustment_store

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)

# A curve whose latest stored day is more than this many BUSINESS days before the window's
# last weekday is reported stale (warn). Covers the sources' own publication lag (the
# Treasury publishes the same afternoon; the BoE's workbook a business day or so later)
# plus one holiday - there's no exchange/holiday calendar to be exact with (TOFIX.md).
STALE_BUSINESS_DAYS = 2
# Also re-plan any never-covered days in this many calendar days BEFORE the window (the
# sources are free). Needed because a source can publish a day LATE and out of the
# scheduled run's short window: the BoE's current-month workbook switches to the new
# month on the 1st, and the previous month's last days only reach its archive once that
# is refreshed (2026-09-03 for August) - uncovered until then, and never re-asked by a
# window that has already moved past them.
GAP_LOOKBACK_DAYS = 30
# Bonds only: a bad print must ALSO be out of line with its peers' median move by this
# much in raw units (percent: 0.05 = 5bp). A source quoting 2 decimals (the US Treasury)
# rounds a quiet tenor's own typical move to ~1bp, inflating its z-score vs its
# neighbours'. Found 2026-09-30 on the Bundesbank's 2-decimal par series (since replaced
# by its full-precision Svensson parameters): the DE 2y was "flagged" on 2022-03-07
# (-9bp vs -9/-6/-6 on 3y/5y/7y) and 2026-03-09/10 (+10/-12bp vs +9/-11 on the 3y), all
# genuine curve moves 2.5-3.5bp off the peer median. With it, nothing in ten years of
# US/UK/DE is flagged, while a lone 40bp spike on one tenor still is.
BOND_MIN_ABS_DEV = 0.05
# Plausible range for a sovereign par yield, percent - a value outside is a parsing /
# scaling error, not a market move.
YIELD_BOUNDS_PCT = (-5.0, 30.0)


def _enabled(curves: dict[str, BondCurve]) -> dict[str, BondCurve]:
    return {c: s for c, s in curves.items() if s.enabled}


def plan_daily_bond_px(
    start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
    curves: dict[str, BondCurve] = BOND_CURVES,
) -> dict[str, list]:
    """``{country: gaps}`` the bond half of the px step WOULD fetch over ``[start, end]``
    (inclusive), plus never-covered days in the GAP_LOOKBACK_DAYS before it. No network -
    also the dry run's view."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    plan = {}
    for country, spec in _enabled(curves).items():
        gaps = pb.plan_bonds_update(spec, start, end + _ONE_DAY, coverage_file=paths.daily_bonds_coverage,
                                    force_refetch=force_refetch)
        gaps += pb.plan_bonds_update(spec, start - pd.Timedelta(days=GAP_LOOKBACK_DAYS), start,
                                     coverage_file=paths.daily_bonds_coverage)
        if gaps:
            plan[country] = merge_intervals(gaps)
    return plan


def fetch_and_store_bonds(
    plan: dict[str, list], curves: dict[str, BondCurve], *, paths: CyclePaths, sources=None,
) -> tuple[int, dict[str, str]]:
    """Fetch and store each planned curve; per-curve failures are collected, not raised."""
    rows, errors = 0, {}
    for country, gaps in plan.items():
        spec = curves[country]
        try:
            fetched = pb.fetch_bonds_raw(spec, gaps, sources=sources)
            rows += pb.store_bonds_raw(spec, fetched, root=paths.daily_bonds_dir, coverage_file=paths.daily_bonds_coverage)
        except Exception as exc:
            errors[country] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            log.warning("bond fetch failed for %s: %s", country, errors[country])
    return rows, errors


def _meta(curves: dict[str, BondCurve]) -> pd.DataFrame:
    """Each tenor's curve and position on it, for the shared bad-print rule."""
    return pd.DataFrame(
        [(bond_ticker(c, t), c, t, s.currency) for c, s in curves.items() for t in s.tenors],
        columns=["ticker", "root", "order", "group"]).set_index("ticker")


def treat_bond_bad_prints(
    start: pd.Timestamp, end: pd.Timestamp, curves: dict[str, BondCurve], paths: CyclePaths,
    *, run_day: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bad prints in ``[start, end]`` on the RAW yields, treated per curve policy
    (``BondCurve.bad_print_policy``) into the adjustments log - exactly like the futures
    settlements (``infra.cycle.px.treat_bad_prints``). Returns ``(treated, pending)``."""
    run_day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if run_day is None else pd.Timestamp(run_day)
    meta = _meta(curves)
    tickers = list(meta.index)
    hist = pb.read_bonds_from_disk(tickers, start - pd.Timedelta(days=180), end + _ONE_DAY + _NEXT_SESSION_LOOKAHEAD,
                                   root=paths.daily_bonds_dir, adjusted=False)
    hist["ticker"] = hist["ticker"].astype(str)
    hist = hist.dropna(subset=["par_yield"]).rename(columns={"par_yield": "value"})
    flagged, pending = peer_outliers_frame(hist, meta, start, end, group_label="currency",
                                           min_abs_dev=BOND_MIN_ABS_DEV)
    adjustment_store.clear(paths.adjustments_dir, store=pb.STORE, source=BAD_PRINT_SOURCE,
                           keys=tickers, start=start, end=end)

    def policy(r):
        return curves[meta.at[r.ticker, "root"]].bad_print_policy, ""

    # a curve with policy "off" has its candidates LOGGED (returned with ``logged_only``), never treated
    off = flagged["ticker"].map(lambda t: curves[meta.at[t, "root"]].bad_print_policy == "off") if len(flagged) else []
    logged = flagged[off] if len(flagged) else flagged
    flagged = flagged[~off] if len(flagged) else flagged
    rows = treatment_rows(flagged, hist, policy=policy, store=pb.STORE, column="par_yield", run_day=run_day)
    treated = pd.DataFrame(rows, columns=adjustment_store.COLUMNS)
    adjustment_store.record(paths.adjustments_dir, treated)
    treated.attrs["logged_only"] = logged
    return treated, pending


def backfill_daily_bond_px(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    force_refetch: bool = False,
    fetch_missing: bool = True,
    curves: dict[str, BondCurve] = BOND_CURVES,
    sources=None,
    run_day: pd.Timestamp | None = None,
) -> dict:
    """Par yields for every enabled curve over ``[start, end]`` (inclusive): plan the gaps
    (Rule 2.1 via the coverage manifest; ``force_refetch`` re-asks the whole window),
    fetch and store them, then detect and treat bad prints."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    enabled = _enabled(curves)
    rows, errors = 0, {}
    if fetch_missing:
        plan = plan_daily_bond_px(start, end, paths=paths, force_refetch=force_refetch, curves=enabled)
        rows, errors = fetch_and_store_bonds(plan, enabled, paths=paths, sources=sources)
    treated, pending = treat_bond_bad_prints(start, end, enabled, paths, run_day=run_day)
    return {"curves": enabled, "fetch_errors": errors, "rows": rows,
            "bad_prints": treated, "pending_bad_prints": pending}


# ------------------------------------------------------------------------ checks
def _out(ctx: StepContext) -> dict:
    return ctx.output.get("bonds", {})


def _curves(ctx: StepContext) -> dict[str, BondCurve]:
    return _out(ctx).get("curves", {})


def _yields(ctx: StepContext, lookback: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame:
    tickers = list(_meta(_curves(ctx)).index)
    df = pb.read_bonds_from_disk(tickers, ctx.start - lookback, ctx.end + _ONE_DAY,
                                 root=ctx.paths.daily_bonds_dir, adjusted=False)  # judge RAW data
    df["ticker"] = df["ticker"].astype(str)
    return df.dropna(subset=["par_yield"])


def _check_fetch_ok(ctx: StepContext):
    errors = _out(ctx).get("fetch_errors", {})
    if not errors:
        return True, f"{len(_curves(ctx))} curve(s), {_out(ctx).get('rows', 0)} rows written", None
    details = pd.DataFrame({"curve": list(errors), "error": list(errors.values())})
    return False, f"{len(errors)} curve fetch error(s): {', '.join(errors)}", details


def _check_complete(ctx: StepContext):
    """Every day a curve published anything in the window has ALL its tenors."""
    df, missing = _yields(ctx), []
    for country, spec in _curves(ctx).items():
        have = df[df["ticker"].str.startswith(f"{country}_BOND_")]
        for day, grp in have.groupby("timestamp"):
            missing += [(country, t, day) for t in pb.curve_tickers(spec) if t not in set(grp["ticker"])]
    if not missing:
        return True, f"every published curve-day complete ({len(df)} yields)", None
    return False, f"{len(missing)} tenor(s) missing on days their curve published", \
        pd.DataFrame(missing, columns=["curve", "ticker", "timestamp"])


def _check_fresh(ctx: StepContext):
    """Warning: a curve's latest stored day lags the window's last weekday by more than
    STALE_BUSINESS_DAYS - a source that stopped publishing, or a holiday run."""
    df = _yields(ctx, lookback=pd.Timedelta(days=14))
    due = last_weekday(ctx.end) - pd.offsets.BDay(STALE_BUSINESS_DAYS)
    stale = []
    for country in _curves(ctx):
        days = df.loc[df["ticker"].str.startswith(f"{country}_BOND_"), "timestamp"]
        latest = days.max() if not days.empty else None
        if latest is None or latest < due:
            stale.append((country, latest))
    if not stale:
        return True, "every curve published within its expected lag", None
    text = ", ".join(f"{c} (latest {d.date() if d is not None else 'none'})" for c, d in stale)
    return False, f"stale curve(s): {text}", pd.DataFrame(stale, columns=["curve", "latest"])


def _check_sane(ctx: StepContext):
    df = _yields(ctx)
    lo, hi = YIELD_BOUNDS_PCT
    bad = df[~np.isfinite(df["par_yield"]) | (df["par_yield"] < lo) | (df["par_yield"] > hi)]
    if bad.empty:
        return True, f"{len(df)} yields finite and within [{lo}, {hi}]%", None
    return False, f"{len(bad)} yield(s) non-finite or outside [{lo}, {hi}]%", bad


def _check_outliers(ctx: StepContext):
    treated = _out(ctx).get("bad_prints", pd.DataFrame())
    if treated.empty:
        n = len(_out(ctx).get("pending_bad_prints", []))
        return True, f"no bad prints ({n} candidate(s) awaiting their next session)", None
    counts = ", ".join(f"{n} {a}" for a, n in treated["action"].value_counts().items())
    details = treated[["key", "timestamp", "action", "original", "adjusted", "detail"]].rename(columns={"key": "ticker"})
    return False, f"{len(treated)} bad print(s) treated ({counts})", details


def _check_outliers_pending(ctx: StepContext):
    pending = _out(ctx).get("pending_bad_prints", pd.DataFrame())
    if pending.empty:
        return True, "no off-peer prints awaiting judgement", None
    return False, f"{len(pending)} off-peer print(s) to be judged once their next session exists", pending


BOND_CHECKS = (
    Check("bonds_fetch_ok", _check_fetch_ok),
    Check("bonds_complete", _check_complete),
    Check("bonds_fresh", _check_fresh, severity=Severity.WARN),
    revision_check(lambda p: p.daily_bonds_dir, ["timestamp", "ticker"], name="bonds_no_revisions",
                   equals_in=lambda ctx: {"ticker": list(_meta(_curves(ctx)).index)}),
    Check("bonds_sane", _check_sane),
    Check("bonds_outliers", _check_outliers, severity=Severity.WARN),
    Check("bonds_outliers_pending", _check_outliers_pending, severity=Severity.WARN),
)
