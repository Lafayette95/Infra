"""Step 2 - ``backfill_daily_derived``: metrics computed purely from the cycle's own
stored data (no API). Unlike the dashboard's on-demand views, derived values ARE
persisted here, because the revision check (b) needs yesterday's values on disk.

Each metric is a ``DerivedMetric`` in ``DERIVED_METRICS``; adding one means adding an
entry (compute function, store, keys, custom checks) - the step, its presence check and
its revision check are generated from the registry.

Every recomputed day REPLACES that day's rows wholesale (prune on ``timestamp``) rather
than upserting: a day whose output shrinks (e.g. a meeting going from two outcome levels
to one) would otherwise keep a stale row from the previous run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from infra.cycle.checks import revision_check
from infra.pipeline.treasury_curves import compute_curves
from infra.cycle.core import Check, Severity, Step, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline.wirp import available_days, build_schedule
from infra.storage import parquet_store

# (rows with a "timestamp" column = the as-of day, {day: why no rows}) over [start, end]
ComputeFn = Callable[[pd.Timestamp, pd.Timestamp, CyclePaths], "tuple[pd.DataFrame, dict]"]


@dataclass(frozen=True)
class ExtraStore:
    """A second (third, ...) store one metric's computation fills: ``compute`` returns its
    frame in ``diag["extra"][name]``; it gets its own replace-the-days write and its own
    revision check. (The Treasury curve fills TreasuryCurves AND TreasuryRV in one fit.)"""
    name: str
    store: Callable[[CyclePaths], Path]
    key_columns: tuple[str, ...]


@dataclass(frozen=True)
class DerivedMetric:
    name: str
    store: Callable[[CyclePaths], Path]
    key_columns: tuple[str, ...]
    compute: ComputeFn
    checks: tuple[Check, ...] = ()  # custom checks (c); presence/revision are generated
    extra_stores: tuple[ExtraStore, ...] = ()
    # how a run replaces its rows (default: delete the window's days by ``timestamp``, write):
    # (store, df, diag, start, end) -> None - for stores keyed by an instant, or shared with
    # rows the cycle doesn't own (the swap closes' hand-built adjusted rows)
    replace: Callable | None = None
    # the generated presence check's severity: WARN where a day can legitimately lack the
    # inputs (a thin DTCC day - a half-day before a holiday prints one swap tenor)
    presence_severity: Severity = Severity.FAIL


# ----------------------------------------------------------------------------- WIRP
WIRP_KEYS = ("timestamp", "meeting_date", "outcome_step")
WIRP_MAX_ABS_MOVE_BPS = 100.0  # a single scheduled meeting beyond this is a bug, not a market


def compute_wirp(start: pd.Timestamp, end: pd.Timestamp, paths: CyclePaths) -> tuple[pd.DataFrame, dict]:
    """WIRP (CLOSE, i.e. settlement-based) for every day in ``[start, end]`` that has ZQ
    settlements on disk - the exact function the dashboard uses, as of each day."""
    kw = dict(contracts_file=paths.contracts_file, close_root=paths.daily_futures_dir)
    days = [d for d in available_days("close", **kw) if start <= d <= end]
    kw["adjustments_dir"] = paths.adjustments_dir  # WIRP reads cleaned settlements
    frames, empty = [], {}
    for day in days:
        schedule, meta = build_schedule("close", today=day, **kw)
        if schedule.empty:
            empty[day] = meta["status"]
            continue
        frames.append(schedule.assign(timestamp=day, anchor_month=meta["anchor_month"],
                                      anchor_rate=meta["anchor_rate"]))
    if not frames:
        return pd.DataFrame(columns=[*WIRP_KEYS]), {"days": days, "empty_days": empty}
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
    df["meeting_date"] = pd.to_datetime(df["meeting_date"]).astype("datetime64[ms]")
    df["outcome_step"] = df["outcome_step"].astype("int32")
    return df, {"days": days, "empty_days": empty}


def _wirp_rows(ctx: StepContext) -> pd.DataFrame:
    df = parquet_store.read_partitioned(ctx.paths.wirp_dir, start=ctx.start,
                                        end=ctx.end + pd.Timedelta(days=1))
    return pd.DataFrame() if df is None else df


def _check_wirp_probabilities(ctx: StepContext):
    df = _wirp_rows(ctx)
    if df.empty:
        return True, "no rows in window", None
    in_range = df["probability"].between(0.0, 1.0)
    sums = df.groupby(["timestamp", "meeting_date"])["probability"].sum()
    bad_sum = sums[~np.isclose(sums, 1.0, atol=1e-9)]
    if in_range.all() and bad_sum.empty:
        return True, f"{sums.size} meeting distributions, all in [0,1] and summing to 1", None
    details = pd.concat([df.loc[~in_range, ["timestamp", "meeting_date", "probability"]],
                         bad_sum.rename("probability").reset_index()], ignore_index=True)
    return False, f"{(~in_range).sum()} out-of-range probability(ies), {len(bad_sum)} bad sum(s)", details


def _check_wirp_moves(ctx: StepContext):
    df = _wirp_rows(ctx)
    if df.empty:
        return True, "no rows in window", None
    bad = df.loc[df["change_bps"].abs() > WIRP_MAX_ABS_MOVE_BPS,
                 ["timestamp", "meeting_date", "method", "change_bps"]].drop_duplicates()
    if bad.empty:
        return True, f"every implied move within +/-{WIRP_MAX_ABS_MOVE_BPS:g}bp", None
    return False, f"{len(bad)} implausible implied move(s)", bad


WIRP = DerivedMetric(
    "wirp", lambda p: p.wirp_dir, WIRP_KEYS, compute_wirp,
    checks=(Check("wirp_probabilities_valid", _check_wirp_probabilities),
            Check("wirp_moves_plausible", _check_wirp_moves)),
)

# --------------------------------------------------------------- our Treasury curve
CURVE_KEYS = ("timestamp", "method")
RV_KEYS = ("timestamp", "cusip", "method")


def compute_treasury_curve(start: pd.Timestamp, end: pd.Timestamp, paths: CyclePaths) -> tuple[pd.DataFrame, dict]:
    """Our Treasury zero curve (spline + Svensson) and every bond's z-spread / carry /
    rolldown for each day in ``[start, end]`` with FedInvest END OF DAY prices -
    ``infra.pipeline.treasury_curves.compute_curves`` (root CLAUDE.md 25), all inputs read
    through ``paths``. Svensson seeds from the STORED fit of the day before the window, so
    this window equals a full sequential rebuild exactly (the revision check's atol 1e-10)."""
    cdf, rdf, diag = compute_curves(start, end, prices_root=paths.treasury_prices_dir,
                                    securities_root=paths.treasury_securities_dir, otr_root=paths.treasury_otr_dir,
                                    auctions_root=paths.tsy_auctions_dir, curves_root=paths.treasury_curves_dir)
    return cdf, {**diag, "extra": {"treasury_rv": rdf}}


def _curve_rows(ctx: StepContext) -> pd.DataFrame:
    df = parquet_store.read_partitioned(ctx.paths.treasury_curves_dir, start=ctx.start, end=ctx.end + pd.Timedelta(days=1))
    return pd.DataFrame() if df is None else df


def _check_curve_fit(ctx: StepContext):
    """(c) each fitted day's fit error is sane (median 1-3bp since 2010; 13bp at worst in 2008)
    and its par curve within [-2, 25]%."""
    df = _curve_rows(ctx)
    if df.empty:
        return True, "no curve rows in window", None
    par = df[[c for c in df.columns if c.startswith("par_")]]
    bad = df[(df["rmse_bp"] > 25) | ~par.apply(lambda s: s.between(-2, 25)).all(axis=1)]
    if bad.empty:
        return True, f"{len(df)} curve fits, fit error max {df['rmse_bp'].max():.2f}bp", None
    return False, f"{len(bad)} implausible curve fit(s)", bad[["timestamp", "method", "n_fit", "rmse_bp"]]


TREASURY_CURVE = DerivedMetric(
    "treasury_curve", lambda p: p.treasury_curves_dir, CURVE_KEYS, compute_treasury_curve,
    checks=(Check("treasury_curve_fit_sane", _check_curve_fit),),
    extra_stores=(ExtraStore("treasury_rv", lambda p: p.treasury_rv_dir, RV_KEYS),),
)

# ----------------------------------------------------------- swap closes (PURE method)
SWAP_CLOSE_KEYS = ("timestamp", "close", "currency", "tenor", "method")
SWAP_RATE_BOUNDS_PCT = (-1.0, 15.0)


def compute_swap_closes(start: pd.Timestamp, end: pd.Timestamp, paths: CyclePaths) -> tuple[pd.DataFrame, dict]:
    """The PURE benchmark swap closes (root CLAUDE.md 16) for every day with an archived DTCC
    file in ``[start - SWAP_CORRECTION_DAYS, end]``: a day's close reads the corrections in
    the files after it, so each run recomputes the last ``SWAP_CORRECTION_DAYS`` and a late
    correction lands (95% of cancellations arrive within a day, 99% within 33). Pure only:
    the futures-adjusted method needs intraday quotes and stays a hand build (TOFIX)."""
    from infra.config import SWAP_CORRECTION_DAYS
    from infra.pipeline import dtcc
    from infra.pipeline.swap_closes import REPORT, compute_closes
    lo = start - pd.Timedelta(days=SWAP_CORRECTION_DAYS)
    have = dtcc.archived_days(REPORT, root=paths.dtcc_dir)
    days = [d for d in pd.date_range(lo, end) if d in have]
    from infra.analytics.sofr_curve import business_days
    bdays = set(business_days(lo, end))  # federal holidays out: a holiday file has few or no prints
    frames, empty, cache = [], {}, {}
    for d in days:
        for k in [k for k in cache if k < d]:
            del cache[k]
        c = compute_closes(d, dtcc_root=paths.dtcc_dir, cache=cache, methods=("pure",))
        if c.empty and d in bdays:
            empty[d] = "no plain par-swap print in any close window that day"
        frames.append(c)
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) \
        else pd.DataFrame(columns=list(SWAP_CLOSE_KEYS))
    if not df.empty:
        df = df.astype({"timestamp": "datetime64[ms]", "tenor": "int32", "n_trades": "int32", "half_window_min": "int32"})
    return df, {"days": [d for d in days if d in bdays], "empty_days": empty, "range": (lo, end)}


def _replace_pure_closes(store, df: pd.DataFrame, diag: dict, start, end) -> None:
    """Delete the recomputed range's PURE rows only (the hand-built adjusted rows stay),
    by the snap INSTANT (closes are keyed by it, not by midnight), then write."""
    lo, hi = diag.get("range", (start, end))
    hi = pd.Timestamp(hi) + pd.Timedelta(days=1)
    parquet_store.delete_where(store, lambda part: pd.to_datetime(part["timestamp"]).ge(pd.Timestamp(lo))
                               & pd.to_datetime(part["timestamp"]).lt(hi) & part["method"].astype(str).eq("pure"))
    if not df.empty:
        parquet_store.write_partitioned(df, store, list(SWAP_CLOSE_KEYS))


def _check_swap_closes(ctx: StepContext):
    """(c) every pure close in the window: rate in bounds, >= 1 trade, finite standard error."""
    df = parquet_store.read_partitioned(ctx.paths.swap_closes_dir, start=ctx.start, end=ctx.end + pd.Timedelta(days=1))
    if df is None or df.empty:
        return True, "no swap closes in window", None
    df = df[df["method"].astype(str) == "pure"]
    lo, hi = SWAP_RATE_BOUNDS_PCT
    bad = df[~df["rate"].between(lo, hi) | (df["n_trades"] < 1) | ~np.isfinite(df["se_bp"].astype(float))]
    if bad.empty:
        return True, f"{len(df)} pure closes in window, all sane", None
    return False, f"{len(bad)} implausible swap close(s)", bad[["timestamp", "close", "currency", "tenor", "rate", "n_trades", "se_bp"]]


SWAP_CLOSES_PURE = DerivedMetric(
    "swap_closes", lambda p: p.swap_closes_dir, SWAP_CLOSE_KEYS, compute_swap_closes,
    checks=(Check("swap_closes_sane", _check_swap_closes),), replace=_replace_pure_closes,
    presence_severity=Severity.WARN,  # like every DTCC check: a thin day must not cost the vintage
)

# ------------------------------------------------- OIS (SOFR) curves from the swap closes
OIS_CURVE_KEYS = ("timestamp", "curve", "node")
OIS_FORWARD_BOUNDS_PCT = (-1.0, 15.0)
OIS_REPRICE_TOL_BP = 0.01


def compute_ois_curve(start: pd.Timestamp, end: pd.Timestamp, paths: CyclePaths) -> tuple[pd.DataFrame, dict]:
    """Every ``OIS_CURVES`` curve over the swap closes' recomputed range (their last
    ``SWAP_CORRECTION_DAYS``: a corrected close moves its curve too). Runs AFTER
    ``swap_closes`` (registry order)."""
    from infra.config import SWAP_CORRECTION_DAYS
    from infra.pipeline.ois_curves import compute_ois_curves
    lo = start - pd.Timedelta(days=SWAP_CORRECTION_DAYS)
    df, diag = compute_ois_curves(lo, end, swap_closes_root=paths.swap_closes_dir, futures_root=paths.daily_futures_dir,
                                  contracts_file=paths.contracts_file, repo_root=paths.repo_dir)
    return df, {**diag, "range": (lo, end)}


def _replace_ois_curves(store, df: pd.DataFrame, diag: dict, start, end) -> None:
    from infra.pipeline.ois_curves import store_ois_curves
    lo, hi = diag.get("range", (start, end))
    store_ois_curves(df, lo, hi, root=store)


def _check_ois_curve(ctx: StepContext):
    """(c) every stored curve in the window: each swap pillar reprices its input close
    (within ``OIS_REPRICE_TOL_BP``), and every forward between nodes lies in bounds."""
    from infra.analytics import swap_curve as sc
    from infra.config import OIS_CURVES, SWAP_CURVES
    from infra.pipeline.ois_curves import read_ois_curves
    df = read_ois_curves(ctx.start, ctx.end + pd.Timedelta(days=1), root=ctx.paths.ois_curves_dir)
    if df.empty:
        return True, "no OIS curves in window", None
    bad, worst = [], 0.0
    for (ts, name), g in df.groupby(["timestamp", "curve"]):
        c, day = sc.curve_from_nodes(g), pd.Timestamp(ts).normalize()
        lag = SWAP_CURVES[OIS_CURVES[name].currency].spot_lag_days
        for r in g[g["source"] != "short_end"].itertuples():
            err = abs(sc.par_rate(c, day, sc.swap_schedule(day, int(r.node[:-1]), lag)) - r.input_rate) * 100.0
            worst = max(worst, err)
            if err > OIS_REPRICE_TOL_BP:
                bad.append({"timestamp": ts, "curve": name, "node": r.node, "issue": f"reprices {err:.3f}bp off"})
        t = np.concatenate([[1e-9], g.sort_values("t_years")["t_years"].to_numpy()])
        f = c.forward(t[:-1], t[1:])
        lo, hi = OIS_FORWARD_BOUNDS_PCT
        for k in np.where((f < lo) | (f > hi))[0]:
            bad.append({"timestamp": ts, "curve": name, "node": g.sort_values("t_years")["node"].iloc[k],
                        "issue": f"forward {f[k]:.2f}% out of bounds"})
    n = df.groupby(["timestamp", "curve"]).ngroups
    if not bad:
        return True, f"{n} curve(s), max repricing error {worst:.1e}bp, forwards in bounds", None
    return False, f"{len(bad)} problem(s) in {n} curve(s)", pd.DataFrame(bad)


OIS_CURVE = DerivedMetric(
    "ois_curve", lambda p: p.ois_curves_dir, OIS_CURVE_KEYS, compute_ois_curve,
    checks=(Check("ois_curve_sane", _check_ois_curve),), replace=_replace_ois_curves,
    presence_severity=Severity.WARN,  # too few tenors on a thin day (2025-06-18, 2026-01-02, 2026-07-02)
)

# ------------------------------------------------------ swap spreads (CMT and on-the-run ASW)
SWAP_SPREAD_KEYS = ("timestamp", "ticker", "source", "cusip")


def compute_swap_spreads(start: pd.Timestamp, end: pd.Timestamp, paths: CyclePaths) -> tuple[pd.DataFrame, dict]:
    """Both sources over the OIS curve's recomputed range (a revised curve moves its
    spreads). Runs AFTER ``ois_curve``; the on-the-run source trails CMT by a day (FedInvest
    posts END OF DAY ~10:00 New York on D+1) and fills in next run."""
    from infra.config import SWAP_CORRECTION_DAYS
    from infra.pipeline.swap_spreads import compute_swap_spreads as compute
    lo = start - pd.Timedelta(days=SWAP_CORRECTION_DAYS)
    df, diag = compute(lo, end, ois_root=paths.ois_curves_dir, bonds_root=paths.daily_bonds_dir,
                       otr_root=paths.treasury_otr_dir, securities_root=paths.treasury_securities_dir,
                       prices_root=paths.treasury_prices_dir)
    return df, {**diag, "range": (lo, end)}


def _replace_swap_spreads(store, df: pd.DataFrame, diag: dict, start, end) -> None:
    from infra.pipeline.swap_spreads import store_swap_spreads
    lo, hi = diag.get("range", (start, end))
    store_swap_spreads(df, lo, hi, root=store)


SWAP_SPREADS = DerivedMetric(
    "swap_spreads", lambda p: p.swap_spreads_dir, SWAP_SPREAD_KEYS, compute_swap_spreads,
    replace=_replace_swap_spreads, presence_severity=Severity.WARN,
)

# order matters: ois_curve reads what swap_closes just wrote, swap_spreads what ois_curve wrote
DERIVED_METRICS: dict[str, DerivedMetric] = {"wirp": WIRP, "treasury_curve": TREASURY_CURVE,
                                             "swap_closes": SWAP_CLOSES_PURE, "ois_curve": OIS_CURVE,
                                             "swap_spreads": SWAP_SPREADS}


# ------------------------------------------------------------------------------ step
def backfill_daily_derived(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    metrics: dict[str, DerivedMetric] | None = None,
) -> dict:
    """Recompute every registered metric over ``[start, end]`` (inclusive) and replace
    those days in its store. Returns per-metric diagnostics for the checks."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    metrics = DERIVED_METRICS if metrics is None else metrics
    out = {}
    for name, metric in metrics.items():
        df, diag = metric.compute(start, end, paths)
        store = metric.store(paths)
        if metric.replace is not None:
            metric.replace(store, df, diag, start, end)
        else:
            parquet_store.prune_rows(store, "timestamp", pd.date_range(start, end, freq="D"))
            if not df.empty:
                parquet_store.write_partitioned(df, store, list(metric.key_columns))
        for extra in metric.extra_stores:
            edf = diag.get("extra", {}).get(extra.name, pd.DataFrame())
            estore = extra.store(paths)
            parquet_store.prune_rows(estore, "timestamp", pd.date_range(start, end, freq="D"))
            if not edf.empty:
                parquet_store.write_partitioned(edf, estore, list(extra.key_columns))
        diag = {k: v for k, v in diag.items() if k != "extra"}
        out[name] = {**diag, "rows": len(df)}
    return out


def _presence_check(metric: DerivedMetric) -> Check:
    """Test (a): every day in the window that HAS input data produced output."""

    def fn(ctx: StepContext):
        diag = ctx.output.get(metric.name, {})
        empty = diag.get("empty_days", {})
        if not empty:
            return True, f"{len(diag.get('days', []))} day(s) computed, {diag.get('rows', 0)} rows", None
        details = pd.DataFrame({"timestamp": list(empty), "reason": list(empty.values())})
        return False, f"{len(empty)} day(s) with input data produced no {metric.name}", details

    return Check(f"{metric.name}_present", fn, metric.presence_severity)


def derived_checks(metrics: dict[str, DerivedMetric]) -> tuple[Check, ...]:
    checks: list[Check] = []
    for m in metrics.values():
        checks += [_presence_check(m),
                   revision_check(m.store, list(m.key_columns), name=f"{m.name}_no_revisions",
                                  rtol=0.0, atol=1e-10),
                   *[revision_check(e.store, list(e.key_columns), name=f"{e.name}_no_revisions",
                                    rtol=0.0, atol=1e-10) for e in m.extra_stores],
                   *m.checks]
    return tuple(checks)


def _run(ctx: StepContext) -> dict:
    return backfill_daily_derived(ctx.start, ctx.end, paths=ctx.paths,
                                  metrics=ctx.options.get("derived_metrics"))


DERIVED_STEP = Step("derived", _run, depends_on=("px", "raw"), checks=derived_checks(DERIVED_METRICS))
