"""Step 3 - ``backfill_daily_bmk`` = ``backfill_daily_risk`` (per risk measure, default
DV01) then ``backfill_daily_pnl`` (pnl per contract, and per unit of the prior day's
risk). Pure local computation over the cycle's stored settlements - no API.

Risk: one model per risk name in ``RISK_MODELS``. DV01 is exact for STIR futures
(price = 100 - rate, so 1bp of rate = 0.01 price points: DV01 = point_value x 0.01).
Bond futures have no DV01 yet - a proper one needs the cheapest-to-deliver bond, which
this project doesn't source - so their rows are stored with a NaN value and the reason
in ``method``, and a warning-level check lists them every run.

Pnl: ``bmk`` names the benchmark pnl definition. Futures use ``futures_price`` = change
in settlement x point value, per 1 long contract; cash bonds will add their own (total
return, yield-change based). ``pnl_per_dv01`` divides by the PRIOR day's DV01 (the risk
held over the day) - for STIR futures that is simply the rate move in bp.

Both stores upsert (keys are deterministic from the universe, unlike WIRP's
price-dependent outcome levels) - pruning whole days would also wipe the rows of roots
a partial run (e.g. ``specs={"ZQ": ...}``) didn't recompute.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from infra.config import DAILY_BACKFILL, FUTURES_ROOTS, DailyBackfillSpec, FuturesRoot
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, Step, StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.universe import UniverseMember, daily_universe
from infra.pipeline import daily as dl
from infra.storage import parquet_store

_ONE_DAY = pd.Timedelta(days=1)
_PRIOR_LOOKBACK = pd.Timedelta(days=14)  # how far back to look for a prior settlement

RISK_KEYS = ["timestamp", "ticker", "risk"]
PNL_KEYS = ["timestamp", "ticker", "bmk"]

# (member, root config) -> (value per contract, method) ; value NaN = unavailable
RiskModel = Callable[[UniverseMember, FuturesRoot], "tuple[float, str]"]


def _dv01(member: UniverseMember, cfg: FuturesRoot) -> tuple[float, str]:
    if cfg.point_value is None:
        return np.nan, "unavailable: point value not verified for this root"
    if cfg.category == "STIR":
        return cfg.point_value * 0.01, "stir_index: point_value x 0.01"
    return np.nan, "unavailable: bond futures DV01 needs the cheapest-to-deliver bond (not yet sourced)"


RISK_MODELS: dict[str, RiskModel] = {"DV01": _dv01}
PLANNED_RISKS = {"carry"}  # recognised names, deliberately not implemented yet


def _risk_dir(paths: CyclePaths):
    return paths.bmk_root / "Risk"


def _pnl_dir(paths: CyclePaths):
    return paths.bmk_root / "Pnl"


def _universe(start, end, paths, specs):
    members, errors = daily_universe(start, end, paths=paths, specs=specs, refresh_contracts=False)
    return members, errors


def _settlements(tickers: list[str], start, end, paths: CyclePaths) -> pd.DataFrame:
    df = dl.read_daily_from_disk(tickers, start, end + _ONE_DAY, root=paths.daily_futures_dir,
                                 adjustments_dir=paths.adjustments_dir)  # cleaned (adjusted) settlements
    df["ticker"] = df["ticker"].astype(str)
    return df.dropna(subset=["settlement_price"]).sort_values(["ticker", "timestamp"])


def _member_days(members: dict[str, UniverseMember], settle: pd.DataFrame, start, end) -> pd.DataFrame:
    """(ticker, timestamp) pairs where the contract is in the universe AND settled."""
    rows = []
    for t, g in settle.groupby("ticker"):
        m = members.get(t)
        if m is None:
            continue
        lo, hi = max(m.first, start), min(m.last, end)
        rows.append(g[(g["timestamp"] >= lo) & (g["timestamp"] <= hi)])
    return pd.concat(rows, ignore_index=True) if rows else settle.iloc[0:0]


def backfill_daily_risk(
    start,
    end,
    *,
    risk: str = "DV01",
    paths: CyclePaths | None = None,
    specs: dict[str, DailyBackfillSpec] = DAILY_BACKFILL,
) -> dict:
    """Risk per contract per settled universe day over ``[start, end]`` (inclusive)."""
    if risk in PLANNED_RISKS:
        raise NotImplementedError(f"risk {risk!r} is planned but not implemented yet")
    if risk not in RISK_MODELS:
        raise KeyError(f"unknown risk {risk!r}; known: {sorted(RISK_MODELS)}")
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    members, errors = _universe(start, end, paths, specs)
    days = _member_days(members, _settlements(list(members), start, end, paths), start, end)
    model = RISK_MODELS[risk]
    out = []
    for ticker, g in days.groupby("ticker"):
        m = members[ticker]
        cfg = FUTURES_ROOTS[m.root]
        value, method = model(m, cfg)
        out.append(pd.DataFrame({"timestamp": g["timestamp"].to_numpy(), "ticker": ticker,
                                 "risk": risk, "root": m.root, "value": value,
                                 "currency": cfg.currency, "method": method}))
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[*RISK_KEYS])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
        df["value"] = df["value"].astype("float64")
        parquet_store.write_partitioned(df, _risk_dir(paths), RISK_KEYS)
    return {"members": members, "universe_errors": errors, "risk": risk, "rows": len(df)}


def backfill_daily_pnl(
    start,
    end,
    *,
    risk: str = "DV01",
    paths: CyclePaths | None = None,
    specs: dict[str, DailyBackfillSpec] = DAILY_BACKFILL,
) -> dict:
    """``futures_price`` pnl per 1 long contract per settled universe day, plus pnl per
    unit of the prior day's ``risk``. A day with no earlier settlement on disk (the very
    first day of a history) gets no row rather than a guessed one."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    members, errors = _universe(start, end, paths, specs)
    settle = _settlements(list(members), start - _PRIOR_LOOKBACK, end, paths)
    settle = settle.assign(prev_timestamp=settle.groupby("ticker")["timestamp"].shift(1),
                           prev_settlement=settle.groupby("ticker")["settlement_price"].shift(1))
    days = _member_days(members, settle, start, end).dropna(subset=["prev_settlement"])
    if days.empty:
        return {"members": members, "universe_errors": errors, "rows": 0}

    root = days["ticker"].map(lambda t: members[t].root)
    cfg = root.map(FUTURES_ROOTS)
    df = pd.DataFrame({
        "timestamp": days["timestamp"].to_numpy(),
        "ticker": days["ticker"].to_numpy(),
        "bmk": "futures_price",
        "root": root.to_numpy(),
        "settlement": days["settlement_price"].to_numpy(),
        "prev_timestamp": days["prev_timestamp"].to_numpy(),
        "prev_settlement": days["prev_settlement"].to_numpy(),
        "currency": cfg.map(lambda c: c.currency).to_numpy(),
    })
    df["price_change"] = df["settlement"] - df["prev_settlement"]
    point_value = cfg.map(lambda c: np.nan if c.point_value is None else c.point_value).to_numpy()
    df["pnl"] = df["price_change"] * point_value

    risk_rows = parquet_store.read_partitioned(_risk_dir(paths), start=start - _PRIOR_LOOKBACK,
                                               end=end + _ONE_DAY, equals_in={"risk": [risk]})
    if risk_rows is not None and not risk_rows.empty:
        risk_rows = risk_rows.rename(columns={"timestamp": "prev_timestamp", "value": "prior_risk"})
        risk_rows["ticker"] = risk_rows["ticker"].astype(str)
        df = df.merge(risk_rows[["prev_timestamp", "ticker", "prior_risk"]],
                      on=["prev_timestamp", "ticker"], how="left")
    else:
        df["prior_risk"] = np.nan
    df[f"pnl_per_{risk.lower()}"] = df["pnl"] / df["prior_risk"]
    df = df.rename(columns={"prior_risk": f"prior_{risk.lower()}"})
    for col in ("timestamp", "prev_timestamp"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ms]")
    parquet_store.write_partitioned(df, _pnl_dir(paths), PNL_KEYS)
    return {"members": members, "universe_errors": errors, "rows": len(df)}


def backfill_daily_bmk(start, end, *, risks=("DV01",), paths: CyclePaths | None = None,
                       specs: dict[str, DailyBackfillSpec] = DAILY_BACKFILL) -> dict:
    """Parent: every requested risk measure first, then pnl (which reads the risk)."""
    out = {f"risk:{r}": backfill_daily_risk(start, end, risk=r, paths=paths, specs=specs) for r in risks}
    out["pnl"] = backfill_daily_pnl(start, end, risk=risks[0], paths=paths, specs=specs)
    return out


# ------------------------------------------------------------------------ checks
def _read(ctx: StepContext, store) -> pd.DataFrame:
    df = parquet_store.read_partitioned(store(ctx.paths), start=ctx.start, end=ctx.end + _ONE_DAY)
    if df is None:
        return pd.DataFrame(columns=["timestamp", "ticker"])
    df["ticker"] = df["ticker"].astype(str)
    return df


def _presence(store, what: str, needs_prior: bool = False):
    """Test (a): EVERY (contract, day) in the window where the contract was in the
    universe and settled has a row - not just the last day, which under the datasets'
    publication lag (see api.available_end) may have no settlements yet and would then
    check nothing. For pnl a row also needs an earlier settlement to diff against."""

    def fn(ctx: StepContext):
        members = ctx.output.get("members", {})
        settled = _settlements(list(members), ctx.start - _PRIOR_LOOKBACK, ctx.end, ctx.paths)
        if needs_prior:
            settled = settled[settled.groupby("ticker")["timestamp"].shift(1).notna()]
        settled = settled[settled["timestamp"] >= ctx.start]
        expected = {(t, d) for t, d in zip(settled["ticker"], settled["timestamp"])
                    if members[t].expected_on(d)}
        rows = _read(ctx, store)
        missing = sorted(expected - set(zip(rows["ticker"], rows["timestamp"])))
        if not missing:
            return True, f"{what} present for all {len(expected)} settled contract-days", None
        return (False, f"{len(missing)} settled contract-day(s) missing {what}",
                pd.DataFrame(missing, columns=["ticker", "timestamp"]))

    return fn


def _check_dv01_coverage(ctx: StepContext):
    """Warning, every run: which roots have no DV01 (so no pnl-per-DV01) and why."""
    df = _read(ctx, _risk_dir)
    missing = df[df["value"].isna()]
    if missing.empty:
        return True, "DV01 available for every contract", None
    by_root = missing.groupby("root")["method"].first().reset_index()
    return False, f"no DV01 for {', '.join(sorted(by_root['root']))}", by_root


def _check_risk_sane(ctx: StepContext):
    df = _read(ctx, _risk_dir)
    bad = df[df["value"].notna() & ~(df["value"] > 0)]
    return (bad.empty, "every available DV01 positive" if bad.empty else f"{len(bad)} non-positive DV01",
            None if bad.empty else bad)


def _check_pnl_consistent(ctx: StepContext):
    """(c): pnl == price change x the root's point value, exactly - catches a wrong spec
    or a unit slip the moment it happens."""
    df = _read(ctx, _pnl_dir)
    if df.empty:
        return True, "no pnl rows in window", None
    pv = df["root"].map(lambda r: FUTURES_ROOTS[r].point_value).astype("float64")
    bad = df[pv.notna() & ~np.isclose(df["pnl"], df["price_change"] * pv, rtol=0, atol=1e-6)]
    if bad.empty:
        return True, f"{len(df)} pnl rows consistent with their point values", None
    return False, f"{len(bad)} pnl row(s) inconsistent with point value", bad


def _run_risk(ctx: StepContext) -> dict:
    return backfill_daily_risk(ctx.start, ctx.end, paths=ctx.paths,
                               risk=ctx.options.get("risk", "DV01"),
                               **{k: ctx.options[k] for k in ("specs",) if k in ctx.options})


def _run_pnl(ctx: StepContext) -> dict:
    return backfill_daily_pnl(ctx.start, ctx.end, paths=ctx.paths,
                              risk=ctx.options.get("risk", "DV01"),
                              **{k: ctx.options[k] for k in ("specs",) if k in ctx.options})


RISK_STEP = Step("bmk_risk", _run_risk, depends_on=("px",), checks=(
    Check("risk_present", _presence(_risk_dir, "risk")),
    revision_check(_risk_dir, RISK_KEYS, name="risk_no_revisions"),
    Check("risk_sane", _check_risk_sane),
    Check("dv01_coverage", _check_dv01_coverage, severity=Severity.WARN),
))

PNL_STEP = Step("bmk_pnl", _run_pnl, depends_on=("px", "bmk_risk"), checks=(
    Check("pnl_present", _presence(_pnl_dir, "pnl", needs_prior=True)),
    revision_check(_pnl_dir, PNL_KEYS, name="pnl_no_revisions", atol=1e-9),
    Check("pnl_consistent", _check_pnl_consistent),
))
