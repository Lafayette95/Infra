"""Near-the-money Treasury futures options for implied vol - Rule 2.3 end to end, reusing
the existing option machinery (``infra.pipeline.options.load_definitions`` for the cached
definition snapshots, ``infra.pipeline.daily_options`` for the settlement store, its
coverage and its fetch). New here: the snapshot grid, the per-day near-the-money
selection (``infra.processing.option_selection``) and a dry-run.

Spec per root: ``infra.config.FUTURES_OPTIONS_IV`` (ZN only for now). Daily settlements
(the ``statistics`` schema) are stored in ``Daily/Options`` like the SR3 options.

    plan = dry_run("ZN", "2019-01-02", "2026-10-01")   # free: prices everything, fetches nothing
    sel = load_iv_options("ZN", start, end)             # paid: definitions, then settlements
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.api import databento_client as api
from infra.config import (DAILY_OPTIONS_COVERAGE_FILE, DAILY_OPTIONS_DIR, DEFINITIONS_DIR, FUTURES_OPTIONS_IV,
                          MAX_COST_USD, OPTIONS_UNIVERSE, FuturesOptionsIVSpec)
from infra.pipeline import daily as dl
from infra.pipeline import daily_options as dopt
from infra.pipeline.options import _definitions_path, load_definitions
from infra.processing.option_selection import fetch_ranges, select_near_the_money

_GRID_EPOCH = pd.Timestamp("2000-01-03")


def snapshot_days(start, end, step_days: int, *, today=None) -> list[pd.Timestamp]:
    """Grid days (fixed epoch, every ``step_days``) from the one at/before ``start`` to the
    first after ``end`` - so every day has a snapshot on each side; a grid day not yet
    queryable is replaced by the day before yesterday (a snapshot can't be from the
    future, and on the current licence GLBX's queryable end runs ~8h behind now, so
    yesterday's whole day isn't there yet - found 2026-10-03, a 422 on the last snapshot)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    last_ok = (pd.Timestamp.now(tz="UTC").tz_localize(None) if today is None else pd.Timestamp(today)).normalize() \
        - pd.Timedelta(days=2)
    first = _GRID_EPOCH + pd.Timedelta(days=((start - _GRID_EPOCH).days // step_days) * step_days)
    days = list(pd.date_range(first, end + pd.Timedelta(days=step_days), freq=f"{step_days}D"))
    return sorted({min(d, last_ok) for d in days})


def definitions(spec: FuturesOptionsIVSpec, start, end, *, fetch_missing: bool = False,
                directory: Path = DEFINITIONS_DIR, client=None) -> pd.DataFrame:
    """Union of the grid snapshots covering ``[start, end]`` (cache-first; paid only with
    ``fetch_missing``)."""
    frames = [load_definitions(spec.parent, d, fetch_missing=fetch_missing, directory=directory, client=client)
              for d in snapshot_days(start, end, spec.snapshot_days)]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True).drop_duplicates("instrument_id", keep="last") if frames else \
        pd.DataFrame(columns=["instrument_id", "underlying", "option_type", "strike", "expiry"])


def underlying_settlements(tickers, start, end) -> pd.DataFrame:
    df = dl.read_daily_from_disk(sorted(set(map(str, tickers))), pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1))
    df["ticker"] = df["ticker"].astype(str)
    return df.dropna(subset=["settlement_price"]).rename(columns={"settlement_price": "price"})[["timestamp", "ticker", "price"]]


def select(spec: FuturesOptionsIVSpec, start, end, *, defs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per day, the selected options (disk only)."""
    defs = definitions(spec, start, end) if defs is None else defs
    if defs.empty:
        return select_near_the_money(defs, pd.DataFrame())
    px = underlying_settlements(defs["underlying"].unique(), start, end)
    return select_near_the_money(defs, px, n_expiries=spec.n_expiries, n_strikes=spec.n_strikes,
                                 min_days=spec.min_days)


def _plan(selection: pd.DataFrame, coverage_file: Path, bucket: str = "Q") -> dict:
    """{(start, end): [ids]} of what's not on disk yet. Instruments are grouped by the
    ``bucket`` period their range starts in, and each group is requested over the group's
    whole span: one request per instrument range was cheapest in data but made ~1,200
    requests at ~10s each for ZN 2019-2026 (found 2026-10-03); per quarter it's ~35, for a
    little more data (settlements cost ~$0.00001 per instrument-month). ``bucket=None``
    = each instrument over its own range."""
    plan = {}
    r = fetch_ranges(selection)
    if r.empty:
        return plan
    if bucket is not None:
        key = pd.to_datetime(r["start"]).dt.to_period(bucket)
        r["start"] = r.groupby(key)["start"].transform("min")
        r["end"] = r.groupby(key)["end"].transform("max")
    for (start, end), g in r.groupby(["start", "end"]):
        ids = [int(x) for x in g["instrument_id"]]
        for gap, todo in dopt.plan_daily_options_update(ids, start, end + pd.Timedelta(days=1),
                                                        coverage_file=coverage_file).items():
            plan.setdefault(gap, []).extend(todo)
    return plan


def dry_run(root: str, start, end, *, client=None, directory: Path = DEFINITIONS_DIR,
            coverage_file: Path = DAILY_OPTIONS_COVERAGE_FILE, stats_sample_month=("2026-09-01", "2026-10-01"),
            assumed_instruments_per_expiry: int = 60) -> dict:
    """Price the whole load without downloading anything (free metadata calls only).

    Definitions: exact, one ``get_cost`` per snapshot not yet cached. Settlements: exact
    when every snapshot is cached (the selection is known, priced per instrument range);
    otherwise an UPPER BOUND - the parent's measured cost per instrument-month over
    ``stats_sample_month``, x ``assumed_instruments_per_expiry`` per expiry over its last
    3 months (a near-the-money band touches far fewer: 6 per expiry per day, moving with
    the futures)."""
    spec = FUTURES_OPTIONS_IV[root]
    dataset = OPTIONS_UNIVERSE[spec.parent]
    client = client or api.get_client()
    days = snapshot_days(start, end, spec.snapshot_days)
    missing = [d for d in days if not _definitions_path(spec.parent, d, directory).exists()]
    def_costs = {d: api.estimate_cost(dataset, "definition", [spec.parent], d, d + pd.Timedelta(days=1), "parent",
                                      client=client) for d in missing}
    out = {"root": root, "parent": spec.parent, "snapshots": len(days), "snapshots_to_fetch": len(missing),
           "definitions_usd": float(sum(def_costs.values()))}
    if not missing:
        plan = _plan(select(spec, start, end, defs=definitions(spec, start, end, directory=directory)), coverage_file)
        out["statistics_usd"] = float(sum(api.estimate_cost(dataset, "statistics", ids, a, b, "instrument_id",
                                                            client=client) for (a, b), ids in plan.items()))
        out["statistics_basis"] = f"exact: {sum(len(v) for v in plan.values())} instrument range(s)"
    else:
        a, b = (pd.Timestamp(x) for x in stats_sample_month)
        month_cost = api.estimate_cost(dataset, "statistics", [spec.parent], a, b, "parent", client=client)
        res = client.symbology.resolve(dataset=dataset, symbols=[spec.parent], stype_in="parent",
                                       stype_out="instrument_id", start_date=f"{a:%Y-%m-%d}",
                                       end_date=f"{a + pd.Timedelta(days=1):%Y-%m-%d}")  # one day: a month
        # times out (504); fewer instruments -> a higher per-instrument rate, so still a bound
        n_inst = max(sum(len(v) for v in res.get("result", {}).values()), 1)
        months = (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.4
        expiries = months * spec.n_expiries / 1.0  # monthly expiries (quarterly + serial), n live at a time
        inst_months = expiries * assumed_instruments_per_expiry * 3
        out["statistics_usd"] = float(month_cost / n_inst * inst_months)
        out["statistics_basis"] = (f"upper bound: ${month_cost:.3f} for {n_inst} instruments in one month "
                                   f"-> {inst_months:,.0f} instrument-months assumed")
    out["total_usd"] = out["definitions_usd"] + out["statistics_usd"]
    return out


def load_iv_options(root: str, start, end, *, client=None, max_cost_usd: float = MAX_COST_USD,
                    directory: Path = DEFINITIONS_DIR, store: Path = DAILY_OPTIONS_DIR,
                    coverage_file: Path = DAILY_OPTIONS_COVERAGE_FILE) -> pd.DataFrame:
    """PAID: fetch missing snapshots, select, fetch the selected options' settlements not on
    disk yet (Rule 2.1), and return the selection."""
    spec = FUTURES_OPTIONS_IV[root]
    defs = definitions(spec, start, end, fetch_missing=True, directory=directory, client=client)
    sel = select(spec, start, end, defs=defs)
    plan = _plan(sel, coverage_file)
    if plan:
        dopt.fetch_and_store_daily_options(plan, defs, dataset=OPTIONS_UNIVERSE[spec.parent], root=store,
                                           coverage_file=coverage_file, max_cost_usd=max_cost_usd, client=client)
    return sel
