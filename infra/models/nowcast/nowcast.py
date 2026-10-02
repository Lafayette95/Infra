"""Parent API: the only module here that reads storage (through infra.pipeline.releases,
never the network - populate data with the daily cycle's ``raw`` step or
``scripts/update_releases.py``). Everything below it is pure.

    model = estimate("c", as_of="2026-06-30")         # versions a-d, spec.py
    nowcast(model, as_of="2026-09-30")                  # current quarter, GDP units
    news(model, "2026-09-23", "2026-09-30")             # what moved it, print by print

Point-in-time throughout (CLAUDE.md 3): ``as_of`` reads only vintages published by the
end of that day, so a backfill loop over days gives exactly what would have been shown
live on each of them. ``as_of=None`` = everything on disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from infra.config import MACRO_RELEASES, RELEASES_DIR, MacroRelease
from infra.models.nowcast import kalman
from infra.models.nowcast.dfm import DFM, fit
from infra.models.nowcast.news import (News, decompose, factor_contributions, prospective, quarter_month,
                                       target_value)
from infra.models.nowcast.panel import Panel, build_panel
from infra.models.nowcast.spec import VERSIONS, ModelSpec, factor_structure
from infra.pipeline.releases import read_releases_from_disk, series_to_fetch


def load_raw(as_of=None, *, releases: dict[str, MacroRelease] = MACRO_RELEASES, root: Path = RELEASES_DIR):
    return read_releases_from_disk(sorted(series_to_fetch(releases)), as_of, root=root)


def _spec(version: str | ModelSpec, overrides: dict) -> ModelSpec:
    spec = VERSIONS[version] if isinstance(version, str) else version
    return spec.with_(**overrides) if overrides else spec


def estimate(version: str | ModelSpec = "c", as_of=None, *, raw: pd.DataFrame | None = None,
             releases: dict[str, MacroRelease] = MACRO_RELEASES, root: Path = RELEASES_DIR, **overrides) -> DFM:
    """Fit a version (a-d, or a full ``ModelSpec``) on the data as published by ``as_of``.
    ``overrides`` tweak the spec (``tau=0.05``, ``global_factor=True``, ...)."""
    spec = _spec(version, overrides)
    raw = load_raw(as_of, releases=releases, root=root) if raw is None else raw
    panel = build_panel(raw, releases, spec, as_of)
    structure = factor_structure(spec, releases, list(panel.data.columns))
    return fit(panel.data, panel.frequency, structure, spec, as_of=as_of)


def model_panel(model: DFM, raw: pd.DataFrame, as_of, *, end=None,
                releases: dict[str, MacroRelease] = MACRO_RELEASES) -> Panel:
    """The panel as of ``as_of``, restricted/aligned to the model's own series."""
    panel = build_panel(raw, releases, model.spec, as_of, end=end)
    for name in ("data", "native", "published"):
        setattr(panel, name, getattr(panel, name).reindex(columns=list(model.series)))
    return panel


@dataclass
class Nowcast:
    quarter: pd.Period
    as_of: pd.Timestamp | None
    value: float  # GDP QoQ % SAAR - the published figure once there is one
    common: pd.Series  # the model's own estimate split by factor (+ mean), GDP pp
    published: bool  # GDP for the quarter already out


def _grid_end(quarter: pd.Period, as_of) -> pd.Timestamp:
    """Last month of the panel: the later of ``quarter``'s and ``as_of``'s quarter ends
    (a backcast still uses everything published since)."""
    later = max(quarter, _default_quarter(as_of))
    return later.asfreq("M", how="end").to_timestamp()


def _default_quarter(as_of) -> pd.Period:
    return pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now()).to_period("Q")


def nowcast(model: DFM, as_of=None, quarter=None, *, raw: pd.DataFrame | None = None,
            releases: dict[str, MacroRelease] = MACRO_RELEASES, root: Path = RELEASES_DIR) -> Nowcast:
    """GDP for ``quarter`` (default: ``as_of``'s) as of ``as_of``."""
    quarter = pd.Period(quarter, freq="Q") if quarter is not None else _default_quarter(as_of)
    raw = load_raw(as_of, releases=releases, root=root) if raw is None else raw
    panel = model_panel(model, raw, as_of, end=_grid_end(quarter, as_of), releases=releases)
    X = model.standardize(panel.data)
    sm = kalman.smooth(model.state_space(), X, lag=False)
    t_q = quarter_month(panel.data.index, quarter)
    published = bool(pd.notna(panel.data.iloc[t_q][model.spec.target]))
    return Nowcast(quarter, None if as_of is None else pd.Timestamp(as_of), target_value(model, X, sm, t_q),
                   factor_contributions(model, sm, t_q), published)


def news(model: DFM, old_as_of, new_as_of, quarter=None, *, raw: pd.DataFrame | None = None,
         releases: dict[str, MacroRelease] = MACRO_RELEASES, root: Path = RELEASES_DIR) -> News:
    """How the nowcast of ``quarter`` (default: ``new_as_of``'s) moved from ``old_as_of``
    to ``new_as_of``, print by print (news.py), parameters held fixed."""
    quarter = pd.Period(quarter, freq="Q") if quarter is not None else _default_quarter(new_as_of)
    raw = load_raw(new_as_of, releases=releases, root=root) if raw is None else raw
    end = _grid_end(quarter, new_as_of)
    old = model_panel(model, raw, old_as_of, end=end, releases=releases)
    new = model_panel(model, raw, new_as_of, end=end, releases=releases)
    out = decompose(model, model.standardize(old.data), model.standardize(new.data), new.data.index, quarter)
    if not out.impacts.empty:
        rows = [new.data.index.get_loc(p) for p in out.impacts["period"]]
        out.impacts["published"] = [new.published.iloc[r][t] for r, t in zip(rows, out.impacts["ticker"])]
        out.impacts["actual_native"] = [new.native.iloc[r][t] for r, t in zip(rows, out.impacts["ticker"])]
        out.impacts["blocks"] = [", ".join(releases[t].blocks) for t in out.impacts["ticker"]]
    return out


def nowcast_history(model: DFM, days, quarter=None, *, raw: pd.DataFrame | None = None,
                    releases: dict[str, MacroRelease] = MACRO_RELEASES, root: Path = RELEASES_DIR) -> pd.DataFrame:
    """The nowcast on each of ``days`` (a backfill: the same ``nowcast`` call per day),
    for a fixed ``quarter`` (default: each day's own quarter)."""
    days = pd.DatetimeIndex(days)
    raw = load_raw(days.max(), releases=releases, root=root) if raw is None else raw
    rows = []
    for day in days:
        nc = nowcast(model, day, quarter, raw=raw, releases=releases)
        rows.append({"as_of": day, "quarter": str(nc.quarter), "nowcast": nc.value, "published": nc.published,
                     **{f"common_{k}": v for k, v in nc.common.items()}})
    return pd.DataFrame(rows)


# A dated source beats a projection for the same release day (infra.pipeline.release_calendar)
_SOURCE_RANK = {"fiscal_data": 0, "nar": 1, "fred_release_dates": 2, "marketwatch": 3, "rule": 4}


def upcoming(model: DFM, as_of=None, days: int = 7, quarter=None, *, raw: pd.DataFrame | None = None,
             releases: dict[str, MacroRelease] = MACRO_RELEASES, root: Path = RELEASES_DIR,
             calendar: pd.DataFrame | None = None, calendar_root: Path | None = None) -> pd.DataFrame:
    """The model's releases due in the ``days`` after ``as_of`` (the release calendar as
    known then): per print, the period it will fill, the model's FORECAST of it, the
    MarketWatch consensus where the calendar showed one by ``as_of`` (it fills next
    week's forecasts in late; the release's own units), and the WEIGHT - GDP pp the
    nowcast of ``quarter`` moves per unit of surprise. ``units``: "release" (the release's
    own units, comparable to the consensus) when the spec has ``use_transforms=False``,
    else "model (transformed)". A print revising an already published period (a GDP
    second estimate, an S&P final) and a weekly print get a ``note`` instead."""
    from infra.pipeline import release_calendar as prc
    from infra.pipeline.econ_calendar import upcoming_consensus
    from infra.reference.events import SERIES

    as_of = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now()).normalize()
    quarter = pd.Period(quarter, freq="Q") if quarter is not None else _default_quarter(as_of)
    cal = prc.read_release_calendar(as_of, **({"root": calendar_root} if calendar_root else {})) \
        if calendar is None else calendar
    lo, hi = as_of + pd.Timedelta(days=1), as_of + pd.Timedelta(days=days + 1)
    cal = cal[(cal["timestamp"] >= lo) & (cal["timestamp"] < hi)].copy()
    cal["_day"] = cal["timestamp"].dt.normalize()
    cal["_rank"] = cal["source"].map(_SOURCE_RANK).fillna(9)
    # one row per (event, day): the best-dated source, with the stage ANY source states (a
    # calendar row named "US Services PMI" is the final its rule says it is)
    stage = cal[cal["stage"] != ""].groupby(["event", "_day"])["stage"].first()
    cal = cal.sort_values("_rank").drop_duplicates(["event", "_day"]).sort_values("timestamp")
    cal["stage"] = [stage.get((e, d), st) for e, d, st in zip(cal["event"], cal["_day"], cal["stage"])]
    raw = load_raw(as_of, releases=releases, root=root) if raw is None else raw
    panel = model_panel(model, raw, as_of, end=_grid_end(quarter, as_of), releases=releases)
    X = model.standardize(panel.data)
    series = list(model.series)
    rows, cells = [], []
    for r in cal.itertuples():
        tickers = [s.bbg_ticker for s in SERIES.values() if s.event == r.event and s.bbg_ticker in series]
        for t in tickers:
            col = series.index(t)
            observed = np.flatnonzero(np.isfinite(X[:, col]))
            step = 3 if panel.frequency[t] == "Q" else 1
            row_ = (observed[-1] + step) if len(observed) else None
            if panel.frequency[t] == "W":
                rows.append((r.timestamp, r.event, t, r.source, r.stage, None, None,
                             "weekly: enters the model only once its month is complete"))
            elif row_ is not None and row_ < len(X) and r.stage not in ("second", "third", "final"):
                cells.append((row_, col))
                rows.append((r.timestamp, r.event, t, r.source, r.stage, panel.data.index[row_], (row_, col), ""))
            else:
                note = ("revises an already published period" if r.stage in ("second", "third", "final")
                        else "its period lies beyond the panel (a later quarter, or a final of a flash)")
                rows.append((r.timestamp, r.event, t, r.source, r.stage, None, None, note))
    out = pd.DataFrame(rows, columns=["timestamp", "event", "ticker", "source", "stage", "period", "_cell", "note"])
    if out.empty:
        return out.drop(columns="_cell")
    pro = prospective(model, X, panel.data.index, quarter, sorted({c for c in cells}))
    pro = {(int(a), int(b)): (f, w) for a, b, f, w in pro[["row", "col", "forecast", "weight"]].itertuples(index=False)}
    mu, sd = model.mu, model.sd
    # model units carry the table's sign; without transforms, flip it back -> the release's own units
    native = not model.spec.use_transforms
    sign = {t: (1 if t == model.spec.target else releases[t].sign) for t in series}
    out["forecast"] = [(mu[c[1]] + sd[c[1]] * pro[c][0]) * (sign[series[c[1]]] if native else 1)
                       if c in pro else np.nan for c in out["_cell"]]
    out["units"] = "release" if native else "model (transformed)"
    out["weight"] = [pro[c][1] / sd[c[1]] * (sign[series[c[1]]] if native else 1) if c in pro else np.nan
                     for c in out["_cell"]]
    cons = []
    for r in out.itertuples():
        rel = releases.get(r.ticker)
        c = upcoming_consensus(rel, as_of, as_of + pd.Timedelta(days=days)) if rel and rel.calendar_pattern else None
        hit = c[c["timestamp"] == r.timestamp.normalize()] if c is not None and len(c) else None
        cons.append(hit["consensus_mw"].iloc[0] if hit is not None and len(hit) else np.nan)
    out["consensus_mw"] = cons
    return out.drop(columns="_cell")
