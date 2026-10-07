"""Positioning measures: assemble inputs from stored data, compute, store, read. Disk only.

Asymmetric reaction (``infra.analytics.positioning.asymmetry``, specs in
``infra.analytics.positioning.config``):

* **moves** in YIELD direction, bp: daily from any series id (``infra.pipeline.series_panel``;
  a level differenced, a moves series as is, x sign x scale); intraday from the event-study
  grid P&L (``infra.pipeline.event_pnl.grid_pnl``, bp of a long futures position -> sign -1);
* **events** (family 1): each release's surprise at its release instant (the feature maker's
  ``surprise:`` input: MarketWatch consensus, NOT Bloomberg's), standardised per release;
  a window = the release day (daily) or (instant - pre, instant + post] on the 1-minute
  grid (intraday), coincident releases sharing a window;
* **store** ``~/Database/Derived/Positioning/Asymmetry``, long: ``timestamp`` (the day the
  value refers to: a daily measure's day, an intraday measure's last grid day, an event's
  day), ``spec``, ``family``, ``measure``, ``instrument``, ``value``. A run replaces its
  spec's whole history (cheap: the inputs are re-read and everything recomputed).

Readable everywhere as series ids ``pos:<spec>:<measure>:<instrument>`` (series_panel).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics.positioning import asymmetry as asym
from infra.analytics.positioning.config import AsymmetrySpec, get_spec
from infra.config import DERIVED_ROOT
from infra.storage import parquet_store

log = logging.getLogger(__name__)

ASYMMETRY_DIR = DERIVED_ROOT / "Positioning" / "Asymmetry"
KEYS = ["timestamp", "spec", "family", "measure", "instrument"]
NY = "America/New_York"


# ------------------------------------------------------------------ inputs

def daily_moves(spec: AsymmetrySpec, start, end) -> pd.DataFrame:
    """Yield-direction bp moves per instrument, each on its own days (a holiday is not a 0)."""
    from infra.pipeline.series_panel import kind_of, read_panel
    panel = read_panel(list(spec.instruments), start, end, as_of=end)
    out = {}
    for col in spec.instruments:
        s = panel[col].dropna()
        mv = s.diff() if kind_of(col) == "level" else s
        out[col] = mv * spec.sign * spec.scale
    return pd.DataFrame(out).reindex(panel.index)


def _grid(cycle_name: str, days):
    from infra.processing import event_windows as ew
    from infra.reference.event_grid import resolve_cycle
    return ew.make_grid(resolve_cycle(cycle_name), days)


def intraday_moves(spec: AsymmetrySpec, start, end) -> tuple[pd.DataFrame, pd.Series]:
    """Grid-step moves (index = grid point, UTC) and each point's local day label."""
    from infra.pipeline.event_pnl import grid_pnl
    from infra.trading_calendar import local_wallclock
    from infra.processing.schedule_rules import business_days
    grid = _grid(spec.cycle, business_days(start, end, "market"))
    steps = grid_pnl(spec.pnl_source, list(spec.instruments), grid) * spec.sign * spec.scale
    day = pd.Series(local_wallclock(steps.index, grid.cycle.timezone).normalize(), index=steps.index)
    return steps, day


def surprises(spec: AsymmetrySpec, start, end) -> pd.DataFrame:
    """Instant x release raw surprises (actual - consensus), as of the release instant."""
    from infra.pipeline.features import _surprise
    cols = {}
    for ticker in spec.releases:
        try:
            s = _surprise(ticker)
        except Exception as e:  # a release whose calendar pattern is broken: skip, say so
            log.warning("asymmetry: no surprises for %s (%s)", ticker, e)
            continue
        if len(s):
            cols[ticker] = s
    if not cols:
        return pd.DataFrame()
    wide = pd.DataFrame(cols).sort_index()
    return wide[(wide.index >= pd.Timestamp(start)) & (wide.index < pd.Timestamp(end) + pd.Timedelta(days=1))]


def _ny_day(instants: pd.DatetimeIndex) -> pd.DatetimeIndex:
    from infra.trading_calendar import local_wallclock
    return pd.DatetimeIndex(local_wallclock(instants, NY)).normalize()


def event_windows_daily(spec: AsymmetrySpec, z: pd.DataFrame, moves: pd.DataFrame):
    """Daily: one window per release DAY; its move = the day's move."""
    day = _ny_day(z.index)
    zd = z.groupby(day).sum(min_count=1)
    return zd, moves.reindex(zd.index), None


def event_windows_intraday(spec: AsymmetrySpec, z: pd.DataFrame):
    """Intraday: one window per release INSTANT on the 1-minute grid: the move over
    (t - pre, t + post] and the continuation over (t + post, t + after]. A window needing a
    point outside its day's grid (e.g. 06:00 NFIB: the first step reaches overnight) is
    left out; any missing step makes it NaN."""
    from infra.pipeline.event_pnl import grid_pnl
    from infra.processing.schedule_rules import business_days
    zi = z.groupby(z.index).sum(min_count=1)
    days = _ny_day(zi.index).unique()
    trading = business_days(days.min(), days.max(), "market")
    grid = _grid(spec.event_cycle, days[days.isin(trading)])
    steps = grid_pnl(spec.pnl_source, list(spec.instruments), grid) * spec.sign * spec.scale
    first = pd.Series(grid.first, index=grid.days)
    idx = steps.index
    vals = steps.to_numpy(dtype=float)
    pre, post, after = (pd.Timedelta(minutes=m) for m in (spec.event_pre_minutes, spec.event_post_minutes,
                                                          spec.event_after_minutes))

    def window(a, b):
        i0, i1 = idx.searchsorted(a, side="right"), idx.searchsorted(b, side="right")
        block = vals[i0:i1]
        expected = int(round((b - a) / grid.step))
        if len(block) != expected or not len(block):
            return np.full(vals.shape[1], np.nan)
        return np.where(np.isnan(block).any(axis=0), np.nan, block.sum(axis=0))

    first_moves, after_moves = [], []
    for t, d in zip(zi.index, _ny_day(zi.index)):
        if d not in first.index or t - pre < first[d]:
            first_moves.append(np.full(vals.shape[1], np.nan))
            after_moves.append(np.full(vals.shape[1], np.nan))
            continue
        first_moves.append(window(t - pre, t + post))
        after_moves.append(window(t + post, t + after))
    y = pd.DataFrame(first_moves, index=zi.index, columns=steps.columns)
    cont = pd.DataFrame(after_moves, index=zi.index, columns=steps.columns)
    return zi, y, cont


# ------------------------------------------------------------------ compute

def _long(frame: pd.DataFrame, label: pd.Series | None, family: str, measure: str, spec: str) -> pd.DataFrame:
    """Wide (index x instrument) -> long rows on day labels (last value per day)."""
    if label is None:
        f = frame.copy()
    else:  # the day's LAST observation (not the last non-NaN: a value that lapsed stays lapsed)
        lab = label.reindex(frame.index)
        last = ~pd.Index(lab.to_numpy()).duplicated(keep="last")
        f = frame[last]
        f.index = lab.to_numpy()[last]
    f.index = pd.DatetimeIndex(f.index, name="timestamp")
    out = f.stack(future_stack=True).dropna().rename("value").reset_index()
    out.columns = ["timestamp", "instrument", "value"]
    out["spec"], out["family"], out["measure"] = spec, family, measure
    return out


def compute(spec: AsymmetrySpec | str, start=None, end=None) -> pd.DataFrame:
    """Every measure of ``spec`` over ``[start, end]``, long. Trailing only: the value on a
    day uses nothing after it (windows, betas, PCs, standardisation and impacts are all
    fitted on earlier data)."""
    spec = get_spec(spec) if isinstance(spec, str) else spec
    start = pd.Timestamp(start or spec.start)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now().normalize()
    if spec.frequency == "daily":
        moves, label = daily_moves(spec, start, end), None
    else:
        moves, label = intraday_moves(spec, start, end)
    rows = []

    # family 2: each instrument (and the factor) on its own
    if spec.factor == "pc1":
        factor, _ = asym.trailing_pc1(moves, spec.pc_window, spec.pc_refit_every, spec.min_obs)
    else:
        factor = moves[spec.factor_instrument].rename("factor")
    single = moves.assign(factor=factor)
    sv, _ = asym.semivariance_asymmetry(single, spec.window, spec.min_obs)
    vol = asym.trailing_vol(single, spec.vol_span, spec.min_obs)
    tail, nu, nd = asym.tail_asymmetry(single, vol, spec.k, spec.window, spec.min_obs, spec.min_big)
    skew = asym.rolling_skew(single, spec.window, spec.min_obs)
    for name, frame in (("semivar_asym", sv), ("skew", skew), ("tail_asym", tail), ("tail_n_up", nu),
                        ("tail_n_down", nd)):
        rows.append(_long(frame, label, "single", name, spec.name))

    # family 3: relative to the factor
    others = moves.drop(columns=[c for c in [spec.factor_instrument] if c])
    rel = asym.relative_asymmetry(others, factor, beta_window=spec.beta_window, window=spec.window, k=spec.k,
                                  vol_span=spec.vol_span, min_obs=spec.min_obs, min_big=spec.min_big)
    for name, frame in rel.items():
        rows.append(_long(frame, label, "relative", name, spec.name))

    # family 1: surprises
    raw = surprises(spec, start, end)
    if not raw.empty:
        z = asym.standardise_surprises(raw, spec.surprise_min_obs)
        if spec.frequency == "daily":
            zw, y, cont = event_windows_daily(spec, z, moves)
        else:
            zw, y, cont = event_windows_intraday(spec, z)
        refit = pd.date_range(zw.index.min().normalize(), zw.index.max(), freq=spec.impact_refit)
        expected = asym.impact_betas(zw, y, refit=refit, lookback=pd.Timedelta(days=spec.impact_lookback_days),
                                     ridge=spec.impact_ridge, min_events=spec.impact_min_events)
        keep = expected.notna().any(axis=1) & y.notna().any(axis=1)
        y, expected = y[keep], expected[keep]
        ev_label = pd.Series(_ny_day(y.index), index=y.index)
        sa = asym.surprise_asymmetry(y, expected, spec.event_window, spec.event_min_side, spec.event_min_side)
        for name, frame in sa.items():
            rows.append(_long(frame, ev_label, "surprise", name, spec.name))
        if cont is not None:
            ca = asym.continuation_asymmetry(y, cont[keep], spec.event_window, spec.event_min_side,
                                             spec.event_min_side)
            for name, frame in ca.items():
                rows.append(_long(frame, ev_label, "continuation", name, spec.name))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEYS + ["value"])
    out = out[(out["timestamp"] >= start.normalize()) & (out["timestamp"] <= end)]
    return out[KEYS + ["value"]].sort_values(KEYS).reset_index(drop=True)


# ------------------------------------------------------------------ store

def store(df: pd.DataFrame, *, root: Path = ASYMMETRY_DIR) -> None:
    """Replace each spec's whole stored history with ``df``'s rows."""
    if df.empty:
        return
    parquet_store.write_partitioned(df, root, KEYS, prune=True, prune_column="spec")


def build(spec: AsymmetrySpec | str, start=None, end=None, *, root: Path = ASYMMETRY_DIR) -> pd.DataFrame:
    df = compute(spec, start, end)
    store(df, root=root)
    return df


def read_asymmetry(spec: str | None = None, measure: str | None = None, instrument: str | None = None,
                   start=None, end=None, *, root: Path = ASYMMETRY_DIR) -> pd.DataFrame:
    filters = {k: [v] for k, v in (("spec", spec), ("measure", measure), ("instrument", instrument)) if v}
    raw = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                         end=None if end is None else pd.Timestamp(end) + pd.Timedelta(days=1),
                                         equals_in=filters or None)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=KEYS + ["value"])
    for c in ("spec", "family", "measure", "instrument"):
        raw[c] = raw[c].astype(str)
    return raw.sort_values(KEYS).reset_index(drop=True)
