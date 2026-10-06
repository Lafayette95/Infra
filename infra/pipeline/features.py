"""The central FEATURE MAKER, point-in-time assembly (spec ``infra/processing/FEATURES.md``,
root CLAUDE.md 31). Disk only.

``feature(expr, start, end)``: ``expr`` = an INPUT, then the feature grammar of
``infra.processing.features`` (``"bmk:otr:US_BOND_10y | chg:20 | norm:vol:60"``). The result is
indexed by AVAILABILITY instant (UTC): a value counts from when its input became public, never
from its label. Inputs:

* a series id (``infra.pipeline.series_panel``): read with ``read_available``; a vintage series
  (macro releases) becomes its STATE at each availability instant (the latest period's value as
  known then); its declared kind (``level`` / ``moves``) drives the grammar;
* ``vol(ID,SPAN)``: EWMA vol of ID's daily changes; ``corr(ID1,ID2,W)`` / ``beta(ID1,ID2,W)``:
  rolling correlation / beta of their daily changes; ``spread(ID1,ID2)`` /
  ``spread(ID1,ID2,BETA)``: ID1 - [BETA x] ID2 (levels);
* ``evt:to:EVENT`` / ``evt:since:EVENT``: business days to the next / since the last occurrence
  of a registry event, as KNOWN that day (release calendar ``known_from``);
* ``surprise:<release ticker>``: actual - MarketWatch consensus (``infra.pipeline.econ_calendar
  .consensus``, NOT Bloomberg's survey), available at the release instant (its event's time);
* ``model:<run>:<column>``: a stored model run's prediction column, available the day after its
  row's label (conservative);
* ``pdiff:<series id>``: a vintage series' latest period's change against the previous period, as
  known at each availability instant (the event study's ``period_diff``).

``align(series, index, source)``: a feature onto another timeline (a model's daily index), the
latest value at or before each instant, carried at most ``FEATURE_FFILL_LIMITS[source]``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from infra.config import FEATURE_FFILL_LIMITS
from infra.processing import features as fx

_DERIVED = re.compile(r"^(vol|corr|beta|spread)\((.*)\)$")
_ONE_DAY = pd.Timedelta(days=1)


def split(expr: str) -> tuple[str, str]:
    """``"ID | steps"`` -> (input, grammar); no steps -> ``"lvl"``."""
    head, _, rest = expr.partition("|")
    return head.strip(), (rest.strip() or "lvl")


def _timeline(series_id: str, start, end) -> pd.Series:
    """A series id's value at each availability instant."""
    from infra.processing.features import state_timeline
    from infra.pipeline.series_panel import availability_of, read_available
    rows = read_available(series_id, start, end)
    if rows.empty:
        return pd.Series(dtype="float64")
    if availability_of(series_id).kind == "publication":
        tl = state_timeline(rows)
    else:
        tl = rows.drop_duplicates("available_at", keep="last")
    return pd.Series(tl["value"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(tl["available_at"])).sort_index()


def _changes(series_id: str, s: pd.Series) -> pd.Series:
    from infra.pipeline.series_panel import kind_of
    return s if kind_of(series_id) == "moves" else s.diff()


def _pair(a: str, b: str, start, end) -> tuple[pd.Series, pd.Series]:
    sa, sb = _timeline(a, start, end), _timeline(b, start, end)
    idx = sa.index.union(sb.index)
    return sa.reindex(idx).ffill(), sb.reindex(idx).ffill()


def _derived(name: str, args: list[str], start, end) -> tuple[pd.Series, str]:
    from infra.pipeline.series_panel import kind_of
    if name == "vol":
        sid, span = args[0], int(args[1])
        s = _timeline(sid, start, end)
        return _changes(sid, s).ewm(span=span, min_periods=max(span // 2, 10)).std(), "level"
    a, b = args[0], args[1]
    sa, sb = _pair(a, b, start, end)
    if name == "spread":
        beta = float(args[2]) if len(args) > 2 else 1.0
        if kind_of(a) == "moves":
            return sa - beta * sb, "moves"
        return sa - beta * sb, "level"
    da, db = _changes(a, sa), _changes(b, sb)
    w = int(args[2])
    if name == "corr":
        return da.rolling(w, min_periods=w // 2).corr(db), "level"
    cov = da.rolling(w, min_periods=w // 2).cov(db)
    return cov / db.rolling(w, min_periods=w // 2).var(), "level"   # beta


def _events(event: str, start, end, direction: str) -> pd.Series:
    from infra.pipeline.release_calendar import read_release_calendar
    from infra.processing import event_windows as ew
    from infra.processing.schedule_rules import business_days
    occ = ew.consolidate(read_release_calendar(None, events=[event]))
    days = business_days(pd.Timestamp(start), pd.Timestamp(end), "market")
    if occ.empty:
        return pd.Series(np.nan, index=days)
    od = pd.DatetimeIndex(occ["day"]).normalize()
    known = pd.DatetimeIndex(pd.to_datetime(occ["known_from"])).normalize()
    # count business days on a calendar reaching well past the window: the next occurrence can lie
    # beyond it (an FOMC meeting six weeks after the window's last day)
    cal = business_days(min(pd.Timestamp(start), od.min()) - pd.Timedelta(days=10),
                        max(pd.Timestamp(end), od.max()) + pd.Timedelta(days=10), "market")
    pos = cal.searchsorted
    out = np.full(len(days), np.nan)
    for i, d in enumerate(days):
        ok = known <= d
        if direction == "to":
            nxt = od[ok & (od >= d)]
            if len(nxt):
                out[i] = pos(nxt.min()) - pos(d)
        else:
            prv = od[ok & (od <= d)]
            if len(prv):
                out[i] = pos(d) - pos(prv.max())
    return pd.Series(out, index=days)


def _surprise(ticker: str) -> pd.Series:
    from infra.config import MACRO_RELEASES
    from infra.pipeline.econ_calendar import consensus
    from infra.reference.events import EVENTS
    from infra.trading_calendar import snap_instants
    rel = MACRO_RELEASES[ticker]
    c = consensus(rel).dropna(subset=["surprise"])
    if c.empty:
        return pd.Series(dtype="float64")
    ev = EVENTS.get(rel.series.event)
    days = pd.DatetimeIndex(pd.to_datetime(c["timestamp"])).normalize()
    if ev is not None and ev.time_local:
        at = snap_instants(days, ev.time_local, ev.timezone)
    else:
        at = snap_instants(days, "23:59", "America/New_York")       # time unknown: the day's end
    s = pd.Series(c["surprise"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(at))
    return s[~s.index.duplicated(keep="last")].sort_index()


def _model(run: str, column: str) -> pd.Series:
    from infra.config import MODEL_RUNS_DIR
    from infra.storage import model_runs as store
    p = store.read_predictions(run, root=MODEL_RUNS_DIR)
    if p.empty or column not in p:
        return pd.Series(dtype="float64")
    s = p[column].astype("float64")
    s.index = pd.DatetimeIndex(s.index).normalize() + _ONE_DAY        # known the day after its label
    return s[~s.index.duplicated(keep="last")].sort_index()


def input_series(inp: str, start, end) -> tuple[pd.Series, str, str]:
    """(series indexed by availability, input kind, source type for the ffill limit)."""
    from infra.pipeline.series_panel import kind_of, parse_id
    m = _DERIVED.match(inp)
    if m:
        s, kind = _derived(m.group(1), [a.strip() for a in m.group(2).split(",")], start, end)
        return s, kind, "derived"
    if inp.startswith("evt:"):
        _, direction, event = inp.split(":", 2)
        return _events(event, start, end, direction), "level", "evt"
    if inp.startswith("surprise:"):
        return _surprise(inp.split(":", 1)[1]), "level", "surprise"
    if inp.startswith("pdiff:"):                 # a vintage series' latest-period change, as known then
        from infra.pipeline.series_panel import read_available
        from infra.processing.features import state_timeline
        sid = inp.split(":", 1)[1]
        tl = state_timeline(read_available(sid, start, end), period_diff=True)
        s = pd.Series(tl["value"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(tl["available_at"]))
        return s.sort_index(), "level", parse_id(sid)[0]
    if inp.startswith("model:"):
        _, run, col = inp.split(":", 2)
        return _model(run, col), "level", "model"
    src, _ = parse_id(inp)
    return _timeline(inp, start, end), kind_of(inp), src


def feature(expr: str, start, end) -> pd.Series:
    """The feature ``expr`` (input | grammar) over ``[start, end]``, indexed by availability."""
    inp, grammar = split(expr)
    s, kind, src = input_series(inp, start, end)
    out = fx.apply(s, grammar, kind) if len(s) else pd.Series(dtype="float64")
    out.attrs = {"feature": expr, "input_kind": kind, "source": src}
    return out


def align(series: pd.Series, index, source: str | None = None, *, limit=None) -> pd.Series:
    """``series`` (indexed by availability) onto ``index``: the latest value at or before each
    instant, at most ``limit`` (default ``FEATURE_FFILL_LIMITS[source]``) old - else NaN."""
    idx = pd.DatetimeIndex(index)
    src = source or series.attrs.get("source", "derived")
    lim = pd.Timedelta(limit if limit is not None else FEATURE_FFILL_LIMITS.get(src, "5D"))
    s = series.dropna().sort_index()
    if s.empty:
        return pd.Series(np.nan, index=idx)
    left = pd.DataFrame({"t": idx.astype("datetime64[ns]")}).reset_index()
    right = pd.DataFrame({"t": s.index.astype("datetime64[ns]"), "v": s.to_numpy()})
    m = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward", tolerance=lim)
    return pd.Series(m.sort_values("index")["v"].to_numpy(), index=idx)
