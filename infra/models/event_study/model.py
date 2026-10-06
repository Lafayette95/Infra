"""The event-study model: prepare (steps 1-2: the events' windows and their P&L) -> fit
(steps 3-4: the tests, per instrument) -> predict (each new event: the fitted expectation and
whether the study passes - the signal - next to the realised move once known).

    study = EventStudy("nfp_morning")
    panel = study.read_panel("2016-01-01", "2026-09-30")    # grid step P&L (infra.pipeline.event_pnl)
    data = study.prepare(panel)                             # windows + their moves, + placebo windows
    study.fit(data, as_of="2025-12-31")                     # completed events up to as_of only
    study.fitted_.table                                     # per instrument: stats, tests, passed
    study.predict(data, start="2025-12-31")                 # later events with the frozen fit
    study.paths(panel, data)                                # step-by-step path of every event

``prepare`` rows (one per window, index = the window's START point - when a trade would be put
on): ``kind`` (``event`` / ``placebo``), ``anchor`` (the anchor occurrence), ``end``,
``start_day`` / ``end_day``, ``legal`` / ``reason``, ``known_from``, ``pattern`` (day gap :
start slot : end slot), and per instrument ``pnl:<inst>`` (the sum of the steps in
(start, end]) and ``missing:<inst>`` (steps without a value: then ``pnl`` is NaN). Illegal
windows are kept (indexed by their anchor) with their reason.

Point in time: ``fit(as_of)`` uses only windows that have ENDED by ``as_of``; ``predict`` gives
a row for each window STARTING after the fit date - the walk-forward of
``infra.models.walk_forward`` / ``infra.models.runs`` (weekly fit-append, daily
predict-append, rebuild) applies unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from infra.models.base import Model
from infra.models.event_study import stats as st
from infra.models.event_study.config import EventStudySpec, get_event_study_spec
from infra.models.stats.common import PARAM_COLUMNS, cutoff_mask, params_frame
from infra.processing import event_windows as ew
from infra.reference.event_grid import resolve_cycle

ROW_STATS = ("t", "n", "passed", "ev_vol")      # fitted statistics copied onto every predicted row
COND_ROW_STATS = ("t", "n", "passed", "ev_vol", "t_did", "cond_passed")
PREP_COLUMNS = ["kind", "anchor", "end", "start_day", "end_day", "legal", "reason", "known_from", "pattern"]


@dataclass
class EventStudyFit:
    as_of: pd.Timestamp
    table: pd.DataFrame          # instrument x (stats + tests)
    code: str


def window_moves(panel: pd.DataFrame, starts, ends) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sum of steps in (start, end], count of missing steps) per window and instrument."""
    idx = panel.index
    vals = panel.to_numpy(dtype="float64")
    c = np.vstack([np.zeros((1, vals.shape[1])), np.nancumsum(vals, axis=0)])
    miss = np.vstack([np.zeros((1, vals.shape[1])), np.cumsum(~np.isfinite(vals), axis=0)])
    s = idx.get_indexer(pd.DatetimeIndex(starts))
    e = idx.get_indexer(pd.DatetimeIndex(ends))
    ok = (s >= 0) & (e >= 0) & (e > s)
    pnl = np.full((len(s), vals.shape[1]), np.nan)
    gaps = np.full((len(s), vals.shape[1]), np.nan)
    pnl[ok] = c[e[ok] + 1] - c[s[ok] + 1]
    gaps[ok] = miss[e[ok] + 1] - miss[s[ok] + 1]
    pnl = np.where(gaps > 0, np.nan, pnl)
    return (pd.DataFrame(pnl, columns=panel.columns), pd.DataFrame(gaps, columns=panel.columns))


class EventStudy(Model):
    """See the module docstring."""

    def __init__(self, spec: EventStudySpec | str | None = None, **overrides):
        self.spec = get_event_study_spec(spec, **overrides)
        self.code = ew.parse_code(self.spec.code)
        self.cycle = resolve_cycle(self.code.cycle)
        self.fitted_: EventStudyFit | None = None

    # ------------------------------------------------------------------ inputs
    def grid(self, start, end) -> ew.Grid:
        from infra.processing.schedule_rules import business_days
        return ew.make_grid(self.cycle, business_days(start, end, self.cycle.calendar))

    def read_panel(self, start, end, **kwargs) -> pd.DataFrame:
        """The grid step P&L of the spec's instruments from the spec's source (disk only)."""
        from infra.pipeline.event_pnl import grid_pnl
        return grid_pnl(self.spec.source, list(self.spec.instruments), self.grid(start, end), **kwargs)

    def occurrences(self, as_of=None) -> pd.DataFrame:
        """The code's events as known at ``as_of`` (``ignore_as_of``: as known now)."""
        from infra.pipeline.release_calendar import read_release_calendar
        events = [e for e, _ in self.code.dt_refs]
        knowledge = None if (self.spec.ignore_as_of or as_of is None) else as_of
        cal = read_release_calendar(None if knowledge is None else knowledge, events=events)
        return ew.consolidate(cal)

    # ------------------------------------------------------------------ 1-2. prepare
    def condition_timeline(self, start, end) -> pd.DataFrame:
        """The condition series' point-in-time rows (``read_available``), from ``history``
        before ``start`` (the partition's warm-up)."""
        from infra.pipeline.series_panel import read_available
        c = self.spec.condition
        return read_available(c.series, pd.Timestamp(start) - pd.Timedelta(c.history), end)

    def _condition(self, starts, timeline: pd.DataFrame, step: pd.Timedelta) -> pd.DataFrame:
        from infra.models.event_study import conditions as ec
        c = self.spec.condition
        states = ec.state_timeline(timeline, period_diff=c.period_diff)
        feat = ec.feature(states, c.steps, grammar=c.feature, kind=ec.input_kind(c.series))
        buckets = ec.partition(feat, c.partition)
        return ec.at_windows(starts, feat, buckets, c.lag_steps * step)

    def prepare(self, raw: pd.DataFrame, *, as_of=None, occurrences: pd.DataFrame | None = None,
                condition_timeline: pd.DataFrame | None = None, **_) -> pd.DataFrame:
        """``raw`` = the grid step P&L panel (``read_panel``). Events as known at ``as_of``
        (default: the panel's last day; ``spec.ignore_as_of``: as known now)."""
        from infra.processing.schedule_rules import business_days
        panel = raw.sort_index()
        days = ew.local_days(panel.index, self.cycle)
        if occurrences is None:
            occurrences = self.occurrences(as_of if as_of is not None else days.max())
        # only events from the data's first day on; the grid runs PAST the data's last day so
        # already-known upcoming events resolve (legal, P&L pending)
        occurrences = occurrences[occurrences["day"] >= days.min()]
        last = max(days.max(), occurrences["day"].max() if len(occurrences) else days.max())
        grid = ew.make_grid(self.cycle, business_days(days.min(), last + pd.Timedelta(days=45),
                                                      self.cycle.calendar))
        windows = ew.resolve(self.code, occurrences, grid, max_gap=self.spec.max_pair_gap)
        windows["pattern"] = ew.pattern_of(windows).where(windows["legal"], "")
        windows["kind"] = "event"
        plac = ew.placebo(windows, grid)
        plac["kind"] = "placebo"
        frame = pd.concat([windows, plac], ignore_index=True) if len(plac) else windows
        frame["legal"] = frame["legal"].astype(bool)
        pnl, gaps = window_moves(panel, frame["start"], frame["end"])
        out = frame[PREP_COLUMNS].copy()
        for inst in panel.columns:
            out[f"pnl:{inst}"] = pnl[inst].to_numpy()
            out[f"missing:{inst}"] = gaps[inst].to_numpy()
        if self.spec.condition is not None:
            tl = condition_timeline if condition_timeline is not None else self.condition_timeline(
                days.min(), max(days.max(), grid.days.max()) + pd.Timedelta(days=1))
            cond = self._condition(frame["start"], tl, grid.step)
            for col in cond.columns:
                out[col] = cond[col].to_numpy()
            # a window whose regime lookup lies beyond the latest value available now: its
            # regime is the LATEST known one, provisional until its start (user decision 2026-10-05)
            latest = pd.Timestamp(pd.DatetimeIndex(tl["available_at"]).max()) if len(tl) else pd.NaT
            lookup = pd.DatetimeIndex(frame["start"]) - self.spec.condition.lag_steps * grid.step
            out["cond_provisional"] = np.asarray(lookup > latest) if pd.notna(latest) else True
        out.index = pd.DatetimeIndex(frame["start"].fillna(frame["anchor"]), name="timestamp")
        out.attrs["instruments"] = list(panel.columns)
        out.attrs["code"] = self.code.encode()
        return out.sort_index(kind="stable")

    # ------------------------------------------------------------------ 3-4. fit
    def fit(self, prepared: pd.DataFrame, as_of=None) -> "EventStudy":
        insts = self._instruments(prepared)
        done_end = pd.DatetimeIndex(prepared["end"])
        if as_of is None:
            as_of = done_end[prepared["legal"].to_numpy()].max()
        as_of = pd.Timestamp(as_of)
        done = cutoff_mask(done_end, as_of) & prepared["legal"].to_numpy() & done_end.notna()
        if self.spec.window is not None:
            done &= np.asarray(prepared.index > as_of - pd.Timedelta(self.spec.window))
        ev = prepared[done & (prepared["kind"] == "event").to_numpy()]
        pl = prepared[done & (prepared["kind"] == "placebo").to_numpy()]
        pl = pl[pl["pattern"].isin(set(ev["pattern"]))]
        rows = {}
        cond = self.spec.condition
        for inst in insts:
            x, y = ev[f"pnl:{inst}"].to_numpy(), pl[f"pnl:{inst}"].to_numpy()
            s = st.event_stats(x, ev.index, y, trim=self.spec.trim)
            rows[inst] = s | st.passes(s, self.spec)
            if cond is None:
                continue
            bx, by = ev["cond_bucket"].to_numpy(dtype="float64"), pl["cond_bucket"].to_numpy(dtype="float64")
            rows[inst] |= st.condition_overall(x, ev["cond_value"].to_numpy(dtype="float64"), y,
                                               pl["cond_value"].to_numpy(dtype="float64"), bx)
            bucket_spec = replace(self.spec, min_obs=cond.min_bucket_obs)
            for b in (-1.0, 0.0, 1.0):
                inb, inp = bx == b, by == b
                rest_x, rest_p = np.isfinite(bx) & ~inb, np.isfinite(by) & ~inp
                sb = st.event_stats(x[inb], ev.index[inb], y[inp], trim=self.spec.trim)
                sb |= st.passes(sb, bucket_spec)
                sb |= st.bucket_stats(x[inb], y[inp], x[rest_x], y[rest_p])
                did_ok = cond.did_t_min is None or (np.isfinite(sb["t_did"]) and abs(sb["t_did"]) >= cond.did_t_min)
                rest_ok = cond.vs_rest_t_min is None or (np.isfinite(sb["t_vs_rest"])
                                                         and abs(sb["t_vs_rest"]) >= cond.vs_rest_t_min)
                sb["test_did"], sb["test_vs_rest"] = float(did_ok), float(rest_ok)
                sb["cond_passed"] = float(sb["passed"] == 1.0 and did_ok and rest_ok)
                rows[f"{inst}|{int(b):+d}"] = sb
        table = pd.DataFrame(rows).T
        table.index.name = "instrument"
        self.fitted_ = EventStudyFit(as_of=as_of, table=table.astype("float64"), code=self.code.encode())
        return self

    def _instruments(self, prepared) -> list[str]:
        return prepared.attrs.get("instruments") or [c.split(":", 1)[1] for c in prepared.columns
                                                    if c.startswith("pnl:")]

    # ------------------------------------------------------------------ predict
    def predict(self, prepared: pd.DataFrame | None = None, *, start=None, end=None) -> pd.DataFrame:
        """Each EVENT window starting after ``start`` up to ``end``: ``end``, ``legal``,
        ``reason``, and per instrument the realised ``pnl:`` (NaN until it has ended / where
        data is missing), ``expected:`` (the fitted mean move) and ``signal:`` (+1 / -1 = the
        mean's sign where the study passes, else 0); ``in_sample``."""
        self.check_fitted()
        if prepared is None:
            raise ValueError("EventStudy.predict needs prepared data")
        ev = prepared[(prepared["kind"] == "event").to_numpy()]
        keep = np.ones(len(ev), dtype=bool)
        if start is not None:
            keep &= ~cutoff_mask(ev.index, start)
        if end is not None:
            keep &= cutoff_mask(ev.index, end)
        ev = ev[keep]
        t = self.fitted_.table
        out = pd.DataFrame({"end": ev["end"], "legal": ev["legal"].astype(float), "reason": ev["reason"],
                            "anchor": ev["anchor"]}, index=ev.index)
        if "cond_bucket" in ev:
            return self._predict_conditional(ev, out)
        for inst in t.index:
            out[f"pnl:{inst}"] = ev[f"pnl:{inst}"].to_numpy() if f"pnl:{inst}" in ev else np.nan
            out[f"expected:{inst}"] = np.where(ev["legal"], t.loc[inst, "mean"], np.nan)
            sig = float(np.sign(t.loc[inst, "mean"]) * t.loc[inst, "passed"]) + 0.0 \
                if np.isfinite(t.loc[inst, "mean"]) else 0.0
            out[f"signal:{inst}"] = np.where(ev["legal"], sig, 0.0)
            for stat in ROW_STATS:
                out[f"{stat}:{inst}"] = np.where(ev["legal"], t.loc[inst, stat], np.nan)
        ends = pd.Series(pd.DatetimeIndex(ev["end"]), index=ev.index)
        ends = ends.fillna(pd.Series(ev.index, index=ev.index))
        out["in_sample"] = cutoff_mask(pd.DatetimeIndex(ends), self.fitted_.as_of)
        out = out.drop(columns=["reason"])
        return out

    def _predict_conditional(self, ev: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        """Each event's expectation and signal come from the bucket its condition was in at its
        start: +1 / -1 (the bucket mean's sign) where the bucket passes its own tests AND the
        conditioning tests (``cond_passed``), else 0. ``expected_all`` = the unconditional mean."""
        t = self.fitted_.table
        out["cond_bucket"] = ev["cond_bucket"].to_numpy()
        if "cond_provisional" in ev:
            out["cond_provisional"] = ev["cond_provisional"].to_numpy(dtype=float)
        legal = ev["legal"].to_numpy(dtype=bool)
        for inst in [i for i in t.index if "|" not in i]:
            out[f"pnl:{inst}"] = ev[f"pnl:{inst}"].to_numpy() if f"pnl:{inst}" in ev else np.nan
            out[f"expected_all:{inst}"] = np.where(legal, t.loc[inst, "mean"], np.nan)
            exp, sig = np.full(len(ev), np.nan), np.zeros(len(ev))
            extra = {stat: np.full(len(ev), np.nan) for stat in COND_ROW_STATS}
            for b in (-1.0, 0.0, 1.0):
                key = f"{inst}|{int(b):+d}"
                if key not in t.index:
                    continue
                m = legal & (ev["cond_bucket"].to_numpy(dtype="float64") == b)
                mean, ok = t.loc[key, "mean"], t.loc[key, "cond_passed"] == 1.0
                exp[m] = mean
                sig[m] = (float(np.sign(mean)) if ok and np.isfinite(mean) else 0.0) + 0.0
                for stat in COND_ROW_STATS:
                    extra[stat][m] = t.loc[key, stat]
            out[f"expected:{inst}"], out[f"signal:{inst}"] = exp, sig
            for stat, v in extra.items():
                out[f"{stat}:{inst}"] = v
        ends = pd.Series(pd.DatetimeIndex(ev["end"]), index=ev.index)
        ends = ends.fillna(pd.Series(ev.index, index=ev.index))
        out["in_sample"] = cutoff_mask(pd.DatetimeIndex(ends), self.fitted_.as_of)
        return out.drop(columns=["reason"])

    # ------------------------------------------------------------------ paths
    def paths(self, panel: pd.DataFrame, prepared: pd.DataFrame, *, relative_to: str = "start") -> pd.DataFrame:
        """Every legal event's path, step by step: long ``timestamp`` (window start),
        ``instrument``, ``step`` (1..n after the start, or -n+1..0 up to the end with
        ``relative_to="end"``), ``at`` (the grid point) and ``cum`` (cumulative move from the
        start)."""
        ev = prepared[(prepared["kind"] == "event").to_numpy() & prepared["legal"].to_numpy()]
        idx = panel.index
        rows = []
        for t0, r in ev.iterrows():
            s, e = idx.get_indexer([t0, r["end"]])
            if s < 0 or e < 0:
                continue
            seg = panel.iloc[s + 1:e + 1]
            n = len(seg)
            steps = np.arange(1, n + 1) if relative_to == "start" else np.arange(-n + 1, 1)
            cum = seg.cumsum(skipna=False)
            for inst in panel.columns:
                rows.append(pd.DataFrame({"timestamp": t0, "instrument": inst, "step": steps, "at": seg.index,
                                          "cum": cum[inst].to_numpy()}))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["timestamp", "instrument", "step", "at", "cum"])

    # ------------------------------------------------------------------ params
    def params(self) -> pd.DataFrame:
        self.check_fitted()
        f = self.fitted_
        meta = pd.DataFrame([{"section": "meta", "row": "as_of", "col": f.as_of.isoformat(), "value": np.nan},
                             {"section": "meta", "row": "code", "col": f.code, "value": np.nan}])
        return pd.concat([meta, params_frame("event_stat", f.table)], ignore_index=True)[PARAM_COLUMNS]

    @classmethod
    def from_params(cls, params: pd.DataFrame, spec: EventStudySpec | str | None = None, **overrides) -> "EventStudy":
        m = cls(spec, **overrides)
        meta = params[params["section"] == "meta"]
        sub = params[params["section"] == "event_stat"]
        table = sub.pivot(index="row", columns="col", values="value")   # keeps NaN statistics
        table = table.reindex(columns=list(dict.fromkeys(sub["col"])), index=list(dict.fromkeys(sub["row"])))
        table.index.name = "instrument"
        m.fitted_ = EventStudyFit(as_of=pd.Timestamp(meta.loc[meta["row"] == "as_of", "col"].iloc[0]),
                                  table=table.astype("float64"),
                                  code=meta.loc[meta["row"] == "code", "col"].iloc[0])
        return m


def make_event_study(spec: EventStudySpec | str | None = None, **overrides) -> EventStudy:
    return EventStudy(spec, **overrides)
