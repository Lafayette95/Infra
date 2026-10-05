"""Conditional event studies and the global availability rules: series availability
(infra.pipeline.series_panel), the condition pipeline (infra/models/event_study/conditions.py),
the conditional tests and the model. Synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.models.event_study import stats as st
from infra.models.event_study.config import ConditionSpec, EventStudySpec
from infra.models.event_study.model import EventStudy
from infra.pipeline import series_panel as sp
from infra.models.event_study import conditions as ec

D = pd.Timestamp
NFP = "US_EMPLOYMENT_SITUATION"


# --------------------------------------------------------------------------- availability
def test_availability_rules_give_utc_instants():
    fri, sat = D("2024-07-05"), D("2024-07-06")
    assert sp.available_at("fut:ZN.v.0", [fri])[0] == D("2024-07-05 21:00")          # 16:00 CT
    assert sp.available_at("repo:SOFR", [fri])[0] == D("2024-07-08 12:00")           # Mon 08:00 ET
    assert sp.available_at("otr:US_BOND_10y", [D("2024-12-31")])[0] == D("2025-01-02 15:00")  # 10:00 EST, New Year
    assert sp.available_at("bar:ZN.v.0", [D("2024-07-05 13:30")])[0] == D("2024-07-05 13:31")
    assert sp.available_at("swap:USD:10y", [fri])[0] == D("2024-07-05 20:30")        # 15:00 ET + 90 min
    assert not sp.availability_of("bond:US_BOND_10y").verified
    with pytest.raises(ValueError):
        sp.available_at("release:PAYEMS", [fri])
    assert sat not in sp.available_at("repo:SOFR", [fri]).normalize()


# --------------------------------------------------------------------------- timeline / features
def test_state_timeline_keeps_the_latest_label_and_its_revisions():
    rows = pd.DataFrame({"label": [D("2024-05-01"), D("2024-06-01"), D("2024-05-01"), D("2024-07-01")],
                         "available_at": [D("2024-06-07 12:30"), D("2024-07-05 12:30"), D("2024-07-05 12:30"),
                                          D("2024-08-02 12:30")],
                         "value": [100.0, 110.0, 101.0, 115.0]})
    tl = ec.state_timeline(rows)
    assert tl["value"].tolist() == [100.0, 110.0, 115.0]
    pdiff = ec.state_timeline(rows, period_diff=True)
    assert np.isnan(pdiff["value"].iloc[0]) and pdiff["value"].tolist()[1:] == [9.0, 5.0]  # 110 - revised 101


def test_partitions_are_trailing():
    x = pd.Series(np.random.default_rng(0).normal(size=300), index=pd.date_range("2020-01-01", periods=300))
    b = ec.partition(x, "rolling_tercile:60")
    x2 = x.copy()
    x2.iloc[200:] += 100.0                                 # the future changes...
    b2 = ec.partition(x2, "rolling_tercile:60")
    np.testing.assert_array_equal(b[:200], b2[:200])        # ...the past regimes don't
    assert set(np.unique(b[~np.isnan(b)])) == {-1.0, 0.0, 1.0} and np.isnan(b[:29]).all()
    assert ec.partition(pd.Series([-2.0, 0.0, 3.0]), "sign").tolist() == [-1.0, 0.0, 1.0]
    assert ec.partition(pd.Series([-2.0, 0.0, 3.0]), "fixed:-1:1").tolist() == [-1.0, 0.0, 1.0]
    z = ec.partition(x, "zscore:60:1")
    assert np.nanmean(z == 0) > 0.5


def test_window_lookup_respects_lag_and_availability():
    feat = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(["2024-07-05 12:00", "2024-07-05 12:30",
                                                               "2024-07-05 13:00"]))
    out = ec.at_windows([D("2024-07-05 12:30"), D("2024-07-05 12:45")], feat, np.array([-1.0, 0.0, 1.0]),
                        pd.Timedelta(minutes=15))
    assert out["cond_value"].tolist() == [1.0, 2.0]   # the 12:30 value is NOT known 15 min before 12:30
    assert out["cond_asof"].iloc[1] == D("2024-07-05 12:30")


# --------------------------------------------------------------------------- tests
def test_bucket_stats_did_closed_form():
    rng = np.random.default_rng(1)
    x_b, y_b = rng.normal(2, 1, 40), rng.normal(1, 1, 400)
    x_r, y_r = rng.normal(0, 1, 80), rng.normal(0, 1, 800)
    s = st.bucket_stats(x_b, y_b, x_r, y_r)
    did = (x_b.mean() - y_b.mean()) - (x_r.mean() - y_r.mean())
    assert s["did"] == pytest.approx(did) and s["t_did"] > 3
    o = st.condition_overall(np.r_[x_b, x_r], np.r_[np.ones(40), np.zeros(80)], y_r, np.zeros(800),
                             np.r_[np.ones(40), np.zeros(80)])
    assert o["slope"] == pytest.approx(x_b.mean() - x_r.mean()) and o["p_kruskal"] < 0.01


# --------------------------------------------------------------------------- the model
def _synthetic(event_effect_in_up=1.0, regime_drift=0.0, seed=0):
    """NFP-like events; a condition series whose regime alternates by month; the event effect
    appears only in the 'up' regime, and/or EVERY window drifts in that regime."""
    from infra.processing import event_windows as ew
    from infra.processing.schedule_rules import business_days
    from infra.reference.event_grid import resolve_cycle
    g = ew.make_grid(resolve_cycle("DEFAULT_CYCLE"), business_days("2016-01-01", "2024-12-31", "market"))
    rng = np.random.default_rng(seed)
    pts = g.instants()
    panel = pd.DataFrame({"X": rng.normal(0, 0.2, len(pts))}, index=pd.DatetimeIndex(pts, name="timestamp"))
    days = pd.DatetimeIndex(g.days)
    level = pd.Series(np.where((days.month % 2) == 0, 1.0, -1.0) + rng.normal(0, 0.05, len(days)), index=days)
    timeline = pd.DataFrame({"label": days, "available_at": days + pd.Timedelta(hours=21), "value": level.to_numpy()})
    up_days = set(days[level.to_numpy() > 0])
    if regime_drift:   # every 08:30 step in the up regime moves (events and placebo alike)
        loc = pd.Series(pts).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        hit = ((loc.dt.hour == 8) & (loc.dt.minute == 45)).to_numpy()
        local_day = pd.DatetimeIndex(loc.dt.tz_localize(None)).normalize()
        prev = days[np.clip(days.searchsorted(local_day) - 1, 0, None)]          # the previous TRADING day
        prev_day_up = np.asarray(pd.Index(prev).isin(list(up_days)))
        panel.loc[hit & prev_day_up, "X"] += regime_drift
    from infra.trading_calendar import snap_instants
    fridays = [d for d in days if d.weekday() == 4 and d.day <= 7]
    occ = pd.DataFrame({"event": NFP, "timestamp": snap_instants(fridays, "08:30", "America/New_York"),  # 08:30 ET all year
                        "day": fridays, "stage": "", "time_source": "registry", "known_from": fridays})
    for d, t in zip(fridays, occ["timestamp"]):
        if days[days.searchsorted(d) - 1] in up_days and event_effect_in_up:
            panel.loc[t + pd.Timedelta(minutes=15), "X"] += event_effect_in_up
    return panel, occ, timeline


def _study(**cond):
    c = ConditionSpec("fut:ZN.v.0", partition="fixed:-0.5:0.5", min_bucket_obs=10, **cond)
    return EventStudy(EventStudySpec(name="c", code=f"{NFP};;;&0_0_&_-1__&0_0_&_4;;DEFAULT_CYCLE", instruments=("X",),
                                     ev_abs_min=None, condition=c))


def test_a_regime_specific_event_effect_passes_the_conditioning():
    panel, occ, tl = _synthetic(event_effect_in_up=1.0)
    m = _study()
    data = m.prepare(panel, occurrences=occ, condition_timeline=tl)
    m.fit(data, as_of="2023-12-31")
    t = m.fitted_.table
    assert t.loc["X|+1", "cond_passed"] == 1.0 and t.loc["X|+1", "t_did"] > 3
    assert t.loc["X|-1", "cond_passed"] == 0.0
    out = m.predict(data, start="2023-12-31")
    up = out["cond_bucket"] == 1.0
    assert (out.loc[up, "signal:X"] == 1.0).all() and (out.loc[~up, "signal:X"] == 0.0).all()


def test_a_regime_that_moves_every_window_is_not_an_event_effect():
    """Every window drifts in the up regime: events in that bucket look great vs the rest,
    but the placebo drifts the same - the difference-in-differences says no."""
    panel, occ, tl = _synthetic(event_effect_in_up=0.0, regime_drift=1.0)
    m = _study()
    m.fit(m.prepare(panel, occurrences=occ, condition_timeline=tl), as_of="2023-12-31")
    t = m.fitted_.table.loc["X|+1"]
    assert t["passed"] == 1.0 and abs(t["t_vs_rest"]) > 3     # the naive comparison is fooled...
    assert abs(t["t_did"]) < 1.5 and t["cond_passed"] == 0.0   # ...the DiD is not


def test_conditional_walk_forward_is_point_in_time(monkeypatch):
    from infra.models.walk_forward import walk_forward
    panel, occ, tl = _synthetic()
    cut = D("2021-06-30")
    import infra.models.event_study.model as mm
    monkeypatch.setattr(mm.EventStudy, "occurrences", lambda self, as_of=None: occ)
    timelines = {"tl": tl}
    monkeypatch.setattr(mm.EventStudy, "condition_timeline", lambda self, s, e: timelines["tl"])
    base = walk_forward(lambda: _study(), panel, "2019-01-01", "2024-12-31", refit="QE")
    shocked = tl.copy()
    shocked.loc[shocked["available_at"] > cut + pd.Timedelta(days=1), "value"] *= -1.0
    timelines["tl"] = shocked
    alt = walk_forward(lambda: _study(), panel, "2019-01-01", "2024-12-31", refit="QE")
    a = base.predictions[base.predictions.index <= cut].drop(columns="fit_as_of")
    b = alt.predictions[alt.predictions.index <= cut].drop(columns="fit_as_of")
    pd.testing.assert_frame_equal(a, b)
    assert not (base.predictions["cond_bucket"] == alt.predictions["cond_bucket"]).all()  # the shock did bite later


def test_conditional_run_incremental_equals_rebuild(tmp_path, monkeypatch):
    from infra.models import runs
    from infra.storage import model_runs as store
    import infra.models.event_study.model as mm
    panel, occ, tl = _synthetic()
    panel = panel[panel.index >= D("2023-01-01")]
    monkeypatch.setattr(mm.EventStudy, "occurrences", lambda self, as_of=None: occ)
    monkeypatch.setattr(mm.EventStudy, "condition_timeline", lambda self, s, e: tl)
    config = runs.RunConfig(name="c", kind="event_study", spec=None, series=("X",), start="2024-01-05",
                            history_start="2023-01-01", refit="ME",
                            overrides={"name": "c", "code": f"{NFP};;;&0_0_&_-1__&0_0_&_4;;DEFAULT_CYCLE",
                                       "min_obs": 5, "ev_abs_min": None,
                                       "condition": ConditionSpec("fut:ZN.v.0", partition="fixed:-0.5:0.5",
                                                                  min_bucket_obs=3)})
    days = pd.DatetimeIndex(sorted(set(panel.index.normalize())))
    last = None
    for day in list(days[days >= D("2024-01-05")][::5]) + [days[-1]]:
        prepared = config.make_model().prepare(panel[panel.index < day + pd.Timedelta(days=1)], occurrences=occ,
                                               condition_timeline=tl)
        for step in ("predict", "fit", "predict"):
            params = store.read_params("c", root=tmp_path)
            if step == "fit":
                store.append_params("c", runs.fit_due(config, prepared, params, day).params, root=tmp_path)
                continue
            after = runs.predict_window_after(params, store.read_predictions("c", root=tmp_path), last_through=last)
            store.upsert_predictions("c", runs.predict_rows(config, prepared, params, after, day), root=tmp_path)
            last = day
    full = runs.rebuild(config, panel, days[-1])
    rep = runs.reconcile(store.read_params("c", root=tmp_path), store.read_predictions("c", root=tmp_path),
                         full.params, full.predictions)
    assert rep["identical"], rep
