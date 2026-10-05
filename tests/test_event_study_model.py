"""Event studies (infra/processing/event_windows.py, infra/pipeline/event_pnl.py,
infra/models/event_study): codes, grid and interval rules, pairing, P&L, tests, the
walk-forward and run machinery, families. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.models.event_study import stats as st
from infra.models.event_study.config import EventStudySpec, FamilySpec, LegRule
from infra.models.event_study.family import benjamini_hochberg, run_family
from infra.models.event_study.model import EventStudy, window_moves
from infra.processing import event_windows as ew
from infra.processing.schedule_rules import business_days
from infra.reference.event_grid import resolve_cycle

D = pd.Timestamp
CYC = resolve_cycle("DEFAULT_CYCLE")
NFP = "US_EMPLOYMENT_SITUATION"


def _grid(start="2024-01-01", end="2024-12-31"):
    return ew.make_grid(CYC, business_days(start, end, "market"))


def _occ(event, instants_utc, time_source="registry", stage="", known_from=None):
    ts = pd.DatetimeIndex(instants_utc)
    return pd.DataFrame({"event": event, "timestamp": ts, "day": ts.normalize(), "stage": stage,
                         "time_source": time_source,
                         "known_from": ts.normalize() if known_from is None else pd.DatetimeIndex(known_from)})


def _resolve(code, occ, grid=None, **kw):
    return ew.resolve(ew.parse_code(code), occ, grid or _grid(), **kw)


# --------------------------------------------------------------------------- codes
def test_code_round_trip_case_and_empty_time_refs():
    c = ew.parse_code("us_tsy_auction_3y__US_TSY_AUCTION_7Y;GRID_START;;&0_0_&_-2__&1_1_%0_2;;DEFAULT_CYCLE")
    assert c.dt_refs[0] == ("US_TSY_AUCTION_3Y", None) and c.end.time_ref == 0 and c.start.step_lag == -2
    assert ew.parse_code(c.encode()) == c
    n = ew.parse_code(f"{NFP};;;&0_0_&_-1__&0_0_&_8;;DEFAULT_CYCLE")
    assert n.time_refs == () and n.encode() == f"{NFP};;;&0_0_&_-1__&0_0_&_8;;DEFAULT_CYCLE"
    s = ew.parse_code("US_GDP:advance;;;&0_0_&_0__&0_0_&_4;;DEFAULT_CYCLE")
    assert s.dt_refs == (("US_GDP", "advance"),)


@pytest.mark.parametrize("bad", ["NOPE;;;&0_0_&_0__&0_0_&_1;;DEFAULT_CYCLE",
                                 f"{NFP};;;&1_0_&_0__&0_0_&_1;;DEFAULT_CYCLE",
                                 f"{NFP};;;&0_0_%0_0__&0_0_&_1;;DEFAULT_CYCLE",
                                 f"{NFP};;;&0_0_&_0__&0_0_&_1;;NO_SUCH_CYCLE",
                                 f"{NFP};;&0_0_&_0;;DEFAULT_CYCLE"])
def test_bad_codes_are_rejected(bad):
    with pytest.raises((ValueError, KeyError)):
        ew.parse_code(bad)


# --------------------------------------------------------------------------- interval rules
def test_grid_has_inclusive_end_points():
    g = _grid("2024-07-01", "2024-07-01")
    pts = g.instants()
    assert len(pts) == 53 and pts[0] == D("2024-07-01 10:00") and pts[-1] == D("2024-07-01 23:00")  # 06:00/19:00 EDT


def test_start_at_the_first_point_and_end_at_the_last_are_legal():
    occ = _occ(NFP, ["2024-07-05 12:30"])  # 08:30 EDT
    w = _resolve(f"{NFP};GRID_START__GRID_END;;&0_0_%0_0__&0_0_%1_0;;DEFAULT_CYCLE", occ)
    assert w["legal"].all() and w["start"].iloc[0] == D("2024-07-05 10:00") and w["end"].iloc[0] == D("2024-07-05 23:00")


@pytest.mark.parametrize("when,reason", [("2024-07-05 09:59", "outside_grid"),   # 05:59 EDT
                                         ("2024-07-05 23:01", "outside_grid"),   # 19:01: not snapped back
                                         ("2024-07-05 23:00", "")])              # 19:00 is a point
def test_reference_time_outside_the_grid_is_illegal(when, reason):
    occ = _occ(NFP, [when])
    w = _resolve(f"{NFP};GRID_START;;&0_0_%0_0__&0_0_&_0;;DEFAULT_CYCLE", occ)
    assert w["reason"].iloc[0] == reason


def test_a_time_between_points_snaps_back_and_lags_count_steps():
    occ = _occ(NFP, ["2024-07-05 12:31"])  # 08:31 EDT -> 08:30
    w = _resolve(f"{NFP};;;&0_0_&_-1__&0_0_&_2;;DEFAULT_CYCLE", occ)
    assert w["start"].iloc[0] == D("2024-07-05 12:15") and w["end"].iloc[0] == D("2024-07-05 13:00")


def test_lags_never_leave_the_day_and_end_must_follow_start():
    occ = _occ(NFP, ["2024-07-05 12:30"])
    assert _resolve(f"{NFP};GRID_END;;&0_0_&_0__&0_0_%0_1;;DEFAULT_CYCLE", occ)["reason"].iloc[0] == "lag_outside_grid"
    assert _resolve(f"{NFP};;;&0_0_&_0__&0_0_&_0;;DEFAULT_CYCLE", occ)["reason"].iloc[0] == "end_not_after_start"


def test_day_lags_count_trading_days_and_the_own_time_travels():
    occ = _occ(NFP, ["2024-07-05 12:30"])  # Friday 08:30
    w = _resolve(f"{NFP};;;&0_0_&_0__&0_1_&_0;;DEFAULT_CYCLE", occ)
    assert w["end"].iloc[0] == D("2024-07-08 12:30")  # Monday 08:30
    sat = _occ(NFP, ["2024-07-06 12:30"])
    assert _resolve(f"{NFP};;;&0_0_&_0__&0_0_&_2;;DEFAULT_CYCLE", sat)["reason"].iloc[0] == "not_a_trading_day"


def test_own_time_needs_a_verified_time():
    occ = _occ("US_TSY_ISSUE_10Y", ["2024-08-15 04:00"], time_source="unknown")
    w = _resolve("US_TSY_ISSUE_10Y;GRID_START__GRID_END;;&0_0_&_0__&0_0_%1_0;;DEFAULT_CYCLE", occ)
    assert w["reason"].iloc[0] == "no_event_time"
    w2 = _resolve("US_TSY_ISSUE_10Y;GRID_START__GRID_END;;&0_0_%0_0__&0_0_%1_0;;DEFAULT_CYCLE", occ)
    assert w2["legal"].iloc[0]


def test_pairing_takes_the_next_occurrence_within_the_gap():
    occ = pd.concat([_occ("US_TSY_AUCTION_3Y", ["2024-07-09 17:00", "2024-08-06 17:00"]),
                     _occ("US_TSY_AUCTION_7Y", ["2024-07-25 17:00"])])
    code = "US_TSY_AUCTION_3Y__US_TSY_AUCTION_7Y;;;&0_0_&_0__&1_0_&_1;;DEFAULT_CYCLE"
    w = _resolve(code, occ, max_gap=22)
    assert w["legal"].tolist() == [True, False] and w["reason"].iloc[1] == "no_pair"
    assert w["end"].iloc[0] == D("2024-07-25 17:15")
    assert _resolve(code, occ, max_gap=5)["reason"].iloc[0] == "no_pair"


def test_placebo_windows_avoid_event_days():
    occ = _occ(NFP, ["2024-07-05 12:30", "2024-08-02 12:30"])
    g = _grid()
    w = ew.resolve(ew.parse_code(f"{NFP};;;&0_0_&_0__&0_0_&_4;;DEFAULT_CYCLE"), occ, g)
    p = ew.placebo(w, g)
    assert len(p) == len(g.days) - 2 and not set(p["start_day"]) & {D("2024-07-05"), D("2024-08-02")}
    assert (p["end"] - p["start"] == pd.Timedelta(hours=1)).all()


def test_consolidate_one_occurrence_per_event_day():
    cal = pd.DataFrame({
        "timestamp": [D("2024-07-05 12:30"), D("2024-07-05 12:30"), D("2024-07-05 04:00"), D("2024-08-02 12:30"),
                      D("2025-01-10 13:30")],
        "event": NFP, "source": ["fred_release_dates", "marketwatch", "marketwatch_unconfirmed", "rule", "rule"],
        "stage": ["", "", "", "", ""], "time_source": ["registry", "source", "source", "registry", "registry"],
        "known_from": [D("2024-07-05"), D("2024-06-28"), D("2024-06-01"), D("2024-07-01"), D("2024-12-01")],
        "last_seen": D("2024-12-31")})
    occ = ew.consolidate(cal.iloc[[0, 1, 2]])
    assert len(occ) == 1 and occ["time_source"].iloc[0] == "source" and occ["known_from"].iloc[0] == D("2024-06-28")
    assert len(ew.consolidate(cal)) == 3  # rule-only days are kept (projections)


# --------------------------------------------------------------------------- moves
def test_window_moves_sum_the_half_open_interval():
    idx = pd.date_range("2024-01-02 11:00", periods=5, freq="15min")
    panel = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0, np.nan]}, index=idx)
    pnl, gaps = window_moves(panel, [idx[0], idx[1], idx[3]], [idx[2], idx[3], idx[4]])
    assert pnl["a"].tolist()[:2] == [6.0, 12.0] and np.isnan(pnl["a"].iloc[2]) and gaps["a"].iloc[2] == 1


def test_bbo_source_measures_each_step_on_one_contract_and_carries_the_halt(monkeypatch):
    from infra.pipeline import event_pnl as ep
    g = _grid("2024-03-14", "2024-03-15")   # Thursday, Friday
    pts = g.instants()
    minutes = pd.date_range(pts.min() - pd.Timedelta(hours=1), pts.max(), freq="1min")
    quotes = pd.concat([pd.DataFrame({"timestamp": minutes, "ticker": t, "bid": base + np.arange(len(minutes)) * 1e-3,
                                      "ask": base + np.arange(len(minutes)) * 1e-3 + 0.01})
                        for t, base in (("ZNM4", 110.0), ("ZNU4", 109.0))])
    quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
    # CME halt 16:00-17:00 CT: no quotes there
    loc = minutes.tz_localize("UTC").tz_convert("America/Chicago")
    halted = (loc.hour == 16)
    quotes = quotes[~np.tile(halted, 2)]
    monkeypatch.setattr(ep, "read_bbo_from_disk", lambda tickers, start, end, root=None: quotes)
    monkeypatch.setattr(ep, "contract_map", lambda inst, days, **kw: pd.Series(
        ["ZNM4" if d < D("2024-03-15") else "ZNU4" for d in days], index=days))
    monkeypatch.setattr(ep, "dv01_prior", lambda con, days, risk_root=None: np.full(len(con), 80.0))
    pts_pnl = ep.grid_pnl("FUTURE_PTS_BBO", ["ZN.v.0"], g)["ZN.v.0"]
    step = pd.Timedelta(minutes=15) / pd.Timedelta(minutes=1) * 1e-3
    intraday = pts_pnl[~pts_pnl.index.isin(g.first)]
    v = intraday.dropna().to_numpy()
    assert np.all(np.isclose(v, step) | np.isclose(v, 0.0) | (v > 0))  # a step's move, 0 in the halt
    # the roll (Thursday 17:00 CT session -> trade date Friday, ZNU4): measured on ZNU4 only - no 1-point jump
    assert intraday.abs().max() < 0.5  # the contracts differ by 1 point: a cross-contract step would show it
    # inside the halt the price is the close's: steps are 0
    halt_pts = pts_pnl.index[(pts_pnl.index.tz_localize("UTC").tz_convert("America/Chicago").hour == 16)
                             & (pts_pnl.index.tz_localize("UTC").tz_convert("America/Chicago").minute > 0)]
    assert (pts_pnl.loc[halt_pts].dropna() == 0).all() and len(pts_pnl.loc[halt_pts].dropna())
    bps = ep.grid_pnl("FUTURE_BPS_BBO", ["ZN.v.0"], g)["ZN.v.0"]
    assert np.allclose(bps.dropna(), pts_pnl.dropna() * 1000 / 80.0)  # x point value / DV01


# --------------------------------------------------------------------------- statistics
def test_event_stats_closed_forms():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 2.0, 60)
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    s = st.event_stats(x, dates, rng.normal(0, 2, 500))
    assert s["t"] == pytest.approx(x.mean() / (x.std(ddof=1) / np.sqrt(60)))
    assert s["hit"] == pytest.approx(np.mean(x > 0))
    assert 0 < s["p_hit"] < 1 and s["ev_vol"] == pytest.approx(x.mean() / s["placebo_std"])
    spec = EventStudySpec(min_obs=50, t_min=1.5, ev_abs_min=0.5, hit_min=0.5, stable_halves=True)
    t = st.passes(s, spec)
    assert t["passed"] == 1.0
    assert st.passes(s, EventStudySpec(min_obs=100))["passed"] == 0.0


def test_benjamini_hochberg():
    # p sorted 0.01, 0.03, 0.04, 0.5 -> raw p*m/rank 0.04, 0.06, 0.0533, 0.5 -> monotone 0.04, 0.0533, 0.0533, 0.5
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5, np.nan]))
    np.testing.assert_allclose(q[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert np.isnan(q[4])
    p = np.array([0.01, 0.02, 0.03, 0.5])
    np.testing.assert_allclose(benjamini_hochberg(p), [0.04, 0.04, 0.04, 0.5])
    assert np.isnan(benjamini_hochberg(np.array([np.nan]))[0])


# --------------------------------------------------------------------------- the model
def _synthetic(effect=1.0, seed=0, start="2018-01-01", end="2024-12-31"):
    g = _grid(start, end)
    rng = np.random.default_rng(seed)
    pts = g.instants()
    panel = pd.DataFrame({"X": rng.normal(0, 0.2, len(pts))}, index=pd.DatetimeIndex(pts, name="timestamp"))
    first_fridays = [d for d in g.days if d.weekday() == 4 and d.day <= 7]
    occ = _occ(NFP, [pd.Timestamp(d) + pd.Timedelta(hours=12, minutes=30) for d in first_fridays])
    for t in occ["timestamp"]:
        nxt = t + pd.Timedelta(minutes=15)
        if nxt in panel.index:
            panel.loc[nxt, "X"] += effect
    return panel, occ


def _study(**kw):
    return EventStudy(EventStudySpec(name="t", code=f"{NFP};;;&0_0_&_-1__&0_0_&_4;;DEFAULT_CYCLE",
                                     instruments=("X",), ev_abs_min=0.5, **kw))


def test_a_planted_effect_is_found_and_the_placebo_is_flat():
    panel, occ = _synthetic(effect=1.0)
    m = _study()
    data = m.prepare(panel, occurrences=occ)
    m.fit(data, as_of="2023-12-31")
    t = m.fitted_.table.loc["X"]
    assert t["passed"] == 1.0 and t["mean"] == pytest.approx(1.0, abs=0.15) and abs(t["placebo_mean"]) < 0.1
    out = m.predict(data, start="2023-12-31")
    assert (out["signal:X"] == 1.0).all() and not out["in_sample"].any()
    paths = m.paths(panel, data)
    assert set(paths["step"]) == {1, 2, 3, 4, 5}


def test_no_effect_fails():
    panel, occ = _synthetic(effect=0.0)
    m = _study()
    m.fit(m.prepare(panel, occurrences=occ), as_of="2023-12-31")
    assert m.fitted_.table.loc["X", "passed"] == 0.0


def test_event_study_walk_forward_is_point_in_time():
    from infra.models.walk_forward import walk_forward
    panel, occ = _synthetic()
    cut = D("2022-06-30")
    fac = lambda: _study()  # noqa: E731
    import infra.models.event_study.model as mm
    orig = mm.EventStudy.occurrences
    mm.EventStudy.occurrences = lambda self, as_of=None: occ
    try:
        base = walk_forward(fac, panel, "2020-01-01", "2024-12-31", refit="QE")
        shocked = panel.copy()
        shocked.loc[shocked.index > cut + pd.Timedelta(days=1)] += 5.0
        alt = walk_forward(fac, shocked, "2020-01-01", "2024-12-31", refit="QE")
    finally:
        mm.EventStudy.occurrences = orig
    a = base.predictions[base.predictions.index <= cut].drop(columns="fit_as_of")
    b = alt.predictions[alt.predictions.index <= cut].drop(columns="fit_as_of")
    pd.testing.assert_frame_equal(a, b)
    pa = base.params[base.params["fit_as_of"] <= cut].reset_index(drop=True)
    pb = alt.params[alt.params["fit_as_of"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(pa, pb)


def test_event_study_incremental_run_equals_rebuild(tmp_path, monkeypatch):
    from infra.models import runs
    from infra.storage import model_runs as store
    import infra.models.event_study.model as mm
    panel, occ = _synthetic(start="2023-01-01", end="2024-12-31")
    monkeypatch.setattr(mm.EventStudy, "occurrences", lambda self, as_of=None: occ)
    config = runs.RunConfig(name="es", kind="event_study", spec=None, series=("X",), start="2024-01-05",
                            history_start="2023-01-01", refit="ME",
                            overrides={"code": f"{NFP};;;&0_0_&_-1__&0_0_&_4;;DEFAULT_CYCLE", "min_obs": 5,
                                       "ev_abs_min": None, "name": "es"})
    days = pd.DatetimeIndex(sorted(set(panel.index.normalize())))
    last = None
    for day in list(days[(days >= D("2024-01-05"))][::3]) + [days[-1]]:
        live = panel[panel.index < day + pd.Timedelta(days=1)]
        prepared = config.make_model().prepare(live, occurrences=occ)
        for step in ("predict", "fit", "predict"):
            params = store.read_params("es", root=tmp_path)
            if step == "fit":
                store.append_params("es", runs.fit_due(config, prepared, params, day).params, root=tmp_path)
                continue
            after = runs.predict_window_after(params, store.read_predictions("es", root=tmp_path), last_through=last)
            store.upsert_predictions("es", runs.predict_rows(config, prepared, params, after, day), root=tmp_path)
            last = day
    full = runs.rebuild(config, panel[panel.index < days[-1] + pd.Timedelta(days=1)], days[-1])
    rep = runs.reconcile(store.read_params("es", root=tmp_path), store.read_predictions("es", root=tmp_path),
                         full.params, full.predictions)
    assert rep["identical"], rep


def test_family_generates_codes_keeps_illegal_ones_and_controls_fdr():
    panel, occ = _synthetic(effect=1.0)
    fam = FamilySpec("f", (NFP,), ("GRID_END",), start=LegRule(step_lags=(-1, 0)),
                     end=LegRule(times=("&", "%0"), step_lags=(0, 1, 4)),
                     study=EventStudySpec(instruments=("X",), ev_abs_min=0.5))
    assert len(fam.codes()) == 2 * 2 * 3
    r = run_family(fam, panel, as_of="2023-12-31", occurrences=occ)
    assert r["code"].nunique() == 12
    assert (r.loc[r["legal_windows"] == 0, "n"] == 0).all() and (r["illegal_reasons"] != "").any()
    hit = r[r["code"].str.contains(r"&0_0_&_-1__&0_0_&_4")]
    assert hit["passed_fdr"].iloc[0] == 1.0
    assert r["q"].notna().any()
