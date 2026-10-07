"""Directional forecasts (infra/models/forecast): the three modes on a planted effect, a wrong-signed
prior switching off, point in time, decision instants, the params round trip, incremental == rebuild,
the evaluation suite and the family's FDR gate. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.models.forecast.config import FORECAST_MODELS, Regressor, get_forecast_spec
from infra.models.forecast.evaluate import evaluate, run_family
from infra.models.forecast.model import ForecastModel

D = pd.Timestamp


def _panel(beta=0.6, n=2500, seed=0, noise_cols=0):
    """y_{t+1} = beta * x_t + e: x at row t (known by its decision) predicts the next day's move."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2012-01-02", periods=n)
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    y[1:] += beta * x[:-1]
    df = pd.DataFrame({"y": y, "x": x}, index=days)
    for i in range(noise_cols):
        df[f"n{i}"] = rng.normal(0, 1, n)
    return df


def _spec(mode="default", weight=1.0, cols=("x",), **kw):
    return get_forecast_spec(mode, regressors=tuple(Regressor(f"bmk:curve:{c}", weight=weight, name=c) for c in cols),
                             horizon_days=1, gap_days=0, **kw)


def test_three_modes_on_a_planted_effect():
    panel = _panel()
    for mode in ("default", "exante", "prior"):
        m = ForecastModel(_spec(mode))
        data = m.prepare(panel)
        m.fit(data, as_of="2018-12-31")
        out = m.predict(data, start="2018-12-31")
        assert m.fitted_.passed
        assert m.fitted_.beta[0] > 0 and (np.sign(out["signal"]) == np.sign(data.loc[out.index, "x"])).mean() > 0.95
    assert ForecastModel(_spec("default")).fit(ForecastModel(_spec()).prepare(panel), as_of="2018-12-31").fitted_.stats["t_x"] > 5


def test_a_wrong_signed_prior_switches_the_regressor_off_and_noise_fails_the_gates():
    panel = _panel()
    m = ForecastModel(_spec("prior", weight=-1.0)).fit(ForecastModel(_spec()).prepare(panel), as_of="2018-12-31")
    assert m.fitted_.beta[0] == 0.0
    noise = _panel(beta=0.0)
    f = ForecastModel(_spec("default")).fit(ForecastModel(_spec()).prepare(noise), as_of="2018-12-31")
    assert not f.fitted_.passed and (f.fitted_.beta == 0).all()


def test_point_in_time_and_params_round_trip():
    panel = _panel()
    cut = D("2016-06-30")
    shocked = panel.copy()
    shocked.loc[shocked.index > cut, "y"] *= 10
    a = ForecastModel(_spec()).fit(ForecastModel(_spec()).prepare(panel), as_of=cut)
    b = ForecastModel(_spec()).fit(ForecastModel(_spec()).prepare(shocked), as_of=cut)
    assert a.fitted_.stats["coef_x"] == b.fitted_.stats["coef_x"]
    r = ForecastModel.from_params(a.params(), _spec())
    data = r.prepare(panel)
    pd.testing.assert_frame_equal(r.predict(data, start=cut), a.predict(data, start=cut))


def test_decision_instants_are_new_york_time_on_the_decision_day():
    m = ForecastModel(get_forecast_spec("default", gap_days=1, decision_time="15:00"))
    days = pd.DatetimeIndex([D("2024-07-01"), D("2024-07-02"), D("2024-12-02"), D("2024-12-03")])
    dec = m.decision_instants(days)
    assert dec[0] == D("2024-07-02 19:00") and dec[2] == D("2024-12-03 20:00")      # EDT / EST
    assert dec[-1] == D("2024-12-04 20:00")                                          # past the data: next business day


def test_read_panel_aligns_regressors_with_limit_and_fill(monkeypatch):
    import infra.pipeline.features as pf
    import infra.pipeline.series_panel as sp
    days = pd.bdate_range("2024-03-01", periods=10)
    monkeypatch.setattr(sp, "read_panel", lambda ids, s, e: pd.DataFrame({ids[0]: np.ones(10)}, index=days))
    surprise = pd.Series([2.0], index=[D("2024-03-05 13:30")])                     # one release, 08:30 New York
    monkeypatch.setattr(pf, "feature", lambda expr, s, e: surprise)
    spec = get_forecast_spec("default", target="bmk:curve:T", gap_days=0,
                             regressors=(Regressor("surprise:X", limit="2D", fill=0.0, name="s"),))
    p = ForecastModel(spec).read_panel("2024-03-01", "2024-03-14")
    assert p.loc[D("2024-03-04"), "s"] == 0.0 and p.loc[D("2024-03-05"), "s"] == 2.0  # known by 15:00 New York
    assert p.loc[D("2024-03-06"), "s"] == 2.0 and p.loc[D("2024-03-07"), "s"] == 0.0  # within 2 days, then "no news"


def test_forecast_run_incremental_equals_rebuild(tmp_path, monkeypatch):
    from infra.models import runs
    from infra.storage import model_runs as store
    panel = _panel(n=1800)
    monkeypatch.setitem(FORECAST_MODELS, "t_fc", _spec("default", name="t_fc"))
    config = runs.RunConfig(name="fc", kind="forecast", spec="t_fc", series=("bmk:curve:T",), start="2015-01-02",
                            history_start="2012-01-02", refit="ME", overrides={"target": "bmk:curve:T"})
    days = panel.index
    for day in list(days[days >= D("2015-01-02")][::13]) + [days[-1]]:
        prepared = config.make_model().prepare(panel[panel.index <= day])
        for step in ("predict", "fit", "predict"):
            params = store.read_params("fc", root=tmp_path)
            if step == "fit":
                store.append_params("fc", runs.fit_due(config, prepared, params, day).params, root=tmp_path)
                continue
            after = runs.predict_window_after(params, store.read_predictions("fc", root=tmp_path))
            store.upsert_predictions("fc", runs.predict_rows(config, prepared, params, after, day), root=tmp_path)
    full = runs.rebuild(config, panel, days[-1])
    rep = runs.reconcile(store.read_params("fc", root=tmp_path), store.read_predictions("fc", root=tmp_path),
                         full.params, full.predictions)
    assert rep["identical"], rep


def test_evaluation_and_family_gate():
    panel = _panel(n=2200, noise_cols=5)
    r = evaluate(_spec("exante", n_placebo=4), panel, "2014-01-02", panel.index[-1], refit="QE",
                 evaluations=("benchmark", "clark_west", "spanning", "permutation", "time_shift"))
    assert r["benchmark"]["sharpe_forecast"] > 2 and r["spanning"]["alpha_t"] > 3
    assert r["permutation"]["p_placebo_ge_real"] == 0.0 and r["time_shift"]["placebo_alpha_t"] < 2
    members = {c: _spec("default", cols=(c,)) for c in ["x"] + [f"n{i}" for i in range(5)]}
    fam = run_family(members, lambda s: panel, "2014-01-02", panel.index[-1], refit="QE")
    s = fam["summary"]
    # noise: on average rarely through; one chance member can persist (expanding-window fits are autocorrelated)
    assert s.loc["x", "fdr_share"] > 0.9 and s.drop(index="x")["fdr_share"].mean() < 0.2
    assert fam["fits"].groupby("fit_as_of")["q"].count().min() == 6


def test_legacy_mode_name_reads_as_fit_mode():
    assert get_forecast_spec("default", mode="prior").fit_mode == "prior"
    with pytest.raises(ValueError):
        get_forecast_spec("default", fit_mode="guess")
