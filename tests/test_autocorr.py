"""Conditional autocorrelation (infra/models/autocorr): a planted X-dependent autocorrelation is
found and traded, no effect fails the gates, gates are swappable, point in time, params round
trip, the evaluation suite. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.models.autocorr.config import get_autocorr_spec
from infra.models.autocorr.evaluate import evaluate
from infra.models.autocorr.gates import GATES
from infra.models.autocorr.model import ConditionalAutocorr

D = pd.Timestamp


def _panel(rho=0.35, n=3000, seed=0):
    """y_t = rho * sign(X_{t-1}) * y_{t-1} + e: momentum when X > 0, reversal when X < 0."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-04", periods=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.98 * x[t - 1] + rng.normal(0, 0.2)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = (rho * np.sign(x[t - 1]) * y[t - 1] if rho else 0.0) + rng.normal(0, 1.0)
    return pd.DataFrame({"T": y, "X": x}, index=days)


def _spec(name="default", **kw):
    return get_autocorr_spec(name, target="T", x="X", x_is_moves=False, x_feature="level", past_days=1,
                             horizon_days=1, gap_days=0, x_partition="rolling_tercile:250", **kw)


def test_a_planted_conditional_autocorrelation_is_found_and_traded():
    panel = _panel()
    m = ConditionalAutocorr(_spec())
    data = m.prepare(panel)
    m.fit(data, as_of="2018-12-31")
    f = m.fitted_
    assert f.passed and f.stats["x_t"] > 3 and f.stats["spread_t"] > 3
    assert f.signal_map[1.0] == 1.0 and f.signal_map[-1.0] == -1.0      # chase when X high, fade when low
    out = m.predict(data, start="2018-12-31")
    row = out[(out.bucket == 1.0) & (out.past > 0)].iloc[0]
    assert row.signal == 1.0 and row.expected_bp > 0


def test_no_effect_fails_strict_gates_and_falls_back():
    data = ConditionalAutocorr(_spec()).prepare(_panel(rho=0.0))
    m = ConditionalAutocorr(_spec()).fit(data, as_of="2018-12-31")
    assert not m.fitted_.passed and set(m.fitted_.signal_map.values()) == {0.0}
    mb = ConditionalAutocorr(_spec(fallback="benchmark")).fit(data, as_of="2018-12-31")
    assert set(mb.fitted_.signal_map.values()) == {mb.fitted_.bench_sign}


def test_gates_are_swappable():
    data = ConditionalAutocorr(_spec()).prepare(_panel(rho=0.0))
    loose = ConditionalAutocorr(_spec("none")).fit(data, as_of="2018-12-31")      # min_obs only
    assert loose.fitted_.passed and list(loose.fitted_.gates) == ["min_obs"]
    with pytest.raises(KeyError):
        ConditionalAutocorr(_spec(gates=("nope",))).fit(data, as_of="2018-12-31")
    assert {"min_obs", "x_t", "bucket_spread_t", "beats_controls", "halves_agree", "monotonic"} <= set(GATES)


def test_fit_is_point_in_time_and_params_round_trip():
    panel = _panel()
    cut = D("2015-06-30")
    shocked = panel.copy()
    shocked.loc[shocked.index > cut, "T"] *= 5.0
    a = ConditionalAutocorr(_spec()).fit(ConditionalAutocorr(_spec()).prepare(panel), as_of=cut)
    b = ConditionalAutocorr(_spec()).fit(ConditionalAutocorr(_spec()).prepare(shocked), as_of=cut)
    assert a.fitted_.stats == b.fitted_.stats
    r = ConditionalAutocorr.from_params(a.params(), _spec())
    data = r.prepare(panel)
    pd.testing.assert_frame_equal(r.predict(data, start=cut), a.predict(data, start=cut))


def test_walk_forward_run_incremental_equals_rebuild(tmp_path, monkeypatch):
    from infra.models import runs
    from infra.storage import model_runs as store
    panel = _panel(n=2200)
    config = runs.RunConfig(name="ac", kind="autocorr", spec="default", series=("T", "X"), start="2014-01-03",
                            history_start="2010-01-04", refit="ME",
                            overrides={"x_is_moves": False, "x_feature": "level", "past_days": 1, "horizon_days": 1,
                                       "gap_days": 0, "x_partition": "rolling_tercile:250"})
    days = pd.DatetimeIndex(sorted(set(panel.index)))
    for day in list(days[days >= D("2014-01-03")][::11]) + [days[-1]]:
        live = panel[panel.index <= day]
        prepared = config.make_model().prepare(live)
        for step in ("predict", "fit", "predict"):
            params = store.read_params("ac", root=tmp_path)
            if step == "fit":
                store.append_params("ac", runs.fit_due(config, prepared, params, day).params, root=tmp_path)
                continue
            after = runs.predict_window_after(params, store.read_predictions("ac", root=tmp_path))
            store.upsert_predictions("ac", runs.predict_rows(config, prepared, params, after, day), root=tmp_path)
    full = runs.rebuild(config, panel, days[-1])
    rep = runs.reconcile(store.read_params("ac", root=tmp_path), store.read_predictions("ac", root=tmp_path),
                         full.params, full.predictions)
    assert rep["identical"], rep


def test_evaluation_suite_on_a_planted_effect():
    panel = _panel(n=2600)
    spec = _spec(n_placebo=4)
    r = evaluate(spec, panel, "2013-01-02", panel.index[-1], refit="QE",
                 evaluations=("benchmark", "clark_west", "spanning", "sharpe_diff", "subperiods", "time_shift",
                              "permutation"))
    assert r["benchmark"]["sharpe_conditional"] > r["benchmark"]["sharpe_benchmark"] + 0.5
    assert r["spanning"]["alpha_t"] > 3 and r["clark_west"]["clark_west_t"] > 2
    assert r["permutation"]["p_placebo_ge_real"] <= 0.25 and r["time_shift"]["placebo_alpha_t"] < r["spanning"]["alpha_t"]


def test_family_fdr_gate_is_per_fit_date_and_stricter_than_own_gates(monkeypatch):
    from infra.models.autocorr import family as fam_mod
    from infra.models.autocorr.family import AutocorrFamilySpec, run_family
    panel = _panel(n=2400)
    rng = np.random.default_rng(5)
    for i in range(6):                                    # six noise X's next to the real one
        panel[f"N{i}"] = np.cumsum(rng.normal(0, 0.2, len(panel)))
    fam = AutocorrFamilySpec("t", pairs=(("T", "X"),) + tuple(("T", f"N{i}") for i in range(6)),
                             x_features=("level",), past_days=(1,), horizon_days=(1,), base="loose")
    monkeypatch.setattr(fam_mod, "get_autocorr_spec",
                        lambda b: _spec(b))                # members inherit the synthetic data settings
    r = run_family(fam, panel, "2013-01-02", panel.index[-1], refit="QE")
    f = r["fits"]
    assert f.groupby("fit_as_of")["q"].count().min() == 7          # one BH set per fit date, all members
    assert (f["passed"] <= f["own_pass"]).all()                      # FDR only removes
    s = r["summary"]
    assert s.loc["T|X|level|k1|h1", "fdr_share"] > 0.8 and s.loc["T|X|level|k1|h1", "alpha_t_fdr"] > 3
    noise = s.drop(index="T|X|level|k1|h1")
    assert noise["fdr_share"].mean() <= noise["own_share"].mean()


def test_cells_mode_finds_an_asymmetric_effect_chase_cannot():
    """fwd = +a only when X > 0 and the past move was DOWN (a counter-move that continues... up):
    symmetric chase averages it away, the 3x3 cells isolate it."""
    rng = np.random.default_rng(3)
    n = 3000
    days = pd.bdate_range("2010-01-04", periods=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.98 * x[t - 1] + rng.normal(0, 0.2)
    y = rng.normal(0, 1.0, n)
    for t in range(1, n):
        if x[t - 1] > 0.3 and y[t - 1] < -0.43:
            y[t] += 0.8
    panel = pd.DataFrame({"T": y, "X": x}, index=days)
    cells = ConditionalAutocorr(_spec("cells", past_partition="rolling_tercile:250")).fit(
        ConditionalAutocorr(_spec("cells", past_partition="rolling_tercile:250")).prepare(panel), as_of="2018-12-31")
    assert cells.fitted_.signal_map["1|-1"] == 1.0 and cells.fitted_.stats["cell_t_1|-1"] > 3
    assert sum(v != 0 for v in cells.fitted_.signal_map.values()) <= 3
    out = cells.predict(cells.prepare(panel), start="2018-12-31")
    assert (out.loc[(out.bucket == 1) & (out.pbucket == -1), "signal"] == 1.0).all()


def test_absmove_feature_is_the_size():
    m = ConditionalAutocorr(get_autocorr_spec("default", target="T", x="X", x_feature="absmove:5"))
    xs = pd.Series(np.r_[np.zeros(40), -np.ones(20)])
    assert (m._x_feature(xs).dropna() >= 0).all()


def test_cells_null_is_the_unconditional_mean_not_zero():
    """A target that merely drifts: no cell differs from the drift, so none may pass."""
    rng = np.random.default_rng(9)
    n = 3000
    days = pd.bdate_range("2010-01-04", periods=n)
    x = np.cumsum(rng.normal(0, 0.2, n))
    panel = pd.DataFrame({"T": rng.normal(0.25, 1.0, n), "X": x}, index=days)          # strong drift, no conditioning
    spec = _spec("cells", past_partition="rolling_tercile:250")
    m = ConditionalAutocorr(spec).fit(ConditionalAutocorr(spec).prepare(panel), as_of="2020-12-31")
    assert sum(v != 0 for v in m.fitted_.signal_map.values()) <= 1                    # ~5% false positives at most
    assert all(m.fitted_.stats[f"cell_mean_{k}"] > 0 for k in ("1|1", "-1|-1"))       # raw means carry the drift


def test_fixed_map_is_a_no_fit_rule():
    panel = _panel(rho=0.0)                               # no effect: a fixed rule still trades, by construction
    spec = _spec("fade_high_chase_low")
    m = ConditionalAutocorr(spec).fit(ConditionalAutocorr(spec).prepare(panel), as_of="2015-12-31")
    assert m.fitted_.passed and m.fitted_.signal_map == {1.0: -1.0, 0.0: 0.0, -1.0: 1.0}
    out = m.predict(m.prepare(panel), start="2015-12-31")
    hi = out[(out.bucket == 1) & (out.past > 0)]
    assert (hi.signal == -1.0).all() and (out.loc[out.bucket == 0, "signal"] == 0).all()
    mirror = ConditionalAutocorr(_spec("chase_high_fade_low")).fit(m.prepare(panel), as_of="2015-12-31")
    pd.testing.assert_series_equal(mirror.predict(m.prepare(panel), start="2015-12-31")["signal"], -out["signal"])


def test_fit_modes_prior_and_legacy_names():
    """prior keeps the fitted map only where it agrees with the stated one; exante works in cells form;
    the legacy names (mode = the form, a bare fixed_map = exante) still build the same spec."""
    from infra.models.autocorr.config import get_autocorr_spec
    panel = _panel(rho=0.35)
    fitted = ConditionalAutocorr(_spec("none")).fit(ConditionalAutocorr(_spec("none")).prepare(panel), as_of="2015-12-31")
    fm = fitted.fitted_.signal_map
    stated = ((1.0, fm[1.0]), (0.0, 0.0), (-1.0, -fm[-1.0]))                          # agree on +, disagree on -
    pr = ConditionalAutocorr(_spec("none", fit_mode="prior", fixed_map=stated)).fit(fitted.prepare(panel), as_of="2015-12-31")
    assert pr.fitted_.signal_map == {1.0: fm[1.0], 0.0: 0.0, -1.0: 0.0}
    cells = (("1|1", 1.0), ("-1|-1", -1.0))
    ex = ConditionalAutocorr(_spec("cells", fit_mode="exante", fixed_map=cells, gates=()))
    ex.fit(ex.prepare(panel), as_of="2015-12-31")
    assert ex.fitted_.signal_map == {"1|1": 1.0, "-1|-1": -1.0}
    out = ex.predict(ex.prepare(panel), start="2015-12-31")
    assert set(out["signal"].unique()) <= {-1.0, 0.0, 1.0} and (out["signal"] != 0).any()
    assert get_autocorr_spec("default", mode="cells").form == "cells"
    assert get_autocorr_spec("default", fixed_map=((1.0, 1.0),), gates=()).fit_mode == "exante"
    with pytest.raises(ValueError):
        get_autocorr_spec("default", fixed_map=((1.0, 1.0),), fit_mode="fitted")
