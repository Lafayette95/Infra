"""infra/models/stats (regressions, PCA, diagnostics), infra/models/prep, the generic
walk-forward and the series panel. Synthetic data only; every estimator is checked
against a closed form or a known data-generating process."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.models.prep import Preprocessor, resample
from infra.models.stats import diagnostics as dg
from infra.models.stats import inference as inf
from infra.models.stats.common import pick
from infra.models.stats.pca import PCA, make_pca, ppca_em
from infra.models.stats.regression import Regression, make_regression
from infra.models.walk_forward import refit_dates, walk_forward

IDX = pd.bdate_range("2020-01-01", periods=600)


def _linear(seed=0, n=600):
    rng = np.random.default_rng(seed)
    x1, x3 = rng.normal(size=n), rng.normal(size=n)
    x2 = rng.normal(size=n) + 0.5 * x1
    y = 1 + 2 * x1 - 0.5 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3}, index=IDX[:n])


def _factor_panel(seed=1, n=600, p=6, k=2, noise=0.3):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(p, k))
    X = rng.normal(size=(n, k)) @ W.T + noise * rng.normal(size=(n, p))
    return pd.DataFrame(X, index=IDX[:n], columns=[f"c{i}" for i in range(p)])


# --------------------------------------------------------------------------- prep
def test_resample_labels_each_bucket_at_its_end():
    s = pd.DataFrame({"a": np.arange(10.0)}, index=pd.bdate_range("2024-01-01", periods=10))
    w = resample(s, "W-FRI")
    assert list(w.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]
    assert w["a"].tolist() == [4.0, 9.0]  # the Friday's own value, never a later one


def test_stateful_scaling_is_frozen_at_fit_and_inverted():
    df = pd.DataFrame({"a": np.arange(20.0)}, index=IDX[:20])
    pp = Preprocessor(("diff", "zscore"))
    prepared = pp.prepare(df)
    pp.fit(prepared.iloc[:10])
    assert prepared.iloc[1:10]["a"].eq(1.0).all()
    z = pp.transform(prepared)
    back = pp.inverse(z["a"], "a")
    pd.testing.assert_series_equal(back, prepared["a"], check_names=False)


def test_stateful_step_before_stateless_is_rejected():
    with pytest.raises(ValueError, match="stateful"):
        Preprocessor(("zscore", "diff")).prepare(pd.DataFrame({"a": [1.0, 2.0]}, index=IDX[:2]))


def test_prepare_can_skip_the_resample_for_another_granularity():
    m = make_regression("ols", prep=("resample:W-FRI",))
    df = _linear()
    assert len(m.prepare(df)) < len(df)
    assert len(m.prepare(df, skip=("resample",))) == len(df)


# --------------------------------------------------------------------------- OLS
def test_ols_matches_closed_form_and_classical_errors():
    df = _linear()
    m = make_regression("ols").fit(make_regression("ols").prepare(df))
    X = np.column_stack([np.ones(len(df)), df[["x1", "x2", "x3"]].to_numpy()])
    b = np.linalg.solve(X.T @ X, X.T @ df["y"].to_numpy())
    e = df["y"].to_numpy() - X @ b
    se = np.sqrt(np.diag(e @ e / (len(df) - 4) * np.linalg.inv(X.T @ X)))
    np.testing.assert_allclose(m.coef.to_numpy(), b, rtol=1e-10)
    np.testing.assert_allclose(m.fitted_.table["se"].to_numpy(), se, rtol=1e-8)
    ss = ((df["y"] - df["y"].mean()) ** 2).sum()
    assert m.fitted_.stats["r2"] == pytest.approx(1 - (e @ e) / ss)


def test_hac_with_zero_lags_is_white():
    rng = np.random.default_rng(3)
    X, e = np.column_stack([np.ones(200), rng.normal(size=200)]), rng.normal(size=200)
    B = np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(inf.sandwich(X, e, B, "HAC", 0), inf.sandwich(X, e, B, "HC0"))


def test_contributions_sum_to_fitted_in_original_units():
    m = make_regression("ols", prep=("diff", "zscore"))
    d = m.prepare(_linear())
    m.fit(d, as_of=IDX[399])
    out = m.predict(d, start=IDX[399], contributions=True)
    np.testing.assert_allclose(pick(out, "contrib").sum(axis=1), out["fitted"], atol=1e-10)
    np.testing.assert_allclose(out["y"], d.iloc[400:]["y"])


def test_y_lead_never_fits_on_a_target_not_yet_known():
    df = _linear()
    m = make_regression("ols", y_lead=5)
    d = m.prepare(df)
    m.fit(d, as_of=IDX[399])
    assert m.fitted_.stats["n"] == 400 - 5  # rows 395..399 have targets after the fit date


def test_univariate_rejects_two_regressors():
    from infra.models.stats.regression import UnivariateRegression
    m = UnivariateRegression("ols")
    with pytest.raises(ValueError, match="one regressor"):
        m.fit(m.prepare(_linear()))


def test_stepwise_drops_the_noise_regressor():
    m = make_regression("stepwise")
    m.fit(m.prepare(_linear()))
    assert m.fitted_.table.loc["x3", "selected"] == 0 and m.coef["x3"] == 0
    assert m.fitted_.table.loc[["x1", "x2"], "selected"].eq(1).all()


# --------------------------------------------------------------------------- penalised
def test_ridge_matches_the_closed_form_on_standardised_data():
    df = _linear()
    m = make_regression("ridge", alpha=0.3)
    m.fit(m.prepare(df))
    Z = df[["x1", "x2", "x3"]]
    mu, sd = Z.mean().to_numpy(), Z.std(ddof=0).to_numpy()
    Zs = (Z.to_numpy() - mu) / sd
    yc = df["y"].to_numpy() - df["y"].mean()
    bz = np.linalg.solve(Zs.T @ Zs / len(df) + 0.3 * np.eye(3), Zs.T @ yc / len(df))
    np.testing.assert_allclose(m.coef[["x1", "x2", "x3"]].to_numpy(), bz / sd, rtol=1e-8)


def test_lasso_zeroes_everything_at_alpha_max_and_cv_picks_a_point_on_the_grid():
    m = make_regression("lasso", alpha=1e6)
    m.fit(m.prepare(_linear()))
    assert (m.coef.drop("const") == 0).all()
    cv = make_regression("lasso")
    cv.fit(cv.prepare(_linear()))
    assert cv.fitted_.stats["alpha"] in cv.fitted_.extra["cv"]["alpha"].to_numpy()
    assert np.isnan(cv.fitted_.table["se"]).all()  # no classical errors after selection


# --------------------------------------------------------------------------- robust / other
def test_huber_resists_outliers_that_move_ols():
    df = _linear()
    df.iloc[::25, 0] += 30.0
    ols = make_regression("ols").fit(make_regression("ols").prepare(df))
    hub = make_regression("huber").fit(make_regression("huber").prepare(df))
    assert abs(hub.coef["x1"] - 2) < abs(ols.coef["x1"] - 2) or abs(hub.coef["const"] - 1) < abs(ols.coef["const"] - 1)
    assert abs(hub.coef["const"] - 1) < 0.15


def test_quantile_regression_on_a_constant_is_the_sample_quantile():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"y": rng.normal(size=301)}, index=IDX[:301])
    for tau in (0.25, 0.5, 0.9):
        m = make_regression("quantile", tau=tau, x=())
        m.fit(m.prepare(df))
        assert m.coef["const"] == pytest.approx(np.quantile(df["y"], tau, method="inverted_cdf"), abs=1e-8)


def test_hockey_stick_recovers_the_knot_and_both_slopes():
    rng = np.random.default_rng(7)
    x = rng.uniform(-2, 2, 600)
    y = 0.2 * x + 1.5 * np.maximum(x - 0.5, 0) + rng.normal(scale=0.1, size=600)
    m = make_regression("hockey")
    d = m.prepare(pd.DataFrame({"y": y, "x": x}, index=IDX))
    m.fit(d)
    st = m.fitted_.stats
    assert st["knot"] == pytest.approx(0.5, abs=0.1)
    assert st["slope_left"] == pytest.approx(0.2, abs=0.05) and st["slope_right"] == pytest.approx(1.7, abs=0.05)
    assert st["knot_ci_low"] <= st["knot"] <= st["knot_ci_high"]
    rebuilt = Regression.from_params(m.params(), "hockey")
    np.testing.assert_allclose(rebuilt.predict(d)["fitted"], m.predict(d)["fitted"])


def test_orthogonal_regression_is_symmetric():
    rng = np.random.default_rng(2)
    t = rng.normal(size=600)
    a, b = t + 0.3 * rng.normal(size=600), 2 * t + 0.3 * rng.normal(size=600)
    df = pd.DataFrame({"a": a, "b": b}, index=IDX)
    ab = make_regression("tls", y="a", x=("b",))
    ba = make_regression("tls", y="b", x=("a",))
    ab.fit(ab.prepare(df))
    ba.fit(ba.prepare(df))
    assert ab.coef["b"] == pytest.approx(1 / ba.coef["a"], rel=1e-10)


@pytest.mark.parametrize("method", ["logit", "probit"])
def test_binary_regressions_recover_their_coefficients(method):
    from scipy import stats
    rng = np.random.default_rng(11)
    n = 20_000
    x = rng.normal(size=n)
    eta = 0.3 + 1.2 * x
    p = 1 / (1 + np.exp(-eta)) if method == "logit" else stats.norm.cdf(eta)
    y = (rng.uniform(size=n) < p).astype(float)
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    m = make_regression(method)
    m.fit(m.prepare(pd.DataFrame({"y": y, "x": x}, index=idx)))
    np.testing.assert_allclose(m.coef.to_numpy(), [0.3, 1.2], atol=0.06)
    assert 0.5 < m.fitted_.stats["auc"] < 1 and m.fitted_.stats["lr_p"] < 1e-10


def test_logit_threshold_binarises_the_target():
    df = _linear()
    m = make_regression("logit", threshold=1.0, x=("x1",))
    m.fit(m.prepare(df))
    out = m.predict()
    assert set(out["y"].dropna().unique()) <= {0.0, 1.0} and out["fitted"].between(0, 1).all()


def test_kalman_tracks_a_drifting_beta_and_rebuilds_from_params():
    rng = np.random.default_rng(4)
    x = rng.normal(size=600)
    beta = 1 + np.cumsum(rng.normal(scale=0.05, size=600))
    df = pd.DataFrame({"y": beta * x + rng.normal(scale=0.3, size=600), "x": x}, index=IDX)
    m = make_regression("kalman")
    d = m.prepare(df)
    m.fit(d, as_of=IDX[399])
    out = m.predict(d, start=IDX[399])
    fixed = make_regression("ols")
    fixed.fit(fixed.prepare(df), as_of=IDX[399])
    err_kf = np.abs(out["beta:x"] - beta[400:]).mean()
    assert err_kf < 0.5 * np.abs(fixed.coef["x"] - beta[400:]).mean()
    assert out["resid_z"].std() == pytest.approx(1.0, abs=0.2)  # calibrated innovations
    rebuilt = Regression.from_params(m.params(), "kalman")
    np.testing.assert_allclose(rebuilt.predict(d, start=IDX[399])["fitted"], out["fitted"])


def test_params_round_trip_reproduces_predictions():
    m = make_regression("ols", prep=("diff", "zscore"))
    d = m.prepare(_linear())
    m.fit(d, as_of=IDX[399])
    rebuilt = Regression.from_params(m.params(), "ols", prep=("diff", "zscore"))
    assert rebuilt.fitted_.as_of == IDX[399]
    pd.testing.assert_frame_equal(rebuilt.predict(d, start=IDX[399]).drop(columns="fitted_se", errors="ignore"),
                                  m.predict(d, start=IDX[399]).drop(columns="fitted_se", errors="ignore"))


# --------------------------------------------------------------------------- PCA
def test_pca_matches_the_covariance_eigendecomposition():
    df = _factor_panel()
    m = make_pca("pca", prep=(), n_components=2).fit(_factor_panel())
    C = np.cov(df.to_numpy(), rowvar=False, bias=True)
    vals, vecs = np.linalg.eigh(C)
    np.testing.assert_allclose(m.explained["eigenvalue"].to_numpy(), vals[::-1], rtol=1e-10)
    for j in range(2):
        assert abs(abs(m.loadings.iloc[:, j].to_numpy() @ vecs[:, -1 - j]) - 1) < 1e-10
        assert m.loadings.iloc[:, j].sum() > 0  # sign convention


def test_pca_scores_are_uncorrelated_in_sample_and_residuals_reconstruct():
    df = _factor_panel()
    m = make_pca("pca", prep=(), n_components=2).fit(df)
    out = m.predict()
    assert abs(pick(out, "score").corr().iloc[0, 1]) < 1e-8
    np.testing.assert_allclose(pick(out, "fitted") + pick(out, "residual"), df.to_numpy(), atol=1e-10)
    fc = dg.factor_correlations(m, df, freq="YE")
    assert fc["corr"].abs().max() > 0  # sub-periods need not be uncorrelated


def test_projection_with_a_missing_column_uses_the_observed_ones():
    df = _factor_panel(noise=1e-6)
    m = make_pca("pca", prep=(), n_components=2).fit(df)
    gappy = df.copy()
    gappy.iloc[-50:, 0] = np.nan
    out = m.predict(gappy, start=IDX[549])
    np.testing.assert_allclose(pick(out, "score").to_numpy(), pick(m.predict(df, start=IDX[549]), "score").to_numpy(),
                               atol=1e-4)
    np.testing.assert_allclose(pick(out, "fitted")["c0"], df["c0"].iloc[-50:], atol=1e-4)  # imputed
    assert (out["n_obs"] == 5).all()


def test_ppca_em_beats_listwise_deletion_with_gaps():
    df = _factor_panel()
    rng = np.random.default_rng(9)
    gappy = df.mask(rng.uniform(size=df.shape) < 0.15)
    truth = make_pca("pca", prep=(), n_components=2).fit(df)
    em = make_pca("missing", prep=(), n_components=2).fit(gappy)
    cc = make_pca("pca", prep=(), n_components=2).fit(gappy)
    ang = lambda m: dg.principal_angles(m.loadings.to_numpy(), truth.loadings.to_numpy()).max()  # noqa: E731
    assert em.fitted_.stats["em_converged"] == 1.0
    assert ang(em) < ang(cc)


def test_ppca_em_on_complete_data_is_the_sample_covariance():
    X = _factor_panel().to_numpy()
    mu, C, info = ppca_em(X, np.ones(len(X)), 2)
    np.testing.assert_allclose(C, np.cov(X, rowvar=False, bias=True), atol=1e-10)


def test_pca_from_params_and_align_to():
    df = _factor_panel()
    m = make_pca("pca", prep=(), n_components=2).fit(df)
    rebuilt = PCA.from_params(m.params(), "pca", prep=(), n_components=2)
    pd.testing.assert_frame_equal(rebuilt.predict(df).drop(columns="in_sample"), m.predict().drop(columns="in_sample"))
    flipped = make_pca("pca", prep=(), n_components=2).fit(df)
    flipped.fitted_.loadings[:, 1] *= -1
    flipped.align_to(m)
    np.testing.assert_allclose(flipped.loadings.to_numpy(), m.loadings.to_numpy())


def test_weighted_pca_needs_weights_and_downweights_the_past():
    with pytest.raises(ValueError, match="halflife"):
        make_pca("weighted", prep=()).fit(_factor_panel())
    df = _factor_panel()
    df.iloc[:300] *= 10.0  # a loud early regime
    eq = make_pca("pca", prep=()).fit(df)
    ew = make_pca("weighted", prep=(), halflife=20.0).fit(df)
    assert ew.explained["eigenvalue"].iloc[0] < eq.explained["eigenvalue"].iloc[0] / 10


def test_pc_neutral_weights_have_zero_exposure():
    m = make_pca("pca", prep=(), n_components=3).fit(_factor_panel())
    w = dg.pc_neutral_weights(m, ["c0", "c1", "c2"], fixed="c1")
    expo = (m.loadings_units.loc[w.index, ["PC1", "PC2"]].T @ w).to_numpy()
    np.testing.assert_allclose(expo, 0.0, atol=1e-10)


def test_eigenvector_stability_is_one_for_a_stable_structure():
    st = dg.eigenvector_stability(lambda: make_pca("pca", prep=(), n_components=2, window=250),
                                  _factor_panel(noise=0.05), IDX[300], IDX[-1], refit="QE")
    assert st["cos_prev"].dropna().min() > 0.99 and st["subspace_angle_ref"].max() < 5


# --------------------------------------------------------------------------- walk-forward
@pytest.mark.parametrize("factory", [lambda: make_regression("ols", prep=("diff",)),
                                     lambda: make_pca("pca", prep=("diff",), n_components=2)])
def test_walk_forward_is_point_in_time(factory):
    """Changing every value after T must not change any prediction or parameter up to T."""
    df = _factor_panel()
    cut = IDX[449]
    base = walk_forward(factory, df, IDX[300], IDX[-1], refit="W-FRI")
    shocked = df.copy()
    shocked.loc[shocked.index > cut] += 50.0
    alt = walk_forward(factory, shocked, IDX[300], IDX[-1], refit="W-FRI")
    a = base.predictions[base.predictions.index <= cut].drop(columns="fit_as_of")
    b = alt.predictions[alt.predictions.index <= cut].drop(columns="fit_as_of")
    pd.testing.assert_frame_equal(a, b)
    pa = base.params[base.params["fit_as_of"] <= cut].reset_index(drop=True)
    pb = alt.params[alt.params["fit_as_of"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(pa, pb)
    assert not base.predictions["in_sample"].any()  # every stitched row is out of sample
    assert base.predictions.index.is_unique


def test_refit_dates_snap_to_rows_and_start_first():
    idx = pd.bdate_range("2024-01-02", periods=30)
    d = refit_dates(idx, "2024-01-03", idx[-1], "W-FRI")
    assert d[0] == pd.Timestamp("2024-01-03") and all(x.weekday() == 4 for x in d[1:])


def test_walk_forward_params_at_rebuilds_the_fit():
    df = _linear()
    res = walk_forward(lambda: make_regression("ols"), df, IDX[300], IDX[-1], refit="ME")
    d = res.fit_dates[2]
    rebuilt = Regression.from_params(res.params_at(d), "ols")
    seg = res.predictions[res.predictions["fit_as_of"] == d]
    np.testing.assert_allclose(rebuilt.predict(df, start=d, end=seg.index.max())["fitted"], seg["fitted"])


# --------------------------------------------------------------------------- panel / storage
def test_series_panel_parses_ids_and_reads_through_sources(monkeypatch):
    from infra.pipeline import series_panel as sp
    with pytest.raises(ValueError, match="bad series id"):
        sp.parse_id("nope")
    calls = {}

    def fake(keys, start, end, as_of):
        calls["keys"] = keys
        return pd.DataFrame({k: [1.0, np.nan, 3.0] for k in keys}, index=pd.bdate_range("2024-01-01", periods=3))

    monkeypatch.setitem(sp.SERIES_SOURCES, "bond", sp.SeriesSource(fake, "fake"))
    out = sp.read_panel(["bond:A", "bond:B"], "2024-01-01", "2024-01-03", as_of="2024-01-02")
    assert calls["keys"] == ["A", "B"] and list(out.columns) == ["bond:A", "bond:B"]
    assert out.index.max() == pd.Timestamp("2024-01-01")  # the all-NaN 01-02 dropped, 01-03 after as_of


def test_model_runs_round_trip(tmp_path):
    from infra.storage.model_runs import list_runs, read_run, save_run
    df = _linear()
    res = walk_forward(lambda: make_regression("ols"), df, IDX[400], IDX[-1], refit="ME")
    save_run("t", res.params, res.predictions, {"spec": "ols"}, root=tmp_path)
    params, pred, meta = read_run("t", root=tmp_path)
    assert list_runs(root=tmp_path) == ["t"] and meta == {"spec": "ols"}
    pd.testing.assert_frame_equal(pred, res.predictions, check_freq=False, check_names=False)
    assert len(params) == len(res.params)


def test_noise_edge_counts_the_true_factors_with_many_series():
    rng = np.random.default_rng(0)
    n, p = 800, 40
    X = rng.normal(size=(n, 3)) @ (rng.normal(size=(3, p)) * [[3], [2], [1]]) + rng.normal(size=(n, p))
    m = make_pca("pca", prep=(), n_components=5).fit(pd.DataFrame(X, index=pd.bdate_range("2020-01-01", periods=n)))
    assert m.fitted_.stats["n_above_mp"] == 3
    assert m.fitted_.stats["noise_var"] == pytest.approx(1.0, abs=0.1)


def test_models_page_runs_every_method_headless():
    from infra.dashboard.models_callbacks import render_pca, render_regression, run_model
    df = _factor_panel()
    for method in ("ols", "ridge", "hockey", "kalman", "logit"):
        cols = ["c0", "c1"] if method == "hockey" else ["c0", "c1", "c2"]
        res = run_model("regression", method, cols, None, None, IDX[400], panel=df[cols],
                        options={"threshold": 0.0} if method == "logit" else None)
        assert len(render_regression(res, "light")) == 6
    for method in ("pca", "missing"):
        res = run_model("pca", method, list(df.columns), None, None, IDX[400], panel=df, diagnostics=True)
        assert len(render_pca(res, "dark")) == 6


def test_similarity_pca_weights_days_like_the_fit_dates_state():
    """Two states with different factor structure: fitted in state B, the similarity-weighted
    PCA recovers B's loading; the plain PCA gets the blend. The state's own columns are not
    decomposed, and an incomplete state row weighs 0."""
    from infra.models.stats.pca import make_pca
    rng = np.random.default_rng(3)
    n = 1200
    state = np.r_[np.zeros(600), np.ones(600)] + rng.normal(0, 0.1, n)
    load_a, load_b = np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 0.3, -0.3, -1.0])
    f = rng.normal(0, 1, n)
    X = np.where(state[:, None] < 0.5, f[:, None] * load_a, f[:, None] * load_b) + rng.normal(0, 0.2, (n, 4))
    idx = pd.bdate_range("2015-01-01", periods=n)
    df = pd.DataFrame(X, index=idx, columns=list("abcd"))
    df["s"] = state
    df.iloc[650, 4] = np.nan
    sim = make_pca("pca", method="similarity", columns=tuple("abcd"), state_columns=("s",), bandwidth=0.5,
                   n_components=1).fit(df, as_of=idx[-1])
    plain = make_pca("pca", columns=tuple("abcd"), n_components=1).fit(df[list("abcd")], as_of=idx[-1])
    cos = lambda f, v: abs(f.fitted_.loadings[:, 0] @ v) / np.linalg.norm(v)  # noqa: E731
    assert sim.fitted_.columns == list("abcd")
    assert cos(sim, load_b) > 0.99 > cos(plain, load_b)
    assert sim.fitted_.stats["n_eff"] < 700 and sim.fitted_.stats["state:s"] == pytest.approx(state[-1])
