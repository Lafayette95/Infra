"""Regime models (infra/models/stats/hmm.py, regimes.py) and the regime-weighted PCA
(regime_pca.py). Synthetic panels with KNOWN regimes and loadings only."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from infra.models.stats import hmm
from infra.models.stats.common import pick
from infra.models.stats.diagnostics import principal_angles
from infra.models.stats.pca import make_pca, project_observed
from infra.models.stats.regime_pca import RegimePCA, make_regime_pca
from infra.models.stats.regimes import HMMRegimes, RuleRegimes, make_regime_model
from infra.models.walk_forward import walk_forward

K = ("a0", "a1", "a2", "a3")


def regime_panel(seed=0, n=2000, N=8, stay=0.99):
    """N level series; regime 1 = 3x volatility AND a different factor structure for the
    first 4 (the K) series."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.uniform() < stay else 1 - s[t - 1]
    W0, W1 = rng.normal(size=(N, 2)), rng.normal(size=(N, 2))
    W1[4:] = W0[4:]
    f, e = rng.normal(size=(n, 2)), rng.normal(size=(n, N))
    chg = np.where(s[:, None] == 0, f @ W0.T + 0.4 * e, 3 * (f @ W1.T) + 1.2 * e)
    idx = pd.bdate_range("2015-01-01", periods=n)
    lvl = pd.DataFrame(np.cumsum(chg, axis=0), index=idx, columns=[f"a{i}" for i in range(N)])
    return lvl, pd.Series(s, index=idx), W0, W1


def _rpca(**kw):
    kw = {"columns": K, "n_components": 2, "prep": ("diff",), "regime_overrides": (("prep", ("diff",)),), **kw}
    return make_regime_pca("regime_pca", **kw)


def _angle(L, W):
    return principal_angles(np.linalg.qr(L)[0], np.linalg.qr(W[:4])[0]).max()


# --------------------------------------------------------------------------- hmm core
def test_forward_backward_matches_brute_force_enumeration():
    rng = np.random.default_rng(0)
    n, R = 5, 2
    log_b = rng.normal(size=(n, R))
    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    pi0 = np.array([0.6, 0.4])
    fb = hmm.forward_backward(log_b, P, pi0)
    joint = {}
    for path in itertools.product(range(R), repeat=n):
        p = pi0[path[0]] * np.exp(log_b[0, path[0]])
        for t in range(1, n):
            p *= P[path[t - 1], path[t]] * np.exp(log_b[t, path[t]])
        joint[path] = p
    Z = sum(joint.values())
    smooth = np.array([[sum(v for k, v in joint.items() if k[t] == r) / Z for r in range(R)] for t in range(n)])
    np.testing.assert_allclose(fb.smoothed, smooth, atol=1e-12)
    assert fb.loglik == pytest.approx(np.log(Z))
    # filtered at the last row = smoothed at the last row; predicted[0] = pi0
    np.testing.assert_allclose(fb.filtered[-1], smooth[-1], atol=1e-12)
    np.testing.assert_allclose(fb.predicted[0], pi0)


def test_an_absorbing_transition_never_produces_nan():
    log_b = np.array([[0.0, -5.0], [-2000.0, 0.0], [0.0, 0.0]])  # day 2 impossible under regime 0
    fb = hmm.forward_backward(log_b, np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 0.0]))
    assert np.isfinite(fb.smoothed).all() and np.isfinite(fb.loglik)


def test_projection_with_gaps_does_not_explode_on_a_badly_identified_factor():
    # factor 2 loads only on the columns that are missing: least squares would be singular
    L = np.linalg.qr(np.array([[1, 0], [1, 0], [1, 1], [1, -1.0]]))[0]
    Z = np.array([[1.0, 1.1, np.nan, np.nan]])
    f = project_observed(Z, L, np.array([10.0, 1.0]), 0.1)
    assert np.isfinite(f).all() and abs(f[0, 1]) < 1.0


# --------------------------------------------------------------------------- regime models
def test_hmm_recovers_the_true_regimes_out_of_sample():
    lvl, truth, *_ = regime_panel()
    m = make_regime_model("hmm2", prep=("diff",))
    d = m.prepare(lvl)
    m.fit(d, as_of=lvl.index[1499])
    out = m.predict(d)
    oos = out[~out["in_sample"]]
    acc = ((oos["p_pred:R1"] > 0.5) == (truth[oos.index] == 1)).mean()
    assert acc > 0.95
    assert out.loc[out["in_sample"], "p_smooth:R1"].notna().any()
    assert oos["p_smooth:R1"].isna().all()  # no hindsight after the fit date


def test_regime_alignment_follows_the_previous_fit():
    lvl, *_ = regime_panel()
    a = make_regime_model("hmm2", prep=("diff",))
    d = a.prepare(lvl)
    a.fit(d, as_of=lvl.index[1499])
    b = make_regime_model("hmm2", prep=("diff",))
    b.fit(d, as_of=lvl.index[1599])
    f = b.fitted_
    swapped = [1, 0]
    f.params.means, f.params.covs = f.params.means[swapped], f.params.covs[swapped]
    f.smoothed, f.filtered, f.predicted = f.smoothed[:, swapped], f.filtered[:, swapped], f.predicted[:, swapped]
    f.filt_last = f.filt_last[swapped]
    perm = b.align_to(a)
    assert list(perm) == [1, 0]
    common = a.smoothed_frame().index.intersection(b.smoothed_frame().index)
    assert np.corrcoef(a.smoothed_frame().loc[common, 1], b.smoothed_frame().loc[common, 1])[0, 1] > 0.9


def test_rule_regimes_threshold_and_softness():
    idx = pd.bdate_range("2024-01-01", periods=5)
    raw = pd.DataFrame({"x": [-2.0, -0.5, 0.5, 2.0, 5.0]}, index=idx)
    hard = RuleRegimes("hmm2", method="rule", prep=(), rule_column="x", rule_thresholds=(0.0, 3.0), n_regimes=3)
    p = hard.fit(hard.prepare(raw)).predict()
    assert p[["p_filt:R0", "p_filt:R1", "p_filt:R2"]].to_numpy().argmax(axis=1).tolist() == [0, 0, 1, 1, 2]
    assert np.isnan(p["p_pred:R0"].iloc[0]) and p["p_pred:R0"].iloc[1] == p["p_filt:R0"].iloc[0]
    soft = RuleRegimes("hmm2", method="rule", prep=(), rule_column="x", rule_thresholds=(0.0,), rule_softness=1.0)
    ps = soft.fit(soft.prepare(raw)).predict()
    np.testing.assert_allclose(ps[["p_filt:R0", "p_filt:R1"]].sum(axis=1), 1.0)
    assert 0 < ps["p_filt:R1"].iloc[1] < 0.5 < ps["p_filt:R1"].iloc[2] < 1


# --------------------------------------------------------------------------- regime PCA
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_regime_pca_recovers_each_regimes_loadings(seed):
    lvl, truth, W0, W1 = regime_panel(seed)
    m = _rpca()
    d = m.prepare(lvl)
    m.fit(d, as_of=lvl.index[1499])
    assert _angle(m.fitted_.L[0], W0) < 8 and _angle(m.fitted_.L[1], W1) < 8
    plain = make_pca("pca", columns=K, n_components=2, prep=("diff",))
    dp = plain.prepare(lvl)
    plain.fit(dp, as_of=lvl.index[1499])
    rv = (pick(m.predict(d, start=lvl.index[1499]), "residual") ** 2).mean().mean()
    pv = (pick(plain.predict(dp, start=lvl.index[1499]), "residual") ** 2).mean().mean()
    assert rv < pv  # out-of-sample residual variance (seeds 0-2: 0.52, 0.94, 0.65 of plain PCA)


def test_regime_pca_z_scores_are_calibrated_in_both_regimes():
    lvl, truth, *_ = regime_panel()
    m = _rpca()
    d = m.prepare(lvl)
    m.fit(d, as_of=lvl.index[1499])
    z = pick(m.predict(d, start=lvl.index[1499]), "resid_z")
    by = z.groupby(truth[z.index].to_numpy()).apply(lambda g: g.stack().std())
    assert by.between(0.75, 1.3).all()


def test_predicted_timing_ignores_the_rows_own_move():
    """With timing="predicted", shocking row t's data cannot change the regime
    probabilities row t uses - nor its loadings."""
    lvl, *_ = regime_panel()
    m = _rpca()
    d = m.prepare(lvl)
    m.fit(d, as_of=lvl.index[1499])
    t = lvl.index[1700]
    base = m.predict(d, start=lvl.index[1499])
    shocked = lvl.copy()
    shocked.loc[shocked.index >= t, "a0"] += 50.0
    alt = m.predict(m.prepare(shocked), start=lvl.index[1499])
    np.testing.assert_allclose(pick(alt, "p").loc[t], pick(base, "p").loc[t])
    filt = _rpca(timing="filtered")
    filt.fit(d, as_of=lvl.index[1499])
    assert not np.allclose(pick(filt.predict(m.prepare(shocked), start=lvl.index[1499]), "p").loc[t],
                           pick(filt.predict(d, start=lvl.index[1499]), "p").loc[t])


@pytest.mark.parametrize("combine", ["covariance", "residual", "robust"])
def test_regime_pca_walk_forward_is_point_in_time(combine):
    lvl, *_ = regime_panel(n=1300)
    cut = lvl.index[1150]
    fac = lambda: _rpca(combine=combine, n_components=1 if combine == "robust" else 2)  # noqa: E731
    base = walk_forward(fac, lvl, lvl.index[1000], lvl.index[-1], refit="ME")
    shocked = lvl.copy()
    shocked.loc[shocked.index > cut] += 50.0
    alt = walk_forward(fac, shocked, lvl.index[1000], lvl.index[-1], refit="ME")
    a = base.predictions[base.predictions.index <= cut].drop(columns="fit_as_of")
    b = alt.predictions[alt.predictions.index <= cut].drop(columns="fit_as_of")
    pd.testing.assert_frame_equal(a, b)
    assert not base.failures
    assert base.param_path("stat", "value", "regime.warm_start").iloc[1:].eq(1.0).all().all()


def test_regime_pca_from_params_reproduces_predictions():
    lvl, *_ = regime_panel(n=1300)
    m = _rpca()
    d = m.prepare(lvl)
    m.fit(d, as_of=lvl.index[1199])
    rebuilt = RegimePCA.from_params(m.params(), "regime_pca", columns=K, n_components=2, prep=("diff",),
                                    regime_overrides=(("prep", ("diff",)),))
    pd.testing.assert_frame_equal(rebuilt.predict(d, start=lvl.index[1199]), m.predict(d, start=lvl.index[1199]))


def test_regime_pca_with_explicit_regimes():
    lvl, truth, W0, W1 = regime_panel()
    probs = pd.DataFrame({"R0": (truth == 0).astype(float), "R1": (truth == 1).astype(float)})
    m = make_regime_pca("regime_pca", columns=K, n_components=2, prep=("diff",),
                        regime_model=RuleRegimes(probabilities=probs))
    m.fit(m.prepare(lvl), as_of=lvl.index[1499])
    assert _angle(m.fitted_.L[0], W0) < 8 and _angle(m.fitted_.L[1], W1) < 8
