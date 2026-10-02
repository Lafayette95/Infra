"""infra/models/nowcast: transforms, panel preparation, Kalman smoother, EM with loading
restrictions (versions a-d), news decomposition, and point-in-time behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.config import MacroRelease
from infra.reference.events import EventSeries
from infra.models.nowcast import kalman, transforms
from infra.models.nowcast.dfm import _aggregate, fit
from infra.models.nowcast.news import decompose
from infra.models.nowcast.nowcast import estimate, news, nowcast, upcoming
from infra.models.nowcast.panel import build_panel, to_monthly
from infra.models.nowcast.spec import VERSIONS, factor_structure

LAB, HOU = "Activity_Labor", "Activity_Housing"


# ------------------------------------------------------------------ transforms
def _monthly(values, start="2000-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float)


def test_parse_rejects_unknown_steps():
    assert transforms.parse("deman_fix:50;ecdf_no_demean_exp:130") == [("deman_fix", 50.0), ("ecdf_no_demean_exp", 130.0)]
    with pytest.raises(ValueError):
        transforms.parse("zscore:12")


def test_rolling_ma_and_deman_fix():
    s = _monthly([1, 2, 3, 4])
    assert transforms.apply(s, "rolling_ma:3").tolist()[2:] == [2.0, 3.0]
    assert transforms.apply(s, "deman_fix:2").tolist() == [-1.0, 0.0, 1.0, 2.0]


def test_ecdf_with_a_huge_half_life_is_the_plain_expanding_rank():
    rng = np.random.default_rng(1)
    s = _monthly(rng.normal(size=40))
    out = transforms.ecdf_exp(s, 1e12, min_obs=1)
    for t in (5, 20, 39):
        past = s.iloc[: t + 1]
        u = ((past < s.iloc[t]).sum() + 0.5) / len(past)
        assert out.iloc[t] == pytest.approx(2 * u - 1)


def test_ecdf_no_demean_keeps_zero_and_sign_ecdf_recentres():
    # an always-positive series: the demeaned ECDF centres it (about half negative), the
    # non-demeaned one keeps every value positive - "no demeaning", like x / RMS
    rng = np.random.default_rng(2)
    s = _monthly(2.0 + rng.random(80))
    dm = transforms.ecdf_exp(s, 130, min_obs=5).dropna()
    nd = transforms.ecdf_no_demean_exp(s, 130, min_obs=5).dropna()
    assert (dm < 0).mean() > 0.2 and (nd > 0).all()
    flipped = transforms.ecdf_no_demean_exp(-s, 130, min_obs=5).dropna()
    np.testing.assert_allclose(flipped, -nd)


def test_transforms_are_trailing_new_data_never_changes_the_past():
    rng = np.random.default_rng(3)
    full = _monthly(rng.normal(size=60))
    for spec in ("rolling_ma:3;ecdf_no_demean_exp:1305", "ecdf_exp:130;gauss", "deman_fix:50;ecdf_no_demean_exp:130"):
        a = transforms.apply(full.iloc[:45], spec, min_obs=3)
        b = transforms.apply(full, spec, min_obs=3).iloc[:45]
        pd.testing.assert_series_equal(a, b)


def test_ecdf_weights_recent_history_more():
    # a level shift: with a short half-life the new regime is soon "normal" again
    s = _monthly([0.0] * 100 + [5.0] * 30)
    short = transforms.ecdf_exp(s, 21, min_obs=1).iloc[-1]
    long = transforms.ecdf_exp(s, 1e9, min_obs=1).iloc[-1]
    assert short < long


def test_gauss_maps_to_normal_scores():
    s = _monthly([-0.5, 0.0, 0.5])
    np.testing.assert_allclose(transforms.gauss(s).to_numpy(), [-0.6744898, 0.0, 0.6744898], atol=1e-6)


# ------------------------------------------------------------------ panel
def test_to_monthly_quarterly_sits_in_the_third_month():
    q = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-04-01"]))
    out = to_monthly(q, "Q")
    assert list(out.index) == list(pd.to_datetime(["2026-03-01", "2026-06-01"]))


def test_to_monthly_weekly_only_complete_months():
    weeks = pd.date_range("2026-01-03", "2026-02-21", freq="W-SAT")  # Jan has 5 Saturdays; Feb data stops early
    out = to_monthly(pd.Series(np.arange(len(weeks), dtype=float), index=weeks), "W")
    assert list(out.index) == [pd.Timestamp("2026-01-01")]
    assert out.iloc[0] == pytest.approx(2.0)


def _rel(ticker, freq, blocks, *, units="level", transform="", sign=1):
    series = EventSeries(ticker, f"E_{ticker}", ticker, "u", "SA", store_id=ticker, frequency=freq, source="fred",
                         derive=units)
    return MacroRelease(ticker, ticker, series, freq == "Q", freq == "Q", freq == "Q",
                        False, True, freq != "Q", sign, transform, "Activity", "Hard", tuple(blocks))


def _vintages(values: pd.Series, lag_days: int, ticker: str, *, months=1, revise_by=0.0, revise_after=30):
    """Raw vintage rows: each period (``months`` long) first published ``lag_days`` after
    it ends, and optionally revised once ``revise_after`` days later."""
    end = values.index + pd.offsets.MonthEnd(months)
    first = pd.DataFrame({"timestamp": end + pd.Timedelta(days=lag_days), "ticker": ticker,
                          "period": values.index, "value": values.to_numpy()})
    if not revise_by:
        return first
    second = first.assign(timestamp=first["timestamp"] + pd.Timedelta(days=revise_after), value=first["value"] + revise_by)
    return pd.concat([first, second], ignore_index=True)


@pytest.fixture(scope="module")
def world():
    """Two blocks (labor, housing), 3 monthly releases each + quarterly GDP, as raw
    vintages - GDP first published 55 days after its quarter ends (Q4 on Feb 24) and
    revised 30 days later."""
    rng = np.random.default_rng(7)
    n_t = 240
    months = pd.date_range("2005-01-01", periods=n_t, freq="MS")
    f = np.zeros((n_t, 2))
    for t in range(1, n_t):
        f[t] = np.array([[0.7, 0.1], [0.0, 0.6]]) @ f[t - 1] + rng.normal(size=2)
    releases, rows = {}, []
    for k in range(6):
        j, block = (0, LAB) if k < 3 else (1, HOU)
        name = f"S{k}"
        releases[name] = _rel(name, "M", [block])
        rows.append(_vintages(pd.Series(0.8 * f[:, j] + 0.6 * rng.normal(size=n_t), index=months), 5 + 3 * k, name))
    z = _aggregate(f)
    q_idx = np.arange(2, n_t, 3)
    gdp = pd.Series(2.0 + 0.3 * z[q_idx, 0] + 0.2 * z[q_idx, 1] + 0.4 * rng.normal(size=len(q_idx)),
                    index=months[q_idx - 2])  # the source dates a quarter by its first month
    releases["GDP"] = _rel("GDP", "Q", [LAB, HOU])
    rows.append(_vintages(gdp, 55, "GDP", months=3, revise_by=0.1))
    raw = pd.concat(rows, ignore_index=True)
    raw["ticker"] = raw["ticker"].astype("category")
    return releases, raw


def _spec(v, **kw):
    return VERSIONS[v].with_(target="GDP", sample_start="2005-01-01", max_iter=60, **kw)


def test_panel_is_point_in_time(world):
    releases, raw = world
    as_of = pd.Timestamp("2019-06-15")
    panel = build_panel(raw, releases, _spec("c"), as_of)
    assert (panel.published.stack().dropna() <= as_of).all()
    # GDP enters untransformed, and as published by as_of (first print, not yet revised)
    q1 = panel.data.loc["2019-03-01", "GDP"]
    first_print = raw[(raw["ticker"] == "GDP") & (raw["period"] == "2019-01-01")].sort_values("timestamp")["value"]
    assert q1 == pytest.approx(first_print.iloc[0])
    assert pd.isna(panel.data.loc["2019-06-01", "GDP"])  # Q2 not out yet


# ------------------------------------------------------------------ versions
def test_factor_structure_per_version(world):
    releases, _ = world
    series = ["GDP", "S0", "S3"]
    a = factor_structure(_spec("a"), releases, series)
    assert a.factors == ("Global",) and a.free.all()
    b = factor_structure(_spec("b"), releases, series)
    c = factor_structure(_spec("c"), releases, series)
    d = factor_structure(_spec("d"), releases, series)
    assert b.factors == c.factors == d.factors == (LAB, HOU)
    assert b.free.all() and not b.penalized.any()
    np.testing.assert_array_equal(c.free, [[True, True], [True, False], [False, True]])
    assert d.free.all() and (d.penalized == ~c.free).all()
    g = factor_structure(_spec("c", global_factor=True), releases, series)
    assert g.factors[0] == "Global" and g.free[:, 0].all()


# ------------------------------------------------------------------ kalman
def test_smoother_matches_the_brute_force_gaussian_posterior():
    """E[s | x] and Cov[s_t, s_t-1 | x] from the Kalman smoother equal the exact
    joint-Gaussian posterior of the stacked system, with missing data."""
    rng = np.random.default_rng(0)
    m, n, n_t = 2, 3, 6
    T = np.array([[0.6, 0.2], [-0.1, 0.5]])
    Q = np.array([[1.0, 0.3], [0.3, 0.5]])
    Z = rng.normal(size=(n, m))
    R = np.array([0.2, 0.5, 0.3])
    a0, P0 = kalman.unconditional(T, Q)
    X = rng.normal(size=(n_t, n))
    X[1, 0] = X[3, :] = X[4, 2] = np.nan
    sm = kalman.smooth(kalman.StateSpace(Z, R, T, Q, a0, P0), X)
    # stacked covariance of s_0..s_{T-1}
    S = np.zeros((n_t * m, n_t * m))
    V = [P0]
    for t in range(1, n_t):
        V.append(T @ V[-1] @ T.T + Q)
    for t in range(n_t):
        for u in range(t, n_t):
            block = np.linalg.matrix_power(T, u - t) @ V[t]
            S[u * m:(u + 1) * m, t * m:(t + 1) * m] = block
            S[t * m:(t + 1) * m, u * m:(u + 1) * m] = block.T
    H = np.kron(np.eye(n_t), Z)
    obs = np.isfinite(X).ravel()
    Hx = H[obs]
    Sxx = Hx @ S @ Hx.T + np.diag(np.tile(R, n_t)[obs])
    gain = S @ Hx.T @ np.linalg.inv(Sxx)
    mean = (gain @ X.ravel()[obs]).reshape(n_t, m)
    cov = S - gain @ Hx @ S
    np.testing.assert_allclose(sm.a, mean, atol=1e-9)
    for t in range(1, n_t):
        np.testing.assert_allclose(sm.P[t], cov[t * m:(t + 1) * m, t * m:(t + 1) * m], atol=1e-9)
        np.testing.assert_allclose(sm.P_lag[t], cov[t * m:(t + 1) * m, (t - 1) * m:t * m], atol=1e-9)


# ------------------------------------------------------------------ estimation
@pytest.fixture(scope="module")
def fitted(world):
    releases, raw = world
    as_of = pd.Timestamp("2024-02-10")
    return {v: estimate(_spec(v, tau=0.3), as_of, raw=raw, releases=releases) for v in "abcd"}


def test_em_likelihood_never_decreases(fitted):
    for v in "abc":  # d maximises a penalised likelihood, not the likelihood itself
        assert np.all(np.diff(fitted[v].loglik) > -1e-6), v


def test_hard_blocks_are_exact_zeros_and_soft_blocks_shrink(fitted):
    free = fitted["c"].structure.free
    lam = {v: fitted[v].params.lam for v in "bcd"}
    assert (lam["c"][~free] == 0).all()
    off = ~fitted["c"].structure.member
    assert np.abs(lam["d"][off]).sum() < np.abs(lam["b"][off]).sum()
    assert np.abs(lam["d"][off]).sum() > 0


def test_block_factors_load_on_their_own_members(fitted):
    lam = fitted["c"].params.lam
    series = list(fitted["c"].series)
    assert all(lam[series.index(s), 0] > 0.5 for s in ("S0", "S1", "S2"))
    assert all(lam[series.index(s), 1] > 0.5 for s in ("S3", "S4", "S5"))


# ------------------------------------------------------------------ nowcast + news
@pytest.mark.parametrize("v", list("abcd"))
def test_news_decomposition_is_exact(world, fitted, v):
    releases, raw = world
    out = news(fitted[v], "2024-02-10", "2024-03-20", raw=raw, releases=releases)
    assert len(out.impacts) > 0
    assert out.new - out.old == pytest.approx(out.revision_impact + out.news_impact, abs=1e-9)
    np.testing.assert_allclose(out.impacts["impact"],
                               out.impacts["weight"] * out.impacts["news"], atol=1e-12)
    assert out.impacts["signal_share"].between(-1e-9, 1 + 1e-9).all()


def test_a_published_gdp_print_has_weight_one_and_ends_the_nowcast(world, fitted):
    releases, raw = world
    model = fitted["c"]
    # Q4 2023 GDP (period 2023-10-01) is first published on 2024-02-24
    out = news(model, "2024-02-10", "2024-03-05", quarter="2023Q4", raw=raw, releases=releases)
    gdp = out.impacts[out.impacts["ticker"] == "GDP"]
    assert len(gdp) == 1 and gdp["weight"].iloc[0] == pytest.approx(1.0)
    assert out.new == pytest.approx(gdp["actual"].iloc[0])
    assert (out.impacts.loc[out.impacts["ticker"] != "GDP", "impact"].abs() < 1e-9).all()


def test_revisions_are_split_from_news(world, fitted):
    releases, raw = world
    # the GDP revision a month after the first print moves the NEXT quarter's nowcast
    out = news(fitted["c"], "2024-03-05", "2024-04-05", quarter="2024Q1", raw=raw, releases=releases)
    assert out.revision_impact != 0.0
    assert "GDP" not in set(out.impacts["ticker"])


def test_nowcast_is_point_in_time(world, fitted):
    releases, raw = world
    day = pd.Timestamp("2023-11-20")
    full = nowcast(fitted["c"], day, raw=raw, releases=releases)
    cut = nowcast(fitted["c"], day, raw=raw[raw["timestamp"] <= day], releases=releases)
    assert full.value == pytest.approx(cut.value)
    assert not full.published
    assert full.common.sum() == pytest.approx(full.value)  # unpublished: nowcast = its common component


def test_decompose_weight_is_the_response_to_a_unit_surprise(fitted, world):
    """A single new print: weight = d nowcast / d print, checked by finite difference."""
    releases, raw = world
    model = fitted["a"]
    panel = build_panel(raw, releases, model.spec, "2024-03-20", end="2024-03-01")
    X_new = model.standardize(panel.data.reindex(columns=list(model.series)))
    X_old = X_new.copy()
    t, i = X_new.shape[0] - 1, list(model.series).index("S0")
    if not np.isfinite(X_new[t, i]):
        t -= 1
    X_old[t:, :] = np.where(np.arange(X_new.shape[1]) == i, np.nan, X_old[t:, :])
    X_old[t + 1:, :] = X_new[t + 1:, :]
    out = decompose(model, X_old, X_new, panel.data.index, "2024Q1")
    row = out.impacts.iloc[0]
    bumped = X_new.copy()
    bumped[t, i] += 0.5
    out2 = decompose(model, X_old, bumped, panel.data.index, "2024Q1")
    assert (out2.new - out.new) / (0.5 * model.sd[i]) == pytest.approx(row["weight"], rel=1e-9)


def test_independent_dynamics_keep_factors_uncorrelated(world):
    releases, raw = world
    m = estimate(_spec("c", factor_dynamics="independent"), "2024-02-10", raw=raw, releases=releases)
    A, Q = m.params.A, m.params.Q
    assert A[0, 1] == 0 and A[1, 0] == 0 and Q[0, 1] == 0 and Q[1, 0] == 0
    assert np.all(np.diff(m.loglik) > -1e-6)


def test_excluded_months_are_never_seen(world):
    releases, raw = world
    spec = _spec("c", exclude=(("2010-01-01", "2010-12-01"),))
    panel = build_panel(raw, releases, spec, "2024-02-10")
    assert panel.data.loc["2010-01-01":"2010-12-01"].isna().all().all()
    assert panel.data.loc["2011-01-01"].notna().any()
    assert panel.native.loc["2010-06-01"].notna().any()  # the record stays; only the model's view is masked


def test_use_transforms_false_enters_the_table_units(world):
    releases = dict(world[0])
    releases["S0"] = _rel("S0", "M", [LAB], transform="ecdf_exp:1305")
    panel_t = build_panel(world[1], releases, _spec("c"), "2024-02-10")
    panel_u = build_panel(world[1], releases, _spec("c", use_transforms=False), "2024-02-10")
    assert panel_t.data["S0"].abs().max() < 1.0  # ECDF-bounded
    pd.testing.assert_series_equal(panel_u.data["S0"], panel_u.native["S0"].where(panel_u.data["S0"].notna()),
                                   check_names=False)



def test_upcoming_forecasts_and_weights(world, fitted, monkeypatch):
    """Releases due after as_of: the next unpublished period of each series, the model's
    forecast, and the GDP weight - which equals the news() weight once that print lands."""
    from infra.reference import events as ev

    releases, raw = world
    monkeypatch.setattr(ev, "SERIES", {"S0": ev.EventSeries("S0", "E_LAB", "s0", "u", "SA", bbg_ticker="S0"),
                                       "S3": ev.EventSeries("S3", "E_HOU", "s3", "u", "SA", bbg_ticker="S3")})
    model, as_of = fitted["c"], pd.Timestamp("2024-02-10")
    cal = pd.DataFrame({"timestamp": pd.to_datetime(["2024-02-15 13:30", "2024-02-20 13:30", "2024-03-30 13:30"]),
                        "event": ["E_LAB", "E_HOU", "E_LAB"], "source": "fred_release_dates", "stage": "",
                        "time_source": "registry", "known_from": as_of, "last_seen": as_of})
    up = upcoming(model, as_of, days=10, raw=raw, releases=releases, calendar=cal)
    assert list(up["ticker"]) == ["S0", "S3"]  # the third release is outside the window
    s0 = up.iloc[0]
    known = raw[raw["timestamp"] <= as_of]
    assert s0["period"] == known.loc[known["ticker"] == "S0", "period"].max() + pd.DateOffset(months=1)
    assert np.isfinite(s0["forecast"]) and np.isfinite(s0["weight"])
    # publish exactly that print the next day: news() must give it the same weight
    landed = raw[(raw["ticker"] == "S0") & (raw["period"] == s0["period"])].head(1).assign(
        timestamp=as_of + pd.Timedelta(days=1))
    out = news(model, as_of, as_of + pd.Timedelta(days=1), quarter="2024Q1",
               raw=pd.concat([known, landed], ignore_index=True), releases=releases)
    assert out.impacts.set_index("ticker").loc["S0", "weight"] == pytest.approx(s0["weight"], rel=1e-6)
