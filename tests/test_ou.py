"""OU mean-reversion maths (infra/analytics/ou.py) against closed forms and Monte Carlo of the exact
discretisation (standardised: dz = -z dt + sqrt(2) dW)."""
from __future__ import annotations

import numpy as np
import pytest

from infra.analytics import ou

DT = 0.002


def _paths(z0, n=20000, t_max=8.0, seed=0, lower=-np.inf, upper=np.inf, target=None):
    """Exact OU steps; returns (first time each path hits `target` / leaves (lower, upper), which side)."""
    rng = np.random.default_rng(seed)
    z = np.full(n, float(z0))
    hit = np.full(n, np.nan)
    side = np.zeros(n)
    e, s = np.exp(-DT), np.sqrt(1 - np.exp(-2 * DT))
    for k in range(1, int(t_max / DT) + 1):
        live = np.isnan(hit)
        if not live.any():
            break
        prev = z[live]
        z[live] = prev * e + s * rng.normal(size=live.sum())
        idx = np.flatnonzero(live)
        cur = z[live]
        u = rng.random(size=(2, len(cur)))

        def crossed(level, j):
            """Crossed at the step's end, or between steps (Brownian bridge: P = exp(-2 d0 d1 / (2 dt)))."""
            d0, d1 = level - prev, level - cur
            same = np.sign(d0) == np.sign(d1)
            return ~same | (u[j] < np.exp(-np.clip(d0 * d1, 0, None) / DT))
        if target is not None:
            done = crossed(target, 0)
            hit[idx[done]] = k * DT
        else:
            up, dn = crossed(upper, 0), crossed(lower, 1)
            both = up & dn
            up = up & ~(both & (u[0] < 0.5))
            dn = dn & ~up
            hit[idx[up | dn]] = k * DT
            side[idx[up]], side[idx[dn]] = 1.0, -1.0
    return hit, side


def test_ar1_estimate_recovers_a_known_ou_and_the_bias_correction_helps():
    rng = np.random.default_rng(1)
    b_true, m, sd = 0.9, 5.0, 1.0
    est, est_bc = [], []
    for _ in range(300):
        x = np.empty(120)
        x[0] = m
        for t in range(1, 120):
            x[t] = m + b_true * (x[t - 1] - m) + sd * rng.normal()
        est.append(ou.fit_ar1(x).b)
        est_bc.append(ou.fit_ar1(x, bias_correct=True).b)
    assert np.mean(est) < b_true - 0.02                                   # biased toward faster reversion
    assert abs(np.mean(est_bc) - b_true) < abs(np.mean(est) - b_true) / 2
    p = ou.ou_params(ou.AR1Fit(a=m * (1 - b_true), b=b_true, se_b=0.01, sd_e=sd, n=1000))
    assert p.m == pytest.approx(m) and p.half_life == pytest.approx(np.log(2) / -np.log(b_true))
    assert p.sd_eq == pytest.approx(sd / np.sqrt(1 - b_true ** 2))
    assert not ou.ou_params(ou.AR1Fit(0.0, 1.01, 0.01, 1.0, 100)).reverting


def test_first_passage_to_the_mean_matches_simulation():
    hit, _ = _paths(1.5, target=0.0)
    assert np.isnan(hit).mean() < 0.01
    for t in (0.3, 1.0, 2.0):
        assert ou.fpt_cdf(1.5, t) == pytest.approx((hit <= t).mean(), abs=0.02)
    assert ou.fpt_median(1.5) == pytest.approx(np.nanmedian(hit), rel=0.04)
    assert ou.expected_hitting_time(1.5, 0.0) == pytest.approx(np.nanmean(hit), rel=0.04)
    assert ou.mean_fpt_to_mean(1.5) == pytest.approx(ou.expected_hitting_time(1.5, 0.0), rel=1e-3)
    # cdf and mean agree: E[T] = integral of the survival function
    ts = np.linspace(0, 40, 40001)
    assert np.trapezoid(1 - ou.fpt_cdf(1.5, ts), ts) == pytest.approx(ou.expected_hitting_time(1.5, 0.0), rel=1e-3)


def test_hitting_a_level_away_from_the_mean_matches_simulation():
    hit, _ = _paths(0.0, target=1.5, t_max=40.0, n=4000)
    assert ou.expected_hitting_time(0.0, 1.5) == pytest.approx(np.nanmean(hit), rel=0.06)
    assert ou.expected_hitting_time(0.0, -1.5) == pytest.approx(ou.expected_hitting_time(0.0, 1.5))


def test_two_barriers_match_simulation():
    hit, side = _paths(-2.0, lower=-2.5, upper=-0.5)
    assert ou.p_hit_upper_first(-2.0, -2.5, -0.5) == pytest.approx((side == 1).mean(), abs=0.02)
    assert ou.expected_exit_time(-2.0, -2.5, -0.5) == pytest.approx(np.nanmean(hit), rel=0.04)
    assert ou.p_target_before_stop(2.0, 0.5, 0.5) == pytest.approx(ou.p_hit_upper_first(-2.0, -2.5, -0.5), abs=1e-3)
    assert ou.expected_trade_time(-2.0, 0.5, 0.5) == pytest.approx(ou.expected_exit_time(-2.0, -2.5, -0.5), rel=1e-2)


def test_horizon_forecast_and_the_optimal_entry():
    p = ou.ou_params(ou.AR1Fit(a=0.0, b=0.8, se_b=0.01, sd_e=1.0, n=500))
    assert ou.expected_level(2.0, p, 3) == pytest.approx(2.0 * 0.8 ** 3)
    assert ou.horizon_sd(p, 1) == pytest.approx(1.0) and ou.horizon_sd(p, 1e6) == pytest.approx(p.sd_eq)
    assert ou.optimal_entry(0.0) < 0.1                                    # no cost: trade as often as possible
    e0, e1 = ou.optimal_entry(0.2), ou.optimal_entry(1.0)
    assert 0.2 < e0 < 1.5 and e1 > e0                                     # costs push the entry out
    assert ou.cycle_rate(e0, -e0, 0.2) >= max(ou.cycle_rate(e0 * 0.6, -e0 * 0.6, 0.2),
                                             ou.cycle_rate(e0 * 1.5, -e0 * 1.5, 0.2))
