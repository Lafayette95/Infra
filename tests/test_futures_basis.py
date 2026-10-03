"""Treasury futures basis (CLAUDE.md 22): delivery windows, forwards, implied futures and
repo, the CTD over bonds x delivery days, and the M0 model - on synthetic inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.analytics import futures_basis as fb
from infra.analytics.sofr_curve import business_days
from infra.models.basis.model import DeterministicBasis
from infra.pipeline.futures_basis import BasisDay
from infra.processing.treasury_prices import accrued_and_yield, full_price_from_yield

D = pd.Timestamp
BD = business_days("2026-01-01", "2027-12-31")


def test_delivery_windows_follow_the_contract_rules():
    ten = fb.delivery_window("ZN", "2026-12-01", BD)
    assert ten == {"first_delivery": D("2026-12-01"), "last_trading": D("2026-12-21"), "last_delivery": D("2026-12-31")}  # Dec 25 skipped
    two = fb.delivery_window("ZT", "2026-12-01", BD)
    assert two["last_trading"] == D("2026-12-31") and two["last_delivery"] == D("2027-01-06")  # Jan 1 a holiday


def test_forward_and_implied_repo_are_inverse():
    coupons = [(D("2026-11-15"), 2.0)]
    fwd = fb.forward_clean(98.0, 1.2, 0.3, coupons, D("2026-10-01"), D("2026-12-31"), 4.0)
    futures = fwd / 0.9  # a market exactly at this bond's implied price
    assert fb.implied_repo_pct(98.0, 1.2, 0.3, coupons, D("2026-10-01"), D("2026-12-31"), futures, 0.9) == pytest.approx(4.0)


def test_zero_carry_bond_forward_equals_spot():
    assert fb.forward_clean(100.0, 0.0, 0.0, [], D("2026-10-01"), D("2026-10-01"), 4.0) == pytest.approx(100.0)


def test_price_from_yield_inverts_the_yield_solver():
    y = accrued_and_yield(97.5, 4.0, D("2033-08-15"), D("2026-10-01"), 2)[1]
    ai = accrued_and_yield(97.5, 4.0, D("2033-08-15"), D("2026-10-01"), 2)[0]
    assert full_price_from_yield(y, 4.0, D("2033-08-15"), D("2026-10-01"), 2) == pytest.approx(97.5 + ai, abs=1e-8)


def _day(futures_price):
    """Two bonds, two delivery days: A is cheap per unit of futures, B has the lower RAW net
    basis (smaller CF) - the CTD must be A."""
    rows = []
    for cusip, price, cf in (("A", 90.0, 0.90), ("B", 60.2, 0.60)):
        for kind, d in (("first", D("2026-12-01")), ("last", D("2026-12-31"))):
            rows.append({"root": "ZN", "contract": "ZNZ6", "cusip": cusip, "cf": cf, "delivery_kind": kind, "delivery": d,
                         "settle": D("2026-10-01"), "price": price, "price_bid": price, "ai_settle": 0.0, "ai_delivery": 0.0,
                         "coupons": [], "repo": 0.0, "repo_base": 0.0, "coupon": 4.0, "maturity": D("2033-08-15"),
                         "yield_eod": 4.5, "dv01": 0.06})
    fut = pd.DataFrame([{"root": "ZN", "contract": "ZNZ6", "futures": futures_price, "futures_source": "bbo_mid",
                         "settlement": np.nan, "bid": None, "ask": None, "first_delivery": D("2026-12-01"),
                         "last_trading": D("2026-12-21"), "last_delivery": D("2026-12-31")}])
    return BasisDay(D("2026-09-30"), D("2026-10-01"), pd.DataFrame(rows), fut)


def test_the_ctd_is_the_lowest_implied_futures_price_not_the_lowest_net_basis():
    m = DeterministicBasis()
    out = m.fit(None).predict(m.prepare(_day(99.0)))
    c = out["contracts"].iloc[0]
    assert c["ctd"] == "A" and c["fair_futures"] == pytest.approx(100.0)
    b = out["bonds"].groupby("cusip")["net_basis"].min()
    assert b["B"] < b["A"]  # B's raw net basis is lower, yet A is cheapest per unit of futures
    assert c["option_value_obs_32"] == pytest.approx(32.0)  # fair 100 - market 99
    assert out["bonds"]["prob"].sum() == 1.0 and c["option_value_model_32"] == 0.0


def test_a_model_must_be_fitted_before_predicting():
    from infra.models.base import NotFittedError
    with pytest.raises(NotFittedError):
        DeterministicBasis().predict(pd.DataFrame({"x": [1]}))


# ------------------------------------------------------------------- M1 maths
from infra.analytics.futures_basis import (bachelier_exchange, clean_price_from_yield, forward_yield,  # noqa: E402
                                          simulate_delivery)


def test_closed_form_price_matches_the_reference_pricer():
    for cpn, mat, at, y in ((4.0, "2033-08-15", "2026-10-01", 4.37), (2.25, "2052-02-15", "2026-12-31", 5.1)):
        ai = accrued_and_yield(np.nan, cpn, D(mat), D(at), 2)[0]
        ref = full_price_from_yield(y, cpn, D(mat), D(at), 2) - ai
        assert float(clean_price_from_yield(y, cpn, D(mat), D(at))) == pytest.approx(ref, abs=1e-10)
        assert forward_yield(ref, cpn, D(mat), D(at)) == pytest.approx(y, abs=1e-9)


def _two_bond_basket():
    """A short low-duration bond (CTD at low yields) and a long high-duration one."""
    deliv = D("2026-12-31")
    cpn, mat, cf = np.array([4.0, 4.0]), [D("2033-08-15"), D("2045-08-15")], np.array([0.85, 0.80])
    fy = np.array([4.4, 4.9])
    fwd = np.array([float(clean_price_from_yield(fy[i], cpn[i], mat[i], deliv)) for i in range(2)])
    return fwd, cf, fy, cpn, mat, deliv


def test_without_volatility_the_one_factor_model_is_deterministic():
    fwd, cf, fy, cpn, mat, deliv = _two_bond_basket()
    _, fair, share = simulate_delivery(fwd, cf, fy, cpn, mat, deliv, np.zeros(1000))
    assert fair == pytest.approx((fwd / cf).min()) and share[np.argmin(fwd / cf)] == 1.0


def test_the_quality_option_is_positive_and_grows_with_volatility():
    fwd, cf, fy, cpn, mat, deliv = _two_bond_basket()
    z = np.random.default_rng(0).standard_normal(5000)
    z = np.concatenate([z, -z])
    m0 = (fwd / cf).min()
    values = [m0 - simulate_delivery(fwd, cf, fy, cpn, mat, deliv, s * z)[1] for s in (20.0, 60.0, 120.0)]
    assert -1e-12 <= values[0] < values[1] < values[2]


def test_each_bond_is_centred_on_its_forward():
    fwd, cf, fy, cpn, mat, deliv = _two_bond_basket()
    z = np.random.default_rng(1).standard_normal(4000)
    implied, _, _ = simulate_delivery(fwd, cf, fy, cpn, mat, deliv, 40.0 * np.concatenate([z, -z]))
    assert np.allclose(implied.mean(axis=0) * cf, fwd)


def test_bachelier_exchange_limits():
    assert bachelier_exchange(-1.0, 0.0) == 0.0 and bachelier_exchange(0.5, 0.0) == 0.5
    assert bachelier_exchange(0.0, 1.0) == pytest.approx(1 / np.sqrt(2 * np.pi))


# ------------------------------------------------------------- M2 factor model
from infra.models.basis.factors import change_panel, fit_factor_model, simulate_shocks  # noqa: E402


def _panel(spread_kind: str, n_days: int = 400, n_bonds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Daily yield changes (bp): a common level random walk plus spreads that are either a
    random walk ("rw") or pure day-to-day noise that reverts ("noise")."""
    rng = np.random.default_rng(seed)
    level = rng.normal(0, 5, n_days)
    if spread_kind == "rw":
        spread_changes = rng.normal(0, 0.5, (n_days, n_bonds))
    else:  # spread level = iid noise around 0, so its changes are the difference of noise
        lv = rng.normal(0, 0.5, (n_days + 1, n_bonds))
        spread_changes = np.diff(lv, axis=0)
    return pd.DataFrame(level[:, None] + spread_changes, columns=[f"B{i}" for i in range(n_bonds)])


def test_spread_covariance_is_taken_at_the_horizon_not_scaled_from_daily():
    rw = fit_factor_model(_panel("rw"), 60, lam_slow=0.999)
    noise = fit_factor_model(_panel("noise"), 60, lam_slow=0.999)
    assert 0.6 < rw.variance_ratio < 1.6  # random-walk spreads: h-day variance ~ h x daily
    assert noise.variance_ratio < 0.1  # reverting noise: h-day variance ~ 2 x noise variance, not 60 x
    assert rw.sigma_level == pytest.approx(5.0, rel=0.25)


def test_young_bonds_borrow_their_predecessor_history():
    days = pd.bdate_range("2025-01-01", periods=60)
    y = pd.DataFrame({"OLD": np.linspace(4.0, 4.6, 60), "NEW": np.r_[np.full(40, np.nan), np.linspace(4.5, 4.7, 20)],
                      "OTHER": np.linspace(3.0, 3.3, 60)}, index=days)
    pred = {"NEW": "OLD", "OLD": None, "OTHER": None}
    mat = {"OLD": D("2035-01-01"), "NEW": D("2036-01-01"), "OTHER": D("2030-01-01")}
    dy, src = change_panel(y, ["NEW", "OTHER"], pred, mat, window=50)
    assert dy["NEW"].notna().all() and src["NEW"] == "predecessor OLD"
    assert dy["NEW"].iloc[0] == pytest.approx((y["OLD"].iloc[11] - y["OLD"].iloc[10]) * 100)


def test_shocks_share_the_level_draws_and_have_the_right_scale():
    fm = fit_factor_model(_panel("noise"), 20)
    z = np.random.default_rng(3).standard_normal(4000)
    z = np.concatenate([z, -z])
    sh = simulate_shocks(fm, 20, z, np.random.default_rng(4))
    assert sh.shape == (8000, 5)
    common = sh.mean(axis=1)
    assert np.corrcoef(common, z)[0, 1] > 0.99  # the level IS the shared draw
    assert common.std() == pytest.approx(fm.sigma_level * np.sqrt(20), rel=0.05)


# ------------------------------------------------------------- expected new issues
from infra.processing.futures_baskets import expected_issue_coupon, expected_issues  # noqa: E402

_TENORS = {"2y": ("Note", "2-Year"), "10y": ("Note", "10-Year")}


def _issues():
    rows = []
    for k, d in enumerate(pd.to_datetime(["2026-06-30", "2026-07-31", "2026-08-31"])):
        rows.append({"cusip": f"TWO{k}", "security_type": "Note", "original_term": "2-Year", "term_months": 24,
                     "issue_date": d, "maturity_date": d + pd.DateOffset(months=24) + pd.offsets.MonthEnd(0)})
    for k, d in enumerate(pd.to_datetime(["2026-02-17", "2026-05-15", "2026-08-17"])):
        rows.append({"cusip": f"TEN{k}", "security_type": "Note", "original_term": "10-Year", "term_months": 120,
                     "issue_date": d, "maturity_date": pd.Timestamp(year=d.year + 10, month=d.month, day=15)})
    return pd.DataFrame(rows)


def test_expected_issues_follow_each_tenors_own_cycle():
    e = expected_issues(_issues(), D("2026-09-01"), D("2026-12-31"), _TENORS)
    two = e[e["tenor"] == "2y"]
    # Oct 31 2026 is a Saturday: a month-end issue rolls FORWARD (as on 2019-07-01)
    assert list(two["issue_date"].dt.date.astype(str)) == ["2026-09-30", "2026-11-02", "2026-11-30", "2026-12-31"]
    assert two["maturity_date"].iloc[0] == D("2028-09-30") and set(two["predecessor"]) == {"TWO2"}
    ten = e[e["tenor"] == "10y"]
    assert list(ten["issue_date"]) == [D("2026-11-16")]  # Nov 15 2026 is a Sunday
    assert ten["maturity_date"].iloc[0] == D("2036-11-15")


def test_nothing_expected_before_as_of_or_after_until():
    assert expected_issues(_issues(), D("2026-10-01"), D("2026-10-31"), _TENORS).empty  # October's lands Nov 2
    e = expected_issues(_issues(), D("2026-10-01"), D("2026-11-05"), _TENORS)
    assert list(e["cusip"]) == ["NEW:2y:2026-11-02"]


def test_a_new_coupon_is_the_highest_eighth_at_or_below_the_yield():
    assert expected_issue_coupon(4.37) == 4.25 and expected_issue_coupon(4.375) == 4.375


def test_scoring_maps_an_expected_issue_to_the_note_it_became():
    from infra.models.basis.validate import resolve_expected_issues
    sec = pd.DataFrame({"cusip": ["REAL"], "security_type": ["Note"], "original_term": ["2-Year"],
                        "issue_date": [D("2026-10-30")]})
    out = resolve_expected_issues(pd.Series(["NEW:2y:2026-10-31", "OLD", "NEW:2y:2027-03-31"]), sec)
    assert list(out) == ["REAL", "OLD", "NEW:2y:2027-03-31"]


# ------------------------------------------------------------- timing options
from infra.analytics.delivery_timing import eom_switch_value, expected_max_normal, wildcard_value  # noqa: E402


def test_expected_max_normal_matches_simulation():
    z = np.random.default_rng(0).standard_normal(400_000)
    for a, v in ((1.0, 0.0), (0.5, 0.3), (2.0, -1.0)):
        assert expected_max_normal(a, v) == pytest.approx(np.maximum(a * z, v).mean(), abs=0.01)


def test_the_wild_card_is_an_optimal_bermudan_exercise():
    """Backward induction equals the best stopping rule found by brute force: exercise
    in window k when the payoff beats the value of waiting."""
    cf, sd, n = 0.6, 1.0, 5
    a = (1 / cf - 1) * sd
    v = wildcard_value(cf, sd, n)
    # thresholds from the same recursion, applied to simulated windows
    thresholds, w = [], 0.0
    for _ in range(n):
        thresholds.append(w)
        w = expected_max_normal(a, w)
    thresholds = thresholds[::-1]  # window k exercises when payoff > value of the remaining windows
    x = a * np.random.default_rng(1).standard_normal((200_000, n))
    paid = np.zeros(len(x)); done = np.zeros(len(x), bool)
    for k in range(n):
        ex = (~done) & (x[:, k] > thresholds[k])
        paid[ex] = x[ex, k]; done |= ex
    assert v == pytest.approx(paid.mean(), rel=0.02)
    assert wildcard_value(1.0, sd, n) == 0.0  # CF = 1: no tail, no wild card
    assert wildcard_value(0.6, sd, n) > wildcard_value(0.8, sd, n) > 0  # bigger tail, more value


def test_end_of_month_switch_needs_a_second_bond():
    z = np.random.default_rng(2).standard_normal(10_000)
    one = eom_switch_value(np.array([90.0]), np.array([0.9]), 100.0, 0, np.array([0.08]), 10.0, z)
    two = eom_switch_value(np.array([90.0, 61.0]), np.array([0.9, 0.6]), 100.0, 0, np.array([0.08, 0.15]), 10.0, z)
    assert one == 0.0 and two > 0.0


def test_wild_card_windows_can_differ_and_events_add_value():
    from infra.analytics.delivery_timing import window_kinds
    flat = wildcard_value(0.6, [1.0, 1.0, 1.0])
    assert flat == pytest.approx(wildcard_value(0.6, 1.0, 3))
    assert wildcard_value(0.6, [1.0, 2.0, 1.0]) > flat  # a high-variance (event) window adds value
    kinds = window_kinds(pd.to_datetime(["2026-03-18", "2026-03-31", "2026-04-30", "2026-04-15"]), [D("2026-03-18")])
    assert kinds == ["fomc", "quarter_end", "month_end", "ordinary"]


def test_fat_tailed_spreads_keep_the_variance_and_add_kurtosis():
    from scipy.stats import kurtosis
    fm = fit_factor_model(_panel("rw"), 20)
    z = np.zeros(200_000)  # no level: look at the spread part alone
    normal = simulate_shocks(fm, 20, z, np.random.default_rng(5))
    fat = simulate_shocks(fm, 20, z, np.random.default_rng(5), spread_df=6.0)
    assert fat.var(axis=0) == pytest.approx(normal.var(axis=0), rel=0.05)
    assert kurtosis(fat[:, 0]) > 2.0 and abs(kurtosis(normal[:, 0])) < 0.2
    with pytest.raises(ValueError):
        simulate_shocks(fm, 20, z[:10], np.random.default_rng(5), spread_df=2.0)


def test_expected_issues_are_priced_at_a_forward_yield_not_spot():
    """Regression (2026-10-02): an expected new issue takes the nearest deliverable's carry
    shift (forward - spot yield). Priced at spot it was ~20bp too cheap under negative
    carry and won the deferred ZT contract's CTD at ~100%."""
    from dataclasses import replace
    from infra.analytics.futures_basis import clean_price_from_yield
    from infra.models.basis.config import BASIS_MODELS
    from infra.models.basis.model import TIERS
    delivery, mat = pd.Timestamp("2024-12-02"), pd.Timestamp("2026-09-30")
    existing = pd.DataFrame({"contract": ["ZTZ4"], "delivery_kind": ["first"], "cusip": ["OLD"], "coupon": [0.875],
                             "maturity": [mat], "yield_eod": [3.85],
                             "fwd": [float(clean_price_from_yield(3.65, 0.875, mat, delivery))]})
    fut = pd.DataFrame({"cusip": ["NEW"], "maturity": [mat], "coupon": [3.875], "cf": [0.9651], "ref_yield": [3.88]})
    c = pd.Series({"contract": "ZTZ4", "delivery_kind": "first", "delivery": delivery})
    for carry, expected in ((True, 3.68), (False, 3.88)):
        m = TIERS["M2"](replace(BASIS_MODELS["M2"], future_issue_carry=carry))
        m._future, m._prepared_bonds = {"ZTZ4": fut}, existing
        x = m._extra_bonds(c).iloc[0]
        assert x["fwd_yield"] == pytest.approx(expected, abs=1e-4)
        assert x["fwd"] == pytest.approx(float(clean_price_from_yield(expected, 3.875, mat, delivery)), abs=1e-6)


def test_wildcard_wait_cost_lowers_value_and_zero_cost_is_the_old_induction():
    from infra.analytics.delivery_timing import wildcard_value
    sds = [0.25] * 20
    free = wildcard_value(0.66, sds)
    costly = wildcard_value(0.66, sds, wait_cost=0.004)
    assert wildcard_value(0.66, sds, wait_cost=0.0) == free
    assert 0 < costly < free
    # waiting prohibitively costly: deliver at the first window whatever the move -> ~0
    assert wildcard_value(0.66, sds, wait_cost=10.0) == pytest.approx(0.0, abs=1e-9)


def test_mark_noise_is_measured_from_reversal_and_removed_from_idio():
    rng = np.random.default_rng(3)
    t, n = 600, 6
    true = np.cumsum(rng.normal(0, 1.0, (t, n)), axis=0)
    marks = true + rng.normal(0, 2.0, (t, n)) * (np.arange(n) == 0)  # bond 0: noisy marks, s = 2bp
    dy = pd.DataFrame(np.diff(marks, axis=0), columns=[f"B{i}" for i in range(n)])
    plain = fit_factor_model(dy, 20, k=1)
    clean = fit_factor_model(dy, 20, k=1, noise_removal=2.0)
    assert plain.noise_var[0] == pytest.approx(4.0, rel=0.35) and plain.noise_var[1:].max() < 1.0
    assert clean.psi[0] < plain.psi[0] - 4.0 and clean.psi[1:] == pytest.approx(plain.psi[1:], rel=0.2)
