"""Offline tests for infra/analytics (Black-76, forward/discount, smile, RND). No API."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.analytics.black76 import implied_vol, price
from infra.analytics.forward import implied_forward_and_discount
from infra.analytics.rnd import extract_rnd
from infra.analytics.smile import fit_smile

D = pd.Timestamp


# ------------------------------------------------------------------------- black76
def test_put_call_parity_holds_exactly_for_our_own_pricer():
    F, T, sigma, discount = 96.0, 0.1, 0.02, 0.999
    for K in [94.0, 95.5, 96.0, 96.5, 98.0]:
        c, p = price(F, K, T, sigma, discount, "C"), price(F, K, T, sigma, discount, "P")
        assert abs((c - p) - discount * (F - K)) < 1e-10


def test_at_expiry_prices_intrinsic_value():
    assert price(96.0, 94.0, 0.0, 0.02, 1.0, "C") == 2.0
    assert price(96.0, 98.0, 0.0, 0.02, 1.0, "C") == 0.0
    assert price(96.0, 98.0, 0.0, 0.02, 1.0, "P") == 2.0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        price(96.0, 95.0, 0.1, 0.02, 1.0, "X")


def test_implied_vol_round_trips():
    F, K, T, discount = 96.0, 95.7, 0.08, 0.998
    for true_sigma in [0.005, 0.02, 0.08]:
        mkt = price(F, K, T, true_sigma, discount, "C")
        recovered = implied_vol(mkt, F, K, T, discount, "C")
        assert abs(recovered - true_sigma) < 1e-6


def test_implied_vol_returns_nan_not_raise_when_unbracketed():
    # a price above the max any vol in [lo,hi] could produce
    out = implied_vol(1000.0, 96.0, 95.7, 0.08, 0.998, "C")
    assert np.isnan(out)


# ------------------------------------------------------------------------- forward
def test_implied_forward_and_discount_recovers_known_values():
    true_F, true_discount, sigma, T = 96.09, 0.9993, 0.015, 0.05
    strikes = np.arange(94.0, 98.0, 0.0625)
    rows = []
    for K in strikes:
        rows.append(("C", K, price(true_F, K, T, sigma, true_discount, "C")))
        rows.append(("P", K, price(true_F, K, T, sigma, true_discount, "P")))
    chain = pd.DataFrame(rows, columns=["option_type", "strike", "settlement_price"])
    F, discount = implied_forward_and_discount(chain)
    assert abs(F - true_F) < 1e-6
    assert abs(discount - true_discount) < 1e-6


def test_forward_requires_both_call_and_put():
    with pytest.raises(ValueError):
        implied_forward_and_discount(pd.DataFrame({"option_type": ["C"], "strike": [96.0], "settlement_price": [1.0]}))


def test_forward_requires_at_least_two_strikes():
    chain = pd.DataFrame({
        "option_type": ["C", "P"], "strike": [96.0, 96.0], "settlement_price": [1.0, 1.0],
    })
    with pytest.raises(ValueError):
        implied_forward_and_discount(chain)


# ------------------------------------------------------------------------- smile
def test_fit_smile_recovers_smooth_curve_at_knots():
    strikes = np.linspace(90, 100, 20)
    vols = 0.02 + 0.0002 * (strikes - 96) ** 2  # a simple smile shape
    smile = fit_smile(strikes, vols)
    assert np.allclose(smile(strikes), vols, atol=1e-9)


def test_fit_smile_drops_nan_vols():
    strikes = np.linspace(90, 100, 10)
    vols = np.full(10, 0.02)
    vols[3] = np.nan
    smile = fit_smile(strikes, vols)  # must not raise
    assert np.isfinite(smile(95.0))


def test_fit_smile_requires_minimum_points():
    with pytest.raises(ValueError):
        fit_smile(np.array([90.0, 91.0]), np.array([0.02, 0.02]))


# --------------------------------------------------------------------------- rnd
def _synthetic_chain(F, discount, sigma, T, strikes) -> pd.DataFrame:
    """A chain with NO smile (flat sigma) - Black-76's terminal distribution is then
    an exact, closed-form lognormal, giving a known-correct answer to test against."""
    rows = []
    day = D("2026-01-01")
    expiry = day + pd.Timedelta(days=round(T * 365))
    for K in strikes:
        rows.append(("SYN", "C", K, price(F, K, T, sigma, discount, "C"), day, expiry))
        rows.append(("SYN", "P", K, price(F, K, T, sigma, discount, "P"), day, expiry))
    return pd.DataFrame(rows, columns=["underlying", "option_type", "strike", "settlement_price", "timestamp", "expiry"])


def _lognormal_pdf(x, F, sigma, T):
    """Closed-form terminal density under Black-76's own assumption (F_T lognormal,
    martingale under the forward measure) - the known-correct answer for a flat-vol chain."""
    return (1.0 / (x * sigma * np.sqrt(2 * np.pi * T))) * np.exp(
        -((np.log(x / F) + 0.5 * sigma**2 * T) ** 2) / (2 * sigma**2 * T)
    )


def test_extract_rnd_recovers_the_known_lognormal_density_for_a_flat_vol_chain():
    F, discount, sigma, T = 96.0, 0.999, 0.02, 0.1
    strikes = np.arange(90.0, 102.0, 0.0625)
    chain = _synthetic_chain(F, discount, sigma, T, strikes)
    out = extract_rnd(chain)

    assert abs(np.sum((out["density"].values[:-1] + out["density"].values[1:]) / 2
                      * np.diff(out["strike"].values)) - 1.0) < 1e-3  # integrates to ~1
    assert (out["density"] >= 0).all()

    # compare against the closed-form answer at a few points well inside the fitted
    # range (away from the truncated-support edges, where error is expected to be larger)
    for x in [94.0, 96.0, 98.0]:
        idx = (out["strike"] - x).abs().idxmin()
        got = out.loc[idx, "density"]
        want = _lognormal_pdf(x, F, sigma, T)
        assert abs(got - want) / want < 0.05, f"at K={x}: got {got}, want {want}"


def test_extract_rnd_peak_is_near_the_forward():
    F, discount, sigma, T = 96.0, 0.999, 0.02, 0.1
    strikes = np.arange(90.0, 102.0, 0.0625)
    chain = _synthetic_chain(F, discount, sigma, T, strikes)
    out = extract_rnd(chain)
    peak_strike = out.loc[out["density"].idxmax(), "strike"]
    assert abs(peak_strike - F) < 0.5


def test_extract_rnd_requires_single_underlying_expiry_day():
    F, discount, sigma, T = 96.0, 0.999, 0.02, 0.1
    strikes = np.arange(94.0, 98.0, 0.25)
    chain = _synthetic_chain(F, discount, sigma, T, strikes)
    chain2 = chain.copy()
    chain2["timestamp"] = D("2026-01-02")
    mixed = pd.concat([chain, chain2], ignore_index=True)
    with pytest.raises(ValueError):
        extract_rnd(mixed)


def test_extract_rnd_rejects_already_expired_option():
    F, discount, sigma, T = 96.0, 0.999, 0.02, 0.1
    strikes = np.arange(94.0, 98.0, 0.25)
    chain = _synthetic_chain(F, discount, sigma, T, strikes)
    chain["expiry"] = chain["timestamp"]  # expiry == valuation day -> T=0
    with pytest.raises(ValueError):
        extract_rnd(chain)
