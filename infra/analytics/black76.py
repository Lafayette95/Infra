"""Black-76 pricing for European options on a future - pure math, no I/O.

Used as the pricing model behind risk-neutral density extraction (infra.analytics.rnd).
SR3 options are actually AMERICAN-exercise (CLAUDE.md section 9) - this module treats
them as European throughout, a documented approximation, not an oversight.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def price(forward: float, strike: float, T: float, sigma: float, discount: float, option_type: str) -> float:
    """Black-76 premium for one option on a future.

    ``forward``: the future/forward price. ``T``: time to the OPTION's own expiry, in
    years. ``sigma``: annualized volatility. ``discount``: discount factor (e.g. from
    ``infra.analytics.forward.implied_forward_and_discount``) applied to the payoff.
    ``option_type``: "C" or "P".
    """
    if option_type not in ("C", "P"):
        raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")
    if T <= 0 or sigma <= 0:
        # At/after expiry, or a degenerate zero-vol request: intrinsic value only.
        intrinsic = max(forward - strike, 0.0) if option_type == "C" else max(strike - forward, 0.0)
        return discount * intrinsic
    d1 = (np.log(forward / strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "C":
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_vol(
    market_price: float, forward: float, strike: float, T: float, discount: float, option_type: str,
    *, lo: float = 1e-6, hi: float = 5.0,
) -> float:
    """Invert ``price`` for sigma via Brent's method.

    Returns NaN (never raises) when no root brackets the market price - e.g. a
    far-out-of-the-money option sitting at its exchange minimum tick, which can fall
    outside what any volatility in ``[lo, hi]`` would produce. Callers (infra.analytics
    .smile.fit_smile) are expected to drop NaNs before fitting.
    """
    def objective(sigma: float) -> float:
        return price(forward, strike, T, sigma, discount, option_type) - market_price

    try:
        lo_val, hi_val = objective(lo), objective(hi)
        if not np.isfinite(lo_val) or not np.isfinite(hi_val) or lo_val * hi_val > 0:
            return float("nan")
        return float(brentq(objective, lo, hi, xtol=1e-10))
    except (ValueError, RuntimeError):
        return float("nan")
