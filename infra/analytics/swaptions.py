"""European swaptions on an OIS curve - normal (Bachelier) pricing, implied vol, gamma.
Pure, no I/O. Rates in DECIMAL inside, vols returned in bp/yr (normal).

A swaption on a swap from ``start`` to ``maturity`` (annual ACT/360 fixed, SOFR OIS) is
priced off the forward swap rate F and the PV annuity A (both from the curve):
payer = A [ (F - K) N(d) + s sqrt(T) n(d) ], receiver = A [ (K - F) N(-d) + s sqrt(T) n(d) ],
d = (F - K) / (s sqrt(T)). Gamma (d2V / dF2) = A n(d) / (s sqrt(T)) - the same for a
payer and a receiver at one strike, which is why an open-interest ledger can do without
the side (the report's Call/Put label doesn't give it: infra.config.SwaptionSpec).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from infra.analytics import swap_curve as sc


def forward_annuity(curve: sc.OisCurve, trade_day, start, maturity) -> tuple[float, float]:
    """(forward swap rate, decimal; PV annuity) of an annual ACT/360 swap from ``start`` to
    ``maturity``: annual dates stepped back from maturity, a short first period folded in
    (no business-day adjustment)."""
    start, maturity = pd.Timestamp(start), pd.Timestamp(maturity)
    ends, k = [], 0
    while (e := maturity - pd.DateOffset(years=k)) > start + pd.Timedelta(days=15):
        ends.append(e)
        k += 1
    ends = ends[::-1]
    starts = [start] + ends[:-1]
    acc = np.array([(e - s).days / 360.0 for s, e in zip(starts, ends)])
    a = float(acc @ curve.discount(sc.year_fraction(trade_day, ends)))
    d0, d1 = curve.discount(sc.year_fraction(trade_day, [start, maturity]))
    return float((d0 - d1) / a), a


def normal_price(f: float, k: float, vol_bp: float, t: float, payer: bool) -> float:
    """Undiscounted price per unit annuity (rate units)."""
    s = vol_bp / 1e4 * np.sqrt(t)
    d = (f - k) / s
    w = 1.0 if payer else -1.0
    return float(w * (f - k) * norm.cdf(w * d) + s * norm.pdf(d))


def implied_normal_vol(price: float, f: float, k: float, t: float, payer: bool) -> float:
    """Normal vol (bp/yr) reproducing ``price`` (per unit annuity); NaN below intrinsic."""
    intrinsic = max((f - k) if payer else (k - f), 0.0)
    if not np.isfinite(price) or price <= intrinsic or t <= 0:
        return float("nan")
    try:
        return brentq(lambda v: normal_price(f, k, v, t, payer) - price, 1e-4, 2000.0, xtol=1e-8)
    except ValueError:
        return float("nan")


def normal_gamma(f: float, k: float, vol_bp: float, t: float, annuity: float) -> float:
    """d2V/dF2 per unit notional, F in decimal."""
    s = vol_bp / 1e4 * np.sqrt(t)
    return float(annuity * norm.pdf((f - k) / s) / s)
