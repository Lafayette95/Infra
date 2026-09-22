"""Fit a smooth implied-volatility smile to observed strikes - cubic spline.

Chosen over an SVI parameterization for v1 (CLAUDE.md section 9): simpler, no
per-day nonlinear calibration, adequate given the dense real strike grids observed
(100+ strikes per SR3 quarterly expiry).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def fit_smile(strikes: np.ndarray, vols: np.ndarray) -> CubicSpline:
    """A callable ``vol(strike)`` fitted to the observed points.

    NaN vols (failed implied-vol inversions - typically far-out-of-the-money options
    sitting at the exchange's minimum tick, see infra.analytics.black76.implied_vol)
    are dropped before fitting. Duplicate strikes are averaged. Extrapolation outside
    the observed strike range is disabled (``extrapolate=False``, returns NaN there) -
    infra.analytics.rnd restricts its output grid to the observed range for exactly
    this reason.
    """
    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(vols, dtype=float)
    mask = np.isfinite(vols) & np.isfinite(strikes)
    strikes, vols = strikes[mask], vols[mask]

    order = np.argsort(strikes)
    strikes, vols = strikes[order], vols[order]
    uniq_strikes, inverse = np.unique(strikes, return_inverse=True)
    if len(uniq_strikes) < len(strikes):
        vols = np.array([vols[inverse == i].mean() for i in range(len(uniq_strikes))])
        strikes = uniq_strikes

    if len(strikes) < 4:
        raise ValueError(f"need at least 4 valid (strike, vol) points to fit a smile, got {len(strikes)}")
    return CubicSpline(strikes, vols, extrapolate=False)
