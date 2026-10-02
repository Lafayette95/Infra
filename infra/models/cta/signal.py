"""Pure building blocks of the CTA trend model (no I/O, no fitted state of their own).

The UBS recipe (Part II, Q7), step by step:

(i)   crossover of exponentially weighted moving averages of the price, at several
      speeds:  x_k = EWMA(level, n_short_k) - EWMA(level, n_long_k);
(ii)  normalised by the underlying's rolling volatility (1y): z_k = x_k / sigma_1y,
      in units of one day's move;
(iii) a response function that makes the result ~uniform on [-1, 1]. Here each pair is
      first put on a common scale (its own RMS, fitted), the pairs are averaged into one
      score, and the response is applied once to that score, so the FINAL signal is the
      uniform one.

Then the position (Q7 b-d): signal x asset-class weight x liquidity factor x portfolio
vol scaling / short-term vol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri


def ewma(level: pd.Series, n: int, init: float | None = None) -> pd.Series:
    """EWMA with weight decay (n-1)/n (alpha = 1/n), recursive (``adjust=False``).

    ``init``: the EWMA's value just before ``level``'s first row, to continue a
    recursion from a stored state (``predict`` after ``fit``). Without it the EWMA
    starts at the first level."""
    alpha = 1.0 / n
    if init is None:
        return level.ewm(alpha=alpha, adjust=False).mean()
    seeded = pd.concat([pd.Series([float(init)]), level.reset_index(drop=True)], ignore_index=True)
    out = seeded.ewm(alpha=alpha, adjust=False).mean().iloc[1:]
    out.index = level.index
    return out


def rolling_vol(returns: pd.Series, window: int, min_obs: int, tail: np.ndarray | None = None) -> pd.Series:
    """Rolling standard deviation of ``returns`` (per observation, not annualised).

    ``tail``: returns just before ``returns``' first row (from a stored state), so the
    window continues exactly across a fit/predict boundary."""
    if tail is None or len(tail) == 0:
        return returns.rolling(window, min_periods=min_obs).std()
    full = pd.concat([pd.Series(tail, dtype="float64"), returns.reset_index(drop=True)], ignore_index=True)
    out = full.rolling(window, min_periods=min_obs).std().iloc[len(tail):]
    out.index = returns.index
    return out


def crossover_z(emas: dict[int, np.ndarray | float], pairs, norm_vol) -> list:
    """Per pair, the vol-normalised crossover ``(EWMA_s - EWMA_l) / norm_vol``.
    Works on scalars, arrays (Monte Carlo paths) and Series alike."""
    return [(emas[s] - emas[l]) / norm_vol for s, l in pairs]


def combined_score(z: list, pair_scale: np.ndarray, weights: np.ndarray):
    """Weighted average of the pairs' crossovers, each divided by its fitted RMS."""
    return sum(w * zk / sc for zk, sc, w in zip(z, pair_scale, weights))


def rms(x: np.ndarray) -> float:
    """Root mean square around 0, not the mean: a crossover's scale without removing
    its average trend, so 0 stays 'no trend'."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if len(x) else np.nan


# ---------------------------------------------------------------- response functions
# Each ``fit_*`` takes the fitted sample of scores and returns a callable mapping scores
# (scalar / array) to signals in [-1, 1]; NaN in -> NaN out.

@dataclass(frozen=True)
class EcdfResponse:
    """2F(score) - 1, F the empirical CDF of the fitted scores (midpoint plotting
    positions, linear between sample points, capped at +-1 outside the sample range).
    Uniform on [-1, 1] in sample by construction - UBS's stated goal for step (iii).
    ``gain`` != 1 steepens it through the normal quantile: 2*Phi(gain * Phi^-1(F)) - 1."""
    xs: np.ndarray
    ps: np.ndarray
    gain: float = 1.0

    def __call__(self, score):
        x = np.asarray(score, float)
        f = np.interp(x, self.xs, self.ps, left=0.0, right=1.0)
        out = 2.0 * f - 1.0 if self.gain == 1.0 else 2.0 * ndtr(self.gain * ndtri(f)) - 1.0
        return np.where(np.isfinite(x), out, np.nan)


@dataclass(frozen=True)
class NormalResponse:
    """2*Phi(gain * score / scale) - 1: uniform if the scores are N(0, scale^2), gain 1."""
    scale: float
    gain: float = 1.0

    def __call__(self, score):
        x = np.asarray(score, float)
        return 2.0 * ndtr(self.gain * x / self.scale) - 1.0


@dataclass(frozen=True)
class TanhResponse:
    """tanh(0.851 * gain * score / scale): the logistic approximation of 2*Phi - 1 (same
    shape, the 'hyperbolic' cap/floor of UBS Fig. 69)."""
    scale: float
    gain: float = 1.0

    def __call__(self, score):
        x = np.asarray(score, float)
        return np.tanh(0.851 * self.gain * x / self.scale)


def fit_ecdf(scores: np.ndarray, gain: float = 1.0) -> EcdfResponse:
    x = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    n = len(x)
    p = (np.arange(n) + 0.5) / n
    xs, inv = np.unique(x, return_inverse=True)
    ps = np.bincount(inv, weights=p) / np.bincount(inv)  # ties -> their mean position
    return EcdfResponse(xs, ps, gain)


def fit_normal(scores: np.ndarray, gain: float = 1.0) -> NormalResponse:
    return NormalResponse(rms(scores), gain)


def fit_tanh(scores: np.ndarray, gain: float = 1.0) -> TanhResponse:
    return TanhResponse(rms(scores), gain)


RESPONSES = {"ecdf": fit_ecdf, "normal": fit_normal, "tanh": fit_tanh}


def fit_response(kind: str, scores: np.ndarray, gain: float = 1.0):
    try:
        return RESPONSES[kind](scores, gain)
    except KeyError:
        raise KeyError(f"unknown response {kind!r}; known: {sorted(RESPONSES)}") from None


# ---------------------------------------------------------------- sizing

def risk_weight(class_weight: float, liquidity: float, pvs) -> float:
    """The constant part of UBS's position formula: asset-class weight x liquidity factor
    x portfolio vol scaling. Position = risk_weight * signal / vol."""
    return class_weight * liquidity * pvs


def portfolio_vol_scaling(unit_positions: pd.DataFrame, returns: pd.DataFrame, target: float | None,
                          window: int, min_obs: int, periods_per_year: int) -> pd.Series:
    """UBS d): the multiplier taking the portfolio to its volatility target, estimated
    on a rolling ``window`` (point in time: the value at t uses returns up to t).

    ``unit_positions``: positions before the multiplier (weight x liquidity x signal /
    vol), held from one observation to the next. ``returns``: each asset's change. Both on
    one calendar; a missing return is no move, a missing position is flat."""
    idx = unit_positions.index
    if target is None:
        return pd.Series(1.0, index=idx)
    held = unit_positions.ffill().shift(1).fillna(0.0)
    strat = (held * returns.reindex(index=idx, columns=unit_positions.columns).fillna(0.0)).sum(axis=1)
    started = (held != 0).any(axis=1).cummax()
    vol = strat.where(started).rolling(window, min_periods=min_obs).std() * np.sqrt(periods_per_year)
    return (target / vol).replace([np.inf, -np.inf], np.nan)
