"""Ornstein-Uhlenbeck (OU) mean reversion: estimation and the trade metrics (pure functions).

The process: dX = kappa (m - X) dt + sigma dW. Estimated from a level series sampled once a
period (a trading day) as an AR(1), X(t+1) = a + b X(t) + e: kappa = -ln b, m = a / (1 - b),
the equilibrium (stationary) sd ``sd_eq = sd(e) / sqrt(1 - b^2)``, half-life = ln 2 / kappa.

Every metric below is computed in STANDARDISED units: the s-score z = (X - m) / sd_eq and time in
units of 1 / kappa, where the process is dz = -z dt + sqrt(2) dW (stationary sd 1). Then a metric
depends on the s-score (and the trade's levels in s-score units) only - so it can be tabulated
once and interpolated - and converts back to days by dividing by kappa.

* ``fit_ar1`` - the AR(1) estimate (optionally bias-corrected: the least-squares b is biased
  toward FASTER reversion in short samples, Kendall 1954: E[b_hat] ~ b - (1 + 3b) / n).
* ``ou_params`` - kappa, m, sd_eq, half-life from an AR(1) fit.
* horizon forecasts: ``expected_level`` and ``horizon_sd`` (exact for the discrete AR(1)).
* first passage to a level: ``fpt_cdf`` / ``fpt_median`` (closed form for the MEAN: the OU is a
  time-changed Brownian motion, so hitting the mean is Levy's first-passage law in the changed
  clock) and ``expected_hitting_time`` (any level, from the scale and speed densities).
* two barriers: ``p_hit_upper_first`` (scale function) and ``expected_exit_time`` (Green's function).
* ``cycle_rate`` / ``optimal_entry``: expected return per unit time of the cycle "wait for the
  entry level, hold to the exit level" (Bertram 2010, "Analytic solutions for optimal statistical
  arbitrage trading", Physica A 389).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import integrate
from scipy.special import erfc, erfcinv, erfcx, erfi

SQRT_PI_2 = np.sqrt(np.pi / 2.0)


# --------------------------------------------------------------------------- estimation
@dataclass(frozen=True)
class AR1Fit:
    a: float
    b: float
    se_b: float
    sd_e: float          # innovation sd (per period)
    n: int               # transitions used

    @property
    def adf_t(self) -> float:
        """(b - 1) / se(b): the Dickey-Fuller statistic (with a constant; 5% critical value ~ -2.86)."""
        return (self.b - 1.0) / self.se_b if self.se_b > 0 else np.nan


def fit_ar1(x: np.ndarray, *, bias_correct: bool = False) -> AR1Fit:
    """Least squares X(t+1) = a + b X(t) + e on consecutive finite pairs."""
    x = np.asarray(x, dtype="float64")
    x0, x1 = x[:-1], x[1:]
    ok = np.isfinite(x0) & np.isfinite(x1)
    x0, x1 = x0[ok], x1[ok]
    n = len(x0)
    if n < 3:
        return AR1Fit(np.nan, np.nan, np.nan, np.nan, n)
    d0 = x0 - x0.mean()
    sxx = float(d0 @ d0)
    if sxx <= 0:
        return AR1Fit(np.nan, np.nan, np.nan, np.nan, n)
    b = float(d0 @ (x1 - x1.mean()) / sxx)
    a = float(x1.mean() - b * x0.mean())
    e = x1 - a - b * x0
    sd_e = float(np.sqrt(e @ e / (n - 2)))
    se_b = sd_e / np.sqrt(sxx)
    if bias_correct:
        b = min(b + (1.0 + 3.0 * b) / n, 0.999999) if b < 1 else b
        a = float(x1.mean() - b * x0.mean())
    return AR1Fit(a, b, float(se_b), sd_e, n)


@dataclass(frozen=True)
class OUParams:
    kappa: float         # per period
    m: float             # equilibrium level
    sd_eq: float         # stationary sd
    sd_e: float          # innovation sd per period
    b: float

    @property
    def half_life(self) -> float:
        return np.log(2.0) / self.kappa if self.kappa > 0 else np.inf

    @property
    def reverting(self) -> bool:
        return bool(np.isfinite(self.kappa) and self.kappa > 0)


def ou_params(f: AR1Fit) -> OUParams:
    """OU parameters of an AR(1) fit; not mean reverting (b >= 1 or b <= 0) -> kappa NaN / inf."""
    if not np.isfinite(f.b) or f.b >= 1.0 or f.b <= 0.0:
        return OUParams(np.nan, np.nan, np.nan, f.sd_e, f.b)
    return OUParams(-np.log(f.b), f.a / (1.0 - f.b), f.sd_e / np.sqrt(1.0 - f.b ** 2), f.sd_e, f.b)


# --------------------------------------------------------------------------- horizon forecasts
def expected_level(x, p: OUParams, h):
    """E[X(t+h) | X(t) = x]."""
    return p.m + (np.asarray(x, dtype="float64") - p.m) * p.b ** np.asarray(h, dtype="float64")


def horizon_sd(p: OUParams, h):
    """sd of X(t+h) given X(t)."""
    return p.sd_eq * np.sqrt(1.0 - p.b ** (2.0 * np.asarray(h, dtype="float64")))


# --------------------------------------------------------------------------- first passage (standardised)
def fpt_cdf(z0, t):
    """P(the standardised OU from z0 has hit 0 by time t) (t in units of 1 / kappa).
    X_t = e^-t (z0 + B(tau(t))) with tau(t) = e^{2t} - 1, so it is Levy's law in the changed clock."""
    z0, t = np.abs(np.asarray(z0, dtype="float64")), np.asarray(t, dtype="float64")
    tau = np.expm1(2.0 * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tau > 0, erfc(z0 / np.sqrt(2.0 * tau)), 0.0)


def fpt_median(z0):
    """Median first-passage time to the mean (units of 1 / kappa)."""
    z0 = np.abs(np.asarray(z0, dtype="float64"))
    tau = z0 ** 2 / (2.0 * erfcinv(0.5) ** 2)
    return 0.5 * np.log1p(tau)


def _from_above(y):
    """s(y) * integral_y^inf m(z) dz with s = e^{y^2/2}, m = e^{-z^2/2}: sqrt(pi/2) erfcx(y / sqrt 2)."""
    return SQRT_PI_2 * erfcx(y / np.sqrt(2.0))


def expected_hitting_time(z0: float, target: float) -> float:
    """E[first time the standardised OU from z0 hits ``target``] (units of 1 / kappa); the other
    side is unbounded. Moving down: integral_target^z0 s(y) integral_y^inf m; up: the mirror."""
    if z0 == target:
        return 0.0
    if z0 < target:                       # by symmetry of the process around 0
        z0, target = -z0, -target
    val, _ = integrate.quad(_from_above, target, z0, limit=200)
    return float(val)


# --------------------------------------------------------------------------- two barriers (standardised)
def _scale(z):
    """S(z) = integral_0^z e^{y^2/2} dy."""
    return SQRT_PI_2 * erfi(np.asarray(z, dtype="float64") / np.sqrt(2.0))


def p_hit_upper_first(z0, lower: float, upper: float):
    """P(hit ``upper`` before ``lower`` | start z0), lower < z0 < upper."""
    s0, sl, su = _scale(z0), _scale(lower), _scale(upper)
    return (s0 - sl) / (su - sl)


def expected_exit_time(z0: float, lower: float, upper: float) -> float:
    """E[time to leave (lower, upper) | z0]: integral G(z0, y) m(y) dy with the Green's function
    G(x, y) = (S(min) - S(lower)) (S(upper) - S(max)) / (S(upper) - S(lower)), m(y) = e^{-y^2/2}."""
    if not lower < z0 < upper:
        return 0.0
    sl, su = float(_scale(lower)), float(_scale(upper))
    s0 = float(_scale(z0))
    g = lambda y: ((min(s0, float(_scale(y))) - sl) * (su - max(s0, float(_scale(y)))) / (su - sl)  # noqa: E731
                   * np.exp(-0.5 * y * y))
    a, _ = integrate.quad(g, lower, z0, limit=200)
    b, _ = integrate.quad(g, z0, upper, limit=200)
    return float(a + b)


# --------------------------------------------------------------------------- the trade cycle (standardised)
def cycle_rate(entry: float, exit_: float, cost: float = 0.0) -> float:
    """Expected return per unit time (s-score units per 1 / kappa) of: wait at ``exit_`` until the
    level reaches ``entry``, then hold until it is back at ``exit_`` (entry > exit_ >= -entry: a
    short at a high s-score; the long side is the mirror). ``cost`` per round trip, s-score units."""
    t = expected_hitting_time(exit_, entry) + expected_hitting_time(entry, exit_)
    return (entry - exit_ - cost) / t if t > 0 else np.nan


@lru_cache(maxsize=256)
def optimal_entry(cost: float = 0.0, exit_: float | None = None) -> float:
    """The entry s-score maximising ``cycle_rate`` (Bertram's symmetric rule when ``exit_`` is None:
    exit at -entry, i.e. the opposite extreme; else a fixed exit level)."""
    grid = np.linspace(0.05, 4.0, 160)
    rates = [cycle_rate(e, -e if exit_ is None else exit_, cost) if (exit_ is None or e > exit_) else -np.inf
             for e in grid]
    return float(grid[int(np.nanargmax(rates))])


# --------------------------------------------------------------------------- tables over the s-score
@lru_cache(maxsize=64)
def _table(kind: str, exit_: float, stop_width: float):
    grid = np.linspace(0.0, 6.0, 241)
    if kind == "mean_fpt":
        vals = [expected_hitting_time(z, 0.0) for z in grid]
    elif kind == "p_target":
        vals = [p_hit_upper_first(-z, -z - stop_width, -exit_) if z > exit_ else 1.0 for z in grid]
    elif kind == "exit_time":
        vals = [expected_exit_time(-z, -z - stop_width, -exit_) if z > exit_ else 0.0 for z in grid]
    else:
        raise ValueError(kind)
    return grid, np.asarray(vals, dtype="float64")


def mean_fpt_to_mean(z):
    """E[first passage time to the mean] from s-score z (units of 1 / kappa), tabulated."""
    g, v = _table("mean_fpt", 0.0, 0.0)
    return np.interp(np.minimum(np.abs(np.asarray(z, dtype="float64")), g[-1]), g, v)


def p_target_before_stop(z, exit_: float, stop_width: float):
    """A position fading s-score z: P(it reaches the target |s| = ``exit_`` before the stop at |s| =
    |z| + ``stop_width``) - by symmetry a function of |z|."""
    g, v = _table("p_target", float(exit_), float(stop_width))
    return np.interp(np.minimum(np.abs(np.asarray(z, dtype="float64")), g[-1]), g, v)


def expected_trade_time(z, exit_: float, stop_width: float):
    """E[time until the faded position exits at the target or the stop] (units of 1 / kappa)."""
    g, v = _table("exit_time", float(exit_), float(stop_width))
    return np.interp(np.minimum(np.abs(np.asarray(z, dtype="float64")), g[-1]), g, v)
