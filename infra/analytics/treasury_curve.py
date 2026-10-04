"""US Treasury zero curve from per-CUSIP prices - pure, no I/O (sub-project
``infra/analytics/CURVES.md`` -> see infra/models/curves/CLAUDE.md for methodology).

Two fitting METHODS, side by side (``FIT_METHODS``), both on the bonds' actual cash flows
(so a bond's coupon doesn't bias the curve the way a yield-to-maturity regression does):

* ``spline`` - the DISCOUNT FUNCTION as a cubic regression spline, D(t) = 1 + sum b_k f_k(t)
  (basis t, t^2, t^3, (t - k)^3_+ at ``KNOTS``; D(0) = 1 built in). Prices are LINEAR in b,
  so it is weighted least squares, and each bond's LEAVE-ONE-OUT residual comes in closed
  form, e_i / (1 - h_ii) - no refit per bond.
* ``svensson`` - the Fed's Gurkaynak-Sack-Wright form: continuously compounded zero rate
  z(t) = b0 + b1 L1 + b2 (L1 - e1) + b3 (L2 - e2), L = (1 - e^{-t/tau}) / (t/tau), e =
  e^{-t/tau}; nonlinear least squares, warm-started from a previous fit.

Weights: price errors / modified duration, so the fit is ~in yield terms (``fit_*``
minimise sum ((P_model - P_market) / (P x D_mod))^2).

Per bond (``bond_metrics``), in yield bp, positive = GAIN to a holder:
* ``zspread_bp``      - the constant added to the curve's zero rates that reprices the bond
                        (negative = rich); ``zspread_loo_bp`` its leave-one-out version
                        (spline: the closed form; the bond doesn't pull the curve to itself).
* ``carry_curve_bp``  - curve carry to ``horizon_days``: today's yield minus the yield of the
                        curve-implied FORWARD price (funding at the curve's own short rate).
* ``rolldown_bp``     - today's yield minus the yield at the horizon if the curve and the
                        z-spread don't change (the bond is just shorter).
Repo carry (funding v1, per CUSIP with specialness) is NOT here - it needs the funding
layer (``infra.analytics.financing``); see the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from infra.processing.treasury_prices import coupon_dates

KNOTS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0)
YEAR = 365.25


# ------------------------------------------------------------------ cash flows
def cash_flows(coupon: float, maturity, settle, per_year: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(times in years from ``settle``, amounts per 100 face) of the remaining coupons and
    the principal."""
    dates = coupon_dates(pd.Timestamp(maturity), pd.Timestamp(settle), per_year)[1:]
    t = np.array([(d - pd.Timestamp(settle)).days / YEAR for d in dates])
    a = np.full(len(dates), coupon / per_year)
    a[-1] += 100.0
    return t, a


def ytm(dirty: float, t: np.ndarray, a: np.ndarray, per_year: int = 2, guess: float = 0.04) -> float:
    """Yield (%, compounded ``per_year``) that reprices the cash flows - a curve-free
    yield used to express everything in bp of the bond's own yield. Newton with the
    analytic derivative (3-4 steps; bracketing Brent as the fallback): the curve build
    solves ~2,700 of these a day and Brent alone made it 1.8s a day."""
    y = guess
    for _ in range(20):
        v = (1 + y / per_year) ** (-per_year * t)
        f = float((a * v).sum()) - dirty
        df = float((-t * a * v / (1 + y / per_year)).sum())
        if df == 0:
            break
        step = f / df
        y -= step
        if abs(step) < 1e-12:
            return 100.0 * y
    g = lambda yy: float((a * (1 + yy / per_year) ** (-per_year * t)).sum()) - dirty
    return 100.0 * brentq(g, -0.05, 0.5)


def modified_duration(dirty: float, t: np.ndarray, a: np.ndarray, y_pct: float, per_year: int = 2) -> float:
    y = y_pct / 100
    v = (1 + y / per_year) ** (-per_year * t)
    return float((t * a * v).sum() / dirty / (1 + y / per_year))


# ------------------------------------------------------------------ spline
def _basis(t: np.ndarray, knots=KNOTS) -> np.ndarray:
    t = np.asarray(t, dtype="float64")
    return np.column_stack([t, t ** 2, t ** 3] + [np.clip(t - k, 0, None) ** 3 for k in knots])


@dataclass
class SplineCurve:
    beta: np.ndarray
    knots: tuple = KNOTS

    def discount(self, t) -> np.ndarray:
        return 1.0 + _basis(np.atleast_1d(t), self.knots) @ self.beta


def fit_spline(bonds: list[tuple[np.ndarray, np.ndarray]], dirty: np.ndarray, weight: np.ndarray, knots=KNOTS):
    """Weighted LS of the discount function. Returns (curve, price residuals market -
    model, leverages h_ii)."""
    X = np.array([(a[:, None] * _basis(t, knots)).sum(axis=0) for t, a in bonds])
    y = dirty - np.array([a.sum() for _, a in bonds])
    sw = np.sqrt(weight)
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    xtx_inv = np.linalg.pinv(Xw.T @ Xw)
    h = np.einsum("ij,jk,ik->i", Xw, xtx_inv, Xw)
    return SplineCurve(beta, knots), y - X @ beta, h


# ------------------------------------------------------------------ Svensson
@dataclass
class SvenssonCurve:
    params: np.ndarray  # b0, b1, b2, b3 (decimal), tau1, tau2 (years)

    def zero(self, t) -> np.ndarray:
        b0, b1, b2, b3, t1, t2 = self.params
        t = np.maximum(np.atleast_1d(np.asarray(t, dtype="float64")), 1e-6)
        e1, e2 = np.exp(-t / t1), np.exp(-t / t2)
        l1, l2 = (1 - e1) / (t / t1), (1 - e2) / (t / t2)
        return b0 + b1 * l1 + b2 * (l1 - e1) + b3 * (l2 - e2)

    def discount(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype="float64"))
        return np.exp(-self.zero(t) * t)


SVENSSON_START = np.array([0.04, -0.01, -0.01, 0.01, 1.5, 10.0])


def fit_svensson(bonds, dirty: np.ndarray, weight: np.ndarray, start: np.ndarray | None = None):
    """Nonlinear weighted LS. Returns (curve, price residuals market - model)."""
    sw = np.sqrt(weight)
    # every bond's cash flows flattened once: one vector op per evaluation, not a loop
    # over bonds (the loop was ~90% of a day's build)
    T = np.concatenate([t for t, _ in bonds]); A = np.concatenate([a for _, a in bonds])
    idx = np.concatenate([np.full(len(t), i) for i, (t, _) in enumerate(bonds)])

    def resid(p):
        model = np.bincount(idx, A * SvenssonCurve(p).discount(T), minlength=len(bonds))
        return (dirty - model) * sw

    x0 = SVENSSON_START if start is None else np.asarray(start, dtype="float64")
    lo = [-0.05, -0.3, -0.5, -0.5, 0.05, 2.0]
    hi = [0.25, 0.3, 0.5, 0.5, 10.0, 40.0]
    x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=400)
    c = SvenssonCurve(r.x)
    return c, dirty - np.bincount(idx, A * c.discount(T), minlength=len(bonds))


FIT_METHODS = {"spline": fit_spline, "svensson": fit_svensson}


# ------------------------------------------------------------------ curve outputs
def zero_rate(curve, t) -> np.ndarray:
    """Continuously compounded zero rate (%)."""
    t = np.maximum(np.atleast_1d(np.asarray(t, dtype="float64")), 1e-6)
    return -np.log(np.maximum(curve.discount(t), 1e-12)) / t * 100.0


def par_yield(curve, maturity_years: float, per_year: int = 2) -> float:
    """Par yield (%, compounded ``per_year``) of a bond issued today."""
    t = np.arange(1, int(round(maturity_years * per_year)) + 1) / per_year
    d = curve.discount(t)
    return float(per_year * (1 - d[-1]) / d.sum() * 100.0)


def zspread(curve, dirty: float, t: np.ndarray, a: np.ndarray) -> float:
    """Constant (bp, continuous) added to the curve's zero rates that reprices the bond
    (Newton, Brent fallback)."""
    d = a * curve.discount(t)
    s = 0.0
    for _ in range(20):
        e = d * np.exp(-s * t)
        f, df = float(e.sum()) - dirty, float((-t * e).sum())
        if df == 0:
            break
        step = f / df
        s -= step
        if abs(step) < 1e-12:
            return 1e4 * s
    g = lambda ss: float((d * np.exp(-ss * t)).sum()) - dirty
    return 1e4 * brentq(g, -0.05, 0.05)


def bond_metrics(curve, dirty: float, accrued: float, t: np.ndarray, a: np.ndarray, *, horizon_days: int = 91,
                 per_year: int = 2) -> dict:
    """z-spread, curve carry and rolldown (bp of the bond's own yield, + = gain) over
    ``horizon_days``. Coupons paid before the horizon are reinvested at the curve (they
    leave the forward); the price at the horizon is clean of the accrued then due."""
    y0 = ytm(dirty, t, a, per_year)
    g0 = y0 / 100
    z = zspread(curve, dirty, t, a) / 1e4
    h = horizon_days / YEAR
    before, after = t <= h, t > h
    # curve carry: forward dirty price at h = (dirty - PV(coupons before h)) / D(h)
    fwd_dirty = (dirty - float((a[before] * curve.discount(t[before])).sum())) / float(curve.discount([h])[0])
    th, ah = t[after] - h, a[after]
    if th.size == 0:
        return {"ytm": y0, "zspread_bp": z * 1e4, "carry_curve_bp": np.nan, "rolldown_bp": np.nan}
    y_fwd = ytm(fwd_dirty, th, ah, per_year, guess=g0)
    # rolldown: the bond at h priced off TODAY's curve (+ its z-spread), i.e. shorter by h
    rolled = float((ah * curve.discount(th) * np.exp(-z * th)).sum())
    y_roll = ytm(rolled, th, ah, per_year, guess=g0)
    return {"ytm": y0, "zspread_bp": z * 1e4, "carry_curve_bp": (y0 - y_fwd) * 100, "rolldown_bp": (y0 - y_roll) * 100}
