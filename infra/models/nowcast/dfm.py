"""Mixed-frequency dynamic factor model, estimated by EM (Banbura & Modugno 2014, the NY
Fed Staff Nowcast's model - Bok et al. 2018, FRBNY SR 830). Pure numpy on an
already-prepared panel (panel.py); the factor pattern comes from spec.py.

Model, in standardized units ``x_it = (y_it - mu_i) / sd_i``:

    monthly   x_it = lam_i . f_t + e_it          e_it = rho_i e_i,t-1 + u_it  (idio="ar1")
                                                 (or iid measurement noise, idio="iid")
    quarterly x_it = lam_i . (f_t + 2f_t-1 + 3f_t-2 + 2f_t-3 + f_t-4) + eps_it
              (observed in the quarter's 3rd month; Mariano & Murasawa 2003 - the
              quarterly growth of a quarterly average, from its monthly growth rates)
    factors   f_t = A_1 f_t-1 + ... + A_p f_t-p + v_t,   v_t ~ N(0, Q)

Loadings may be fixed at 0 (``FactorStructure.free``, version c) or carry a N(0, tau^2)
prior (``penalized``, version d): the M-step for that release's loadings becomes a
ridge (MAP) regression. The prior is scaled against the release's idiosyncratic variance
(its regression noise once the common component is taken out); for an AR(1) idiosyncratic
state that treats the idio as the noise of the loading regression, ignoring its serial
correlation - an approximation in the penalty only, the likelihood stays exact.

Factors are rescaled to unit unconditional variance after every M-step. That is a free
normalization for (a)-(c) (the likelihood is invariant), and what makes (d)'s ``tau``
mean something: without it the model could inflate a factor's scale to shrink the
loadings under the prior for free.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import linalg

from infra.models.nowcast import kalman
from infra.models.nowcast.spec import FactorStructure, ModelSpec

log = logging.getLogger(__name__)

QUARTERLY_WEIGHTS = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
KAPPA = 1e-4  # measurement noise of a monthly series whose idio is in the state
_MIN_VAR = 1e-4
_RHO_MAX = 0.99


@dataclass
class Params:
    lam: np.ndarray  # (n, r)
    A: np.ndarray  # (r, r * p)
    Q: np.ndarray  # (r, r)
    rho: np.ndarray  # (n,) AR(1) of the idio (monthly, idio="ar1"); 0 otherwise
    sig2: np.ndarray  # (n,) idio innovation variance (ar1) / measurement variance (iid, quarterly)


@dataclass
class DFM:
    spec: ModelSpec
    structure: FactorStructure
    quarterly: np.ndarray  # (n,) bool
    mu: np.ndarray  # (n,) standardization, frozen at estimation
    sd: np.ndarray
    params: Params
    months: pd.DatetimeIndex  # the estimation sample
    as_of: pd.Timestamp | None
    loglik: list[float] = field(default_factory=list)

    @property
    def series(self) -> tuple[str, ...]:
        return self.structure.series

    @property
    def r(self) -> int:
        return len(self.structure.factors)

    def standardize(self, data: pd.DataFrame) -> np.ndarray:
        return ((data[list(self.series)].to_numpy(dtype=float)) - self.mu) / self.sd

    def state_space(self) -> kalman.StateSpace:
        return build_state_space(self.params, self.quarterly, self.spec)

    def layout(self) -> "Layout":
        return Layout.of(self.r, self.spec.factor_lags, self.quarterly, self.spec.idio)


@dataclass(frozen=True)
class Layout:
    """Where everything sits in the state vector."""
    r: int
    p: int
    L: int  # factor lags carried in the state
    idio_index: dict[int, int]  # series i -> its idio state (idio="ar1", monthly only)

    @property
    def m(self) -> int:
        return self.r * self.L + len(self.idio_index)

    def f(self, lag: int = 0) -> slice:
        return slice(lag * self.r, (lag + 1) * self.r)

    @classmethod
    def of(cls, r: int, p: int, quarterly: np.ndarray, idio: str) -> "Layout":
        L = max(p, len(QUARTERLY_WEIGHTS) if quarterly.any() else 1)
        monthly = [i for i in range(len(quarterly)) if not quarterly[i]] if idio == "ar1" else []
        return cls(r, p, L, {i: r * L + k for k, i in enumerate(monthly)})


def measurement_row(lam_i: np.ndarray, i: int, quarterly: bool, lay: Layout) -> np.ndarray:
    """Row of Z for series i: its loadings on the state (common part + its own idio)."""
    z = np.zeros(lay.m)
    if quarterly:
        for lag, w in enumerate(QUARTERLY_WEIGHTS):
            z[lay.f(lag)] = w * lam_i
    else:
        z[lay.f(0)] = lam_i
    if i in lay.idio_index:
        z[lay.idio_index[i]] = 1.0
    return z


def build_state_space(p: Params, quarterly: np.ndarray, spec: ModelSpec) -> kalman.StateSpace:
    n, r = p.lam.shape
    lay = Layout.of(r, spec.factor_lags, quarterly, spec.idio)
    m = lay.m
    Z = np.vstack([measurement_row(p.lam[i], i, quarterly[i], lay) for i in range(n)])
    R = np.array([KAPPA if i in lay.idio_index else p.sig2[i] for i in range(n)])
    T = np.zeros((m, m))
    T[: r, : r * lay.p] = p.A
    for lag in range(1, lay.L):
        T[lay.f(lag), lay.f(lag - 1)] = np.eye(r)
    Q = np.zeros((m, m))
    Q[: r, : r] = p.Q
    for i, k in lay.idio_index.items():
        T[k, k] = p.rho[i]
        Q[k, k] = p.sig2[i]
    a0, P0 = kalman.unconditional(T, Q)
    return kalman.StateSpace(Z, R, T, Q, a0, P0)


# --------------------------------------------------------------------- initialization
def _pc1(block: np.ndarray) -> np.ndarray:
    block = block - block.mean(axis=0)
    u, s, _ = np.linalg.svd(block, full_matrices=False)
    f = u[:, 0] * s[0]
    return f / (f.std() or 1.0)


def _lagged(f: np.ndarray, lags: int) -> np.ndarray:
    """(T, r*lags): [f_{t-1}, ..., f_{t-lags}], zero-padded at the start."""
    n_t, r = f.shape
    out = np.zeros((n_t, r * lags))
    for lag in range(1, lags + 1):
        out[lag:, (lag - 1) * r: lag * r] = f[:-lag]
    return out


def _aggregate(f: np.ndarray) -> np.ndarray:
    """(T, r): sum_l w_l f_{t-l} - the quarterly regressor."""
    z = QUARTERLY_WEIGHTS[0] * f.copy()
    for lag, w in enumerate(QUARTERLY_WEIGHTS[1:], start=1):
        z[lag:] += w * f[:-lag]
    return z


def initialize(X: np.ndarray, structure: FactorStructure, quarterly: np.ndarray, spec: ModelSpec) -> Params:
    """Block principal components on the zero-filled monthly panel, then OLS."""
    n = X.shape[1]
    r = len(structure.factors)
    filled = np.where(np.isfinite(X), X, 0.0)
    monthly = ~quarterly
    f0 = np.empty((X.shape[0], r))
    for j in range(r):
        cols = structure.member[:, j] & monthly
        if not cols.any():
            cols = monthly
        f0[:, j] = _pc1(filled[:, cols])
    # starting loadings: own blocks only unless the version is unrestricted (b)
    start_free = structure.free if spec.restriction == "none" else structure.member
    z0 = _aggregate(f0)
    lam, resid = np.zeros((n, r)), np.full(X.shape, np.nan)
    for i in range(n):
        obs, J = np.isfinite(X[:, i]), start_free[i]
        reg = (z0 if quarterly[i] else f0)[obs][:, J]
        coef = np.linalg.lstsq(reg, X[obs, i], rcond=None)[0]
        lam[i, J] = coef
        resid[obs, i] = X[obs, i] - reg @ coef
    p = spec.factor_lags
    lagged = _lagged(f0, p)[p:]
    A, Q = _var_step(f0[p:].T @ f0[p:], f0[p:].T @ lagged, lagged.T @ lagged, r, p, len(lagged), spec.factor_dynamics)
    rho, sig2 = np.zeros(n), np.ones(n)
    for i in range(n):
        e = resid[:, i]
        pair = np.isfinite(e[1:]) & np.isfinite(e[:-1])
        var = np.nanvar(e) if np.isfinite(e).sum() > 1 else 1.0
        if spec.idio == "ar1" and not quarterly[i] and pair.sum() > 2:
            rho[i] = np.clip(np.dot(e[1:][pair], e[:-1][pair]) / np.dot(e[:-1][pair], e[:-1][pair]), -_RHO_MAX, _RHO_MAX)
            sig2[i] = max(var * (1.0 - rho[i] ** 2), _MIN_VAR)
        else:
            sig2[i] = max(var, _MIN_VAR)
    return normalize(Params(lam, A, Q, rho, sig2))


def normalize(p: Params) -> Params:
    """Rescale every factor to unit unconditional variance (loadings absorb the scale)."""
    r = p.lam.shape[1]
    lags = p.A.shape[1] // r
    comp = np.zeros((r * lags, r * lags))
    comp[:r] = p.A
    comp[r:, : r * (lags - 1)] = np.eye(r * (lags - 1))
    Qc = np.zeros_like(comp)
    Qc[:r, :r] = p.Q
    if np.max(np.abs(np.linalg.eigvals(comp))) >= 1.0:
        return p
    V = linalg.solve_discrete_lyapunov(comp, Qc)
    d = np.sqrt(np.clip(np.diag(V)[:r], 1e-12, None))
    Dinv = np.diag(1.0 / d)
    A = np.hstack([Dinv @ p.A[:, k * r:(k + 1) * r] @ np.diag(d) for k in range(lags)])
    return Params(p.lam * d, A, Dinv @ p.Q @ Dinv, p.rho, p.sig2)


# --------------------------------------------------------------------------- EM
def _moments(sm: kalman.Smoothed) -> tuple[np.ndarray, np.ndarray]:
    """E[s_t s_t'] and E[s_t s_t-1'] per t."""
    a = sm.a
    Ess = sm.P + a[:, :, None] * a[:, None, :]
    Ess1 = sm.P_lag.copy()
    Ess1[1:] += a[1:, :, None] * a[:-1, None, :]
    return Ess, Ess1


def _own_lags(j: int, r: int, p: int) -> list[int]:
    return [lag * r + j for lag in range(p)]


def _var_step(S11, S10, S00, r: int, p: int, n: int, dynamics: str) -> tuple[np.ndarray, np.ndarray]:
    """Factor VAR from its sufficient statistics: ``full`` = one VAR, ``independent`` =
    each factor regressed on its own lags only, diagonal shock covariance."""
    if dynamics == "full":
        A = np.linalg.solve(S00.T, S10.T).T
        Q = (S11 - A @ S10.T) / n
        return A, (Q + Q.T) / 2.0 + 1e-8 * np.eye(r)
    if dynamics != "independent":
        raise ValueError(f"unknown factor_dynamics {dynamics!r}")
    A, q = np.zeros((r, r * p)), np.zeros(r)
    for j in range(r):
        own = _own_lags(j, r, p)
        A[j, own] = np.linalg.solve(S00[np.ix_(own, own)], S10[j, own])
        q[j] = (S11[j, j] - A[j, own] @ S10[j, own]) / n
    return A, np.diag(np.maximum(q, 1e-8))


def m_step(X: np.ndarray, sm: kalman.Smoothed, p: Params, structure: FactorStructure, quarterly: np.ndarray,
           spec: ModelSpec) -> Params:
    n, r = p.lam.shape
    lay = Layout.of(r, spec.factor_lags, quarterly, spec.idio)
    Ess, Ess1 = _moments(sm)
    n_t = X.shape[0]
    rp = r * lay.p
    # factor VAR
    S11 = Ess[1:, :r, :r].sum(0)
    S10 = Ess1[1:, :r, :rp].sum(0)
    S00 = Ess[:-1, :rp, :rp].sum(0)
    A, Q = _var_step(S11, S10, S00, r, lay.p, n_t - 1, spec.factor_dynamics)
    # idiosyncratic AR(1)
    rho, sig2 = p.rho.copy(), p.sig2.copy()
    for i, k in lay.idio_index.items():
        s_e1 = Ess1[1:, k, k].sum()
        s_00 = Ess[:-1, k, k].sum()
        rho[i] = np.clip(s_e1 / s_00, -_RHO_MAX, _RHO_MAX)
        sig2[i] = max((Ess[1:, k, k].sum() - rho[i] * s_e1) / (n_t - 1), _MIN_VAR)
    # loadings (+ measurement variances where the idio is not in the state)
    lam = np.zeros_like(p.lam)
    weights = np.concatenate([w * np.eye(r) for w in QUARTERLY_WEIGHTS], axis=1)  # z = W s[:r*5]
    for i in range(n):
        obs = np.isfinite(X[:, i])
        J = structure.free[i]
        if not obs.any() or not J.any():
            continue
        x = X[obs, i]
        if quarterly[i]:
            span = slice(0, r * len(QUARTERLY_WEIGHTS))
            Ez = sm.a[obs][:, span] @ weights.T
            Ezz = np.einsum("ij,tjk,lk->til", weights, Ess[obs][:, span, span], weights)
        else:
            Ez = sm.a[obs][:, :r]
            Ezz = Ess[obs][:, :r, :r]
        num = x @ Ez[:, J]
        if i in lay.idio_index:
            num = num - Ess[obs][:, :r, lay.idio_index[i]][:, J].sum(0)
            noise = sig2[i] / (1.0 - rho[i] ** 2)
        else:
            noise = p.sig2[i]
        den = Ezz[:, J][:, :, J].sum(0)
        pen = structure.penalized[i, J]
        if pen.any():
            den = den + np.diag(pen * noise / spec.tau ** 2)
        lam[i, J] = np.linalg.solve(den, num)
        if i not in lay.idio_index:
            li = lam[i]
            resid2 = x ** 2 - 2.0 * x * (Ez @ li) + np.einsum("j,tjk,k->t", li, Ezz, li)
            sig2[i] = max(resid2.mean(), _MIN_VAR)
    return normalize(Params(lam, A, Q, rho, sig2))


def fit(data: pd.DataFrame, frequency: dict[str, str], structure: FactorStructure, spec: ModelSpec,
        *, as_of=None) -> DFM:
    """Estimate on ``data`` (months x release tickers, the model units of panel.py)."""
    series = list(structure.series)
    values = data[series].to_numpy(dtype=float)
    mu, sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    X = (values - mu) / sd
    quarterly = np.array([frequency[s] == "Q" for s in series])
    params = initialize(X, structure, quarterly, spec)
    model = DFM(spec, structure, quarterly, mu, sd, params, data.index, None if as_of is None else pd.Timestamp(as_of))
    for it in range(spec.max_iter):
        sm = kalman.smooth(model.state_space(), X)
        model.loglik.append(sm.loglik)
        if it > 0:
            prev = model.loglik[-2]
            if abs(sm.loglik - prev) / max(abs(prev), 1.0) < spec.tol:
                break
        model.params = m_step(X, sm, model.params, structure, quarterly, spec)
    log.info("DFM %s: %d EM iterations, loglik %.1f", spec.name, len(model.loglik), model.loglik[-1])
    return orient(model)


def orient(model: DFM) -> DFM:
    """Sign convention: each factor points the way its own members load on average
    (after ``sign``, members point "stronger economy / higher prices" up)."""
    p = model.params
    r = p.lam.shape[1]
    flip = np.ones(r)
    for j in range(r):
        members = model.structure.member[:, j]
        if p.lam[members, j].sum() < 0:
            flip[j] = -1.0
    F = np.diag(flip)
    lags = p.A.shape[1] // r
    A = np.hstack([F @ p.A[:, k * r:(k + 1) * r] @ F for k in range(lags)])
    model.params = Params(p.lam * flip, A, F @ p.Q @ F, p.rho, p.sig2)
    return model
