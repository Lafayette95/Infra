"""Linear Gaussian state space, Kalman filter + Rauch-Tung-Striebel smoother, with
missing data. Pure numpy, model-agnostic (knows nothing about factors or releases).

    x_t = Z s_t + eps_t,        eps_t ~ N(0, diag(R))
    s_t = T s_{t-1} + eta_t,    eta_t ~ N(0, Q),    s_0 ~ N(a0, P0)

Missing data (the ragged edge, mixed frequencies) is handled by dropping the missing rows
of ``x_t`` at each step - the exact likelihood, equivalent to giving them infinite noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

_LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class StateSpace:
    Z: np.ndarray  # (n, m)
    R: np.ndarray  # (n,) measurement noise variances
    T: np.ndarray  # (m, m)
    Q: np.ndarray  # (m, m)
    a0: np.ndarray  # (m,)
    P0: np.ndarray  # (m, m)


@dataclass
class Smoothed:
    a: np.ndarray  # (T, m) E[s_t | all data]
    P: np.ndarray  # (T, m, m) Var[s_t | all data]
    P_lag: np.ndarray  # (T, m, m) Cov[s_t, s_{t-1} | all data]; [0] is zero
    loglik: float


def unconditional(T: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stationary mean (0) and covariance of the state, a large diffuse-ish prior if the
    transition is not stable."""
    m = T.shape[0]
    try:
        if np.max(np.abs(np.linalg.eigvals(T))) < 1.0:
            P = linalg.solve_discrete_lyapunov(T, Q)
            return np.zeros(m), (P + P.T) / 2.0
    except (np.linalg.LinAlgError, ValueError):
        pass
    return np.zeros(m), np.eye(m) * 10.0


def smooth(ss: StateSpace, X: np.ndarray, *, lag: bool = True) -> Smoothed:
    """Filter then smooth ``X`` (T x n, NaN = missing)."""
    n_t, m = X.shape[0], ss.T.shape[0]
    a_pred = np.empty((n_t, m))
    P_pred = np.empty((n_t, m, m))
    a_filt = np.empty((n_t, m))
    P_filt = np.empty((n_t, m, m))
    a, P = ss.a0.copy(), ss.P0.copy()
    loglik = 0.0
    observed = np.isfinite(X)
    for t in range(n_t):
        a_pred[t], P_pred[t] = a, P
        obs = observed[t]
        if obs.any():
            Zt = ss.Z[obs]
            v = X[t, obs] - Zt @ a
            PZ = P @ Zt.T
            F = Zt @ PZ + np.diag(ss.R[obs])
            try:
                c = linalg.cho_factor(F, lower=True, check_finite=False)
                Finv_v = linalg.cho_solve(c, v, check_finite=False)
                K_T = linalg.cho_solve(c, PZ.T, check_finite=False)  # F^-1 Z P = K'
                logdet = 2.0 * np.log(np.diag(c[0])).sum()
            except linalg.LinAlgError:
                Finv = np.linalg.pinv(F)
                Finv_v, K_T = Finv @ v, Finv @ PZ.T
                logdet = np.linalg.slogdet(F)[1]
            a = a + PZ @ Finv_v
            P = P - PZ @ K_T
            P = (P + P.T) / 2.0
            loglik -= 0.5 * (obs.sum() * _LOG_2PI + logdet + v @ Finv_v)
        a_filt[t], P_filt[t] = a, P
        a = ss.T @ a
        P = ss.T @ P @ ss.T.T + ss.Q
    a_s = np.empty_like(a_filt)
    P_s = np.empty_like(P_filt)
    P_lag = np.zeros_like(P_filt)
    a_s[-1], P_s[-1] = a_filt[-1], P_filt[-1]
    for t in range(n_t - 2, -1, -1):
        TP = ss.T @ P_filt[t]
        try:
            J = linalg.solve(P_pred[t + 1], TP, assume_a="pos", check_finite=False).T
        except (linalg.LinAlgError, ValueError):
            J = (np.linalg.pinv(P_pred[t + 1]) @ TP).T
        a_s[t] = a_filt[t] + J @ (a_s[t + 1] - a_pred[t + 1])
        P_s[t] = P_filt[t] + J @ (P_s[t + 1] - P_pred[t + 1]) @ J.T
        P_s[t] = (P_s[t] + P_s[t].T) / 2.0
        if lag:
            P_lag[t + 1] = P_s[t + 1] @ J.T
    return Smoothed(a_s, P_s, P_lag, loglik)
