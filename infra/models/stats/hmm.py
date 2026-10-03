"""Hidden Markov model machinery, pure numpy: the forward-backward recursions and a
Gaussian-emission EM that tolerates missing features.

Generic on purpose: ``forward_backward`` takes ANY per-row, per-regime log-likelihood, so a
Markov-switching regression (emission = the regression's residual density per regime)
reuses it unchanged - only the M-step differs.

Probabilities, all ``n x R``:

* ``filtered``  p(s_t | x_1..x_t)      - known at t (point in time)
* ``predicted`` p(s_t | x_1..x_{t-1})  - known BEFORE row t's own data (= filtered_{t-1} @ P)
* ``smoothed``  p(s_t | x_1..x_T)      - uses the whole sample: in-sample estimation only,
  never a trading signal (look-ahead)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FBResult:
    filtered: np.ndarray
    predicted: np.ndarray
    smoothed: np.ndarray
    xi_sum: np.ndarray        # sum over t of p(s_{t-1}=i, s_t=j | all) - the M-step's transition counts
    loglik: float


def forward_backward(log_b: np.ndarray, P: np.ndarray, pi0: np.ndarray, *, smooth: bool = True) -> FBResult:
    """Scaled forward-backward. ``log_b``: n x R log emission densities (a row of zeros =
    no information that day); ``P``: R x R transitions (rows sum to 1); ``pi0``: the
    regime distribution BEFORE the first row (so predicted[0] = pi0)."""
    n, R = log_b.shape
    shift = np.max(log_b, axis=1, keepdims=True)
    shift[~np.isfinite(shift)] = 0.0
    b = np.exp(log_b - shift)
    filt, pred, c = np.empty((n, R)), np.empty((n, R)), np.empty(n)
    P = floor_probs(P)
    prev = floor_probs(pi0)
    for t in range(n):
        p = prev if t == 0 else filt[t - 1] @ P
        pred[t] = p
        x = p * b[t]
        s = x.sum()
        if not s > 0:  # every regime impossible (numerical): fall back to the prediction
            x, s = p.copy(), 1.0
        c[t] = s
        filt[t] = x / s
    loglik = float(np.sum(np.log(c)) + np.sum(shift))
    if not smooth:
        return FBResult(filt, pred, np.full((n, R), np.nan), np.zeros((R, R)), loglik)
    beta = np.ones((n, R))
    xi = np.zeros((R, R))
    for t in range(n - 2, -1, -1):
        bb = b[t + 1] * beta[t + 1]
        beta[t] = (P @ bb) / c[t + 1]
        xi += P * np.outer(filt[t], bb) / c[t + 1]
    sm = filt * beta
    sm /= sm.sum(axis=1, keepdims=True)
    return FBResult(filt, pred, sm, xi, loglik)


def gaussian_loglik(X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """n x R log densities of each row under each regime's Gaussian, marginalised over the
    row's MISSING entries (a row with nothing observed scores 0 = uninformative)."""
    n, d = X.shape
    R = len(means)
    obs = np.isfinite(X)
    out = np.zeros((n, R))
    keys: dict[bytes, list[int]] = {}
    for i in range(n):
        keys.setdefault(obs[i].tobytes(), []).append(i)
    for key, rows in keys.items():
        o = np.frombuffer(key, dtype=bool)
        if not o.any():
            continue
        rows = np.asarray(rows)
        Xo = X[np.ix_(rows, np.flatnonzero(o))]
        for r in range(R):
            C = covs[r][np.ix_(o, o)]
            sign, logdet = np.linalg.slogdet(C)
            D = Xo - means[r][o]
            q = np.einsum("ij,ij->i", D @ np.linalg.inv(C), D)
            out[rows, r] = -0.5 * (q + logdet + o.sum() * np.log(2 * np.pi))
    return out


def _kmeans(X: np.ndarray, R: int, rng: np.random.Generator, iters: int = 50) -> np.ndarray:
    """Labels from k-means++ / Lloyd on complete rows (initialisation only)."""
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, R):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
    C = np.array(centers)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        new = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(axis=2), axis=1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for r in range(R):
            if np.any(lab == r):
                C[r] = X[lab == r].mean(axis=0)
    return lab


PROB_FLOOR = 1e-8  # no transition / initial state is ever exactly impossible


def floor_probs(P: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    """Clip a stochastic matrix (or vector) at ``floor`` and renormalise its rows. Found
    2026-10-03 (US/DE/UK curves): a refit inherited a transition with an exact 0; on a day
    extremely unlikely under the regime the filter was sure of, predicted x likelihood was 0
    for every regime and the recursion returned NaN."""
    Q = np.clip(np.asarray(P, dtype="float64"), floor, None)
    return Q / Q.sum(axis=-1, keepdims=True)


@dataclass
class HMMParams:
    means: np.ndarray     # R x d
    covs: np.ndarray      # R x d x d
    P: np.ndarray         # R x R
    pi0: np.ndarray       # R (distribution before the first row)


def m_step_gaussian(X, gamma, covariance: str, reg: float, floor_var: np.ndarray):
    """Weighted means / covariances from the COMPLETE rows (rows with gaps inform the
    filter, not the parameters - see the module docstring of ``regimes``)."""
    complete = np.isfinite(X).all(axis=1)
    Xc, G = X[complete], gamma[complete]
    R, d = G.shape[1], X.shape[1]
    means, covs = np.zeros((R, d)), np.zeros((R, d, d))
    for r in range(R):
        w = G[:, r]
        W = w.sum()
        if W < 1e-8:
            means[r], covs[r] = np.nanmean(X, axis=0), np.diag(floor_var / reg if reg else floor_var)
            continue
        mu = w @ Xc / W
        D = Xc - mu
        C = (D * w[:, None]).T @ D / W
        if covariance == "diag":
            C = np.diag(np.diag(C))
        means[r] = mu
        covs[r] = C + reg * np.diag(floor_var)
    return means, covs


def transition_m_step(xi_sum: np.ndarray, sticky: float = 0.0) -> np.ndarray:
    """Transition probabilities from the expected transition counts, with ``sticky``
    pseudo-counts on each diagonal (a Dirichlet prior favouring staying; MAP estimate,
    the "sticky HMM" of Fox et al. 2011). 0 = maximum likelihood."""
    counts = xi_sum + sticky * np.eye(len(xi_sum))
    return floor_probs(counts / np.clip(counts.sum(axis=1, keepdims=True), 1e-300, None))


def fit_gaussian_hmm(X: np.ndarray, R: int, *, covariance: str = "full", n_init: int = 3, max_iter: int = 200,
                     tol: float = 1e-6, reg: float = 1e-3, seed: int = 0, stay: float = 0.95,
                     sticky: float = 0.0, init: HMMParams | None = None) -> tuple[HMMParams, FBResult, dict]:
    """EM (Baum-Welch) for a Gaussian HMM. ``init``: warm start (one run from those
    parameters, e.g. the previous refit's) instead of ``n_init`` k-means starts. ``reg``:
    a ridge of ``reg`` x each feature's variance on every covariance (no regime collapses
    onto a singular matrix). Returns the best run's parameters, its smoothed pass and
    info (loglik, iterations, converged, regime occupancy)."""
    complete = np.isfinite(X).all(axis=1)
    if complete.sum() < R * (X.shape[1] + 2):
        raise ValueError(f"HMM: {int(complete.sum())} complete feature rows for {R} regimes x {X.shape[1]} features")
    floor_var = np.nanvar(X[complete], axis=0)
    floor_var[floor_var <= 0] = 1.0
    starts = []
    if init is not None:
        starts.append(init)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_init):
            lab = _kmeans(X[complete], R, rng)
            G = np.zeros((len(X), R))
            G[np.flatnonzero(complete), lab] = 1.0
            means, covs = m_step_gaussian(X, G, covariance, reg, floor_var)
            P = np.full((R, R), (1 - stay) / max(R - 1, 1)) + np.eye(R) * (stay - (1 - stay) / max(R - 1, 1))
            starts.append(HMMParams(means, covs, P if R > 1 else np.ones((1, 1)), np.full(R, 1.0 / R)))
    best = None
    for p0 in starts:
        p = HMMParams(p0.means.copy(), p0.covs.copy(), p0.P.copy(), p0.pi0.copy())
        prev_ll, it, fb = -np.inf, 0, None
        for it in range(1, max_iter + 1):
            fb = forward_backward(gaussian_loglik(X, p.means, p.covs), p.P, p.pi0)
            p.means, p.covs = m_step_gaussian(X, fb.smoothed, covariance, reg, floor_var)
            p.P = transition_m_step(fb.xi_sum, sticky)
            p.pi0 = floor_probs(fb.smoothed[0])
            if not (np.isfinite(p.means).all() and np.isfinite(p.covs).all() and np.isfinite(fb.loglik)):
                break
            if abs(fb.loglik - prev_ll) < tol * (1 + abs(fb.loglik)):
                break
            prev_ll = fb.loglik
        if not (np.isfinite(p.means).all() and np.isfinite(p.covs).all()):
            continue
        fb = forward_backward(gaussian_loglik(X, p.means, p.covs), p.P, p.pi0)
        if not (np.isfinite(fb.loglik) and np.isfinite(fb.smoothed).all()):
            continue
        info = {"loglik": fb.loglik, "iterations": it, "converged": float(it < max_iter)}
        if best is None or fb.loglik > best[1].loglik:
            best = (p, fb, info)
    if best is None:
        if init is not None:  # a warm start that failed numerically: start cold instead
            return fit_gaussian_hmm(X, R, covariance=covariance, n_init=n_init, max_iter=max_iter, tol=tol,
                                    reg=reg, seed=seed, stay=stay, sticky=sticky, init=None)
        raise RuntimeError("HMM: every EM start failed numerically")
    p, fb, info = best
    info["occupancy"] = fb.smoothed.mean(axis=0)
    return p, fb, info
