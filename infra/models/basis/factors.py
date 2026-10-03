"""M2's yield-change factor model for one delivery basket (infra/models/basis/CLAUDE.md).
Pure, no I/O.

    y_i(T) - y_i(now) = L + s_i,   L = the basket's mean move,  s_i = bond i's spread move

* L, the LEVEL: the basket's equal-weight mean change; a random walk whose daily vol is a
  FAST EWMA (``lam_fast``), scaled by sqrt(horizon).
* s, the SPREADS (each bond's yield minus the basket mean): their covariance is estimated
  AT THE HORIZON, from overlapping h-day changes under SLOW EWMA weights - NOT scaled up
  from daily changes. Spreads mean-revert hard (measured 2026-10-02, 2024-2026: the
  variance of 60-day spread changes is 4% (ZT), 19% (ZN), ~30% (ZB, UB) of 60x the daily
  variance - daily pricing noise and relative-value reversion), so sqrt(h) scaling
  overstated the relative variance 3-25x. Top ``k`` principal components + a diagonal
  remainder (idiosyncratic) at that horizon. These are the switch drivers, which a PCA on
  yield LEVELS hides in components holding under 2% of the variance.
Bonds too young for the window get their PREDECESSOR's changes (the previous original issue
of the same security type and term - a constant spread drops out of changes), then the
nearest-maturity basket bond's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorModel:
    cusips: tuple[str, ...]
    horizon_bd: int  # the horizon the spread covariance was estimated at
    phi: np.ndarray  # (n, k) spread-factor loadings at the horizon (unit-variance factors), bp
    psi: np.ndarray  # (n,) idiosyncratic spread variance at the horizon (bp^2)
    sigma_level: float  # level vol (bp/day), fast EWMA
    explained: dict  # variance shares at the horizon, for diagnostics
    variance_ratio: float  # median over bonds: var(h-day spread change) / (h x var(1-day))
    backfilled: dict  # cusip -> where its missing history came from
    noise_var: np.ndarray | None = None  # (n,) mark-noise variance per bond (bp^2), see fit_factor_model


def ewma_weights(n: int, lam: float) -> np.ndarray:
    w = lam ** np.arange(n - 1, -1, -1, dtype="float64")
    return w / w.sum()


def change_panel(yields: pd.DataFrame, cusips: list[str], predecessor: dict[str, str | None],
                 maturity: dict[str, pd.Timestamp], window: int) -> tuple[pd.DataFrame, dict]:
    """Daily yield changes (bp) of ``cusips`` over the last ``window`` days of ``yields``
    (wide, %, by day), each bond's gaps filled from its predecessor chain, then from the
    nearest-maturity basket bond with a full window."""
    dy = yields.diff().iloc[1:].tail(window) * 100.0
    out, source = pd.DataFrame(index=dy.index), {}
    for c in cusips:
        s = dy[c].copy() if c in dy.columns else pd.Series(np.nan, index=dy.index)
        hops, p = [], predecessor.get(c)
        while s.isna().any() and p is not None and len(hops) < 4:
            if p in dy.columns:
                s = s.fillna(dy[p])
                hops.append(p)
            p = predecessor.get(p)
        out[c] = s
        if hops:
            source[c] = "predecessor " + " > ".join(hops)
    full = [c for c in cusips if out[c].notna().all()]
    for c in cusips:
        if out[c].isna().any() and full:
            near = min(full, key=lambda f: abs((maturity[f] - maturity[c]).days))
            out[c] = out[c].fillna(out[near])
            source[c] = (source.get(c, "") + f" + nearest {near}").strip(" +")
    return out.dropna(how="any"), source


def fit_factor_model(dy: pd.DataFrame, horizon_bd: int, *, k: int = 3, lam_slow: float = 0.99,
                     lam_fast: float = 0.94, backfilled: dict | None = None, min_idio_share: float = 0.01,
                     max_horizon_share: float = 0.34, noise_removal: float = 0.0) -> FactorModel:
    """Fit on a complete panel of DAILY changes (rows = days, columns = bonds), with the
    spread covariance taken at ``horizon_bd`` (capped at ``max_horizon_share`` of the
    window so enough overlapping changes remain).

    ``noise_removal``: how many MARK-NOISE variances to take off each bond's idiosyncratic
    variance (0 = none, as first built; 1 = the start-of-horizon mark; 2 = both ends of an
    h-day change). A price mark with independent noise of variance s^2 gives its daily
    changes a lag-1 autocovariance of -s^2, so s^2 = max(-autocov, 0) per bond, on the
    spread changes. Motivation (2026-10-02/03): on ZB's ~50-bond basket M2 put only a
    median 61% on the realised CTD (M1 93%), every bond's own noise leaking probability to
    bonds that never win; FedInvest's marks for old off-the-runs are the noisiest part."""
    x = dy.to_numpy(dtype="float64")
    t, n = x.shape
    if t < 40 or n == 0:
        raise ValueError(f"too little history to fit: {t} days x {n} bonds")
    h = int(max(1, min(horizon_bd, int(t * max_horizon_share))))
    level = x.mean(axis=1)
    spread_cum = np.cumsum(x - level[:, None], axis=0)
    dh = spread_cum[h:] - spread_cum[:-h]  # overlapping h-day spread changes
    w = ewma_weights(dh.shape[0], lam_slow)
    cov = (w[:, None] * dh).T @ dh
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = vals[::-1].clip(min=0.0), vecs[:, ::-1]
    k = max(0, min(k, n - 1))
    phi = vecs[:, :k] * np.sqrt(vals[:k])[None, :]
    sx = x - level[:, None]
    sx = sx - sx.mean(axis=0)
    noise = np.maximum(-(sx[1:] * sx[:-1]).mean(axis=0), 0.0)
    psi = np.maximum(np.diag(cov) - (phi ** 2).sum(axis=1) - noise_removal * noise, min_idio_share * np.diag(cov))
    wf = ewma_weights(t, lam_fast)
    sigma_level = float(np.sqrt((wf * level ** 2).sum()))
    level_var_h = sigma_level ** 2 * h
    total = float(level_var_h * n + np.trace(cov))
    explained = {"level": level_var_h * n / total, "spread_factors": float(vals[:k].sum() / total),
                 "idio": float(psi.sum() / total)}
    d1 = (x - level[:, None]).var(axis=0)
    ratio = float(np.median(np.diag(cov) / np.where(d1 > 0, h * d1, np.nan)))
    return FactorModel(tuple(dy.columns), h, phi, psi, sigma_level, explained, ratio, backfilled or {}, noise)


def simulate_shocks(model: FactorModel, horizon_bd: int, z_level: np.ndarray, rng: np.random.Generator,
                    spread_df: float | None = None, idio_scale: float = 1.0) -> np.ndarray:
    """(paths x bonds) yield shocks (bp) at delivery: the level from ``z_level`` (standard
    normals shared with the caller, so tiers use the same level draws) x sigma x
    sqrt(horizon), plus spread factors and idiosyncratic noise at the model's own horizon -
    all antithetic with the level. ``spread_df``: if set, the spread part (factors + idio)
    is multivariate Student-t with that many degrees of freedom - ONE mixing draw per path,
    shared by the whole basket (a quiet regime vs a jump regime), scaled so the variance is
    exactly the normal's. Measured 2026-10-02: CTD/runner-up spread changes have excess
    kurtosis 2.6-5.2 (a t with ~6 df has 3), and a normal with the RIGHT variance
    over-predicts moderate moves, hence too many switches."""
    paths = z_level.size
    half = paths // 2
    zs = rng.standard_normal((half, model.phi.shape[1]))
    ze = rng.standard_normal((half, len(model.cusips)))
    zs, ze = np.vstack([zs, -zs]), np.vstack([ze, -ze])
    spread = zs @ model.phi.T + ze * np.sqrt(model.psi * idio_scale)[None, :]  # idio_scale: a variance multiplier
    if spread_df is not None:
        if spread_df <= 2:
            raise ValueError("spread_df must exceed 2 (finite variance)")
        w = (spread_df - 2) / rng.chisquare(spread_df, half)  # E[w] = 1: variance kept
        spread = spread * np.sqrt(np.concatenate([w, w]))[:, None]
    level = model.sigma_level * np.sqrt(max(horizon_bd, 0)) * z_level
    return level[:, None] + spread
