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
    idio_corr: np.ndarray | None = None  # (n, n) correlation of the idiosyncratic moves (None = independent)
    idio_kernel: tuple | None = None  # (a, length_years) of idio_corr, for diagnostics
    level_beta: np.ndarray | None = None  # (n,) each bond's spread move per unit of level move (None = 0)
    joint: bool = False  # True: phi are components of TOTAL yield changes (level inside PC1), no separate level


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
                     max_horizon_share: float = 0.34, noise_removal: float = 0.0,
                     maturities_years: np.ndarray | None = None, level_betas: bool = False) -> FactorModel:
    """Fit on a complete panel of DAILY changes (rows = days, columns = bonds), with the
    spread covariance taken at ``horizon_bd`` (capped at ``max_horizon_share`` of the
    window so enough overlapping changes remain).

    ``noise_removal``: how many MARK-NOISE variances to take off each bond's idiosyncratic
    variance (0 = none, as first built; 1 = the start-of-horizon mark; 2 = both ends of an
    h-day change). A price mark with independent noise of variance s^2 gives its daily
    changes a lag-1 autocovariance of -s^2, so s^2 = max(-autocov, 0) per bond, on the
    spread changes. Motivation (2026-10-02/03): on ZB's ~50-bond basket M2 put only a
    median 61% on the realised CTD (M1 93%), every bond's own noise leaking probability to
    bonds that never win; FedInvest's marks for old off-the-runs are the noisiest part.

    ``maturities_years`` (one per column): if given, the idiosyncratic moves are CORRELATED
    by maturity distance, rho_ij = a x exp(-|m_i - m_j| / l) (i != j; 1 on the diagonal - a
    mix of an exponential kernel and the identity, so always a valid correlation), a and l
    fitted to the residual (spread covariance minus the factors) correlations of pairs
    under 2 years apart. Found 2026-10-03 on ZB: the residuals of bonds maturing within 3
    months of each other correlate 0.35-0.63, within 3-6 months ~0.4, fading by a year;
    taken as independent, a near-twin CTD / runner-up pair got ~2x its variance (z-score
    sd 0.51 for pairs < 6 months apart).

    ``level_betas``: the spreads are taken to move with the level, s_i = beta_i x L + rest,
    beta fitted on the same h-day changes (slow EWMA), and the PCA / idiosyncratic split
    runs on the REST; ``simulate_shocks`` then adds beta x the level shock. Found
    2026-10-03 on ZB: in a level move the 15y bonds' spreads move +0.04..+0.07 per unit and
    the 25y bonds' -0.04..-0.06 (the curve flattens as yields fall), the level explaining
    15-45% of a bond's spread variance; drawn independently of the level, a CTD / runner-
    up pair with opposite betas loses the covariance that offsets its DV01-difference term,
    and M2 overstated ZB's pair variance in every year (z-score sd 0.34-0.77)."""
    x = dy.to_numpy(dtype="float64")
    t, n = x.shape
    if t < 40 or n == 0:
        raise ValueError(f"too little history to fit: {t} days x {n} bonds")
    h = int(max(1, min(horizon_bd, int(t * max_horizon_share))))
    level = x.mean(axis=1)
    spread_cum = np.cumsum(x - level[:, None], axis=0)
    dh = spread_cum[h:] - spread_cum[:-h]  # overlapping h-day spread changes
    w = ewma_weights(dh.shape[0], lam_slow)
    beta = None
    if level_betas:
        lvl_cum = np.cumsum(level)
        lh = lvl_cum[h:] - lvl_cum[:-h]
        denom = float((w * lh) @ lh)
        if denom > 0:
            beta = (w * lh) @ dh / denom
            dh = dh - lh[:, None] * beta[None, :]
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
    idio_corr, kernel = None, None
    if maturities_years is not None and n > 1:
        res = cov - phi @ phi.T
        d = np.sqrt(np.clip(np.diag(res), 1e-12, None))
        rc = res / np.outer(d, d)
        m = np.asarray(maturities_years, dtype="float64")
        gap = np.abs(m[:, None] - m[None, :])
        iu = np.triu_indices(n, 1)
        g, r = gap[iu], rc[iu]
        near = g < 2.0
        best = (0.0, 0.5, np.inf)
        if near.sum() >= 5:
            for ell in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0):
                k_ = np.exp(-g[near] / ell)
                a = float(np.clip((k_ @ r[near]) / (k_ @ k_), 0.0, 0.95))
                err = float(((r[near] - a * k_) ** 2).sum())
                if err < best[2]:
                    best = (a, ell, err)
        a, ell = best[0], best[1]
        if a > 0:
            idio_corr = a * np.exp(-gap / ell)
            np.fill_diagonal(idio_corr, 1.0)
            kernel = (a, ell)
    return FactorModel(tuple(dy.columns), h, phi, psi, sigma_level, explained, ratio, backfilled or {}, noise,
                       idio_corr, kernel, beta)


def fit_joint_model(dy: pd.DataFrame, horizon_bd: int, *, k: int = 4, lam_slow: float = 0.99,
                    lam_fast: float = 0.94, backfilled: dict | None = None, min_idio_share: float = 0.01,
                    max_horizon_share: float = 0.34) -> FactorModel:
    """The JOINT horizon PCA (user's original M2 design, built 2026-10-03): the covariance
    of h-day TOTAL yield changes of the basket (level included, slow EWMA, overlapping
    changes), top ``k`` components + a diagonal idiosyncratic remainder. PC1 is the
    level-like component and carries the curve's own co-movement (its loadings aren't flat:
    on ZB the 15y bonds load more than the 25y ones - the flattening that the split model
    lost by drawing the level and the spreads independently). Two speeds kept: PC1's
    loadings are rescaled so the basket-level variance at the horizon matches the FAST
    EWMA of the daily basket-level changes (``lam_fast``) x h; the rest stays slow. The
    returned model has ``joint=True``: ``simulate_shocks`` draws PC1 with the caller's
    ``z_level`` (the same draws M1 uses) and the other components independently."""
    x = dy.to_numpy(dtype="float64")
    t, n = x.shape
    if t < 40 or n == 0:
        raise ValueError(f"too little history to fit: {t} days x {n} bonds")
    h = int(max(1, min(horizon_bd, int(t * max_horizon_share))))
    cum = np.cumsum(x, axis=0)
    dh = cum[h:] - cum[:-h]
    w = ewma_weights(dh.shape[0], lam_slow)
    cov = (w[:, None] * dh).T @ dh
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = vals[::-1].clip(min=0.0), vecs[:, ::-1]
    k = max(1, min(k, n))
    phi = vecs[:, :k] * np.sqrt(vals[:k])[None, :]
    if phi[:, 0].sum() < 0:
        phi[:, 0] = -phi[:, 0]  # PC1 oriented as "all yields up"
    # floor on the SPREAD variance (each bond net of the basket mean), as in the split model:
    # 1% of the TOTAL h-day variance - level-dominated - would be a ~5bp sd of pure noise per
    # bond on ZB (found 2026-10-03), several times the real idiosyncratic size
    dm = dh - dh.mean(axis=1, keepdims=True)
    psi = np.maximum(np.diag(cov) - (phi ** 2).sum(axis=1), min_idio_share * ((w[:, None] * dm) * dm).sum(axis=0))
    level = x.mean(axis=1)
    wf = ewma_weights(t, lam_fast)
    sigma_level = float(np.sqrt((wf * level ** 2).sum()))
    ones = np.full(n, 1.0 / n)
    slow_level_var = float(ones @ cov @ ones)
    pc1_level_var = float((ones @ phi[:, 0]) ** 2)
    rest = max(slow_level_var - pc1_level_var, 0.0)
    target = sigma_level ** 2 * h
    if pc1_level_var > 0 and target > rest:
        phi[:, 0] *= np.sqrt((target - rest) / pc1_level_var)
    total = float(np.trace(cov))
    explained = {"level": float(vals[0] / total) if total else np.nan,
                 "spread_factors": float(vals[1:k].sum() / total) if total else np.nan,
                 "idio": float(psi.sum() / total) if total else np.nan}
    return FactorModel(tuple(dy.columns), h, phi, psi, sigma_level, explained, np.nan, backfilled or {}, joint=True)


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
    if model.joint:  # PC1 takes the shared level draws; the factors are already AT the horizon
        zo = rng.standard_normal((half, model.phi.shape[1] - 1))
        ze = rng.standard_normal((half, len(model.cusips)))
        zs = np.column_stack([z_level, np.vstack([zo, -zo])])
        ze = np.vstack([ze, -ze])
        spread = zs[:, 1:] @ model.phi[:, 1:].T + ze * np.sqrt(model.psi * idio_scale)[None, :]
        if spread_df is not None:
            if spread_df <= 2:
                raise ValueError("spread_df must exceed 2 (finite variance)")
            wt = (spread_df - 2) / rng.chisquare(spread_df, half)
            spread = spread * np.sqrt(np.concatenate([wt, wt]))[:, None]
        return z_level[:, None] * model.phi[:, 0][None, :] + spread
    zs = rng.standard_normal((half, model.phi.shape[1]))
    ze = rng.standard_normal((half, len(model.cusips)))
    zs, ze = np.vstack([zs, -zs]), np.vstack([ze, -ze])
    if model.idio_corr is not None:  # idiosyncratic moves correlated by maturity distance
        ze = ze @ np.linalg.cholesky(model.idio_corr + 1e-10 * np.eye(len(model.cusips))).T
    spread = zs @ model.phi.T + ze * np.sqrt(model.psi * idio_scale)[None, :]  # idio_scale: a variance multiplier
    if spread_df is not None:
        if spread_df <= 2:
            raise ValueError("spread_df must exceed 2 (finite variance)")
        w = (spread_df - 2) / rng.chisquare(spread_df, half)  # E[w] = 1: variance kept
        spread = spread * np.sqrt(np.concatenate([w, w]))[:, None]
    level = model.sigma_level * np.sqrt(max(horizon_bd, 0)) * z_level
    load = 1.0 if model.level_beta is None else 1.0 + model.level_beta[None, :]
    return level[:, None] * load + spread
