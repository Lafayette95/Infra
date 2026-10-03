"""Known-problem visibility for PCA and regressions: do the fitted structures hold up?

PCA (the classic failure modes of a rates PCA):

* ``factor_correlations`` - PCs are uncorrelated over the FIT sample by construction, but
  not inside sub-periods: a level/slope correlation of 0.6 in one year means the factors
  were not separate risks then (and a "PC-neutral" hedge wasn't neutral).
* ``eigenvector_stability`` - refit on a schedule and measure how far each loading vector
  and the k-dimensional subspace move (|cosine| to the previous fit and to a reference;
  principal angles between subspaces). A PC whose eigenvalue is close to the next one's
  (``gap`` near 1) is not identified: it rotates freely inside the pair even though the
  pair's subspace is stable - so read the per-vector cosine together with the gap and the
  subspace angle.
* ``parallel_analysis`` - how many PCs beat the same data with each column shuffled
  (destroys co-movement, keeps each column's distribution); complements the
  Marchenko-Pastur edge reported by every fit.
* ``bootstrap_loadings`` - moving-block bootstrap bands for the loadings.

Regression:

* ``coefficient_stability`` - the same regression fitted separately in each sub-period.
* ``rolling_fit`` - coefficient / R^2 paths from refits (a thin wrapper of ``fit_path``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.models.stats.common import pick
from infra.models.walk_forward import fit_path, refit_dates


# --------------------------------------------------------------------------- PCA
def factor_correlations(model, prepared: pd.DataFrame, *, freq: str | None = "YE", window: int | None = None,
                        start=None, end=None) -> pd.DataFrame:
    """Correlation of every PC pair within each sub-period (``freq``: calendar buckets) or
    over a rolling ``window`` of rows, using the FITTED model's scores (one fixed set of
    loadings - the question is whether ITS factors stay uncorrelated). Long frame:
    ``period, pc_i, pc_j, corr, n``."""
    scores = pick(model.predict(prepared, start=start, end=end), "score").dropna()
    pcs = list(scores.columns)
    rows = []
    if window is not None:
        for i, a in enumerate(pcs):
            for b in pcs[i + 1:]:
                c = scores[a].rolling(window).corr(scores[b])
                rows.append(pd.DataFrame({"period": c.index, "pc_i": a, "pc_j": b, "corr": c.to_numpy(),
                                          "n": window}))
        return pd.concat(rows, ignore_index=True).dropna(subset=["corr"])
    for period, g in scores.groupby(pd.Grouper(freq=freq)):
        if len(g) < 5:
            continue
        C = g.corr()
        for i, a in enumerate(pcs):
            for b in pcs[i + 1:]:
                rows.append({"period": period, "pc_i": a, "pc_j": b, "corr": float(C.loc[a, b]), "n": len(g)})
    return pd.DataFrame(rows, columns=["period", "pc_i", "pc_j", "corr", "n"])


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between the column spaces of A and B (orthonormal
    columns): 0 = same subspace."""
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def eigenvector_stability(make_model, raw: pd.DataFrame, start, end, *, refit: str | int = "ME",
                          reference: str = "first") -> pd.DataFrame:
    """Refit at each ``refit`` date and track, per fit date and PC: ``eigenvalue``,
    ``explained``, ``gap`` (lambda_i / lambda_{i+1}), ``cos_prev`` (|cosine| of the loading
    vector with the previous fit's), ``cos_ref`` (with the ``reference`` fit: ``"first"``
    or ``"last"``), and for the whole top-k subspace ``subspace_angle_prev`` /
    ``subspace_angle_ref`` (largest principal angle, degrees). Loadings are compared on
    the columns common to both fits."""
    probe = make_model() if callable(make_model) and not hasattr(make_model, "fit") else make_model
    dates = refit_dates(probe.prepare(raw).index, start, end, refit)
    fits = fit_path(make_model, raw, dates)
    if not fits:
        return pd.DataFrame()
    keys = list(fits)
    ref = fits[keys[0] if reference == "first" else keys[-1]]
    rows, prev = [], None
    for d, m in fits.items():
        L = m.loadings
        ev = m.explained
        k = L.shape[1]

        def compare(other):
            common = [c for c in L.index if c in other.loadings.index]
            A, B = L.loc[common].to_numpy(), other.loadings.loc[common].to_numpy()
            A, B = np.linalg.qr(A)[0], np.linalg.qr(B)[0]
            kk = min(A.shape[1], B.shape[1])
            cos = np.abs(np.sum(L.loc[common].to_numpy()[:, :kk] * other.loadings.loc[common].to_numpy()[:, :kk],
                                axis=0))
            return cos, float(principal_angles(A, B).max())

        cos_ref, ang_ref = compare(ref)
        cos_prev, ang_prev = compare(prev) if prev is not None else (np.full(k, np.nan), np.nan)
        for j in range(k):
            lam = ev["eigenvalue"].to_numpy()
            rows.append({"fit_as_of": d, "pc": f"PC{j + 1}", "eigenvalue": lam[j], "explained": ev["explained"].iloc[j],
                         "gap": lam[j] / lam[j + 1] if j + 1 < len(lam) and lam[j + 1] > 0 else np.nan,
                         "cos_prev": cos_prev[j] if j < len(cos_prev) else np.nan,
                         "cos_ref": cos_ref[j] if j < len(cos_ref) else np.nan,
                         "subspace_angle_prev": ang_prev, "subspace_angle_ref": ang_ref})
        prev = m
    return pd.DataFrame(rows)


def loading_path(make_model, raw: pd.DataFrame, start, end, *, refit: str | int = "ME",
                 units: bool = False) -> pd.DataFrame:
    """Loadings at every refit, long: ``fit_as_of, column, pc, loading`` (``units=True``:
    one-sd moves in model units, ``loadings_units``)."""
    probe = make_model() if callable(make_model) and not hasattr(make_model, "fit") else make_model
    dates = refit_dates(probe.prepare(raw).index, start, end, refit)
    rows = []
    for d, m in fit_path(make_model, raw, dates).items():
        L = m.loadings_units if units else m.loadings
        long = L.reset_index(names="column").melt(id_vars="column", var_name="pc", value_name="loading")
        rows.append(long.assign(fit_as_of=d))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parallel_analysis(model, prepared: pd.DataFrame, *, n_iter: int = 100, quantile: float = 0.95,
                      seed: int = 0) -> pd.DataFrame:
    """Horn's parallel analysis on the fitted model's sample: eigenvalues of the data's
    CORRELATION matrix vs the ``quantile`` of those with each column independently
    permuted (destroys co-movement only). ``keep`` = PCs that beat the null. Always on
    correlations, whatever the model's ``scale``: on a covariance the null's eigenvalues
    are just the largest column variances, which says nothing about structure."""
    model.check_fitted()
    f = model.fitted_
    sample = model.prep.transform(f.sample.reindex(columns=f.columns)).dropna()
    X = sample.to_numpy(dtype="float64")
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_iter):
        P = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        null.append(np.sort(np.linalg.eigvalsh(np.cov(P, rowvar=False)))[::-1])
    real = np.sort(np.linalg.eigvalsh(np.cov(Z, rowvar=False)))[::-1]
    q = np.quantile(np.array(null), quantile, axis=0)
    return pd.DataFrame({"eigenvalue": real, "null_q": q, "keep": real > q},
                        index=[f"PC{j + 1}" for j in range(len(real))])


def bootstrap_loadings(model, *, n_boot: int = 200, block: int = 20, seed: int = 0,
                       level: float = 0.90) -> pd.DataFrame:
    """Moving-block bootstrap of the fitted model's sample (blocks of ``block`` rows keep
    short-range dependence): refit, sign-align to the original, report each loading's
    ``low`` / ``high`` band at ``level``. Long: ``column, pc, loading, low, high``."""
    model.check_fitted()
    f = model.fitted_
    sample = f.sample.reindex(columns=f.columns)
    n = len(sample)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        boot = sample.iloc[idx].copy()
        boot.index = sample.index  # keep a valid, increasing time index
        m = type(model)(model.spec)
        try:
            m.fit(boot, as_of=sample.index[-1])
        except ValueError:
            continue
        m.align_to(model, permute=True)
        draws.append(m.loadings.reindex(index=model.loadings.index).to_numpy())
    D = np.array(draws)
    lo, hi = np.quantile(D, (1 - level) / 2, axis=0), np.quantile(D, (1 + level) / 2, axis=0)
    L = model.loadings
    out = []
    for i, c in enumerate(L.index):
        for j, pc in enumerate(L.columns):
            out.append({"column": c, "pc": pc, "loading": L.iloc[i, j], "low": lo[i, j], "high": hi[i, j]})
    return pd.DataFrame(out)


def pc_neutral_weights(model, legs: list[str], fixed: str, *, neutralise: tuple[int, ...] = (1, 2),
                       fixed_weight: float = 1.0) -> pd.Series:
    """Weights on ``legs`` (one of which is ``fixed`` at ``fixed_weight``) that make the
    position's exposure to the listed PCs zero: e.g. a PCA-weighted 2s5s10s fly, long the
    5y belly, neutral to level (PC1) and slope (PC2). Exposures are in model units
    (``loadings_units``): neutral to a one-sd move of each PC."""
    L = model.loadings_units
    free = [c for c in legs if c != fixed]
    pcs = [f"PC{i}" for i in neutralise]
    if len(free) != len(pcs):
        raise ValueError(f"{len(pcs)} PCs to neutralise need {len(pcs)} free legs, got {free}")
    A = L.loc[free, pcs].to_numpy().T
    b = -fixed_weight * L.loc[fixed, pcs].to_numpy()
    w = np.linalg.solve(A, b)
    return pd.Series({fixed: fixed_weight, **dict(zip(free, w))}).reindex(legs)


# --------------------------------------------------------------------------- regression
def coefficient_stability(make_model, raw: pd.DataFrame, *, freq: str = "YE", min_obs: int = 30) -> pd.DataFrame:
    """The regression fitted SEPARATELY on each calendar sub-period: ``period, term, coef,
    se, t, r2, n``. Coefficients that wander far outside each other's error bands mean
    the relation isn't stable (a Chow-test view, without assuming where the break is)."""
    probe = make_model() if callable(make_model) and not hasattr(make_model, "fit") else make_model
    data = probe.prepare(raw)
    rows = []
    for period, g in data.groupby(pd.Grouper(freq=freq)):
        g.attrs = data.attrs
        m = make_model() if callable(make_model) and not hasattr(make_model, "fit") else type(make_model)(make_model.spec)
        try:
            m.fit(g, as_of=g.index.max())
        except (ValueError, np.linalg.LinAlgError):
            continue
        if m.fitted_.stats.get("n", 0) < min_obs:
            continue
        t = m.fitted_.table
        for term in t.index:
            rows.append({"period": period, "term": term, "coef": t.loc[term, "coef"], "se": t.loc[term, "se"],
                         "t": t.loc[term, "t"], "r2": m.fitted_.stats.get("r2", np.nan), "n": m.fitted_.stats["n"]})
    return pd.DataFrame(rows)


def rolling_fit(make_model, raw: pd.DataFrame, start, end, *, refit: str | int = "ME") -> pd.DataFrame:
    """Coefficients and fit statistics at each refit: ``fit_as_of`` x (``coef:<term>``,
    ``se:<term>``, ``r2``, ``resid_std``, ``half_life``)."""
    probe = make_model() if callable(make_model) and not hasattr(make_model, "fit") else make_model
    dates = refit_dates(probe.prepare(raw).index, start, end, refit)
    rows = []
    for d, m in fit_path(make_model, raw, dates).items():
        t, s = m.fitted_.table, m.fitted_.stats
        row = {"fit_as_of": d, **{f"coef:{k}": v for k, v in t["coef"].items()},
               **{f"se:{k}": v for k, v in t["se"].items()}}
        row.update({k: s.get(k, np.nan) for k in ("r2", "resid_std", "half_life", "n")})
        rows.append(row)
    return pd.DataFrame(rows).set_index("fit_as_of") if rows else pd.DataFrame()
