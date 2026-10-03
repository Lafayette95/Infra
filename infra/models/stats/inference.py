"""Pure inference helpers for the statistical models: coefficient covariances (classical,
White, Newey-West), coefficient tables, and the residual diagnostics every regression
reports. numpy/scipy only (no statsmodels in this environment); each formula is the
textbook one, checked in ``tests/test_stats_models.py`` against closed forms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

COV_KINDS = ("nonrobust", "HC0", "HC1", "HAC")


def hac_default_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 (n/100)^(2/9))."""
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def sandwich(X: np.ndarray, psi: np.ndarray, bread: np.ndarray, kind: str, lags: int | None = None) -> np.ndarray:
    """``bread @ meat @ bread`` for an M-estimator whose score is ``x_i * psi_i``.

    ``bread`` = the inverse Hessian (OLS: (X'X)^-1, logit: (X'DX)^-1). ``kind``: HC0, HC1
    (HC0 x n/(n-k)), HAC (Newey-West, Bartlett kernel, ``lags`` default
    ``hac_default_lags``)."""
    n, k = X.shape
    S = X * psi[:, None]
    meat = S.T @ S
    if kind == "HAC":
        L = hac_default_lags(n) if lags is None else int(lags)
        for lag in range(1, L + 1):
            w = 1.0 - lag / (L + 1.0)
            G = S[lag:].T @ S[:-lag]
            meat += w * (G + G.T)
    elif kind not in ("HC0", "HC1"):
        raise ValueError(f"sandwich kind {kind!r}; one of HC0, HC1, HAC")
    cov = bread @ meat @ bread
    if kind == "HC1" and n > k:
        cov *= n / (n - k)
    return cov


def coef_table(names, beta, cov, df: float | None, *, level: float = 0.95) -> pd.DataFrame:
    """``coef, se, t, p, ci_low, ci_high`` per coefficient. ``df`` None -> normal
    (z) inference, else Student t. ``cov`` None -> coefficients only (e.g. lasso)."""
    beta = np.asarray(beta, dtype="float64")
    out = pd.DataFrame({"coef": beta}, index=pd.Index(list(names), name="term"))
    if cov is None:
        for c in ("se", "t", "p", "ci_low", "ci_high"):
            out[c] = np.nan
        return out
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    dist = stats.norm if df is None else stats.t(df)
    q = dist.ppf(0.5 + level / 2.0)
    out["se"], out["t"] = se, t
    out["p"] = 2.0 * dist.sf(np.abs(t))
    out["ci_low"], out["ci_high"] = beta - q * se, beta + q * se
    return out


def vif(X: np.ndarray, names) -> pd.Series:
    """Variance inflation factor of each non-constant column: 1 / (1 - R^2 of that column
    on the others). > 10 is the usual collinearity alarm."""
    X = np.asarray(X, dtype="float64")
    out = {}
    for j, name in enumerate(names):
        y = X[:, j]
        if np.ptp(y) == 0:
            continue
        others = np.column_stack([np.ones(len(y)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        r = y - others @ beta
        r2 = 1.0 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
        out[name] = 1.0 / max(1.0 - r2, 1e-12)
    return pd.Series(out, dtype="float64")


# --------------------------------------------------------------------------- residuals
def durbin_watson(e: np.ndarray) -> float:
    e = np.asarray(e, dtype="float64")
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2)) if len(e) > 1 else np.nan


def ljung_box(e: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """(Q, p): no residual autocorrelation up to ``lags``."""
    e = np.asarray(e, dtype="float64") - np.mean(e)
    n = len(e)
    lags = min(lags, n - 2)
    if lags < 1:
        return np.nan, np.nan
    denom = e @ e
    q = n * (n + 2) * sum((e[k:] @ e[:-k] / denom) ** 2 / (n - k) for k in range(1, lags + 1))
    return float(q), float(stats.chi2.sf(q, lags))


def jarque_bera(e: np.ndarray) -> tuple[float, float]:
    """(JB, p): residual normality from skew and kurtosis."""
    e = np.asarray(e, dtype="float64")
    if len(e) < 4:
        return np.nan, np.nan
    s, k = stats.skew(e), stats.kurtosis(e, fisher=False)
    jb = len(e) / 6.0 * (s ** 2 + (k - 3.0) ** 2 / 4.0)
    return float(jb), float(stats.chi2.sf(jb, 2))


def breusch_pagan(e: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """(LM, p) Koenker's studentised form: n R^2 of e^2 on X (X with constant)."""
    e2 = np.asarray(e, dtype="float64") ** 2
    X = np.asarray(X, dtype="float64")
    if X.shape[1] < 2 or len(e2) <= X.shape[1]:
        return np.nan, np.nan
    beta, *_ = np.linalg.lstsq(X, e2, rcond=None)
    r = e2 - X @ beta
    r2 = 1.0 - (r @ r) / ((e2 - e2.mean()) @ (e2 - e2.mean()))
    lm = len(e2) * r2
    return float(lm), float(stats.chi2.sf(lm, X.shape[1] - 1))


def ar1(e: np.ndarray) -> tuple[float, float]:
    """(phi, half-life in rows) of e_t = c + phi e_{t-1} + u: how fast a residual mean
    reverts. Half-life NaN if phi is outside (0, 1) (no reversion / oscillating)."""
    e = np.asarray(e, dtype="float64")
    if len(e) < 3:
        return np.nan, np.nan
    x, y = e[:-1] - e[:-1].mean(), e[1:] - e[1:].mean()
    phi = float((x @ y) / (x @ x)) if x @ x > 0 else np.nan
    hl = float(-np.log(2.0) / np.log(phi)) if 0.0 < phi < 1.0 else np.nan
    return phi, hl


# Critical values of the ADF t-statistic. "constant": the plain unit-root test with an
# intercept (MacKinnon 2010, asymptotic). "eg2": the same statistic on the residual of a
# 2-variable cointegrating regression (Engle-Granger; MacKinnon 2010, N=2, constant).
ADF_CRITICAL = {"constant": {0.01: -3.43, 0.05: -2.86, 0.10: -2.57},
                "eg2": {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}}


def adf(e: np.ndarray, lags: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic (constant, ``lags`` lagged differences). More
    negative = more stationary; compare with ``ADF_CRITICAL``."""
    e = np.asarray(e, dtype="float64")
    d = np.diff(e)
    n = len(d) - lags
    if n < lags + 5:
        return np.nan
    cols = [np.ones(n), e[lags:-1]] + [d[lags - j:-j] for j in range(1, lags + 1)]
    X = np.column_stack(cols)
    y = d[lags:]
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    r = y - X @ beta
    s2 = (r @ r) / (n - X.shape[1])
    return float(beta[1] / np.sqrt(s2 * XtX_inv[1, 1]))


def residual_stats(e: np.ndarray, X: np.ndarray | None = None) -> dict[str, float]:
    """The residual diagnostics every regression reports."""
    phi, hl = ar1(e)
    lb, lb_p = ljung_box(e)
    jb, jb_p = jarque_bera(e)
    out = {"dw": durbin_watson(e), "resid_ar1": phi, "half_life": hl, "adf_t": adf(e),
           "ljung_box": lb, "ljung_box_p": lb_p, "jarque_bera": jb, "jarque_bera_p": jb_p}
    if X is not None:
        out["breusch_pagan"], out["breusch_pagan_p"] = breusch_pagan(e, X)
    return out
