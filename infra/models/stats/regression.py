"""Regressions on any series: prepare -> fit (point in time) -> predict (fitted, residual).

    model = make_regression("ols_hac", y="bond:US_BOND_10y", x=("bond:US_BOND_2y",))
    data = model.prepare(panel)                  # stateless prep (+ lags / lead)
    model.fit(data, as_of="2026-06-30")          # sample up to as_of only; window, weights
    model.summary()                              # coefficient table + fit statistics
    out = model.predict(data, start="2026-06-30")  # rows after the fit date, params fixed:
                                                 # y, fitted, residual, resid_z, in_sample
    model.params()                               # tidy frame: stack one per refit, rebuild later

Classes (``REGRESSION_CLASSES``, chosen by ``RegressionSpec.method``):

* ``LinearRegression`` - OLS / WLS (exponential weights), classical, White or Newey-West
  errors. ``UnivariateRegression`` / ``MultivariateRegression`` pin the regressor count.
* ``StepwiseRegression`` - forward / backward / both, by p-value, AIC or BIC.
* ``RidgeRegression``, ``LassoRegression``, ``ElasticNetRegression`` - penalised (glmnet
  objective, standardised X, alpha fixed or by forward-chaining CV).
* ``HuberRegression`` - robust to outliers (bad prints, event days).
* ``QuantileRegression`` - any conditional quantile (LAD at tau 0.5).
* ``HockeyStickRegression`` - piecewise linear in one regressor, knot(s) estimated.
* ``TotalLeastSquares`` - orthogonal / Deming: errors in y AND x (symmetric hedge ratios).
* ``LogitRegression`` / ``ProbitRegression`` - binary target (y > threshold).
* ``KalmanRegression`` - coefficients that drift (random walk), filtered forward.

Units: ``fitted`` / ``residual`` come back in y's PREPARED units (after stateless steps,
before any stateful scaling, which is undone); ``resid_z`` is the residual over its
fit-sample standard deviation - the usual relative-value signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import optimize, sparse, stats

from infra.models.stats import inference as inf
from infra.models.stats.common import (PARAM_COLUMNS, StatModel, decay_weights, effective_n, params_frame,
                                       section)
from infra.models.stats.config import RegressionSpec, get_regression_spec

log = logging.getLogger(__name__)

CONST = "const"


@dataclass
class Estimate:
    """What an estimator returns (model units: prepared + stateful-scaled)."""
    beta: np.ndarray
    cov: np.ndarray | None
    df_resid: float | None            # None = normal (z) inference
    stats: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    XtX_inv: np.ndarray | None = None  # for prediction standard errors (OLS family)
    sigma2: float = np.nan


@dataclass
class RegressionFit:
    as_of: pd.Timestamp
    target: str
    regressors: list[str]
    terms: list[str]
    beta: pd.Series
    table: pd.DataFrame
    stats: dict
    resid_std: float
    extra: dict
    XtX_inv: np.ndarray | None
    sigma2: float
    sample: pd.DataFrame | None       # the (unscaled) fit sample, for predict() with no data


# --------------------------------------------------------------------------- estimators
def _weighted(X, y, w):
    if w is None:
        return X, y
    sw = np.sqrt(w / w.mean())
    return X * sw[:, None], y * sw


def ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None, cov: str, hac_lags: int | None) -> Estimate:
    Xw, yw = _weighted(X, y, w)
    n, k = Xw.shape
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    beta = XtX_inv @ Xw.T @ yw
    e = yw - Xw @ beta
    df = max(n - k, 1)
    sigma2 = float(e @ e / df)
    if cov == "nonrobust":
        V = sigma2 * XtX_inv
    else:
        V = inf.sandwich(Xw, e, XtX_inv, cov, hac_lags)
    out = Estimate(beta=beta, cov=V, df_resid=df, XtX_inv=XtX_inv, sigma2=sigma2)
    has_const = bool(np.any(np.all(X == 1.0, axis=0)))
    k_slopes = k - 1 if has_const else k
    if k_slopes > 0:
        # Wald test that every slope is 0 (with the chosen covariance)
        idx = [j for j in range(k) if not np.all(X[:, j] == 1.0)]
        b, Vs = beta[idx], V[np.ix_(idx, idx)]
        F = float(b @ np.linalg.pinv(Vs) @ b / len(idx))
        out.stats.update(F=F, F_p=float(stats.f.sf(F, len(idx), df)))
    ll = -0.5 * n * (np.log(2 * np.pi) + np.log(e @ e / n) + 1)
    out.stats.update(loglik=ll, aic=-2 * ll + 2 * k, bic=-2 * ll + np.log(n) * k, cond_number=float(np.linalg.cond(Xw)))
    return out


def _standardise(X, w, names):
    """(Xs, mean, std) over the non-constant columns, weighted; constants dropped."""
    keep = [j for j, nme in enumerate(names) if nme != CONST]
    Z = X[:, keep]
    ww = np.ones(len(Z)) if w is None else w / w.mean()
    mu = np.average(Z, axis=0, weights=ww)
    sd = np.sqrt(np.average((Z - mu) ** 2, axis=0, weights=ww))
    sd[sd == 0] = 1.0
    return (Z - mu) / sd, mu, sd, keep, ww


def enet_path_fit(Z, y, ww, alpha, l1_ratio, *, beta0=None, tol=1e-8, max_iter=10_000) -> np.ndarray:
    """Weighted elastic net by coordinate descent on centred y and standardised Z."""
    n, p = Z.shape
    W = ww.sum()
    if l1_ratio == 0.0:
        A = (Z.T * ww) @ Z / W + alpha * np.eye(p)
        return np.linalg.solve(A, (Z.T * ww) @ y / W)
    b = np.zeros(p) if beta0 is None else beta0.copy()
    z = (ww[:, None] * Z ** 2).sum(axis=0) / W
    r = y - Z @ b
    for _ in range(max_iter):
        dmax = 0.0
        for j in range(p):
            rho = (ww * Z[:, j]) @ r / W + z[j] * b[j]
            new = np.sign(rho) * max(abs(rho) - alpha * l1_ratio, 0.0) / (z[j] + alpha * (1.0 - l1_ratio))
            if new != b[j]:
                r -= Z[:, j] * (new - b[j])
                dmax = max(dmax, abs(new - b[j]))
                b[j] = new
        if dmax < tol:
            break
    return b


def alpha_grid(Z, y, ww, l1_ratio, n: int) -> np.ndarray:
    W = ww.sum()
    if l1_ratio > 0:
        amax = np.max(np.abs((Z.T * ww) @ y / W)) / l1_ratio
    else:
        amax = 1e3
    amax = max(amax, 1e-8)
    return amax * np.logspace(0, -4, n)


def forward_chain_cv(Z, y, ww, l1_ratio, grid, folds: int) -> tuple[float, pd.DataFrame]:
    """Pick alpha by forward-chaining CV: split the sample into ``folds + 1`` contiguous
    blocks; fold i trains on blocks 0..i and tests on block i+1 (never on the past with
    a model fitted on the future). Returns (alpha minimising mean test MSE, the curve)."""
    n = len(y)
    edges = np.linspace(0, n, folds + 2).astype(int)
    rows = []
    for a in grid:
        errs = []
        for i in range(folds):
            tr, te = slice(0, edges[i + 1]), slice(edges[i + 1], edges[i + 2])
            if edges[i + 1] < 5 or edges[i + 2] - edges[i + 1] < 1:
                continue
            Ztr, ytr, wtr = Z[tr], y[tr], ww[tr]
            mu = np.average(Ztr, axis=0, weights=wtr)
            ym = np.average(ytr, weights=wtr)
            b = enet_path_fit(Ztr - mu, ytr - ym, wtr, a, l1_ratio)
            pred = ym + (Z[te] - mu) @ b
            errs.append(np.mean((y[te] - pred) ** 2))
        rows.append({"alpha": a, "cv_mse": float(np.mean(errs)) if errs else np.nan})
    curve = pd.DataFrame(rows)
    best = float(curve.loc[curve["cv_mse"].idxmin(), "alpha"]) if curve["cv_mse"].notna().any() else float(grid[-1])
    return best, curve


# --------------------------------------------------------------------------- base class
class Regression(StatModel):
    """Common machinery; subclasses implement ``estimate`` (and, if the design or the
    link differs, ``features`` / ``link``)."""

    method: str | None = None

    def __init__(self, spec: RegressionSpec | str | None = None, **overrides):
        spec = get_regression_spec(spec, **overrides)
        if self.method is not None and spec.method != self.method:
            spec = replace(spec, method=self.method)
        super().__init__(spec)

    # ------------------------------------------------------------------ columns
    def target_column(self, raw_columns) -> str:
        return self.spec.y if self.spec.y is not None else list(raw_columns)[0]

    def base_regressors(self, raw_columns) -> list[str]:
        y = self.target_column(raw_columns)
        return list(self.spec.x) if self.spec.x is not None else [c for c in raw_columns if c != y]

    def input_columns(self, raw: pd.DataFrame) -> list[str]:
        y = self.target_column(raw.columns)
        return [y] + [c for c in self.base_regressors(raw.columns) if c != y]

    def target_name(self, y: str) -> str:
        return f"{y}.lead{self.spec.y_lead}" if self.spec.y_lead else y

    # ------------------------------------------------------------------ 1. prepare
    def prepare(self, raw: pd.DataFrame, *, skip=()) -> pd.DataFrame:
        """Stateless prep, then the lag / lead columns (``x.lag1``, ``y.lead5``)."""
        base = super().prepare(raw, skip=skip)
        y = self.target_column(raw.columns)
        xs = [c for c in self.base_regressors(raw.columns) if c != y]
        out = {}
        target = self.target_name(y)
        out[target] = base[y].shift(-self.spec.y_lead) if self.spec.y_lead else base[y]
        for c in xs:
            for lag in self.spec.x_lags:
                name = c if lag == 0 else f"{c}.lag{lag}"
                out[name] = base[c].shift(lag) if lag else base[c]
                if lag and c in self.prep.by_column:
                    self.prep.by_column[name] = self.prep.by_column[c]
        if self.spec.y_lead and y in self.prep.by_column:
            self.prep.by_column[target] = self.prep.by_column[y]
        frame = pd.DataFrame(out, index=base.index)
        frame.attrs["target"], frame.attrs["y_lead"] = target, self.spec.y_lead
        return frame

    def _columns(self, prepared: pd.DataFrame) -> tuple[str, list[str]]:
        target = prepared.attrs.get("target", prepared.columns[0])
        return target, [c for c in prepared.columns if c != target]

    # ------------------------------------------------------------------ design
    def features(self, scaled: pd.DataFrame, regressors: list[str]) -> pd.DataFrame:
        F = scaled[regressors].copy()
        if self.spec.add_const:
            F.insert(0, CONST, 1.0)
        return F

    def link(self, eta: np.ndarray) -> np.ndarray:
        return eta

    def transform_target(self, y: pd.Series) -> pd.Series:
        return y

    # ------------------------------------------------------------------ 2. fit
    def fit(self, prepared: pd.DataFrame, as_of=None) -> "Regression":
        target, regressors = self._columns(prepared)
        sample, as_of = self.fit_sample(prepared, as_of)
        h = int(prepared.attrs.get("y_lead", 0) or 0)
        if h:
            # the target of row t is known at t+h: keep only rows whose target is known by as_of
            pos = prepared.index.get_indexer(sample.index)
            target_time = prepared.index[np.minimum(pos + h, len(prepared) - 1)]
            known = (pos + h < len(prepared)) & np.asarray(
                pd.Series(target_time) <= pd.Timestamp(as_of) + (pd.Timedelta(days=1) - pd.Timedelta(1)
                                                                  if pd.Timestamp(as_of) == pd.Timestamp(as_of).normalize()
                                                                  else pd.Timedelta(0)))
            sample = sample[known]
        sample = sample.dropna(subset=[target] + regressors)
        if len(sample) < max(self.spec.min_obs, len(regressors) + 2):
            raise ValueError(f"{type(self).__name__}: {len(sample)} usable rows up to {pd.Timestamp(as_of).date()} "
                             f"(min_obs {self.spec.min_obs})")
        w = decay_weights(len(sample), self.spec.halflife)
        self.prep.fit(sample, w)
        scaled = self.prep.transform(sample)
        yv = self.transform_target(scaled[target]).to_numpy(dtype="float64")
        F = self.features(scaled, regressors)
        est = self.estimate(F.to_numpy(dtype="float64"), yv, w, list(F.columns), scaled, regressors)
        terms = list(F.columns) if "terms" not in est.extra else est.extra.pop("terms")
        beta = pd.Series(est.beta, index=terms)
        table = inf.coef_table(terms, est.beta, est.cov, est.df_resid)
        self.fitted_ = RegressionFit(as_of=pd.Timestamp(as_of), target=target, regressors=regressors, terms=terms,
                                     beta=beta, table=table, stats={}, resid_std=np.nan, extra=est.extra,
                                     XtX_inv=est.XtX_inv, sigma2=est.sigma2, sample=sample)
        Fx = self.features(scaled, regressors)
        fitted = self.link(Fx[terms].to_numpy(dtype="float64") @ est.beta)
        e = yv - fitted
        self.fitted_.resid_std = self.residual_scale(e, w)
        self.fitted_.stats = self.fit_stats(yv, fitted, w, len(terms), Fx[terms].to_numpy(dtype="float64")) | est.stats
        if len(regressors) > 1:
            v = inf.vif(scaled[regressors].to_numpy(dtype="float64"), regressors)
            self.fitted_.table["vif"] = v.reindex(table.index)
        sd_y = np.std(yv)
        std_coef = {t: est.beta[i] * np.std(Fx[t].to_numpy(dtype="float64")) / sd_y if sd_y else np.nan
                    for i, t in enumerate(terms) if t != CONST}
        self.fitted_.table["std_coef"] = pd.Series(std_coef).reindex(table.index)
        return self

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        raise NotImplementedError

    def residual_scale(self, e: np.ndarray, w) -> float:
        ww = np.ones_like(e) if w is None else w
        return float(np.sqrt(np.average(e ** 2, weights=ww) * len(e) / max(len(e) - 1, 1)))

    def fit_stats(self, y, fitted, w, k, X) -> dict:
        e = y - fitted
        ww = np.ones_like(y) if w is None else w
        ybar = np.average(y, weights=ww)
        sst = float(np.sum(ww * (y - ybar) ** 2))
        sse = float(np.sum(ww * e ** 2))
        n = len(y)
        n_eff = effective_n(w, n)
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        out = {"n": n, "n_eff": n_eff, "k": k, "r2": r2,
               "adj_r2": 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1) if np.isfinite(r2) else np.nan,
               "rmse": float(np.sqrt(sse / ww.sum())), "resid_std": self.fitted_.resid_std,
               "corr": float(np.corrcoef(y, fitted)[0, 1]) if np.std(fitted) > 0 else np.nan}
        out.update(inf.residual_stats(e, X))
        return out

    # ------------------------------------------------------------------ 3. predict
    def predict(self, prepared: pd.DataFrame | None = None, *, start=None, end=None,
                contributions: bool = False) -> pd.DataFrame:
        """Per row of ``prepared`` (strictly after ``start``, up to ``end``; with no data,
        the fit sample itself): ``y``, ``fitted``, ``residual``, ``resid_z``,
        ``in_sample`` (row known by the fit date), ``fitted_se`` where the model has one,
        and with ``contributions=True`` each term's ``contrib:<term>`` (sums to fitted)."""
        self.check_fitted()
        f = self.fitted_
        frame = f.sample if prepared is None else self.predict_window(prepared, start, end)
        if frame is None:
            raise ValueError("no data: this fit was rebuilt from params; pass prepared data")
        scaled = self.prep.transform(frame)
        F = self.features(scaled, f.regressors)[f.terms]
        Xv = F.to_numpy(dtype="float64")
        eta = Xv @ f.beta.to_numpy()
        fitted = self.link(eta)
        out = pd.DataFrame(index=frame.index)
        has_y = f.target in scaled.columns
        yv = self.transform_target(scaled[f.target]).to_numpy(dtype="float64") if has_y else np.full(len(frame), np.nan)
        out["y"] = self.prep.inverse(yv, f.target) if not self.binary else yv
        out["fitted"] = self.prep.inverse(fitted, f.target) if not self.binary else fitted
        resid = yv - fitted
        out["residual"] = self.prep.inverse(resid, f.target, shift=False) if not self.binary else resid
        out["resid_z"] = self.standardised_residual(resid, fitted)
        if f.XtX_inv is not None and np.isfinite(f.sigma2) and not self.binary:
            lev = np.einsum("ij,jk,ik->i", Xv, f.XtX_inv, Xv)
            out["fitted_se"] = self.prep.inverse(np.sqrt(f.sigma2 * (1.0 + lev)), f.target, shift=False)
        out["in_sample"] = self.in_sample(frame.index)
        if contributions:
            for j, t in enumerate(f.terms):
                c = Xv[:, j] * f.beta.iloc[j]
                out[f"contrib:{t}"] = c if self.binary else self.prep.inverse(c, f.target, shift=(t == CONST))
            if not self.spec.add_const and not self.binary:
                out[f"contrib:{CONST}"] = self.prep.inverse(np.zeros(len(out)), f.target)
        return out

    binary = False

    def standardised_residual(self, resid, fitted):
        return resid / self.fitted_.resid_std if self.fitted_.resid_std else np.full(len(resid), np.nan)

    # ------------------------------------------------------------------ reporting
    @property
    def coef(self) -> pd.Series:
        self.check_fitted()
        return self.fitted_.beta

    def summary(self) -> str:
        self.check_fitted()
        f = self.fitted_
        head = (f"{type(self).__name__} [{self.spec.name}] y={f.target}  as_of={f.as_of.date()}  "
                f"n={f.stats.get('n')}  cov={self.spec.cov}")
        keys = ("r2", "adj_r2", "F", "F_p", "rmse", "aic", "bic", "dw", "half_life", "adf_t", "loglik",
                "pseudo_r2", "lr", "lr_p", "auc", "alpha", "df_eff", "knot", "slope_left", "slope_right")
        s = "  ".join(f"{k}={f.stats[k]:.4g}" for k in keys if k in f.stats and np.isfinite(f.stats[k]))
        return f"{head}\n{f.table.to_string(float_format=lambda v: f'{v:.4g}')}\n{s}"

    def params(self) -> pd.DataFrame:
        self.check_fitted()
        f = self.fitted_
        parts = [params_frame("coef", f.table), params_frame("stat", f.stats), self.prep.params()]
        parts += [params_frame(k, v) for k, v in f.extra.items() if isinstance(v, (pd.DataFrame, pd.Series, dict))]
        meta = pd.DataFrame([{"section": "meta", "row": "target", "col": f.target, "value": np.nan},
                             {"section": "meta", "row": "as_of", "col": f.as_of.isoformat(), "value": np.nan}]
                            + [{"section": "meta", "row": "regressor", "col": r, "value": float(i)}
                               for i, r in enumerate(f.regressors)])
        return pd.concat([meta] + [p for p in parts if not p.empty], ignore_index=True)[PARAM_COLUMNS]

    @classmethod
    def from_params(cls, params: pd.DataFrame, spec: RegressionSpec | str | None = None, *, as_of=None,
                    **overrides) -> "Regression":
        """Rebuild a fitted model from ``params()`` (e.g. a walk-forward's stored row for
        one fit date) - enough to ``predict`` new data; no fit sample attached."""
        model = make_regression(spec, **overrides) if cls is Regression else cls(spec, **overrides)
        meta = params[params["section"] == "meta"]
        target = meta.loc[meta["row"] == "target", "col"].iloc[0]
        regs = meta[meta["row"] == "regressor"].sort_values("value")["col"].tolist()
        table = section(params, "coef")
        stats_ = section(params, "stat")["value"].to_dict()
        extra = {s: section(params, s) for s in params["section"].unique()
                 if s not in ("coef", "stat", "prep", "meta")}
        model.prep.load_params(params)
        stored = meta.loc[meta["row"] == "as_of", "col"]
        as_of = pd.Timestamp(as_of if as_of is not None else (stored.iloc[0] if len(stored) else pd.NaT))
        model.fitted_ = RegressionFit(
            as_of=as_of, target=target, regressors=regs,
            terms=list(table.index), beta=table["coef"], table=table, stats=stats_,
            resid_std=float(stats_.get("resid_std", np.nan)), extra=extra, XtX_inv=None, sigma2=np.nan, sample=None)
        model.load_extra(extra)
        return model

    def load_extra(self, extra: dict) -> None:
        """Subclass hook: restore method-specific state from ``params``."""


# --------------------------------------------------------------------------- OLS family
class LinearRegression(Regression):
    """OLS / WLS. ``spec.cov``: nonrobust, HC0, HC1 (White) or HAC (Newey-West)."""
    method = "ols"

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        return ols(X, y, w, self.spec.cov, self.spec.hac_lags)


class UnivariateRegression(LinearRegression):
    """OLS with exactly one regressor (x_lags included)."""

    def fit(self, prepared, as_of=None):
        _, regs = self._columns(prepared)
        if len(regs) != 1:
            raise ValueError(f"UnivariateRegression takes one regressor, got {regs}")
        return super().fit(prepared, as_of)


class MultivariateRegression(LinearRegression):
    """OLS with any number of regressors (VIF reported for each)."""


class StepwiseRegression(LinearRegression):
    """Selects regressors at fit time (so each refit of a walk-forward chooses its own);
    excluded ones keep coefficient 0 and ``selected`` 0 in the table."""
    method = "stepwise"

    def _score(self, X, y, w, cols) -> tuple[float, Estimate]:
        est = ols(X[:, cols], y, w, self.spec.cov, self.spec.hac_lags)
        return (est.stats["aic"] if self.spec.criterion == "aic" else est.stats["bic"]), est

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        sp = self.spec
        fixed = [i for i, nme in enumerate(names) if nme == CONST]
        cand = [i for i in range(len(names)) if i not in fixed]
        chosen = list(cand) if sp.direction == "backward" else []
        max_vars = sp.max_vars or len(cand)

        def pvals(cols):
            est = ols(X[:, cols], y, w, sp.cov, sp.hac_lags)
            t = est.beta / np.sqrt(np.clip(np.diag(est.cov), 1e-300, None))
            return dict(zip(cols, 2 * stats.t(est.df_resid).sf(np.abs(t))))

        for _ in range(4 * len(cand) + 4):
            changed = False
            if sp.direction in ("forward", "both") and len(chosen) < max_vars:
                best, best_val = None, None
                base_crit = self._score(X, y, w, fixed + chosen)[0] if sp.criterion != "pvalue" and fixed + chosen else np.inf
                for j in [c for c in cand if c not in chosen]:
                    cols = fixed + chosen + [j]
                    val = pvals(cols)[j] if sp.criterion == "pvalue" else self._score(X, y, w, cols)[0]
                    if best_val is None or val < best_val:
                        best, best_val = j, val
                if best is not None and (best_val < sp.p_enter if sp.criterion == "pvalue" else best_val < base_crit):
                    chosen.append(best)
                    changed = True
            if sp.direction in ("backward", "both") and chosen:
                cols = fixed + chosen
                if sp.criterion == "pvalue":
                    p = pvals(cols)
                    worst = max(chosen, key=lambda c: p[c])
                    drop = p[worst] > sp.p_remove
                else:
                    base_crit = self._score(X, y, w, cols)[0]
                    trial = {c: self._score(X, y, w, [x for x in cols if x != c])[0] if len(cols) > 1 else np.inf
                             for c in chosen}
                    worst = min(trial, key=trial.get)
                    drop = trial[worst] < base_crit
                if drop:
                    chosen.remove(worst)
                    changed = True
            if not changed:
                break
        cols = sorted(fixed + chosen)
        if not cols:
            cols = fixed
        est = ols(X[:, cols], y, w, sp.cov, sp.hac_lags) if cols else Estimate(beta=np.zeros(0), cov=None, df_resid=len(y))
        k = len(names)
        beta, V = np.zeros(k), np.full((k, k), np.nan)
        beta[cols] = est.beta
        if cols:
            V[np.ix_(cols, cols)] = est.cov
        XtX = np.zeros((k, k))
        if cols:
            XtX[np.ix_(cols, cols)] = est.XtX_inv
        out = Estimate(beta=beta, cov=V, df_resid=est.df_resid, stats=dict(est.stats), XtX_inv=XtX, sigma2=est.sigma2)
        out.extra["selected"] = pd.Series({names[i]: float(i in cols) for i in range(k)})
        return out

    def fit(self, prepared, as_of=None):
        super().fit(prepared, as_of)
        sel = self.fitted_.extra["selected"]
        self.fitted_.table["selected"] = sel.reindex(self.fitted_.table.index)
        unsel = sel[sel == 0].index
        self.fitted_.table.loc[unsel, ["se", "t", "p", "ci_low", "ci_high"]] = np.nan
        return self


class PenalizedRegression(LinearRegression):
    """Elastic net family; ``l1_ratio`` 0 = ridge, 1 = lasso. Inference: ridge reports
    standard errors CONDITIONAL on alpha (shrunk, biased toward 0); lasso / elastic net
    report none (no valid classical errors after selection)."""
    method = "elasticnet"
    l1_ratio: float | None = None

    def _l1(self) -> float:
        return self.spec.l1_ratio if self.l1_ratio is None else self.l1_ratio

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        l1 = self._l1()
        Z, mu, sd, keep, ww = _standardise(X, w, names)
        ym = np.average(y, weights=ww)
        yc = y - ym
        extra = {}
        if self.spec.alpha == "cv":
            grid = alpha_grid(Z - 0, yc, ww, l1, self.spec.cv_grid)
            alpha, curve = forward_chain_cv(Z, y, ww, l1, grid, self.spec.cv_folds)
            extra["cv"] = curve.set_index(curve["alpha"].map(lambda a: f"{a:.6g}"))[["alpha", "cv_mse"]]
        else:
            alpha = float(self.spec.alpha)
        bz = enet_path_fit(Z, yc, ww, alpha, l1)
        slopes = bz / sd
        beta = np.zeros(len(names))
        beta[keep] = slopes
        const_idx = [i for i, nme in enumerate(names) if nme == CONST]
        if const_idx:
            beta[const_idx[0]] = ym - mu @ slopes
        e = y - X @ beta
        n = len(y)
        W = ww.sum()
        V, df_eff = None, float(np.count_nonzero(bz)) + len(const_idx)
        if l1 == 0.0:
            A = (Z.T * ww) @ Z / W + alpha * np.eye(Z.shape[1])
            Ainv = np.linalg.inv(A)
            H = Z @ Ainv @ (Z.T * ww) / W
            df_eff = float(np.trace(H)) + len(const_idx)
            sigma2 = float(np.sum(ww * e ** 2) / max(n - df_eff, 1))
            Vz = sigma2 / W ** 2 * Ainv @ ((Z.T * ww) @ Z) @ Ainv
            V = np.full((len(names), len(names)), np.nan)
            V[np.ix_(keep, keep)] = Vz / np.outer(sd, sd)
        sigma2 = float(np.sum(ww * e ** 2) / max(n - df_eff, 1))
        ll = -0.5 * n * (np.log(2 * np.pi) + np.log(np.sum(ww * e ** 2) / W) + 1)
        st = {"alpha": alpha, "l1_ratio": l1, "df_eff": df_eff, "loglik": ll,
              "aic": -2 * ll + 2 * df_eff, "bic": -2 * ll + np.log(n) * df_eff}
        return Estimate(beta=beta, cov=V, df_resid=max(n - df_eff, 1), stats=st, extra=extra, sigma2=sigma2)


class RidgeRegression(PenalizedRegression):
    method = "ridge"
    l1_ratio = 0.0


class LassoRegression(PenalizedRegression):
    method = "lasso"
    l1_ratio = 1.0


class ElasticNetRegression(PenalizedRegression):
    method = "elasticnet"


class HuberRegression(LinearRegression):
    """Huber M-estimator by IRLS, scale = MAD of the residuals re-estimated each pass;
    observations beyond ``huber_k`` scales are down-weighted (not dropped). Errors from
    the sandwich (``cov`` nonrobust -> HC1)."""
    method = "huber"

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        k = self.spec.huber_k
        ww = np.ones(len(y)) if w is None else w / w.mean()
        beta = ols(X, y, w, "nonrobust", None).beta
        for _ in range(200):
            r = y - X @ beta
            s = float(np.median(np.abs(r - np.median(r))) / 0.6745) or 1e-12
            u = r / s
            hw = np.minimum(1.0, k / np.maximum(np.abs(u), 1e-12))
            sw = np.sqrt(ww * hw)
            new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
            if np.max(np.abs(new - beta)) < 1e-10 * (1 + np.max(np.abs(beta))):
                beta = new
                break
            beta = new
        r = y - X @ beta
        u = r / s
        psi = np.clip(u, -k, k) * s * np.sqrt(ww)
        dpsi = (np.abs(u) <= k).astype(float) * ww
        Xs = X * np.sqrt(ww)[:, None] / np.sqrt(ww)[:, None]
        bread = np.linalg.pinv((X * dpsi[:, None]).T @ X)
        kind = "HC1" if self.spec.cov == "nonrobust" else self.spec.cov
        V = inf.sandwich(Xs, psi, bread, kind, self.spec.hac_lags)
        st = {"scale": s, "share_downweighted": float(np.mean(np.abs(u) > k))}
        return Estimate(beta=beta, cov=V, df_resid=max(len(y) - X.shape[1], 1), stats=st,
                        extra={"weights_min": {"value": float(hw.min())}})


class QuantileRegression(LinearRegression):
    """Conditional quantile ``tau`` (linear program, HiGHS). Errors: iid sparsity
    (Hall-Sheather bandwidth) for ``cov="nonrobust"``, else a Powell-kernel sandwich
    (HC / HAC)."""
    method = "quantile"

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        tau = self.spec.tau
        n, k = X.shape
        ww = np.ones(n) if w is None else w / w.mean()
        c = np.concatenate([np.zeros(k), tau * ww, (1 - tau) * ww])
        A = sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)]).tocsc()
        bounds = [(None, None)] * k + [(0, None)] * (2 * n)
        res = optimize.linprog(c, A_eq=A, b_eq=y, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"quantile regression LP failed: {res.message}")
        beta = res.x[:k]
        r = y - X @ beta
        # Hall-Sheather bandwidth (in probability), then the sparsity s = 1/f(F^-1(tau))
        z = stats.norm.ppf(tau)
        h = n ** (-1 / 3) * stats.norm.ppf(0.975) ** (2 / 3) * (1.5 * stats.norm.pdf(z) ** 2 / (2 * z ** 2 + 1)) ** (1 / 3)
        h = min(h, tau - 1e-3, 1 - tau - 1e-3)
        XtX_inv = np.linalg.pinv(X.T @ X)
        if self.spec.cov == "nonrobust":
            s = (np.quantile(r, tau + h) - np.quantile(r, tau - h)) / (2 * h)
            V = tau * (1 - tau) * s ** 2 * XtX_inv
        else:
            kappa = min(np.std(r), stats.iqr(r) / 1.34) or np.std(r)
            cn = kappa * (stats.norm.ppf(tau + h) - stats.norm.ppf(tau - h))
            fhat = (np.abs(r) < cn) / (2 * cn)
            bread = np.linalg.pinv((X * fhat[:, None]).T @ X)
            psi = tau - (r < 0).astype(float)
            V = inf.sandwich(X, psi, bread, self.spec.cov, self.spec.hac_lags)
        obj = float(np.sum(ww * r * (tau - (r < 0))))
        r0 = y - np.quantile(y, tau)
        obj0 = float(np.sum(ww * r0 * (tau - (r0 < 0))))
        st = {"tau": tau, "pseudo_r2": 1 - obj / obj0 if obj0 > 0 else np.nan, "check_loss": obj}
        return Estimate(beta=beta, cov=V, df_resid=max(n - k, 1), stats=st)

    def residual_scale(self, e, w):
        return float(np.median(np.abs(e - np.median(e))) / 0.6745) or super().residual_scale(e, w)


class HockeyStickRegression(LinearRegression):
    """Piecewise linear (continuous) in ONE regressor, ``knot_var``: adds a hinge
    ``max(x - knot, 0)`` per knot. Knots fixed (``spec.knots``, in the variable's prepared
    units) or estimated by grid search of the SSR over ``knot_range`` quantiles; the
    profile is kept (``extra['profile']``) and gives a likelihood-ratio interval for the
    knot. Coefficient errors are CONDITIONAL on the knot, and ``F_vs_linear``'s p-value
    is naive (a knot only exists under the alternative - Davies' problem): treat it as a
    descriptive statistic."""
    method = "hockey"

    def _knot_var(self, regressors) -> str:
        return self.spec.knot_var or regressors[0]

    def features(self, scaled, regressors):
        F = super().features(scaled, regressors)
        knots = getattr(self, "_knots", None)
        if knots is not None:
            x = scaled[self._knot_var(regressors)]
            for i, kn in enumerate(knots):
                F[f"hinge{i}"] = np.maximum(x - kn, 0.0)
        return F

    def _ssr(self, X, y, w, knots, x):
        H = np.column_stack([X] + [np.maximum(x - kn, 0.0) for kn in knots])
        Hw, yw = _weighted(H, y, w)
        b = np.linalg.lstsq(Hw, yw, rcond=None)[0]
        e = yw - Hw @ b
        return float(e @ e)

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        sp = self.spec
        kv = self._knot_var(regressors)
        x = X[:, names.index(kv)]
        grid = np.quantile(x, np.linspace(sp.knot_range[0], sp.knot_range[1], sp.knot_grid))
        extra = {}
        if sp.knots is not None:
            knots = list(self.prep.transform(pd.DataFrame({kv: list(sp.knots)}))[kv]) if kv in self.prep.scaling_ else list(sp.knots)
        else:
            knots = []
            for _ in range(sp.n_knots):
                ssr = np.array([self._ssr(X, y, w, knots + [g], x) for g in grid])
                knots.append(float(grid[int(np.argmin(ssr))]))
                if len(knots) == 1:
                    extra["profile"] = pd.DataFrame({"knot": self.prep.inverse(grid, kv), "ssr": ssr},
                                                    index=[f"g{i}" for i in range(len(grid))])
                    profile_ssr = ssr
            knots.sort()
        self._knots = knots
        H = np.column_stack([X] + [np.maximum(x - kn, 0.0) for kn in knots])
        est = ols(H, y, w, sp.cov, sp.hac_lags)
        terms = names + [f"hinge{i}" for i in range(len(knots))]
        lin = ols(X, y, w, "nonrobust", None)
        Xw, yw = _weighted(X, y, w)
        ssr_lin = float(np.sum((yw - Xw @ lin.beta) ** 2))
        Hw, _ = _weighted(H, y, w)
        ssr_h = float(np.sum((yw - Hw @ est.beta) ** 2))
        q, df = len(knots), est.df_resid
        F = ((ssr_lin - ssr_h) / q) / (ssr_h / df) if ssr_h > 0 else np.nan
        bx = est.beta[names.index(kv)]
        st = dict(est.stats)
        st.update(F_vs_linear=F, F_vs_linear_p_naive=float(stats.f.sf(F, q, df)),
                  slope_left=bx, slope_right=bx + est.beta[len(names)])
        knots_raw = self.prep.inverse(np.array(knots), kv)
        for i, kn in enumerate(knots_raw):
            st[f"knot{i}" if i else "knot"] = float(kn)
        if len(knots) == 1 and "profile" in extra:
            crit = ssr_h * (1 + stats.f.ppf(0.95, 1, df) / df)
            inside = extra["profile"]["knot"].to_numpy()[profile_ssr <= crit]
            st["knot_ci_low"], st["knot_ci_high"] = float(inside.min()), float(inside.max())
        extra["knots"] = pd.Series({f"k{i}": float(k) for i, k in enumerate(knots)})  # model units
        extra["terms"] = terms
        est.stats, est.extra = st, extra
        return est

    def load_extra(self, extra):
        if "knots" in extra:
            self._knots = list(extra["knots"]["value"].to_numpy())


class TotalLeastSquares(LinearRegression):
    """Errors-in-variables. One regressor: Deming regression with ``variance_ratio`` =
    var(error in y) / var(error in x) (1 = orthogonal). Several: orthogonal TLS from the
    smallest singular vector of the centred [X y]. ``fitted`` is the vertical fit, so the
    residual is comparable to OLS's. Errors: delete-block jackknife (20 blocks,
    contiguous - keeps short-range dependence inside a block)."""
    method = "tls"

    def _beta(self, X, y, names, w):
        ww = np.ones(len(y)) if w is None else w / w.mean()
        keep = [j for j, nme in enumerate(names) if nme != CONST]
        Z = X[:, keep]
        mz, my = np.average(Z, axis=0, weights=ww), np.average(y, weights=ww)
        Zc, yc = Z - mz, y - my
        if len(keep) == 1:
            d = self.spec.variance_ratio
            sxx = np.average(Zc[:, 0] ** 2, weights=ww)
            syy = np.average(yc ** 2, weights=ww)
            sxy = np.average(Zc[:, 0] * yc, weights=ww)
            b = np.array([(syy - d * sxx + np.sqrt((syy - d * sxx) ** 2 + 4 * d * sxy ** 2)) / (2 * sxy)])
        else:
            if self.spec.variance_ratio != 1.0:
                raise ValueError("multivariate TLS supports variance_ratio=1 (orthogonal) only")
            M = np.column_stack([Zc, yc]) * np.sqrt(ww)[:, None]
            v = np.linalg.svd(M, full_matrices=False)[2][-1]
            b = -v[:-1] / v[-1]
        beta = np.zeros(len(names))
        beta[keep] = b
        if CONST in names:
            beta[names.index(CONST)] = my - mz @ b
        return beta

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        beta = self._beta(X, y, names, w)
        n, G = len(y), 20
        edges = np.linspace(0, n, G + 1).astype(int)
        reps = []
        for g in range(G):
            m = np.ones(n, dtype=bool)
            m[edges[g]:edges[g + 1]] = False
            reps.append(self._beta(X[m], y[m], names, None if w is None else w[m]))
        reps = np.array(reps)
        V = (G - 1) / G * (reps - reps.mean(axis=0)).T @ (reps - reps.mean(axis=0))
        return Estimate(beta=beta, cov=V, df_resid=G - 1, stats={"variance_ratio": self.spec.variance_ratio})


# --------------------------------------------------------------------------- binary
class LogitRegression(Regression):
    """Binary target (``y > threshold`` if a threshold is set, else y must be 0/1), by
    Newton-IRLS. ``fitted`` = probability, ``residual`` = y - p, ``resid_z`` = Pearson
    residual. Stats: log-likelihood, McFadden pseudo R^2, LR test vs the constant model,
    AIC/BIC, accuracy at 0.5, Brier score, AUC. A coefficient that diverges (perfect
    separation) is flagged ``separation=1``."""
    method = "logit"
    binary = True

    def transform_target(self, y):
        if self.spec.threshold is not None:
            return (y > self.spec.threshold).astype(float).where(y.notna())
        return y

    def link(self, eta):
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))

    def _info(self, eta):
        """(probability, score factor dpsi/(p(1-p)) ... ) per link: returns p, weight for
        the expected information, and the score multiplier on (y - p)."""
        p = self.link(eta)
        v = np.clip(p * (1 - p), 1e-12, None)
        return p, v, np.ones_like(p)

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("binary regression needs a 0/1 target: set spec.threshold")
        ww = np.ones(len(y)) if w is None else w / w.mean()
        beta = np.zeros(X.shape[1])
        separated = False
        for _ in range(100):
            eta = X @ beta
            p, info_w, mult = self._info(eta)
            score = X.T @ (ww * mult * (y - p))
            H = (X * (ww * info_w)[:, None]).T @ X
            step = np.linalg.lstsq(H, score, rcond=None)[0]
            beta = beta + step
            if np.max(np.abs(beta)) > 50:
                separated = True
                break
            if np.max(np.abs(step)) < 1e-10:
                break
        eta = X @ beta
        p, info_w, mult = self._info(eta)
        bread = np.linalg.pinv((X * (ww * info_w)[:, None]).T @ X)
        if self.spec.cov == "nonrobust":
            V = bread
        else:
            V = inf.sandwich(X, ww * mult * (y - p), bread, self.spec.cov, self.spec.hac_lags)
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        ll = float(np.sum(ww * (y * np.log(pc) + (1 - y) * np.log(1 - pc))))
        ybar = np.clip(np.average(y, weights=ww), 1e-12, 1 - 1e-12)
        ll0 = float(np.sum(ww * (y * np.log(ybar) + (1 - y) * np.log(1 - ybar))))
        k, n = X.shape[1], len(y)
        lr = 2 * (ll - ll0)
        st = {"loglik": ll, "loglik_null": ll0, "pseudo_r2": 1 - ll / ll0 if ll0 else np.nan, "lr": lr,
              "lr_p": float(stats.chi2.sf(lr, max(k - 1, 1))), "aic": -2 * ll + 2 * k, "bic": -2 * ll + np.log(n) * k,
              "accuracy": float(np.mean((p > 0.5) == (y > 0.5))), "brier": float(np.mean((p - y) ** 2)),
              "auc": auc(y, p), "base_rate": float(np.mean(y)), "separation": float(separated)}
        return Estimate(beta=beta, cov=V, df_resid=None, stats=st)

    def fit_stats(self, y, fitted, w, k, X):
        n = len(y)
        return {"n": n, "n_eff": effective_n(w, n), "k": k}

    def residual_scale(self, e, w):
        return 1.0

    def standardised_residual(self, resid, fitted):
        return resid / np.sqrt(np.clip(fitted * (1 - fitted), 1e-12, None))


class ProbitRegression(LogitRegression):
    """As ``LogitRegression`` with the normal CDF link (expected information)."""
    method = "probit"

    def link(self, eta):
        return stats.norm.cdf(eta)

    def _info(self, eta):
        p = np.clip(stats.norm.cdf(eta), 1e-12, 1 - 1e-12)
        phi = stats.norm.pdf(eta)
        v = phi ** 2 / (p * (1 - p))
        return p, v, phi / (p * (1 - p))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney rank form)."""
    pos, neg = y > 0.5, y <= 0.5
    if not pos.any() or not neg.any():
        return np.nan
    ranks = stats.rankdata(p)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


# --------------------------------------------------------------------------- kalman
def kalman_filter(X, y, beta0, P0, q, r, *, update=True):
    """Random-walk coefficients: beta_t = beta_{t-1} + N(0, q I); y_t = x_t beta_t + N(0, r).
    Returns one-step predictions, innovations, their variances, filtered betas and the
    final (beta, P). A NaN y skips the update (prediction only)."""
    n, k = X.shape
    b, P = beta0.astype(float).copy(), P0.astype(float).copy()
    pred, innov, fvar, betas = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full((n, k), np.nan)
    Q = q * np.eye(k)
    for t in range(n):
        x = X[t]
        if update:
            P = P + Q
        if not np.all(np.isfinite(x)):
            betas[t] = b
            continue
        pred[t] = x @ b
        F = x @ P @ x + r
        fvar[t] = F
        if update and np.isfinite(y[t]):
            v = y[t] - pred[t]
            innov[t] = v
            K = P @ x / F
            b = b + K * v
            P = P - np.outer(K, x @ P)
        elif np.isfinite(y[t]):
            innov[t] = y[t] - pred[t]
        betas[t] = b
    return pred, innov, fvar, betas, b, P


class KalmanRegression(Regression):
    """Time-varying coefficients (random walk), Kalman-filtered. ``fit`` estimates the
    noise variances (state ``q``, observation ``r``) by maximum likelihood of the one-step
    prediction errors on the fit sample (or fixes ``q = delta/(1-delta) r`` with
    ``kalman_delta``), runs the filter through it and keeps the final state. ``predict``
    continues filtering from that state (``kalman_update=False`` freezes the last beta -
    use that for data at another granularity). ``fitted`` is the ONE-STEP-AHEAD prediction
    (beta_{t|t-1}), so ``residual`` is an honest innovation and ``resid_z`` the
    standardised innovation; ``beta:<term>`` are the filtered coefficients.
    Regressors are standardised internally on the fit sample (isotropic state noise)."""
    method = "kalman"

    def estimate(self, X, y, w, names, scaled, regressors) -> Estimate:
        n, k = X.shape
        keep = [j for j, nme in enumerate(names) if nme != CONST]
        mu, sd = X[:, keep].mean(axis=0), X[:, keep].std(axis=0)
        sd[sd == 0] = 1.0
        Z = X.copy()
        Z[:, keep] = (X[:, keep] - mu) / sd
        burn = min(max(30, 5 * k), n // 3)
        init = ols(Z[:burn], y[:burn], None, "nonrobust", None)
        b0, P0 = init.beta, init.cov * 10.0
        var_y = float(np.var(y)) or 1.0

        def nll(theta):
            q, r = np.exp(theta)
            _, v, F, *_ = kalman_filter(Z[burn:], y[burn:], b0, P0, q, r)
            ok = np.isfinite(v) & (F > 0)
            return 0.5 * np.sum(np.log(F[ok]) + v[ok] ** 2 / F[ok])

        if self.spec.kalman_delta is not None:
            ratio = self.spec.kalman_delta / (1 - self.spec.kalman_delta)
            res = optimize.minimize_scalar(lambda lr: nll(np.array([np.log(ratio) + lr, lr])),
                                           bounds=(np.log(var_y) - 15, np.log(var_y) + 3), method="bounded")
            r = float(np.exp(res.x))
            q = ratio * r
        else:
            warm = getattr(self, "_warm_qr", None)
            start = np.log(warm) if warm is not None else np.log([1e-4 * var_y, 0.5 * var_y])
            res = optimize.minimize(nll, start, method="Nelder-Mead",
                                    options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
            q, r = np.exp(res.x)
        pred, innov, fvar, betas, bT, PT = kalman_filter(Z, y, b0, P0, q, r)
        self._kf = dict(mu=mu, sd=sd, keep=keep, q=float(q), r=float(r), bT=bT, PT=PT)
        self._sample_path = (pred, innov, fvar, betas)
        ll = -0.5 * np.nansum(np.log(2 * np.pi * fvar[burn:]) + innov[burn:] ** 2 / fvar[burn:])
        # report the final coefficients in the regressors' (scaled) units
        beta = bT.copy()
        beta[keep] = bT[keep] / sd
        if CONST in names:
            beta[names.index(CONST)] = bT[names.index(CONST)] - (bT[keep] / sd) @ mu
        st = {"q": float(q), "r": float(r), "signal_to_noise": float(q / r), "loglik": float(ll),
              "warm_start": float(getattr(self, "_warm_qr", None) is not None),
              "mle_evaluations": float(getattr(res, "nfev", np.nan))}
        extra = {"kalman": pd.Series({"q": float(q), "r": float(r)}),
                 "kalman_scale": pd.DataFrame({"mu": mu, "sd": sd}, index=[names[j] for j in keep]),
                 "state_beta": pd.Series(bT, index=names), "state_P": pd.DataFrame(PT, index=names, columns=names)}
        return Estimate(beta=beta, cov=None, df_resid=None, stats=st, extra=extra)

    def warm_start_from(self, previous) -> None:
        """Start the next fit's (q, r) search from ``previous``'s estimate (a walk-forward or
        an appended weekly refit): the maximum likelihood is the same, found in a fraction
        of the evaluations. Differences against a cold search stay at the optimiser's
        tolerance."""
        if isinstance(previous, KalmanRegression) and previous.is_fitted:
            st = previous.fitted_.stats
            if np.isfinite(st.get("q", np.nan)) and np.isfinite(st.get("r", np.nan)) and st["q"] > 0 and st["r"] > 0:
                self._warm_qr = np.array([st["q"], st["r"]])

    def _Z(self, F: pd.DataFrame) -> np.ndarray:
        kf = self._kf
        Z = F.to_numpy(dtype="float64").copy()
        Z[:, kf["keep"]] = (Z[:, kf["keep"]] - kf["mu"]) / kf["sd"]
        return Z

    def fit(self, prepared, as_of=None):
        super().fit(prepared, as_of)
        pred, innov, fvar, betas = self._sample_path
        f = self.fitted_
        z = innov / np.sqrt(fvar)
        f.resid_std = float(np.nanstd(innov))
        f.stats.update(inf.residual_stats(innov[np.isfinite(innov)]))
        # R^2 of the ONE-STEP-AHEAD predictions (the base class's would apply the final
        # beta to the whole sample - hindsight)
        y = self.transform_target(self.prep.transform(f.sample)[f.target]).to_numpy(dtype="float64")
        ok = np.isfinite(innov) & np.isfinite(y)
        f.stats["r2"] = float(1 - np.sum(innov[ok] ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))
        f.stats["adj_r2"] = f.stats["corr"] = np.nan
        f.stats["innovation_z_std"] = float(np.nanstd(z))
        return self

    def predict(self, prepared=None, *, start=None, end=None, contributions=False) -> pd.DataFrame:
        self.check_fitted()
        f, kf = self.fitted_, self._kf
        frame = f.sample if prepared is None else self.predict_window(prepared, start, end)
        scaled = self.prep.transform(frame)
        Fr = self.features(scaled, f.regressors)[f.terms]
        y = scaled[f.target].to_numpy(dtype="float64") if f.target in scaled else np.full(len(frame), np.nan)
        insample = self.in_sample(frame.index)
        k = len(f.terms)
        pred, fvar = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        betas = np.full((len(frame), k), np.nan)
        if insample.any() and getattr(self, "_sample_path", None) is not None:
            # rows up to the fit date: the fit's own filter path (rows outside the fit sample stay NaN)
            s_pred, _, s_fvar, s_betas = self._sample_path
            pos = pd.Index(f.sample.index).get_indexer(frame.index[insample]) if f.sample is not None else \
                np.full(int(insample.sum()), -1)
            hit = pos >= 0
            rows = np.flatnonzero(insample)[hit]
            pred[rows], fvar[rows], betas[rows] = s_pred[pos[hit]], s_fvar[pos[hit]], s_betas[pos[hit]]
        if (~insample).any():
            later = ~insample
            p_, _, fv_, b_, *_ = kalman_filter(self._Z(Fr[later]), y[later], kf["bT"], kf["PT"], kf["q"], kf["r"],
                                               update=self.spec.kalman_update)
            pred[later], fvar[later], betas[later] = p_, fv_, b_
        out = pd.DataFrame(index=frame.index)
        out["y"] = self.prep.inverse(y, f.target)
        out["fitted"] = self.prep.inverse(pred, f.target)
        out["residual"] = self.prep.inverse(y - pred, f.target, shift=False)
        out["resid_z"] = (y - pred) / np.sqrt(fvar)
        out["in_sample"] = insample
        # filtered coefficients back in the regressors' (scaled) units
        keep, sd, mu = kf["keep"], np.asarray(kf["sd"]), np.asarray(kf["mu"])
        slopes = betas[:, keep] / sd
        for j, t in enumerate(f.terms):
            if j in keep:
                out[f"beta:{t}"] = slopes[:, keep.index(j)]
            else:  # the intercept, back from the standardised regressors
                out[f"beta:{t}"] = betas[:, j] - slopes @ mu
        return out

    def load_extra(self, extra):
        names = list(extra["state_beta"].index)
        sc = extra["kalman_scale"]
        self._kf = dict(mu=sc["mu"].to_numpy(), sd=sc["sd"].to_numpy(),
                        keep=[names.index(c) for c in sc.index], q=float(extra["kalman"].loc["q", "value"]),
                        r=float(extra["kalman"].loc["r", "value"]), bT=extra["state_beta"]["value"].to_numpy(),
                        PT=extra["state_P"].reindex(index=names, columns=names).to_numpy())


REGRESSION_CLASSES: dict[str, type[Regression]] = {
    "ols": LinearRegression, "stepwise": StepwiseRegression, "ridge": RidgeRegression,
    "lasso": LassoRegression, "elasticnet": ElasticNetRegression, "huber": HuberRegression,
    "quantile": QuantileRegression, "hockey": HockeyStickRegression, "tls": TotalLeastSquares,
    "logit": LogitRegression, "probit": ProbitRegression, "kalman": KalmanRegression,
}


def make_regression(spec: RegressionSpec | str | None = None, **overrides) -> Regression:
    """The right class for ``spec.method`` (a spec, a ``REGRESSION_MODELS`` name, or None
    = ``ols``), with field ``overrides`` (``y=``, ``x=``, ``alpha=`` ...)."""
    spec = get_regression_spec(spec, **overrides)
    return REGRESSION_CLASSES[spec.method](spec)
