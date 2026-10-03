"""Principal components of any set of series: prepare -> fit (point in time) -> predict.

    pca = make_pca("curve_changes", columns=tuple(f"bond:US_BOND_{t}y" for t in (2, 5, 10, 30)))
    data = pca.prepare(panel)               # stateless prep (diff, x100 = bp)
    pca.fit(data, as_of="2026-06-30")       # loadings, eigenvalues, mean/scale frozen
    out = pca.predict(data, start="2026-06-30")
    pick(out, "score")                      # factor scores per row
    pick(out, "residual")                   # x - reconstruction from n_components PCs
    pca.loadings, pca.explained             # tables

Classes (``PCA_CLASSES``, by ``PCASpec.method``):

* ``PCA`` - complete rows only (listwise deletion in the fit sample).
* ``WeightedPCA`` - exponential TIME weights (``halflife``) and/or VARIABLE weights
  (``var_weights``: how much each series counts in the decomposition).
* ``MissingDataPCA`` - fits on rows with gaps (markets closed on different days):
  ``missing="em"`` (default; maximum-likelihood probabilistic PCA, Tipping & Bishop 1999) or
  ``"pairwise"`` (covariance from pairwise-complete rows, projected to the nearest PSD).

Every class projects a row with gaps the same way: its scores are the least-squares fit of
the OBSERVED entries on their loadings (needs at least ``n_components`` observed), so a
cross-market model keeps producing factors while one market is shut - intraday too.

Signs are fixed (loadings sum positive, ``spec.sign``) so a refit doesn't flip a factor;
``align_to(other)`` matches signs (and optionally order) to a previous fit, which
``infra.models.walk_forward`` does at every refit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from infra.models.stats.common import PARAM_COLUMNS, StatModel, decay_weights, effective_n, params_frame, section
from infra.models.stats.config import PCASpec, get_pca_spec


@dataclass
class PCAFit:
    as_of: pd.Timestamp
    columns: list[str]
    mean: np.ndarray             # per column, after prep (model units)
    scale: np.ndarray            # per column: std if spec.scale else 1
    var_weight: np.ndarray       # per column
    eigenvalues: np.ndarray      # all p, descending (of the weighted, scaled covariance)
    loadings: np.ndarray         # p x k, orthonormal (in the weighted, scaled space)
    resid_std: np.ndarray        # per column, fit sample (model units)
    stats: dict
    sample: pd.DataFrame | None
    noise_var: float = 0.0       # per-entry noise (scaled space) for projecting rows with gaps


def project_observed(Z: np.ndarray, L: np.ndarray, prior_var: np.ndarray, noise_var) -> np.ndarray:
    """Scores of centred rows ``Z`` (n x p, NaN = not observed) on loadings ``L`` (p x k,
    or n x p x k per row). Complete rows: the classical projection ``Z L``. Rows with gaps:
    the posterior mean of the scores given the OBSERVED entries, with prior f ~ N(0,
    diag(prior_var)) (each PC's own variance; k, or n x k per row) and noise ``noise_var``
    per entry (a scalar, or n per row) - every row is projected on its own, never with
    anything averaged over the batch, so a row's output can't depend on which other rows
    are predicted with it:
    ``(L_o' L_o + noise_var diag(1/prior_var))^-1 L_o' z_o``. Plain least squares on the
    observed entries is ill-conditioned when they barely identify a factor - found
    2026-10-03: on US-only days, a US/DE/UK PCA's cross-country factor came out at ~100
    standard deviations and a regime HMM made a regime of those days. The prior shrinks
    a badly identified factor toward 0 instead. Rows with nothing observed -> NaN."""
    n, p = Z.shape
    per_row = L.ndim == 3
    k = L.shape[-1]
    out = np.full((n, k), np.nan)
    obs = np.isfinite(Z)
    full = obs.all(axis=1)
    out[full] = np.einsum("ni,nik->nk", Z[full], L[full]) if per_row else Z[full] @ L
    prior = np.broadcast_to(np.asarray(prior_var, dtype="float64"), (n, k))
    noise = np.broadcast_to(np.asarray(noise_var, dtype="float64"), (n,))
    for i in np.flatnonzero(~full & obs.any(axis=1)):
        o = obs[i]
        Lo = (L[i] if per_row else L)[o]
        ridge = noise[i] / np.clip(prior[i], 1e-300, None)
        out[i] = np.linalg.solve(Lo.T @ Lo + np.diag(ridge), Lo.T @ Z[i, o])
    return out


def mp_upper_edge(p: int, n_eff: float, sigma2: float = 1.0) -> float:
    """Marchenko-Pastur upper edge: the largest eigenvalue pure noise of variance
    ``sigma2`` reaches with p variables and n observations. Eigenvalues above it carry
    structure; below it they are indistinguishable from noise."""
    return sigma2 * (1.0 + np.sqrt(p / max(n_eff, 1.0))) ** 2


MP_MIN_VARIABLES = 20  # below this there is no noise "bulk" to estimate the noise level from


def noise_edge(eigenvalues: np.ndarray, n_eff: float, k: int, max_iter: int = 50) -> tuple[float, float]:
    """(noise variance, MP upper edge). The textbook noise variance, trace/p, counts the
    signal as noise: on a US curve PCA PC1 carries ~90% and lifts the edge above the slope
    PC (checked 2026-10-03, 6 tenors). So:

    * p >= ``MP_MIN_VARIABLES``: the noise variance is the mean of the eigenvalues BELOW
      the edge, iterated to a fixed point (Laloux et al. 2000);
    * fewer variables: Marchenko-Pastur is asymptotic and there is no bulk (the iteration
      slides down to the smallest eigenvalue), so the noise is the mean of the DISCARDED
      eigenvalues (k+1..p) - probabilistic PCA's maximum-likelihood noise. Use
      ``diagnostics.parallel_analysis`` as the small-p test of how many PCs to keep."""
    vals = np.asarray(eigenvalues, dtype="float64")
    p = len(vals)
    if p < MP_MIN_VARIABLES:
        sigma2 = float(vals[k:].mean()) if p > k else float(vals[-1])
        return sigma2, mp_upper_edge(p, n_eff, sigma2)
    sigma2 = float(vals.mean())
    for _ in range(max_iter):
        below = vals[vals <= mp_upper_edge(p, n_eff, sigma2)]
        new = float(below.mean()) if len(below) else sigma2
        if abs(new - sigma2) <= 1e-12 * max(sigma2, 1e-300):
            break
        sigma2 = new
    return sigma2, mp_upper_edge(p, n_eff, sigma2)


def nearest_psd(C: np.ndarray) -> np.ndarray:
    v, V = np.linalg.eigh((C + C.T) / 2.0)
    return (V * np.clip(v, 0.0, None)) @ V.T


def orient(L: np.ndarray, how: str) -> np.ndarray:
    L = L.copy()
    for j in range(L.shape[1]):
        s = L[:, j].sum() if how == "sum" else L[np.argmax(np.abs(L[:, j])), j]
        if how == "sum" and abs(s) < 1e-8:
            s = L[np.argmax(np.abs(L[:, j])), j]
        if s < 0:
            L[:, j] *= -1.0
    return L


class PCA(StatModel):
    """See the module docstring."""

    method = "pca"

    def __init__(self, spec: PCASpec | str | None = None, **overrides):
        spec = get_pca_spec(spec, **overrides)
        if spec.method != self.method:
            spec = replace(spec, method=self.method)
        super().__init__(spec)

    def input_columns(self, raw):
        return list(self.spec.columns) if self.spec.columns is not None else list(raw.columns)

    # ------------------------------------------------------------------ fit
    def _var_weights(self, cols) -> np.ndarray:
        vw = dict(self.spec.var_weights)
        return np.array([float(vw.get(c, 1.0)) for c in cols])

    def _moments(self, X: np.ndarray, w: np.ndarray):
        """Weighted mean and covariance of complete rows."""
        mu = np.average(X, axis=0, weights=w)
        Xc = X - mu
        C = (Xc * w[:, None]).T @ Xc / w.sum()
        return mu, C

    def _fit_covariance(self, sample: pd.DataFrame, w: np.ndarray):
        """(mean, covariance, rows used) - complete rows only. Subclasses override."""
        ok = sample.notna().all(axis=1).to_numpy()
        X = sample.to_numpy(dtype="float64")[ok]
        mu, C = self._moments(X, w[ok])
        return mu, C, ok

    def fit(self, prepared: pd.DataFrame, as_of=None) -> "PCA":
        sample, as_of = self.fit_sample(prepared, as_of)
        sample = sample.dropna(how="all")
        cols = list(sample.columns)
        if self.spec.min_coverage > 0:
            cov_share = sample.notna().mean()
            cols = [c for c in cols if cov_share[c] >= self.spec.min_coverage]
            sample = sample[cols]
        k = min(self.spec.n_components, len(cols))
        w_all = decay_weights(len(sample), self.spec.halflife)
        w_all = np.ones(len(sample)) if w_all is None else w_all
        self.prep.fit(sample, w_all)
        scaled = self.prep.transform(sample)
        mu, C, used = self._fit_covariance(scaled, w_all)
        if used.sum() < max(self.spec.min_obs, k + 1):
            raise ValueError(f"{type(self).__name__}: {int(used.sum())} usable rows up to "
                             f"{pd.Timestamp(as_of).date()} (min_obs {self.spec.min_obs})")
        sd = np.sqrt(np.clip(np.diag(C), 1e-300, None)) if self.spec.scale else np.ones(len(cols))
        vw = self._var_weights(cols)
        D = np.sqrt(vw) / sd
        Cs = C * np.outer(D, D)
        vals, vecs = np.linalg.eigh(Cs)
        order = np.argsort(vals)[::-1]
        vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
        L = orient(vecs[:, :k], self.spec.sign)
        w_used = w_all[used]
        n_eff = effective_n(w_used if self.spec.halflife else None, int(used.sum()))
        total = vals.sum()
        sigma2, edge = noise_edge(vals, n_eff, k)
        self.fitted_ = PCAFit(as_of=pd.Timestamp(as_of), columns=cols, mean=mu, scale=sd, var_weight=vw,
                              eigenvalues=vals, loadings=L, resid_std=np.full(len(cols), np.nan), stats={},
                              sample=sample, noise_var=float(vals[k:].mean()) if len(vals) > k else 0.0)
        rec = self._project(scaled.to_numpy(dtype="float64"))[2]
        resid = scaled.to_numpy(dtype="float64") - rec
        self.fitted_.resid_std = np.sqrt(np.nanmean(resid ** 2, axis=0))
        self.fitted_.stats = {
            "n": int(used.sum()), "n_rows": len(sample), "n_eff": n_eff, "p": len(cols), "k": k,
            "explained_k": float(vals[:k].sum() / total) if total > 0 else np.nan,
            "mp_edge": edge, "noise_var": sigma2, "n_above_mp": int(np.sum(vals > edge)),
            "gap_k": float(vals[k - 1] / vals[k]) if k < len(vals) and vals[k] > 0 else np.nan,
            "projection_noise_var": self.fitted_.noise_var,
        } | self._extra_stats()
        return self

    def _extra_stats(self) -> dict:
        return {}

    # ------------------------------------------------------------------ projection
    def _project(self, X: np.ndarray):
        """(scores n x k, observed count, reconstruction n x p in model units); rows with
        gaps are projected on their observed entries (``project_observed``)."""
        f = self.fitted_
        D = np.sqrt(f.var_weight) / f.scale
        Z = (X - f.mean) * D
        obs = np.isfinite(Z)
        k = f.loadings.shape[1]
        scores = project_observed(Z, f.loadings, f.eigenvalues[:k], f.noise_var)
        rec = scores @ f.loadings.T / D + f.mean
        return scores, obs.sum(axis=1), rec

    def predict(self, prepared: pd.DataFrame | None = None, *, start=None, end=None) -> pd.DataFrame:
        """Per row: ``score:PCi``, ``fitted:<col>`` (reconstruction from the k PCs),
        ``residual:<col>`` (x - fitted), ``resid_z:<col>`` (over the fit-sample residual
        std), ``n_obs`` (observed columns) and ``in_sample``. Values in each column's
        prepared units (stateful scaling undone)."""
        self.check_fitted()
        f = self.fitted_
        frame = f.sample if prepared is None else self.predict_window(prepared, start, end)
        if frame is None:
            raise ValueError("no data: this fit was rebuilt from params; pass prepared data")
        frame = frame.reindex(columns=f.columns)
        scaled = self.prep.transform(frame)
        X = scaled.to_numpy(dtype="float64")
        scores, n_obs, rec = self._project(X)
        out = pd.DataFrame(index=frame.index)
        for j in range(scores.shape[1]):
            out[f"score:PC{j + 1}"] = scores[:, j]
        resid = X - rec
        for i, c in enumerate(f.columns):
            out[f"fitted:{c}"] = self.prep.inverse(rec[:, i], c)
        for i, c in enumerate(f.columns):
            out[f"residual:{c}"] = self.prep.inverse(resid[:, i], c, shift=False)
        for i, c in enumerate(f.columns):
            out[f"resid_z:{c}"] = resid[:, i] / f.resid_std[i] if f.resid_std[i] > 0 else np.nan
        out["n_obs"] = n_obs
        out["in_sample"] = self.in_sample(frame.index)
        return out

    transform = predict  # sklearn's name for the same step

    # ------------------------------------------------------------------ tables
    @property
    def pc_names(self) -> list[str]:
        return [f"PC{j + 1}" for j in range(self.fitted_.loadings.shape[1])]

    @property
    def loadings(self) -> pd.DataFrame:
        """Unit-norm eigenvectors (in the weighted / scaled space), columns x PCs."""
        self.check_fitted()
        return pd.DataFrame(self.fitted_.loadings, index=self.fitted_.columns, columns=self.pc_names)

    @property
    def loadings_units(self) -> pd.DataFrame:
        """What a ONE-standard-deviation move of each PC does to each column, in the
        column's model units (bp for a ``diff, mult:100`` prep): the curve shape of each
        factor."""
        self.check_fitted()
        f = self.fitted_
        D = np.sqrt(f.var_weight) / f.scale
        k = f.loadings.shape[1]
        return pd.DataFrame(f.loadings * np.sqrt(f.eigenvalues[:k]) / D[:, None], index=f.columns,
                            columns=self.pc_names)

    @property
    def explained(self) -> pd.DataFrame:
        self.check_fitted()
        ev = self.fitted_.eigenvalues
        total = ev.sum()
        return pd.DataFrame({"eigenvalue": ev, "explained": ev / total, "cumulative": np.cumsum(ev) / total,
                             "above_mp": ev > self.fitted_.stats["mp_edge"]},
                            index=[f"PC{j + 1}" for j in range(len(ev))])

    def summary(self) -> str:
        self.check_fitted()
        s = self.fitted_.stats
        head = (f"{type(self).__name__} [{self.spec.name}] as_of={self.fitted_.as_of.date()} n={s['n']} "
                f"p={s['p']} k={s['k']} explained={s['explained_k']:.1%} above MP edge={s['n_above_mp']}")
        k = self.fitted_.loadings.shape[1]
        return f"{head}\n{self.explained.head(max(k, 5)).to_string(float_format=lambda v: f'{v:.4g}')}\n" \
               f"{self.loadings_units.to_string(float_format=lambda v: f'{v:.4g}')}"

    def align_to(self, reference: "PCA", *, permute: bool = False) -> "PCA":
        """Flip (and with ``permute``, reorder) this fit's PCs to match ``reference``'s
        loadings on the common columns: keeps factor identities stable across refits when
        signs or near-equal eigenvalues would otherwise swap them."""
        self.check_fitted()
        ref = reference.loadings
        mine = self.loadings
        common = [c for c in mine.index if c in ref.index]
        if not common:
            return self
        A, B = mine.loc[common].to_numpy(), ref.loc[common].to_numpy()
        k = min(A.shape[1], B.shape[1])
        L = self.fitted_.loadings.copy()
        order = list(range(L.shape[1]))
        if permute:
            M = np.abs(A[:, :k].T @ B[:, :k])
            rows, cols = linear_sum_assignment(-M)
            perm = [int(r) for _, r in sorted(zip(cols, rows))]
            order = perm + [j for j in range(L.shape[1]) if j not in perm]
            L = L[:, order]
            self.fitted_.eigenvalues[:len(order)] = self.fitted_.eigenvalues[order]
            A = A[:, order]
        for j in range(k):
            if A[:, j] @ B[:, j] < 0:
                L[:, j] *= -1.0
        self.fitted_.loadings = L
        return self

    # ------------------------------------------------------------------ params
    def params(self) -> pd.DataFrame:
        self.check_fitted()
        f = self.fitted_
        cols = pd.DataFrame({"mean": f.mean, "scale": f.scale, "var_weight": f.var_weight,
                             "resid_std": f.resid_std}, index=f.columns)
        ev = pd.DataFrame({"eigenvalue": f.eigenvalues}, index=[f"PC{j + 1}" for j in range(len(f.eigenvalues))])
        meta = pd.DataFrame([{"section": "meta", "row": "as_of", "col": f.as_of.isoformat(), "value": np.nan}])
        parts = [meta, params_frame("loading", self.loadings), params_frame("column", cols),
                 params_frame("eigen", ev), params_frame("stat", f.stats), self.prep.params()]
        return pd.concat([p for p in parts if not p.empty], ignore_index=True)[PARAM_COLUMNS]

    @classmethod
    def from_params(cls, params: pd.DataFrame, spec: PCASpec | str | None = None, **overrides) -> "PCA":
        model = make_pca(spec, **overrides) if cls is PCA else cls(spec, **overrides)
        load = section(params, "loading")
        cols = section(params, "column").reindex(load.index)
        ev = section(params, "eigen")["eigenvalue"]
        ev = ev.reindex(sorted(ev.index, key=lambda s: int(s[2:])))
        meta = params[(params["section"] == "meta") & (params["row"] == "as_of")]
        model.prep.load_params(params)
        model.fitted_ = PCAFit(
            as_of=pd.Timestamp(meta["col"].iloc[0]) if len(meta) else pd.NaT, columns=list(load.index),
            mean=cols["mean"].to_numpy(), scale=cols["scale"].to_numpy(), var_weight=cols["var_weight"].to_numpy(),
            eigenvalues=ev.to_numpy(), loadings=load[sorted(load.columns, key=lambda s: int(s[2:]))].to_numpy(),
            resid_std=cols["resid_std"].to_numpy(), stats=section(params, "stat")["value"].to_dict(), sample=None,
            noise_var=float(section(params, "stat")["value"].get("projection_noise_var", 0.0)))
        return model


class WeightedPCA(PCA):
    """Exponential time weights (``halflife``, rows: recent data counts more - a 'live'
    covariance) and/or variable weights (``var_weights``). Effective sample size (Kish)
    replaces n in the Marchenko-Pastur edge."""
    method = "weighted"

    def fit(self, prepared, as_of=None):
        if self.spec.halflife is None and not self.spec.var_weights:
            raise ValueError("WeightedPCA needs halflife and/or var_weights (else use PCA)")
        return super().fit(prepared, as_of)


def ppca_em(Y: np.ndarray, w: np.ndarray, k: int, *, max_iter: int = 500, tol: float = 1e-7,
            init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Maximum likelihood of the probabilistic PCA model ``y = mu + W z + e``,
    ``z ~ N(0, I_k)``, ``e ~ N(0, sigma2 I)`` on rows with gaps (Tipping & Bishop 1999),
    by EM. Per missingness pattern the E-step gives the missing entries' conditional mean
    AND covariance, and both enter the expected covariance - so imputed cells keep their
    idiosyncratic noise. (Plain "fill with the rank-k reconstruction and repeat" drops that
    term: the imputed cells look perfectly factor-driven and the explained share is
    overstated - found on the US/DE/UK curve 2026-10-03: 92% vs 86% complete-case, PC2
    turned into a spurious US-vs-UK factor, and it never converged.)
    Returns (mean, the model-implied covariance's unconstrained counterpart = the
    expected sample covariance, info)."""
    n, p = Y.shape
    obs = np.isfinite(Y)
    W_ = w / w.sum()
    mu = np.array([np.average(Y[obs[:, j], j], weights=w[obs[:, j]]) for j in range(p)])
    C = init if init is not None else np.diag(np.nanvar(Y, axis=0))
    patterns: dict[bytes, np.ndarray] = {}
    for i in range(n):
        patterns.setdefault(obs[i].tobytes(), []).append(i)
    groups = [(np.frombuffer(key, dtype=bool), np.array(rows)) for key, rows in patterns.items()]
    prev_ll, it, ll = -np.inf, 0, np.nan
    for it in range(1, max_iter + 1):
        vals, vecs = np.linalg.eigh((C + C.T) / 2)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        sigma2 = max(float(vals[k:].mean()) if p > k else 1e-12, 1e-12)
        Wm = vecs[:, :k] * np.sqrt(np.clip(vals[:k] - sigma2, 0.0, None))
        Sx, Sxx, ll = np.zeros(p), np.zeros((p, p)), 0.0
        for o, rows in groups:
            m = ~o
            Wo = Wm[o]
            M = Wo.T @ Wo + sigma2 * np.eye(k)
            Minv = np.linalg.inv(M)
            D = Y[np.ix_(rows, np.flatnonzero(o))] - mu[o]
            Ez = D @ Wo @ Minv
            Xh = np.empty((len(rows), p))
            Xh[:, o] = Y[np.ix_(rows, np.flatnonzero(o))]
            Xh[:, m] = mu[m] + Ez @ Wm[m].T
            wr = W_[rows]
            Sx += wr @ Xh
            Sxx += (Xh * wr[:, None]).T @ Xh
            if m.any():
                Cm = sigma2 * np.eye(int(m.sum())) + Wm[m] @ (sigma2 * Minv) @ Wm[m].T
                Sxx[np.ix_(m, m)] += wr.sum() * Cm
            # observed-data log-likelihood of this pattern
            Co = Wo @ Wo.T + sigma2 * np.eye(int(o.sum()))
            sign, logdet = np.linalg.slogdet(Co)
            q = np.einsum("ij,jk,ik->i", D, np.linalg.inv(Co), D)
            ll += float(-0.5 * wr @ (logdet + q + o.sum() * np.log(2 * np.pi)))
        mu = Sx
        C = Sxx - np.outer(mu, mu)
        if abs(ll - prev_ll) < tol * (1.0 + abs(ll)):
            break
        prev_ll = ll
    return mu, C, {"iterations": it, "em_loglik": ll, "em_converged": float(it < max_iter),
                   "missing_share": float(1 - obs.mean())}


class MissingDataPCA(PCA):
    """Fits on incomplete rows (markets closed on different days).

    * ``missing="em"`` (default): maximum-likelihood probabilistic PCA by EM (``ppca_em``);
      the eigenvectors come from the expected covariance. Rows observed in only some
      markets still inform those markets' variances and their links to the others.
    * ``"pairwise"``: each covariance entry from the rows where both columns are observed,
      clipped to the nearest PSD matrix. Fast; each entry comes from a different sample,
      which can distort the matrix when overlaps are small.

    Never zero-fill a closed market's change, and never difference across a forward-filled
    level: both invent a move of 0 followed by a catch-up move. Differencing a level with a
    gap leaves the gap day AND the reopening day missing, which is what this class is for.
    Scaling (``scale=True``) and variable weights are applied with pairwise variances
    before the EM (fixed during it)."""
    method = "missing"

    def _pairwise(self, X, obs, w):
        p = X.shape[1]
        mu = np.array([np.average(X[obs[:, j], j], weights=w[obs[:, j]]) for j in range(p)])
        Xc = np.where(obs, X - mu, 0.0)
        Wm = obs * w[:, None]
        C = (Xc * Wm).T @ (Xc * obs) / np.clip(Wm.T @ obs.astype(float), 1e-300, None)
        return mu, nearest_psd(C)

    def _fit_covariance(self, sample, w):
        X = sample.to_numpy(dtype="float64")
        obs = np.isfinite(X)
        rows = obs.any(axis=1)
        X, obs, w = X[rows], obs[rows], w[rows]
        used = np.zeros(len(sample), dtype=bool)
        used[np.flatnonzero(rows)] = True
        how = "em" if self.spec.missing == "drop" else self.spec.missing
        mu, C = self._pairwise(X, obs, w)
        if how == "pairwise":
            self._em_info = {"iterations": 0, "missing_share": float(1 - obs.mean())}
            return mu, C, used
        if how != "em":
            raise ValueError(f"missing {self.spec.missing!r} (em | pairwise)")
        sd = np.sqrt(np.clip(np.diag(C), 1e-300, None)) if self.spec.scale else np.ones(X.shape[1])
        D = np.sqrt(self._var_weights(list(sample.columns))) / sd
        k = min(self.spec.n_components, X.shape[1])
        muY, CY, info = ppca_em(X * D, w, k, max_iter=self.spec.em_max_iter, tol=self.spec.em_tol,
                                init=C * np.outer(D, D))
        self._em_info = info
        return muY / D, CY / np.outer(D, D), used

    def _extra_stats(self):
        return dict(getattr(self, "_em_info", {}))


PCA_CLASSES: dict[str, type[PCA]] = {"pca": PCA, "weighted": WeightedPCA, "missing": MissingDataPCA}


def make_pca(spec: PCASpec | str | None = None, **overrides) -> PCA:
    """The right class for ``spec.method`` (a spec, a ``PCA_MODELS`` name or a method)."""
    spec = get_pca_spec(spec, **overrides)
    return PCA_CLASSES[spec.method](spec)
