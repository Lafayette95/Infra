"""PCA of K series weighted by uncertain regimes inferred from a richer set of N series.

    m = make_regime_pca("regime_pca", columns=tuple(k_ids), regime_overrides=(("columns", tuple(n_ids)),))
    data = m.prepare(panel)                 # panel holds the N and the K series (K may be inside N)
    m.fit(data, as_of="2025-12-31")
    out = m.predict(data, start="2025-12-31")
    pick(out, "resid_z")                    # residual z per K series - the trading signal
    pick(out, "p")                          # the regime probabilities each row used

Fit, on rows known by ``as_of``:

1. the regime model (``infra.models.stats.regimes``, any of them: HMM on N-series features
   by default, or explicit rules) gives each fit-sample row SMOOTHED regime probabilities -
   legitimate inside the fit sample, which only holds data up to ``as_of`` anyway;
2. per regime r, the weighted mean and covariance of the K series, weights = p_r(t) x the
   optional time decay; complete rows, or probabilistic-PCA EM (``missing="em"``);
3. each regime's moments are shrunk toward the pooled ones by its effective sample size:
   lambda_r = n_eff_r / (n_eff_r + kappa), kappa = ``shrink_obs`` (default K(K+1)/2, one
   prior observation per covariance term) - a rare regime's covariance is otherwise noise.

Predict, per row with regime probabilities q (``timing``: the PREDICTED probability
p(s_t | data to t-1) by default - fixed before the row's own move, so a large idiosyncratic
move in a K series can't shift the regime and with it the loadings that measure that same
move; ``"filtered"`` uses p(s_t | data to t). Rows up to the fit date use the smoothed
probabilities and are flagged ``in_sample``):

* ``combine="covariance"`` (default): blend the regime moments into ONE distribution,
  m = sum q_r mu_r, C = sum q_r (C_r + (mu_r - m)(mu_r - m)'), take its top-k PCs
  (signs anchored to the pooled PCA's), residual = x - reconstruction. One well-defined
  residual and hedge; loadings move smoothly with q. ``resid_z`` divides by the
  MODEL-IMPLIED residual sd, sqrt(sum_{j>k} lambda_j v_ij^2) of that day's blended
  covariance (scaled back by the fit-sample ratio of realised to implied, so z has unit
  variance in sample).
* ``"residual"``: each regime's own PCA residual, blended with q (variance blended too).
  Simple, but not the residual of any single model.
* ``"robust"``: neutral to EVERY regime's top-k factors at once (residual orthogonal to the
  span of all of them, a fixed basis): the hedge holds whichever regime is true; scores are
  on that basis (``score:B<i>``), not PCs. Needs K > R x k.

Never "most likely regime only": the residual would jump whenever the argmax flips, with no
price having moved - phantom signals and turnover.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from infra.models.stats.common import (PARAM_COLUMNS, StatModel, cutoff_mask, decay_weights, effective_n,
                                       params_frame, section)
from infra.models.stats.config import RegimePCASpec, get_regime_pca_spec, get_regime_spec
from infra.models.stats.pca import orient, ppca_em, project_observed
from infra.models.stats.regimes import REGIME_CLASSES, RegimeModel, make_regime_model

R_PREFIX, K_PREFIX = "R|", "K|"


@dataclass
class RegimePCAFit:
    as_of: pd.Timestamp
    columns: list[str]
    k: int
    D: np.ndarray            # per column: 1/sd (scale=True) or 1
    mu: np.ndarray           # R x K, model units (prepared + stateful prep), already shrunk
    cov: np.ndarray          # R x K x K
    pooled_mu: np.ndarray
    pooled_cov: np.ndarray
    L0: np.ndarray           # K x k pooled loadings (sign anchor), in the D-scaled space
    L: np.ndarray            # R x K x k per-regime loadings (D-scaled space)
    regime_var: np.ndarray   # R x K model-implied residual variance of each regime's own PCA (D space)
    B: np.ndarray            # K x b robust basis (D space)
    z_scale: np.ndarray      # K: realised / implied residual sd in sample (so z ~ unit variance)
    occupancy: np.ndarray    # R: mean smoothed probability (fill for rows without one)
    stats: dict
    sample: pd.DataFrame | None


def _moments(Z: np.ndarray, w: np.ndarray, k: int, missing: str):
    """Weighted (mean, covariance, n_eff) of rows; complete rows or PPCA-EM."""
    if missing == "em":
        rows = np.isfinite(Z).any(axis=1) & (w > 0)
        if rows.sum() < 3:
            return None
        mu, C, _ = ppca_em(Z[rows], w[rows], k)
        return mu, C, effective_n(w[rows], int(rows.sum()))
    ok = np.isfinite(Z).all(axis=1) & (w > 0)
    if ok.sum() < 3:
        return None
    ww = w[ok]
    mu = np.average(Z[ok], axis=0, weights=ww)
    D = Z[ok] - mu
    return mu, (D * ww[:, None]).T @ D / ww.sum(), effective_n(ww, int(ok.sum()))


def _top(C: np.ndarray, k: int, anchor: np.ndarray | None):
    vals, vecs = np.linalg.eigh((C + C.T) / 2)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]
    L = orient(vecs[:, :k], "sum")
    if anchor is not None:
        L = L * np.where(np.sum(L * anchor, axis=0) < 0, -1.0, 1.0)
    resid_var = np.sum(vecs[:, k:] ** 2 * vals[k:], axis=1)
    return L, vals, resid_var


class RegimePCA(StatModel):
    """See the module docstring."""

    def __init__(self, spec: RegimePCASpec | str | None = None, *, regime_model: RegimeModel | None = None,
                 **overrides):
        """``regime_model``: an explicit regime model instance (e.g. ``RuleRegimes`` with a
        probability frame); otherwise built from ``spec.regime`` + ``regime_overrides``."""
        spec = get_regime_pca_spec(spec, **overrides)
        super().__init__(spec)
        self.regime = regime_model if regime_model is not None else make_regime_model(
            get_regime_spec(spec.regime, **dict(spec.regime_overrides)))

    # ------------------------------------------------------------------ prepare
    def k_columns(self, raw_columns) -> list[str]:
        return list(self.spec.columns) if self.spec.columns is not None else list(raw_columns)

    def prepare(self, raw: pd.DataFrame, *, skip=()) -> pd.DataFrame:
        """Both models' stateless prep, side by side: ``R|<col>`` (the regime model's
        inputs) and ``K|<col>`` (the PCA's)."""
        k = self.prep.prepare(raw[self.k_columns(raw.columns)], skip=skip)
        r = self.regime.prepare(raw, skip=skip)
        out = pd.concat([r.add_prefix(R_PREFIX), k.add_prefix(K_PREFIX)], axis=1)
        out.index.name = "timestamp"
        return out

    @staticmethod
    def _split(prepared: pd.DataFrame):
        r = prepared[[c for c in prepared.columns if c.startswith(R_PREFIX)]]
        k = prepared[[c for c in prepared.columns if c.startswith(K_PREFIX)]]
        return r.rename(columns=lambda c: c[len(R_PREFIX):]), k.rename(columns=lambda c: c[len(K_PREFIX):])

    def warm_start_from(self, previous: "RegimePCA") -> None:
        if isinstance(previous, RegimePCA) and previous.is_fitted:
            self.regime.warm_start_from(previous.regime)

    # ------------------------------------------------------------------ fit
    def fit(self, prepared: pd.DataFrame, as_of=None) -> "RegimePCA":
        sp = self.spec
        r_part, k_part = self._split(prepared)
        as_of = k_part.index.max() if as_of is None else pd.Timestamp(as_of)
        self.regime.fit(r_part, as_of=as_of)
        sample, as_of = self.fit_sample(k_part, as_of)
        sample = sample.dropna(how="all")
        if len(sample) < sp.min_obs:
            raise ValueError(f"RegimePCA: {len(sample)} rows up to {as_of.date()} (min_obs {sp.min_obs})")
        cols = list(sample.columns)
        K, R = len(cols), self.regime.n_regimes
        k = min(sp.n_components, K - 1)
        decay = decay_weights(len(sample), sp.halflife)
        decay = np.ones(len(sample)) if decay is None else decay
        self.prep.fit(sample, decay)
        Z = self.prep.transform(sample).to_numpy(dtype="float64")
        probs = self.regime.predict(r_part, end=as_of)
        P = probs.reindex(sample.index)[[f"p_smooth:R{r}" for r in range(R)]].to_numpy(dtype="float64")
        known = np.isfinite(P).all(axis=1)
        occupancy = P[known].mean(axis=0) if known.any() else np.full(R, 1.0 / R)
        P[~known] = occupancy
        pooled = _moments(Z, decay, k, sp.missing)
        if pooled is None:
            raise ValueError("RegimePCA: no usable rows for the pooled moments")
        mu0, C0, n0 = pooled
        kappa = sp.shrink_obs if sp.shrink_obs is not None else K * (K + 1) / 2
        mus, covs, stats = np.zeros((R, K)), np.zeros((R, K, K)), {"n": int(len(sample)), "n_eff": n0,
                                                                    "kappa": kappa}
        for r in range(R):
            mo = _moments(Z, P[:, r] * decay, k, sp.missing)
            mu_r, C_r, ne = (mu0, C0, 0.0) if mo is None else mo
            lam = ne / (ne + kappa)
            mus[r] = lam * mu_r + (1 - lam) * mu0
            covs[r] = lam * C_r + (1 - lam) * C0
            stats[f"n_eff_R{r}"], stats[f"lambda_R{r}"] = ne, lam
            stats[f"occupancy_R{r}"] = float(occupancy[r])
        D = 1.0 / np.sqrt(np.clip(np.diag(C0), 1e-300, None)) if sp.scale else np.ones(K)
        DD = np.outer(D, D)
        L0, vals0, _ = _top(C0 * DD, k, None)
        L, regime_var = np.zeros((R, K, k)), np.zeros((R, K))
        for r in range(R):
            L[r], vals, regime_var[r] = _top(covs[r] * DD, k, L0)
            stats[f"explained_k_R{r}"] = float(vals[:k].sum() / vals.sum()) if vals.sum() > 0 else np.nan
        stats["explained_k_pooled"] = float(vals0[:k].sum() / vals0.sum()) if vals0.sum() > 0 else np.nan
        U, s, _ = np.linalg.svd(np.concatenate(list(L), axis=1), full_matrices=False)
        rank = min(int(np.sum(s > 1e-8 * s.max())), K - 1)
        B = orient(U[:, :rank], "sum")
        stats["robust_rank"] = rank
        stats.update({f"regime.{key}": v for key, v in self.regime.fitted_.stats.items()})
        self.fitted_ = RegimePCAFit(as_of=as_of, columns=cols, k=k, D=D, mu=mus, cov=covs, pooled_mu=mu0,
                                    pooled_cov=C0, L0=L0, L=L, regime_var=regime_var, B=B, z_scale=np.ones(K),
                                    occupancy=occupancy, stats=stats,
                                    sample=prepared[cutoff_mask(prepared.index, as_of)])
        # calibrate z: realised in-sample residual sd / model-implied (each series)
        core = self._core(Z, P)
        ratio = np.sqrt(np.nanmean(core["resid"] ** 2, axis=0) / np.nanmean(core["var"], axis=0))
        self.fitted_.z_scale = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
        for i, c in enumerate(cols):
            stats[f"z_scale:{c}"] = float(self.fitted_.z_scale[i])
        return self

    # ------------------------------------------------------------------ core
    def _core(self, Z: np.ndarray, q: np.ndarray) -> dict:
        """scores, reconstruction, residual, model-implied residual variance - all in the
        D-scaled model space - for rows Z (prepared + stateful prep) with probabilities q."""
        f = self.fitted_
        D = f.D
        X = Z * D
        mus = f.mu * D
        covs = f.cov * np.outer(D, D)
        mode = self.spec.combine
        if mode == "residual":
            n, K = X.shape
            scores = np.zeros((n, f.k))
            resid = np.zeros((n, K))
            var = q @ f.regime_var
            for r in range(len(mus)):
                Xc = X - mus[r]
                lam_r = np.sort(np.linalg.eigvalsh(covs[r]))[::-1]
                fr = project_observed(Xc, f.L[r], lam_r[:f.k], float(lam_r[f.k:].mean()))
                scores += q[:, [r]] * fr
                resid += q[:, [r]] * (Xc - fr @ f.L[r].T)
            return {"scores": scores, "resid": resid, "recon": X - resid, "var": var}
        m = q @ mus
        dev = mus[None, :, :] - m[:, None, :]
        C = np.einsum("nr,rij->nij", q, covs) + np.einsum("nr,nri,nrj->nij", q, dev, dev)
        Xc = X - m
        if mode == "robust":
            Bm = f.B
            bv = np.einsum("ib,nij,jb->nb", Bm, C, Bm).mean(axis=0) if len(C) else np.ones(Bm.shape[1])
            noise = float(np.mean(np.diagonal(C, axis1=1, axis2=2))) * 1e-3 if len(C) else 0.0
            scores = project_observed(Xc, Bm, bv, noise)
            recon_c = scores @ Bm.T
            M = np.eye(len(Bm)) - Bm @ Bm.T
            var = np.einsum("ij,njk,ik->ni", M, C, M)
        else:
            vals, vecs = np.linalg.eigh(C)
            vals, vecs = np.clip(vals[:, ::-1], 0, None), vecs[:, :, ::-1]
            Lt = vecs[:, :, :f.k]
            Lt = Lt * np.where(np.einsum("nik,ik->nk", Lt, f.L0) < 0, -1.0, 1.0)[:, None, :]
            # rows with gaps: per-row prior = that row's blended eigenvalues (mean over rows is
            # used for the ridge - project_observed takes one prior vector)
            prior = vals[:, :f.k].mean(axis=0) if len(vals) else np.ones(f.k)
            noise = float(vals[:, f.k:].mean()) if len(vals) and vals.shape[1] > f.k else 0.0
            scores = project_observed(Xc, Lt, prior, noise)
            recon_c = np.einsum("nk,nik->ni", scores, Lt)
            var = np.einsum("nij,nj->ni", vecs[:, :, f.k:] ** 2, vals[:, f.k:])
        return {"scores": scores, "resid": Xc - recon_c, "recon": recon_c + m, "var": var}

    # ------------------------------------------------------------------ predict
    def regime_probabilities(self, prepared: pd.DataFrame, index: pd.DatetimeIndex, *, end=None) -> np.ndarray:
        """The probabilities each row uses: smoothed up to the fit date, ``timing``
        (predicted / filtered) after it; rows with none -> the fit-sample occupancy."""
        f = self.fitted_
        R = self.regime.n_regimes
        r_part, _ = self._split(prepared)
        probs = self.regime.predict(r_part, end=end).reindex(index)
        later = "p_pred" if self.spec.timing == "predicted" else "p_filt"
        insample = cutoff_mask(index, f.as_of)
        q = np.where(insample[:, None], probs[[f"p_smooth:R{r}" for r in range(R)]].to_numpy(dtype="float64"),
                     probs[[f"{later}:R{r}" for r in range(R)]].to_numpy(dtype="float64"))
        bad = ~np.isfinite(q).all(axis=1)
        q[bad] = f.occupancy
        return q

    def predict(self, prepared: pd.DataFrame | None = None, *, start=None, end=None) -> pd.DataFrame:
        """Per row after ``start`` up to ``end``: ``score:PCi`` (``score:B<i>`` for
        robust), ``fitted:<col>``, ``residual:<col>`` (prepared units), ``resid_z:<col>``,
        ``p:R<r>`` (the probabilities used), ``n_obs``, ``in_sample``. Pass the WHOLE
        prepared frame: the regime features need the history before ``start``."""
        self.check_fitted()
        f = self.fitted_
        frame = f.sample if prepared is None else prepared
        if frame is None:
            raise ValueError("no data: this fit was rebuilt from params; pass prepared data")
        _, k_part = self._split(frame)
        rows = self.predict_window(k_part.reindex(columns=f.columns), start, end)
        q = self.regime_probabilities(frame, rows.index, end=end)
        Z = self.prep.transform(rows).to_numpy(dtype="float64")
        core = self._core(Z, q)
        D = f.D
        out = pd.DataFrame(index=rows.index)
        prefix = "B" if self.spec.combine == "robust" else "PC"
        for j in range(core["scores"].shape[1]):
            out[f"score:{prefix}{j + 1}"] = core["scores"][:, j]
        for i, c in enumerate(f.columns):
            out[f"fitted:{c}"] = self.prep.inverse(core["recon"][:, i] / D[i], c)
        for i, c in enumerate(f.columns):
            out[f"residual:{c}"] = self.prep.inverse(core["resid"][:, i] / D[i], c, shift=False)
        sd = np.sqrt(np.clip(core["var"], 1e-300, None)) * f.z_scale
        for i, c in enumerate(f.columns):
            out[f"resid_z:{c}"] = core["resid"][:, i] / sd[:, i]
        for r in range(q.shape[1]):
            out[f"p:R{r}"] = q[:, r]
        out["n_obs"] = np.isfinite(Z).sum(axis=1)
        out["in_sample"] = cutoff_mask(rows.index, f.as_of)
        return out

    transform = predict

    # ------------------------------------------------------------------ tables
    def loadings_by_regime(self, *, units: bool = True) -> pd.DataFrame:
        """Long ``regime, column, pc, loading``: each regime's own PCA (``units``: a 1-sd
        move of the PC in each column's model units)."""
        self.check_fitted()
        f = self.fitted_
        rows = []
        for r in range(len(f.L)):
            vals = np.sort(np.linalg.eigvalsh(f.cov[r] * np.outer(f.D, f.D)))[::-1][:f.k]
            L = f.L[r] * (np.sqrt(vals) / f.D[:, None] if units else 1.0)
            for i, c in enumerate(f.columns):
                for j in range(f.k):
                    rows.append({"regime": f"R{r}", "column": c, "pc": f"PC{j + 1}", "loading": L[i, j]})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        self.check_fitted()
        s = self.fitted_.stats
        R = self.regime.n_regimes
        lines = [f"RegimePCA [{self.spec.name}] as_of={self.fitted_.as_of.date()} combine={self.spec.combine} "
                 f"timing={self.spec.timing} n={s['n']} k={self.fitted_.k}"]
        for r in range(R):
            lines.append(f"  R{r}: occupancy {s[f'occupancy_R{r}']:.1%}  n_eff {s[f'n_eff_R{r}']:.0f}  "
                         f"shrink lambda {s[f'lambda_R{r}']:.2f}  explained {s[f'explained_k_R{r}']:.1%}  "
                         f"duration {s.get(f'regime.duration_R{r}', np.nan):.0f} rows")
        lines.append(f"  pooled explained {s['explained_k_pooled']:.1%}")
        lb = self.loadings_by_regime().pivot_table(index=["regime", "pc"], columns="column", values="loading", sort=False)
        return "\n".join(lines) + "\n" + lb.round(3).to_string()

    # ------------------------------------------------------------------ refits
    def align_to(self, reference: "RegimePCA") -> "RegimePCA":
        """Relabel regimes like ``reference`` (smoothed-probability overlap) and keep the
        pooled loadings' (and robust basis') signs, so scores and regimes are continuous
        across walk-forward refits."""
        self.check_fitted()
        if not isinstance(reference, RegimePCA) or not reference.is_fitted:
            return self
        perm = self.regime.align_to(reference.regime)
        f = self.fitted_
        f.mu, f.cov, f.L, f.regime_var, f.occupancy = f.mu[perm], f.cov[perm], f.L[perm], f.regime_var[perm], f.occupancy[perm]
        stats = dict(f.stats)
        for key in ("n_eff", "lambda", "occupancy", "explained_k"):
            for r in range(len(perm)):
                stats[f"{key}_R{r}"] = f.stats[f"{key}_R{perm[r]}"]
        f.stats = stats
        if reference.fitted_.columns == f.columns:
            flip = np.where(np.sum(f.L0 * reference.fitted_.L0, axis=0) < 0, -1.0, 1.0)
            f.L0, f.L = f.L0 * flip, f.L * flip
            nb = min(f.B.shape[1], reference.fitted_.B.shape[1])
            fb = np.where(np.sum(f.B[:, :nb] * reference.fitted_.B[:, :nb], axis=0) < 0, -1.0, 1.0)
            f.B[:, :nb] *= fb
        return self

    # ------------------------------------------------------------------ params
    def params(self) -> pd.DataFrame:
        self.check_fitted()
        f = self.fitted_
        names = [f"R{r}" for r in range(len(f.mu))]
        pcs = [f"PC{j + 1}" for j in range(f.k)]
        reg = self.regime.params()
        parts = [pd.DataFrame([{"section": "meta", "row": "as_of", "col": f.as_of.isoformat(), "value": np.nan},
                               {"section": "meta", "row": "k", "col": "", "value": float(f.k)}]
                              + [{"section": "meta", "row": "column", "col": c, "value": float(i)}
                                 for i, c in enumerate(f.columns)]),
                 reg.assign(section="regime." + reg["section"]),
                 params_frame("rmean", pd.DataFrame(f.mu, index=names, columns=f.columns)),
                 params_frame("pooled", pd.DataFrame({"mean": f.pooled_mu, "D": f.D, "z_scale": f.z_scale},
                                                     index=f.columns)),
                 params_frame("pooled_cov", pd.DataFrame(f.pooled_cov, index=f.columns, columns=f.columns)),
                 params_frame("L0", pd.DataFrame(f.L0, index=f.columns, columns=pcs)),
                 params_frame("B", pd.DataFrame(f.B, index=f.columns, columns=[f"B{j + 1}" for j in range(f.B.shape[1])])),
                 params_frame("regime_var", pd.DataFrame(f.regime_var, index=names, columns=f.columns)),
                 params_frame("occupancy", pd.Series(f.occupancy, index=names)),
                 params_frame("stat", f.stats), self.prep.params()]
        for r, nm in enumerate(names):
            parts.append(params_frame(f"rcov:{nm}", pd.DataFrame(f.cov[r], index=f.columns, columns=f.columns)))
            parts.append(params_frame(f"L:{nm}", pd.DataFrame(f.L[r], index=f.columns, columns=pcs)))
        return pd.concat([p for p in parts if not p.empty], ignore_index=True)[PARAM_COLUMNS]

    @classmethod
    def from_params(cls, params: pd.DataFrame, spec: RegimePCASpec | str | None = None, **overrides) -> "RegimePCA":
        """Rebuild a predict-ready model (HMM regimes) from ``params()``."""
        m = cls(spec, **overrides)
        sub = params[params["section"].str.startswith("regime.")]
        sub = sub.assign(section=sub["section"].str.slice(len("regime.")))
        rspec = m.regime.spec
        m.regime = REGIME_CLASSES[rspec.method].from_params(sub, rspec) if rspec.method == "hmm" else m.regime
        meta = params[params["section"] == "meta"]
        cols = meta[meta["row"] == "column"].sort_values("value")["col"].tolist()
        k = int(meta.loc[meta["row"] == "k", "value"].iloc[0])
        mean = section(params, "rmean").reindex(columns=cols)
        names = list(mean.index)
        pooled = section(params, "pooled").reindex(cols)
        Bt = section(params, "B").reindex(cols)
        Bt = Bt[sorted(Bt.columns, key=lambda c: int(c[1:]))]
        pcs = [f"PC{j + 1}" for j in range(k)]
        m.prep.load_params(params)
        m.fitted_ = RegimePCAFit(
            as_of=pd.Timestamp(meta.loc[meta["row"] == "as_of", "col"].iloc[0]), columns=cols, k=k,
            D=pooled["D"].to_numpy(), mu=mean.to_numpy(),
            cov=np.array([section(params, f"rcov:{nm}").reindex(index=cols, columns=cols).to_numpy() for nm in names]),
            pooled_mu=pooled["mean"].to_numpy(),
            pooled_cov=section(params, "pooled_cov").reindex(index=cols, columns=cols).to_numpy(),
            L0=section(params, "L0").reindex(index=cols, columns=pcs).to_numpy(),
            L=np.array([section(params, f"L:{nm}").reindex(index=cols, columns=pcs).to_numpy() for nm in names]),
            regime_var=section(params, "regime_var").reindex(index=names, columns=cols).to_numpy(),
            B=Bt.to_numpy(), z_scale=pooled["z_scale"].to_numpy(),
            occupancy=section(params, "occupancy").reindex(names)["value"].to_numpy(),
            stats=section(params, "stat")["value"].to_dict(), sample=None)
        return m


def make_regime_pca(spec: RegimePCASpec | str | None = None, *, regime_model: RegimeModel | None = None,
                    **overrides) -> RegimePCA:
    return RegimePCA(spec, regime_model=regime_model, **overrides)
