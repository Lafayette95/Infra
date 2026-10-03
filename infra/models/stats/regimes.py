"""Regime models: per row, the probability of each regime - from a hidden Markov model on
features of a (rich) set of N series, or from an explicit rule. All share one interface,
so ``RegimePCA`` (and later a regime-weighted / Markov-switching regression) take any of
them.

    reg = make_regime_model("hmm2", columns=tuple(n_asset_ids))
    data = reg.prepare(panel)
    reg.fit(data, as_of="2025-12-31")
    probs = reg.predict(data)        # p_filt:<r>, p_pred:<r>, p_smooth:<r> (in sample only), in_sample

Features (``FEATURE_BUILDERS``, pluggable; ``RegimeSpec.features`` + ``feature_args``):

* ``pca_vol``: a correlation PCA of the N prepared series (missing-data aware, fitted on
  the fit sample, frozen), then per factor any of: the daily score itself
  (``score:PCi``, ``scores=True``), the log rolling volatility of the score (``vol:PCi``,
  ``vols=True``), its rolling drift t-statistic (``drift:PCi`` = rolling sum / (rolling
  std x sqrt(window)), ``drifts=True``). All trailing. DEFAULT (``RegimeSpec``): daily
  scores only - a Gaussian HMM with a covariance per regime then separates volatility AND
  correlation regimes with no window lag (evidence in ``RegimeSpec.feature_args``). Few features from many series: a full
  Gaussian on N raw series would need N(N+1)/2 covariance terms per regime.
* ``columns``: the prepared input columns themselves (a funding spread, a vol index ...).

Probabilities (see ``hmm``): ``p_filt`` is known at t, ``p_pred`` before row t's data,
``p_smooth`` uses the whole fit sample and exists only for rows up to the fit date - for
estimating things inside the fit sample, never as a signal.

Rows with gaps: the filter uses whatever features a row has (marginal likelihood); the
parameters are estimated from complete feature rows only (an approximation: a full
missing-data M-step would also use the partial rows; features are complete on almost every
row once the rolling window has filled).

Regime labels: the first fit orders regimes calmest first (by the mean of the ``vol:``
features, else by the trace of the regime covariance - the total variance with daily
scores); ``align_to(previous_fit)`` then relabels each refit to match the
previous one by the overlap of their smoothed probabilities on the dates both cover
(robust to the feature PCA itself being refitted), and ``walk_forward`` calls it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from infra.models.stats import hmm
from infra.models.stats.common import PARAM_COLUMNS, StatModel, cutoff_mask, params_frame, section
from infra.models.stats.config import RegimeSpec, get_regime_spec
from infra.models.stats.pca import PCA, make_pca


# --------------------------------------------------------------------------- features
class FeatureBuilder:
    """``fit`` (stateful parts, rows up to ``as_of``) -> ``transform`` (any rows; trailing)."""

    def fit(self, frame: pd.DataFrame, as_of, reference: "FeatureBuilder | None" = None) -> "FeatureBuilder":
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def params(self) -> pd.DataFrame:
        return pd.DataFrame(columns=PARAM_COLUMNS)

    def load_params(self, params: pd.DataFrame) -> "FeatureBuilder":
        return self


@dataclass
class ColumnFeatures(FeatureBuilder):
    columns: tuple[str, ...] | None = None

    def transform(self, frame):
        return frame[list(self.columns)] if self.columns else frame


@dataclass
class PCAVolFeatures(FeatureBuilder):
    n_factors: int = 3
    window: int = 10
    vols: bool = True
    drifts: bool = False
    scores: bool = False            # the daily factor scores themselves (no window, no lag)
    scale: bool = True             # correlation PCA: N series in different units
    pca_: PCA | None = field(default=None, repr=False)

    def fit(self, frame, as_of, reference=None):
        k = min(self.n_factors, frame.shape[1])
        self.pca_ = make_pca("missing", prep=(), n_components=k, scale=self.scale).fit(frame, as_of=as_of)
        if reference is not None and getattr(reference, "pca_", None) is not None:
            self.pca_.align_to(reference.pca_)
        return self

    def transform(self, frame):
        scores = self.pca_.predict(frame.reindex(columns=self.pca_.fitted_.columns))
        scores = scores[[c for c in scores.columns if c.startswith("score:")]]
        scores.columns = [c.split(":", 1)[1] for c in scores.columns]
        w, mp = self.window, max(self.window // 2, 3)
        out = {}
        if self.scores:
            for c in scores.columns:
                out[f"score:{c}"] = scores[c]
        sd = scores.rolling(w, min_periods=mp).std()
        if self.vols:
            for c in scores.columns:
                out[f"vol:{c}"] = np.log(sd[c].where(sd[c] > 0))
        if self.drifts:
            s = scores.rolling(w, min_periods=mp).sum()
            n = scores.rolling(w, min_periods=mp).count()
            for c in scores.columns:
                out[f"drift:{c}"] = s[c] / (sd[c] * np.sqrt(n[c]))
        return pd.DataFrame(out, index=frame.index)

    def params(self):
        p = self.pca_.params()
        return p.assign(section="feat." + p["section"])

    def load_params(self, params):
        sub = params[params["section"].str.startswith("feat.")]
        sub = sub.assign(section=sub["section"].str.slice(5))
        self.pca_ = PCA.from_params(sub, "missing", prep=(), n_components=self.n_factors, scale=self.scale)
        return self


FEATURE_BUILDERS: dict[str, Callable[..., FeatureBuilder]] = {"pca_vol": PCAVolFeatures, "columns": ColumnFeatures}


def make_features(name: str, args) -> FeatureBuilder:
    return FEATURE_BUILDERS[name](**dict(args))


# --------------------------------------------------------------------------- base
class RegimeModel(StatModel):
    """Common interface: ``n_regimes``, ``fit``, ``predict`` -> ``p_filt:/p_pred:/p_smooth:``
    columns + ``in_sample``, ``smoothed_`` (fit sample), ``align_to``, ``params``."""

    def __init__(self, spec: RegimeSpec | str | None = None, **overrides):
        super().__init__(get_regime_spec(spec, **overrides))

    def input_columns(self, raw):
        return list(self.spec.columns) if self.spec.columns is not None else list(raw.columns)

    @property
    def n_regimes(self) -> int:
        return self.spec.n_regimes

    @property
    def regime_names(self) -> list[str]:
        return [f"R{r}" for r in range(self.n_regimes)]

    def _frame(self, index, filt, pred, smooth, insample) -> pd.DataFrame:
        out = pd.DataFrame(index=index)
        for kind, arr in (("p_filt", filt), ("p_pred", pred), ("p_smooth", smooth)):
            for r, name in enumerate(self.regime_names):
                out[f"{kind}:{name}"] = arr[:, r]
        out["in_sample"] = insample
        return out

    def align_to(self, reference: "RegimeModel") -> np.ndarray:
        """Relabel to match ``reference`` by smoothed-probability overlap on common dates.
        Returns the permutation applied (new label r = old label perm[r])."""
        return np.arange(self.n_regimes)

    def warm_start_from(self, previous: "RegimeModel") -> None:
        """Hook: initialise the next fit from a previous one (walk-forward refits)."""


# --------------------------------------------------------------------------- HMM
@dataclass
class HMMFit:
    as_of: pd.Timestamp
    params: hmm.HMMParams
    filt_last: np.ndarray                 # filtered probability at the fit date (the state predict continues)
    sample_index: pd.DatetimeIndex        # rows the HMM was fitted on
    filtered: np.ndarray
    predicted: np.ndarray
    smoothed: np.ndarray
    feature_names: list[str]
    stats: dict
    sample: pd.DataFrame | None


class HMMRegimes(RegimeModel):
    """Gaussian hidden Markov model on features of the input series (module docstring)."""

    def __init__(self, spec: RegimeSpec | str | None = None, **overrides):
        super().__init__(spec, **overrides)
        self.features = make_features(self.spec.features, self.spec.feature_args)
        self._warm: HMMRegimes | None = None

    def warm_start_from(self, previous):
        if isinstance(previous, HMMRegimes) and previous.is_fitted:
            self._warm = previous

    def fit(self, prepared: pd.DataFrame, as_of=None) -> "HMMRegimes":
        sample, as_of = self.fit_sample(prepared, as_of)
        upto = prepared[cutoff_mask(prepared.index, as_of)]
        self.prep.fit(sample)
        upto_s = self.prep.transform(upto)
        warm = self._warm
        self.features.fit(self.prep.transform(sample), as_of, reference=warm.features if warm else None)
        F = self.features.transform(upto_s).loc[sample.index]
        F = F[F.notna().any(axis=1)]
        if len(F) < self.spec.min_obs:
            raise ValueError(f"HMMRegimes: {len(F)} feature rows up to {pd.Timestamp(as_of).date()} "
                             f"(min_obs {self.spec.min_obs})")
        X = F.to_numpy(dtype="float64")
        init = None
        if warm is not None and list(warm.fitted_.feature_names) == list(F.columns):
            init = warm.fitted_.params
        sp = self.spec
        params, fb, info = hmm.fit_gaussian_hmm(X, sp.n_regimes, covariance=sp.covariance, n_init=sp.n_init,
                                                max_iter=sp.max_iter, tol=sp.tol, reg=sp.reg, seed=sp.seed,
                                                stay=sp.stay, sticky=sp.sticky, init=init)
        if init is None:  # canonical order on a cold start: calmest first
            vol = [j for j, c in enumerate(F.columns) if c.startswith("vol:")]
            key = params.means[:, vol].sum(axis=1) if vol else np.array([np.trace(c) for c in params.covs])
            # (daily scores: the trace of the regime covariance IS its total variance)
            order = np.argsort(key)
            params, fb = _permute(params, fb, order)
        stats = {"loglik": info["loglik"], "iterations": info["iterations"], "converged": info["converged"],
                 "n": len(F), "warm_start": float(init is not None)}
        for r in range(sp.n_regimes):
            stats[f"occupancy_R{r}"] = float(fb.smoothed[:, r].mean())
            stay = params.P[r, r]
            stats[f"duration_R{r}"] = float(1.0 / (1.0 - stay)) if stay < 1 else np.inf
        self.fitted_ = HMMFit(as_of=pd.Timestamp(as_of), params=params, filt_last=fb.filtered[-1],
                              sample_index=F.index, filtered=fb.filtered, predicted=fb.predicted,
                              smoothed=fb.smoothed, feature_names=list(F.columns), stats=stats, sample=sample)
        self._warm = None
        return self

    def feature_frame(self, prepared: pd.DataFrame) -> pd.DataFrame:
        self.check_fitted()
        return self.features.transform(self.prep.transform(prepared))[self.fitted_.feature_names]

    def predict(self, prepared: pd.DataFrame | None = None, *, start=None, end=None) -> pd.DataFrame:
        """Probabilities for the rows of ``prepared`` after ``start`` up to ``end``. Rows up
        to the fit date come from the fit's own pass (all three kinds); later rows continue
        the filter from the fit date with the parameters frozen (filtered and predicted
        only). Features are computed on the WHOLE ``prepared`` frame (rolling windows need
        the history before ``start``), so pass the full frame."""
        self.check_fitted()
        f = self.fitted_
        R = self.n_regimes
        frame = f.sample if prepared is None else prepared
        if end is not None:
            frame = frame[cutoff_mask(frame.index, end)]
        n = len(frame)
        filt, pred, smooth = (np.full((n, R), np.nan) for _ in range(3))
        insample = cutoff_mask(frame.index, f.as_of)
        pos = pd.Index(f.sample_index).get_indexer(frame.index)
        hit = (pos >= 0) & insample
        filt[hit], pred[hit], smooth[hit] = f.filtered[pos[hit]], f.predicted[pos[hit]], f.smoothed[pos[hit]]
        later = ~insample
        if later.any():
            X = self.feature_frame(frame)[later].to_numpy(dtype="float64")
            p = f.params
            fb = hmm.forward_backward(hmm.gaussian_loglik(X, p.means, p.covs), p.P, f.filt_last @ p.P, smooth=False)
            filt[later], pred[later] = fb.filtered, fb.predicted
        out = self._frame(frame.index, filt, pred, smooth, insample)
        return self.predict_window(out, start, None)

    def align_to(self, reference):
        if not isinstance(reference, RegimeModel) or not reference.is_fitted:
            return np.arange(self.n_regimes)
        mine = pd.DataFrame(self.fitted_.smoothed, index=self.fitted_.sample_index)
        ref = reference.smoothed_frame()
        common = mine.index.intersection(ref.index)
        if len(common) < 5:
            return np.arange(self.n_regimes)
        M = ref.loc[common].to_numpy().T @ mine.loc[common].to_numpy()   # [ref r, mine s]
        rows, cols = linear_sum_assignment(-M)
        perm = np.empty(self.n_regimes, dtype=int)
        perm[rows] = cols                                                    # new label r <- old label perm[r]
        if not np.array_equal(perm, np.arange(self.n_regimes)):
            f = self.fitted_
            fb = hmm.FBResult(f.filtered, f.predicted, f.smoothed, np.zeros((1, 1)), 0.0)
            f.params, fb = _permute(f.params, fb, perm)
            f.filtered, f.predicted, f.smoothed = fb.filtered, fb.predicted, fb.smoothed
            f.filt_last = f.filt_last[perm]
            st = dict(f.stats)
            for r in range(self.n_regimes):
                for k in ("occupancy", "duration"):
                    st[f"{k}_R{r}"] = f.stats[f"{k}_R{perm[r]}"]
            f.stats = st
        return perm

    def smoothed_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.fitted_.smoothed, index=self.fitted_.sample_index)

    # ---------------------------------------------------------------- params
    def params(self) -> pd.DataFrame:
        self.check_fitted()
        f, p = self.fitted_, self.fitted_.params
        names, feats = self.regime_names, f.feature_names
        parts = [pd.DataFrame([{"section": "meta", "row": "as_of", "col": f.as_of.isoformat(), "value": np.nan}]
                              + [{"section": "meta", "row": "feature", "col": c, "value": float(i)}
                                 for i, c in enumerate(feats)]),
                 params_frame("hmm_mean", pd.DataFrame(p.means, index=names, columns=feats)),
                 params_frame("hmm_P", pd.DataFrame(p.P, index=names, columns=names)),
                 params_frame("hmm_state", pd.DataFrame({"pi0": p.pi0, "filt_last": f.filt_last}, index=names)),
                 params_frame("stat", f.stats), self.prep.params(), self.features.params()]
        for r, nm in enumerate(names):
            parts.append(params_frame(f"hmm_cov:{nm}", pd.DataFrame(p.covs[r], index=feats, columns=feats)))
        return pd.concat([x for x in parts if not x.empty], ignore_index=True)[PARAM_COLUMNS]

    @classmethod
    def from_params(cls, params: pd.DataFrame, spec=None, **overrides) -> "HMMRegimes":
        m = cls(spec, **overrides)
        meta = params[params["section"] == "meta"]
        feats = meta[meta["row"] == "feature"].sort_values("value")["col"].tolist()
        names = m.regime_names
        means = section(params, "hmm_mean").reindex(index=names, columns=feats).to_numpy()
        covs = np.array([section(params, f"hmm_cov:{nm}").reindex(index=feats, columns=feats).to_numpy()
                         for nm in names])
        P = section(params, "hmm_P").reindex(index=names, columns=names).to_numpy()
        state = section(params, "hmm_state").reindex(index=names)
        m.prep.load_params(params)
        m.features.load_params(params)
        m.fitted_ = HMMFit(as_of=pd.Timestamp(meta.loc[meta["row"] == "as_of", "col"].iloc[0]),
                           params=hmm.HMMParams(means, covs, P, state["pi0"].to_numpy()),
                           filt_last=state["filt_last"].to_numpy(), sample_index=pd.DatetimeIndex([]),
                           filtered=np.zeros((0, len(names))), predicted=np.zeros((0, len(names))),
                           smoothed=np.zeros((0, len(names))), feature_names=feats,
                           stats=section(params, "stat")["value"].to_dict(), sample=None)
        return m


def _permute(params: hmm.HMMParams, fb: hmm.FBResult, order) -> tuple[hmm.HMMParams, hmm.FBResult]:
    """Relabel regimes: new r = old order[r]."""
    o = np.asarray(order)
    p = hmm.HMMParams(params.means[o], params.covs[o], params.P[np.ix_(o, o)], params.pi0[o])
    f = hmm.FBResult(fb.filtered[:, o], fb.predicted[:, o], fb.smoothed[:, o], fb.xi_sum, fb.loglik)
    return p, f


# --------------------------------------------------------------------------- explicit
class RuleRegimes(RegimeModel):
    """Explicit regimes from a rule on the prepared input columns, or from a given
    probability frame (``probabilities=``, columns = regimes, a row = what was known at
    that timestamp). The rule's value at t is the FILTERED probability (it must only use
    data known at t); ``p_pred`` is the previous row's; ``p_smooth`` = ``p_filt`` (a rule
    has no hindsight - so a regime defined with hindsight, like "hiking cycle until the
    last hike", must not be passed here).

    Built-in rule (``RegimeSpec.rule``): ``"threshold"`` - regime = which interval of
    ``rule_thresholds`` the (prepared) ``rule_column`` falls in; ``rule_softness`` > 0 makes
    each boundary a logistic of that width instead of a step."""

    def __init__(self, spec=None, probabilities: pd.DataFrame | None = None, **overrides):
        if probabilities is not None:
            overrides.setdefault("n_regimes", probabilities.shape[1])
            overrides.setdefault("method", "rule")
        super().__init__(spec, **overrides)
        self.probabilities = probabilities

    def input_columns(self, raw):
        if self.probabilities is not None:
            return []
        return [self.spec.rule_column]

    def prepare(self, raw, *, skip=()):
        if self.probabilities is not None:
            return pd.DataFrame(index=raw.index)
        return super().prepare(raw, skip=skip)

    def _probs(self, prepared: pd.DataFrame) -> np.ndarray:
        if self.probabilities is not None:
            return self.probabilities.reindex(prepared.index).to_numpy(dtype="float64")
        if self.spec.rule != "threshold":
            raise ValueError(f"unknown rule {self.spec.rule!r}")
        x = prepared[self.spec.rule_column].to_numpy(dtype="float64")
        cuts = list(self.spec.rule_thresholds)
        if len(cuts) + 1 != self.n_regimes:
            raise ValueError(f"{len(cuts)} thresholds make {len(cuts) + 1} regimes, spec says {self.n_regimes}")
        s = self.spec.rule_softness
        above = np.column_stack([(x > c).astype(float) if not s else 1 / (1 + np.exp(-(x - c) / s)) for c in cuts]) \
            if cuts else np.zeros((len(x), 0))
        # P(regime r) = P(above cut r-1) - P(above cut r), with P(above cut -1) = 1, P(above cut R-1) = 0
        cum = np.column_stack([np.ones(len(x)), above, np.zeros(len(x))])
        p = cum[:, :-1] - cum[:, 1:]
        p[~np.isfinite(x)] = np.nan
        return p

    def fit(self, prepared, as_of=None):
        sample, as_of = self.fit_sample(prepared, as_of)
        p = self._probs(sample)
        self.fitted_ = HMMFit(as_of=pd.Timestamp(as_of), params=None, filt_last=p[-1] if len(p) else None,
                              sample_index=sample.index, filtered=p, predicted=np.vstack([np.full((1, p.shape[1]), np.nan), p[:-1]]),
                              smoothed=p, feature_names=[], stats={"n": len(p)}, sample=sample)
        return self

    def predict(self, prepared=None, *, start=None, end=None):
        self.check_fitted()
        frame = self.fitted_.sample if prepared is None else prepared
        if end is not None:
            frame = frame[cutoff_mask(frame.index, end)]
        p = self._probs(frame)
        pred = np.vstack([np.full((1, p.shape[1]), np.nan), p[:-1]]) if len(p) else p
        out = self._frame(frame.index, p, pred, p, cutoff_mask(frame.index, self.fitted_.as_of))
        return self.predict_window(out, start, None)

    def smoothed_frame(self):
        return pd.DataFrame(self.fitted_.smoothed, index=self.fitted_.sample_index)

    def params(self):
        return pd.DataFrame([{"section": "meta", "row": "as_of", "col": self.fitted_.as_of.isoformat(), "value": np.nan}])


REGIME_CLASSES = {"hmm": HMMRegimes, "rule": RuleRegimes}


def make_regime_model(spec: RegimeSpec | str | None = None, **overrides) -> RegimeModel:
    spec = get_regime_spec(spec, **overrides)
    return REGIME_CLASSES[spec.method](spec)
