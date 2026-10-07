"""Regression and PCA parameters, kept apart from the code (root CLAUDE.md 3 / models 0):
a spec per parametrisation, toggled by name (``REGRESSION_MODELS``, ``PCA_MODELS``), or
built ad hoc (``RegressionSpec(method="ridge", alpha=0.1)``) - the dashboard does that.

A spec says HOW to model; it never says where the data comes from. The column names in
``y`` / ``x`` / ``columns`` are whatever the input frame calls them (a series id from
``infra.pipeline.series_panel`` such as ``bond:US_BOND_10y``, or anything else).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

REGRESSION_METHODS = ("ols", "stepwise", "ridge", "lasso", "elasticnet", "huber", "quantile",
                      "hockey", "tls", "logit", "probit", "kalman")
PCA_METHODS = ("pca", "weighted", "missing", "similarity")
REGIME_METHODS = ("hmm", "rule")
REGIME_PCA_COMBINES = ("covariance", "residual", "robust")


@dataclass(frozen=True)
class RegressionSpec:
    name: str = "custom"
    method: str = "ols"
    description: str = ""
    # --- data ---------------------------------------------------------------
    y: str | None = None                       # target column (None = the input's first)
    x: tuple[str, ...] | None = None           # regressors (None = every other column)
    add_const: bool = True
    prep: tuple[str, ...] = ()                 # infra.models.prep steps, every column
    prep_by_column: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (column, steps) overrides
    x_lags: tuple[int, ...] = (0,)             # regress on x_t, x_{t-1}, ... (0 = contemporaneous)
    y_lead: int = 0                            # target y_{t+h}: a forecasting regression
    # --- fit sample -----------------------------------------------------------
    window: int | str | None = None            # rows (int) or a Timedelta ("730D"); None = expanding
    min_obs: int = 20
    halflife: float | None = None              # exponential observation weights, in rows (WLS)
    # --- inference --------------------------------------------------------------
    cov: str = "nonrobust"                     # nonrobust | HC0 | HC1 | HAC
    hac_lags: int | None = None                # None = Newey-West rule of thumb
    # --- penalised (glmnet objective: 1/2n |y - Xb|^2 + alpha (l1 |b|_1 + (1-l1)/2 |b|^2),
    #     X standardised on the fit sample, intercept never penalised) ------------
    alpha: float | str = 1.0                   # a number, or "cv": forward-chaining CV
    l1_ratio: float = 0.5                      # elasticnet only (ridge 0, lasso 1)
    cv_folds: int = 5
    cv_grid: int = 30
    # --- stepwise ---------------------------------------------------------------
    direction: str = "both"                    # forward | backward | both
    criterion: str = "pvalue"                  # pvalue | aic | bic
    p_enter: float = 0.05
    p_remove: float = 0.10
    max_vars: int | None = None
    # --- robust -------------------------------------------------------------------
    huber_k: float = 1.345                     # in units of the robust residual scale
    tau: float = 0.5                           # quantile regression's quantile
    # --- hockey stick (piecewise linear in ONE regressor) --------------------------
    knot_var: str | None = None                # the hinge regressor (None = first x)
    n_knots: int = 1
    knots: tuple[float, ...] | None = None     # fixed knots (None = estimated by grid search)
    knot_range: tuple[float, float] = (0.10, 0.90)  # quantiles of knot_var searched
    knot_grid: int = 60
    # --- total least squares ---------------------------------------------------------
    variance_ratio: float = 1.0                # var(error in y) / var(error in x), Deming
    # --- binary (logit / probit) -------------------------------------------------------
    threshold: float | None = None             # y > threshold -> 1 (None: y must be 0/1)
    # --- kalman (time-varying coefficients, random walk) -----------------------------------
    kalman_delta: float | None = None          # state noise share (Q = delta/(1-delta) R); None = MLE
    kalman_update: bool = True                 # predict keeps filtering (False: freeze the last beta)

    def __post_init__(self):
        if self.method not in REGRESSION_METHODS:
            raise ValueError(f"unknown regression method {self.method!r}; one of {REGRESSION_METHODS}")


@dataclass(frozen=True)
class PCASpec:
    name: str = "custom"
    method: str = "pca"
    description: str = ""
    columns: tuple[str, ...] | None = None     # None = every input column
    n_components: int = 3
    prep: tuple[str, ...] = ()
    prep_by_column: tuple[tuple[str, tuple[str, ...]], ...] = ()
    scale: bool = False                        # True: correlation PCA (unit variance), False: covariance
    window: int | str | None = None
    min_obs: int = 20
    # weighted PCA
    halflife: float | None = None              # exponential observation (time) weights, rows
    var_weights: tuple[tuple[str, float], ...] = ()  # (column, weight): importance of each variable
    # similarity weights (method "similarity"): rows weighted by closeness of their state to the fit date's
    state_columns: tuple[str, ...] = ()        # the N state series (not decomposed)
    bandwidth: float = 1.0                     # Gaussian kernel width, in standard deviations of the state
    # missing data
    missing: str = "drop"                      # drop (complete rows) | em | pairwise
    em_max_iter: int = 500
    em_tol: float = 1e-7
    min_coverage: float = 0.0                  # drop a column observed on fewer fit rows than this share
    # sign: each PC is oriented so its loadings SUM positive (PC1 = everything up)
    sign: str = "sum"                          # sum | max (largest |loading| positive)

    def __post_init__(self):
        if self.method not in PCA_METHODS:
            raise ValueError(f"unknown PCA method {self.method!r}; one of {PCA_METHODS}")


@dataclass(frozen=True)
class RegimeSpec:
    """A regime model on N input series (``infra.models.stats.regimes``)."""
    name: str = "custom"
    method: str = "hmm"
    description: str = ""
    columns: tuple[str, ...] | None = None     # the N series (None = every input column)
    n_regimes: int = 2
    prep: tuple[str, ...] = ("diff", "mult:100")
    prep_by_column: tuple[tuple[str, tuple[str, ...]], ...] = ()
    window: int | str | None = None
    min_obs: int = 250
    # features (regimes.FEATURE_BUILDERS): name + keyword arguments
    features: str = "pca_vol"
    # default: the DAILY factor scores (a Gaussian HMM with a covariance per regime - vol AND
    # correlation regimes, no window lag). Synthetic 2-regime panels, 4 seeds (2026-10-03,
    # tests/test_regimes.py): out-of-sample regime accuracy 98.4-98.8%, regime loadings as
    # good as with the TRUE labels (<= 6 deg); 10-row rolling vols lagged each switch (94-97%),
    # and the few stressed rows they mislabelled calm (9x the variance) rotated the calm
    # loadings by up to 27 deg, once 77 deg. 21-row windows were worse; adding drift
    # features let a drift split hijack the HMM once in 3 seeds (71%).
    feature_args: tuple[tuple[str, object], ...] = (("n_factors", 3), ("vols", False), ("scores", True))
    # hmm
    covariance: str = "full"                   # full | diag
    n_init: int = 3                            # k-means starts on a cold fit (a warm refit runs once)
    max_iter: int = 200
    tol: float = 1e-6
    reg: float = 1e-3                          # covariance ridge, x each feature's variance
    stay: float = 0.95                         # initial probability of staying in a regime
    sticky: float = 0.0                        # pseudo-counts of staying (Dirichlet prior on the
                                               # transition diagonal): forces persistent regimes
    # refits start EM from the previous fit (faster; regimes stay near the previous optimum -
    # continuity) or cold, n_init k-means starts every time (each fit depends on its window
    # only: path-INDEPENDENT, so a rebuild from any start date reproduces it)
    warm_start: bool = True
    seed: int = 0
    # rule
    rule: str = "threshold"
    rule_column: str | None = None
    rule_thresholds: tuple[float, ...] = ()
    rule_softness: float = 0.0
    rule_columns: tuple[str, ...] = ()            # grid rule: the columns ...
    rule_grid: tuple[tuple[float, ...], ...] = ()  # ... and each one's thresholds

    def __post_init__(self):
        if self.method not in REGIME_METHODS:
            raise ValueError(f"unknown regime method {self.method!r}; one of {REGIME_METHODS}")


@dataclass(frozen=True)
class RegimePCASpec:
    """PCA of K series weighted by the regime probabilities of a regime model on N series
    (``infra.models.stats.regime_pca``)."""
    name: str = "custom"
    description: str = ""
    columns: tuple[str, ...] | None = None     # the K series (None = every input column)
    regime: RegimeSpec | str = "hmm2"
    regime_overrides: tuple[tuple[str, object], ...] = ()
    n_components: int = 3
    prep: tuple[str, ...] = ("diff", "mult:100")
    prep_by_column: tuple[tuple[str, tuple[str, ...]], ...] = ()
    scale: bool = False
    window: int | str | None = None
    min_obs: int = 250
    halflife: float | None = None              # time decay, multiplied with the regime probabilities
    missing: str = "drop"                      # drop (complete rows) | em (PPCA-EM per regime)
    combine: str = "covariance"                # covariance | residual | robust
    timing: str = "predicted"                  # predicted (p(s_t | data to t-1)) | filtered (p(s_t | data to t))
    shrink_obs: float | None = None            # prior observations pulling each regime toward the pooled
                                               # moments; None = K(K+1)/2 (one per covariance term)

    def __post_init__(self):
        if self.combine not in REGIME_PCA_COMBINES:
            raise ValueError(f"combine {self.combine!r}; one of {REGIME_PCA_COMBINES}")
        if self.timing not in ("predicted", "filtered"):
            raise ValueError(f"timing {self.timing!r} (predicted | filtered)")


def _named(**specs):
    return {k: replace(v, name=k) for k, v in specs.items()}


REGRESSION_MODELS: dict[str, RegressionSpec] = _named(
    ols=RegressionSpec(method="ols", description="OLS, classical errors"),
    ols_hac=RegressionSpec(method="ols", cov="HAC", description="OLS, Newey-West errors (overlapping / autocorrelated)"),
    ols_changes=RegressionSpec(method="ols", prep=("diff",), cov="HAC", description="OLS on daily changes"),
    hedge_ratio=RegressionSpec(method="ols", prep=("diff",), cov="HAC", halflife=126.0,
                               window=504, description="beta of changes, 6m half-life, 2y window"),
    stepwise=RegressionSpec(method="stepwise", description="stepwise by p-value (enter 5%, remove 10%)"),
    ridge=RegressionSpec(method="ridge", alpha="cv"),
    lasso=RegressionSpec(method="lasso", alpha="cv"),
    elasticnet=RegressionSpec(method="elasticnet", alpha="cv", l1_ratio=0.5),
    huber=RegressionSpec(method="huber"),
    lad=RegressionSpec(method="quantile", tau=0.5, description="median (least absolute deviation)"),
    hockey=RegressionSpec(method="hockey"),
    tls=RegressionSpec(method="tls", description="orthogonal regression: symmetric in y and x"),
    logit=RegressionSpec(method="logit"),
    probit=RegressionSpec(method="probit"),
    kalman_beta=RegressionSpec(method="kalman", prep=("diff",), description="time-varying beta of changes"),
)

PCA_MODELS: dict[str, PCASpec] = _named(
    curve_levels=PCASpec(method="pca", n_components=3, description="covariance PCA of levels"),
    curve_changes=PCASpec(method="pca", n_components=3, prep=("diff", "mult:100"),
                          description="covariance PCA of daily changes, bp"),
    curve_changes_ew=PCASpec(method="weighted", n_components=3, prep=("diff", "mult:100"), halflife=126.0,
                             description="exponentially weighted (6m half-life) PCA of changes, bp"),
    cross_market=PCASpec(method="missing", n_components=3, prep=("diff", "mult:100"), missing="em",
                         description="changes across markets with different holidays (EM-PCA)"),
    correlation=PCASpec(method="pca", scale=True, prep=("diff",), description="correlation PCA of changes"),
)


REGIME_MODELS: dict[str, RegimeSpec] = _named(
    hmm2=RegimeSpec(n_regimes=2, description="2-regime Gaussian HMM on the N series' daily PCA factor scores"),
    hmm3=RegimeSpec(n_regimes=3, description="3-regime Gaussian HMM on the N series' daily PCA factor scores"),
    hmm2_vol=RegimeSpec(n_regimes=2, feature_args=(("n_factors", 3), ("window", 10)),
                        description="2-regime HMM on 10-row factor volatilities (slower regimes; lags switches)"),
    hmm2_trend=RegimeSpec(n_regimes=2, feature_args=(("n_factors", 3), ("window", 21), ("drifts", True)),
                          description="2-regime HMM on factor volatilities AND drift t-stats (21 rows)"),
)

REGIME_PCA_MODELS: dict[str, RegimePCASpec] = _named(
    regime_pca=RegimePCASpec(description="K-series PCA of bp changes, weighted by a 2-regime HMM on N series"),
    regime_pca3=RegimePCASpec(regime="hmm3", description="same, 3 regimes"),
    regime_pca_robust=RegimePCASpec(combine="robust", description="residual neutral to every regime's factors"),
)


def _lookup(spec, registry, methods, cls, default):
    """A spec, a registry name, a bare method name (defaults for that method) or None."""
    if spec is None:
        return registry[default]
    if not isinstance(spec, str):
        return spec
    if spec in registry:
        return registry[spec]
    if spec in methods:
        return cls(name=spec, method=spec)
    raise KeyError(f"unknown spec {spec!r}: not in the registry {sorted(registry)} nor a method {methods}")


def get_regression_spec(spec: RegressionSpec | str | None = None, **overrides) -> RegressionSpec:
    base = _lookup(spec, REGRESSION_MODELS, REGRESSION_METHODS, RegressionSpec, "ols")
    return replace(base, **overrides) if overrides else base


def get_pca_spec(spec: PCASpec | str | None = None, **overrides) -> PCASpec:
    base = _lookup(spec, PCA_MODELS, PCA_METHODS, PCASpec, "curve_changes")
    return replace(base, **overrides) if overrides else base


def get_regime_spec(spec: RegimeSpec | str | None = None, **overrides) -> RegimeSpec:
    base = _lookup(spec, REGIME_MODELS, REGIME_METHODS, RegimeSpec, "hmm2")
    return replace(base, **overrides) if overrides else base


def get_regime_pca_spec(spec: RegimePCASpec | str | None = None, **overrides) -> RegimePCASpec:
    if spec is None or isinstance(spec, RegimePCASpec):
        base = spec or REGIME_PCA_MODELS["regime_pca"]
    elif spec in REGIME_PCA_MODELS:
        base = REGIME_PCA_MODELS[spec]
    else:
        raise KeyError(f"unknown regime PCA spec {spec!r}; known {sorted(REGIME_PCA_MODELS)}")
    return replace(base, **overrides) if overrides else base
