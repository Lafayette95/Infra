# infra/models/stats - Regression & PCA on any series (additive to the root and `infra/models` CLAUDE.md)

Everything in the root `CLAUDE.md` and `infra/models/CLAUDE.md` still applies (conda env,
one-way dependencies, point-in-time `as_of`, UTC, prepare -> fit -> predict). Known edge
cases go in the ONE root `TOFIX.md`, under headings prefixed "Stats:". This file says what
the models MEAN and HOW to use them. Built 2026-10-03.

## 1. What it is
*   **Two model families on the `base.Model` pattern, input-agnostic:** any wide
    `timestamp x series` frame goes in. Column names are just names; the convenience
    reader for stored data is `infra.pipeline.series_panel.read_panel(ids, start, end,
    as_of=)` with ids `<source>:<key>` (`bond:US_BOND_10y`, `fut:ZN.v.0`, `stir:SR3.c.4`,
    `swap:USD:10y`, `repo:SOFR`, `release:PAYEMS`, `bar:ZN.v.0`, ... - its docstring has the
    table). Disk only, never fetches.
*   **Separation of fit and transform is the point.** `fit(prepared, as_of)` estimates on
    rows known by `as_of` only (inside `spec.window`, with `spec.halflife` weights) and
    freezes everything - coefficients, loadings, AND the stateful prep (z-score mean/std).
    `predict(prepared, start=, end=)` applies that frozen fit to any rows, typically the
    ones after the fit date, possibly at another granularity
    (`prepare(raw, skip=("resample",))`). Every output row says whether it was `in_sample`.
*   **`params()` is a complete, tidy (`section, row, col, value`) export** and
    `Regression.from_params` / `PCA.from_params` rebuild a predict-ready model from it. That
    is what makes use case (a) work: store the params of each weekly refit, rebuild any of
    them later (live, intraday, or to audit a backtest).

## 2. The two use cases (see also `infra/models/CLAUDE.md` 0a)
*   **(a) Point-in-time refit for a strategy - the important one.**
    `infra.models.walk_forward.walk_forward(factory, panel, start, end, refit="W-FRI")`:
    fit at each refit date on data up to it, predict until the next refit with parameters
    frozen, stitch. `res.params` (one params frame per refit, `fit_as_of`),
    `res.predictions` (out-of-sample rows only, each tagged with the fit behind it),
    `res.param_path("coef", "coef")` (coefficient paths), `res.params_at(day)` (rebuild that
    day's fit). CLI: `scripts/run_walk_forward.py ... --save NAME` writes
    `Database/Derived/ModelRuns/NAME/{params,predictions}.parquet`
    (`infra.storage.model_runs`). **Pinned by a test:** shocking every value after T leaves
    every prediction and parameter up to T unchanged
    (`tests/test_stats_models.py::test_walk_forward_is_point_in_time`). Vintage data
    (`release:`): pass `raw` as a callable `as_of -> frame`, so each refit reads the
    history as published then.
*   **(b) Interactive exploration: the dashboard's `/models` page**
    (`infra/dashboard/models_{layout,callbacks,charts}.py`). Pick series (dropdown of
    examples + free-typed ids), regression or PCA, method, prep (level / change / bp change
    / log / % change, resample, z-score), window, half-life, method parameters and a FIT
    DATE: everything after it is shaded and out of sample. "Diagnostics" adds coefficient
    paths (regression) or factor-correlation-by-year and eigenvector stability (PCA). The
    page's compute is `models_callbacks.run_model` (plain Python, tested headless). Same
    classes and specs as (a), so what you see is exactly what a backtest would use.

*   **Operated live** per `infra/models/CLAUDE.md` 0b: `scripts/model_run.py create` a
    run, `rebuild --promote` its history, then `run` daily (predict-append + weekly
    fit-append), `rebuild` periodically to reconcile. `KalmanRegression` warm-starts its
    (q, r) search from the previous refit; HMM regimes their EM.

## 3. Preparation (`infra/models/prep.py`, shared by every model)
*   String steps like the nowcast's transforms: `resample:W-FRI[:how]`, `log`, `diff[:n]`,
    `pct`, `logdiff`, `lag[:n]`, `mult:k`, `add:k`, `neg`, `ffill[:n]`, `ewm_z:hl` -
    STATELESS, trailing, run in `prepare` - then STATEFUL `demean`, `scale`, `zscore`,
    `winsor:k`, fitted on the fit sample and frozen. Stateful must come last (enforced).
    `spec.prep` applies to every column, `spec.prep_by_column` overrides per column.
*   `resample` labels and closes buckets on the RIGHT, so a weekly bar is dated by its last
    input (never earlier). For intraday -> trading-day bars use
    `infra.processing.resample` (trading-day semantics, root 6e), not this.
*   Outputs come back in each column's PREPARED units (stateless steps applied, stateful
    scaling undone), so `diff, mult:100` on a yield gives fitted/residuals in bp.

## 4. Regressions (`regression.py`; specs `config.REGRESSION_MODELS`, `RegressionSpec`)
*   **Data fields:** `y` (default: the first column), `x` (default: all others), `x_lags`
    (`(0, 1)` adds `x.lag1`), `y_lead` (forecasting: target y_{t+h}; the fit drops rows whose
    target is not yet known at `as_of`), `add_const`.
*   **Classes** (`make_regression(spec_or_name, **overrides)` picks by `method`):
    `LinearRegression` (OLS/WLS) + `UnivariateRegression` / `MultivariateRegression`;
    `StepwiseRegression` (forward/backward/both by p-value, AIC, BIC; re-selects at every
    refit); `RidgeRegression` / `LassoRegression` / `ElasticNetRegression` (glmnet
    objective, X standardised, `alpha="cv"` = forward-chaining CV - never validates on the
    past with a model fitted on the future); `HuberRegression`; `QuantileRegression`
    (`tau`; LAD = 0.5); `HockeyStickRegression` (continuous piecewise-linear in `knot_var`,
    knots fixed or grid-searched, SSR profile kept, profile-likelihood knot interval);
    `TotalLeastSquares` (Deming with `variance_ratio`, orthogonal for several regressors:
    the symmetric hedge ratio, `beta(a on b) = 1/beta(b on a)`, tested); `LogitRegression` /
    `ProbitRegression` (`threshold` binarises y); `KalmanRegression` (random-walk
    coefficients; q, r by MLE or `kalman_delta`; fitted = one-step-ahead, so the residual is
    an honest innovation; `kalman_update=False` freezes the last beta for other
    granularities).
*   **Errors (`cov`):** `nonrobust`, `HC0`/`HC1` (White), `HAC` (Newey-West, Bartlett,
    `hac_lags` default `floor(4 (n/100)^(2/9))`). Use HAC for anything overlapping or in
    levels. Ridge: errors CONDITIONAL on alpha (shrunk). Lasso / elastic net: none (no valid
    classical errors after selection). Stepwise: the selected model's errors ignore the
    selection (optimistic). Huber/quantile: sandwich (quantile `nonrobust` = iid sparsity,
    Hall-Sheather bandwidth). TLS: delete-block jackknife (20 contiguous blocks). Hockey:
    conditional on the knot; `F_vs_linear_p_naive` is NOT a valid test (Davies' problem).
*   **Every fit reports:** n, effective n (weights), R^2/adj, F (Wald, with the chosen
    covariance), log-likelihood/AIC/BIC, condition number, VIF and standardised coefficient
    per term, and residual diagnostics (`inference.residual_stats`): Durbin-Watson,
    Ljung-Box(10), Jarque-Bera, Breusch-Pagan, AR(1) and HALF-LIFE of the residual, ADF
    t-statistic (compare with `inference.ADF_CRITICAL["eg2"]` when the regression is a
    2-variable cointegrating one - Engle-Granger).
*   **`predict` columns:** `y`, `fitted`, `residual`, `resid_z` (residual / fit-sample
    residual std - the RV signal), `fitted_se` (OLS family: prediction standard error),
    `in_sample`, and with `contributions=True` `contrib:<term>` (sums to `fitted`); Kalman
    adds `beta:<term>`; logit `fitted` = probability, `resid_z` = Pearson residual.
*   **Checked on synthetic data (tests):** OLS = the closed form (coefs and classical
    errors); HAC(0) = White; ridge = its closed form; lasso at alpha_max = all zeros;
    quantile on a constant = the sample quantile; logit/probit recover their coefficients
    (20k obs, +-0.06); hockey recovers knot 0.5 and slopes 0.2/1.7; Kalman tracks a
    drifting beta with less than half a fixed OLS beta's error and calibrated innovations
    (z sd ~1); Huber resists 30-sigma outliers. On real data (US CMT, 2021-2026): 10y
    changes on 2y + 30y, HAC, fit to 2026-04: R^2 0.95, coefficients 0.33 / 0.79; the
    `hedge_ratio` spec (2y window, 6m half-life) refitted weekly over 2025-2026 (92 refits)
    moves only between 0.36 / 0.75 and 0.36 / 0.76 in its last 5 weeks.

## 5. PCA (`pca.py`; specs `config.PCA_MODELS`, `PCASpec`)
*   **Classes:** `PCA` (complete rows), `WeightedPCA` (time weights `halflife` and/or
    variable weights `var_weights`), `MissingDataPCA` (`missing="em"`: maximum-likelihood
    probabilistic PCA by EM, Tipping & Bishop 1999 - the default; `"pairwise"`: pairwise
    covariance clipped to PSD). `scale=True` = correlation PCA. `transform` = `predict`.
*   **Why PPCA-EM, not "impute with the rank-k reconstruction and repeat":** the latter
    drops the imputed cells' idiosyncratic variance, so the factor structure looks stronger
    than it is. Found 2026-10-03 on US/DE/UK 2/10/30y bp changes (2019-2025, 4.8% of cells
    missing): explained 92% vs 86% complete-case, PC2 turned into a spurious US-vs-UK
    factor, 500 iterations without converging. PPCA-EM: 86.3%, the same loadings as
    complete-case, 4 iterations, and it used 1,809 rows instead of 1,593. On a synthetic
    2-factor panel with 15% of cells missing it recovers the full-data subspace best
    (largest principal angle 0.6 deg vs 1.4 complete-case, 1.5 pairwise; tested).
*   **Cross-market gaps:** keep the panel's NaN (`read_panel(how="outer")`). Never zero-fill
    a closed market's change, never difference across a forward-filled level (a fake 0 then
    a catch-up move). `diff` of a level with a gap leaves the gap day and the reopening day
    missing, which `MissingDataPCA` handles.
*   **Projection with gaps (every class, `pca.project_observed`):** a complete row's scores
    are the classical projection; a row with gaps gets the posterior mean of its scores
    given its OBSERVED entries (prior: each PC's own variance; noise: the mean discarded
    eigenvalue), and `fitted:<col>` also fills the missing columns. So a cross-market model
    keeps producing factors while one market is shut, intraday too. **Not plain least
    squares on the observed entries** - found 2026-10-03: on US-only days a US/DE/UK PCA's
    cross-country factor is barely identified, least squares put it at ~100 sd, and a regime
    HMM fed those scores made a "regime" of those days. The prior shrinks a badly
    identified factor toward 0 instead.
*   **Outputs:** `score:PCi`, `fitted:<col>` (reconstruction from k PCs), `residual:<col>`
    (rich/cheap vs the factor model), `resid_z:<col>`, `n_obs`, `in_sample`. Tables:
    `loadings` (unit eigenvectors), `loadings_units` (a 1-sd move of each PC in each
    column's units - the curve shapes), `explained` (eigenvalue, share, cumulative,
    `above_mp`).
*   **Signs** are fixed (loadings sum positive) and `align_to(prev, permute=)` matches a
    refit to the previous one; `walk_forward` and `fit_path` align automatically.
*   **Noise edge (`noise_edge`, stat `mp_edge`, `n_above_mp`):** Marchenko-Pastur upper
    edge with Kish's effective n. The textbook noise variance (trace/p) counts the signal
    as noise - on the US curve PC1 carries ~90% and pushed the edge above the slope PC. So
    with >= 20 series the noise is the mean of the eigenvalues below the edge, iterated
    (finds exactly 3 factors in a 40-series 3-factor test, noise 0.99 vs 1); with fewer it
    is the mean of the DISCARDED eigenvalues (PPCA's ML noise) - MP is asymptotic and a
    6-tenor curve has no noise bulk. For small p, `diagnostics.parallel_analysis` is the
    better test of how many PCs to keep.
*   **Real data (US CMT 2/3/5/7/10/20/30y bp changes, 2016-2025):** level / slope /
    curvature explain 87.8 / 9.9 / 1.1%; PCA-weighted 2s5s10s fly (long 5y, neutral to
    PC1 and PC2): -0.55 2y, -0.56 10y.

## 6. Known-problem visibility (`diagnostics.py`)
*   **Factor correlation in sub-periods** (`factor_correlations`, by year or rolling): the
    PCs are uncorrelated over the fit sample by construction, not inside sub-periods. US
    curve 2016-2025, one fixed fit: PC1-PC2 correlation +0.68 in 2020, +0.55 in 2021, -0.37
    in 2023 - in those years a "PC-neutral" position was not neutral.
*   **Eigenvector instability** (`eigenvector_stability`): refit on a schedule; per PC
    `cos_prev` / `cos_ref` (|cosine| of the loading vector), `gap` (lambda_i / lambda_{i+1})
    and, for the whole top-k subspace, the largest principal angle. Read them together: a
    PC with a gap near 1 is not identified and can rotate inside its pair while the
    subspace stays put. US curve, quarterly 2y-window refits 2018-2026: median cos to the
    previous fit 0.9998 / 0.9997 / 0.998 for PC1-3, median gaps 10.3 / 7.7 / 2.4.
    `loading_path` gives the loadings over time for charts.
*   **`parallel_analysis`** (Horn, on correlations), **`bootstrap_loadings`**
    (moving-block bootstrap bands), **`pc_neutral_weights`** (hedge/fly weights neutral to
    chosen PCs).
*   **Regression:** `coefficient_stability` (the regression fitted separately per year: a
    Chow view without assuming the break date), `rolling_fit` (coefficient and R^2 paths).

## 7. Regimes and regime-weighted PCA (`hmm.py`, `regimes.py`, `regime_pca.py`; built 2026-10-03)
*   **The use case:** trade PCA residuals of K series where the factor structure depends on
    uncertain regimes inferred from a RICHER set of N series (N > K; K may be inside N).
    `RegimePCA` = a regime model (on N) + one weighted PCA per regime (on K).
*   **Regime models, one interface** (`predict` -> `p_filt:R<r>` known at t, `p_pred:R<r>`
    known before row t's data, `p_smooth:R<r>` whole fit sample, in-sample rows only):
    `HMMRegimes` (Gaussian HMM, EM / Baum-Welch, k-means starts, warm-started from the
    previous refit in a walk-forward, ``sticky`` Dirichlet prior on staying) and
    `RuleRegimes` (threshold rule on an input, hard or logistic-soft, or a given probability
    frame - must be point in time: no "hiking cycle until the last hike"). The HMM core
    (`hmm.forward_backward`) takes any per-row log-likelihood, so a Markov-switching
    regression only needs its own M-step. Pluggable features (`regimes.FEATURE_BUILDERS`):
    `pca_vol` = a correlation PCA of the N series, then daily scores (DEFAULT) and/or
    rolling vols / drift t-stats; `columns` = given inputs.
*   **Why daily scores by default:** synthetic 2-regime panels (vol x3 plus a different
    structure), 4 seeds: daily scores give 98.4-98.8% out-of-sample regime accuracy and
    regime loadings as good as with the true labels; 10-row rolling vols lag each switch
    (94-97%), and the few stressed rows they label calm (9x the variance) rotated the calm
    loadings up to 27 deg (once 77). Drift features once hijacked the split (71%).
*   **Regime labels across refits:** cold start orders calmest first; `align_to` relabels
    each refit by smoothed-probability overlap with the previous one (basis-free).
*   **Fit (`RegimePCA`):** smoothed probabilities weight each fit-sample row (legitimate:
    the fit only holds data to `as_of`); per regime the weighted mean / covariance (complete
    rows or PPCA-EM), shrunk to the pooled by lambda = n_eff / (n_eff + kappa), kappa =
    K(K+1)/2 by default.
*   **Predict:** after the fit date each row uses the PREDICTED probability (timing
    `"predicted"`, default): fixed before the row's own move, so an idiosyncratic move in a
    K series can't shift the regime and the loadings measuring it (pinned by
    `tests/test_regimes.py::test_predicted_timing_ignores_the_rows_own_move`). Combine
    (`combine`): `covariance` (default; blend the regime moments incl. the between-regime
    mean term, PCA of the blend each row, signs anchored to the pooled PCA), `residual`
    (blend each regime's own residual), `robust` (neutral to every regime's factors at
    once). `resid_z` = residual / the model-implied residual sd that day, scaled so the
    fit sample has unit variance. Never argmax switching (phantom moves).
*   **Real data (2026-10-03): K = US 2/5/10/30y bp changes, N = US/DE/UK 2/10/30y (+US 5y),
    2y-window monthly walk-forward 2022-2026.** The HMM (no sticky) finds a persistent
    2022-23 hiking / high-vol regime (91-97% of those years; mean durations 95 / 35 days).
    The regime curve SHAPES differ by only ~5 deg, so residuals are not better (variance
    0.699 vs 0.683 plain PCA) - the gain is the SCALE: residual z sd 0.92 calm / 0.95
    stressed vs plain PCA's 0.80 / 1.20, and false |z|>2 in the stressed regime 3.5% vs
    6.9% (normal 4.6%). Out of sample, a 2-regime daily-score HMM's predicted probability
    can still swing within days (0.15 -> 0.92 over three days in Sep 2026 even at
    sticky=200): use `sticky` (~1000) or `hmm2_vol` when the trade horizon needs a slow
    regime. Expect structure (shape) gains where the factor structure genuinely changes -
    cross-market K, or a curve with a pinned front end.
*   **Synthetic checks (tests):** forward-backward = brute-force enumeration; absorbing
    transitions never give NaN (`hmm.floor_probs`: a refit once inherited an exact-0
    transition and returned NaN); regime loadings within 8 deg of the truth; residual
    variance 0.52 / 0.94 / 0.65 of plain PCA on seeds 0-2; z calibrated in both regimes;
    walk-forward shock test for all three combine modes; params round trip.

## 7a. Covariance conditioned on third variables - framework C (2026-10-06/07)
*   **The user's framework C:** bias the covariance (the PCA) by third variables, trade the residuals.
    Residual trading itself runs through the other frameworks: a stored run's residual is a series
    (`model:<run>:residual:<col>` in `infra.pipeline.series_panel` = that residual's daily P&L), so
    "fade the residual" is an A spec (ex ante, weight -1) or a B1 spec on it.
*   **Three ways to condition, by how fast the state moves:**
    *   **Fast states (daily moves) - `RegimePCA` with the HMM** on the N series' daily factor scores
        (section 7): the regime probability updates daily, so the loadings follow a switch within days.
        Its Gaussian HMM assumes observations independent given the regime - right for daily moves,
        WRONG for slow states: on levels, slope, a 6-month change or an EWMA vol it collapses into
        one wide regime or into eras (found 2026-10-06: one regime held every day from 2015; z-scoring
        the features over 3 years did not help).
    *   **Slow states, discrete - rule regimes, `rule="grid"`** (`RegimeSpec.rule_columns` /
        `rule_grid`): regime = the combination of intervals on several columns (last column varying
        fastest), soft boundaries multiply. E.g. cycle priced (hikes / cuts) x vol above / below its
        3-year norm = 4 states.
    *   **Slow states, continuous - `SimilarityPCA`** (PCA method `similarity`; `PCASpec.state_columns`,
        `bandwidth`): each day weighted by how close its state (z-scored over the fit sample) is to the
        state on the fit date, Gaussian kernel (`bandwidth` in standard deviations, mean squared distance
        across the state series); an incomplete state weighs 0; the loadings are FROZEN until the next
        refit, so a residual stays one portfolio for the month (RegimePCA's blend moves with the day's
        probability: right for fast regimes, but the residual's weights then change daily).
    *   They combine naturally (similarity on the slow state x the HMM probability of the fast regime) -
        not built.
*   **Run inputs can be any feature:** `feat:<expression>` (series_panel) puts a feature-maker
    expression into a run, labelled by the DAY it became available (conservative).
*   **First test (2026-10-06/07; the US on-the-run curve 2/3/5/7/10/30y, bmk otr P&L, 3 factors, monthly
    refits; signal: fade each residual's 20-day sum, z over a year, 1 / 5 / 20 days; scratch runs
    `c_*` in `ModelRuns`):**
    *   Plain PCA residuals revert at 1-5 days, belly strongest (2013-2026 Sharpe 0.4-0.8 at 1 day,
        spanning t ~2 at 3y / 5y); nothing at 20 days; strong in 2010-14, ~0.2 since 2015. Gross, and
        part of the 1-day reversion is likely noise in the end-of-day marks.
    *   HMM on the curve's own scores (K = N): residuals 0.98-0.995 correlated with plain, no gain.
    *   The user's slow states (front-end shape = SR1 12 months out minus SR1 now - hikes / cuts priced;
        10y EWMA vol; 10y level; 2s10s): rule grid (cycle sign x vol) and similarity (bandwidth 0.5 / 1.0),
        on 2019-09.. (the SR1 strip's history), fades evaluated 2021-10..2026-09 against a plain PCA on
        the same sample: NO gain - the front end WORSE (2y / 3y incremental t -1.8 to -3.0 at 1 day), no
        tenor with incremental t >= 2. Likely: the plain fade barely works in this window (the front end
        trended with policy), and conditioning costs effective sample (15-40% at bandwidth 0.5), adding
        loading noise that doesn't revert. Short sample: a 2012+ test needs a pre-2018 front-end series
        (`TOFIX.md`).

## 8. Running it
```python
from infra.pipeline.series_panel import read_panel
from infra.models.stats.regression import make_regression
from infra.models.stats.pca import make_pca
from infra.models.stats.common import pick
from infra.models.walk_forward import walk_forward

ids = ["bond:US_BOND_10y", "bond:US_BOND_2y", "bond:US_BOND_30y"]
panel = read_panel(ids, "2020-01-01", "2026-09-30")
m = make_regression("ols_hac", y=ids[0], x=tuple(ids[1:]), prep=("diff", "mult:100"))
d = m.prepare(panel); m.fit(d, as_of="2026-06-30"); print(m.summary())
out = m.predict(d, start="2026-06-30")            # frozen fit, later rows
res = walk_forward(lambda: make_regression("hedge_ratio", y=ids[0], x=tuple(ids[1:])),
                   panel, "2023-01-01", "2026-09-30", refit="W-FRI")
```
`scripts/run_walk_forward.py --list` shows every named spec; `regime_pca` runs take
`--series` (K), `--regime-series` (N) and `--regime-set key=value`.
