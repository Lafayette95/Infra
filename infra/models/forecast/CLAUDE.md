# infra/models/forecast - Directional forecasts, framework A (additive to root and `infra/models` CLAUDE.md)

Everything in the root `CLAUDE.md` and `infra/models/CLAUDE.md` applies (point in time, UTC,
prepare -> fit -> predict, the operating model of 0b, named specs, the feature maker of root 31).
Known gaps: the ONE root `TOFIX.md`, headings "Forecast: ...". Built 2026-10-06 from the user's spec.

## 1. The question
*   **Do point-in-time features predict an instrument's forward move?** Framework A of the user's
    four (A direction, B autocorrelation, C covariance, D time/catalyst). Time-series first; the
    cross-sectional variant is not built.
*   **Three modes, one model** (user decision 2026-10-06), as with every framework here:
    *   `exante`: no fit - forecast = sum of weight x feature, ANY feature string of the feature
        maker with a stated direction (a sign, a weight); the fit only reports statistics. Honest
        only if the direction comes from outside the data.
    *   `fitted`: walk-forward regression (default OLS with HAC errors, user decision), used when its
        gates pass, else flat. Adapts, but a weak effect refitted monthly flip-flops (the B1 lesson).
    *   `prior`: the SIGN is fixed ex ante (the weights' signs), the data only sizes it (ridge,
        `prior_lambda`); a coefficient on the wrong side is set to 0 - the robustness of the
        ex-ante rule, scaled by the data, never flipped by a noisy month.

## 2. The model (`model.py`, `config.py`)
*   **Rows are the target's days D; the decision is taken on day D + `gap_days` at
    `decision_time` New York** (default 15:00, before the 15:30 cash marks). Every regressor is the
    feature maker's value AVAILABLE by that instant (`infra.pipeline.features.align`, with the
    regressor's own carry `limit` and `fill` - a surprise is carried a few days, then 0 = "no news").
    The forward move is the target's sum over the `horizon_days` closes after the decision day's
    close, / (EWMA vol at D x sqrt h). `gap_days` 1 by default: the target's own day-D move (and so
    the momentum benchmark) is public only on D+1.
*   **Fit(as_of)** uses rows whose forward window ended before `as_of`'s day: OLS (or ridge) of the
    forward move on [1, regressors], HAC lags = h, always computed for the statistics (`coef_*`,
    `t_*`, `r2`, `drift`); the mode picks the coefficients. The intercept (drift) never enters the
    forecast: it is a benchmark.
*   **Gates** (`GATES`: `min_obs`, `coef_t` - every regressor's |t| -, `r2`), swappable per spec.
*   **Predict:** `forecast` (vol units), `signal` = forecast / (2 x its in-sample sd) clipped to
    [-1, 1], `expected_bp`, and the benchmarks' signals `drift_signal`, `mom_signal`.
*   **Regressors** (`Regressor`: a feature expression, a weight, a carry limit, a fill, a label);
    helper `surprise(ticker, weight)` = `surprise:<ticker> | lvl | norm:z0:36 | clip:3`, carried
    4 days, then 0.
*   **Run kind `forecast`** (`infra.models.runs`; regressors come from the NAMED spec, since they
    don't serialise into a run's JSON) - incremental == rebuild exactly (tested).

## 3. Evaluation and families (`evaluate.py`)
*   Staggered daily books (B1's `staggered_pnl`); `benchmark` (Sharpe of the forecast, drift and
    own-momentum books), `clark_west` (out-of-sample R2 vs the drift forecast and the Clark-West
    t), `spanning` (the forecast book ~ drift + momentum books jointly: the intercept is what the
    regressors add), `subperiods`, placebos `permutation` (regressor columns block-permuted) and
    `time_shift` (a year back).
*   `run_family(members, panel_of, ...)`: single-regressor fitted members, Benjamini-Hochberg across
    members per fit date (point in time), each traded only where its own gates pass AND q <= 0.10.
    Expanding-window fits are strongly autocorrelated, so a chance early significance persists for
    many fits - read a family's discoveries with that in mind.

## 4. First test case: macro surprises against duration (2026-10-06)
*   **Setup:** target the 10y on-the-run yield P&L (`bmk:otr:US_BOND_10y`, + = rally); regressor each
    release's surprise (`surprise(ticker)`: z on its own trailing 36-print RMS, clipped at 3, carried 4
    days, then 0) for the 14 releases with 100+ consensus prints (NFP, unemployment, claims, ISM mfg /
    services, Philly, Empire, UMich, Conference Board, durables, IP, housing starts, existing home
    sales, GDP); decisions 15:00 New York, so on release day the position comes AFTER the morning's
    reaction - this tests POST-release drift (slow absorption); horizons 1 / 5 / 20 days; walk-forward
    2012-2026, monthly refits.
*   **Ex-ante** (strong data -> duration keeps selling off, weight = -sign): mean Sharpe across releases
    -0.05 / -0.04 / +0.02 (1 / 5 / 20 days), half positive; one (existing home sales, 20 days, Sharpe
    0.51, spanning t 2.0) of 42 clears t 2 - chance-sized. **The composite surprise index** (all 14):
    Sharpe -0.28 / +0.04 / +0.07, spanning t -1.1 / +0.2 / +0.3, permutation p 0.84 / 0.42 / 0.34: no
    post-release drift.
*   **Prior** (that sign fixed, size fitted): worse (mean -0.14 / -0.09 / -0.03), as expected when the
    stated sign is mostly wrong or absent - the sign constraint switches most releases off.
*   **NFP and GDP point the OTHER way:** the ex-ante rule loses on NFP at every horizon (-0.19 / -0.37 /
    -0.37) and GDP (-0.41 / -0.33 / -0.48) - after a strong print duration tends to RALLY over the next
    days, i.e. the release-day move partly REVERSES (over-reaction). Selection caveat: 2 of 14 picked
    after looking; a candidate for the event-study / autocorrelation frameworks (the release-day move
    and its reversal), not a finding.
*   **Fitted family** (42 members, FDR per fit date): raw p < 0.05 on 5.0 members a fit (2.1 by chance),
    FDR-passed 2.1 a fit on 100 of 178 fits - but the gated books add little out of sample (best
    spanning t 1.2, durable goods at 5 days): the persistent early significances of expanding-window
    fits, not a tradeable effect.

## 5. Running it
```python
from infra.models.forecast.config import get_forecast_spec, surprise
from infra.models.forecast.model import ForecastModel
from infra.models.forecast.evaluate import evaluate
spec = get_forecast_spec("exante", target="bmk:otr:US_BOND_10y", horizon_days=5,
                         regressors=(surprise("NFP TCH Index", weight=-1),))
panel = ForecastModel(spec).read_panel("2009-01-01", "2026-09-30")
evaluate(spec, panel, "2012-01-03", "2026-09-30")
```
