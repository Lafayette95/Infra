# infra/models/meanrev - Mean reversion of factor residuals, cross-sectional (additive to root and `infra/models` CLAUDE.md)

Everything in the root `CLAUDE.md` and `infra/models/CLAUDE.md` applies (point in time, UTC,
prepare -> fit -> predict, the operating model of 0b, named specs, fit modes of 0c). Known gaps:
the ONE root `TOFIX.md`, headings "Meanrev: ...". Built 2026-10-08 (user design discussion
2026-10-07: framework C's trading part, an add-on to PCA residuals, with the OU trade metrics of a
fixed-income RV mean-reversion chapter).

## 1. What it does
*   **Wraps a factor model and owns its fit** (`MeanRevSpec.factor` = `pca` - plain, weighted,
    `similarity` - or `regime_pca`; `factor_spec` + `factor_overrides`; `columns` = the K
    instruments, daily moves in bp; `regime_columns` = the regime model's N, or the similarity
    states). One model instance covers ALL K residuals of the fit: cross-sectional.
*   **At each fit date:**
    1.  fit the factor model on data up to the date;
    2.  RE-APPLY that fit to the last `window` rows and cumulate each residual into a LEVEL from the
        window's first row (the current loadings applied to the recent past - point in time; the
        stored daily residuals of a run would mix portfolios across refits);
    3.  read each residual's portfolio weights in the K instruments off a regression of its daily
        residual on the instruments' moves over the window (exact for a constant-loading PCA,
        `r2_proj` = 1; below 1 for a regime blend whose loadings move by day);
    4.  an OU / AR(1) on each level (`infra.analytics.ou`; Kendall bias correction by default),
        per residual or `pooling="pooled"` (one b, own mean and vol each), with gates per residual
        (`min_obs`, `reverting` = 0 < b < 1, `adf_t` = (b - 1) / se(b) <= -threshold, `half_life` in a
        band, `halves` = both halves of the window reverting).
*   **Fit modes (`infra/models/CLAUDE.md` 0c; the stated direction is REVERSION):** `exante` = fade
    the level's z over the window, no OU, no gates; `fitted` = the OU s-score where every gate
    passes; `prior` = the OU s-score wherever the residual reverts at all (b < 1) - a residual
    estimated as trending is switched off, never chased.
*   **Signal per residual:** `linear` = -s / `entry` clipped to [-1, 1], or `threshold` (open at |s| >=
    `entry`, close at |s| <= `exit` or when s crosses the mean); optional `cross_section="demean"`;
    sized by 1 / the residual's daily sd (`risk_scale`). **Positions are NETTED into the instruments**
    (`pos:<instrument>` = sum of residual sizes x their weights): the K residuals of a k-factor fit
    span only K - k independent directions, so trading them as K separate bets would double count.
    The netted book is neutral to the fit's factors by construction.
*   **Predict continues each level from the fit date** (the stored `level_end` + out-of-sample
    residuals; the threshold rule from the stored `state_end`): a prediction depends on the stored
    params and post-fit data only. Found 2026-10-08 while making the run reconcile: a regime PCA
    rebuilt from its params does NOT reproduce its in-sample regime path (so neither its in-sample
    residuals), only the out-of-sample one - recomputing the window in predict broke
    incremental == rebuild (`TOFIX.md`).

## 2. The OU trade metrics (`infra/analytics/ou.py`, pure; per residual and row in `predict`)
*   In STANDARDISED units (s-score z = (X - m) / sd_eq, time in units of 1 / kappa; dz = -z dt +
    sqrt(2) dW), where most metrics depend on |z| only and are tabulated once:
    *   `s:` the s-score; `z:` the level's z over the window; `level:` (bp);
    *   `exp_move:` / `sd_h:` / `sharpe_h:` over `horizon_days` (closed form, exact for the AR(1));
    *   `fpt_median:` (closed form: the OU is a time-changed Brownian motion, so hitting the mean is
        Levy's first-passage law in the clock tau(t) = e^{2t} - 1) and `fpt_mean:` (scale / speed
        densities) - days to the mean;
    *   `p_target:` P(reaching |s| = `exit` before the stop at |s| + `stop_width`) (scale function)
        and `trade_days:` the expected time to either (Green's function);
    *   per fit, `opt_entry`: the entry s-score maximising the expected return per unit time of the
        cycle "wait for the entry, hold to the opposite extreme" at the given cost (Bertram 2010);
        with no cost it is ~0 (trade as often as possible), costs push it out.
*   Every formula is checked against Monte Carlo of the exact discretisation with a Brownian-bridge
    crossing correction (`tests/test_ou.py`; without the correction the discrete simulation detects
    crossings late and looked 2-6% off).

## 3. First real test (2026-10-08; US on-the-run curve 2/3/5/7/10/30y bmk P&L, 3 factors, monthly refits)
*   **2013-2026 (gross, netted book):** best `prior` with a 250-row window, Sharpe 0.71 (plain PCA)
    / 0.59 (HMM regime PCA); `fitted` (every gate) 0.14-0.19; `exante` 0.0-0.16; `threshold` ~0;
    `pooled` ~0.1. Sub-periods (window 120): all of it is 2010-2019 (prior 1.42 in 2015-19), then
    flat to negative from 2020.
*   **The gates select the wrong residuals:** `fitted` keeps ~1/3 of residual-fits, with a median
    half-life of 4.5 days, against 12-14 days for all of them - the ADF gate favours the fastest
    reverting levels, which are likely the mark noise (end-of-day prices), not the tradeable
    dislocations. `prior` (every reverting residual) does better.
*   **2021-10..2026-09 (the similarity sample):** everything ~0 or negative (best prior +0.18); the
    similarity- and regime-weighted factors add nothing (as with the simple fades, stats CLAUDE.md 7a).
*   **Window 60 trades nothing** under the gated specs: 59 transitions fall short of `min_obs` 60.
*   Not yet: costs (the book's turnover is 15-35% of gross a day), the off-the-run CUSIP universe
    (a wider cross-section, the natural next test), a placebo on real data.

## 4. Running it
```python
from infra.models.meanrev.evaluate import evaluate
K = tuple(f"bmk:otr:US_BOND_{t}y" for t in (2, 3, 5, 7, 10, 30))
panel = read_panel(list(K), "2009-03-02", "2026-09-30").dropna()
evaluate("prior", panel, "2012-01-03", "2026-09-30", columns=K, window=250)
```
Run kind `meanrev` (`series` = the K instruments, `regime_series` = the N / states; overrides as the
spec's fields, `factor_overrides` as a dict) - incremental == rebuild exactly for a plain and a
regime-PCA factor (`tests/test_model_runs.py`).
