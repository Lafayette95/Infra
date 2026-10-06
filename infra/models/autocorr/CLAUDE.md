# infra/models/autocorr - Conditional autocorrelation, framework B1 (additive to root and `infra/models` CLAUDE.md)

Everything in the root `CLAUDE.md` and `infra/models/CLAUDE.md` applies (point in time, UTC,
prepare -> fit -> predict, the operating model of 0b, named specs). Known gaps: the ONE root
`TOFIX.md`, headings "Autocorr: ...". Built 2026-10-06 from the user's spec.

## 1. The question
*   **Does a third variable X decide whether an instrument's price action continues (chase) or
    reverts (fade)?** Framework B1 of the user's four (A direction, B autocorrelation, C
    covariance, D time/catalyst). Model: `r_fwd = a + b r_past + c X + d (X x r_past)`; `b` is the
    unconditional momentum / reversal (the BENCHMARK), `c` X's directional effect (framework A, not
    this), `d` the B1 effect.
*   **One-dimensional form:** `chase = sign(past) x fwd` is the P&L of chasing the past move. The
    benchmark says E[chase] is a constant; B1 says E[chase | X] varies with X. Testing "does X
    predict chase" removes most of X's directional effect automatically (chase is signed by the
    past move, not by X).
*   **Buckets AND regression (user spec):** X in terciles (point-in-time trailing partition) is the
    nonparametric version - per-bucket mean chase, top-minus-bottom spread; the continuous
    regression `chase ~ X` is the parametric cross-check. Terciles rather than quadrants: the
    middle bucket absorbs the sign-at-zero noise.

## 2. The model (`model.py`, `config.py`)
*   **Features come from the central feature maker** (`infra.processing.features`, root CLAUDE.md 31): `x_feature` takes its grammar (`chg:20|norm:vol:60|abs`) or the legacy names `move:K` / `absmove:K` / `level` / `z:W` (aliases, outputs identical - tested); the past move is `chg:k|norm:vol:<vol_span>`.
*   **Inputs:** any two series ids (`infra.pipeline.series_panel`), the target as DAILY moves
    (bp, + = a long position made money - e.g. `bmk:curve:US_BOND_10y`, the persisted benchmark
    P&L, or a yield structure `bmk:curve:FLY__US_BOND_5y__US_BOND_7y__US_BOND_10y`), X as daily
    moves (`x_is_moves`) or a level.
*   **Prepare** (per information day D, data known at D): `past` = k-day move / (EWMA vol x sqrt k);
    `fwd` = the h-day move over the window starting `gap_days` after D / (vol x sqrt h) - `gap_days`
    1 by default, since FedInvest-based yields for D post ~10:00 New York on D+1; `chase`; X's
    feature (`x_feature`: `move:K` = X's K-day move / its vol, `level`, `z:W`) and its bucket
    (`x_partition`, default `rolling_tercile:504`, from `infra.models.event_study.conditions`);
    controls `vol_z` (vol against its trailing year) and `abs_past`.
*   **Fit(as_of)** uses only rows whose forward window ENDED before `as_of`'s day. Statistics, HAC
    errors with lags = h: `bench_mean` / `bench_t`; `x_coef` / `x_t` (chase ~ X standardised in the
    fit sample); per-bucket means and counts, `spread` / `spread_t` (top minus bottom); `ctrl_t`
    (X's t with `controls` - vol regime, |past| - in the regression: X must add beyond "it was a
    volatile period"); `halves` (both halves of the sample agree on the spread's sign).
*   **Gates (`gates.py`, `GATES` registry; `AutocorrSpec.gates` + `thresholds`, swappable):**
    `min_obs`, `x_t`, `bucket_spread_t`, `beats_controls`, `halves_agree`, `monotonic`. All named
    gates must pass for the conditioning to be used; then the signal map is chase (+1) / fade (-1)
    per bucket by the sign of its mean. A failed fit `fallback`s to flat or to the benchmark's own
    rule. Named specs: `default` (strict, |t| >= 2 on X, spread and controls), `loose` (1.5, no
    controls gate), `none` (min_obs only - the conditional rule always: research).
*   **Predict:** `signal = sign(past) x map[bucket]`, `bench_signal`, `chase_or_fade`,
    `expected_bp` over the forward window. Run kind `autocorr` (`infra.models.runs`): weekly
    fit-append, daily predict-append, rebuild - incremental == rebuild exactly (tested).

## 3. Does X add value? The evaluation suite (`evaluate.py`, `EVALUATIONS`, swappable)
*   P&L is a STAGGERED daily book (each day's signal holds 1/h of the position over its own
    forward window, in units of the target's vol) - a real daily series, not overlapping sums.
*   `benchmark`: Sharpe of the conditional and the benchmark books; `clark_west`: out-of-sample R2
    of the conditional forecast of chase against the benchmark's, and the Clark-West t (nested
    models - plain MSE / Diebold-Mariano is biased against the larger model); `spanning`:
    conditional P&L ~ benchmark P&L (HAC) - the INTERCEPT is what X adds, the slope how much is
    the same bet; `sharpe_diff`: block-bootstrap of the Sharpe difference; `subperiods`;
    placebos run through the same walk-forward - `permutation` (X's raw series in blocks of
    `placebo_block_days`, keeping its persistence), `synthetic_ar1` (an AR(1) with X's
    persistence and scale: persistent regressors find spurious structure), `time_shift` (X a year
    back: same properties, no timing information); p = share of placebos whose spanning t is at
    least the real one. A conditional book that never traded counts as alpha 0.
*   **How to decide** (user discussion 2026-10-06): X adds value if the interaction is
    significant in sample with the controls, beats the benchmark out of sample (Clark-West and a
    positive spanning intercept, net of the extra turnover), sits in the tail of the placebos,
    and holds in most sub-periods and on more than one instrument.

## 3a. Families and the false-discovery gate (`family.py`, built 2026-10-06)
*   **A family = a grid declared up front** (`AutocorrFamilySpec`, registry `AUTOCORR_FAMILIES`):
    (target, X) pairs x X features x past horizons x forward horizons over one base spec. Every
    member is walked forward with its conditioning always computed; its own gates are re-applied
    from its stored statistics.
*   **The `fdr` gate, point in time:** at each fit date, Benjamini-Hochberg across ALL members'
    p-values of `fdr_stat` (default `x_t`) -> q; a member trades on that fit only if its own gates
    pass AND q <= `fdr_q` (0.10). It lives at the family level (it needs every member's fit), like
    the event families' FDR table.
*   **Scope:** the correction covers the family it's given. A new X is a new family and "resets" -
    so keep every family's per-fit p-values (`run_family`'s `fits`) and periodically correct
    across families together (`global_q`). Online FDR (LORD / alpha-investing) is the principled
    version once many families exist (`TOFIX.md`).

## 4. Test case: duration vs flies, both ways (2026-10-06)
*   **Setup:** our curve's yield P&L (`bmk:curve:...`), duration = `US_BOND_10y`, flies 2s5s10,
    5s7s10, 2s10s30, 5s10s30; past 5 days, forward 5, gap 1; X = the other instrument's 20-day
    move (`move:20`); monthly walk-forward 2012-2026 (fits from 2008-09). Spec `none` (always
    conditional) with the full suite and 50 placebos each; `loose` and `default` gated.
*   **Result: X adds nothing, either way, on any fly.** Spanning t -2.0 .. +0.4, Clark-West t
    -2.6 .. +0.1, out-of-sample R2 below the benchmark's everywhere; placebo X do as well or better
    (permutation p 0.34-0.92, AR(1) p 0.36-0.96; X shifted a year back scored HIGHER than the real X
    on the three duration-conditioned flies 2s5s10, 5s7s10, 5s10s30). The strict gates passed on 0-0.6% of fits, the loose
    ones only on 5s7s10 given duration (37% of fits) - and lost there (Sharpe -0.07). The
    framework rejecting a weak X is the intended behaviour.
*   **Side finding, the benchmarks themselves:** the walk-forward unconditional rule (chase or fade
    by the fitted sign) earned Sharpe +0.40 on 5s7s10 (it chases) and +0.25 on 2s5s10, but -0.48
    on 2s10s30 and -0.29 on 5s10s30 - their unconditional sign flipped out of sample. Duration
    itself: +0.09.
*   **The grid (family `duration_flies`, 216 members: 8 pairs x X features move:5/20/60 x past
    1/5/20 x forward 1/5/20 days; 178 monthly fits 2012-2026; ~3 minutes):**
    *   raw p < 0.05 on **10.4 members per fit - exactly the 10.8 expected by chance**;
    *   own (strict) gates alone: 5.5 members per fit; the 47 members that ever traded had a mean
        spanning t of -0.30, none beyond +-2;
    *   FDR-passed: 0.49 per fit, on 13 of 178 fit dates, ALL in 2012-2013 (and one in 2018) -
        long-horizon members (k20 / h20) fitted on short early samples, where HAC errors with 20
        lags on few independent observations over-reject. None since 2014; the 11 members that ever
        passed made money in 2 cases (spanning t +0.3, +0.8) and lost in the rest.
    *   **Conclusion: no conditional autocorrelation between 10y duration and these flies at any
        tested horizon.** The family-level correction did its job: it would have prevented trading
        ~10 chance discoveries a month.

## 4b. Second round: the user's quadrant spec and |X| (2026-10-06)
*   **Why:** the user knew the effect empirically, so we asked what OUR test could miss: (1) `chase`
    is symmetric in the past move's sign, so an asymmetric effect (a fly that co-moved with duration
    reverts, one that counter-moved continues) cancels inside an X bucket; (2) the x_t / spread tests
    only see monotonic dependence, missing "after a LARGE duration move either way".
*   **Built:** `mode="cells"` - the 3x3 of X bucket x the target's own past-move bucket, a signal per
    cell (the user's quadrant spec), the family's FDR over CELLS, spanning against BOTH benchmarks
    jointly (own momentum rule, X's direction-only rule `xdir_signal`); `x_feature="absmove:K"`
    for |X|. Specs `cells`, `cells_loose`; families `duration_flies_cells` (288 members x 9 cells) and
    `duration_flies_abs` (216), on the ON-THE-RUN P&L (away from fitted-curve noise).
*   **A mistake of ours, found and fixed:** the first cell test's null was "cell mean = 0", so a
    target that merely TRENDED (duration 2012-2020) made every cell look significant: 234 raw
    p < 0.05 a fit (130 by chance), 52 FDR 'discoveries' a fit - none of which added anything out of
    sample. The null is now the target's unconditional mean (or its X row's, `demean_by_x`); the
    traded sign stays the cell's raw mean (`cell_mean`), significance is its excess (`cell_excess`).
*   **Results (178 monthly fits, 2012-2026):**
    *   `duration_flies_abs`: raw p < 0.05 on 7.7 members a fit (10.8 by chance), ZERO FDR discoveries.
    *   `duration_flies_cells` (corrected null): raw p < 0.05 on 79.5 cells a fit (130 by chance),
        0.33 FDR discoveries a fit, only 2012-2016; the 6 members that ever passed had a mean joint
        spanning t of -0.64, none above +0.44; own gates alone: 219 members traded, mean t -0.24.
    *   Cross-family (`global_q` over the three families, ~3,000 tests a fit): nothing robust.
*   **Conclusion: on these instruments, horizons and features, neither the symmetric, the
    asymmetric (cell), nor the |X| version of duration-conditioned fly autocorrelation (or the
    reverse) exists out of sample.** If the effect is known empirically, the remaining suspects are
    the ones we did NOT test: other flies or curves, intraday horizons (the lead-lag of a stale belly
    is real at minutes-to-hours), a different X (positioning, vol, risk aversion), or a sample
    ending around 2015 (several curve effects were 3-5x stronger in 2009-2013).

## 4c. No-fit rules (`fixed_map`; the user's prior, 2026-10-06)
*   **`AutocorrSpec.fixed_map`:** a pre-stated signal per X bucket, no fitting and no gates (the fit
    only reports statistics) - like the fixed 5s30s auction rule. Specs `fade_high_chase_low` (fade
    the target's move when X is in its + tercile, chase in the - tercile, flat in the middle) and its
    mirror `chase_high_fade_low` (exactly the negative P&L).
*   **Run** (the user's recollection of a known empirical effect, so no false-discovery step): 5-day
    past / forward, X = its own 5-day move, 8 pairs (4 flies, both roles), on-the-run and curve P&L,
    2012-2026, 50 placebos each on-the-run.
*   **Result:** fly given duration leans to the MIRROR (chase the fly after duration RALLIES, fade it
    after sell-offs): positive on every fly on-the-run, strongest on **5s7s10: Sharpe +0.48 (t 1.8),
    positive in all four sub-periods, 11 of 15 years, the real duration beating ~98% of placebo X** -
    but only +0.07 on our curve (source-dependent: on-the-run 5y/7y/10y specifics, or chance in one
    of 16 tries), and much of it is X's DIRECTION (duration's move alone predicts 5s7s10's direction:
    Sharpe 0.58; beyond it and the fly's own momentum the rule adds a joint t ~ +0.75). Duration
    given the fly leans to `fade_high_chase_low` on all 8 (+0.04 .. +0.24), not significant.

## 6. Running it
```python
from infra.models.autocorr.evaluate import evaluate
from infra.pipeline.series_panel import read_panel
T, X = "bmk:curve:FLY__US_BOND_5y__US_BOND_7y__US_BOND_10y", "bmk:curve:US_BOND_10y"
r = evaluate("none", read_panel([T, X], "2008-09-01", "2026-09-30"), "2012-01-03", "2026-09-30", target=T, x=X)
```
