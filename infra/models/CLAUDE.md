# infra/models - Models: Nowcasting, Inflation, CTA, Basis, Stats (additive to the root CLAUDE.md)

Everything in the root `CLAUDE.md` still applies (conda env, modularity, one-way
dependencies, point-in-time `as_of`, `TOFIX.md`, UTC). This file only adds what is
specific to the models sub-project.

## 0. How a model is built: prepare -> fit -> predict (all models, from 2026-10-02)
*   **Each model is a class following `infra.models.base.Model`** (user decision
    2026-10-02), sklearn style:
    1.  `prepare(raw)`: inputs -> the model's aligned, cleaned, standardised form. Pure, no
        fitting. The SAME call prepares the fit sample and new data.
    2.  `fit(prepared, as_of)`: estimate the slow parameters from data up to `as_of` only,
        keep them plus the state `predict` continues from; returns `self` (fitted attributes
        end in `_`). The heavy step, meant to run on a schedule (daily/weekly).
    3.  `predict(prepared)`: the light step on new data only, parameters held fixed -
        possibly at another granularity (an intraday price on a daily fit).
*   **Parameters live in a config registry** (a spec dataclass per parametrisation, toggled
    by name, e.g. `CTA_MODELS`), apart from the code; **inputs are abstract**: reading stored
    data is a separate `inputs` module per model.
*   **Shared machinery for every model (2026-10-03):** `infra/models/prep.py` (string prep
    steps: stateless ones in `prepare`, stateful scaling fitted in `fit` and frozen - so
    resample / diff / normalise is a helper of ANY model, not of one) and
    `infra/models/walk_forward.py` (the point-in-time refit loop, below). `Model.params()`
    (optional) exports the fit as a tidy `section, row, col, value` frame.
*   **The nowcast predates this** (functions `estimate` = fit, `nowcast`/`news` = predict,
    `panel.py` = prepare). It maps onto the pattern; it was not refactored.
*   **Operating model:** section 0b (weekly fit-append, daily predict-append, periodic
    rebuild + reconciliation). **Scheduling is open:** the daily cycle may not import
    `infra/models`, so the jobs need their own runner (`TOFIX.md`).

## 0a. The two use cases every model must serve (user requirement, 2026-10-03)
The SAME classes, specs and parameters serve both; never a second implementation for one
of them.
*   **(a) Point-in-time continuous refit for a strategy - the important one.** A backtest
    refits on a schedule (say every Friday) on data up to that date ONLY, saves the
    parameters as of that refit, and uses that frozen fit to compute the following week's
    outputs (fitted values, residuals, factors, signals). Stitched, that is the history as
    a live user would have seen it. Generic loop: `infra.models.walk_forward.walk_forward`
    (any model with `prepare` / `fit(as_of)` / `params()` / `predict(start=, end=)`;
    the CTA keeps its own because its predict continues a recursive state). Rules that
    make it safe: every stateless prep step is trailing; stateful scaling is fitted inside
    `fit`; `fit` takes only rows known by `as_of`; outputs after the fit date are flagged
    `in_sample=False`; `from_params` rebuilds any stored fit. Saved runs:
    `scripts/run_walk_forward.py --save NAME` -> `Database/Derived/ModelRuns/NAME`.
    A new model must pass the shock test in
    `tests/test_stats_models.py::test_walk_forward_is_point_in_time` (change every value
    after T; nothing up to T may move).
*   **(b) On-demand interactive use in Dash.** Select the input series, the model and its
    parameters, a fit date, and see the fit and its out-of-sample behaviour (the `/models`
    page for the statistical models, `infra/models/stats/CLAUDE.md` 2). The page builds
    the same spec objects a walk-forward uses.

## 0b. Operating a model: weekly fit-append, daily predict-append, periodic rebuild (user decision 2026-10-03)
The operating model for EVERY model (root CLAUDE.md 3b states it; this is the full
specification). Use case (a) of 0a, run live.

*   **A run** = `infra.models.runs.RunConfig`: kind (`regression` | `pca` | `regime_pca`;
    new families register here), spec name + overrides, input series (`regime_series` for
    regime models), first refit date `start`, refit rule (`W-FRI`, `ME`, or every N rows),
    `history_start` (the first day ever read - preparation always starts here). Stored as
    `Database/Derived/ModelRuns/<run>/meta.json`, with `params.parquet` (one block per fit
    date, `fit_as_of`) and `predictions.parquet` (one row per timestamp, `fit_as_of` = the
    fit behind it).
*   **The three jobs** (`infra/jobs/model_runs.py`, CLI `scripts/model_run.py`; event families: `infra/jobs/family_runs.py`, root CLAUDE.md 27):
    1.  `run` = the daily job: predict the new rows, append any due fit, then re-predict
        the rows after a fit it just appended - so the run is walk-forward-consistent
        whenever the job ends. (`fit` / `predict` do one half each.)
    2.  Fit-append: `runs.fit_due` fits each due refit date after the last stored (or
        failed) one, in order, warm-started / aligned from the previous fit rebuilt from its
        params. A failed fit is recorded in `meta.json` and not retried (the walk-forward
        skips it too).
    3.  `rebuild [--promote]`: the full walk-forward into `<run>/_rebuild/` and a
        reconciliation report against the stored run; `--promote` archives the stored run
        to `<run>/_archive/<timestamp>/` and replaces it. A new run's history is built this
        way (one pass), not by a first `run` fitting hundreds of dates.
*   **The invariant: incremental == full walk-forward, exactly.** Then a rebuild's
    differences are DATA REVISIONS or BUGS, never noise. `runs.reconcile` reports fit dates
    on one side only, per params section the largest difference and the FIRST fit date that
    differs (a regime chain diverging after a revision shows as one date and everything
    after it), and per prediction column the largest difference and the first differing
    row. Tests: `tests/test_model_runs.py` (OLS, Kalman, PCA, regime PCA; daily schedule
    and a catch-up schedule that skips days); real data 2026-10-03: a US-curve regime PCA
    backfilled to 2026-08-31, run on 5 scattered days to 2026-09-30, reconciled identical
    with a fresh rebuild.
*   **Rules that keep the invariant (each found or pinned 2026-10-03):**
    *   **A refit target is due only once data exists on/after it**
        (`walk_forward.refit_dates`): otherwise a not-yet-loaded Friday would be snapped to
        Thursday, and a later rebuild (seeing Friday) would disagree. A genuine holiday is
        snapped back once a later row exists.
    *   **Which fit a row uses:** the latest fit dated strictly BEFORE it. The daily predict
        recomputes from the earliest of: the last fit (so a revised day this week shows up
        as a "changed" row - logged), the last stored row, and the first stored row whose
        fit is no longer the right one (a catch-up run appends several fits after
        predicting those days with the old one - found when only the last new fit's rows
        were being redone).
    *   **Row-local outputs:** a row's output depends only on its own data, the history
        before it, and its fit - never on statistics of the batch it was predicted in (the
        regime PCA's projection of rows with gaps once used a prior averaged over the batch;
        `tests/test_model_runs.py::test_regime_pca_rows_with_gaps_do_not_depend_on_the_batch`).
    *   **Preparation from the fixed history start**, every time: diffs, EWM z-scores
        (unbounded memory) and resampling must give the same values as in the full run.
    *   **The previous fit is rebuilt exactly** from its params (`from_params`; regime
        models also `restore_path`, which recomputes the fit's own regime path from its
        params: needed to align the next fit's regime labels and to warm-start it).
*   **Per model: what the weekly fit and the daily predict carry.**

    | Model | Fit (weekly) | Predict (daily) | Across refits |
    |---|---|---|---|
    | OLS/WLS, Huber, quantile, TLS, logit/probit, hockey, ridge with fixed alpha | window only | stateless: params + the row | independent |
    | ridge/lasso/elastic net with alpha by CV, stepwise | window only | stateless | independent, but discrete choices (alpha grid point, selected regressors) can JUMP after a tiny revision |
    | Kalman regression | MLE of (q, r) on the window, warm-started from the previous fit's (q, r): 55 vs 90 likelihood evaluations, same result to 1e-5 | STATEFUL: filters forward from the fit's final state; the daily job re-filters the few rows since the fit (deterministic - no daily state stored) | optimiser start only (tolerance-level) |
    | PCA (plain / weighted / missing-data) | window only | stateless projection | sign alignment only (score signs, never residuals) |
    | HMM regimes, regime PCA | EM, warm-started from the previous fit (`RegimeSpec.warm_start`) | STATEFUL: forward filter from the fit's final filtered probability; re-filtered daily from the fit, like Kalman | PATH-DEPENDENT (below) |
    | Event study | the tests on events ENDED by the fit date | stateless: each new event's window P&L and the fitted signal; a window still open when stored is recomputed by the next predict (`last_predict_through`) | independent |
    | CTA | its own `walk_forward` (recursive state) | - | not on this machinery yet |

*   **Markov / HMM state, the two kinds:**
    *   *Within predict* (today's probability depends on yesterday's): no state is stored
        day to day - the fit holds the filter state at its date, and each daily job
        re-filters the rows since. Deterministic, so it reconciles exactly.
    *   *Across fits*: a warm-started EM fit depends on the previous fit (its starting
        point, and the label alignment), so the fits form a CHAIN. A rebuild reproduces it
        exactly only by replaying the chain from the same first refit on the same data (it
        does: same `start`). A data revision can move one fit into a different local
        optimum, and that propagates to every later fit - which is what the periodic
        rebuild is for: the report gives the first fit where the chain diverged.
        `RegimeSpec.warm_start=False` makes every fit cold (k-means multi-start: each fit
        depends on its window only, PATH-INDEPENDENT; ~2s per weekly fit, so affordable on
        this schedule), at the cost of regime CONTINUITY (a cold fit can land in a different
        local optimum than last week's, so the regime definition can hop). Default: warm.
    *   In-sample (smoothed) probabilities are re-estimated by every fit - the past regime
        weights a fit uses can change from one week to the next; that is estimation, not
        look-ahead. Out of sample only predicted / filtered probabilities are ever used.
*   **Adding a model to this machinery:** it must have `prepare`, `fit(prepared, as_of)`,
    `params()`, `from_params`, and `predict(prepared, start=, end=)` returning row-local
    outputs; optional `warm_start_from` / `align_to` / `restore_path`; then a `RunConfig`
    kind, and the incremental-equals-rebuild test in `tests/test_model_runs.py`.

## 1. Scope and layering
*   **`infra/models` is a consumer layer, like `infra/analytics`**: pure computation on data
    the pipeline already stored. It never calls an API and never writes storage. Modules
    that read storage: `infra/models/nowcast/nowcast.py` (through
    `infra.pipeline.releases.read_releases_from_disk`) and `infra/models/cta/inputs.py`
    (through `infra.pipeline.relative_daily.load_relative_daily(fetch_missing=False)` and
    `infra.pipeline.daily.read_daily_from_disk`; the continuous-futures reader moved to
    `infra.pipeline.series_panel.continuous_futures` on 2026-10-03, unchanged), and
    `infra/models/stats` (any series through `infra.pipeline.series_panel.read_panel`).
*   **Dependencies point one way.** `infra/api`, `processing`, `pipeline`, `cycle` and
    `storage` never import `infra.models`. Only `infra/dashboard` may, from above. This is
    enforced by `tests/test_architecture.py`.
*   **The data lives in the main pipeline, not here:**
    *   Release config: `infra.config.MACRO_RELEASES` / `MacroRelease`, one row per release
        of the user's table: HOW the nowcast uses a series (staging flags, sign,
        transform, categories).
    *   WHAT each series is and where it comes from lives in its registry entry,
        `MacroRelease.series` (`infra.reference.events.SERIES`): source, stored id,
        `derive` (units), frequency, calendar pattern and scale. `release.source` etc. are
        read-through properties. Moved there 2026-10-02 with no behaviour change: the
        nowcast, the calendar validation and the rule validation were byte-identical before
        and after.
    *   Client: `infra/api/fred_client.py`.
    *   Pure vintage logic: `infra/processing/releases.py`.
    *   Read/plan/fetch/store: `infra/pipeline/releases.py`.
    *   Daily-cycle step: `infra/cycle/raw_releases.py`, registered as `raw` step source
        `"releases"`, i.e. `backfill_daily_raw_data`.
    *   Store: `~/Database/RawData/Releases`. Coverage manifest:
        `~/Database/RawData/_coverage/releases.parquet`.
    *   Standalone script: `scripts/update_releases.py` (`--dry-run`, `--verify`).

## 2. The release table (`MACRO_RELEASES`)
*   **Transcribed 2026-09-30 from the user's OneNote "Data Releases" table.** The source
    was a photo, read at full resolution; every cell was legible.
*   **Columns map 1:1 onto fields:**
    *   Staging flags: `Advanced`, `Preliminary`, `Final`, `ShortHistory`, `BbgMedian`,
        `NoStage`.
    *   `Sign` and `Transform`.
    *   `Cat1` (Activity/Price) and `Cat2` (Survey/Hard).
    *   Subcategory columns: `Activity_*` and `Price_*` → `blocks`.
*   **The Bloomberg ticker is the key and a reference only.** There is no Bloomberg data.
    Each release is rebuilt from a free official series:
    *   `source` + `series_id` is what gets stored, as published.
    *   `units` converts it to the table's units per vintage: `level`, `diff`, `pct` or
        `yoy`.
*   **The 9 releases with no official free history come from archived economic
    calendars (`source="calendar"`, series ids `MW:<ticker>`).** These are ISM
    manufacturing and services, the S&P Global (formerly Markit) manufacturing, services
    and composite PMIs, MNI Chicago, the Conference Board, NFIB and NAR existing home
    sales. See section 3a.
*   **Richmond Fed (`RCHSINDX`) still has no source:** it is not on FRED, and not on the
    calendar (see `TOFIX.md`).
*   **The Bloomberg consensus median (`BbgMedian`) is not available and not needed.** The
    DFM's own forecast is the baseline each surprise is measured against.

*   **Series choices, each verified against FRED's own metadata (`update_releases.py
    --verify`) and the calendar cross-check, 2026-10-01:**
    *   **PPI:** `PPIFID` (NSA, as the table's "YoY NSA" says), not `PPIFIS` (SA).
    *   **GDP:** the level `GDPC1` (`units="saar"`), whose vintages start 1991, rather than
        BEA's growth series (vintages only from 2014). Growth computed from the level
        matches BEA's published growth within 0.05pp.
    *   **Empire:** the SA headline `GACDISA066MSFRBNY`. The NSA series was used until the
        cross-check showed a 0% match.
    *   **Wholesale inventories:** `I42IMSM144SCEN`, from the Monthly Wholesale Trade
        release. `WHLSLRIMSA` belongs to a release a median 6 days after the first print.
    *   **UMich:** `fred+prelims`. FRED carries finals only, so the mid-month prelims come
        from the calendar.

## 3. Vintages (the real-time data)
*   **FRED/ALFRED stores every published value with its publication day**
    (`realtime_start`). Advance/Preliminary/Final estimates and later revisions are
    therefore separate rows, dated when they were published. No separate release calendar
    is needed.
*   **Raw store row:** `(timestamp = publication day, ticker = source id, period, value)`.
    *   Partitioned by publication day, so an `as_of` read is a partition-pruned
        point-in-time read.
    *   `value` is float64, not ×10000 fixed-point. Rule 6b is for prices, and macro values
        span about 10 orders of magnitude: 6.1M initial claims ×10000 overflows int32.
*   **A row exists only where the published value changed** (`drop_unchanged`; ALFRED
    also only starts a new period on a change). So the k-th row is the k-th DISTINCT
    value, not necessarily the k-th scheduled estimate. Verified 2026-09-30: Q2 2026 GDP's
    unchanged second estimate (27 Aug) leaves no row. Labelling Advance/Second/Third needs
    the source's release calendar. The model never needs the labels.
*   **GDP comes from the real GDP level `GDPC1` (`units="saar"`),** not BEA's published
    growth series. The level has vintages since 1991-12; the growth series only since
    2014-09. The derived growth matches the published figure within 0.05pp (BEA's
    one-decimal rounding) at every vintage checked, 2026-09-30.
*   **First archived vintage per series:** see `TOFIX.md` ("pseudo-real-time"). A fully
    real-time panel exists from mid-2018.
*   **Coverage is on the publication-day axis and must stay contiguous from FRED's epoch
    (1776-07-04).**
    *   FRED also returns values published before the requested start that are still
        current. They come back dated at, or clipped to, the start.
    *   Those rows can only be told apart from a genuine publication on that day if
        everything earlier is already stored.
    *   So a new series is bootstrapped with its whole vintage history, and later requests
        start where coverage ends, never leaving a hole.
    *   `force_refetch` (the scheduled run's revision window) re-asks the trailing days.
        Rule 2.1 otherwise holds.
*   **A publication day is claimed covered only once it's over** (`DAY_COMPLETE_LAG`:
    06:00 UTC the next day).
*   **Point-in-time everywhere.**
    *   `read_releases_from_disk(as_of=)` → `processing.releases.snapshot(as_of)` →
        `derive_units`. A derived value, such as a payroll change, is dated by its latest
        input.
    *   All transforms are trailing.
    *   The same `nowcast(model, as_of)` call therefore serves live use and backfills
        (`nowcast_history`).

## 3a. Calendar-sourced releases (archived MarketWatch economic calendar)
*   **Where it comes from.** MarketWatch's U.S. economic calendar lists, for each release,
    its day and time, actual, consensus median and previous. The Wayback Machine has
    archived it since 2009 (`infra.config.CALENDAR_PAGES`).
    *   **Coverage, measured 2026-10-01:** every week from Apr 2020 to Sep 2026 has a
        capture. Before that, 366 of 590 weeks (62%): 2009–11 thin, 2012–19 25–47 weeks a
        year.
    *   **Spot check:** a capture checked against the real releases matched (e.g. ISM Aug
        2024 47.2, S&P final manufacturing Aug 2024 47.9, Conference Board Aug 2024
        103.3).
*   **Two layers:**
    1.  `scripts/backfill_econ_calendar.py` harvests captures into the calendar store
        `RawData/EconCalendar`: one row per (release day, report, period), with values
        parsed and as shown, including the consensus.
    2.  The releases pipeline's `"calendar"` source (`infra.pipeline.econ_calendar.
        calendar_vintages`, local, no network) turns those rows into vintages in the SAME
        `RawData/Releases` store. Everything downstream reads them exactly like FRED data.
*   **Parsing facts** (verified across 2010, 2018, 2020, 2024 and 2026 captures; details in
    `infra/processing/econ_calendar.py`):
    *   Columns are mapped from each table's own header row, because layouts changed.
    *   The year is inferred from the weekday. A frozen old-URL page captured in Oct 2020
        still shows April 2020, so the capture date can't be used.
    *   Values carry units and typos (`5.02 mln`, `4.43 miln`, `-$78.8B`), which are
        normalised.
    *   Report names drift ("ISM" → "ISM Report On Business Manufacturing PMI"; Markit →
        S&P). Each release's `calendar_pattern` must match every observed variant and
        nothing else. `tests/test_econ_calendar.py::NAMES` lists them;
        `backfill_econ_calendar.py --names` audits the store.
*   **What a value means in time:**
    *   An `actual` is that period's print, dated on its release day.
    *   A `previous` is the prior period as it stood on the release day. It is a revision
        when it differs, and it also fills a month whose own week was never archived.
        It's dated on the release day (late at worst, never early).
    *   **Not for flash/final releases (the three S&P PMIs):** there, the final row's
        "previous" is the SAME month's flash (2024-09-03: final 47.9, previous 48.0 = the
        August flash; July was 49.6). So those releases take actuals only, and flash and
        final each count as a print on its own day.
*   **Merging captures:** a row's "previous" changes at release (pre-release pages show the
    old value). Per row, a post-release capture beats a pre-release one, then the later
    capture wins. A stale page archived years later never overwrites a post-release row.
*   **Terms of use:** MarketWatch's terms prohibit automated scraping, and ISM, S&P
    Global, MNI and the Conference Board license their data. The calendar history is read
    from the Internet Archive's copies at personal-research volume. Treat the stored values
    as for personal research only: never redistribute or publish them.
*   **Not live yet.** The archive's capture timing makes it ~67% reliable for T−1 by a
    06:00 ET run (measured over Apr–Sep 2026). A direct live source is the next step
    (`TOFIX.md`). Until then the calendar series go stale after the last harvest, and
    `releases_fresh` warns.
*   **Typo filter** (`infra.processing.econ_calendar.suspect_prints`, applied when
    vintages are built). A typo is a unit or decimal slip: off by about 6× or more from
    at least two of three references (the next release's restatement of that period, and
    the row's own previous and consensus), with the restatement deciding whenever one
    exists. Values that are small for the series are never compared by ratio.
    *   **Calibrated 2026-10-01 on 1,256 cross-checked prints:** it flags exactly the four
        real slips (unemployment "50" for 5.0, claims "368" for 368k, housing starts "6.28M"
        for 628k, existing home sales "5.49" for 5.49M) and none of 1,232 good prints.
        Two earlier rules (revision size, then ratio only) flagged 17 and 14 good prints.
    *   A flagged print is left out. Its period then takes its value from the next
        release's "previous", dated that later day.
*   **Cross-check against FRED** (`infra.pipeline.econ_calendar.crosscheck`,
    `scripts/validate_econ_calendar.py`). Every FRED-sourced release has a
    `calendar_pattern` + `calendar_scale` too (`infra.config._CALENDAR_CROSSCHECK`;
    left out where the calendar's units differ: PPI YoY, retail ex autos & gas). Each
    calendar actual is compared with FRED's value for that period as published THE SAME
    DAY. A match validates the parsing, the period, the units and the release day at once:
    the guardrail for the calendar-only releases, which have nothing to be checked
    against.
    *   **Result 2026-10-01 (through mid-2021):**

        | Release | Match |
        |---|---|
        | GDP (incl. revision stages) | 100% |
        | Payrolls | 100% |
        | Empire | 100% |
        | Housing starts | 99% |
        | Industrial production | 99% |
        | Unemployment rate | 99% |
        | Wholesale inventories | 99% |
        | Durable goods | 98% |
        | Claims | 97% |
        | Philly | 97% |
        | CFNAI | 80% (the calendar sometimes quotes the 3-month average) |
    *   **It caught two bugs in our FRED setup.** EMPRGBCI was the NSA Empire series (0%
        match, now `GACDISA066MSFRBNY`). MWINCHNG came from a release a median 6 days
        after the number's first publication (now `I42IMSM144SCEN`, Monthly Wholesale
        Trade).
*   **UMich preliminaries** (`source="fred+prelims"`). FRED only carries the end-of-month
    final, so `infra.pipeline.releases.fred_with_calendar_prelims` adds the calendar's
    mid-month preliminary as an earlier vintage of the same month.
    *   A row is a preliminary if named so, never if named final, else if released by day
        20 (`PRELIM_LAST_DAY`).
    *   The model sees the prelim as news and the final as a revision, as the table's
        Preliminary/Final flags intend.
    *   These rows are calendar data inside the FRED series' store, so the cross-check
        compares UMich's finals only.
*   **Consensus** (`infra.pipeline.econ_calendar.consensus(release, as_of)`): per print,
    `consensus_mw`, `actual` and `surprise`, in the release's own units and point-in-time.
    It's MarketWatch's median, not Bloomberg's survey: a free stand-in for the table's
    `BbgMedian`, labelled so it's never mistaken for it. The model doesn't use it.

## 4. Data preparation (`nowcast/panel.py`, `nowcast/transforms.py`), separate from the model
*   **Pipeline:** `snapshot(as_of)` → `derive_units` → the table's transform × `sign` →
    `to_monthly`, which produces one monthly grid:
    *   Quarterly values sit in the quarter's third month.
    *   Weekly values are the mean of a complete month's weeks only. A partial month would
        otherwise change as weeks arrive, turning news into revisions.
*   **The target (GDP) skips the transform and sign.** It enters in native units (QoQ %
    SAAR), so every nowcast and contribution is in GDP units.
*   **Transform steps:**
    *   `rolling_ma:N` and `deman_fix:K` (subtract a fixed level: PMI 50, NFIB 100).
    *   `ecdf_exp:W`: exponentially weighted ECDF, W = half-life in business days (user
        2026-09-30; 1305 ≈ 5y, 130 ≈ 6m). Output `2F−1`, centred on the series' own
        history.
    *   `ecdf_no_demean_exp:W`: `sign(x)·F(|x|)`, which keeps 0 as the anchor. It is the
        ECDF analogue of a z-score without demeaning (x / RMS around 0). User-confirmed
        2026-09-30.
    *   `gauss`: normal score. Not in the table; `ModelSpec.gaussianize` appends it.
*   **`ECDF_MIN_OBS = 12`:** no ECDF value until the series has 12 observations.

## 5. The model (`nowcast/dfm.py`, `nowcast/kalman.py`, `nowcast/spec.py`)
*   **Mixed-frequency DFM estimated by EM.** This follows Bańbura & Modugno (2014), the
    model behind the NY Fed Staff Nowcast (Bok et al. 2018, FRBNY SR 830).
    *   Monthly: `x = λ·f + e`, with an AR(1) idiosyncratic component in the state
        (`idio="ar1"`, NY Fed style) or iid noise.
    *   Quarterly GDP: loads on `f_t + 2f_{t-1} + 3f_{t-2} + 2f_{t-3} + f_{t-4}` (Mariano &
        Murasawa 2003).
    *   Factors: VAR(p).
    *   The Kalman filter and smoother are exact with missing data. `tests/test_nowcast.py`
        checks them against the brute-force joint-Gaussian posterior.
*   **Versions (`spec.VERSIONS`).** Only the loading pattern differs; data prep and EM are
    shared:
    *   **a**: one factor.
    *   **b**: one factor per category (`block_level`: `sub` = the Activity_*/Price_*
        columns, or `cat1`/`cat2`). Loadings are unrestricted, and each factor is
        initialised from its category's first principal component. The labels are only
        identified up to rotation.
    *   **c**: hard blocks. Loadings outside a release's own categories are exactly 0.
    *   **d**: soft blocks. Off-block loadings get a N(0, τ²) prior, so the M-step is a
        ridge/MAP step.
        *   τ → 0 recovers c; τ → ∞ recovers b.
        *   The penalty is scaled by the release's idiosyncratic variance. With AR(1)
            idio, that treats the idio as the regression noise and ignores its
            autocorrelation. This is an approximation in the penalty only; see `TOFIX.md`.
    *   **Options:**
        *   `global_factor=True` adds a factor every release loads on (NY Fed: global +
            blocks).
        *   `factor_dynamics="independent"` gives each factor its own AR(p) with
            uncorrelated shocks (Bańbura–Modugno / NY Fed); the default `"full"` is one VAR
            across factors.
        *   `use_transforms=False` skips the table's transforms (inputs in the table's
            units).
        *   `exclude`: months the model never sees. Default Mar 2020–Jun 2021 (COVID).
            Unmasked, the fit did not converge and block contributions reached ±13pp,
            verified on real data 2026-09-30.
        *   `factor_lags`, `idio`, `tau` and `sample_start`.
    *   **Real-data comparison and open default choices:** see `TOFIX.md` ("per-block GDP
        attribution is fragile", "single-series blocks are degenerate").
*   **Normalization.** Every factor is rescaled to unit unconditional variance after each
    M-step. This is free for a–c. It is required for d: otherwise the model could inflate a
    factor's scale to escape the prior.
*   **Sign.** Each factor points the way its own members load on average (`orient`).
*   **Parameters are frozen at estimation**, including μ/σ standardization. `nowcast` and
    `news` hold them fixed.

## 6. Nowcast and news (`nowcast/news.py`, `nowcast/nowcast.py`)
*   **Nowcast:** `E[GDP_q | data]` in GDP units. Once the quarter's GDP is published, that
    value is the nowcast.
*   **`common`:** the model's own estimate split by factor, plus the mean. Always
    available, so it gives "which block is driving GDP" in pp.
*   **News:** Bańbura & Modugno (2014), the NY Fed's impact table.
    *   `new − old = revisions + Σ weight_j·(actual_j − forecast_j)`, and it sums
        exactly.
    *   `weight_j` is the smoother's gain on that print's surprise in the GDP nowcast: how
        much of it is read as GDP signal rather than noise, net of the other prints in the
        same batch. It is computed as the nowcast's response to a unit bump of print j,
        with the other new prints at their expectation. That is exact and
        order-independent, since the conditional mean is affine in the new data.
    *   `signal_share_j` is the same bump's effect on the print's own common component:
        news (1) versus idiosyncratic noise (0).
*   **Coming up** (`nowcast.upcoming(model, as_of, days)`): the model's releases due in
    the next `days`, from the release calendar as known at `as_of`. Per print it gives:
    *   the period it will fill (the series' next unpublished month);
    *   the model's forecast;
    *   the MarketWatch consensus, where the calendar showed one by `as_of`;
    *   the weight: GDP pp per unit of surprise, the same unit-bump response as `news`.
        A test pins this to the `news()` weight once that print lands.
    *   `units` is `release` (comparable to the consensus) only with `use_transforms=False`;
        otherwise it's the model's transformed units.
    *   Revisions (stage second/third/final) and weekly claims get a `note` instead.
    *   **Limits:**
        *   It works going forward. The harvested calendar kept only each row's last
            capture, so for past weeks a release is known only from its own day; the view
            can't be replayed historically from the calendar.
        *   The consensus for NEXT week appears late; MarketWatch fills its forecasts during
            that week.
*   **Revisions are separated first:** old values re-published differently.

## 7. Running it
*   `scripts/update_releases.py`: fetch (the daily cycle does this every run too).
    `--verify` prints FRED's own metadata per configured id.
*   **Without `FRED_API_KEY`, the `raw` step fetches nothing and warns**
    (`releases_configured`) instead of failing, so a missing key never blocks the daily
    vintage.
*   `scripts/run_nowcast.py --version c [--as-of D] [--news-from D0] [--tau ...]`.
*   **Python:**
    ```python
    from infra.models.nowcast.nowcast import estimate, nowcast, news
    model = estimate("c", as_of="2026-06-30")
    nowcast(model, as_of="2026-09-30")
    news(model, "2026-09-23", "2026-09-30")
    ```

## 8. Inflation
*   **Its own sub-project, `infra/models/inflation/`, with its own `CLAUDE.md`** (CPI / PPI
    / PCE component trees, their vintages and weights). Read that file before touching
    inflation work.

## 9. CTA positioning
*   **Its own sub-project, `infra/models/cta/`, with its own `CLAUDE.md`**: a bottom-up
    replica of a trend-following CTA (UBS Q-Series 2022), the first model built on the
    `base.Model` pattern. Read that file before touching it.

## 10. Treasury futures basis
*   **Its own sub-project, `infra/models/basis/`, with its own `CLAUDE.md`**: the delivery-option
    / cheapest-to-deliver models (pricing tiers M0 ... M3, add-ons T, IV and MS, an
    explanatory basis-spread layer) and its
    validation bench. Its pure maths lives BELOW the models (`infra/analytics/futures_basis.py`)
    so the daily cycle can use the futures DV01 without importing `infra/models`. Read that
    file before touching it.

## 11. Regression & PCA on any series
*   **Its own sub-project, `infra/models/stats/`, with its own `CLAUDE.md`**: regression
    classes (OLS/WLS, stepwise, ridge/lasso/elastic net, Huber, quantile, hockey stick,
    total least squares, logit/probit, Kalman time-varying beta) and PCA classes (plain,
    weighted, missing data by probabilistic-PCA EM), regime models (Gaussian HMM, rules) and
    a regime-weighted PCA (K series' structure under regimes inferred from N series), their
    diagnostics (factor correlation
    by sub-period, eigenvector stability, parallel analysis, bootstrap loadings,
    coefficient stability), input-agnostic, serving both use cases of 0a. Read that file
    before touching it.

## 12. Event studies
*   **Its own sub-project, `infra/models/event_study/`, with its own `CLAUDE.md`**: windows
    defined by an event code (date+time events from the central registry, time-only events,
    day and grid-step lags, a trading grid), window P&L from a pluggable source, per-instrument
    tests against placebo windows, families of codes with false-discovery control; a run kind
    on the 0b operating model. Read that file before touching it.
