# infra/analytics/positioning - Positioning measures (additive to the root CLAUDE.md)

Everything in the root `CLAUDE.md` applies (point in time, UTC, disk only). Gaps and the
roadmap: the ONE root `TOFIX.md`, headings "Positioning: ...". This file says what the
measures MEAN and how they behaved on real data.

## 1. Scope
*   **Positioning has two meanings here (user, 2026-10-07):** (1) by PLAYER / sector - who
    holds the risk; (2) AGGRESSOR - who paid up to trade, at what prices. Every measure
    states which. Roadmap of both: `TOFIX.md`, "Positioning: roadmap".
*   **Measures with no fitted state live here** (pure code) with assembly / storage in
    `infra/pipeline/positioning.py`, BELOW the models, so the daily cycle can run them
    later; anything with a fit / predict step is a model (`infra/models`, e.g. the CTA).
    Stored under `~/Database/Derived/Positioning/`; readable anywhere as series ids
    `pos:<spec>:<measure>:<instrument>` (`infra.pipeline.series_panel`, so also in the
    feature maker).

## 2. Asymmetric reaction (`asymmetry.py`, specs `config.py`, built 2026-10-07)
*   **Idea:** a crowded position shows as amplified moves AGAINST it (stops, forced
    unwinds) and muted moves in its favour - inferred from prices, no positioning data.
*   **Sign convention:** moves in YIELD direction (+ = yields up); measure > 0 = yield-up
    moves amplified = consistent with crowded LONG duration (relative: long that
    instrument against the factor).
*   **Three families (user decision: all three, daily first then intraday):**
    1.  **Surprise** (`family = surprise / continuation`): each release's surprise
        (MarketWatch consensus - not Bloomberg's) standardised by its own earlier
        surprises; the usual IMPACT fitted jointly across releases by ridge (coincident
        releases share a window: payrolls + unemployment rate aren't double counted),
        refitted monthly on the previous 5 years; the expected move = impact x surprise.
        Then over the last 40 events: slopes of realised on expected, separately on
        hawkish and dovish news, `surprise_asym = (b_up - b_down) / (|b_up| + |b_down|)`
        (bounded: the unbounded ratio exploded, sd 420), read WITH `b_all` (the pooled
        slope: near 0 = the market isn't reacting and the asymmetry is noise). Intraday
        only: `cont_asym` = does the first 15 minutes' move continue over the next 105,
        sell-offs vs rallies.
    2.  **One instrument, big moves** (`family = single`): `semivar_asym` (realised
        semivariance, around 0), `tail_asym` (moves beyond k x trailing vol), `skew`
        (rolling skewness around the window's own mean).
    3.  **Relative to the usual covariance** (`family = relative`; no events, no
        timestamps): residual vs a factor - the panel's trailing level PC or a named
        instrument (configurable) - with the beta estimated on a window ENDING BEFORE the
        measurement window (else it absorbs the asymmetry); `rel_asym` = excess beta on
        big factor-up moves minus on big factor-down moves; `resid_skew`,
        `resid_semivar_asym`.
*   **Every value is a ratio of rolling sums**, trailing only, stored with its counts
    (`n_up`, `n_down`, `tail_n_*`).
*   **Specs:** `ust_daily` (OTR 2/5/10/30y, FedInvest END OF DAY, from 2008-09; 63-day
    windows, 252-day betas, k = 1), `ust_daily_vs10y` (factor = the 10y), `ust_intraday`
    (ZT/ZF/ZN/ZB/UB bbo-1m mids in bp on the 15-minute grid from 2018-10; ~20-day windows,
    6-month betas, k = 2; release windows on the 1-minute grid, (t - 1m, t + 15m]),
    `ust_intraday_vsZN`. Run times: daily ~6 s, intraday ~45 s, all four ~2 min.

## 3. What the real data says (validation 2026-10-07)
*   **Semivariance and tail asymmetry around 0 mostly report the TREND**, not crowding:
    correlation with the CTA model's positions (`ubs2022_cal`, walk-forward) -0.56..-0.71
    at every tenor, and the 2y's apparent predictive power (+0.20 with the next month's
    yield change) is momentum. The de-trended `skew` cuts the CTA correlation to
    -0.16..-0.27 and the predictive correlation to ~0. Use `skew` / the relative measures
    for crowding; keep `semivar_asym` as a direction-of-pain gauge only.
*   **The relative measure is the cleanest:** CTA correlation -0.12..+0.14 (trend-free by
    construction); predictive +0.15 for the 2y (monthly, n = 190, ~2 standard errors -
    weak, one of ~20 tests), ~0 or negative elsewhere.
*   **CFTC TFF** (z-scored vs its trailing year): |correlations| 0.1-0.25, mixed signs - as
    expected from a benchmark contaminated by the basis trade; not a usable ground truth.
*   **Surprise asymmetry correlates NEGATIVELY with CTA positions (-0.25..-0.40):** after
    rallies the market reacts MORE to dovish news - the opposite of the crowding sign.
    More likely attention / regime (easing phases listen to weak data) than positioning;
    unresolved (`TOFIX.md`).
*   **Persistence is low** (63-day-ahead autocorrelation 0.05-0.3) except intraday
    `cont_asym` on ZT/ZF/ZN (0.27-0.43) and intraday `surprise_asym` on the long end
    (0.3); intraday semivariance mean-reverts (-0.3). Daily and intraday versions of the
    same measure agree at ~0.3.
*   **Bottom line:** a working, point-in-time measurement framework; as a CROWDING signal,
    only the relative measures and the de-trended skew are clean, and none is yet shown
    to predict. Next tests: episodes (2020-03, 2022, 2023-03 SVB, 2023-10), and forward
    SKEW / reversal size rather than forward direction.
