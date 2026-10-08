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

## 4. Cross-asset, multi-factor (built 2026-10-07)
*   **Data:** 16 CME macro roots (`infra.config.MACRO_ROOTS`: ES NQ RTY NKD, 6E 6J 6B 6A 6C 6S
    6M, CL NG, GC SI HG), daily settlements from 2010-07 (`scripts/backfill_daily_bulk.py`),
    read as `fret:<root>.v.0` = SAME-CONTRACT log returns. Never take log differences of the
    additively back-adjusted level (`fut:`): it divides each move by a shifted level - crude's
    level went negative before 2014, and every macro future's vol came out wrong (NQ 14% vs
    20%, crude 64% vs 43%). STIR as `fut:SR1.c.3` (same-contract change, x -100 = bp), never
    `stir:` (raw rank settlement: every roll a fake jump).
*   **Orientation:** every instrument "+ = a loss for a long holder" (futures -100 x log
    return, yields +100 x change), so positive asymmetry = crowded long everywhere.
*   **Multi-factor expected move** (`multifactor_relative`): residual vs a trailing OLS on K
    factors (first 4 PCs of the VOL-SCALED panel, or named: 10y, S&P, EUR, crude, gold),
    fitted every 21 days on 504 days ending before the measurement window, >= 252 rows per fit
    (60 overfit: ES out-of-sample R^2 -4.5); asymmetry on big EXPECTED moves. Specs
    `macro_daily`, `macro_daily_named`, `macro_daily_stir` (the last from 2022, SR1's history
    start + the fit's warm-up). ~35 s for all three.
*   **Fit quality:** median out-of-sample R^2 0.5-0.65 for FX majors, equity indices, the 5-30y,
    gold; ~0 for crude, natural gas, the yen. It COLLAPSES in regime changes (10y -0.7 in 2023,
    the stock-bond correlation flip): there the residual measures changing covariance, not
    positioning.
*   **Validation (equity + FX futures vs CFTC TFF, 2010-2026): nothing.** Multi-factor asym vs
    leveraged-fund net (z vs its trailing year) -0.06 on average, residual skew -0.11 (both
    the wrong sign, small); forward 20-day pain move: pooled correlation +-0.02 (~t 0.15).
    Semivariance vs leveraged funds -0.26, asset managers -0.33: trend again - speculative
    CFTC positioning is itself largely trend-following. Likely reason: ~9 big moves per side
    in a 63-day window can't measure an excess beta. Next: intraday (bbo-1m) for the macro
    roots - ~53 observations a day.

## 5. Aggressor positioning: signed volume (built 2026-10-08)
*   **Data:** every ZN trade with its AGGRESSOR side (Databento `trades`: `B` buyer-initiated,
    `A` seller-initiated, `N` none), 2026-04-07..10-07, ZNM6/ZNU6/ZNZ6/ZNH7 (front and second
    around both rolls), 13.3M trades, $16.65, 146 MB. Raw ticks are STORED (any later cut -
    1-second, volume buckets, trade sizes, blocks - is local, never re-bought); the 1-minute
    signed bars are built from them (root CLAUDE.md 14). Extending = more history or the other
    five Treasury futures (one year of all six: $104, priced 2026-10-08).
*   **Terms:** signed volume = buy - sell aggressor volume ("trade pressure" when normalised:
    `imbalance` = signed / (buy + sell)). NOT order flow imbalance in the Cont-Kukanov-Stoikov
    sense (quote-queue changes, passive orders too) - a separate, later measure.
*   **First look (front contract, 128 trading days; day-clustered t):**
    *   **Same-time link is strong:** correlation of signed volume with the price change 0.48
        (5 min), 0.57 (15), 0.61 (60); impact ~0.25-0.30 ticks (1/64) per 1,000 net contracts.
    *   **No persistence:** 5-minute signed volume autocorrelation ~0 at every lag (1-12).
    *   **No short-term prediction:** past 5 / 15-minute flow -> next 5 / 15 minutes: 0.00
        (t 0.1-0.3), also net of the window's own price move. Past HOUR -> next hour: -0.05
        (t -2.2): a slight reversal of heavy hourly flow - borderline, one of several tests.
    *   **Absorption** (30-min windows with top-10% |flow| and a below-median move, n = 101): no
        follow-through either way (-0.7 ticks in the flow's direction over the next hour, t -1).
    *   **`N` (no aggressor) = 1.2% of volume:** mostly the outright legs of spread trades - the
        hook for roll research (tagging roll trades), deliberately not pursued yet (user, 2026-10-08:
        outrights first; the spread instruments need a Rule 2.2 exception).
*   **Reading:** flow explains price at the same time and predicts nothing at intraday horizons
    on its own - the efficient-market baseline. The positioning use is slower: cumulative flow
    over days, combined with daily open interest (the entry-price distribution, TOFIX roadmap) -
    which needs more history than six months.

## 6. Multi-day "programs" in aggressor flow (2026-10-08, `flow.py`, `scripts/study_flow_programs.py`)
*   **Hypothesis (user):** intraday flow effects don't hold on average, but large participants
    work orders over DAYS (index extension, rebalances, hedges, CTAs): same-sign net flow on
    consecutive days, steady through the day, front-loaded the next morning.
*   **Setup:** ZN, all contracts summed (a roll done as two outright trades nets out; spread
    legs carry no aggressor), CME trading days; sessions overnight 18:00-03:00, London
    03:00-08:20, New York 08:20-17:00 (New York time). PRE-REGISTERED tests, discovery
    2026-04-07..07-31 (84 days), validation 08-01..10-07 (49 days).
*   **Results - nothing replicates:**
    *   daily net flow persistence (lags 1-5): discovery -0.06..+0.03, validation -0.23..+0.18,
        no consistent sign;
    *   day t's flow -> day t+1's flow by session: London -0.20 then +0.07; New York +0.11 then
        +0.21 (t 1.0, 1.4 - the only same-sign pair, weak); price: no continuation;
    *   steady vs bursty days: discovery both ~0; validation +0.25 vs -0.12 (n 24 each) - not
        replicated;
    *   the CTA model's position change: same-day +0.38 / +0.30, but almost all of it through
        the day's price move (the model's signal updates with the close; flow moves price):
        net of it +0.10 / +0.17 (t 0.9, 1.2); next day ~0.
*   **The open-interest proxy** (OI change x price-change sign, all contracts): correlates only
    +0.24 (t 2.8) with the real daily flow here - a weak proxy - yet over 2014-2026 it IS
    persistent: lag 1 +0.13 (t 7.1), lag 2 +0.05 (t 2.9). Price signs aren't autocorrelated,
    so this is positions being added in the same direction on consecutive days: the
    program signature, seen in the slow data - looked like the best lead; section 7 shows it
    was mostly rolls and outliers.
*   **Power:** 133 days give a daily-level standard error of ~0.09 (~0.14 per half). Only
    effects of ~0.3 would show; the hypothesis isn't rejected, just not visible at this size.

## 7. The open-interest proxy, dissected (2026-10-08, `scripts/study_oi_programs.py`)
*   **ZN's persistence was an artifact.** Rank correlation +0.03 (t 1.8) instead of +0.13;
    excluding roll windows (+-7 days of a front switch and the 10 days before it, 27% of days)
    +0.03 (t 1.5); by year it is ~0 except 2019 (+0.19) and 2026 (+0.45). What persists is
    |OI change| (+0.42): the big open-interest swings of roll and expiry weeks, a few
    large-magnitude days carrying the product's correlation. Excluding payroll / CPI / FOMC
    weeks changes nothing (+0.16).
*   **Across 22 roots (2010/2014-2026, one root's standard error ~0.016):** Treasury futures
    ~0 once rolls are excluded (-0.07..+0.05). FX, metals, crude and equity index futures show
    real program-like persistence, ex-roll: 6E +0.06, 6B +0.07, 6A +0.09, 6C +0.11, 6M +0.07,
    GC +0.11, SI +0.11, HG +0.07, CL +0.13, NQ +0.11, RTY +0.07 (t ~3-7 each); NKD -0.20 (thin).
*   **No price relevance anywhere:** proxy vs the next day's price change within +-0.03 for
    every root (ZN +0.007).
*   **Who (weekly CFTC TFF changes vs the weekly proxy, 2010/2014-2026):** Treasuries ~0 for
    every category. FX and equity index: speculators on the proxy's side (leveraged funds
    +0.07..+0.12, asset managers +0.05..+0.13) and dealers against it (-0.09..-0.19) - the
    proxy reads speculative adding absorbed by dealers. Weekly position changes are themselves
    persistent in FX for every category (+0.19..+0.38; dealers most), mean-reverting for
    Treasury dealers (-0.08..-0.13).
*   **Reading:** in Treasuries neither the ticks (section 6) nor the slow proxy show programs;
    in FX / commodities / equity indices programs exist as persistent position building, but
    carry no next-day price information in this form.

## 8. Curves (2026-10-08): known flows and asymmetry on structures, data on disk only
*   **Auction cycle** (`scripts/study_curve_flows.py`; Lou, Yan & Zhang 2013): P&L of LONG the
    auctioned tenor against its on-the-run neighbours (bmk OTR yield P&L, 2008-2026; flies
    3y 2/3/5, 5y 3/5/7, 7y 5/7/10, 10y 7/10/30, 20y 10/20/30 from 2020, 2y vs 3y, 30y vs 10y;
    complete windows only). BEFORE (closes D-5 -> D-1) it cheapens: 3y -0.47bp (t -4.6), 5y
    -0.27 (t -3.6), 10y -0.18 (t -1.8), 20y -0.28 (t -1.8), 30y -0.32 (t -1.2); 7y none.
    After (D -> D+5): 30y recovers +0.71 (t 2.1); 2y keeps cheapening -0.69 (t -2.9) - possibly
    the on-the-run switch at its issue a few days later, unverified. Real, but fractions of a bp.
*   **Month-end:** the 30y RALLIES into the last 3 business days (+0.65bp/day, t 3.3;
    refunding months +0.90, t 2.6) - but the curve STEEPENS then (5s30s +0.31, t 2.4; 10s30s
    +0.15, t 2.1): the belly richens more than the long end. Index extension shows as a
    duration rally, not the long-end flattening expected.
*   **Asymmetry on curve structures** (spec `ust_curves_daily`: 2s10s, 5s30s, 2s5s, 10s30s, 2/5/10
    and 5/10/30 flies vs the 10y level; + = a loss for a long structure):
    *   relative asymmetry and semivariance: no forward information (mean -0.03);
    *   **SKEW predicts reversal:** structures whose last 63 days were skewed towards sharp
        losses for longs recover over the next 20 days. Negative on all six structures (mean
        -0.10; 5/10/30 fly -0.18, t -2.6; 2s10s -0.13); the SAME in both halves (2008-2017
        pooled -0.089, 2018-2026 -0.090); and stronger net of the structure's own past 63-day
        move (pooled -0.13, t ~-3.2 counting the six structures as ~3 independent; the past move
        itself +0.04, t 1.1). Residual skew weaker after 2018.
    *   Reading: a flush (sharp moves against the crowded side) followed by a bounce - the
        first measure here to survive a split-sample check. Section 9 tests it properly (the
        pooled -0.13 here was an early, looser estimate).

## 9. Curve skew: horizons, mean-reversion overlap, other countries (2026-10-08, `scripts/study_curve_skew.py`)
*   **Setup:** six structures per curve (2s10s, 5s30s, 2s5s, 10s30s, 2/5/10, 5/10/30), daily
    moves in bp, duration-neutral, + = a loss for a long holder. Signal: 63-day skew. Controls:
    the past 63-day move, and the level z (cumulative move vs its trailing 252-day mean / sd -
    the fade-the-level of `infra/models/meanrev`'s exante mode, applied to the structure).
    Forward: the next h days, sampled every h (non-overlapping); pooled over the structures,
    standardised, t counting the six as ~3 independent.
*   **US on-the-run (bmk P&L, 2008-2026): negative at every horizon** - h = 5, 10, 20, 40, 60 days:
    -0.04 (t -1.9), -0.05, -0.09 (t -2.2), -0.06, -0.10; unchanged by the controls. Strongest
    on the long end: 5/10/30 fly -0.22, 5s30s -0.11 (h = 20). US CMT (2016-2026) the same sign
    (-0.075 at 20 days, t -1.4; 5/10/30 -0.17, 5s30s -0.19).
*   **Input trap found:** differencing the `otr:` yield series puts a fake move on every
    on-the-run switch day (a new bond, another maturity); it cut the effect to -0.03. Structures
    must come from the bmk P&L (same bond across a switch) or a fitted curve (CMT).
*   **No overlap with mean reversion:** the skew coefficient doesn't move with the level-z
    control, and the level z itself comes out POSITIVE (+0.06..+0.33 by horizon, t up to 2.2): over
    1-3 months the structures' levels TREND rather than revert - curve mean reversion in this
    simple form is not what the skew is picking up.
*   **Other countries, weaker:** DE (Bundesbank curve, 2016-2026) -0.04 at 20 days (t -0.7),
    negative on 5 of 6 structures; UK (BoE curve) ~0 at 20 days, -0.16 at 40 days (t -2.2) but
    gone net of the past move.
*   **Reading:** a modest, persistent US effect concentrated in the long end (5/10/30, 5s30s),
    about -0.05 to -0.10 standardised, robust to the controls and the sample split; not a
    general law across curves. Costs not assessed (user: not now).

## 10. Skew on `meanrev`'s PCA residuals (2026-10-08, `scripts/study_meanrev_skew.py`; research only)
*   **Setup:** `infra/models/meanrev`'s own walk-forward (spec `prior`, US OTR 2/3/5/7/10/30y bmk
    P&L, 3-factor PCA, window 250, monthly refits, 2012-2026; its code untouched). Per residual:
    daily moves = day-to-day changes of its `level:` within a fit (the level is re-based at
    each refit; that day dropped), their 63-day skew (>= 50 days), its OU s-score `s:`, the past
    63-day move; forward = the next h days (windows may miss <= 2 refit days). Pooled over the 6
    residuals, standardised, t counting them as ~half independent.
*   **Skew does NOT predict the residuals:** -0.01 / -0.04 / -0.04 at h = 5 / 10 / 20 (t >= -1.2),
    with or without the s-score. So the curve-structure skew effect (section 9) lives in the
    slope / curvature FACTORS the structures are exposed to, not in the bond-specific residual
    `meanrev` trades.
*   **`meanrev`'s s-score does predict reversion:** -0.04 (t -1.9), -0.06 (t -2.1), -0.10 (t -2.4).
*   **...and more so after a skewed flush in the dislocation's own direction** (sign of the skew
    = sign of the s-score: the residual got there through sharp moves): h = 20 -0.14 (t -2.5, n
    574) vs -0.05 (t -0.7, n 490) otherwise; h = 10 -0.07 vs -0.05; h = 5 -0.05 vs -0.04. Same
    ordering at every horizon, but the difference itself is ~1 standard error: a candidate
    conditioning for `meanrev` (fade harder after a flush), not an established improvement.

## 11. Month-end Treasury index extension (2026-10-08, `index_extension.py`, `scripts/study_index_extension.py`)
*   **Computed from the index's rules**, data on disk: fixed-coupon notes and bonds, >= 1 year
    to maturity, >= $300m, market-value weights; members fixed through the month, switched at
    the month-end close (new issues settled by then enter, < 1 year leave); duration = coupon-bond
    modified duration at FedInvest END OF DAY yields; amounts = auctions' `total_accepted`
    (competitive + non-competitive + the Fed's add-ons - whether the real index nets out the
    Fed's holdings is unverified; our market value ~$17tn suggests it may).
*   **Result 2009-2026 (213 months):** extension mean 0.074 years (0.016-0.154); refunding months
    (Feb/May/Aug/Nov) ~0.10, others ~0.06 - the known pattern; index duration ~6.0 in 2025-26.
*   **Not yet validated against a published number:** iShares (GOVT tracks the ICE US Treasury
    Core index, AGG the Bloomberg Aggregate) serves its holdings / characteristics only to a
    browser session - every scripted request gets the web page (not worked around). Planned
    check: GOVT's duration change across month-ends from holdings files the user downloads.
*   **Does the size matter?** Not for the last 3 days (correlations ~0, although the curve rallies
    then regardless: 30y +2bp, 10y +2.5bp). Over the last 10 days a bigger extension comes with
    more 5s30s FLATTENING (-0.17, t ~-2.5) and a bigger 30y rally (+0.12, t ~1.7): buyers
    position ahead of the known extension, not at the close. The first days of the next month
    give 1-2bp back.

## 12. Entry-price inventory of open positions (2026-10-08, `inventory.py`, `scripts/study_entry_prices.py`)
*   **Model:** two books per market, positions opened by BUYERS and by SELLERS, from total open
    interest: an OI increase is booked at the day's price (back-adjusted settlement, so an entry
    before a roll compares correctly after it) in the book of the day's aggressor side; an OI
    decrease closes the OTHER side's book proportionally (aggressive buying = shorts covering).
    Outputs per day: each book's size, average entry, share underwater, the price's distance from
    the entries in daily vols, and `balance` = the buyers' book's share.
*   **Tick vs proxy (ZN, 2026-04..10, both starting empty):** the day's aggressor side from ticks
    agrees with the price-change sign on 73% of days; the pain features of the two versions
    correlate 0.70-0.91 (net gap 0.91, net underwater 0.83) - the price-sign proxy is usable for
    them over long history. Not for `balance` (0.08): its errors accumulate.
*   **Predictive test, proxy books, 22 roots, 2010/2014-2026, split in halves:**
    *   `balance` (share of open positions put on by buyers - with the proxy: open interest built
        on UP days) is NEGATIVE in both halves at both horizons: h = 5 -0.032 / -0.032, h = 20 -0.064
        / -0.060, positive in only 27-36% of roots (RTY -0.34, ES -0.15, 6M -0.14). Open interest
        built in rallies precedes underperformance - a contrarian crowding signal, stable but small
        (pooled t ~-1.3 per half, counting the roots as ~4x correlated).
    *   The gaps (distance from the average entries): ~0 in the first half, -0.05..-0.09 in the
        second - unstable. Net underwater: flips sign between halves - nothing.
*   **Not in rates** (2026-10-08): Treasury futures (2015-2026) ~0 / slightly positive in the first
    half (+0.01..+0.05), negative in the second (-0.05..-0.09) only through ZT / ZF (-0.21 at 20
    days, ~-0.44 at 60 - the 2022-2025 cycle's turning points), ZN the opposite (+0.41 at 60), SR1
    (2019-) flips sign. Treasury futures' open interest is dominated by the basis trade and
    hedging, so OI built on up days isn't directional crowding there - consistent with section
    7 (no program persistence in Treasuries). For rates, positioning must separate the basis
    trade out (the CFTC basis decomposition) or work in curve space (sections 8-9).

