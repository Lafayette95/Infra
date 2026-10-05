# infra/models/event_study - Event studies (additive to the root and `infra/models` CLAUDE.md)

Everything in the root `CLAUDE.md` and `infra/models/CLAUDE.md` applies (point in time, UTC,
prepare -> fit -> predict, the operating model of 0b). Known gaps: the ONE root `TOFIX.md`,
headings "Event study: ...". Built 2026-10-05 from the user's spec.

## 1. What it is
*   **Patterns in windows relative to one event, or between two events**, absolute or
    CONDITIONAL on a third series (section 5). A study = an EVENT CODE (the window) x instruments x
    a P&L source x test thresholds (`config.EventStudySpec`, registry `EVENT_STUDIES`).
*   **Model lingo:** steps 1-2 (events -> windows -> window P&L) are `prepare`; steps 3-4 (the
    tests, per instrument, and what passes) are `fit`; `predict` gives each later event the
    fitted expectation and the signal (+1 / -1 where the study passes, else 0) next to the
    realised move. The operating model applies unchanged: a run of kind `event_study`
    (`scripts/model_run.py create ... --kind event_study --spec <study> --series <instruments>`),
    weekly fit-append, daily predict-append, rebuild + reconciliation.

## 2. Events and the code
*   **Date+time events** are the central registry's (`infra.reference.events`, root CLAUDE.md
    17): releases, auctions, central-bank decisions, Treasury lifecycle, futures calendars; their
    occurrences come from the release-calendar store, consolidated to ONE per event and local
    day (`event_windows.consolidate`: the best time - the source's own > the registry's >
    unknown -, the earliest `known_from` of all sources, a stage if any source has one;
    unconfirmed schedule rows dropped; rule projections only where nothing else lists the day).
    Events are read as known at the panel's last day (`as_of`), or as known now
    (`EventStudySpec.ignore_as_of`).
*   **Time-only events and grids** are `infra.reference.event_grid`: `TIME_EVENTS` (`GRID_START`,
    `GRID_END` = the cycle's own first / last point; `US_CASH_OPEN` 08:20 ET (the user's
    definition; the example's "futures open" meant this); `CME_GLOBEX_OPEN` 17:00 CT,
    `CME_GLOBEX_CLOSE` 16:00 CT, `CME_TSY_SETTLE` 14:00 CT; the benchmark snaps NY1500 / NY1530 /
    NY1600 / LDN1615 from `SWAP_CLOSES`) and `CYCLES` (`15MIN_NO_OVERNIGHT`: 06:00-19:00 New York
    every 15 minutes, market calendar; `1MIN_NO_OVERNIGHT`) with `CYCLE_ALIASES`
    (`DEFAULT_CYCLE` -> `15MIN_NO_OVERNIGHT`: moving the default grid is one line, then studies
    are re-run).
*   **The code** (`infra/processing/event_windows.py`, `parse_code` / `EventCode.encode`):
    `DT_REFS;TIME_REFS;;START_LEG__END_LEG;;CYCLE`, refs `__`-separated and indexed from 0 (user
    decision 2026-10-05), event ids case-insensitive, `ID:stage` to keep one stage; a leg
    `&i_D_T_L` = event i's day moved D TRADING days of the cycle's calendar, at time T (`&` = the
    event's own clock time, carried to the new day in the event's own zone; `%j` = time-only
    event j), moved L grid steps. E.g. the user's example
    `US_TSY_AUCTION_3Y__US_TSY_AUCTION_7Y;GRID_START;;&0_0_&_-2__&1_1_%0_2;;DEFAULT_CYCLE`.
*   **Pairing** (user decision): each occurrence of event 0 pairs with the FIRST occurrence of
    each other event at or after it, within `max_pair_gap` trading days (default 22 - the 7y
    auction comes ~12 trading days after the 3y each month, so 10 paired almost nothing).
*   **The interval rules** (user decision 2026-10-05): grid points run start .. end INCLUSIVE
    (06:00 and 19:00 are points); a window is (start point, end point] - it may START at 06:00
    and END at 19:00. A leg's reference time outside [first point, last point] is illegal
    (`outside_grid`: 19:01 is NOT snapped back to 19:00); inside, it snaps to the point at or
    before it; a step lag leaving the day is illegal (`lag_outside_grid`, never rolls into
    another day); `&` on a day-level event is `no_event_time`; a day-lag-0 leg on a
    non-trading day is `not_a_trading_day`; `end_not_after_start`; `no_pair`. Illegal windows
    are kept with their reason.
*   **Day lags count the cycle's trading days**: the market calendar
    (`schedule_rules.business_days(..., "market")`, NYSE-style). Consequence: an NFP on Good
    Friday (2021, 2023, 2026) is `not_a_trading_day` although CME rates trade a short session
    (`TOFIX.md`).

## 3. P&L (`infra/pipeline/event_pnl.py`)
*   **A grid's step P&L per instrument**, from a pluggable source (`PNL_SOURCES`): the move over
    (previous point, point]; a day's first point reaches back to the previous trading day's last
    point (the OVERNIGHT move - only a multi-day window contains it). Window P&L = the sum of its
    steps; a missing step makes it NaN (`missing:<inst>` counts them).
*   **`FUTURE_BPS_BBO`** (the user's "bps pnl", built on the fly): bbo-1m quote mids of the
    contract an instrument maps to (`ZN.v.0` -> its contract per CME trading day), each step on
    ONE contract (the one mapped at the step's end, at both ends: a roll is never a move), in bp
    = price change x point value / the contract's DV01 on the PRIOR trading day (bmk risk store;
    STIR = x 100). `FUTURE_PTS_BBO` = the same in price points. Where the market is SHUT - CME's
    daily halt (17:00-18:00 ET, inside the default grid) or after a close with no session
    following (Friday evening, holiday eves) - the price is the close's: without that, 11% of
    ZN's grid points were NaN (found 2026-10-05); with it 99.35%.
*   **Coverage today:** bbo-1m quotes for ZT / ZF / ZN / ZB / UB from 2015, TN from 2016 (both
    contracts around each roll); bp needs the bmk DV01: back-filled 2026-10-05, so bp from
    2018-10 (the funding model's start), 98-99% of grid steps from 2019; `FUTURE_PTS_BBO` (points)
    from 2015.
    A persisted 15-minute bmk store can replace the on-the-fly source under the same interface.

## 4. The tests (`stats.py`, thresholds in the spec; `None` = off)
*   Size: n, mean, median, std, t / p (one-sample). Robustness: hit rate with a two-sided
    binomial p, Wilcoxon signed-rank p (the median: one large event can't carry it), trimmed mean
    / t (`trim` per tail). Economic size: `ev_abs` (source units, bp) and `ev_vol` = mean /
    placebo sd. Baseline: the PLACEBO windows (the same window shape - day gap, start and end
    slot - on every trading day with no event window), `excess` = mean - placebo mean and its
    Welch t: a drift every day has is not an event effect. Stability: halves of the sample, share
    of years with the overall sign.
*   **Tests** (`passes`): `min_obs` (always), `t_min` (default 1.5), `ev_abs_min` (2 bp),
    `ev_vol_min`, `hit_min`, `placebo_t_min`, `stable_halves`, `year_share_min`; `passed` = all
    enabled. Per instrument, so a study passes on ZB and fails on ZT independently.
*   **Families** (`family.py`; as RUNS: section 4a; `FamilySpec` + `LegRule`: a cartesian product of refs, day lags,
    times and step lags per leg -> codes; registry `EVENT_FAMILIES`): every code fitted on one
    shared panel, illegal codes kept (n = 0, with the reasons), and **Benjamini-Hochberg**
    false-discovery control over all (code, instrument) tests (`q`, `passed_fdr`): a family
    produces false positives by construction.

## 4a. Family runs (`infra/jobs/family_runs.py`, built 2026-10-05)
*   **A family on the operating model:** one ordinary run per code (`<name>/c000` ..., spec
    `family:<family>#<i>`, resolved by `config.get_event_study_spec`) plus the family's
    point-in-time false-discovery table `<name>/fdr.parquet` (`family.fdr_table`: per fit DAY,
    BH across every code's (code, instrument) p_t - the codes' fits sit at different instants
    of the same day, since refit targets snap to each code's window starts - and across the
    conditional buckets' p_did; `passed_fdr` = own tests AND q <= `fdr_q`), recomputed from
    the codes' stored params after every job, so it always matches them. The panel, the
    occurrences and the condition timeline are read once per job for all codes.
*   **Incremental == rebuild holds per code**, so for the family
    (`tests/test_strategies.py`); real data (40 NFP codes, monthly refits 2024-01..2026-09):
    rebuild 17 s, a daily catch-up 6 s, reconciled identical.
*   **Predictions carry the fitted statistics** (`t:`, `n:`, `passed:`, `ev_vol:` per
    instrument; conditional: per the row's bucket, plus `t_did:`, `cond_passed:` and
    `cond_provisional` = the regime rests on a value not yet known at the window start's
    lookup now, i.e. the latest known one) and the `anchor`: what a strategy needs per row.
    `runs.predict_upcoming` / `family_runs.upcoming` give the events already scheduled in the
    next N days with the latest fit (a plan, not stored by the run).

## 5. Conditional studies (`conditions.py`, `ConditionSpec`; built 2026-10-05)
*   **The condition is another series** (`ConditionSpec.series`, any `series_panel` id: a price
    we track, a yield, a repo rate, a macro print), one per study for now (user decision).
*   **Point in time through the GLOBAL availability rules** (`infra.pipeline.series_panel`,
    root CLAUDE.md 26): each value counts from when it became public to the market (a settlement
    by the session close, SOFR at 08:00 the next morning, FedInvest at 10:00 the next morning, a
    macro vintage at its release instant ...), not from its day label - a fixed "1 day" lag would
    leak FedInvest's END OF DAY into an 08:30 window. `lag_steps` (default 1 grid step) is a
    safety margin on top. Conditioning a window on the print it contains is therefore impossible
    unless the window starts after the print.
*   **Feature** (`steps`: the prep steps along the series' own point-in-time timeline - `diff:20`
    = the 20-observation change, `ewm_z:hl` ...; `period_diff` for vintage series: the latest
    print's change vs the previous period, as known then) and **regimes** (`partition`, swappable,
    `PARTITIONERS`): `rolling_tercile:W` (default; balanced buckets), `zscore:W:k` (tails),
    `sign`, `fixed:a:b` - each from the trailing W timeline rows only.
*   **The tests**, per instrument and bucket (rows `ZN.v.0|-1`, `|+0`, `|+1` next to the overall
    row): the standalone tests on the bucket's events (`min_bucket_obs`); `t_vs_rest` (Welch,
    the bucket vs the other buckets - never vs the whole population, which contains it); and
    the **difference-in-differences** `did` / `t_did` = (bucket events - bucket placebo) -
    (rest events - rest placebo): whether the EVENT effect differs in the regime, net of what the
    regime does to every window. `cond_passed` = the bucket's own tests and `did_t_min` (and
    `vs_rest_t_min` if set). Overall: `p_kruskal` (do the buckets differ at all) and the SLOPE of
    the event move on the continuous feature, with the placebo slope and `t_slope_did` (no
    thresholds). `predict` uses the bucket the event was in at its start. Families add
    `q_did` (BH over the buckets' DiD p-values).
*   **Why the DiD (pinned by a test):** a regime that moves EVERY window (placebo too) makes the
    naive bucket-vs-rest t 9-13 on synthetic data, while the DiD stays null - over 30 seeds mean
    t 0.24, |t| > 1.5 in 3% (conservative). A regime-specific event effect passes it (t > 3).
*   **Real data** (2026-10-05, points, fit 2015-2024, `nfp_morning_by_trend`: ZN's 20-day trend
    into the print, rolling terciles over 2 years): after a 20-day rally in ZN (top tercile) ZN
    sells off after NFP (-0.15 pt, t -1.9, DiD t -1.6): passes; ZB's unconditional effect is not
    a trend effect (no ZB bucket passes the DiD); the slope of ZT's move on the trend is t -2.0
    net of placebo.

## 6. Real-data checks (2026-10-05, ZT / ZN / ZB, price points, fit 2015-2024)
*   `nfp_morning` (08:15 -> 10:30 ET, 120 events): ZB mean -0.20 pt, t -2.44, Wilcoxon p 0.018,
    same sign in 90% of years, placebo excess t -2.33: passes; ZT / ZN don't. The path: the move
    is in the 08:30-08:45 step (sd 0.36 vs 0.08 the step before).
*   `nfp_intraday` family (40 codes x ZT, ZN): 8 (code, instrument) pairs pass their own tests
    (ZN sells off in the 30 minutes after the print, |t| up to 2.8), none survives BH at q 0.10
    (lowest q 0.20) - the reason the family control exists.

## 7. Running it
```python
from infra.models.event_study.model import EventStudy
study = EventStudy("nfp_morning", source="FUTURE_PTS_BBO")
panel = study.read_panel("2015-01-01", "2026-09-30")
data = study.prepare(panel)
study.fit(data, as_of="2024-12-31"); study.fitted_.table
study.predict(data, start="2024-12-31"); study.paths(panel, data)
from infra.models.event_study.family import run_family
run_family("nfp_intraday", panel, as_of="2024-12-31")
EventStudy("nfp_morning_by_trend", source="FUTURE_PTS_BBO")   # a conditional study: same calls
```
