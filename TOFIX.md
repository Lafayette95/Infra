# Known edge cases / deferred fixes

Things found during implementation that are real but were judged too narrow, rare, or
complex to fix immediately. Each entry has enough detail to pick back up later without
re-deriving the analysis. Remove an entry once it's actually fixed (and say so in the
commit that fixes it, rather than leaving a stale line here).

---

## Intraday resample buckets can span a trading-day roll boundary

**Found:** 2026-09-21, while adding exchange trading-day bucketing (CLAUDE.md 6e).
**Where:** `infra/processing/resample.py` (`resample_ohlcv`), documented inline there too.
**Status:** open, not fixed.

**The issue:** `"1D"` bars and roll-day assignment (`infra.relative`) now bucket by the
exchange's trading day (e.g. CME rolls at 16:00 CT / a UTC-shifting boundary), but
intraday timeframes (`1m`..`4h`) still bucket on UTC-clock-aligned boundaries (00:00,
04:00, ... UTC) - unchanged, out of scope for that fix. Since a CME roll no longer
falls on a UTC-clock boundary, a coarsened intraday bucket (most likely `4h`, possibly
`1h`) that happens to contain the *exact moment* of a roll can span two different
absolute contracts. `resample_ohlcv`'s `"first"` aggregation of the `contract` column
then silently reports only the first contract's label, and the bucket's OHLC values are
computed across bars from two different underlying instruments.

**Why it's low priority:** it only affects one coarsened intraday candle, at the one
moment a roll actually happens (quarterly at most for a calendar-ranked `.c.N` series;
occasional, and now smoothed, for `.v.N`). The raw 1-minute data on disk, the `"1D"`
daily bars, and the roll-day assignment itself are all unaffected and correct - this is
purely a coarsened-intraday-chart cosmetic/precision issue.

**Fix options considered, not yet chosen:**
1. When building an intraday bucket, also group by `contract` (not just the time
   bucket) so a bucket straddling a roll splits into two shorter candles instead of
   blending them. Complication: `pd.Grouper`-based grouping labels both split pieces
   with the *same* bucket-start timestamp, so the two rows would collide on the same
   x-value - needs each split segment re-labelled by its own first bar's real
   timestamp (moves away from a fixed resample grid toward variable-width candles).
2. Leave the bucket as-is but visibly mark it (e.g. a distinct dashboard hover note
   "contains a roll") instead of silently showing one contract's label.
3. Do nothing further - document as a known, accepted limitation (current state).

**Next step if picked back up:** decide with the user which of the above (or another
approach) before implementing option 1's re-labelling scheme, since it changes what an
intraday candle's x-value means.

---

## ICE open interest can misattribute a same-day re-publish when `ts_ref` is null

**Found:** 2026-09-21, while building the daily settlement/OI pipeline (CLAUDE.md 8).
**Where:** `infra/processing/statistics.py` (`resolve_trading_day`).
**Status:** open, not fixed.

**The issue:** `resolve_trading_day` prefers Databento's `ts_ref` field (the exchange's
own reference date for a statistic) and falls back to `trading_day(ts_recv, dataset)`
only when `ts_ref` is null. For OPEN_INTEREST specifically, `ts_ref` correctly points to
the PRIOR trading day (OI published one morning reports the previous session's
close) - but real ICE Gilt data shows `ts_ref` is populated on the FIRST open-interest
update of a session and null on a later same-value re-publish. When that happens, the
fallback computes `trading_day(ts_recv, dataset)` for the re-publish, which lands on the
CURRENT day - even though the value is still describing the PRIOR day's open interest.
Concretely (real data, 2025-03-12, ICE Gilt `R   FMM0025!`): an OI update at 11:05 UTC
carries `ts_ref = 2025-03-11` (correct); a second update at 18:00 UTC with the *same*
quantity (1,051,182) carries `ts_ref = NaT`, and the fallback would tag it 2025-03-12 -
a spurious row, since it's really still Mar 11's figure being re-broadcast.

**Why it's low priority:** narrow - one venue (ICE), one stat type (open interest), and
only triggers when `ts_ref` happens to be missing on that specific message (CME's OI
`ts_ref` was populated on every sample checked; Eurex doesn't publish OI settlement in
this shape at all in the samples checked). `last()`-per-day aggregation in
`clean_daily_statistics` means the spurious row would just carry the same (correct)
quantity value under the wrong day label - a duplicate/misdated row, not a wrong number.

**Fix options considered, not yet chosen:**
1. When `ts_ref` is null for an OPEN_INTEREST row, compare its `quantity` to the most
   recent OI row (any day) for the same ticker; if identical, treat it as a re-publish
   of that same trading day rather than falling back to `trading_day(ts_recv)`.
2. For OPEN_INTEREST specifically, always prefer the LAST non-null `ts_ref` seen that
   trading session rather than per-row fallback, since a session's OI is one fact
   republished, not a new one each time.
3. Do nothing further - the spurious row carries a correct value under a wrong day
   label, and de-duplication on `["timestamp", "ticker"]` means it doesn't corrupt
   anything else; a downstream reader who diffs consecutive days would see one day
   with a repeated/flat OI reading, not a nonsense number.

**Next step if picked back up:** get a few more real ICE (and Eurex) samples across
different days to see how often the null-`ts_ref` case recurs before choosing a fix -
this was observed on a single sample day, not yet characterized at scale.

---

## `infra.pipeline.options.load_options` has no storage-path override, risking real-DB pollution in tests

**Found:** 2026-09-21, while building the daily options settlement pipeline
(`infra/pipeline/daily_options.py`).
**Where:** `infra/pipeline/options.py` (`load_options`).
**Status:** open, not fixed.

**The issue:** `load_options` (the 1-minute options parent function) does not expose
`root`/`coverage_file`/`directory` parameters the way `load_futures`,
`load_daily`/`load_daily_options` (after this same fix was applied there) all do - it
always writes to and reads from the real `OPTIONS_DIR`/`OPTIONS_COVERAGE_FILE`/
`DEFINITIONS_DIR` paths. A test that monkeypatches the API and calls `load_options`
directly (the natural, obvious way to test it) would silently write fake data into the
real `~/Database` on disk, with no way to redirect it to a `tmp_path`.

**How this was found:** `load_daily_options` (built alongside this pipeline, mirroring
`load_options`'s shape) initially had the exact same gap. A test for it ran against the
real default paths and wrote 2 rows of fake settlement data into the real
`~/Database/Daily/Options` and a fake `SR3.OPT_2025-03-12.parquet` into the real
`~/Database/definitions` before being caught and cleaned up (no real data was actually
overwritten - both were fresh files - but it could have been). `load_daily_options` was
fixed to accept `definitions_directory`/`root`/`coverage_file` overrides;
`load_options` itself was left as-is, since fixing it wasn't part of this pipeline.

**Why it's not fixed now:** out of scope for the daily-options-settlement work; touching
already-shipped `infra/pipeline/options.py` for an unrelated reason adds review surface
to a change that wasn't asked for.

**Fix:** add `directory: Path = DEFINITIONS_DIR`, `root: Path = OPTIONS_DIR`,
`coverage_file: Path = OPTIONS_COVERAGE_FILE` parameters to `load_options`, threaded
through to its internal `load_definitions`/`plan_options_update`/
`fetch_and_store_options`/`read_options_from_disk` calls - the exact same mechanical
change already applied to `load_daily_options` (see its current signature for the
pattern to copy).

**Next step if picked back up:** apply that same signature change, then check whether
any existing test already calls `load_options` against real default paths (a search
for `load_options(` in `tests/` at that time didn't turn up an end-to-end test of it at
all - if one gets added later without this fix, it's at risk of the same leak this
entry describes).

---

## Daily cycle has no exchange holiday calendar - can't tell a holiday from missing data

**Found:** 2026-09-28, while building the daily cycle's presence check (CLAUDE.md 12).
**Where:** `infra/cycle/px.py` (`_check_present`, `_check_dataset_present`).
**Status:** open, worked around.

**The issue:** test (a) "today's data is there" needs to know which days each exchange
was open. The project only knows weekends. On an exchange holiday no contract of that
dataset has a settlement - indistinguishable from "the run happened before settlement
was published" or "Databento hasn't made it available yet".

**Current workaround:** `px_present` (fail) only judges datasets that published at least
one settlement that day, so a holiday can't fail the cycle; `px_dataset_present` (warn)
reports any dataset with nothing at all. Consequence: a genuinely missing whole-dataset
day only warns - it does not fail the cycle.

**Fix options:** (1) a real exchange calendar, e.g. `pandas_market_calendars` (new
dependency; its CME rates / CBOT / Eurex calendars would need verifying against the
exchanges' own holiday notices, per this project's verify-first habit); (2) a hand-kept
holiday list per dataset in `infra/config.py` with source URLs, like `FOMC_MEETINGS`.
Either lets `px_dataset_present` become a hard `fail` on non-holidays.

---

## Bond futures have no DV01 (so no pnl-per-DV01) - needs a CTD model

**Found:** 2026-09-28, building the daily cycle's bmk step (CLAUDE.md 12).
**Where:** `infra/cycle/bmk.py` (`_dv01`, the `RISK_MODELS["DV01"]` entry).
**Status:** open, deliberately deferred - a CTD model is planned as a separate side
project.

**The issue:** a bond future's DV01 is the cheapest-to-deliver bond's DV01 divided by its
conversion factor (roughly - plus delivery-option effects). That needs cash-bond
reference data (deliverable basket, coupons, maturities, conversion factors) and CTD
prices/yields, none of which this project sources. STIR futures don't have the problem
(price = 100 - rate, so DV01 = point value x 0.01 exactly).

**Update 2026-10-02 - the inputs now exist** (CLAUDE.md 18): deliverable baskets with
per-contract conversion factors (`Reference/Treasuries/FuturesBaskets`: CME's own files
from 2023-12-09, computed and cross-checked before that), coupons and maturities
(`Reference/Treasuries/Securities`), and daily END OF DAY prices / yields per CUSIP
(`Daily/TreasuryPrices`, from 2008). Repo for carry: SOFR (FRED). So the model is now a
build, not a data problem: forward price of each deliverable to delivery, implied repo,
cheapest-to-deliver, forward DV01 / conversion factor, with the switch option via a
parallel yield bump across the basket.

**Current state:** every bond-futures risk row is stored with `value = NaN` and the reason
in `method`; the warn-level `dv01_coverage` check lists the affected roots every run.
Bond futures pnl in CURRENCY is unaffected and real (settlement change x point value);
only `pnl_per_dv01` is NaN for them.

**When the CTD model lands:** add it as the bond branch of `_dv01` (or a separate entry in
`RISK_MODELS` dispatched by category), backfill `bmk_risk` then `bmk_pnl` over history -
`pnl_per_dv01` fills in with no other change, since pnl already divides by the prior
day's stored DV01.

**Interim option considered, not taken:** an empirical DV01 from regressing futures
price moves on a benchmark yield series (would need a yield feed - the natural first
user of `backfill_daily_raw_data`). Rejected for now in favour of doing it properly.

---

## UK par yields derived from the BoE spot curve don't reproduce the BoE's own par series

**Found:** 2026-09-30, building the cash-bond px pipeline (CLAUDE.md 13).
**Where:** `infra/processing/bond_curves.py` (`PAR_METHODS`, `par_from_spot`), config
`BOND_CURVES["UK"].par_method`.
**Status:** open. The method is modular by design (user decision 2026-09-30) so it can
be swapped once the gap is understood.

**The issue:** the BoE publishes its nominal gilt curve as SPOT (zero-coupon) yields on
a 0.5y grid (0.5-40y), and par yields only for 5/10/20y (IADB `IUDSNPY`, `IUDMNPY`,
`IUDLNPY`). We derive par for all seven tenors from the spot grid (semi-annual coupons,
discount factors straight off the grid). Against the BoE's own par series, over
2026-08/09 (same dates, verified not a date offset): 5y -0.6bp (sd 0.5), 10y -3.6bp
(sd 1.5), 20y +1.6bp (sd 0.9). Reading the spot rates as continuously compounded is worse
overall (+5.1 / +3.1 / +9.3bp), so `semiannual_from_spot` is the default; the BoE's site
doesn't state the spot curve's compounding.

**Why it matters:** the goal is a curve matching what a desk sees (e.g. Bloomberg); a few
bp at 10y is material for RV. Not yet checked against Bloomberg at all (no access here).

**Options:** (1) find the BoE's par methodology (coupon dates on the real gilt cycle -
7 Jun/7 Dec - rather than valuation date + 0.5y steps? a different spot compounding?)
and add it as a new `PAR_METHODS` entry; (2) use the BoE's own par for 5/10/20y and
derive only 2/3/7/30y (rejected for now: two methods on one curve); (3) calibrate
against Bloomberg values for a few dates, once the user provides them.

---

## Bad-print rule can't judge a tenor whose typical move is exactly 0

**Found:** 2026-09-30 (CLAUDE.md 13).
**Status:** open, minor.

The bad-print rule can't judge a tenor whose typical daily move is
exactly 0 - Treasury quotes 2 decimals, so the US 2y during zero rates (65 days,
2020-21) was skipped. (The related 2-decimal effect - a quiet tenor's z-score inflated
vs its neighbours' - IS handled: `infra.cycle.px_bonds.BOND_MIN_ABS_DEV`.)

---

## Nowcast: calendar-sourced releases have no LIVE source; Richmond Fed has none at all

**Found:** 2026-09-30/10-01. **Where:** `infra/pipeline/econ_calendar.py`, `infra.config.CALENDAR_PAGES`.
**Status:** open.

**The issue:** ISM ×2, S&P Global PMIs ×3, MNI Chicago, the Conference Board, NFIB and NAR
existing home sales have no official free history. Their history now comes from archived
MarketWatch calendar pages (Wayback Machine, `infra/models/CLAUDE.md` 3a). That is a
HISTORY source only:
*   Its capture timing makes it about 67% reliable for T−1 by the 06:00 ET run (measured
    Apr–Sep 2026; median 12.7h, 90th percentile 55h after a 10:00 ET release).
*   The live MarketWatch page answers scripts with HTTP 401.

So these series go stale after each harvest, and `releases_fresh` warns.

**Options (user, 2026-10-01: build the history first, then look for a live source):**
(1) a calendar page that serves plain HTML to a script (Briefing.com, Yahoo Finance,
broker calendars - none checked yet); (2) triggering a Wayback "Save Page Now" capture
before the run (a daily public capture under our IP - needs the user's OK); (3) a
browser-driven fetch (heavy for a headless 06:00 job).

**Also still missing:** the Richmond Fed composite (`RCHSINDX`). It is free from the
Richmond Fed itself, but not on FRED and not on the MarketWatch calendar; it needs its own
client. The earlier plan of free proxies (Kansas City/Dallas Fed surveys, Philly non-mfg,
NY Fed services, new home sales/permits) is no longer needed for the nine calendar
releases, but would still add breadth to the thin Manufacturing/Housing blocks.

**Older history is patchy:** 2009–11 has few captures; 2012–19 has 25–47 weeks a year.
A month whose release week was never archived is only recovered from the next month's
"previous" (dated late), and for the S&P PMIs (actuals only) not at all. Any second
archived calendar (Briefing.com, Yahoo) would fill gaps and cross-check values.

---

## Nowcast: history published before ALFRED's vintage archive is only pseudo-real-time

**Found:** 2026-09-30. **Where:** `infra/processing/releases.py`, `infra/models/nowcast/panel.py`.
**Status:** open, not yet measured on real data.

**The issue:** ALFRED keeps vintages only from some date per series (for many series this
is the 1990s or later). Every observation older than a series' first archived vintage
arrives in one bulk "publication" on that first vintage day, already revised. For
estimation this doesn't matter: the model is fit on a snapshot, and revised history is
standard. It does matter for a real-time BACKTEST (news, or `nowcast_history`) over such
dates. There, as-of views before the first vintage have no data at all, and the first
vintage day shows a huge spurious batch of "news".

**Measured 2026-09-30 (first archived vintage per series):**
*   GDP (`GDPC1`): 1991-12.
*   `PAYEMS`: 1955. `UNRATE`: 1960. `HOUST`: 1960. `CPIAUCNS`: 1949. `INDPRO`: 1927.
*   `DGORDER`: 1999. `UMCSENT`: 1998. `PCEPI`: 2000.
*   `ICSA`: 2009. `CFNAI`: 2011. `WHLSLRIMSA`: 2013. `PPIFID`: 2014. Empire: 2014.
    Philly: 2015.
*   Retail ex autos and gas (`MARTSSM44W72USS`): 2018-05.
*   ADP: 2022-08 (ADP's relaunch).

So a fully real-time panel only exists from mid-2018, or from 2022-08 with ADP.

**Options:** (1) start backtests after every model series' first vintage (simplest; a
check could enforce it); (2) impute each old period's publication day as
period end + the series' typical first-print lag (median of the archived era), marked
`pseudo=True`; (3) Philly Fed's Real-Time Data Set for the few series it covers. Measure
each series' first vintage date once real data is on disk, then decide.

---

## Nowcast: weekly claims only enter as COMPLETE months

**Found:** 2026-09-30. **Where:** `infra/models/nowcast/panel.py` (`to_monthly`, `"W"`).
**Status:** accepted for v1.

**The issue:** a month's claims value exists only once every week ending in it is
published. A partial month would change as weeks arrive, and each new week would show up
as a revision rather than news. The cost is that mid-month weekly claims prints move
nothing until the month completes. The NY Fed has the same monthly treatment. The proper
fix is a weekly-in-monthly measurement equation: each week observed at its month, loading
on that month's factor with its own idiosyncratic component, so every week is its own
news.

---

## Nowcast (version d): soft-prior penalty treats an AR(1) idio as white noise

**Found:** 2026-09-30. **Where:** `infra/models/nowcast/dfm.py` (`m_step`, penalized rows).
**Status:** accepted approximation.

**The issue:** with `idio="ar1"`, a monthly release's idiosyncratic component is in the
state, so the exact loading regression has only the tiny `KAPPA` noise. A MAP penalty
`KAPPA/tau^2` would do nothing. The penalty is therefore scaled by the idio's
unconditional variance `sig2/(1-rho^2)`, which treats the idio as the regression noise and
ignores its serial correlation. The likelihood and E-step stay exact; only the strength of
the shrinkage is approximate. With `idio="iid"` it is the exact MAP. Exact fix: a
GLS-weighted penalty using the idio's AR(1) precision, or a Gibbs sampler for (d).

---

## Releases: a value DELETED in a later vintage is not recorded

**Found:** 2026-09-30. **Where:** `infra/processing/releases.py` (`drop_unchanged`).
**Status:** open, rare.

**The issue:** the store keeps a row per changed value. If a later vintage REMOVES an
observation (FRED `realtime_end` closes with no successor, e.g. a discontinued period), the
store still shows the last value as current. The `realtime_end` of the last row is not
stored. Fix: store a tombstone row (value NaN) at the day the value stopped being current,
and have `snapshot` drop NaN.

---

## Nowcast: single-series blocks are degenerate (Housing, Price_WholeSales)

**Found:** 2026-09-30, on the first real-data fit.
**Where:** `infra/models/nowcast/spec.py` (`factor_structure`), `infra.config.MACRO_RELEASES`.
**Status:** open, needs a user decision.

**The issue:** once the proprietary series are dropped, `Activity_Housing` has one monthly
member (housing starts, plus GDP) and `Price_WholeSales` has one (PPI). Each block factor
then IS that series: its idiosyncratic variance goes to the floor, its `signal_share` in
the news table is about 1.0 (seen: housing starts 0.998, PPI 0.996), and the factor adds a
free parameter set but no pooling. Existing home sales was Housing's second member; a
free proxy (new home sales, permits: the proxies entry above) would fix Housing.

**Options:** (1) require at least 2 monthly members per block and fold smaller blocks into
the global factor, or into their Cat1 parent (PPI into a single Price block); (2) add the
housing proxies; (3) accept it (the nowcast is fine, only that block's signal share is
not meaningful).

---

## Nowcast: per-block GDP attribution is fragile (correlated block factors)

**Found:** 2026-09-30, on real data. **Where:** `infra/models/nowcast/dfm.py`, `spec.py`.
**Status:** open, needs a user decision on the defaults for versions b/c/d.

**The issue:** as specified (b/c/d = one factor per category, full VAR, no global factor),
the block factors are highly correlated, and GDP's loadings on them come out with
offsetting signs:
*   b: Consumption −0.74, OtherBusiness +0.60.
*   Unmasked c: block contributions of −12.8pp against +13.2pp.

The nowcast itself is stable, but "which block drives GDP" is not interpretable.
`global_factor=True` + `factor_dynamics="independent"` (the NY Fed / Bańbura–Modugno
structure: blocks carry only local co-movement, uncorrelated with the global cycle)
brings block contributions down to tenths of a pp. GDP's own loadings are still tiny and
of mixed sign, because GDP is quarterly, noisy and loads on 5 factors.

**Real-data fit (in-sample GDP R², 1990–2019, COVID excluded):**

| Configuration | With table transforms | With `use_transforms=False` |
|---|---|---|
| a | 0.34 | 0.44 |
| b | 0.58 | – |
| c | 0.54 | – |
| c + global + independent | 0.51 | 0.59 |
| d, τ=0.3 | 0.59 | – |

**Next step:** the user picks the defaults. Then run a real-time out-of-sample backtest
(re-estimate each quarter on the vintage as of that day, compare with the first GDP print,
from mid-2018: see the pseudo-real-time entry above) to choose between versions on
forecast accuracy rather than in-sample fit.


---

## Intraday WIRP: on an FOMC decision day, that meeting is already treated as past

**Found:** 2026-09-30, building intraday WIRP (CLAUDE.md 14).
**Where:** `infra.pipeline.wirp.build_schedule` (meeting inclusion: `end_date > today`,
day-granular) as called by `intraday_schedules` at each grid time.
**Status:** open.

**The issue:** a meeting counts as "upcoming" only while its decision date is after the
as-of DAY, so on the decision day itself it's already treated as past for the whole
day - including the hours before the 14:00 ET statement, which is exactly the window an
intraday event study wants. That meeting's month is then priced from settlements (the
past-chain rule), and it drops out of the displayed schedule.

**Why not fixed now:** doing it right needs the decision INSTANT (14:00 America/New_York,
18:00 or 19:00 UTC by DST) compared with the intraday time - a timezone conversion
outside `infra/dashboard`/`infra.trading_calendar`, which CLAUDE.md 7 forbids unless it
lives in the trading-calendar layer. **Options:** (1) add an FOMC-decision-instant helper
to `infra.trading_calendar` (the sanctioned home for exchange/calendar time logic) and
make meeting inclusion instant-based when `build_schedule` gets an intraday `today`;
(2) store decision instants in UTC in `FOMC_MEETINGS` (verified per meeting).

---

## Inflation: bulk snapshots: vintages only from the first snapshot, versions only if seen

**Found:** 2026-09-30, building `infra/pipeline/bulk_series.py`. **Status:** open, by design.

**The issue:** BLS and BEA serve only the latest revised history, so vintages exist only from
the first snapshot on. On first sight every historical value is stamped with that snapshot's
publication day: PCE 2026-09-30, CPI 2026-09-11, PPI 2026-09-10. A
point-in-time read before that day returns nothing from these stores. A version is also only
captured if a run happens while it's current. A monthly file and a daily cycle make a miss
unlikely. A same-day correction overwrites that day's rows, since the store is
day-granular. Like the FRED releases store (entry above), a value DELETED by a later version
is not recorded. **Options:** for earlier vintages of the headline series, use ALFRED
(`MACRO_RELEASES`). For components, BLS/BEA archived release files (BEA's "Archive" of NIPA
releases, BLS's archived CPI detailed reports) could be back-filled one release at a time,
if deep component vintages turn out to matter.

---

## Inflation: CPI weights - only Table 1 (U.S. city average) is stored

**Found:** 2026-09-30, building `infra/pipeline/cpi_weights.py`. **Status:** open.

**The issue:** BLS's relative-importance files also carry Tables 2-7: metro areas, regions,
population-size classes, and the areas' own weights. `parse_xlsx` reads only "Table 1". The
area tables exist only in the 2020+ xlsx files (the .txt archives are Table 1 only), so
area weights would have history from 2020. **Why not now:** national CPI modelling needs
Table 1. **Fix:** a `table` column, and `parse_xlsx` over sheets 2-7. The column headers
pair each area with CPI-U/CPI-W, and area names need mapping to `cu.area` codes.

---

## Daily statistics requests are slow over long ranges, and the relative loaders fetch one contract at a time

**Found:** 2026-10-01, back-filling cleared volume (CLAUDE.md 5, 8) into the daily store.
**Where:** `infra.pipeline.daily.fetch_and_store_daily`, called per contract by `infra.pipeline.relative.volume_ranked_mapping` and `load_relative_daily`.
**Status:** open, not fixed.

**The issue:** a single-contract `statistics` request over ~15 months took ~230s of
Databento server time (SR3Z6, 2026-10-01; storing it took 0.1s). The cost is tiny
(~$0.000003 per contract-day); the wait is the problem. A `.v.N` series fetches its
candidate pool's statistics one contract at a time, so its FIRST request over a long
window can take minutes per contract. Later requests are instant: Rule 2.1 caches every
fetched range.

**Why not fixed now:** it only bites the first load of a long, never-fetched window, and
the daily cycle's own px step already fetches concurrently (`infra.cycle.px`). The one-off
back-fill worked around it with one request per batch of up to 40 contracts, split by
symbol afterwards (session scratch script, not kept).

**Options:** (a) batch the pool's contracts into one `get_range` (symbols list) and split
by `symbol`, as the back-fill did. That's the most effective, but needs a per-symbol
coverage split, since contracts' gaps differ. (b) Fetch the pool concurrently, as
`infra.cycle.px` does (network-only `fetch_daily_raw` in threads, then sequential
`store_daily_raw`). (c) Both.

---

## Calendar / auctions: remaining gaps

**Found:** 2026-10-01. **Where:** `infra/pipeline/release_calendar.py`, `infra/pipeline/tsy_auctions.py`.
**Status:** open. Wired into the daily cycle 2026-10-02 (`infra/cycle/raw_reference.py`); still to do:
*   Agency full-year schedules (BEA, Census, Fed G.17; BLS with `BLS_CONTACT_EMAIL`) are
    reachable but not fetched. FRED already dates these releases ~3 months ahead.
*   No forward date at all for the S&P flash PMIs (no rule; S&P's release calendar page
    returns 403 to scripts) or UMich (its "release schedule" page lists past reports only).
*   Six archived calendar pages (2025-26, odd query-string variants) are listed by the
    archive but return 404. Their weeks are covered by neighbouring days' pages.
*   The harvested calendar store keeps only each row's LAST capture, so for past weeks a
    release is known only from its own day: the nowcast's "coming up" view cannot be
    replayed historically from it. A live calendar source would record first sightings
    going forward (`known_from`).
---

## Reference: bills are not in the on/off-the-run map

**Found:** 2026-10-01, building the Treasury OTR map (CLAUDE.md 18).
**Where:** `infra.config.TREASURY_OTR_TENORS`, `infra.processing.treasury_otr`.
**Status:** open, not fixed.

**The issue:** the map ranks ORIGINAL issues by their original term, which is right for coupons (a 10y reopening stays the same series) and wrong for bills. Since 2016 the only original-issue bill terms are 26, 17, 52 and 8 weeks; the 4-week and 13-week bills are auctioned as REOPENINGS of older 26/52-week bills (today's 13-week bill is a CUSIP first issued as a 26-week bill). The reference table holds one row per CUSIP (the original issue), so it can't see those auctions. **Why not now:** the consumers so far (futures baskets, coupon benchmarks) only need coupons. **To do:** rank bills from the auction rows themselves (each auction's own ``security_term``, reopenings included) - read through `infra.pipeline.tsy_auctions.read_auctions(nominal_only=False)` - and add them to the map, e.g. as tenors `4w`/`13w`/`26w`/`52w`.

---

## Reference: futures basket rules before 2023-12 are not verified

**Found:** 2026-10-01, computing Treasury futures baskets (CLAUDE.md 18).
**Where:** `infra.config.TREASURY_BASKET_RULES`, `infra.processing.futures_baskets`.
**Status:** open, not verified.

**The issue:** computed baskets (`source = "computed"`) apply TODAY's deliverable-grade rules, which are validated against every CME file from 2023-12-09 on (CME's own baskets and factors). Before that there's no CME file to check against, and the rules may have changed: in particular ZN's remaining-term window (now 6.5-8y), and the contracts' start dates (TN first listed 2016, Z3N relaunched 2021, TWE more recently - a computed basket for a contract that didn't trade yet means nothing). **Why not now:** CME's site blocks this machine (2026-10-01), and its FTP keeps files only from 2023-12-09. **Options:** (a) give each rule an effective contract month once the history is confirmed (CME contract-spec PDFs or rule-change notices, read by hand, or archived copies via the Wayback Machine - `infra.api.wayback_client`); (b) check computed baskets against CTD behaviour (implied repo of the cheapest bond vs SOFR) in the old period.

---

## Tails: history harvest unfinished; failure analysis pending

**Found:** 2026-10-01. **Where:** `scripts/backfill_auction_tails.py`, `infra/processing/auction_tails.py`.
**Status:** open.

**State at the end of 2026-10-01:** 737 archived recaps processed (ZeroHedge 700 of 1,111;
ForexLive's 229 not reached), 295 passed both checks, 294 auctions with a checked tail
(2011-08 .. 2023-02). The run stalled at 22:06 when the Internet Archive degraded: CDX
timed out, playback returned 503 and then redirected to 7.8KB stubs. It was stopped and
can be resumed: rerun the script. Passed articles are never re-fetched; everything else is
retried under `PARSER_VERSION` 3.

**Still to do once it completes:**
*   **Diagnose the 79 "failed check" articles** (45 of them on the modern site): is it a
    mis-read WI or high yield, a stated tail that contradicts high yield - WI, or a tail
    above 15bp?
*   **Old-site pass rate (2011-18) was 31%.** Measure what v3 recovers (URL-path dates,
    high yield in the title). Many old-site captures lack the article text altogether.
*   **Coverage report** by year and tenor (`backfill_auction_tails.py --report`), against
    the archive-coverage estimate in `CLAUDE.md` §17.

---

## Ops: production runs from a working tree that several sessions edit at once

**Found:** 2026-10-01. **Status:** open (workflow).

**The issue:** the daily cycle (launchd/Prefect, 06:00 New York) imports whatever is in
this checkout, and on 2026-10-01 four sessions were editing it with uncommitted work. At
22:41 one session's half-finished edit left `infra/config.py` unimportable (`CME_TCF_DIR`
used `RAW_DATA_ROOT` before its definition). Every test failed until it was fixed a
minute later. Had that state been live at 06:00, the whole run would have failed and the
day's vintage would have been lost.

**Options:**
1. Run production from a separate clean checkout or worktree at a known commit, updated
   on purpose.
2. Have the scheduled run refuse to start on an import failure, alert immediately, and
   retry (the watchdog only notices a missed vintage hours later).
3. Have sessions work in worktrees and merge to main.


---

## Derived: OTR yields and swap closes are not in the daily cycle yet

**Found:** 2026-10-02, wiring the Treasury reference data and prices into the cycle (CLAUDE.md 12, 16, 18).
**Where:** `infra.pipeline.bond_yields.build_otr_yields`, `infra.pipeline.swap_closes.backfill_swap_closes`; the `derived` step (`infra/cycle/derived.py`, `DERIVED_METRICS`).
**Status:** open, deferred (user decision 2026-10-02: hold `derived` until the intraday data is scheduled).

**The issue:** both are computed by hand today, so their stores only extend when someone runs them. `Derived/OTRYields` stops at the last `build_otr_yields()` and `Derived/SwapCloses` at 2026-09-30. **Why not now:** the futures-adjusted swap closes need each day's bond-futures `bbo-1m` quotes for the `.v.0` contracts. That's a daily intraday fetch, which belongs with scheduling the intraday cycle (`infra/cycle/intraday.py`, still run by hand). Wiring only half now would leave the adjusted closes silently stale. **To do, together:** (1) schedule the intraday fetch, including bond-futures `bbo-1m` for `SWAP_HEDGES` roots' `.v.0` (a few cents a day); (2) add OTR yields and swap closes (pure + adjusted, recomputing the last `SWAP_CORRECTION_DAYS` for late corrections) to `DERIVED_METRICS`, with presence / sanity / revision checks; (3) the OTR yields' check vs CMT (within a few bp, outside auction-to-issue days).

---

## Bonds: FedInvest has no END OF DAY prices on 2014-09-12 and 2014-11-21

**Found:** 2026-10-02, backfilling Treasury prices 2008-09..2015-12 (CLAUDE.md 18).
**Where:** `Daily/TreasuryPrices`; `infra.pipeline.treasury_prices.store_prices_day`.
**Status:** open, by design for now.

**The issue:** on those two days FedInvest's page lists every security (362 and 369 rows) with the ~1pm BUY/SELL prices but END OF DAY all zero, so the pipeline (correctly) doesn't store them as complete days and leaves them uncovered. Every other business day 2008-09-02..2026-09-30 is stored or a holiday. **Why not fixed:** substituting the 1pm price would silently mix a different time of day into a 3:30pm-consistent series. **Options:** (a) leave the gaps (current); (b) store those days with `price_eod` NaN and `price_buy`/`price_sell` filled, marked, and cover them; (c) fill END OF DAY from the 1pm price plus the day's CMT move. The cycle's 30-day gap lookback never re-asks them; only a manual backfill over 2014 would.

---

## Swap closes: the futures-adjusted method's hedge gaps (short end, EUR, GBP)

**Found:** 2026-10-01, building the futures-adjusted swap closes (CLAUDE.md 16).
**Where:** `infra.config.SWAP_HEDGES`, `infra.pipeline.swap_hedge`.
**Status:** open, not fixed.

**The issue:** (1) 1-3y USD swaps are hedged with ZT, a 2y Treasury future - a reasonable proxy over a <=90-minute move, but the SR3 strip is the right hedge for the short end (it IS the SOFR curve). (2) EUR and GBP have no adjusted closes at all: Bund futures (Eurex, data from 2025-03) and gilt futures (ICE, disabled in the cycle) have no stored intraday quotes. **Why not now:** both need new `bbo-1m` backfills (SR3; Eurex/ICE), not yet priced, and the intraday fetch isn't scheduled. **To do:** price the SR3 `bbo-1m` backfill for the strip covering 1-3y; hedge 1-3y with the strip's forward-weighted move (no hedge ratio needed: price = 100 - rate); for EUR/GBP, decide whether Eurex/ICE quotes are worth adding.

---

## Treasury: no prices for a new issue between its auction and issue date

**Found:** 2026-10-01, building the OTR yield benchmark (CLAUDE.md 18).
**Where:** `Daily/TreasuryPrices`, `infra.pipeline.bond_yields` (the OTR series uses the "issue" convention).
**Status:** open, no free source.

**The issue:** a new issue trades when-issued from its auction but has no FedInvest price until its issue date, so the OTR yield series switches bond at the issue (CMT switches at the auction; the two differ by up to ~10bp at the 2y when the front is steeply inverted - 2023). **Checked 2026-10-02:** FINRA TRACE disseminates exactly those trades (on-the-run coupons from the day after the auction, end of day, from 2024-03-25), but free access is a web grid for personal non-commercial use and its terms forbid automated copying; automated access is the paid end-of-day file ($750/month). User decision: stay put. **Free option not yet built:** on auction-to-issue days, CMT's yield at the tenor is essentially the when-issued yield (CMT switches at the auction, bid side ~3:30pm) - nearly exact for the 2y, close for the others - so an "auction"-convention OTR yield series could use CMT for those days (yields only, no prices).

---

## Treasury: prices reach the daily cycle a day late

**Found:** 2026-10-02, the first scheduled run with Treasury prices (CLAUDE.md 12, 18).
**Where:** `infra.cycle.px_treasuries`; schedule `infra.cycle.flows.SCHEDULE_CRON` (06:00 New York).
**Status:** open, acceptable for now.

**The issue:** FedInvest posts day D's END OF DAY between 06:00 and ~10:00 New York on D+1 (absent at 06:00, posted by 09:57 on 2026-10-02), so the 06:00 run leaves D pending and stores it a run later - Treasury prices lag the futures settlements by a day. Nothing consumes them same-day yet. **Options:** (a) leave it; (b) a small extra Treasury-only fetch around 11:00 New York (a separate schedule calling `backfill_daily_treasury_px`); (c) pin the posting time down over a few days first.

---

## Treasury: OTR yields run ~1-1.5bp below CMT in 2008-2015

**Found:** 2026-10-02, backfilling Treasury prices to 2008 (CLAUDE.md 18).
**Where:** `Derived/OTRYields` vs FRED's `DGS*` (CMT).
**Status:** open, unexplained (not a known bug).

**The issue:** 2016-2026 the OTR yields sit -0.3..+0.3bp from CMT on average; 2008-2015 they run -0.9..-1.6bp (30y: 0.0), with about twice the dispersion. Plausible causes, none verified: the Treasury's CMT fitting method changed in December 2021 (quasi-cubic Hermite spline -> monotone convex); FedInvest's END OF DAY pricing may have been sourced differently then; crisis-era volatility. **To check if it matters:** compare a few 2010-2015 days' FedInvest prices with another source, and read the Treasury's notice on the 2021 method change.

---

## Repo: per-CUSIP spread to spline (specialness signal) - held for the curve-building sub-project

**Found:** 2026-10-02, planning the CTD model's financing inputs (CLAUDE.md 19).
**Where:** not built. Would be a derived store `Derived/SpreadToSpline`, keys `timestamp`, `cusip`.
**Status:** deliberately deferred (user decision 2026-10-02) until the curve-building sub-project exists, which will own the fitted Treasury curve.

**The idea:** a bond that is special in repo trades RICH (owning it earns cheap financing), so its yield sits below a smooth curve fitted to its neighbours. Spread to spline complements the NY Fed lending fee (`Daily/SecLending`): the fee says "expensive to BORROW" but exists only for bonds the Fed holds and dealers borrowed; the spread says "expensive to OWN" for every bond, every day, and separates "rich because special" (negative spread AND a lending fee) from "rich for other reasons" (negative spread, no fee). It is also, more or less, what picks the cheapest-to-deliver. **Inputs already exist:** FedInvest END OF DAY prices and yields per CUSIP from 2008 (`Daily/TreasuryPrices`), the reference table and the OTR map. **Design choices to make there:** which bonds enter the fit (usually off-the-runs only: exclude on-the-run, 1-old and < ~1y to maturity, since those carry the very effects being measured), the curve family (Svensson as the Fed's own GSW curve, or a smoothing spline on yields), a robust loss so one bad print can't bend it. **Kept forward-compatible now:** every per-CUSIP store is keyed `(timestamp, cusip)` (prices, lending), the lending signal is a pure function (`infra.processing.sec_lending.excess_fee`), so a later per-CUSIP specialness view only joins stores - nothing built now has to change.

---

## Cycle: the bad-print rule flags genuine STIR moves on shock days

**Found:** 2026-10-02, backfilling SR1 settlements 2018-2025 (CLAUDE.md 12, 20).
**Where:** `infra.cycle.bad_prints` / `infra.cycle.px.peer_outliers`; policy `DailyBackfillSpec.strict_ranks`.
**Status:** open - the 5 false treatments were removed from the adjustments log by hand; the rule is unchanged.

**The issue:** the rule (|z| >= 5, |peer deviation| >= 6, and the deviation reversing next session) was calibrated on 2025-2026 settlements, which had no emergency moves. On SR1's 2018-2025 history it flagged SR1H0 2020-03-03 (the emergency 50bp cut), SR1H0 + SR1J0 2020-03-09/10 (the crash days before the March 15 cut) and SR1H2 2022-02-10 (the hot CPI print), and set them to NA (ranks 0-1 are strict). All five are smooth, genuine paths: the front contract moves far more than its peers when the next weeks' policy is repriced, and partial retracements next day are normal in a shock. **Risk:** the scheduled run would do the same to ZQ / SR1 / SR3 front contracts on the next emergency move or big data surprise - NA where the data matters most. **Options:** exempt days with an FOMC decision (incl. unscheduled, `FOMC_MEETINGS`) or a top-tier release (CPI, payrolls - the release calendar, CLAUDE.md 17); require the reversal to undo most of the move rather than any of it; for monthly STIR fronts, check the print against the realised fixings it settles on (a front SR1 / ZQ contract's fair value is largely known); recalibrate on 2018-2026 now that the history exists.

---

## Financing: v1 simplifications to revisit

**Found:** 2026-10-02 (CLAUDE.md 20). **Where:** `infra/analytics/sofr_curve.py`, `infra/analytics/specialness.py`, `infra.config.FINANCING_MODELS`.
**Status:** open - documented v1 choices, each a candidate for a v2 layer.

* **Year-end turn imposed from history** (median of the last 3). Near a year-end the December SR1 contract, with few unfixed days left, implies the turn itself (2025: ~+19bp implied vs +16 actual vs 3bp imposed); fit the turn as a parameter once December is mostly fixed.
* **Month- and quarter-end spikes are smeared into the levels.** Measured harmless for financing averages (no premium on ordinary month-ends; quarter-ends ~0.5bp in the DVP-SOFR spread), but visible in fit residuals near month-ends.
* **Specialness forecasts slightly over-react** (realised moves ~0.85-0.9 per unit forecast): a shrinkage factor fitted out of sample would fix the slope.
* **Not borrowed = 0 specialness** assumes the Fed held the bond; holdings are only reported for bonds that were lent. SOMA holdings by CUSIP (the NY Fed's SOMA API) would remove the assumption.
* **The successor's issue date is expected from the tenor's median cycle until announced** (about a week before the auction); Treasury's tentative auction schedule (section 17) would give it a quarter ahead.
* **No delivery-squeeze effect:** the cheapest-to-deliver's specialness into delivery (open interest against available supply) is not modelled; flag it in the CTD model rather than forecast it.
* **SIFMA calendar approximated** by federal holidays + Good Friday (SIFMA sometimes recommends only an early close on Good Friday).

---

## Repo: the DTCC GCF index is a biased proxy for general collateral before 2018

**Found:** 2026-10-02 (CLAUDE.md 19).
**Where:** `Daily/Repo`, series `DTCC_GCF_TSY` (2005-2024).
**Status:** open - matters for financing before SOFR (total-return series and CTD history 2008-2018).

**The issue:** the only free Treasury GC series before 2018-04 is DTCC's GCF index, and GCF (an inter-dealer, blind-brokered market) trades above the broad market by a REGIME-DEPENDENT amount: over 2018-2024 it sat above SOFR by 0.3bp (2021-22, abundant reserves) to 7.7bp (2018, scarce reserves), and spikes 200bp+ at year-ends. A constant splice shift would be wrong in some years by several bp. **Options:** (a) use GCF as is before 2018 and accept a few bp of bias (simplest; small next to most carry calculations); (b) model the GCF-SOFR basis on reserves (Fed H.4.1, on FRED) and apply it backwards; (c) find another pre-2018 series - the NY Fed published indicative SOFR history from 2014-08 in a one-off release (not in its API), worth checking.

---

## Repo: the minimum lending fee schedule is inferred, not sourced

**Found:** 2026-10-02 (CLAUDE.md 19).
**Where:** `infra.config.SEC_LENDING_MIN_FEE`.
**Status:** open, low risk.

**The issue:** the program's minimum fee (the floor under every lending fee) was inferred from the data: each regime's level is its most common daily minimum, each switch the first day at the new level. It fits the data (96.9% of days have their lowest fee exactly at the floor, none below), and `sec_lending_sane` warns if a future fee ever goes below it (a schedule change), but the dates are not yet matched to the NY Fed's own announcements. **To do:** find the NY Fed's notices for each change (1999 start 150bp; 2001-09-18; 2003-06-26; 2004-07-01; 2007-08-21; 2008-10-08; 2008-12-18; 2009-04-08) and cite them in the config.

---

## Repo: 3 lent Treasuries predate the auctions store

**Found:** 2026-10-02 (CLAUDE.md 19).
**Where:** `infra.cycle.px_repo._check_lending_sane`; CUSIPs 912810CE6, 912810CC0, 912810CG1 (8.75% 2008, 8.375% 2008, 9.125% 2009 bonds, issued 1978-79).
**Status:** open, harmless - they only fail the "known Treasury" clause over a history run before 2002.

**The issue:** the Treasury reference data starts with the auctions store's first auction (1979-10-31); these bonds were issued earlier, so they're unknown to it. **Options:** leave it (they matured in 2008-09 and never appear in a daily window); or add a tiny static list of pre-1979 securities if the reference table ever needs to reach back that far.

