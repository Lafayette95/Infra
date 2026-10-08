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

## Derived: OTR yields, swap closes and futures snaps are not in the daily cycle yet

**Update 2026-10-07 - the futures-ADJUSTED swap closes are in `derived` too** (`swap_closes` metric: both methods, one hedge book per run, both methods' rows replaced over the recomputed window; USD, and EUR / GBP on Eurex / Long Gilt hedges - CLAUDE.md 16). The OIS curves switched to `method="best"` (adjusted where present, else pure) the same day, user decision, and their consumers were rebuilt (CLAUDE.md 16). Still open here: the OTR yields, the futures snaps.

**Update 2026-10-06 (later) - the hedge quotes ARE scheduled:** the daily `intraday` step fetches `bbo-1m` for every `SWAP_HEDGES` root's `.v.0` (root CLAUDE.md 14). That unblocks the futures-ADJUSTED swap closes in `derived` (`swap_closes` metric: `methods=("pure", "adjusted")`, pass the cycle-paths hedge book, replace both methods' rows) and then switching `OIS_CURVES["USD_SOFR"].method` to `adjusted` - not done yet (user's call: it changes the stored OIS curve and spreads). The ZQ intraday px / intraday WIRP and the futures snaps (which also need SR3 / non-front contracts' quotes) remain by hand.

**Update 2026-10-06 - the PURE swap closes are in `derived`** (metric `swap_closes`, user decision: step 1 of the swap-spread plan below), recomputing the last `SWAP_CORRECTION_DAYS` and replacing only pure rows. Still open here: the futures-ADJUSTED closes (hand build, `Derived/SwapCloses` adjusted rows stop at 2026-09-30 - the half this entry warned about, now an explicit known gap, not silent: `swap_closes_present` judges the pure method only), the OTR yields, the futures snaps.

**Found:** 2026-10-02, wiring the Treasury reference data and prices into the cycle (CLAUDE.md 12, 16, 18).
**Where:** `infra.pipeline.bond_yields.build_otr_yields`, `infra.pipeline.swap_closes.backfill_swap_closes`; the `derived` step (`infra/cycle/derived.py`, `DERIVED_METRICS`).
**Status:** open, deferred (user decision 2026-10-02: hold `derived` until the intraday data is scheduled).

**Added 2026-10-02:** the futures snap store (`Derived/FuturesSnaps`, CLAUDE.md 23) is in the same position - built by hand with `infra.pipeline.futures_snaps.build_futures_snaps`, from bbo-1m quotes that are themselves not fetched daily yet. When the intraday data is scheduled, `derived` should build the day's snaps BEFORE the swap closes and the basis models, which read them.

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
**Status:** (2) done 2026-10-07 - EUR / GBP adjusted closes (CLAUDE.md 16); (1) open.

**The issue:** (1) 1-3y USD swaps are hedged with ZT, a 2y Treasury future - a reasonable proxy over a <=90-minute move, but the SR3 strip is the right hedge for the short end (it IS the SOFR curve). (2) EUR and GBP have no adjusted closes at all: Bund futures (Eurex, data from 2025-03) and gilt futures (ICE, disabled in the cycle) have no stored intraday quotes. **Why not now:** both need new `bbo-1m` backfills (SR3; Eurex/ICE), not yet priced, and the intraday fetch isn't scheduled. **To do:** price the SR3 `bbo-1m` backfill for the strip covering 1-3y; hedge 1-3y with the strip's forward-weighted move (no hedge ratio needed: price = 100 - rate); for EUR/GBP, decide whether Eurex/ICE quotes are worth adding.

---

## Treasury: no prices for a new issue between its auction and issue date

**Found:** 2026-10-01, building the OTR yield benchmark (CLAUDE.md 18).
**Where:** `Daily/TreasuryPrices`, `infra.pipeline.bond_yields` (the OTR series uses the "issue" convention).
**Status:** open, no free source.

**The issue:** a new issue trades when-issued from its auction but has no FedInvest price until its issue date, so the OTR yield series switches bond at the issue (CMT switches at the auction; the two differ by up to ~10bp at the 2y when the front is steeply inverted - 2023). **Checked 2026-10-02:** FINRA TRACE disseminates exactly those trades (on-the-run coupons from the day after the auction, end of day, from 2024-03-25), but free access is a web grid for personal non-commercial use and its terms forbid automated copying; automated access is the paid end-of-day file ($750/month). User decision: stay put. **Free option not yet built:** on auction-to-issue days, CMT's yield at the tenor is essentially the when-issued yield (CMT switches at the auction, bid side ~3:30pm) - nearly exact for the 2y, close for the others - so an "auction"-convention OTR yield series could use CMT for those days (yields only, no prices).

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


---

## CTA: calibration rests on one published snapshot; no performance-based calibration yet

**Found:** 2026-10-02 (`infra/models/cta/CLAUDE.md` section 5).
**Where:** `infra.models.cta.config.CTA_MODELS["ubs2022_cal"]`, `infra/models/cta/compare.py`.
**Status:** open - the main next step for the CTA model.

**The issue:** UBS's note gives no EWMA speeds and no response function, so `ubs2022` uses
published defaults (Baz et al. 2015) and `ubs2022_cal` picks speeds and a response gain to
match UBS's own 2022-09-02 rates snapshot (16 numbers, 2 parameters, ONE date). That
reproduces UBS's model, not real CTAs, and could be a coincidence of that date.
**Options, best first:** (a) calibrate on REALISED CTA performance: build the model's
portfolio P&L (sum of position(t-1) x return(t), vol targeted) and choose parameters
maximising its out-of-sample correlation with a CTA index's returns (UBS: its proxy has 64%
correlation with BarclayHedge's index and 76% with SG's). Free series: the SG CTA / SG
Trend indices (daily; free download from SG's prime services index page historically -
verify access), BarclayHedge CTA index (monthly, free with registration), HFRX Systematic
Diversified (daily, HFR, registration), AQR's "Time Series Momentum: Factors, Monthly" data
set (free, decades of history), and fully free daily ETF prices of replicators - DBMF
(replicates the SG CTA index, from 2019), KMLM (MLM trend index, published rules), CTA.
**Caveat:** our universe is rates futures only; an index return also carries equities, FX
and commodities, so either fetch those futures (Databento cost) or regress the index on
our rates P&L plus free proxies of the other sleeves and calibrate on the rates part.
(b) more UBS snapshots (the note promises monthly updates) - more dates, same model.
(c) a clearer scan of pp. 24 and 40 (Fig. 75, 107, 108) adds the other US contracts' t-2w
and t+2w values to `paper.py`.

---

## CTA: top-down positioning (regress CTA performance on asset returns) - tabled

**Found:** 2026-10-02 (user idea, tabled by the user).
**Where:** would sit next to `infra/models/cta` (a second model on the same `base.Model` pattern).
**Status:** tabled.

**The idea:** a CTA index's daily return is (approximately) sum_i exposure_i(t-1) x
return_i(t), so regressing it on asset returns over a rolling window gives the exposures
top-down (Sharpe 1992 style analysis; Fung & Hsieh 2001 for trend followers; DBMF's
"dynamic beta" replication is exactly this, and DBMF publishes its futures holdings daily -
a free, ready-made top-down estimate). **Problems:** many correlated assets vs few
observations (needs ridge/LASSO or sign constraints), exposures move faster than any
rolling window (needs exponential weights or a Kalman filter), index returns are net of
fees and some are lagged/smoothed. **The useful combination:** keep the bottom-up model's
SHAPE (which assets, which sign, how saturated) and let the top-down regression estimate
only a few SCALE factors - one per asset class, beta_class(t) x bottom-up position - in a
Kalman filter on the index return. Few parameters, well identified, and the scale is exactly
the missing "size footprint" (AUM x leverage per class) that turns our [-1, 1] positions
into contracts and flows in % of ADV.

---

## CTA: no scheduled fit/predict, no stored outputs, no live input

**Found:** 2026-10-02.
**Where:** `infra/models/cta`; `infra/cycle` (may not import `infra/models`, `tests/test_architecture.py`).
**Status:** open - design decision needed.

**The issue:** the model is built for a scheduled fit (daily/weekly) and a light predict
(daily, or intraday on a daily fit), but nothing runs it on a schedule, keeps fitted
state, or stores outputs, and models may not write storage. **Options:** (a) a separate
Prefect flow outside `infra/cycle` (e.g. `infra/models/cta/jobs.py` + a script), after the
daily cycle, writing to a `Derived/CTA` store through `infra.storage` - needs the
"models never write storage" rule relaxed for a job layer; (b) move the CTA's computation
below the models layer (`infra/analytics/cta`, like WIRP) so the `derived` step can run
it - but it is a model with fitted state, which is what `infra/models` is for; (c) keep
it on demand. Fitted state is plain dataclasses/pandas (picklable). **Live input:** an
intraday back-adjusted price needs the front contract's latest bbo-1m/ohlcv-1m price
minus its last settlement, added to the continuous series' last value - not written yet.

---

## CTA: universe and unit gaps against UBS's model

**Found:** 2026-10-02.
**Where:** `infra.models.cta.config.CTA_UNIVERSES`, `infra/models/cta/inputs.py`.
**Status:** open, by data availability.

* **Rates only, and few markets:** UBS runs ~100 markets (bonds, STIR, equities, FX,
  credit, commodities); we have US bond futures (from 2014-12), Eurex Schatz/Bobl/Bund/BTP
  (from 2025-07 only: no signal history to fit an ECDF, so their signals start late), no
  Buxl, OAT, JGB, CGB, KTB, ACGB, no Long Gilt (ICE disabled in the cycle). STIR: SR3 only
  from 2025-03, so UBS's money-market results (ED4 etc.) aren't comparable yet.
  The portfolio vol scaling is therefore estimated on a much smaller portfolio than UBS's.
* **Price vs yield space:** UBS sizes rates in $DV01 with yield-bp vols; we size on futures
  price vol. The ratio is the contract's duration, which drifts slowly (CTD switches), so
  normalised positions differ a little; converting can use the bond futures DV01 now in
  `Bmk/Risk` (the deterministic-CTD model, since 2026-10-02).
* **Fit-sample length:** our ECDF and position scale use history from 2015; UBS's use ~10-30
  years, so "extreme" means extreme relative to a shorter, mostly low-vol history.
* **Monte Carlo simplifications** (as UBS): assets simulated independently, constant
  forecast vol, normalisation and portfolio scaling held over the horizon.

---

## Basis: futures trade rich against funding v1 on ZT/ZF/ZN/ZB, and UB's option value is unexplained

**Found:** 2026-10-02, the basis models' 2019-2026 bench (`infra/models/basis/CLAUDE.md` 5).
**Where:** `infra/models/basis`; observed option value = fair (M0) - market futures at 15:30.
**Status:** open - the explanatory spread layer's job (`infra/models/basis/CLAUDE.md` 1), with M2T/M3 for the option side.

**The issue:** (1) the observed option value is NEGATIVE on 58-87% of days for ZB, ZF, ZN, ZT (medians -0.4 to -0.7/32): the futures are RICH against our funding, the basis-trade premium documented for 2019-20 (CTD implied repo +13 to +20bp over SOFR) - a pure delivery-option model can't produce a negative value. Plan (2026-10-02): an EXPLANATORY spread layer - regress the CTD's implied repo minus funding v1 on funding and positioning drivers (SOFR p75 - median, which also tests v1; SOFR p99 - p75; the DVP term premium; quarter-ends; CFTC positioning; reserves) - not a statistical richness term fitted to the residual, which would add nothing. (2) UB's observed value has a median of ~6/32 while M1 explains ~0 of it (yields far below the 6% notional, so parallel moves rarely switch). Candidates: relative moves among the long bonds (M2), the end-of-month and wild-card options (add-on T - built 2026-10-02: the wild card alone gives UB 10.5/32 vs 12.3 observed in the final month), a cash bid/mid bias on old long bonds (`cash_mid_frac`; their posted spread is 1-2/32).

---

## Basis: the cash bid/mid adjustment is a guess

**Found:** 2026-10-02 (CLAUDE.md 18, 22). **Where:** `infra.models.basis.config.BasisSpec.cash_mid_frac`.
**Status:** open.

**The issue:** FedInvest's END OF DAY is the bid (its Sell column), and its posted Buy/Sell spread is a fixed convention (0.5/1/2 32nds by maturity), not the market's. The basis needs the mid; the model adds `cash_mid_frac` (0.5) x half the posted spread, i.e. a quarter of it - a guess between bid and posted mid. The bias it can leave is up to ~1/32 on ZB's and UB's old long bonds. **Options:** calibrate the fraction from the data (the value that makes CTD net basis most consistent across roots / most stationary), or a real mid source (TRACE end-of-day is paid, user decision 2026-10-02: no).

---

## Basis: only each root's front contract has a 15:30 futures quote

**Found:** 2026-10-02. **Where:** `Derived/FuturesSnaps` (from bbo-1m, which the front-quotes backfill stores for each root's `.v.0` only).
**Status:** open, minor.

**The issue:** the basis models run on contracts quoted at 15:30 only; deferred contracts would fall back to the 15:00 settlement (half an hour off the cash close: 2-5/32 of basis noise). Around a roll both contracts matter (the back month becomes front over a few days). **Options:** extend the front-quotes backfill to `.v.1` (cost: priced per month like the front); or accept settlement-based numbers for deferreds, flagged (current fallback).

---

## Basis: the expected-issue generator ignores holidays

**Found:** 2026-10-02. **Where:** `infra.processing.futures_baskets.expected_issues`.
**Status:** open, minor.

**The issue:** a predicted issue day is rolled forward past weekends only; an issue day falling on a federal holiday is predicted a day early. Backtest 2019-2026: 815/845 issues within 5 days (all usable) but only 747 exact. A day's error only matters for issues landing right at a contract's last delivery day. **Fix:** roll with the SIFMA-approximate calendar (`infra.analytics.sofr_curve.business_days`) instead of weekdays.

---

## Basis: the wild-card window's variance - remaining caveats after the event fix

**Found:** 2026-10-02, building the timing options (`infra/models/basis/CLAUDE.md`, `infra.analytics.delivery_timing`).
**Where:** `BasisSpec.wildcard_window`; `infra.models.basis.model.OneFactorBasis._timing`.
**Status:** partly fixed 2026-10-02 - event days now get their own variance; the rest is secondary (user decision).

**Fixed:** each wild-card window has its own variance: the root's ORDINARY-day share of daily variance in 15:00 -> 19:00 New York (ZT 4.0% ... UB 6.7%) x a multiplier for FOMC days (1.7-6.3x by root), quarter-ends (2.0-2.8x) and month-ends (1.3-3.1x), measured 2019-2026 (`BasisSpec.wildcard_window`; day kinds `infra.analytics.delivery_timing.window_kinds`). Found moot: the LAST intention day's later deadline (20:00 Chicago = 21:00 New York, 1.5-1.75x the variance of a 19:00 window) - that day falls after the last trading day, when the futures price is already frozen, so it belongs to the end-of-month period, not to a wild-card window.
**Remaining:** (1) the shares and multipliers are constants calibrated on 2019-2026, so a backtest before 2026 uses later data in a parameter (a mild look-ahead); re-measuring them point in time needs the intraday quotes at fit time. (2) Other after-close events aren't tagged (e.g. ZN's largest window move, +29.5/32 on 2025-04-02, the tariff announcement) - the release calendar (CLAUDE.md 17) could add scheduled ones; unscheduled ones can't be anticipated. (3) The window spans CME's 17:00-18:00 New York halt (futures don't trade, cash thinly does), and futures moves stand in for the CTD's cash moves - both secondary.

---

## Swap closes: the futures-adjusted hedge ratio could use the basis model's futures DV01

**Found:** 2026-10-02 (user question), after the basis model's futures DV01 was validated (`infra/models/basis/CLAUDE.md` 7: slope 0.97-1.00, R^2 >= 0.977).
**Where:** `infra.pipeline.swap_hedge` / `infra.processing.swap_hedge` (`SWAP_HEDGES`, `SWAP_HEDGE_CMT`).
**Status:** open - a candidate improvement, untested.

**The idea:** the hedge ratio (swap-rate bp per futures point) is a 60-business-day regression of futures settlement changes on the CMT par yield, which bundles two legs: futures point -> the CTD's yield (exactly 1 / the model's futures DV01, point in time, adjusting at once at rolls and CTD switches where a 60-day regression lags) and the CTD's yield -> the swap tenor (a curve beta the DV01 doesn't know - e.g. 1-3y swaps hedged with ZT). **Candidate:** ratio = (curve beta of the swap tenor on the CTD's yield, regressed) x (1 / model futures DV01). **Test:** the swap closes' held-out test (adjusted MAE 0.21bp today, CLAUDE.md 16); switch only if it wins.

---

## CFTC TFF: a shutdown-delayed release is LATER than `known_from` says

**Found:** 2026-10-02, building the TFF store (CLAUDE.md 24).
**Where:** `infra/processing/cftc_tff.py` (`known_from` = report Tuesday + 3 days, 15:30 New York).
**Status:** open - narrow, history only.

`known_from` is the SCHEDULED release. During a government shutdown CFTC stops publishing and catches up later (2013-10, 2018-12..2019-01, 2025-10..11), so for those weeks the data became public weeks after `known_from`: a point-in-time backtest reading them on their scheduled day uses data nobody had yet. Live runs are unaffected in effect (the data isn't on disk until it's published), but the stored `known_from` is still the schedule. **Options:** (1) a small table of the catch-up release dates per shutdown (from CFTC's own release notices) overriding `known_from`; (2) store a `first_seen` (the run that first fetched the row) going forward - doesn't fix history. (1) is the real fix; worth doing before the spread layer is backtested across 2018-19 or 2025.

---

## Primary dealer statistics: series breaks and MBS settlement-class dates

**Found:** 2026-10-02, building the store (CLAUDE.md 24).
**Where:** `infra/pipeline/primary_dealer.py`, `infra/processing/primary_dealer.py`.
**Status:** open - documentation / modelling caveats, nothing wrong in the stored data.

(1) **Series breaks:** the FR 2004 form changed in 2001, 2013, 2015, 2022 and 2024 (`pd/list/seriesbreaks.json`). The CSV carries every break's history under its keys, but only the CURRENT break's descriptions are catalogued (the list endpoint returns only those), and a key's coverage can change across a break (e.g. Treasury positions split by maturity from 2013; `PDPOSGST-TOT` starts 2013-04-03). A consumer building a long history must splice across breaks deliberately - **option:** a per-break catalog (the API may serve older breaks' series lists by another path; not found yet). (2) **MBS settlement-class series** are dated on days other than the Wednesday as-of date (settlement classes, possibly forward dates); `known_from` takes the first release at least 8 days after each date, which is conservative (never too early) but can be a week late for them. Verify against the FR 2004 instructions before using those series point in time.

---

## Basis: add-on T is inverted against UB's observed option value across carry regimes

**Update 2026-10-03 - option (1) DONE, now the default** (user decision): the Bermudan negative-carry rule with the end-of-month continuation net of carry (`BasisSpec.timing_carry_bermudan`) takes UB's negative-carry T from 3.15 to 9.69/32 (observed 11.27), 2023 from 2.5 to 8.8 (12.2). T is now ~9-12/32 in BOTH regimes (essentially the wild card - the end-of-month nets -0.8 / +0.13); what stays open is the OBSERVED value's swing with the carry regime (positive 4.7 vs negative 11.3), see below - most likely futures richness, for the spread layer.

**Found:** 2026-10-03, re-scoring M2 / M2T on the 14-day sample (`infra/models/basis/CLAUDE.md` 3e).
**Where:** `infra/models/basis/model.py` (`_timing`), `infra/analytics/delivery_timing.py`, and funding v1 (`infra/analytics/financing.py`).
**Status:** open - diagnosis, not yet a fix.

UB, medians in 32nds: positive carry model 11.8 vs observed 4.6; negative carry model 3.5 vs observed 10.9; observed peaks in 2023 (12.2) where the model is lowest (2.5). **Options:** (1) replace the negative-carry rule ("deliver at the first window, no end-of-month option") with the Bermudan that weighs each window's wild card against a day's negative carry - the short still holds a wild card every day up to delivery; (2) test whether the residual is a FUNDING effect - re-run M0's observed value with a term-repo base (the DVP 8-30 / >30-day buckets, `Daily/Repo`) instead of rolling overnight, and see whether UB's 2023 residual collapses; (3) leave it to the explanatory spread layer with carry regime / term premium as regressors.

**(2) tested 2026-10-03 and REJECTED** (a test only, nothing changed in the model - user decision): replacing each CTD's v1 rate with OFR's DVP >30-day rate (`OFR_DVP_G30`, known from D+1, CTD assumed not special) moves funding by -2..+9bp per year and the observed residuals by <= ~1/32 - UB 2023 12.2 -> 12.1, 2024 9.2 -> 8.3. (The premise was also backwards: HIGHER funding raises the fair price and the residual; only LOWER true funding could shrink it.) UB's 2023-24 residual equals an implied repo ~200bp BELOW funding (10-50bp in other years): no funding curve does that. **Next candidates, all UB-specific:** the CTD choice (the market's vs M0's), the long-bond cash marks (FedInvest's bid on 25y+ bonds, amplified by CF ~0.6-0.75), or real UB futures cheapness (positioning - the TFF data is now stored). Side finding: term funding narrows the shorter roots' futures richness only in scarce-reserve 2019 (ZN -0.8 -> -0.1, ZT -1.3 -> -0.7/32).

**Dissected 2026-10-03 (UBU3, 2023-06-01), three more candidates REJECTED:** (a) the CTD choice - 912810SE9 (3.375% Nov-2048, CF 0.6623) is cheapest by 52/32 over the runner-up, with sane cash marks (3.95%), and the futures sit 8/32 under its implied price (implied repo 4.56% vs funding 5.28%); (b) CTD specialness - the 72bp gap is exactly what ~75bp of specialness would give, but SE9 was lent by the NY Fed on 170 days 2022-06..2024 at the 5bp MINIMUM fee (once 11bp), so it wasn't special; (c) positioning (TFF, point in time, every 14th day 2019-2026) - leveraged funds 30-50% net short UB's open interest every year, correlation with the residual 0.00; asset managers -0.21; dealers +0.44 (residual larger when dealers are less short - weak, 8 yearly points); ZB / ZN nothing. **What remains:** a UB-specific delivery-option value the models underprice, hump-shaped 2019 (2.8) -> 2023 (12.2) -> 2026 (2.4)/32, tracking neither yield level nor carry cleanly. Next: implied vol (the ZN options now stored; UB's own options would be the real test) and M3's per-path timing.

---

## Basis: M2 overstates the CTD-pair spread variance on ZB

**Found:** 2026-10-02/03 (`infra/models/basis/CLAUDE.md` 3f).
**Where:** `infra/models/basis/factors.py` (`fit_factor_model`: the horizon covariance of a ~50-bond basket).
**Status:** open - M1 is the better tier for ZB until fixed (Brier 0.291 vs M2 0.350 on the 14-day sample).

**Update 2026-10-03:** two causes found and built as options (`idio_maturity_corr`, `level_betas` / `joint_pca`, all default off), closing about a quarter of the gap (0.350 -> 0.334) but costing UB (0.177 -> 0.192) - see `infra/models/basis/CLAUDE.md` 3f. Still open: the rest of the gap; label noise ruled out. Original entry: ZB's CTD/runner-up spread changes have a z-score sd of 0.63 under M2 (ideal 1): ~2.5x too much predicted variance, and M2 puts a median 61% on the realised CTD vs M1's 93%. Ruled out: fat tails (`spread_df`), mark noise (`idio_noise_removal` - FedInvest marks barely reverse). Removing the idiosyncratic part entirely recovers half the gap (0.319). **Options:** (1) shrink the idiosyncratic covariance toward a structured target (e.g. by maturity distance) instead of a free diagonal; (2) estimate the horizon covariance over a calmer / regime-matched window (the 500-day EWMA spans 2020-22's dislocations); (3) model the spread of each bond to the CTD rather than to the basket mean (the pair that decides). A root-specific scale would fit the sample and should be avoided.


## Stats: model runs have no scheduler yet

**Found:** 2026-10-03. **Where:** `scripts/model_run.py`, `infra/models/runs.py`.
**Status:** open (the jobs exist; nothing triggers them).

**The issue:** the operating model (root CLAUDE.md 3b: weekly fit-append, daily
predict-append, periodic rebuild) is implemented and tested, but no scheduler runs it, so
a strategy reading a run sees whatever was last run by hand. The daily cycle may not import
`infra/models` (and must not fail on a model). **Options:** a separate Prefect flow (or
launchd job) after the daily cycle that runs `model_run.py run <name>` for every run folder
with a `meta.json`, and `rebuild` on a calendar (monthly), alerting when a reconciliation is
not identical; same gap as "CTA: no scheduled fit/predict".

## Stats: `release:` series are indexed by observation period, and one snapshot per read

**Found:** 2026-10-03. **Where:** `infra/pipeline/series_panel.py` (`_release`).
**Status:** open.

**The issue:** a macro series comes back on its PERIOD axis (May payrolls dated
2025-05-01) as published by `as_of`, not on the axis of when each value became known. Mixed
with daily market data, May's value sits at 05-01 although it was published in June:
a regression of daily moves on it would use it ~5 weeks early. And a frame read once with
`as_of` = the backtest's end carries LATER revisions into earlier refits. **Mitigations
today:** pass `raw` to `walk_forward` as a callable `as_of -> read_panel(..., as_of=as_of)`
(vintage-correct per refit); keep macro-only regressions on the period axis. **Fix
options:** a `release_known:` source indexed by publication day (first-print or latest
vintage), built from `infra.processing.releases` (`timestamp` = publication day).

## Stats: Kalman regression is slow (pure-Python filter)

**Found:** 2026-10-03. **Where:** `infra/models/stats/regression.py` (`kalman_filter`,
`KalmanRegression.estimate`).
**Status:** open, low priority.

**The issue:** ~1s per fit on 1,300 daily rows (the MLE runs the filter a few hundred
times); the dashboard's "diagnostics" box with Kalman refits monthly over 5 years took
~35s. Weekly walk-forwards over many years take minutes. Partly done 2026-10-03: refits
warm-start the MLE from the previous fit's (q, r) (55 vs 90 likelihood evaluations, same
result to 1e-5), and under the weekly fit-append schedule (root CLAUDE.md 3b) a fit is one
per week. **Remaining options:** vectorise the filter for a scalar observation; only
estimate (q, r) on a slower schedule and re-filter at each refit.

## Stats: hockey-stick knot inference is grid-limited and its F p-value is naive

**Found:** 2026-10-03. **Where:** `HockeyStickRegression` (`regression.py`).
**Status:** open, documented.

**The issue:** the knot is the best of `knot_grid` (60) quantile points, so with low noise
the profile-likelihood interval collapses to one or two grid points (`knot_ci_low ==
knot`), and `F_vs_linear_p_naive` uses an F distribution that does not hold when the knot
exists only under the alternative (Davies 1987). **Options:** refine the knot by a 1-D
optimisation between the grid neighbours; sup-F p-values by Hansen (1996)'s simulation or a
residual bootstrap.

## Stats: logit/probit have no penalty; separation is only flagged

**Found:** 2026-10-03. **Where:** `LogitRegression.estimate`.
**Status:** open.

**The issue:** with a regressor that separates the classes (common with thresholds on a
small sample) the MLE diverges; the fit stops at |beta| > 50 and reports `separation=1`,
but the coefficients are then meaningless. **Options:** a ridge (L2) or Firth penalty in
the IRLS step, as a spec field.

## Stats: Marchenko-Pastur edge with few series uses the discarded eigenvalues

**Found:** 2026-10-03, on the US curve (6 tenors). **Where:** `pca.noise_edge`.
**Status:** documented choice, revisit if misleading.

**The issue:** below `MP_MIN_VARIABLES` (20) there is no noise bulk; the noise level is then
the mean of the eigenvalues after the k retained ones, so `n_above_mp` depends on k (US
curve, k=3: PC1-PC4 above, PC4 marginally). It is a heuristic there, not a test.
`diagnostics.parallel_analysis` is the better small-p tool; the dashboard does not show it
yet.

## Stats: HMM parameters are estimated from complete feature rows only

**Found:** 2026-10-03, building `infra/models/stats/hmm.py`.
**Where:** `hmm.m_step_gaussian`.
**Status:** open, low priority.

**The issue:** the filter uses rows with gaps (marginal likelihood over the observed
features), but the M-step's means and covariances use complete feature rows only. With the
default daily-score features every row with any observed N series has complete scores
(the scores themselves are projected over the gaps), so today this only matters for the
`columns` feature builder on gappy inputs. **Fix:** the conditional-Gaussian M-step (expected
missing entries and their covariance per regime, as `pca.ppca_em` does).

## Stats: a daily-score HMM's predicted regime can swing within days out of sample

**Found:** 2026-10-03, US/DE/UK curves walk-forward (`infra/models/stats/CLAUDE.md` 7).
**Status:** open, a modelling choice per use.

**The issue:** the default features (daily factor scores) make each day's emission very
informative, so out of sample the predicted probability of the 2-regime HMM moved 0.15 ->
0.92 within three days (Sep 2026) even with `sticky=200`. Fine for the z-score scaling,
noisy if a strategy keys on the regime itself. **Options:** larger `sticky` (pseudo-counts
~ the regime durations wanted), `hmm2_vol` (rolling-vol features, slower), a minimum-duration
(semi-Markov) HMM, or smoothing the probability path used for trading (point-in-time:
trailing only). Not chosen yet: needs a strategy to judge against.

## Stats: Markov-switching regression not built

**Found:** 2026-10-03 (decision). **Status:** deferred.

The external-regime path covers "regimes from a richer data set" for PCA (`RegimePCA`);
the regression analogue (regime-weighted regression with `RegimeModel` probabilities) and a
true Markov-switching regression (regimes defined by the regression's own fit: emission =
residual density per regime, through `hmm.forward_backward`, M-step = the existing weighted
estimators) are both still to build. Also queued by the user: Johansen / VECM and
regression with autocorrelated errors (Prais-Winsten).

---

## Basis add-on MS: per-bond richness forecasts are contaminated by the global curve fit

**Found:** 2026-10-03 (`infra/models/basis/CLAUDE.md` 3g).
**Where:** `infra/analytics/event_study.py` (`curve_residuals`: one cubic regression spline per day, 1-30y), used by `infra/analytics/event_drift.py` (MS add-on, `BasisSpec.ms_calendar`, off).
**Status:** open - MS off until fixed.

Both MS modes hurt the bench (events: TN 0.050 -> 0.150; aging: UB 0.177 -> 0.392). The aging failure on UB - old 30y bonds whose true aging is ~0 - points at the measure: a bond's residual to ONE smooth curve changes as it rolls down the maturity axis through regions the spline fits systematically differently (the long end, near knots), and that fit error becomes fake "aging". **Options:** (1) local richness - each bond's yield minus a fit through its nearest neighbours EXCLUDING itself (leave-one-out local regression), so a slide along the curve doesn't change the reference; (2) matched pairs - richness relative to the adjacent issues of the same series; (3) a finer, more flexible curve (more knots, coupon effect) - helps but keeps the problem in thin sectors. Then re-estimate both profiles and re-bench; also add the profiles' dispersion as extra variance.

---

## Basis: the observed option value doesn't move with the delivery options - no target can test vol inputs

**Found:** 2026-10-03, the spread study's change-based IV test (`infra/models/basis/CLAUDE.md`, the IV row).
**Where:** the bench's target, `option_value_obs_32` = M0 fair (15:30 cash) - futures (`infra/models/basis/model.py`, `validate.option_value_bench`).
**Status:** open - research; add-on IV stays off meanwhile.

Over 2,100 consecutive same-contract changes (5 business days apart, 2019-2026) the models' option-value changes barely correlate with the observed changes (-0.42 UB .. +0.21 TN), so neither EWMA nor implied vol can be judged against it. UB's -0.42 looks systematic: a candidate is the TAIL (CF ~0.6) - a level move changes the measured fair-minus-market with the opposite sign to the model's response. **Options:** (1) a vol-responsive target - Treasury futures options' implied delivery-option value, or the P&L of a DV01-hedged basis position held to delivery; (2) decompose the observed change into level (beta to the futures move), carry and residual, and check UB's tail hypothesis; (3) a denser 15:00 / 15:30 cash source to cut snap noise (TRACE is paid - see memory).


## Basis: delivery windows are counted on the FEDERAL calendar (Dec 2021 off by one day)

**Found:** 2026-10-05, building the futures contract calendar events (root CLAUDE.md 17).
**Where:** callers of `infra.analytics.futures_basis.delivery_window` that pass
`infra.analytics.sofr_curve.business_days` (federal holidays + Good Friday), e.g. the basis
models' inputs.
**Status:** open (the basis sub-project's code; not changed here).

**The issue:** CME counts delivery-window business days on its own (market) calendar. The
federal calendar observes a SATURDAY New Year's Day on Friday 31 December, when markets are
open (2021, 2010): with it, the Dec-2021 Treasury contracts' last trading days come out one
day early (ZNZ1 rule 2021-12-20, CME/stored 2021-12-21; ZFZ1 2021-12-30 vs 2021-12-31) and
the delivery window shifts with them. The event calendar now counts on
`infra.processing.schedule_rules.business_days(..., "market")`, which matches every stored
expiry (299 contracts). **Fix:** pass the market calendar to `delivery_window` in the basis
inputs (or have `delivery_window` default to it); re-run the bench around Dec 2021.

## Events: refunding / borrowing-estimate dates before 2016, and unscheduled-meeting times

**Found:** 2026-10-05. **Where:** `infra/processing/event_dates.py`, `infra.config`.
**Status:** open, low priority.

**The issue:** `US_TSY_REFUNDING` / `US_TSY_BORROWING_ESTIMATES` are emitted from 2016 only:
the Monday 15:00 / Wednesday 08:30 pattern is verified for 2016 and 2026, while the auctions
store would date refunding statements back to 1979 (Wednesdays every quarter since 2000). The
unscheduled central-bank decisions (ECB 18 Mar 2020 evening, BoE 11 and 19 Mar 2020, Fed 3 and
15 Mar 2020) are day-level: their release times were not verified. **Options:** verify a few
older refunding statements (home.treasury.gov press releases) and lower `REFUNDING_FROM`;
add each unscheduled meeting's verified time to its config entry.

## Event study: bp P&L only from 2025-07 (bmk DV01 history)

**Found:** 2026-10-05. **Where:** `infra/pipeline/event_pnl.py` (`FUTURE_BPS_BBO`), the bmk
risk store. **Status:** open.

**The issue:** bp steps divide by the contract's prior-day DV01 from `Bmk/Risk`, which for the
Treasury roots starts 2025-07; quotes go back to 2015, so `FUTURE_BPS_BBO` is NaN before then
and studies in bp have ~15 months of history. **Options:** back-fill bmk risk
(`infra.cycle.bmk.backfill_daily_risk`; Treasury DV01 needs FedInvest prices - from 2008 - and
funding - SR1 from 2018-05, so 2018-05 onward is possible); meanwhile use `FUTURE_PTS_BBO`.
**Update 2026-10-05:** back-filled 2018-05 .. 2025-06 (M0 DV01, ~13 s per month): 99-100%
of Treasury contract-days from 2019, 95-96% for Oct-Dec 2018 (December was failing on the
5 Dec 2018 closure - fixed in `fit_sofr_path`, re-run). Before 2018-10 there is no DV01: the
funding model's SOFR path starts 2018-10-01. So bp studies start 2018-10.

## Event study: open items from the first build

**Found:** 2026-10-05. **Status:** open.

*   **Good Friday:** the grid's market calendar has no Good Friday, so an NFP on Good Friday
    (2021, 2023, 2026) is `not_a_trading_day`, while CME rates trade a short session. A
    CME-rates calendar (open on such Fridays) would keep them.
*   **Costs:** no test compares the expected move with the bid-ask at the window's ends yet (the
    bbo store has both sides: a `half_spread` per step from the same source would do it).
*   **Overlapping windows:** consecutive events whose windows overlap are treated as independent
    (plain t); a family with long windows should use overlap-aware errors.
*   ~~Family persistence~~ fixed 2026-10-05: family runs (`infra/jobs/family_runs.py`, one run
    per code + a point-in-time FDR table).
*   **Conditional studies, next steps:** one condition per study (two-dimensional regimes
    later); no SURPRISE feature yet (actual - MarketWatch consensus,
    `infra.pipeline.econ_calendar.consensus`, needs its own availability: the consensus as known
    before the print); availability rules are single per source (date-ranged variants if a
    publication time changes); CMT / BoE / Bundesbank / OFR rules are conservative, not verified.
*   **OHLC and executable P&L sources** (user decision 2026-10-05: BBO mid only for now). Both
    are a price-type parameter on the same source (mapping, one-contract steps, halt carry and
    DV01 conversion shared): `..._OHLC` = the close of the 1-minute bar ENDING at the point (a
    bar is stamped at its start - point in time), a cross-check (an effect only in trade prices
    is microstructure: bid-ask bounce, stale prints) that also brings volume; `..._BBO_EXEC` =
    enter at the ask / exit at the bid (long), the cost-inclusive move - the "costs" item above.
    ohlcv-1m holds only 2026-08-23..09-20 for the bond roots: price the back-fill with a dry run
    first.

## Strategies: open items from the first build (CEVT)

**Found:** 2026-10-05. **Where:** `infra/strategies`, `infra/jobs/strategy_runs.py`. **Status:** open.

*   ~~No strategy P&L or transaction costs~~ fixed 2026-10-05: the accounting layer
    (`infra/strategies/accounting.py`, root CLAUDE.md 27). Its open items:
    *   **`spread_paid` = 0.5 is an assumption** (passively filled on half the trades): research
        it - e.g. how often a resting order at the touch would have filled within the label's
        interval, from bbo-1s around the trades, which can differ a lot by time (an 08:30
        release vs a quiet afternoon).
    *   **No market impact:** a trade beyond the top-of-book size is only flagged
        (`exceeds_top_of_book`; real data: up to 110 ZN contracts at NFP against 1-20 at the
        touch). Options: a depth-based cost (needs MBP-10, a new paid schema), or a square-root
        impact on volume.
    *   **Friday `GRID_END` exits can't trade** (CME shuts 17:00 ET; the 06:00-19:00 grid runs to
        19:00): the accounting defers them to the reopen (weekend risk), while the event study
        values them at the 17:00 close - its P&L is optimistic for such codes. Options: a cycle
        whose Friday ends at the close, an accounting policy `advance` (trade the last executable
        label BEFORE a known closure for exits), or illegal-window rules in the event study for
        a leg falling in a closure.
    *   **Deferral is per contract**: a roll whose new contract can't trade while the old one can
        leaves the position flat for the gap (flagged as `deferred`); a "roll as a pair" policy
        would wait for both. **Measured 2026-10-05 - it breaks STRUCTURES at the print:** on the
        layered NFP run (6 structures, 2021-2026), every one of 52 deferrals was `wide_spread` at
        08:30 ET on NFP Fridays: some legs traded, others waited one label, so for the 15 minutes
        across the release the book held broken hedges (ZB at -335 vs a +337 target). That
        legging moved realised gross by +-$400k on single days and -$740k in total ($0.81m
        executed vs $1.55m intended). Options: defer ATOMICALLY per trade decision (all legs of a
        label's trade wait if any can't go), a "no new risk inside a release minute" rule (enter
        before 08:30, as the 08:15 codes do), or price the legs at the touch instead of waiting.
    *   **Daily costs from NY1500 snaps** exist only for contracts the snap store holds (each bond
        root's front, the ZQ strip); others use the root's trailing median (`cost_fallback`).
*   **Instruments are independent** in the PER-INSTRUMENT position sizing (no covariance): ZT and
    ZN views in the same direction double the risk the target assumes. Addressed 2026-10-05 for
    the Treasury futures by the LAYERED sizing (views on structures, per-layer budgets, a
    full-covariance portfolio cap; root CLAUDE.md 28); per-instrument sizing is unchanged and
    still independent (fine for one instrument, wrong for several correlated ones).
*   **Not scheduled:** like model runs (root CLAUDE.md 3b), the jobs need a runner after the
    daily cycle (which may not import above the pipeline); `scripts/strategy_run.py` by hand.
*   **Firm series are recomputed whole each run** (fine at ~36k labels a few years; for long
    1-minute grids, restrict to labels after the last stored one plus open windows, like
    `predict_window_after`).
*   **Overlapping windows of one code** (consecutive events whose windows overlap) are
    aggregated like different codes; harmless for intraday NFP windows, not for multi-day ones.
*   **The plan uses the latest fit for every upcoming event** and the family's latest FDR
    verdict; a refit due before the event can change it (that is what plan vintages record).
*   **The realised-vol scaling** has a unit test but no real strategy using it yet; the first
    always-on strategy should check its warm-up (`realised_min_obs` days of P&L, flat before).


## Daily futures: bad / stale vendor settlements the bad-print rule can't see (UBM0, March 2020)

**Found:** 2026-10-05 (curve-structure diagnostics). **Where:** `Daily/Futures` raw store (Databento
`statistics`), `infra.cycle.px.peer_outliers` / `infra.cycle.bad_prints`. **Status:** open.

**The issue:** UBM0's settlement is 211.1875 on BOTH 2020-03-16 and 2020-03-17, while the NY1500
quote mid moved +11.17 then -11.66 points (397 and 373 ticks off; every other bond future moved
13-33bp on the 17th). It made the WN daily bp move -3 / 0 / -35.5bp over 16-18 March and put
kurtosis > 110 on every structure with a WN leg. The bad-print rule cannot flag it: it needs the
contract's OWN move to be large (|z| >= 5), and a stale print moves 0.
**Scale:** settlement vs NY1500 bbo mid over 2015-2026 (16,987 front bond-future contract-days):
median miss 1 tick, p99 8 ticks; this is the only genuine bad episode. The other >= 16-tick misses
are snap timing on early-close days (Friday before Memorial Day 2016, year ends), not bad
settlements. Unchanged settlements while the complex moved are mostly genuine (TU at the zero
bound moves less than a tick).
**Options:** a `settle_vs_quote` check in the px step (settlement change vs the change in the bbo
mid at the settlement instant, in ticks; needs bbo-1m for the contract and an early-close
calendar for the snap), treated like a bad print (NA / roll); or a stale-print clause in
`peer_outliers` (own move ~0 while peers' median |z| is large, and the next session catches up).
Not fixed now: found during research; downstream users (bmk pnl 2020-03, CTA, any daily structure
study) should exclude 2020-03-16..18 for UB until then.

## Strategies: layered structures - open items from the first build

**Found:** 2026-10-05. **Where:** `infra/reference/structures.py`, `infra/pipeline/structures.py`,
`infra/strategies/layered.py`, `infra/strategies/config/layers.py`. **Status:** open.

*   **Budgets are placeholders** (macro $1m, front $0.3m, micro $0.5m, cap $1.5m a year). The
    front end's LOWER vol budget is the user's interim choice (2026-10-05); the planned
    replacement is a TAIL budget (expected shortfall, or the worst historical stress - SVB,
    March 2020 - scaled to today's book), since 22% of the hedged TU's variance sits on 5 days
    and a 60-day EWMA is lowest just before a jump (a longer span or a floor for that layer).
*   **Cost-aware sizing:** vol-parity puts huge notional on low-vol structures (a full fly view
    ~800 UXY contracts); round-trip cost is 26-77% of a day's vol for fly / front / micro
    (strategies doc 5). Options: budgets net of expected cost, a minimum holding period per
    layer, a size cap per leg vs the touch (`exceeds_top_of_book`), or only daily horizons for
    those layers.
*   **The hedged TU breaks at the zero bound:** 2021 betas (duration 0.38 vs ~0.8 otherwise); a
    hedge fitted while the front is pinned under-hedges on the way out. Options: a regime-aware
    window, betas floored at a long-run value, or a conventional TU_FV fallback when the fit's R2
    collapses.
*   **Execution routing TY <-> UXY** (user agreed as an optional later layer): duration is
    defined in TY; execution could hold it in whichever is cheaper per bp at the moment (UXY in
    quiet hours, TY at events for depth), a small cost minimisation subject to the structure
    exposures.
    **Extended 2026-10-06 (user intent):** routing is also where RV signals act - instrument
    selection biased by rich / cheap signals (the auction-cycle flies first), across futures
    and eventually cash bonds; see `infra/strategies/CLAUDE.md` 5b.
*   **CME inter-commodity spread books** (NOB, FYT, TUT...; user: research for later): curve /
    fly costs are legged at outright half-spreads, probably overstated. Storing the spread
    instruments' bbo would let the accounting price a structure trade as one.
*   **P&L attribution per structure / layer** is not stored: the accounting reports per future.
    Add `positions (usd_dv01:<structure>)` x structure bp moves per label as a research table.
*   **Hedge warm-up:** hedged structures need 120 daily moves (DV01 history from 2018-10), so
    structure P&L and layered positions start ~2019-04; days without a complete state use the
    latest complete one before them.
*   **No `struct:` series in `series_panel`** yet (structure daily bp moves for the regression /
    PCA page and conditions): `infra.pipeline.structures.structure_state(...).moves` has them.

## Event study / bmk: bp history starts 2018-10 (no Treasury DV01 before the funding model)

**Found:** 2026-10-05. **Where:** `Bmk/Risk` (M0 futures DV01, `infra.pipeline.futures_basis.
deterministic_futures_dv01`), `FUTURE_BPS_BBO` / `STRUCT_BPS_BBO`, `infra.pipeline.structures`.
**Status:** open.

**The issue:** every study and the layered structures work in bp (move x point value / prior-day
DV01), and the Treasury futures DV01 needs the funding model, whose SOFR path starts 2018-10.
So bp history is ~8 years (~95 NFPs) while quotes go back to 2015 (~140 NFPs in points). The
real-data checks in the event-study doc (sections 5-6) were run in points for that reason and
should be re-run in bp. **Option:** a DV01 PROXY before 2018-10 - the same cheapest-to-deliver
forward DV01 / conversion factor, with funding replaced by a simple proxy (e.g. the CTD's own
repo-free forward, or GC from the DTCC GCF index 2005-2024): funding barely moves a forward DV01
(it changes the forward price by carry, a few 32nds), and the baskets (computed from 2016-01)
and FedInvest prices (2008-09) exist. Store it flagged (`model = "M0_proxy"`) so bp before
2018-10 is visibly a proxy; check it against the real M0 DV01 over 2018-10..2026 first.

## Daily: open items from the first build (frequency, yield P&L, daily studies and strategies)

**Found:** 2026-10-05. **Where:** root CLAUDE.md 12 (bmk yield P&L), 29. **Status:** open.

*   **Views on yields have no execution map:** a daily strategy on `US_BOND_10y` raises at sizing.
    Needed: yield view -> futures exposure via the measured yield beta of each future (TY tracks
    ~7y, UXY the on-the-run 10y; the swap-hedge regression `SWAP_HEDGE_CMT` is the template), or
    cash bonds in the accounting.
*   **Futures as a benchmark P&L (sanity check (a)) not stored yet:** futures bp vs the yield P&L
    at the same instant - the futures at the NY1530 snap against the 15:30 cash marks (UXY vs the
    on-the-run 10y, TY vs its CTD's yield) - as a stored daily comparison / check; the residual is
    net basis + model DV01 error.
*   **Total-return bmk with carry** (user, planned): price action + carry + rolldown - financing,
    per CUSIP (TreasuryRV carry / rolldown, the funding model).
*   ~~**`yield_curve` stops at the last hand build** of TreasuryCurves~~ - fixed 2026-10-05:
    the curve is in the daily cycle's `derived` step (metric `treasury_curve`, stores
    TreasuryCurves + TreasuryRV), so it advances every run.
*   **CMT before 2016** isn't in our store (the Treasury's yearly CSVs go back to 1990, free):
    a backfill would give CMT-based daily studies 2008+ like the other sources.
*   **Daily study defaults:** the 2bp size floor and overlap-unaware errors were set for intraday
    windows (see the event-study open items).
*   **`ZB.v.0` held into first notice on 2021-08-31** (ZBU1; caught by `held_in_delivery` in the
    daily layered run): the bond futures' 2-day volume lookback still left v.0 on the expiring
    contract that day. Check how often any bond root's v.0 sits on or after first notice.

---

## Swap spread bmk P&L: plan, and the CTD-aware version for later

**Found:** 2026-10-06 (user discussion). **Where:** swap closes (root CLAUDE.md 16), our Treasury curve (25a), the bmk yield P&L sources.
**Status:** steps 1-3 built (2026-10-06): the pure swap closes, the OIS (SOFR) curve and the two swap-spread series (CMT, on-the-run par-par ASW) are in `derived`, their price-action P&L in `bmk_pnl` (`swsp_cmt` / `swsp_otr`; root CLAUDE.md 12, 16). Next: level 2 - carry + rolldown on both legs (bond: coupon - repo via funding v1, rolldown on our curve; swap: fixed - compounded SOFR, rolldown on the OIS curve). Open on the P&L: `swsp_cmt` carries CMT's auction-switch jump (2y/3y, a few bp on auction days, as `yield_cmt` does); 7 other 2y days in two years where the sources disagree by > 3bp, not investigated. Open on the curve: (a) it uses the PURE closes - switch `OIS_CURVES["USD_SOFR"].method` to `adjusted` once the adjusted closes are in the cycle (needs the intraday fetch): the remaining 30 reversing forward kinks sit at 15-30y, the pure closes' own noise in thin tenors (fly residuals move 2.5-4bp/day there), and in the 6m-1y segment; (b) payment dates = accrual ends (SOFR OIS pays 2 business days later: a few hundredths of a bp), no end-of-month roll; (c) EUR/GBP curves not built (thin prints, no stored futures for a short end); (d) quoted swap-spread convention verified as the PLAIN difference (spread-over trades: fixed = Treasury yield + spread, the swap keeping annual ACT/360 - Clarus); Bloomberg's own methodology page not found.

**Agreed order:** (1) the PURE swap closes into the daily cycle's `derived` step (no intraday dependency, like the curve); (2) a SOFR discount curve bootstrapped from them (reusing the curve code: spline on the discount function, cash-flow based); (3) two swap-spread series, the way the bond P&L has OTR / CMT / our-curve sources: **CMT swap spread** (SOFR swap par at NY1530 minus the CMT par yield at the same tenor - market convention, includes the on-the-run premium; CMT chosen as the reference: it is ~the on-the-run's own 15:30 bid yield interpolated to the tenor, and synchronous with the NY1530 close; convention to verify: swap annual ACT/360 vs CMT semi-annual bond basis, convert before subtracting) and the **on-the-run's par-par ASW** ((PV of the bond's cash flows on the SOFR curve - dirty price) / the floating leg's annuity on the SOFR curve; z-spread to SOFR as a cross-check). Then the bmk P&L with carry + rolldown on both legs (bond: coupon - repo via funding v1, rolldown on our curve; swap: fixed - compounded SOFR, rolldown on the SOFR curve).
**Later (user, 2026-10-06):** the **CTD-aware** version - the swap spread / ASW of the futures' cheapest-to-deliver (what a basis trader holds) - the same way a CTD-aware cash-bond P&L comes later. Also later: our-curve (off-the-run, premium-free) swap spread as an RV variant; a held-position (level 3) P&L.

## Autocorr: open items from the first build (framework B1)

**Found:** 2026-10-06. **Where:** `infra/models/autocorr`. **Status:** open.

*   ~~The (k, h, X feature, source) grid is a family~~ built 2026-10-06 (`infra/models/autocorr/family.py`,
    point-in-time FDR gate per fit date; the duration-flies grid of 216 members found nothing).
    Still open: **cross-family control** - each new X is a new family; `global_q` corrects several
    families together but nothing does it routinely, and online FDR (LORD / alpha-investing) would
    be the principled version once many X's have been tried.
*   **Cell tests and correlated cells:** the corrected cell test gives FEWER raw p < 0.05 than chance
    (79.5 vs 130 a fit) - the 9 cells share one regression and overlapping windows, and HAC on
    non-contiguous cell members is approximate. Conservative, not wrong; a block bootstrap of the
    cell excess would calibrate it.
*   **Long horizons on short samples over-reject:** the only FDR discoveries in the grid were
    k20 / h20 members in 2012-2013 (fits on ~4 years, HAC with 20 lags on few independent
    windows). Options: non-overlapping sampling for long h, a minimum of independent windows
    (n / h) as a gate, or a fixed-b / bootstrap correction of the HAC t.
*   **Not wired into a strategy:** predictions carry `signal` / `expected_bp`; a strategy would turn
    them into views (the common forecast object discussed 2026-10-06) - not built.
*   **Evaluation P&L ignores costs:** the staggered book trades a little every day; net-of-cost
    comparisons (the conditional rule flips with the regime) are not computed yet.
*   **Placebos are slow-ish** (~0.9 s per walk-forward; 100 placebos = 1.5 min per evaluation):
    fine for research, too slow for a large grid - vectorise the fit across refit dates if needed.
*   **The `bmk:` source's availability is one conservative rule** (D+1 10:00 New York) for every
    yield source, though CMT is out the same evening.

---

## Swaptions: open items from the first build

**Found:** 2026-10-06. **Where:** `infra.pipeline.swaptions`, `infra.processing.dtcc_swaptions` (root CLAUDE.md 16). **Status:** open.

*   **Gamma map (step 3) not built:** gross gamma per strike from `open_interest(as_of)` x the surface (`infra.analytics.swaptions.normal_gamma`, $DV01 per bp of the forward), plus a net view under a stated sign assumption - the report has no direction.
*   **Offsetting unwinds aren't netted:** an unwind traded as a new opposite trade (same terms, months later, lower premium - seen in the identical-terms pairs: 2,880 same-label pairs open on 2026-10-05) adds open interest instead of removing it; without a side it can't be netted. Straddles (Call+Put pairs, 10,900) are genuinely two options.
*   **Capped notionals** count at their floor (~25% of open notional): open interest is understated in the largest trades.
*   **15% of open notional has no underlying maturity** (tenor bucket `unknown`).
*   **Vols beyond ~2y expiry are implausible** (220-260bp normal at 5y-10y): a premium convention (deferred / forward premium? callables reported as swaptions?) not understood; the surface stops at 2.25y.
*   ~~**Intraday forward mismatch**~~ fixed 2026-10-06: each print's forward is moved to its trade time by the hedge futures (root CLAUDE.md 16).
*   **Linking dipped to 67% in 2026Q2** (81-93% other quarters): not investigated.
*   **No skew series - the DTCC flow is too thin (2026-10-06):** a pooled regression (vol on moneyness, moneyness^2, a level per day) blew up even with forwards moved to the trade time (risk reversal swinging hundreds of bp: poorly identified with prints bunched at few strikes); a robust bucket estimator (median vol of receivers 25-75bp OTM minus payers 25-75bp OTM) is stable where it has data, but coverage is thin: 10-day pool, best point 3m x 10y on 192 of 482 days (moving 30bp a day there); 20-day pool, 3m x 10y on 346 days, moving 12bp week to week against a skew of ~1bp. The forward adjustment was necessary, not sufficient. **Realistic source:** listed option chains with real strikes across the smile - ZN / OZN beyond near-the-money (`infra.pipeline.futures_options_iv` selects near-the-money only) or SR3 options daily (two snapshots stored) - a priced Databento fetch.

---

## Volatility risk premium: open items

**Found:** 2026-10-06. **Where:** `infra.pipeline.vrp` (root CLAUDE.md 16). **Status:** open.

*   **Short swaption history** (2 years, one regime): ZN since 2019 shows the premium near zero or negative for whole years (2019, 2022-23) - don't read the swaption numbers as a constant.
*   **Realised side from pure-close curves:** the forward swap rate's daily changes carry the pure closes' noise (~0.35bp a close, ~1% of the daily variance at 10y; more in the 15-30y segment) - switch to the adjusted-close curve when it exists.
*   **Implied is the surface's ATM median**, at a surface point whose prints span an expiry bucket (1m = 0.05-0.14y), against a realised horizon of exactly 1 / 3 months.
*   **ZN is price vol** (points/yr): comparable to the swaptions only through the ratio; a yield-vol version needs the futures DV01 (bmk risk store).

## Features: open items from the first build (the central feature maker)

**Found:** 2026-10-06. **Where:** `infra/processing/features.py`, `infra/pipeline/features.py`
(spec `infra/processing/FEATURES.md` 5). **Status:** open.

*   **Inputs not built:** `resid(ID; FACTORS; W)` (residual vs factors, trailing fit - the
    `infra.analytics.structures.rolling_betas` machinery would serve) and `rev(ID)` (macro
    revisions). (`pdiff:` built 2026-10-06.)
*   ~~Migrations not done~~ done 2026-10-06: `prep.py`'s stateless steps run through the grammar's
    literal transforms, the event study's conditions take a grammar `feature` (legacy `steps`
    kept) - parity tested against copies of the old code. The event study's `period_diff` flag
    still exists beside the new `pdiff:` input (both give the same series).
*   **Intraday features** (the 15-minute grid): daily first by the user's decision; the grammar's
    units are observations, so it would work on a grid, but availability and `align` limits need
    an intraday pass.
*   **`norm:vol` assumes independent daily changes** (sqrt N; the user's choice): with momentum
    or mean reversion the true N-period dispersion differs; `norm:z` on the change itself is the
    alternative that absorbs it.
*   **`surprise` availability** uses the registry event's usual time; a release that moved time
    (or one with no registry time - the day's end is used) is not modelled per occurrence.

## Forecast: open items from the first build (framework A)

**Found:** 2026-10-06. **Where:** `infra/models/forecast`. **Status:** open.

*   **CPI has only 5 consensus prints** in the harvested calendar (the CPI release's
    `calendar_pattern` barely matches the calendar's report names): surprises exist for 14 releases
    with 100+ prints, not CPI / PCE. Check the pattern against the stored report names
    (`backfill_econ_calendar.py --names`).
*   **Evaluation code is duplicated in spirit between B1 and A** (`infra/models/autocorr/evaluate.py`,
    `infra/models/forecast/evaluate.py`; A imports B1's `staggered_pnl` / joint spanning): move the
    shared pieces (staggered books, Clark-West, spanning, placebos, BH families) into one
    `infra/models/evaluation.py`.
*   **B1 is a special case of A** (its regressor is sign(past) x X): it could be re-expressed as an A
    spec once A is proven; kept separate for now.
*   **Cross-sectional A** (a panel / rank model across countries x structures) is not built.
*   **Surprises test post-release drift only** (decisions at 15:00 New York, after the morning's
    reaction). The release-day move itself, and its reversal (NFP / GDP hint), belong to the
    event-study and autocorrelation frameworks.

---

## Listed option chains (ZN, SOFR): priced, on hold

**Found:** 2026-10-06. **Where:** would extend `infra.pipeline.futures_options_iv` (today ZN near-the-money only) / `infra.pipeline.daily_options`. **Status:** on hold (user decision 2026-10-06: priced, not fetched).

**Why:** the better source for skew, the short end of the vol term structure and a REAL open-interest / gamma map: CME settles every strike daily and publishes open interest per strike (Databento `statistics`), unlike the DTCC swaption flow (too thin for skew - see "Swaptions: open items"). Direction is still unknown; the candidates are aggressor-side `trades` on liquid strikes and the CFTC TFF combined-minus-futures-only options delta by category (already stored).

**Products (parents resolving on GLBX, 2026-09-29):** ZN `OZN.OPT` (monthly/quarterly), weeklies `ZN1` / `ZN2` / `ZN3` (Friday), `WY1` (Wednesday), `VY1` (Monday); SOFR `SR3.OPT` plus mid-curves `S0`, `S2`-`S5` (listed from 2020); `SR1.OPT` exists too. A parent also carries spreads/combos: outrights were 44% (OZN) / 66% (SR3) of the parent's statistics cost in Sep 2026.

**Dry-run prices (free metadata, 2026-10-06; est. = parent x outright share, max = parent):** ZN monthlies 2016-2026 ~$4.1 (max $9.3), all ZN products ~$6.3 (max $14.3); SOFR all products 2020-2026 ~$12 (max $18); definition snapshots ~$0.0102 each - monthly grid ~$8 (ZN) / ~$5 (SOFR); weekly snapshots for ZN weeklies' history (they live < 1 month) ~$28 extra; ongoing ~$0.003 (ZN) + ~$0.009 (SOFR) per trading day, ~$7/yr both with snapshots. Proposed order when resumed: ZN monthlies 2016+, SOFR 2020+, ZN weeklies forward only (~$22 up front). Script: the session's scratch `price_chains.py` (rebuild from this description: `estimate_cost` per parent per year, and per outright id from a cached definition snapshot).

---

## Inflation swaps: open items from the first build

**Found:** 2026-10-07. **Where:** `infra.pipeline.inflation_swaps` (root CLAUDE.md 16). **Status:** open.

*   **USD CPI fixings not decoded:** ~7,500 no-upfront 12-month swaps dated the 1st of a month to the 1st a year later, mostly traded months after their start date (2024-09..2026-10). They look like the CPI fixings market (market-implied YoY CPI for single months - a direct benchmark for the inflation nowcast), but no fixed month offset reproduces realised CPI-U NSA YoY for fixings already fully known: best maturity - 3/4 months, median error 0.16pp, where an exact mapping would give ~0. Next: the convention (CME / ISDA fixing conventions; whether the SDR reports the fixing month in the dates or an index level; interpolation), e.g. by matching prints around CPI release days.
*   **EUR (HICPx, 126 trades a day) and UK (RPI, 94) not built:** only ~9% start at spot, maturities cluster on the 15th, half carry an upfront - standardized-date conventions to map before a spot-tenor close makes sense.
*   **No seasonality adjustment:** ZC rates are on CPI-U NSA with the 3-month lag; short tenors and forwards starting mid-year carry CPI seasonality (1y moves ~7bp a day vs 1.9bp at 10y).
*   **TIPS not built yet:** FedInvest lists TIPS (we filter them out - `TREASURY_TYPES`), index ratios follow from CPI-U NSA (stored); a real curve would give the long history (2008+) and the per-bond TIPS-ZC basis. Planned next (user, 2026-10-07).
*   **The pure method only:** an adjusted method (moving prints by the futures, as the OIS closes) would need a breakeven-futures hedge ratio - none obvious; the wide window is the trade-off instead.

## C: covariance conditioned on slow states - short sample, untested combinations

**Found:** 2026-10-07. **Where:** `infra/models/stats` (rule `grid`, `SimilarityPCA`), scratch runs `c_*` in
`Database/Derived/ModelRuns`. **Status:** open (research).

*   **The front-end shape (SR1 12 months out minus SR1 now) only exists from 2019-09** (SR1's launch;
    the 12-month contract settles properly from 2019-09-27 - its first day settled at 0, a "+9,779bp"
    shape). SR3 is on disk only from 2025-03 (the cycle's universe). So the conditioned-covariance test
    covers 2021-10..2026-09 only. A 2012+ test needs a pre-2018 front-end series: Eurodollar (GE)
    settlements from Databento (daily statistics, a few dollars - price it with a dry run first), or
    the 2y's 6-month change as a cruder proxy (tried: the HMM on it collapsed, a rule on it untested).
*   **Not built:** similarity on the slow state x the HMM probability of the fast regime (combined
    weights); conditioning on flow features (positioning, vol of vol) instead of the macro state.
*   **The `c_*` runs are scratch** (no spec in a registry; their `feat:` inputs and overrides live in
    their `meta.json`): delete or re-create them as named specs if the work continues.

---

## Cross-currency basis: open items from the first build

**Found:** 2026-10-07. **Where:** `infra.pipeline.xccy_basis`, `infra.processing.dtcc_xccy` (root CLAUDE.md 16). **Status:** open.

*   **The short end is thin:** 3m / 6m closes on 22-45% of days (about one print a day) - the short-end hedge market is FX swaps / forwards. Candidates: DTCC's FOREX report (FX forwards ~120 and FX swaps ~50 a day in EUR/USD; ARCHIVED since 2026-10-07 - `DTCC_REPORTS` includes FOREX, backlog from 2024-09-30 - nothing extracts it yet) or CME FX futures (6E / 6J / 6B / 6C via Databento, quarterly, from 2010, cents).
*   **No external validation:** no free basis series to compare against; levels and daily moves are plausible. A CIP check against FX forwards (FOREX report) would be the first real test.
*   **CHF (~8 clean prints a day) and AUD (trades against BBSW, not OIS) not built.**
*   **Hedged yields (the purpose) not built yet:** need EUR / GBP / JPY / CAD OIS curves (EUR, GBP pure closes exist; JPY / CAD OIS products in the same files not extracted), the foreign bond yields (DE, UK stored; JGBs from Japan's MoF CSV; Canada from the Bank of Canada's free API) and the two hedge conventions (rolling 3m FX hedge; term hedge via the cross-currency swap). User: discuss after.

---

## Positioning: roadmap (by player/sector vs aggressor)

**Found:** 2026-10-07 (user brainstorm). **Where:** proposed layout below; data already stored: CFTC TFF and NY Fed primary dealer statistics (CLAUDE.md 24), SR3 / ZN option OI by strike (statistics), sec lending (19), the CTA model (21). **Status:** roadmap, nothing built beyond the CTA model and the two raw stores.

**Two meanings, never blurred (user definition):** (1) **player/sector** - who holds the risk and who is FORCED to trade (slow, levels, crowding); every long has a short, so this only means something per sector, and the sectors must add up to supply outstanding (Treasury MSPD; Fed Z.1 quarterly as the anchor) - a weekly "Treasury flow of funds" nowcast from the fast proxies below, reconciled to the identity, is the unifying target. (2) **aggressor** - who paid up to trade and at what prices (fast, flows, pain levels, who is trapped). Rules-based player models sit between them: they forecast FUTURE aggressor flow (the CTA's expected flows), and aggressor data tests whether they traded.

**Where it lives (proposal, user question 2026-10-07):** these are mostly deterministic measures, not models, so NOT `infra/models`. Same rule as the rest of the repo: (a) raw sources in `api` / `processing` / `pipeline` / `RawData` (TFF, FR 2004 already there); (b) a measure with no fit (signed volume, entry-price distribution, index extension, convexity need, dealer gamma, residual specialness) = pure code in `infra/analytics/positioning/` + read/store in `infra/pipeline/positioning.py`, persisted to `~/Database/Derived/Positioning/<measure>` through the cycle's `DERIVED_METRICS` (BELOW the models, so the daily cycle can run it - the cycle may not import `infra/models`); (c) anything with fitted parameters and a predict step stays in `infra/models` (CTA; the top-down scale filter; risk-parity / pension player models); (d) every measure is exposed to strategies and frameworks as a feature-maker input (a `pos:<measure>:<column>` input kind; model outputs already arrive via `model:<run>:<column>`). Name `Derived` is the storage area, not a code package.

**(1) Player / sector - sources and critique**
* **CFTC TFF** (stored): leveraged funds' Treasury futures short is mostly the futures leg of the cash-futures BASIS trade, asset managers' long the other side - so use LF short as basis-trade SIZE, and dealers' net / AM net vs benchmark duration for direction. Sum across contracts in DV01 (futures DV01 from `infra/analytics/futures_basis.py`). Also SR3 in TFF. First real use (2026-10-03, UB basis residual): correlation 0.00 - positioning didn't explain it.
* **Primary dealers FR 2004** (stored): weekly net positions by maturity bucket = the mirror of customers in aggregate; plus financing and fails. Mind series breaks (TOFIX "Primary dealer statistics").
* **Repo / specialness**: use the RESIDUAL vs our lifecycle profile (`infra/analytics/specialness.py`) as short demand; NY Fed lending bids vs accepted; when-issued bids before settlement; fails (FR 2004); sponsored repo growth (FICC, verify frequency) as basis-trade proxy. Confounders: reopenings, SOMA, quarter-ends.
* **Custody / holdings, free**: Fed H.4.1 foreign official custody (weekly), TIC SLT (monthly), H.8 bank holdings (weekly), Z.1 (quarterly, sector identity), SEC N-PORT (fund holdings by CUSIP, quarterly, lagged), N-MFP (MMFs), Treasury investor-class auction allotments (monthly; we hold the auctions store). Paid: State Street's flow indicators.
* **Surveys**: JPM Treasury client survey and BofA FMS are proprietary (no legitimate free history); free: NY Fed SPD/SMP (expectations vs market pricing = a mispositioning gauge), SCOOS (leverage terms). Low priority, contrarian only.
* **Top-down regressions**: bond fund / ETF daily returns on key-rate factors -> active duration of core / core-plus managers (the "real-money duration" proxy), VALIDATED against N-PORT holdings; ETF shares outstanding (TLT, IEF...) for flow-implied demand; HF index style analysis (regularised, time-varying). Best use: keep bottom-up SHAPE, estimate a few SCALE factors (see "CTA: top-down positioning").
* **Rules-based player models**, priority order: (i) **index month-end extension** - passive trackers add duration at month end; computable from our securities table + OTR map + FedInvest prices, all on disk; (ii) **pension / balanced rebalancing** at month/quarter end (needs equity index data); (iii) **mortgage convexity hedging** - iShares MBB's published daily effective duration change x agency MBS outstanding; (iv) **basis traders** - size from implied repo vs funding (the basis models) and CME margins, validated against TFF LF shorts; (v) **risk parity / vol targeting** (needs equity futures); (vi) **dealer option gamma** on SR3 / ZN from OI by strike (sign assumption to test: customers long puts / short calls); (vii) **hedged foreign buyers** (Japanese lifers: hedged yield pickup vs JGBs; MoF weekly/monthly flows to validate).

**(2) Aggressor - design**
* **Signed volume from the exchange flag**: Databento `trades` carries the aggressor side - no Lee-Ready. Cumulative signed volume per contract in DV01, summed by root (rolls cancel). Cost: front 1-2 contracts per root, windows or a small daily universe, under the same coverage/cost guards as the 1-second stores; never a chain.
* **Entry-price distribution by side**: book each day's OI change at the day's VWAP, split by the dominant aggressor (OI up + aggressive buying = new longs; OI down + aggressive selling = longs liquidated), run off proportionally as OI falls -> share of aggressor longs/shorts underwater, stop / pain levels. Improves the classic price+OI quadrant, which says positions opened, not who is net.
* **Size / urgency**: blocks and outsized prints (cleared volume minus regular-trade volume ~ block activity per day, already in the daily store).
* **Absorption**: persistent aggressive buying with no price progress = a large passive seller (links back to (1)).
* **Rates-specific cuts**: aggressor flow along the SR3 strip (where on the policy path people pay up); front vs back contract in the roll window (longs vs shorts rolling); signed flow around releases (calendar + 1-second data).
* **Not available**: swaps - DTCC public reports carry no direction (activity only).

**New ideas from the brainstorm**
* **Asymmetric reaction function as a crowding signal** (user: was going to suggest it): markets react more to surprises AGAINST a crowded position. Per release, the yield response to the surprise (feature `surprise:` + intraday / 1-second moves, the event-study machinery) split by surprise SIGN, rolling; a widening asymmetry = crowding inferred from prices alone, testable against TFF / CTA positioning. A trailing estimate with no predict step -> analytics/feature layer (like `beta(...)`), not a model.
* Swap spreads (basis / receiver activity) and SR3 options skew (hedging demand) as cross-checks.
* **Validation discipline** (as UBS did for the CTA): judge every proxy by what it PREDICTS - forward returns, reversals at extremes - not by plausibility.

**Suggested order:** TFF basis decomposition (data stored) -> index month-end extension -> MBS convexity (MBB) -> option-OI dealer gamma -> residual specialness -> aggressor signed volume (needs `trades` fetch) -> asymmetric reaction -> custody / allotments into the sector-balance-sheet nowcast.

**Progress:** asymmetric reaction BUILT 2026-10-07 (user moved it first; `infra/analytics/positioning`, root CLAUDE.md 33); open items in "Positioning: asymmetric reaction - open items".

---

## Positioning: asymmetric reaction - open items

**Found:** 2026-10-07 (first build and validation, `infra/analytics/positioning/CLAUDE.md` 3). **Where:** `infra/analytics/positioning/asymmetry.py`, `infra/pipeline/positioning.py`. **Status:** open.

* **No predictive evidence yet.** Only the relative measure shows anything (2y, +0.15 with the next month's yield change, ~2 standard errors, one of ~20 tests). Next tests: known episodes (2020-03, 2022, 2023-03, 2023-10); forward SKEW and reversal size instead of forward direction; conditioning a strategy on the measure through the feature maker (`pos:` ids) with the frameworks' evaluation.
* **The surprise measure's sign puzzle:** negative correlation with CTA positions (-0.25..-0.40): after rallies dovish news moves yields more. Candidates: attention regimes (easing phases), the expected move's own fit (a 5-year impact fit lags importance shifts), consensus quality (MarketWatch median, no dispersion). Test by regime (hiking / holding / cutting) and by release.
* **Today's daily surprise slopes can be negative** (2026-09: dovish news moved yields UP, `b_down` -2, `surprise_asym` pinned at 1): the bounded ratio reads a sign flip as maximal asymmetry. Read it with `b_all`; consider NaN when `b_all` is below a floor, or a separate "inverted reaction" flag.
* **CPI is missing from the surprises** (5 consensus prints - the calendar pattern, TOFIX "Forecast"), and PCE has 29: the biggest rates releases of 2021-2024 aren't in family 1.
* **Thresholds and windows are untuned** (k, windows, the 5-year impact lookback, 40-event windows): chosen a priori, not optimised; tune only out of sample against a stated target.
* **Intraday release windows** drop releases at the grid's first point (06:00 NFIB) and use one window length for every release; a 1-second version for the big releases is possible (on-demand 1-second stores).
* **Not in the daily cycle:** a `derived` entry (`DERIVED_METRICS`) rebuilding the daily spec each run (~6 s) and the intraday weekly (~45 s); checks: counts present, values in range.

## Foreign OIS curves (EUR / GBP / JPY / CAD): open items from the first build (2026-10-07)
*   **Holiday calendars:** every non-USD curve schedules on WEEKDAYS (`SwapCurveSpec.calendar = "weekday"`, `infra.analytics.swap_curve._is_bday`). The effective date or a payment date next to a TARGET / UK / Tokyo / Toronto holiday can roll one day differently than the market's, which is a fraction of a bp in par. Fix: encode the four holiday calendars (or the `holidays` package, not in the env) as named calendars.
*   **ESR settlement time vs the 16:15 London close is not verified**, so the EUR short end takes the PREVIOUS day's ESR settlements to stay point in time (`infra.pipeline.ois_curves.esr_periods`). The strip's 1y sits -0.1bp from the 1y close (sd 3.8bp, mostly the day's move). If ESR settles before 16:15 London, use the same day. Also: the live quarter's elapsed €STR fixings are taken at the futures' rate, since we store no €STR fixings (the ECB publishes them free).
*   **JPY and CAD have no short end** (no TONA / CORRA futures on Databento; MX and OSE aren't on it): a flat forward to the first pillar. Free options: the BoJ / BoC overnight fixings plus a policy-meeting step path, as for SOFR.
*   **CAD is the thinnest curve:** a curve on 62% of days, zeros moving 5.4-7.5bp a day against USD's ~5. More days would need a wider window, which costs noise; a futures-adjusted close would need Canadian bond futures (CGB on MX, not on Databento).
*   **Futures-adjusted closes for EUR / GBP: built 2026-10-07** (CLAUDE.md 16), in the daily cycle; the OIS curves use them where present (`method="best"`, 2026-10-07). GBP 2-3y have no hedge (one gilt future).
*   **The GBP short end is the BoE curve of the day before** (the BoE posts day D by the next morning). A day with a big front-end move (an MPC day) bends the curve between that 1-day-old short end and today's 2y pillar. SONIA futures (`SO3`, ICE) would fix it, but ICE is disabled in the daily cycle on cost (section 12).

## UK / DE / JP / CA: the multi-benchmark structure beyond the official curves (2026-10-07)
*   **Done:** the official par curves' benchmark yield P&L (`yield_boe`, `yield_bundesbank`, CLAUDE.md 12). **Missing** (what the US has): per-bond daily prices, a bond reference table and on/off-the-run map, an `otr` yield source, our own fitted curve with per-bond rich/cheap, futures delivery baskets / conversion factors / CTD; JP and CA have no curve at all.
*   **Germany (2026-10-07): reference and per-ISIN prices done** (CLAUDE.md 18, `infra.pipeline.bunds`). The on/off-the-run map and the `yield_otr` bmk for `DE_BOND_<t>y` are done too. Our fitted Bund curve with rich/cheap is done too (from 2022-06, CLAUDE.md 25a). Still to do: the curve BEFORE 2022-06 (no published dirty price, so no first-coupon dates and the 3-decimal accrued; and the universe's short end is missing - needs the long-first-coupon rule applied from the issue day alone, then validated on the overlap); the `yield_curve` bmk and the Eurex baskets / CFs / CTD / DV01 are done (2026-10-07). Open: the Eurex basis before 2025-03-20 (no stored Eurex quotes before; settlements only from 2025-07 for three roots); a net basis needs EUR term funding (an EUR OIS curve exists from 2024-09); the CTD is by implied repo with spot DV01 (no delivery-option or forward-DV01 model as for the US); BTP futures (FBTP) have no basket (Italian bond prices not stored). Caveats: before mid-2022 the short end of the universe is missing (the Bundesbank drops a bond ~4 years after maturity; ~60% priced in 2020, ~35% in 2015), so a fitted curve before then lacks its front; the issue day is approximated as first auction + 2 weekdays; the prices are an 11:15 Frankfurt snapshot (`BUND_PRICES_LOCAL_TIME`), so a pairing with the Eurex settlement (17:15) or a London close must account for the gap.
*   **UK gilt prices are blocked at the source:** the DMO data portal (`www.dmo.gov.uk/data/...`, reports such as D10B gilt reference prices) answers every scripted request with a ShieldSquare bot captcha (checked 2026-10-07 from a residential IP, no VPN - the Bundesbank API answered normally at the same moment). Not worked around, by policy. Options: (1) ask the DMO for automated access / its data terms; (2) a one-off history export done by hand in a browser, imported from a file (daily updates would still need (1)); (3) another free source with per-gilt end-of-day prices - none checked yet (LSE pages carry delayed prices but scraping terms are unclear; Tradeweb FTSE closing prices are licensed).

## Japan: per-bond JGB prices and the rest of the structure (2026-10-07)
*   **Done:** the MoF constant-maturity curve `JP_BOND_<t>y` from 1990 and its `yield_mof` bmk (CLAUDE.md 13).
*   **Per-bond prices are blocked at the source:** JSDA's Reference Statistical Prices (`market.jsda.or.jp/en/statistics/bonds/prices/otc/`, one CSV a day, `files/<year>/ES<yymmdd>.csv`, archive pages back to 2002, posted 10:00 Tokyo for the 15:00 quotes of the day before) answered 429 Too Many Requests to every scripted request on 2026-10-07, including single requests minutes apart. Its terms of use were not reachable either. Not worked around. Options: (1) ask JSDA about automated access and terms; (2) a one-off manual download of the archive, imported from files (no daily update); (3) another source - none checked.
*   Without them: no JP on/off-the-run map, `otr` source or fitted curve. JGB futures (OSE) aren't on Databento, so no basis either.

## Canada: beyond the benchmark yields (2026-10-07)
*   **Done:** BoC benchmark yields `CA_BOND_<t>y` from 2001 and their `yield_boc` bmk (CLAUDE.md 13).
*   **Benchmark switches are corrected from 2011** (CLAUDE.md 13: the BoC page's effective dates + a zero-curve spread). Open: switches before 2011 stay raw - no page captures, and inferring them from the zero curve (Viterbi over candidate bonds) found only half to two-thirds of the page's switches 2011-2026; the long bond's switches are labelled but not corrected; the switch dating has no rule of thumb - the 2y switched a median 92 days after the new bond's last auction, the 10y 7 days.
*   **The BoC fitted zero curve** (0.25-30y, from 1986) is published weekly with a two-week lag: usable as a history / validation source, not daily.
*   **No per-bond prices found yet; no futures:** Montreal Exchange (CGB) isn't on Databento.

## Bundesbank official curve: occasional curve-wide jumps (2026-10-07)
*   **Where:** `yield_bundesbank` (`DE_BOND_<t>y` from the Bundesbank's Svensson parameters, CLAUDE.md 13).
*   **What:** on 6 of 1,108 days 2022-2026 (2 in 2022, 4 in late Aug 2026: 08-27, 08-31, 09-01, 09-02) the curve moves +-10-13bp at every tenor while the on-the-run Bund, our own Bund curve and the Bund future agree within a few bp. The peer bad-print rule can't see a curve-wide move.
*   **Done:** the synchronized family uses our curve instead (CLAUDE.md 12); `yield_sources_agree` (warn) lists such days in the unsynced P&L.
*   **Open:** treat them in `yield_bundesbank` itself (e.g. an outlier test against our curve / the OTR, then NA) - not done, since that bmk is the raw official series by design.

## Hedged yields / FX-implied rates: open items (2026-10-07)
*   **Where:** `infra.pipeline.hedged_yields`, `infra.pipeline.fx_implied` (CLAUDE.md 16). v2 (`rolling_3m_fx`, FX swaps) is done and the default; v1 (`rolling_3m`, OIS + basis) stays alongside.
*   **The OIS short ends behind v1 are wrong where the FX swaps disagree:** EUR before 2025-07-01 (no ESR settlements stored: the curve is flat to its 1y; ~70bp too low in 2024Q4). Fix: backfill ESR settlements 2024-08..2025-06 (priced 2026-10-07 at ~$0.006 for statistics; needs the user's go), then rebuild the EUR curve and everything after it in `derived`. JPY / CAD have no short end at all: a TONA / CORRA fixing source, or the FX-implied rate minus the basis.
*   **Only 3m from FX swaps.** The matched hedge still uses OIS + basis; longer FX forwards (1y+) are thin in the archive. Revisit when the cross-currency basis is checked against them.
*   **Thin days:** GBP has FX swaps on ~42% of days (often one swap), so the carried spread does much of the work there.
*   **`scripts/run_daily_cycle.py --dry-run` fails before 2025-03-10** (Eurex's `definition` cost estimate starts before XEUR.EOBI's available start, 422). The estimate should be clamped like the real fetch. Not fixed: `infra/api/databento_client.py` was being edited by another session.
