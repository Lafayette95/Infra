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
