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
