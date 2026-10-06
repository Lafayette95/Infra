# The central feature maker (spec and status)

Root `CLAUDE.md` 31 says where it lives; this file is the spec. Designed with the user
2026-10-06 (daily first). Code: `infra/processing/features.py` (pure: grammar + steps),
`infra/pipeline/features.py` (point-in-time assembly: inputs, availability, alignment). Tests:
`tests/test_features.py`.

## 1. Purpose
One point-in-time way to turn any stored series into model inputs, shared by every framework
(A direction, B1/B2 autocorrelation, C covariance, D conditions), strategies and the dashboard.
It replaces the partial versions that grew separately: `infra/models/prep.py` string steps, the
event study's `conditions.feature` + `PARTITIONERS`, B1's `_x_feature`.

## 2. Inputs
*   **Any series id** (`infra.pipeline.series_panel`), read on its AVAILABILITY timeline
    (`read_available`): a row counts from when it became public, never from its label. A vintage
    series (macro releases) becomes its STATE at each availability instant (`state_timeline`: the
    latest period's value as known then).
*   **Each source declares its input kind** (`SeriesSource.kind`, `series_panel.kind_of`):
    `level` (yields, settlements, indices, spreads; the default) or `moves` (the `bmk:` P&L). A
    change is a DIFFERENCE of a level and a SUM of moves; the level of moves is their cumulative sum.
*   **Derived inputs:** `vol(ID,SPAN)` (EWMA vol of daily changes), `corr(ID1,ID2,W)`,
    `beta(ID1,ID2,W)` (rolling, of daily changes), `spread(ID1,ID2[,BETA])`; `evt:to:EVENT` /
    `evt:since:EVENT` (business days to the next / since the last occurrence of a registry event, AS
    KNOWN that day - release-calendar `known_from`); `surprise:<release ticker>` (actual -
    MarketWatch consensus, `econ_calendar.consensus` - NOT Bloomberg's survey; available at the
    release instant = its registry event's time, else the day's end; user decision 2026-10-06);
    `model:<run>:<column>` (a stored model run's prediction, available the day after its label).

## 3. The grammar
A feature = an input, then a KIND, then MODIFIERS, separated by `|`:
`bmk:otr:US_BOND_10y | chg:20 | norm:vol:60 | abs | part:tercile:504`.

| Kind | Meaning |
|---|---|
| `lvl` | the level (moves: their cumulative sum) |
| `gap:ewm:HL` | level - EWMA(level, half-life HL): stretch from trend |
| `chg:N` | change over N observations, boxcar (difference of levels / sum of moves) |
| `chg:ewm:HL` | EWMA of daily changes |
| `x:F:S` | EWMA(level, F) - EWMA(level, S): crossover trend |
| `acc:N:M` | chg:N now - chg:N M observations ago |
| `accr:F:S` | per-observation change over F minus over S (F < S) |
| `range:N` | position in the trailing N range, 0..1 |
| `dd:N` | drawdown from the trailing N high (<= 0) |

| Modifier | Meaning |
|---|---|
| `norm:vol:SPAN` | / EWMA vol of daily changes x the kind's scale: **sqrt N for chg:N** (user decision: kept; it assumes independent daily changes), sqrt((1-l)/(1+l)) for chg:ewm, sqrt(1/F - 1/S) for accr, sqrt(2N) for acc, 1 otherwise - so a vol-normalised change has unit variance on independent moves |
| `norm:z:W` / `norm:z0:W` | trailing z-score, demeaned / not demeaned (/ RMS) |
| `norm:rank:W` / `norm:robust:W` | trailing percentile 0..1 / (x - median) / (1.4826 MAD) |
| `abs`, `sign`, `clip:K`, `pow:P` | shape (`pow` keeps the sign) |
| `lag:N` | N more observations of lag |
| `part:tercile:W`, `part:z:W:K`, `part:sign`, `part:fixed:A:B` | buckets -1 / 0 / +1 |

## 4. The point-in-time contract
1.  **Trailing only**, explicit warm-up (NaN until enough rows; never a short-window value).
2.  **Anything needing a fit sample** (a z-score on the fit window, a fitted beta) is NOT a feature
    step: it is fitted in a model's `fit` and frozen (the `prep.py` stateless / stateful split).
3.  **Output indexed by availability instant**, with `attrs` (feature string, input kind, source).
4.  **Alignment** onto another timeline (`align`): the latest value at or before each instant,
    carried at most `FEATURE_FFILL_LIMITS[source]` (`infra/config.py`; user decision 2026-10-06:
    a limit per source type - daily market data 5 days, monthly macro / surprises 100 days, event
    time 1 day, 1-minute bars 1 hour) - beyond it NaN, never a stale value.
5.  **Daily first** (user decision): intraday grids later.

## 5. Status (2026-10-06)
*   **Built:** the grammar, every kind and modifier; series, `vol` / `corr` / `beta` / `spread`,
    `evt:to` / `evt:since`, `surprise`, `model` inputs; `align`; partitions moved here (the event
    study's `conditions.partition` / `PARTITIONERS` delegate); `state_timeline` moved here.
*   **Migrated, outputs identical (tested):** B1's X features (`move:K`, `absmove:K`, `level`,
    `z:W` are aliases of `chg:K|norm:vol:SPAN`, `...|abs`, `lvl`, `lvl|norm:z:W`; any other
    `x_feature` string is the grammar itself) and its past move.
*   **Not yet** (`TOFIX.md`): `resid(ID; FACTORS; W)`, `rev(ID)`, `pdiff(ID)` as inputs (the event
    study's `period_diff` still does the latter); `prep.py`'s stateless steps as grammar aliases;
    the event study's condition `steps` as a grammar string; intraday features.
