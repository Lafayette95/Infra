# Databento API Implementation Guidelines for Python

### **Project Goals**
    1. Build an automated Python data pipeline that pulls data from Databento API (STIR Futures and Options)
    2. Process and clean the data using pandas, saving it to a local parquet.
    3. Build an interactive Plotly Dash web app to display key metrics, trends, and charts.

## 1. System Architecture & Constraints
*   **Environment Configuration:** 
*   This project uses a Conda environment named `infra-env` ( which uses Python 3.12.14; Use PEP 8 styling )
*   To install packages or run scripts, NEVER use bare `python` or `pip`. ALWAYS use the explicit environment prefix paths:
    1. Run python: `/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python`
    2. Install packages: `/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/pip install <package>`
*   **Asset Universes:**
    *   `GLBX.MDP3` (CME Globex): SOFR futures & options, €STR futures & options, US Treasury Futures.
    *   `XEUR.EOBI` (Eurex): Euro-Bund, Bobl, Schatz, BTP futures & options.
    *   `IFLL.IMPACT` (ICE Europe Financials): SONIA futures & options, UK Gilt futures.
*   **Data Level:** Level 1 (Top-of-Book / Best Bid & Offer).
*   **Target Schema:** `ohlcv-1m` (1-Minute Bars containing open, high, low, close, volume).

## 2. Mandatory Cost-Protection Rules (Strict Budget Guardrails)
*   **Rule 2.1: Implement Local Storage Caching Always**
    Before executing ANY historical API data request, the script must verify if data exists in a local `.parquet` file exists. If it exists, read it into Pandas with `pd.read_parquet()`. Never allow duplicate queries to charge the Databento wallet multiple times.
*   **Rule 2.2: Absolute Contracts Only (API + Database)**
    The API is queried ONLY by absolute contract raw symbol (`SRZ4`, `ZNH5`) via `stype_in="raw_symbol"`. NEVER pass wildcards (`SR3*`), empty symbols, spread instruments, or relative/continuous tickers (`SR3.c.0`) to the API. Only absolute tickers are saved to the database. Relative tickers (`SR3.c.0` calendar, `SR3.v.0` trailing-average-volume) are resolved locally from the contracts table (built from `definition` snapshots, outrights only: `instrument_class == "F"`) by `infra/relative`, so only the contracts actually needed are downloaded.
*   **Rule 2.3: Safe Filtering for Options Chains**
    Options chains cause a data payload explosion. To fetch options data efficiently:
    1. Query the tiny `definition` schema first to return active contract IDs text data.
    2. Filter the resulting dataframe locally in Python by strike/expiry.
    3. Query the pricing data (`ohlcv-1m` or `tbbo`) ONLY using the precise array of isolated contract IDs.

## 3. Code Architecture Guidelines
*   Modularity re-usability is SUPREME: including parent functions and child functions wrappers for more precise usage
*   Explicitely, I want
    1. Read function to first check the existing relevant parquet files. Evaluate which data is not available on disk, and only then query API (and save it down)
    2. All functions in 1. are  distinct funtions (reading/writing from file, checking missing data, reading API); API distinct code file, reading/writing parquet distinct file, dash distinct etc
    3. Code lives in ~/Repos/Infra; DataBase lives separately in ~/Database/ohlcv-1m/Futures
*   **Known edge cases that are found but not fixed immediately MUST be recorded in `TOFIX.md`** (repo root), not just mentioned in a conversation, commit message, or PR description and then lost. Each entry needs enough detail (where, why it happens, why it wasn't fixed now, options considered) that someone can pick it up later without re-deriving the analysis. Remove the entry once actually fixed, noting that in the fixing commit.

## 4. Reading Data Schema Definitions & Structural Mapping
When interacting with Databento payloads, map to these native schemas:
*   `ohlcv-1m`: Use for bars. Key fields returned: `symbol`, `open`, `high`, `low`, `close`, `volume`. Index is `ts_event` (start time of bar). **Trade-based, not quote-based**: `open`/`high`/`low`/`close` are actual executed transaction prices (not bid/ask/mid), and `volume` is summed trade size — confirmed against Databento's own schema docs. For bid/ask instead, use `tbbo` below.
*   `tbbo`: Top of Book Best Bid/Offer (Level 1 Tick stream). Contains: `bid_price_0`, `ask_price_0`, `bid_size_0`, `ask_size_0`.
*   `definition`: Instrument metadata. Key fields: `instrument_id`, `raw_symbol`, `strike_price`, `expiration_date`.

## 5. Reading Symbology Best Practices
Stored/queried tickers are ABSOLUTE (Databento `raw_symbol`); relative tickers exist only in code:
*   **Futures roots (parent symbol `<root>.FUT`), all in `FUTURES_ROOTS` (`infra/config.py`):**
    *   STIR: `SR3` (3M SOFR), `ESR` (3M €STR) on `GLBX.MDP3`; `SO3` (3M SONIA) on `IFLL.IMPACT`.
    *   US Treasuries (CBOT, on `GLBX.MDP3`): `ZT` (2Y), `ZF` (5Y), `ZN` (10Y), `TN` (Ultra 10Y), `ZB` (Classic Bond), `UB` (Ultra Bond).
    *   Eurex (`XEUR.EOBI`, data only from 2025-03-10): `FGBL` (Bund), `FGBM` (Bobl), `FGBS` (Schatz), `FBTP` (BTP). `GG` is NOT a valid Eurex root.
    *   ICE (`IFLL.IMPACT`, data from 2018-12-23): `R` = UK Long Gilt. `G`, `SOA` (1M SONIA) and `SON` are not used.
*   **ICE symbol quirk:** raw symbols look like `R   FMH0025!`. Each expiry also lists a non-trading `_Z` twin one day earlier, and `R` has off-cycle April/May-coded contracts; both would corrupt relative ranks, so ICE roots filter with `ticker_regex` (quarterly `H/M/U/Z` `!` contracts only).
*   **Absolute examples:** `SRZ4`, `ZNH5`, `FGBL SI 20250606 PS` (Eurex raw symbols are not CME-style: rank by definition `expiry`, never by parsing symbols). SR3's letter lags its expiry by a quarter (`SRZ4` expires 2025-03-18).
*   **Relative notation:** `<root>.<c|v>.<rank>`, e.g. `SR3.c.0`. Ranking uses the quarterly expiry cycle (SR3 serial contracts are excluded), configured per root in `infra/config.py`. `c.0` on ZN/Bund sits in an expiring, thin contract for weeks; prefer `v.0` there.
*   **3-Month SOFR Option Root:** `SR3.OPT` (`OQ` does not resolve - verified against the real API 2026-09-21; `SR3.OPT` returns 3,358 real option contracts, e.g. `SR3U6 C9762.5`, same convention as the futures root plus `.OPT`).

## 6. Database Architecture Rules: STIR Futures & Options (1-Min OHLCV)

## 6a. Storage & Indexing Strategy
*   **Index Structure:** Disk storage must be **completely flat**. Row indexes (`index=False`) are banned in the files.
*   **In-Memory Lifecycle:** Reconstruct MultiIndexes (`timestamp`, `ticker`) exclusively in RAM *after* data is loaded.
*   **Column Layout:** Value metrics (`open`, `high`, `low`, `close`, `volume`) must remain separate columns to ensure columnar query acceleration.

## 6b. Fixed-Point Scaling Constraints
*   **Price Transformation:** All price columns must be multiplied by `10000` and cast to `int32` before writing.
*   **Read Decoding:** Scaled columns must be divided by `10000.0` inside memory before computations occur.
*   **Target Columns:** `open`, `high`, `low`, `close`, and options `strike`.

## 6c. Data Schema Specifications

### Futures Parameter Mapping
*   `timestamp`: 64-bit Timestamp (`datetime64[ms]` or `INT64`)
*   `ticker`: String Category/Dictionary (`category` or `BYTE_ARRAY`)
*   `open`, `high`, `low`, `close`: Fixed-Point Integers (`int32` or `INT32`)
*   `volume`: Unscaled Integer (`int32` or `INT32`)
*   `open_interest`: Unscaled Integer (`int32` or `INT32`)

### Options Parameter Mapping
*   `timestamp`: 64-bit Timestamp (`datetime64[ms]` or `INT64`)
*   `underlying`: String Category matching the future contract ticker (`category`)
*   `option_type`: String Category limited to `P` or `C` (`category`)
*   `strike`: Fixed-Point Integer (`int32`)
*   `expiry`: 32-bit Integer Day Epoch (`datetime64[D]` or `INT32`)
*   `open`, `high`, `low`, `close`: Fixed-Point Integers (`int32`), fallback to `float32` only if premium drops below `0.0001`
*   `volume`, `open_interest`: Unscaled Integers (`int32`)

## 6d. File I/O & Partitioning Engine Settings
*   **Partition Columns:** `["year", "quarter"]` (calculated from the record timestamp).
*   **Partition Format:** Hive-style hierarchical file organization.
*   **Prohibited Partitions:** Never partition by `ticker`, `symbol`, or `day`.
*   **Read Optimization:** Force PyArrow dataset predicate pushdown using C++ expressions before moving bytes to Pandas.
*   **Compression Engine:** `ZSTD` (Compression Level: `5`).

## 6e. Trading-Day Semantics
*   **"Daily" means the exchange's TRADING day, not the naive UTC calendar day.** Config (per-dataset session hours, verified against the exchange, with a source URL) lives in `infra.config.TRADING_HOURS` / `TradingSession`. The pure bucketing function is `infra.trading_calendar.trading_day(index, dataset)`. Consumed by `infra.relative` (roll-day and `.v.N` volume-ranking assignment — `daily_volume`, `apply_mapping`) and `infra.processing.resample`'s `"1D"` bucket (requires `dataset`, e.g. `"GLBX.MDP3"`).
*   **CME Globex / CBOT** (`SR3`, `ESR`, and CBOT Treasuries `ZT`/`ZF`/`ZN`/`TN`/`ZB`/`UB`): session opens 17:00 CT and closes 16:00 CT the next day; the exchange's own convention labels the WHOLE session by the day it **closes** (e.g. Sunday evening's session is trade date Monday). `TradingSession.crosses_midnight = True`.
*   **Eurex** (`FGBL`/`FGBM`/`FGBS`/`FBTP`) **and ICE Futures Europe** (`SO3`/`R`) do **NOT** follow that pattern — their sessions sit entirely inside one local calendar day (Europe/Berlin and Europe/London respectively), so the trading day is simply that local date, no next-day label shift. `crosses_midnight = False`. Re-verify against `TradingSession.source` if hours change (ICE's SONIA/Gilt hours changed 2026-07-06).
*   **Fetch/coverage windows stay separate and unaffected.** The download-deduplication windows in `infra/coverage` and `infra/storage` (Rule 2.1) keep using simple UTC-day windows — that's a caching/cost concern, not a trading-day one, and must not be conflated with it.
*   **Intraday buckets (1m..4h) stay UTC-clock-aligned**, unaffected by this section — see the caveat this implies, documented in `infra/processing/resample.py` (and `TOFIX.md`).

## 7. Timezone Handling
*   **Everything is UTC except the pixels on screen.** Data on disk, everything returned by `infra/pipeline` (incl. `load_series`), all of `infra/processing` (incl. resampling quarter boundaries), `infra/relative`, `infra/storage` and `infra/api` MUST always be tz-naive UTC. NEVER localize, convert, or shift a timestamp anywhere outside `infra/dashboard` — with the sole exception of `infra/trading_calendar.trading_day`, whose job is precisely to compute the trading-day LABEL (6e), never to alter a stored timestamp.
*   **Timezone display conversion is Dash-only.** It lives exclusively in `infra/dashboard` (e.g. `infra/dashboard/timezones.py`), as a pure, isolated conversion applied ONLY to the values handed to Plotly for rendering (x-axis ticks, hover labels). It must never mutate a DataFrame coming out of `infra/pipeline` — build a separate display index/copy, never write back to `bars`/`df`.
*   **Bucket boundaries are a data concept and stay fixed regardless of display timezone.** A `1D` bar is bucketed by the exchange trading day (6e) and a roll is assigned to a trading day the same way, no matter what timezone is selected for display; only the rendered label of an already-bucketed instant may shift. Never let a display timezone change which rows fall in the same bucket, which day a roll is assigned to, or any grouping/aggregation key. Trading-day bucketing (6e, a data concept) and display timezone (this section, a rendering concept) are orthogonal — they compose, but must never be confused: 6e decides which day a bar belongs to; this section only decides how that already-decided instant is drawn on screen.

## 8. Daily Settlement & Open Interest Pipeline
*   **Deliberately a separate pipeline from section 6's 1-min OHLCV rules**, not a variant of it: different schema (`statistics`, not `ohlcv-1m`), different cadence (once per trading day), a genuinely different code path (`infra/pipeline/daily.py`, `infra/processing/statistics.py`). It reuses the generic, dataset-agnostic primitives as-is (`infra.storage.parquet_store`, `infra.storage.coverage_store`, `infra.coverage.intervals`) — only pointed at a new root and a new coverage manifest, no new storage code.
*   **Storage roots:** `~/Database/Daily/Futures` and `~/Database/Daily/Options` (siblings of `~/Database/ohlcv-1m`, NOT nested under it), same hive `year=/quarter=` partitioning, flat, ZSTD level 5 — CLAUDE.md 6a/6d apply unchanged. Separate coverage manifests (`~/Database/Daily/_coverage/futures.parquet`, `.../options.parquet`) so gap-tracking never crosses between pipelines, or with `ohlcv-1m`'s.
*   **Scope: futures (absolute + relative tickers) and options, both supported.**
*   **Futures schema:** `timestamp` (the trading day, `infra.trading_calendar`), `ticker` (absolute contract, category on decode), `settlement_price` (nullable `Int32`, ×10000 fixed-point per Rule 6b — nullable because a day can have OI without a settlement yet, or vice versa), `open_interest` (nullable `Int32`, unscaled). Key columns: `["timestamp", "ticker"]`.
*   **Options schema:** `timestamp`, `underlying`, `option_type` (`C`/`P`), `strike` (`int32`, ×10000 — never null, comes from `definition`, not the statistic itself), `expiry` (`int32` day-epoch, matching the 1-min options schema, 6c), `settlement_price` / `open_interest` (nullable `Int32`, same reasoning as futures). Key columns: `["timestamp", "underlying", "option_type", "strike", "expiry"]`. Implements Rule 2.3 exactly as the 1-min options pipeline does: `infra.pipeline.options.load_definitions` (reused as-is, no new definitions-fetching code) → filter locally → `api.fetch_statistics_by_instrument_ids` (the isolated array only). One difference worth knowing: Databento caps a single request at 2,000 symbols (`api.MAX_SYMBOLS_PER_REQUEST`, verified empirically — a 3,358-contract SR3 chain request failed without batching), so both `fetch_statistics_by_instrument_ids` and `fetch_ohlcv_by_instrument_ids` batch internally now.
*   **`infra/pipeline/daily.py` handles futures, `infra/pipeline/daily_options.py` handles options** — mirrors the `infra/pipeline/futures.py` / `infra/pipeline/options.py` split exactly. (`daily.py` was NOT renamed to `daily_futures.py` for this symmetry, to avoid disrupting its already-shipped call sites — a minor, accepted naming asymmetry.)
*   **Source fields, verified against the real API 2026-09-21:** the `statistics` schema's `stat_type` enum (`databento_dbn.StatType`) — `SETTLEMENT_PRICE = 3`, `OPEN_INTEREST = 9`. Multiple updates can arrive per trading day (a preliminary settlement, then a final); the LAST one (by `ts_recv`) wins.
*   **Trading-day resolution is NOT simply `trading_day(ts_recv, dataset)`.** Databento's own `ts_ref` field is the exchange's reference date for a statistic and is preferred when present (`infra.processing.statistics.resolve_trading_day`, shared by both futures and options - verified against real data for both). Two verified quirks:
    1. **Open interest's `ts_ref` is the PRIOR trading day**, not the day it was published — OI published early one session actually reports the end-of-day figure as of the PREVIOUS session's close. Bucketing by `ts_recv` instead would silently misclassify it by one day (confirmed against real CME futures `SR3Z4` AND a real SR3H5 option contract: an OI update received 2025-03-12 01:50 UTC carries `ts_ref = 2025-03-11`).
    2. **`ts_ref` is null for Eurex** (always, in samples) **and sometimes for ICE** — for both, falling back to `trading_day(ts_recv, dataset)` is exact anyway, since neither venue's session crosses midnight (6e).
*   **Known residual gap (`TOFIX.md`):** when ICE's `ts_ref` is null specifically for an OPEN_INTEREST row, the `ts_recv`-based fallback can misattribute a same-value re-publish of the prior day's OI to the current day instead. Narrow (one venue, one stat type, only when `ts_ref` happens to be missing) and not yet fixed.
*   **Relative tickers (`SR3.c.0`, `SR3.v.0`) work on daily data too** (`infra/pipeline/relative_daily.py`), reusing `infra.relative`'s contract/mapping machinery (`group_by_root`, `needed_contract_windows`, `calendar_mapping`, `volume_mapping`, `apply_mapping`) UNCHANGED — only the data source differs (`infra.pipeline.daily` instead of `infra.pipeline.futures`). This works because a daily row's `timestamp` IS already the trading day, and `infra.trading_calendar.trading_day` is idempotent on an already-resolved trading day (verified across a full year, both DST transitions, every configured dataset). `.v.N` ranking still needs `daily_volume()` from 1-minute OHLCV bars (the daily statistics data carries open interest, not volume) — so a volume-ranked daily series has an implicit dependency on the 1-minute pipeline; `load_relative_daily` fetches 1-minute bars too when needed for ranking (`fetch_missing`-gated, same cost guardrail), but only as a ranking input — they're never part of the returned series.

## 9. Risk-Neutral Density Analytics
*   **A derived analytics layer, not a data pipeline** (`infra/analytics/`): pure functions, no API calls, no new storage — reads already-fetched data via `infra.pipeline.daily_options`/`infra.pipeline.daily`. Extracted densities are computed on demand, never persisted.
*   **Methodology (Breeden-Litzenberger via butterfly spreads on a fitted smile, not raw finite differences):** `forward.py` (`implied_forward_and_discount`) fits the option chain's own put-call parity (`C - P = discount × (forward - strike)`, linear in strike) to get the forward price and discount factor — **no separate risk-free-rate curve pipeline needed**; both fall out of the chain itself, verified against real `SR3U6` data 2026-09-21 (implied forward matched the future's own settlement to 4 decimal places, fit was line-straight across ~100 strikes). `black76.py` prices/inverts implied vol (out-of-the-money side at each strike — puts below the forward, calls above — since deep in-the-money options have near-zero vega and a poorly-conditioned inversion). `smile.py` fits a cubic spline to the resulting vol-vs-strike smile (chosen over SVI for v1: simpler, no per-day nonlinear calibration, adequate given the dense real strike grids observed — 100+ strikes per SR3 quarterly expiry). `rnd.py` rebuilds a dense synthetic call-price curve from the fitted smile and applies the discrete second-difference, clips negative noise to 0, and normalizes to integrate to 1 over the **observed strike range only** (truncated support, not the true unconditional density).
*   **SR3 options are AMERICAN-exercise, not European** (verified via CME's own FAQ — "SOFR Options utilize the American Style option exercise"). Breeden-Litzenberger is derived for European options; this is a documented approximation, not an oversight. In practice the put-call-parity fit was still extremely clean on real data (CME's settlement price is model-derived, not a raw last-trade), suggesting the distortion is small for short-dated, near-the-money SR3 quarterlies — treat deep in-the-money / longer-dated tails with more skepticism than the body of the distribution.
*   **Scale note:** SR3 option strikes are quoted at exactly 100× the underlying future's own price scale (e.g. strike `9619.52` ↔ future `96.1952`) — confirmed empirically via the parity-implied forward matching the future's real settlement. `infra.analytics` works entirely in whatever scale the input chain uses (strike-axis-invariant); the returned density is "probability per unit of the input strike axis," not automatically re-scaled to the future's natural units.
*   **Practical range note:** avoid extracting a density extremely close to the option's own expiry (single-digit days) — the true distribution is genuinely a near-delta-function then (correct behavior, not a bug), but is hard to resolve usefully on a strike grid sized for the chain's normal spread. Verified stable and sensible on real `SR3U6` data across 6 consecutive trading days at T≈17 days: forward and peak strike drift smoothly day to day, density stays unimodal with a modest persistent skew, always integrates to 1, always non-negative.
*   **`grid_points`** (`extract_rnd`, default 400) controls the finite-difference resolution — far finer than real strike spacing, since it operates on the smooth *fitted* smile, not raw market prices.

## 10. Dashboard Structure (Multi-Page)
*   **One Dash app, multiple pages** (`dash.register_page`), not a separate app per concern — reuses the same server, assets, CSS and theme system; picking this over a second Dash process/port was a deliberate call (simpler to run, Dash's own built-in answer to "different view, same app").
*   **The theme control moved to a shared, persistent app shell** (`infra/dashboard/shell.py`, `build_shell` = nav links + theme toggle wrapping `dash.page_container`) — it is NOT duplicated per page. Each page's own `Output`/`Input` still reads `theme` (an `Input`, e.g. to color its own Plotly figures), but only the shell owns `Output("root", "className")`; a second callback writing the same Output would be a Dash error, and duplicating the theme control's id across pages would collide (Dash Pages shares one DOM/id-namespace across all registered pages, even though only one page's content is mounted at a time).
*   **Page modules self-register at import time** (`dash.register_page(__name__, path=..., name=..., layout=...)`), no `pages/` folder — kept flat with the rest of `infra/dashboard`, one file per concern (`infra/dashboard/layout.py` → `/` "Futures", `infra/dashboard/rnd_layout.py` → `/rnd` "Risk-Neutral Density"). `dash.register_page` requires the `Dash(use_pages=True)` instance to already exist, so `infra/dashboard/app.py`'s `create_app()` constructs the app FIRST, then imports the page modules (triggering their registration), then wires up `app.layout`/callbacks.
*   **The RND page is read-only, never fetches** — `infra/dashboard/rnd_selectors.py`/`rnd_callbacks.py` read directly from whatever's already cached under `Database/Daily/Options` (bypassing `infra.pipeline.daily_options.load_daily_options`, which needs a `definitions_day` matching an exact prior snapshot — the page's chosen valuation day usually isn't that day). Populate data via `scripts/update_daily_options.py` first. Strike/density are shown on the future's natural `/100` scale (CLAUDE.md 9's scale note), not the raw stored scale.
*   **Bug found and fixed while building this:** the stored `expiry` column is an `int32` day-epoch (CLAUDE.md 6c), not a datetime. Reading it directly with `pd.to_datetime()` silently reads the small integer as *nanoseconds* since epoch, giving a nonsense date near 1970-01-01 - `infra.processing.statistics.decode_expiry_column` is the one correct decoder (factored out of `decode_daily_options` for exactly this reason: a caller reading just the `expiry` column for a dropdown, without decoding the whole frame, still needs to decode it right). `infra/dashboard/rnd_selectors.py` uses it; a regression test (`tests/test_rnd_dashboard.py`) guards against reintroducing the bug.
