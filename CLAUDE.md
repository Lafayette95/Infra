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
