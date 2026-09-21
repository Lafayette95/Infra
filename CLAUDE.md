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
    *   `XEUR.EOBI` (Eurex): Euro-Bund, Bobl, Schatz futures & options.
    *   `IFLL.IMPACT` (ICE Europe Financials): SONIA futures & options, UK Gilt futures.
*   **Data Level:** Level 1 (Top-of-Book / Best Bid & Offer).
*   **Target Schema:** `ohlcv-1m` (1-Minute Bars containing open, high, low, close, volume).

## 2. Mandatory Cost-Protection Rules (Strict Budget Guardrails)
*   **Rule 2.1: Implement Local Storage Caching Always**
    Before executing ANY historical API data request, the script must verify if data exists in a local `.parquet` file exists. If it exists, read it into Pandas with `pd.read_parquet()`. Never allow duplicate queries to charge the Databento wallet multiple times.
*   **Rule 2.2: Continuous Contract Symology**
    When querying historical futures lines over long multi-year horizons, NEVER pass wildcards like `SR3*` or empty asset fields. Instead, strictly pass the continuous front-month contract formatting (`SR3.c.0` for 3-Month SOFR) to avoid downloading exponential rows of illiquid or dead expiries.
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

## 4. Reading Data Schema Definitions & Structural Mapping
When interacting with Databento payloads, map to these native schemas:
*   `ohlcv-1m`: Use for bars. Key fields returned: `symbol`, `open`, `high`, `low`, `close`, `volume`. Index is `ts_event` (start time of bar).
*   `tbbo`: Top of Book Best Bid/Offer (Level 1 Tick stream). Contains: `bid_price_0`, `ask_price_0`, `bid_size_0`, `ask_size_0`.
*   `definition`: Instrument metadata. Key fields: `instrument_id`, `raw_symbol`, `strike_price`, `expiration_date`.

## 5. Reading Symbology Best Practices
Ensure proper ticker naming structures when querying:
*   **3-Month SOFR Continuous:** `SR3.c.0`
*   **US 10-Year Note Continuous:** `ZN.c.0`
*   **Euro-Bund Continuous:** `GG.c.0` (Verify root symbol via active definitions if required)
*   **3-Month SOFR Option Root:** `OQ` (e.g., utilize `OQ` inside definitions framework).

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
