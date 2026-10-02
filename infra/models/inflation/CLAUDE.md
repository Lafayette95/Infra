# infra/models/inflation - US Inflation (additive to the root CLAUDE.md)

Everything in the root `CLAUDE.md` and in `infra/models/CLAUDE.md` still applies (conda
env, modularity, one-way dependencies, point-in-time `as_of`, UTC). Known edge cases still
go in the ONE root `TOFIX.md`, under headings prefixed "Inflation:". This file only adds
what is specific to inflation.

## 1. Scope and layering
*   **This package holds inflation MODELS** (none yet): pure computation on stored data,
    like the rest of `infra/models`. It never fetches or writes storage, and nothing below
    `infra/models` imports it (`tests/test_architecture.py`).
*   **The DATA layer stays in the main pipeline**, because the daily cycle runs it and the
    cycle may not import `infra/models`. The root `CLAUDE.md` section 15 has the short
    version. The pieces:
    *   Config: `infra.config.BULK_DATASETS`.
    *   Clients: `infra/api/bls_client.py`, `infra/api/bea_client.py`.
    *   Read/plan/fetch/store: `infra/pipeline/bulk_series.py`.
    *   Daily-cycle source: `infra/cycle/raw_bulk.py`, the `raw` step's `"bulk"`.
    *   Stores: `~/Database/RawData/BLS`, `.../BEA`. Catalogs: `RawData/_catalog`.
        Coverage: `RawData/_coverage/bulk_series.parquet`.
    *   Script: `scripts/update_bulk_series.py`.
    *   The methodology of all of the above lives HERE.

## 2. The component trees
*   **Datasets:**
    *   `cpi`: BLS `cu`, CPI-U, every item × area, SA/NSA, ~4,000 current series. IDs look
        like `CUUR0000SA0`: `U` = NSA, `S` = SA; `0000` = U.S. city average; `SA0` = all
        items. Monthly periods only. The annual averages (`M13`) and the half-yearly
        values of areas published semiannually (`S01`-`S03`) are not stored.
    *   `ppi_commodity`: BLS `wp`, ~5,200 series. This holds the final demand-intermediate
        demand tree. Headline PPI final demand is `WPUFD4`.
    *   `ppi_industry`: BLS `pc`, PPI by NAICS industry.
    *   `pce`: BEA NIPA tables U20404 / U20405 / U20406 (the ~390-line "underlying
        detail" tree: price indexes, nominal spending, real spending) plus T20804/5/6
        (the monthly headline tables). Headline is `DPCERG`, core is `DPCCRG`.
        Values keep BEA's own scale; the catalog's `scale` column gives it (-6 =
        millions).
*   **Why the bulk files:** the BLS API allows 50 series per call and 500 queries a day,
    and the BEA API needs a key and pages by table. One bulk file holds the whole
    survey, is free and has no limits.
*   **The sources keep NO vintages, so the store builds them:**
    *   A file only ever holds the latest revised history.
    *   Each new file version is stamped with its HTTP `Last-Modified` day. That is the
        publication: 08:30 New York on release day.
    *   `drop_unchanged` keeps only the values that version changed.
    *   Rows use the same layout as the FRED releases (section 3), so
        `infra.processing.releases.snapshot(raw, as_of)` gives the point-in-time view.
    *   **Vintages exist only from the first snapshot on.** On first sight the whole
        history is stamped with that snapshot's publication day: PCE 2026-09-30, CPI
        2026-09-11, PPI 2026-09-10. An `as_of` before that returns nothing from these
        stores. For earlier real-time history of the headlines, use ALFRED
        (`CPIAUCNS`, `PPIFID`, `PCEPI` in `MACRO_RELEASES`).
    *   **Revision patterns:**
        *   CPI NSA is never revised. CPI SA is revised each February, for 5 years, when
            new seasonal factors come out.
        *   PPI is revised for 4 months (footnote `P` = preliminary; not stored, since
            a value change is what matters).
        *   PCE is revised every month, plus annual and comprehensive updates.
    *   Known limits are in `TOFIX.md` ("Bulk inflation snapshots").
*   **Verified on real data 2026-09-30:**
    *   Each file's `Last-Modified` day equals FRED's publication day for the same print.
    *   BEA `DPCERG` equals FRED `PCEPI` exactly on all 812 months.
    *   `CUUR0000SA0` equals `CPIAUCNS`, and `WPUFD4` equals `PPIFID`, for August 2026.
*   **Navigating the tree:**
    *   Each dataset's catalog (latest only) has BLS codes plus `item_name` / `area_name` /
        `group_name`, and BEA `table_lines` (`U20404:374`) plus `parents`.
    *   Python: `infra.pipeline.bulk_series.read_catalog("cpi")`.
    *   Command line: `scripts/update_bulk_series.py --search cpi shelter`.
    *   Values: `read_bulk_from_disk(dataset, tickers, as_of)`, or
        `load_bulk_series(...)`, which fetches a newer version first when `as_of` is
        today.
*   **Weights for bottom-up aggregation:** PCE uses the nominal spending shares in U20405,
    already stored and vintaged. CPI uses the relative-importance tables (section 3).

## 3. CPI weights (relative importance)
*   **What:** BLS's "Table 1": each item's relative importance, U.S. city average, CPI-U
    and CPI-W, in percent of all items, as of DECEMBER of weight year Y. It is released
    with the January Y+1 CPI and is the weight set the index aggregates from during Y+1.
    BLS moves it month to month by relative price changes and doesn't publish those
    interim weights; recompute them as `w_i x (P_i,t / P_i,Dec)` if needed.
*   **Stored:** every weight year 1987-2025 (13,252 rows, ~0.6 MB) in
    `~/Database/RawData/CPIWeights`. Coverage `RawData/_coverage/cpi_weights.parquet` is
    keyed by weight year.
    *   Row: `timestamp` (the publication day), `weight_year`, `section`, `line`,
        `indent`, `item_name`, `item_code`, `cpi_u`, `cpi_w`.
    *   Read: `infra.pipeline.cpi_weights.read_cpi_weights(as_of, weight_year)`. It
        returns only weights PUBLISHED by `as_of`: the latest such weight year, or the
        one asked for.
    *   Code: `infra/pipeline/cpi_weights.py`, parsers in
        `infra/processing/cpi_weights.py`, fetcher `bls_client.fetch_relative_importance`.
*   **Fetched once a year, never re-fetched:**
    *   `due_weight_year(today)` is Y-1 from February 1, else Y-2.
    *   A run makes NO request while every due year is on disk (most of the year).
    *   From February 1 until BLS posts the new year, a run makes one small request a day.
        A year not yet posted is not recorded as covered, so it's asked again the next day.
    *   `cpi_weights_fresh` warns once a due year is 45 days overdue.
    *   The daily cycle's `raw` step runs this as its `cpi_weights` source, AFTER `releases`
        and `bulk`.
*   **Publication day = the January Y+1 CPI's first print, from the ALFRED `CPIAUCNS`
    vintages on disk** (exact for every year, e.g. weight year 2025 → 2026-02-13), else the
    day it was fetched. This assumes the table is released WITH the January CPI, as BLS
    states. The files' own `Last-Modified` is useless here: `2025.xlsx` shows 2026-07-14,
    touched long after release.
*   **Three layouts, one parser each** (verified on every year 2026-09-30):
    *   1987-1996 `.txt` (`coded`): a flat list sorted by the item codes of the time. The
        codes are kept; `indent` and `section` are null. The hierarchy is in the codes
        (`SA1` > `SA11` > `SA111`).
    *   1997-2019 `.txt` (`dotted`): the tree by indentation, with wrapped names joined.
    *   2020+ `.xlsx`: an explicit indent column.
    *   The dotted and xlsx layouts give 294 expenditure rows + 28 special aggregates
        (`section`).
    *   Years before 2020 come out of BLS's per-decade zip archives. A decade is looked up
        in its archive too once BLS moves its loose files there.
*   **`indent` is as published, and not a strict tree in a few places.** For example,
    "Alcoholic beverages" sits one level under "Food", identically in the text and xlsx
    layouts. Never derive a parent's weight by summing children: every published aggregate
    carries its own weight.
*   **`item_code`** is filled from the CPI catalog by normalized, unique item name:
    *   83-92% matched for the dotted/xlsx years.
    *   100% for the coded years, as published. Those are OLD codes that may differ from
        today's.
    *   Unmatched are mostly BLS's "Unsampled ..." items, which have no index of their own.
    *   Series id = `CUUR0000` + `item_code` (NSA, U.S. city average).
*   **Verified 2026-09-30:** the eight major groups (indent 1) weighted with December-Y
    weights reproduce the published Jan/Dec change of headline CPI-U NSA to within 0.0002pp
    for Y = 2023, 2024 and 2025 (e.g. 0.3696% vs 0.3697%). This checks the parsing, the
    code matching and the weight-year convention together.
*   **Not stored:** Tables 2-7 (metro areas, regions, size classes; xlsx years only).
    See `TOFIX.md`.

## 4. Point-in-time history: what exists where
*   **Three point-in-time sources, all in the vintage layout of `infra.processing.releases`**
    (`timestamp` = publication day; `snapshot(raw, as_of)` = the values as known at the end
    of `as_of`):
    *   **ALFRED** (FRED's archive of every past version), via the macro-release pipeline
        into `RawData/Releases`:
        *   The headlines in `MACRO_RELEASES`: `CPIAUCNS` (from 1949), `PCEPI` (2000),
            `PPIFID` (2014).
        *   38 components in `infra.config.INFLATION_ALFRED_SERIES`, each with its first
            vintage noted there:
            *   CPI SA: headline 1972; core, food and energy 1996; the major groups 2009;
                core goods/services, shelter, rent, OER, vehicles, medical etc. 2011.
            *   PCE: core 2000; goods, services, food and energy 2013; ex food, energy and
                housing 2023.
            *   PPI final demand splits: 2014-15.
        *   They are fetched and stored like releases (`all_series_to_fetch`), but are NOT
            nowcast inputs (`series_to_fetch` leaves them out).
    *   **The bulk snapshots** (section 2): every CPI/PPI/PCE series, point-in-time from
        Sep 2026 only.
    *   **CPI weights** (section 3): every year since 1987, point-in-time by construction.
*   **Not on ALFRED at all** (checked 2026-09-30), so their point-in-time history starts
    with the snapshots: CPI motor vehicle insurance; PCE health care, financial services
    and insurance, housing and utilities.
*   **`CUSR0000SASL2RS` is NOT the market's "supercore".** It still holds energy services
    (Jan 2024: 0.63% first print, against the ~0.85% "core services less shelter" quoted
    then). Supercore = `CUSR0000SASLE` less `CUSR0000SAH1`, recombined with the CPI weights.
*   **NSA CPI was deliberately NOT reconstructed** (user decision 2026-09-30). NSA CPI is
    never revised, so today's values stamped with each month's release day would be a
    faithful history, but it would be reconstructed, not observed.

## 5. Backtesting rule: always point-in-time
*   **Anything evaluated historically uses values AS KNOWN THEN**: `snapshot(raw, as_of)` at
    each decision time, never the latest revised history. Revisions are large at the
    component level (core PCE Jun 2024 MoM: 0.182% first print, 0.213% today), and a
    backtest on revised data uses information nobody had.
*   **Compute changes WITHIN one vintage, never across vintages.** Comprehensive NIPA
    revisions REBASE the PCE indexes (e.g. 2012=100 → 2017=100). A level from one vintage
    divided by a level from another is meaningless. `derive_units` works on one snapshot
    for this reason.
*   **Pick the evaluation target deliberately:**
    *   A signal trading the print: the FIRST print (what the market saw), with surprise
        measured against a forecast. There is no consensus feed.
    *   A nowcast's accuracy: first release, or a fixed later vintage. Say which.
*   **Before a series' first vintage there is no point-in-time data.** Only revised history
    exists there, so start the backtest at the first vintage, or accept the look-ahead
    knowingly. NSA CPI is the exception, being never revised.
*   **Timing is day-granular.** A value published on D (08:30 New York) is usable from
    D's close; intraday reaction studies need the release time too.
