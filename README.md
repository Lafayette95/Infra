# Infra — STIR / rates data pipeline + Dash app

Databento (`ohlcv-1m`) → cleaned pandas → local hive-partitioned parquet → Plotly Dash.
Code lives here; data lives in `~/Database/ohlcv-1m/{Futures,Options}` (see `infra/config.py`).

## Layout
```
infra/
  config.py                  paths, constants, instrument universe
  api/databento_client.py    ONLY file that calls Databento (cost guard, continuous-symbol check)
  coverage/intervals.py      pure "which date ranges are missing" logic
  storage/parquet_store.py   parquet read (pyarrow pushdown) / write (year/quarter, ZSTD 5)
  storage/coverage_store.py  manifest of already-queried ranges (prevents re-charging)
  processing/transforms.py   clean / fixed-point encode / decode / MultiIndex (RAM only)
  processing/definitions.py  option definition normalise + local filter
  processing/resample.py     timeframe aggregation
  pipeline/futures.py        load_futures (ABSOLUTE contracts) + read / plan / fetch children
  pipeline/contracts.py      contracts table from periodic `definition` snapshots
  pipeline/relative.py       relative series (SR3.c.0 / SR3.v.0) resolved to absolute contracts
  pipeline/series.py         load_series / plan_series: one entry point for both kinds
  relative/                  pure roll logic: symbology, calendar + volume rolls, apply_mapping
  pipeline/options.py        load_options  (definitions -> filter -> ids -> bars)
  dashboard/                 app, layout, callbacks, charts, theme, assets/
scripts/                     update_futures.py, update_options.py, run_dashboard.py
tests/                       offline tests (no API, no cost)
```

## Run (always the infra-env interpreter)
```
PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
cp .env.example .env            # add DATABENTO_API_KEY
$PY scripts/update_futures.py --tickers SR3.c.0 SR3.v.0 --start 2025-01-01 --dry-run   # gaps + est. cost
$PY scripts/update_futures.py --tickers SR3.c.0 SR3.v.0 --start 2025-01-01             # fetch gaps only
$PY scripts/run_dashboard.py                                 # http://127.0.0.1:8050
$PY -m pytest
```

## Supported futures (`FUTURES_ROOTS` in `infra/config.py`)
| Category | Roots | Dataset |
|---|---|---|
| STIR | `SR3` (SOFR), `ESR` (€STR) | GLBX.MDP3 |
| STIR | `SO3` (SONIA) | IFLL.IMPACT |
| US Treasuries | `ZT`, `ZF`, `ZN`, `TN` (Ultra 10Y), `ZB`, `UB` (Ultra Bond) | GLBX.MDP3 |
| Eurex | `FGBL`, `FGBM`, `FGBS`, `FBTP` (data from 2025-03-10) | XEUR.EOBI |
| ICE | `R` (Long Gilt) | IFLL.IMPACT |

Relative tickers: `<root>.c.<n>` (calendar) / `<root>.v.<n>` (prior-day volume), e.g. `ZN.v.0`.
Prefer `.v.0` for bonds: the calendar front sits in an expiring, thin contract for weeks.

## Cost protection
- Disk first: only date ranges absent from the coverage manifest are requested.
- Every request is priced via `metadata.get_cost` and refused above `INFRA_MAX_COST_USD` (default $5).
- The API and database use ABSOLUTE contracts only (`SRZ4`); wildcards and relative tickers raise at the API boundary.
- Relative tickers (`SR3.c.0`, `SR3.v.0`) are resolved locally, so only the contracts actually needed are fetched.
- Relative series are unadjusted across rolls; roll day is the UTC calendar day.
- Options: `definition` → local filter → bars for the isolated instrument ids only.
- The dashboard never hits the API unless "Fetch missing" is ticked and Load is pressed.
