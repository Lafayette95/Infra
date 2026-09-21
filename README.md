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
  pipeline/futures.py        load_futures  (parent) + read / plan / fetch children
  pipeline/options.py        load_options  (definitions -> filter -> ids -> bars)
  dashboard/                 app, layout, callbacks, charts, theme, assets/
scripts/                     update_futures.py, update_options.py, run_dashboard.py
tests/                       offline tests (no API, no cost)
```

## Run (always the infra-env interpreter)
```
PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
cp .env.example .env            # add DATABENTO_API_KEY
$PY scripts/update_futures.py --start 2025-01-01 --dry-run   # gaps + estimated cost only
$PY scripts/update_futures.py --start 2025-01-01             # fetch gaps only
$PY scripts/run_dashboard.py                                 # http://127.0.0.1:8050
$PY -m pytest
```

## Cost protection
- Disk first: only date ranges absent from the coverage manifest are requested.
- Every request is priced via `metadata.get_cost` and refused above `INFRA_MAX_COST_USD` (default $5).
- Futures accept continuous symbols only (`SR3.c.0`); wildcards raise.
- Options: `definition` → local filter → bars for the isolated instrument ids only.
- The dashboard never hits the API unless "Fetch missing" is ticked and Load is pressed.
