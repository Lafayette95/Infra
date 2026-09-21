"""Central configuration: paths, storage constants and the instrument universe.

Code lives in ~/Repos/Infra; the database lives separately under ~/Database.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- database paths
DATABASE_ROOT = Path(os.environ.get("INFRA_DATABASE_ROOT", "~/Database")).expanduser()
OHLCV_ROOT = DATABASE_ROOT / "ohlcv-1m"
FUTURES_DIR = OHLCV_ROOT / "Futures"
OPTIONS_DIR = OHLCV_ROOT / "Options"
COVERAGE_DIR = OHLCV_ROOT / "_coverage"  # which (key, date range) were already queried
DEFINITIONS_DIR = DATABASE_ROOT / "definitions"  # cached `definition` schema pulls

FUTURES_COVERAGE_FILE = COVERAGE_DIR / "futures.parquet"
OPTIONS_COVERAGE_FILE = COVERAGE_DIR / "options.parquet"

# ------------------------------------------------------------------ API settings
SCHEMA_OHLCV = "ohlcv-1m"
SCHEMA_DEFINITION = "definition"
API_KEY_ENV = "DATABENTO_API_KEY"

# Hard budget guardrail: a single API request whose estimated cost exceeds this
# raises instead of downloading. Override with INFRA_MAX_COST_USD.
MAX_COST_USD = float(os.environ.get("INFRA_MAX_COST_USD", "5.0"))

# --------------------------------------------------------------- storage settings
PRICE_SCALE = 10_000  # fixed-point multiplier for price columns
PARTITION_COLUMNS = ["year", "quarter"]  # never ticker / symbol / day
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 5
PARQUET_FILE_NAME = "part-0.parquet"

# ----------------------------------------------------------------------- universe
# Continuous front-month symbols only (Rule 2.2) -> dataset that serves them.
FUTURES_UNIVERSE: dict[str, str] = {
    "SR3.c.0": "GLBX.MDP3",  # 3-Month SOFR
    "ZN.c.0": "GLBX.MDP3",  # US 10-Year Note
    "GG.c.0": "XEUR.EOBI",  # Euro-Bund (verify root symbol via definitions)
}

# Option parent symbols (Rule 2.3): parent symbol -> dataset.
OPTIONS_UNIVERSE: dict[str, str] = {
    "OQ.OPT": "GLBX.MDP3",  # 3-Month SOFR options (verify root via definitions)
}


def dataset_for_ticker(ticker: str) -> str:
    """Return the Databento dataset for a configured continuous ticker."""
    try:
        return FUTURES_UNIVERSE[ticker]
    except KeyError as exc:
        raise KeyError(
            f"{ticker!r} is not in FUTURES_UNIVERSE; add it to infra/config.py "
            "or pass `dataset=` explicitly."
        ) from exc
