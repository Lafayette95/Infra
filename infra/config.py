"""Central configuration: paths, storage constants and the instrument universe.

Code lives in ~/Repos/Infra; the database lives separately under ~/Database.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
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
FUTURES_DEFS_COVERAGE_FILE = COVERAGE_DIR / "futures_definitions.parquet"
OPTIONS_COVERAGE_FILE = COVERAGE_DIR / "options.parquet"

# Master table of absolute futures contracts (root, ticker, expiry, ...).
FUTURES_CONTRACTS_FILE = DEFINITIONS_DIR / "Futures" / "contracts.parquet"

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
# The database stores ABSOLUTE contracts only (raw symbols such as ``SRZ4``).
# "Relative" tickers (``SR3.c.0``, ``SR3.v.1``) are resolved locally from the
# contracts table - see infra/relative.
@dataclass(frozen=True)
class FuturesRoot:
    root: str  # product code, e.g. "SR3"
    dataset: str  # Databento dataset serving it
    parent: str  # parent symbol used to list its contracts
    name: str  # human label (dashboard)
    category: str  # "STIR" | "Bonds"
    expiry_months: tuple[int, ...] = (3, 6, 9, 12)  # cycle ranked by relative tickers
    roll_offset_days: int = 0  # calendar roll this many days before expiry
    # Keep only outrights whose raw symbol matches (drops non-trading twins, serial months).
    ticker_regex: str | None = None


_CME, _EUREX, _ICE = "GLBX.MDP3", "XEUR.EOBI", "IFLL.IMPACT"
# ICE symbols look like "R   FMH0025!": keep quarterly (H/M/U/Z) "!" contracts only.
_ICE_QUARTERLY = r"FM[HMUZ]\d{4}!$"

FUTURES_ROOTS: dict[str, FuturesRoot] = {
    r.root: r
    for r in (
        # ---- STIR
        FuturesRoot("SR3", _CME, "SR3.FUT", "3M SOFR", "STIR"),
        FuturesRoot("ESR", _CME, "ESR.FUT", "3M €STR", "STIR"),
        FuturesRoot("SO3", _ICE, "SO3.FUT", "3M SONIA", "STIR", ticker_regex=_ICE_QUARTERLY),
        # ---- Bonds: US Treasuries (CBOT, via GLBX.MDP3)
        FuturesRoot("ZT", _CME, "ZT.FUT", "US 2Y Note", "Bonds"),
        FuturesRoot("ZF", _CME, "ZF.FUT", "US 5Y Note", "Bonds"),
        FuturesRoot("ZN", _CME, "ZN.FUT", "US 10Y Note", "Bonds"),
        FuturesRoot("TN", _CME, "TN.FUT", "US Ultra 10Y Note", "Bonds"),
        FuturesRoot("ZB", _CME, "ZB.FUT", "US Classic Bond", "Bonds"),
        FuturesRoot("UB", _CME, "UB.FUT", "US Ultra Bond", "Bonds"),
        # ---- Bonds: Eurex (XEUR.EOBI only has data from 2025-03-10)
        FuturesRoot("FGBL", _EUREX, "FGBL.FUT", "Euro-Bund", "Bonds"),
        FuturesRoot("FGBM", _EUREX, "FGBM.FUT", "Euro-Bobl", "Bonds"),
        FuturesRoot("FGBS", _EUREX, "FGBS.FUT", "Euro-Schatz", "Bonds"),
        FuturesRoot("FBTP", _EUREX, "FBTP.FUT", "Euro-BTP", "Bonds"),
        # ---- Bonds: ICE Futures Europe (data from 2018-12-23)
        FuturesRoot("R", _ICE, "R.FUT", "UK Long Gilt", "Bonds", ticker_regex=_ICE_QUARTERLY),
    )
}

# Relative (kind, rank) pairs offered per root (dashboard Expiry dropdown, defaults).
DEFAULT_RELATIVE_RANKS: tuple[tuple[str, int], ...] = (
    ("c", 0), ("c", 1), ("c", 2), ("c", 3), ("v", 0), ("v", 1),
)
DEFAULT_RELATIVE_TICKERS: list[str] = [
    f"{root}.{kind}.{rank}" for root in FUTURES_ROOTS for kind, rank in DEFAULT_RELATIVE_RANKS
]

DEFINITION_SNAPSHOT_DAYS = 30  # definition snapshot cadence used to discover contracts
VOLUME_EXTRA_CANDIDATES = 3  # calendar ranks beyond the requested v.N ranked by volume

# Option parent symbols (Rule 2.3): parent symbol -> dataset.
OPTIONS_UNIVERSE: dict[str, str] = {
    "OQ.OPT": "GLBX.MDP3",  # 3-Month SOFR options (verify root via definitions)
}
