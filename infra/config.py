"""Central configuration: paths, storage constants and the instrument universe.

Code lives in ~/Repos/Infra; the database lives separately under ~/Database.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
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

# Top-of-book QUOTES sampled every minute (Databento ``bbo-1m``) - a sibling of ohlcv-1m,
# not nested in it: a different schema (bid/ask, not trade bars) with its own coverage.
# Unlike trade bars, a quote exists every minute whether or not the contract traded, so
# thin contracts still have a current price (intraday WIRP, infra.pipeline.wirp).
BBO_1M_ROOT = DATABASE_ROOT / "bbo-1m"
BBO_FUTURES_DIR = BBO_1M_ROOT / "Futures"
BBO_FUTURES_COVERAGE_FILE = BBO_1M_ROOT / "_coverage" / "futures.parquet"
# 1-SECOND trade bars and quotes, fetched ON DEMAND only for small event windows
# (infra.pipeline.ohlcv_1s / infra.pipeline.bbo) - never by a cycle. Cached like
# everything else (Rule 2.1): a window is only ever paid for once.
OHLCV_1S_ROOT = DATABASE_ROOT / "ohlcv-1s"
OHLCV_1S_FUTURES_DIR = OHLCV_1S_ROOT / "Futures"
OHLCV_1S_FUTURES_COVERAGE_FILE = OHLCV_1S_ROOT / "_coverage" / "futures.parquet"
BBO_1S_ROOT = DATABASE_ROOT / "bbo-1s"
BBO_1S_FUTURES_DIR = BBO_1S_ROOT / "Futures"
BBO_1S_FUTURES_COVERAGE_FILE = BBO_1S_ROOT / "_coverage" / "futures.parquet"

# Master table of absolute futures contracts (root, ticker, expiry, ...).
FUTURES_CONTRACTS_FILE = DEFINITIONS_DIR / "Futures" / "contracts.parquet"

# Daily settlement price / open interest - deliberately a SEPARATE root from ohlcv-1m
# (different cadence, different pipeline shape; see infra/pipeline/daily.py and
# CLAUDE.md section 8), reusing the same generic parquet/coverage primitives.
DAILY_ROOT = DATABASE_ROOT / "Daily"
DAILY_FUTURES_DIR = DAILY_ROOT / "Futures"
DAILY_COVERAGE_DIR = DAILY_ROOT / "_coverage"
DAILY_FUTURES_COVERAGE_FILE = DAILY_COVERAGE_DIR / "futures.parquet"
DAILY_OPTIONS_DIR = DAILY_ROOT / "Options"
DAILY_OPTIONS_COVERAGE_FILE = DAILY_COVERAGE_DIR / "options.parquet"
# Daily cash-bond par yields (constant-maturity curves from official sources, not
# Databento - see BOND_CURVES below and infra/pipeline/bonds.py). Coverage is keyed by
# CURVE ("US", "UK"), since one request always returns the whole curve.
DAILY_BONDS_DIR = DAILY_ROOT / "Bonds"
DAILY_BONDS_COVERAGE_FILE = DAILY_COVERAGE_DIR / "bonds.parquet"

# Daily-cycle outputs (infra/cycle, CLAUDE.md section 12). Derived metrics ARE persisted
# here (unlike the dashboard's on-demand analytics) because the cycle's revision checks
# need yesterday's values on disk to compare against.
DERIVED_ROOT = DATABASE_ROOT / "Derived"
WIRP_DIR = DERIVED_ROOT / "WIRP"
# Intraday WIRP: the same schedule, recomputed on a fixed UTC grid from bbo-1m quote mids
# (infra.pipeline.wirp.intraday_schedules). One row set per grid time.
WIRP_INTRADAY_DIR = DERIVED_ROOT / "WIRP_intraday"
WIRP_INTRADAY_GRID = "15min"
# Ad-hoc 1-SECOND WIRP for small event windows (infra.pipeline.wirp.store_wirp_1s) - its
# own store, never the 15-minute one (whose backfill replaces whole days).
WIRP_1S_DIR = DERIVED_ROOT / "WIRP_1s"
# Benchmark swap closes snapped from DTCC trades (SWAP_CLOSES; CLAUDE.md 16) - one row per
# (snap instant, close, currency, tenor, method).
SWAP_CLOSES_DIR = DERIVED_ROOT / "SwapCloses"
BMK_ROOT = DATABASE_ROOT / "Bmk"
# Full dated snapshots of the database, one per cycle run day (``_vintages/YYYY-MM-DD``) -
# the baseline each run's "no revisions" check compares against.
VINTAGE_ROOT = DATABASE_ROOT / "_vintages"
# Only the newest vintages are kept (by count, not calendar days - the cycle skips
# weekends): today's and the one before, which is all the revision check ever compares
# against. Older ones are deleted after each successful backup (user decision 2026-09-30).
VINTAGES_KEPT = 2
# Sidecar log of every value the pipeline TOUCHED (a bad print NA'd or rolled, ...): the
# data stores keep exactly what the vendor delivered; readers overlay this at read time.
# infra.storage.adjustment_store, CLAUDE.md section 12.
ADJUSTMENTS_DIR = DATABASE_ROOT / "_adjustments"
# Reference data (the daily cycle's ``ref`` step, first): what instruments exist and their
# fixed characteristics - not prices. CLAUDE.md section 18.
REFERENCE_ROOT = DATABASE_ROOT / "Reference"
TREASURY_REF_DIR = REFERENCE_ROOT / "Treasuries"
# One row per CUSIP: its static characteristics, derived from the raw auctions store
# (TSY_AUCTIONS_DIR) - no amounts: those grow with reopenings and live, point in time,
# in the auctions.
TREASURY_SECURITIES_DIR = TREASURY_REF_DIR / "Securities"
# Security types kept. TIPS and floating-rate notes are excluded for now (user decision
# 2026-10-01): FiscalData marks them by flag, not type (inflation_index_security /
# floating_rate), so they are dropped by those flags.
TREASURY_TYPES = ("Bill", "Note", "Bond")
# On/off-the-run map (Reference/Treasuries/OTR): per business day and tenor, rank 0 = the
# on-the-run issue, 1 = 1-old, ... down to TREASURY_OTR_DEPTH, original issues only (a
# reopening of the 10y/20y/30y is the same series), not yet matured. Two switch
# conventions are stored: "issue" (DEFAULT: a new issue counts from its issue date - it
# has no price before, user decision 2026-10-01) and "auction" (market convention: from
# the auction, when-issued). COUPONS only for now: a 4/13-week bill is a REOPENING of an
# older 26/52-week bill, so bills need the auction-level terms (TOFIX.md).
TREASURY_OTR_DIR = TREASURY_REF_DIR / "OTR"
TREASURY_OTR_TENORS: dict[str, tuple[str, str]] = {
    "2y": ("Note", "2-Year"), "3y": ("Note", "3-Year"), "5y": ("Note", "5-Year"), "7y": ("Note", "7-Year"),
    "10y": ("Note", "10-Year"), "20y": ("Bond", "20-Year"), "30y": ("Bond", "30-Year"),
}
TREASURY_OTR_DEPTH = 5
TREASURY_OTR_CONVENTIONS = ("issue", "auction")
# Where the Treasury history (OTR map, per-CUSIP prices) starts; FedInvest has 2008-09-02
# on - earlier days: TOFIX.md.
TREASURY_PRICES_START = "2016-01-04"
TREASURY_OTR_DEFAULT_CONVENTION = "issue"
# Treasury futures delivery baskets with per-contract conversion factors
# (Reference/Treasuries/FuturesBaskets, keys timestamp / root / contract / cusip). From
# CME's own daily files where they exist - ftp.cmegroup.com/settle/TCF/TCF_YYYYMMDD.csv,
# every deliverable CUSIP for the next three contract months of each Treasury future, from
# 2023-12-09 (verified 2026-10-01; archived raw in RawData/CME_TCF) - and computed from the
# eligibility rules and CME's conversion-factor formula before that.
CME_TCF_DIR = DATABASE_ROOT / "RawData" / "CME_TCF"  # under RAW_DATA_ROOT (defined below)
TREASURY_BASKETS_DIR = TREASURY_REF_DIR / "FuturesBaskets"
# CME's product code in the TCF files -> our futures root (FUTURES_ROOTS where we trade it;
# Z3N - 3y note - and TWE - 20y bond - are kept too, they're in the files anyway).
# Deliverable-grade rules, for computing baskets where CME's files don't reach (CLAUDE.md
# 17). Remaining term is measured from the FIRST day of the delivery month (the max from
# its LAST day where ``max_from_last_day``); ``cf_months`` = 1 (whole months) or 3 (whole
# quarters) is how CME rounds that term in the conversion factor. Checked against every
# CME file 2023-12..2026-10; whether they held unchanged before 2023-12 is NOT verified
# (TOFIX.md).
@dataclass(frozen=True)
class BasketRule:
    min_remaining_months: int
    max_remaining_months: int | None = None
    max_from_last_day: bool = False
    max_inclusive: bool = True
    max_original_months: int | None = None
    original_months: int | None = None  # exact original term required (TN: 10y notes only)
    cf_months: int = 3


TREASURY_BASKET_RULES: dict[str, BasketRule] = {
    "ZT": BasketRule(21, 24, max_from_last_day=True, max_original_months=63, cf_months=1),
    "Z3N": BasketRule(33, 36, max_from_last_day=True, max_original_months=84, cf_months=1),
    "ZF": BasketRule(50, max_original_months=63, cf_months=1),
    "ZN": BasketRule(78, 96, max_original_months=120),
    "TN": BasketRule(113, 120, original_months=120),
    "ZB": BasketRule(180, 300, max_inclusive=False),
    "UB": BasketRule(300),
    "TWE": BasketRule(230, 239),
}
# Computed baskets cover these roots only, from their first listed day: TN launched
# 2016-01-11; Z3N (relaunched 2021) and TWE start dates aren't verified, so they get no
# computed history (CME's files cover them from 2023-12).
TREASURY_BASKET_HISTORY: dict[str, str] = {"ZT": "2016-01-04", "ZF": "2016-01-04", "ZN": "2016-01-04",
                                           "TN": "2016-01-11", "ZB": "2016-01-04", "UB": "2016-01-04"}
CME_TCF_ROOTS: dict[str, str] = {"26": "ZT", "3YR": "Z3N", "25": "ZF", "21": "ZN", "TN": "TN", "17": "ZB",
                                 "TWE": "TWE", "UBE": "UB"}
# Non-price raw inputs (the daily cycle's ``raw`` step, 1b). Macro releases are stored as
# every published VINTAGE of each source series (a release date per value), not just the
# latest revised history - see MACRO_RELEASES below and infra/pipeline/releases.py.
RAW_DATA_ROOT = DATABASE_ROOT / "RawData"
RELEASES_DIR = RAW_DATA_ROOT / "Releases"
RELEASES_COVERAGE_FILE = RAW_DATA_ROOT / "_coverage" / "releases.parquet"
# Economic-calendar rows (actual / consensus / previous per release, as the calendar page
# showed them), harvested from archived copies of MarketWatch's U.S. economic calendar -
# the free source for releases with no official free history (ISM, S&P Global PMIs, ...).
# See CALENDAR_PAGES and infra/pipeline/econ_calendar.py. Coverage is keyed by page and
# on the CAPTURE-day axis ("every archived capture of these days was processed").
CALENDAR_DIR = RAW_DATA_ROOT / "EconCalendar"
CALENDAR_COVERAGE_FILE = RAW_DATA_ROOT / "_coverage" / "econ_calendar.parquet"
# WHEN scheduled events happen (infra.reference.events says WHAT they are): one row per
# (event, release instant, source), point-in-time via known_from / last_seen - see
# infra/pipeline/release_calendar.py.
RELEASE_CALENDAR_DIR = RAW_DATA_ROOT / "ReleaseCalendar"
# US Treasury auctions (Fiscal Data auctions_query): one row per auction, results once held;
# coverage keyed "auctions" on the auction-date axis, claimed only for HELD auctions.
TSY_AUCTIONS_DIR = RAW_DATA_ROOT / "TsyAuctions"
TSY_AUCTIONS_COVERAGE_FILE = RAW_DATA_ROOT / "_coverage" / "tsy_auctions.parquet"
# Auction TAILS (high yield - when-issued yield at the close), one row per (auction,
# source) - archived auction recaps (ZeroHedge, ForexLive) for history; more sources (a
# futures-based estimate) can be added side by side. Coverage keyed by source, on the
# article-URL axis (a processed URL is never fetched again).
TSY_TAILS_DIR = RAW_DATA_ROOT / "TsyAuctionTails"
TSY_TAILS_COVERAGE_FILE = RAW_DATA_ROOT / "_coverage" / "tsy_auction_tails.parquet"
# Full-granularity inflation detail (CPI / PPI / PCE component trees) snapshotted from the
# agencies' own bulk files - see BULK_DATASETS below and infra/pipeline/bulk_series.py.
# One store per source (``BulkDataset.store``) under RAW_DATA_ROOT; coverage keyed by dataset.
BULK_COVERAGE_FILE = RAW_DATA_ROOT / "_coverage" / "bulk_series.parquet"
BULK_CATALOG_DIR = RAW_DATA_ROOT / "_catalog"
# CPI relative importance (the item weights), one weight year = December Y, fetched once a
# year - infra/pipeline/cpi_weights.py. BLS publishes them from December 1987 on.
CPI_WEIGHTS_DIR = RAW_DATA_ROOT / "CPIWeights"
CPI_WEIGHTS_COVERAGE_FILE = RAW_DATA_ROOT / "_coverage" / "cpi_weights.parquet"
CPI_WEIGHTS_FIRST_YEAR = 1987
# DTCC public price dissemination (CFTC Part 43 swap trades): each day's CUMULATIVE report
# zip, archived byte-for-byte - infra/pipeline/dtcc.py. DTCC itself keeps only a rolling ~2
# years (first file still published when the archive started: 2024-09-30, verified
# 2026-10-01), so a day not archived within that window is lost for good.
DTCC_DIR = RAW_DATA_ROOT / "DTCC"
DTCC_REPORTS = ("RATES",)  # CFTC cumulative report kinds archived (also: CREDITS, FOREX, ...)
DTCC_FIRST_DAY = "2024-09-30"  # never request earlier days: DTCC no longer has them
DTCC_RETENTION_DAYS = 700  # look back this far for unarchived days (inside DTCC's ~730)

# ------------------------------------------------------------------ API settings
SCHEMA_OHLCV = "ohlcv-1m"
SCHEMA_OHLCV_1S = "ohlcv-1s"
SCHEMA_BBO_1M = "bbo-1m"
SCHEMA_BBO_1S = "bbo-1s"
SCHEMA_DEFINITION = "definition"
SCHEMA_STATISTICS = "statistics"
API_KEY_ENV = "DATABENTO_API_KEY"
FRED_API_KEY_ENV = "FRED_API_KEY"  # free key, fred.stlouisfed.org -> My Account -> API Keys
# download.bls.gov answers 403 unless the User-Agent carries a contact email (verified
# 2026-09-30: browser, generic and python-urllib agents are all refused).
BLS_CONTACT_ENV = "BLS_CONTACT_EMAIL"

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
    # ``.v.N`` ranks the N+1+this nearest contracts by volume; None = VOLUME_EXTRA_CANDIDATES.
    # Bond futures: 1, i.e. v.0 picks between the front two only - volume never leads
    # further out, and every candidate is a contract whose volume must be fetched.
    volume_extra_candidates: int | None = None
    # ``.v.N`` averages this many prior trading days' volume; None = VOLUME_LOOKBACK_DAYS.
    # Bond futures: 2 - they roll once a quarter, sharply and one way, so a 5-day average
    # only crosses ~4 sessions after the daily volume does (ZN Nov 2025: volume crossed on
    # 11-25, 3 days before first notice; the 5-day average only on 12-01, when the front
    # traded 35k against 1.16m), leaving v.0 in a dying contract.
    volume_lookback_days: int | None = None
    # Keep only outrights whose raw symbol matches (drops non-trading twins, serial months).
    ticker_regex: str | None = None
    # Currency value of a 1.00 move in the quoted price, per contract (daily cycle pnl /
    # DV01, CLAUDE.md 12). None = not yet verified against the exchange's contract specs.
    point_value: float | None = None
    currency: str | None = None


_CME, _EUREX, _ICE = "GLBX.MDP3", "XEUR.EOBI", "IFLL.IMPACT"
_FRONT_TWO = 1  # volume_extra_candidates for bond futures: v.0 is one of the front two
_BOND_LOOKBACK = 2  # volume_lookback_days for bond futures (see FuturesRoot)
# Point values verified 2026-09-28 against the exchanges' own contract specs:
#   SR3 $2,500 x IMM index ($25/bp) - cmegroup.com .../three-month-sofr.contractSpecs.html
#   SR1, ZQ $4,167 x index ($41.67/bp) - .../one-month-sofr and .../30-day-federal-fund specs
#   ESR EUR2,500 x index (EUR25/bp) - .../euro-short-term-rate.contractSpecs.html
#   ZT $200,000 face = $2,000/pt; ZF/ZN/TN/ZB/UB $100,000 face = $1,000/pt - CME Treasury specs
#   FGBL/FGBM/FGBS/FBTP EUR100,000 nominal = EUR1,000/pt (0.01 = EUR10) - eurex.com product pages
# ICE (SO3, R) deliberately left unset: disabled in the daily cycle, not yet verified.
# ICE symbols look like "R   FMH0025!": keep quarterly (H/M/U/Z) "!" contracts only.
_ICE_QUARTERLY = r"FM[HMUZ]\d{4}!$"

FUTURES_ROOTS: dict[str, FuturesRoot] = {
    r.root: r
    for r in (
        # ---- STIR
        FuturesRoot("SR3", _CME, "SR3.FUT", "3M SOFR", "STIR", point_value=2500.0, currency="USD"),
        FuturesRoot("ESR", _CME, "ESR.FUT", "3M €STR", "STIR", point_value=2500.0, currency="EUR"),
        FuturesRoot("SO3", _ICE, "SO3.FUT", "3M SONIA", "STIR", ticker_regex=_ICE_QUARTERLY),
        # Both MONTHLY cycle (all 12 months), not quarterly like the rest of this
        # universe - verified against the real API 2026-09-21 (SR1: 26 outrights, ZQ:
        # 61 outrights, both with every calendar month present).
        FuturesRoot("SR1", _CME, "SR1.FUT", "1M SOFR", "STIR",
                   expiry_months=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), point_value=4167.0, currency="USD"),
        FuturesRoot("ZQ", _CME, "ZQ.FUT", "30-Day Fed Funds", "STIR",
                   expiry_months=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), point_value=4167.0, currency="USD"),
        # ---- Bonds: US Treasuries (CBOT, via GLBX.MDP3)
        FuturesRoot("ZT", _CME, "ZT.FUT", "US 2Y Note", "Bonds", point_value=2000.0, currency="USD",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        FuturesRoot("ZF", _CME, "ZF.FUT", "US 5Y Note", "Bonds", point_value=1000.0, currency="USD",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        FuturesRoot("ZN", _CME, "ZN.FUT", "US 10Y Note", "Bonds", point_value=1000.0, currency="USD",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        FuturesRoot("TN", _CME, "TN.FUT", "US Ultra 10Y Note", "Bonds", point_value=1000.0, currency="USD",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        FuturesRoot("ZB", _CME, "ZB.FUT", "US Classic Bond", "Bonds", point_value=1000.0, currency="USD",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        FuturesRoot("UB", _CME, "UB.FUT", "US Ultra Bond", "Bonds", point_value=1000.0, currency="USD",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        # ---- Bonds: Eurex (XEUR.EOBI only has data from 2025-03-10)
        FuturesRoot("FGBL", _EUREX, "FGBL.FUT", "Euro-Bund", "Bonds", point_value=1000.0, currency="EUR",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        FuturesRoot("FGBM", _EUREX, "FGBM.FUT", "Euro-Bobl", "Bonds", point_value=1000.0, currency="EUR",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        FuturesRoot("FGBS", _EUREX, "FGBS.FUT", "Euro-Schatz", "Bonds", point_value=1000.0, currency="EUR",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        FuturesRoot("FBTP", _EUREX, "FBTP.FUT", "Euro-BTP", "Bonds", point_value=1000.0, currency="EUR",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        # ---- Bonds: ICE Futures Europe (data from 2018-12-23)
        FuturesRoot("R", _ICE, "R.FUT", "UK Long Gilt", "Bonds", ticker_regex=_ICE_QUARTERLY,
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
    )
}

# ------------------------------------------------------------------ trading calendar
# What "trading day" means per exchange - the source of truth for infra.trading_calendar
# (used by infra.relative for roll-day/volume-ranking assignment and by
# infra.processing.resample for the "1D" bucket; CLAUDE.md section 6e). This is a DATA
# concept, distinct from the DISPLAY timezone a dashboard viewer picks (section 7).
@dataclass(frozen=True)
class TradingSession:
    """One exchange's trading-day definition. Verify against ``source`` if this drifts.

    ``crosses_midnight=False`` (Eurex, ICE Futures Europe): the session sits inside one
    local calendar day, so the trading day is simply that local date; ``open_time`` /
    ``close_time`` are informational only.
    ``crosses_midnight=True`` (CME Globex/CBOT): the session opens in the evening of one
    local calendar day and closes the afternoon of the next. The exchange's own "trade
    date" convention labels the WHOLE session by the day it closes - e.g. the session
    opening Sunday 17:00 CT and closing Monday 16:00 CT is trade date Monday.
    """
    exchange: str
    timezone: str  # IANA zone open_time/close_time are expressed in
    open_time: str  # "HH:MM" local session open
    close_time: str  # "HH:MM" local session close; the trading-day boundary
    crosses_midnight: bool
    source: str  # verified against, on the date below - re-check if hours change


TRADING_HOURS: dict[str, TradingSession] = {
    # Sun 17:00 CT -> Fri 16:00 CT, Mon-Thu maintenance halt 16:00-17:00 CT; a session
    # is labelled by the day it closes (CME's own trade-date convention). Same platform
    # hours for STIR (SR3, ESR) and CBOT Treasuries (ZT/ZF/ZN/TN/ZB/UB) alike.
    # Verified 2026-09-21: https://www.cmegroup.com/trading-hours.html and
    # https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/1184202754/Globex+Trade+Date+Behavior+Change
    "GLBX.MDP3": TradingSession(
        exchange="CME Globex / CBOT", timezone="America/Chicago",
        open_time="17:00", close_time="16:00", crosses_midnight=True,
        source="https://www.cmegroup.com/trading-hours.html",
    ),
    # Continuous trading ~02:10-22:00 CET/CEST, entirely inside one calendar day - no
    # next-day label shift. Verified 2026-09-21:
    # https://www.eurex.com/ex-en/markets/int/long-term-interest-rates/fix/government-bonds/Euro-Bund-Futures-137298
    "XEUR.EOBI": TradingSession(
        exchange="Eurex", timezone="Europe/Berlin",
        open_time="02:10", close_time="22:00", crosses_midnight=False,
        source="https://www.eurex.com/ex-en/markets/int/long-term-interest-rates/fix/government-bonds/Euro-Bund-Futures-137298",
    ),
    # 01:00-21:00 London time, entirely inside one calendar day - no next-day label
    # shift. Effective 2026-07-06 (aligned with European contract hours); verify again
    # if it predates that change. Verified 2026-09-21:
    # https://www.newsquawk.com/headlines/from-6th-july-2026-ice-will-be-aligning-the-trading-hours-of-uk-fistir-contracts-with-those-of-european-contracts-as-such-gilt-and-sonia-futures-will-open-at-1am-london-time-and-close-at-9pm
    "IFLL.IMPACT": TradingSession(
        exchange="ICE Futures Europe", timezone="Europe/London",
        open_time="01:00", close_time="21:00", crosses_midnight=False,
        source="https://www.newsquawk.com/headlines/from-6th-july-2026-ice-will-be-aligning-the-trading-hours-of-uk-fistir-contracts-with-those-of-european-contracts-as-such-gilt-and-sonia-futures-will-open-at-1am-london-time-and-close-at-9pm",
    ),
}


# ------------------------------------------------------------------ swap curves (DTCC)
# Which DTCC trades are a currency's plain OIS par swaps (CLAUDE.md 16), verified against
# the RATES files 2026-09-16..30: the product is identified by ``UPI FISN`` plus the
# floating index in ``UPI Underlier Name`` (``Product name`` is always empty).
@dataclass(frozen=True)
class SwapCurveSpec:
    fisn: str  # UPI FISN of the product
    underlier: str  # substring of ``UPI Underlier Name`` naming the floating index
    spot_lag_days: int  # business days from trade to the standard effective date
    tenors: tuple[int, ...]  # whole-year par tenors snapped
    fixed_frequencies: tuple[str, ...] = ("YEAR", "EXPI")  # annual; EXPI = one payment (<=1y)


SWAP_CURVES: dict[str, SwapCurveSpec] = {
    "USD": SwapCurveSpec("NA/Swap OIS USD", "SOFR", 2, (1, 2, 3, 5, 7, 10, 15, 20, 30)),
    "EUR": SwapCurveSpec("NA/Swap OIS EUR", "EuroSTR", 2, (1, 2, 3, 5, 7, 10, 15, 20, 30)),
    "GBP": SwapCurveSpec("NA/Swap OIS GBP", "SONIA", 0, (1, 2, 3, 5, 7, 10, 15, 20, 30)),
}
# Spot-start tolerance: the effective date may land up to this many calendar days after
# trade + spot lag (holidays, which plain business days ignore); maturity may miss
# effective + N years by this many days (date rolling) and still count as tenor N.
SWAP_SPOT_TOLERANCE_DAYS = 2
SWAP_TENOR_TOLERANCE_DAYS = 4
# Off-market trades: a fixed coupon set by agreement (often round, e.g. 3.50%), with or
# without a reported upfront fee, can sit 70-170bp from the market (found 2026-10-01 in the
# first backfill: 2.50% at 30y against 4.19%, eight 1.83% prints at 5y against 3.57%).
# Those with a reported fee are dropped outright; the rest by distance from a reference:
# the median of same-tenor trades within +-SWAP_OFF_MARKET_WINDOW_HOURS (at least 3), else
# the day's same-tenor median (at least 3), else the neighbouring tenors' day medians,
# interpolated - with a wider limit for that last, coarser reference.
SWAP_OFF_MARKET_WINDOW_HOURS = 2.0
# Platforms whose prints are not a market price. "BILT" (bilateral, off-platform) is
# where large off-market coupons cluster - e.g. ~$23bn of 1y at round 3.30-3.49% against
# a 3.75% market (2026-03-30), big enough to BECOME the local median and beat the filter
# below. Measured over March 2026 against the on-platform median of the same tenor
# within +-1h (USD): BILT median 2.0bp off, 41% beyond 5bp, 7% beyond 20bp; every
# electronic platform (TWSF, BBSF, BMTF, TREU, ...) 0.1-0.3bp median, ~0% beyond 5bp;
# off-platform XOFF 0.3bp, kept.
SWAP_EXCLUDED_PLATFORMS: tuple[str, ...] = ("BILT",)
SWAP_OFF_MARKET_BP = 25.0
# Measured 2025-03..06 (USD, 26,039 prints): good prints sit up to 29.5bp from their
# neighbouring tenors' interpolated day medians (99.9th pct 23.7bp) - curve shape, so this
# coarse reference catches garbage (382%, 0%), not a lone print 15-20bp off.
SWAP_OFF_MARKET_CURVE_BP = 30.0
# A day's closes apply the corrections and cancellations published in the following
# files up to this many days later (99% of cancellations arrive within 33 days, 95%
# within a day; corrections arriving later are mostly fixes to much older trades -
# September 2026). The cycle should recompute at least this far back.
SWAP_CORRECTION_DAYS = 10


# ------------------------------------------------------------- swap benchmark closes
# The benchmark "closes" swap curves are snapped at from DTCC trades (CLAUDE.md section 16).
# Written in the venue's LOCAL time with its IANA zone - the one form that means the same
# thing across both DST regimes (London and New York switch on different dates) - and
# converted to a UTC instant per day the moment code reads it
# (infra.trading_calendar.snap_instants); everything downstream sees UTC only (CLAUDE.md 7).
@dataclass(frozen=True)
class SwapCloseWeighting:
    """How much each trade counts toward a snap: inverse of its expected squared error vs
    the snap's true rate, in bp^2 -

        var = trade_noise^2 + drift^2 * |dt| + (hedge_error * futures_move)^2

    ``dt`` = hours between the trade and the snap; ``futures_move`` = bp the hedge
    future(s) moved over that time (adjusted method only - the pure method has no hedge
    term). ``drift`` is what's left unhedged: the whole rate move for a pure snap, only the
    swap-vs-futures spread move for an adjusted one - so the same formula weights steeply
    by time in the first and gently in the second.
    """
    trade_noise_bp: float  # same-moment dispersion between prints
    pure_drift_bp_per_sqrt_hour: float  # rate drift, unhedged
    adjusted_drift_bp_per_sqrt_hour: float  # swap-vs-futures spread drift, after the hedge
    hedge_error: float  # relative error of the hedge ratio (0.1 = 10%)
    # A print further from the snap's first estimate than this many times the larger of
    # the window's robust spread and its own expected error (sqrt of ``variance``) is
    # dropped - an off-market or mis-reported trade.
    outlier_k: float = 4.0

    def variance(self, dt_hours, futures_move_bp=0.0, *, adjusted: bool):
        drift = self.adjusted_drift_bp_per_sqrt_hour if adjusted else self.pure_drift_bp_per_sqrt_hour
        hedge = self.hedge_error * futures_move_bp if adjusted else 0.0
        return self.trade_noise_bp ** 2 + drift ** 2 * abs(dt_hours) + hedge ** 2

    def weight(self, dt_hours, futures_move_bp=0.0, *, adjusted: bool):
        return 1.0 / self.variance(dt_hours, futures_move_bp, adjusted=adjusted)


@dataclass(frozen=True)
class SwapCloseSpec:
    """One benchmark close. ``local_time`` in ``timezone`` (config is the only non-UTC
    place besides the dashboard); ``matches`` says what the snap exists to compare with.

    Pure: trades within +-``pure_half_window_min`` of the snap, widened to
    +-``pure_fallback_half_window_min`` when fewer than ``pure_min_trades`` fall inside.
    Futures-adjusted: trades within +-``adjusted_half_window_min``, each moved to the snap
    by its hedge future's move."""
    local_time: str  # "HH:MM" in ``timezone``
    timezone: str  # IANA zone
    currencies: tuple[str, ...]  # swap currencies snapped at this close
    matches: str
    source: str  # where the reference time was verified
    # Calibrated 2026-10-01 (held-out test, see SWAP_CLOSE_WEIGHTING): pure accuracy is
    # flat from +-5 to +-60 min (MAE 0.32-0.34bp), so the window only buys coverage - +-15
    # found prints on 76% of snaps, +-30 on 91%, +-45 on 96%; adjusted error is flat from
    # +-45 to +-120 with +-90 lowest overall (RMSE 0.528bp).
    pure_half_window_min: int = 30
    pure_fallback_half_window_min: int = 60
    pure_min_trades: int = 3
    adjusted_half_window_min: int = 90


# CALIBRATED 2026-10-01 on 185,045 USD par trades (2024-09-30..2026-09-30):
# * variance model from 2.4m same-tenor trade pairs <= 3h apart (E[d^2] vs the gap):
#   noise 0.27bp (1-7y) / 0.39bp (10-30y) - the adjusted fit's intercept, the clean one;
#   unhedged drift 2.36 / 2.11 bp/sqrt(h) over gaps <= 30 min (the pure windows' range;
#   the pure curve is CONCAVE - steep in the first 15-30 min, flatter after); swap-vs-
#   futures drift 0.18 / 0.25 bp/sqrt(h); hedge error 0.115 / 0.141 (30y alone 0.195).
#   One all-tenor set is used: tenor differences are moderate and, below, irrelevant.
# * held-out test (3,923 snaps with prints within +-2 min of the snap as the truth): the
#   weights barely move the weighted median (current vs measured within ~0.01bp), and
#   outlier_k is irrelevant (2.5..6 or none, within 0.005bp - the upstream off-market
#   filters do that work; 4 kept as a safety net). The adjusted method beats the pure one
#   by ~1/3 (MAE 0.214 vs 0.322bp, p95 0.67 vs 1.0bp, like-for-like), near the truth's
#   own noise floor. Windows: see SwapCloseSpec.
SWAP_CLOSE_WEIGHTING = SwapCloseWeighting(trade_noise_bp=0.35, pure_drift_bp_per_sqrt_hour=2.2,
                                          adjusted_drift_bp_per_sqrt_hour=0.22, hedge_error=0.13)

# Futures-adjusted closes: each print is moved to the snap by its hedge future's move,
# rate_at_snap = rate + ratio x (mid at snap - mid at the print), ``ratio`` in bp of yield
# per point of price. The hedge is the root's ``.v.0`` contract (CLAUDE.md 5; bonds roll
# on a 2-day volume average) quoted by ``bbo-1m`` mids; ``ratio`` is the regression of its
# daily settlement changes (same contract both days) on the matching CMT par yield's, over
# the prior SWAP_HEDGE_RATIO_DAYS business days - known before the day it's used. Checked
# 2026-10-01 over 2024-09..2026-09: R2 0.90 (ZT) - 0.95 (ZN), ratios -58.6 (ZT) .. -6.0 (UB)
# bp/pt, in line with the contracts' DV01s.
# Tenor -> hedge root, by the root's cheapest-to-deliver maturity. 1-3y use ZT until SR3
# quotes exist (an SR3-strip hedge is the better short end). USD only: Bund (Eurex, quotes
# not stored) and gilt (ICE, disabled) futures aren't available as hedges yet.
SWAP_HEDGES: dict[str, dict[int, str]] = {
    "USD": {1: "ZT", 2: "ZT", 3: "ZT", 5: "ZF", 7: "ZN", 10: "TN", 15: "ZB", 20: "ZB", 30: "UB"},
}
SWAP_HEDGE_CMT: dict[str, str] = {"ZT": "US_BOND_2y", "ZF": "US_BOND_5y", "ZN": "US_BOND_7y",
                                  "TN": "US_BOND_10y", "ZB": "US_BOND_20y", "UB": "US_BOND_30y"}
SWAP_HEDGE_RATIO_DAYS = 60

SWAP_CLOSES: dict[str, SwapCloseSpec] = {
    # CME settles SR3 (13:59-14:00 CT) and Treasury futures (13:59:30-14:00 CT) on the
    # last minute(s) before 14:00 Chicago = 15:00 New York.
    "NY1500": SwapCloseSpec(
        "15:00", "America/New_York", ("USD",), "CME SR3 / Treasury futures settlements",
        source="https://cmegroupclientsite.atlassian.net/wiki/display/EPICSANDBOX/Treasuries (verified 2026-10-01)",
    ),
    # Treasury par yield curve (CMT): indicative bid-side quotes the NY Fed collects at or
    # near 3:30pm; matched by FedInvest END OF DAY prices to ~0.5bp (verified 2026-10-01).
    "NY1530": SwapCloseSpec(
        "15:30", "America/New_York", ("USD",), "Treasury CMT par curve",
        source="https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics",
    ),
    # Late US close, no official counterpart; DTCC USD prints thin out after 16:30.
    "NY1600": SwapCloseSpec(
        "16:00", "America/New_York", ("USD",), "US late close (no official benchmark)",
        source="DTCC print density, 2026-09-16..30",
    ),
    # Gilts: Tradeweb FTSE Gilt Closing Prices sample 16:14-16:16 London (the BoE yield
    # curves' input since 2017-07-24); ICE Long Gilt futures settle 16:13-16:15 London.
    # Believed (unverified) to also match Eurex Bund settlement, 17:15 Frankfurt.
    "LDN1615": SwapCloseSpec(
        "16:15", "Europe/London", ("USD", "EUR", "GBP"), "Gilt closing prices / BoE curves / Long Gilt futures",
        source="https://www.ice.com/publicdocs/futures/Designated_Settlement_Periods_Volume_Thresholds.pdf; "
               "Tradeweb FTSE Gilt Closing Prices calculation guide (verified 2026-10-01)",
    ),
}

# ------------------------------------------------------------------ daily cycle
# Operational parameters of the scheduled daily cycle (infra/cycle, CLAUDE.md section 12).
# Deliberately separate from FUTURES_ROOTS: that describes the instrument universe itself,
# this describes how the scheduled pipeline operates on it - a root can stay in the
# universe while being switched off here (ICE, for now).
@dataclass(frozen=True)
class DailyBackfillSpec:
    n_contracts: int  # nearest unexpired contracts on the root's expiry cycle (c.0 .. c.N-1)
    enabled: bool = True
    # What happens to a detected bad print, by the contract's rank that day: ranks below
    # strict_ranks are "NA" (strict - the value is dropped, never replaced), the rest "roll"
    # (the last good settlement is carried forward). CLAUDE.md section 12.
    strict_ranks: int = 0

    def bad_print_policy(self, rank: int) -> str:
        return "NA" if rank < self.strict_ranks else "roll"


DAILY_BACKFILL: dict[str, DailyBackfillSpec] = {
    # Bad-print policy (strict_ranks): STIR - the first HALF of the saved contracts
    # (n_contracts // 2) are strict "NA", the rest "roll"; bond futures - the front
    # contract strict, the deferred "roll".
    # STIR, monthly cycle: 12 = a full year of monthly expiries.
    "ZQ": DailyBackfillSpec(12, strict_ranks=6),
    "SR1": DailyBackfillSpec(12, strict_ranks=6),
    # STIR, quarterly cycle (serials excluded by FuturesRoot.expiry_months).
    "SR3": DailyBackfillSpec(21, strict_ranks=10),
    "ESR": DailyBackfillSpec(6, strict_ranks=3),
    # Bond futures: front + one deferred.
    "ZT": DailyBackfillSpec(2, strict_ranks=1),
    "ZF": DailyBackfillSpec(2, strict_ranks=1),
    "ZN": DailyBackfillSpec(2, strict_ranks=1),
    "TN": DailyBackfillSpec(2, strict_ranks=1),
    "ZB": DailyBackfillSpec(2, strict_ranks=1),
    "UB": DailyBackfillSpec(2, strict_ranks=1),
    "FGBL": DailyBackfillSpec(2, strict_ranks=1),
    "FGBM": DailyBackfillSpec(2, strict_ranks=1),
    "FGBS": DailyBackfillSpec(2, strict_ranks=1),
    "FBTP": DailyBackfillSpec(2, strict_ranks=1),
    # ICE excluded for now: ~99% of the daily cycle's API cost (2026-09-28 cost check).
    "SO3": DailyBackfillSpec(0, enabled=False),
    "R": DailyBackfillSpec(0, enabled=False),
}

# ------------------------------------------------------------------ cash-bond curves
# Daily constant-maturity PAR yields per sovereign, stored as absolute tickers
# ``<country>_BOND_<tenor>y`` (e.g. "US_BOND_10y"), in percent. Each curve has its own
# free official source (infra/api/<source>_client.py) - no Databento, no cost. EVERY
# curve is on one basis - par with SEMI-ANNUAL coupons (user decision 2026-09-30): the
# US as published (the Treasury's own basis), UK and DE derived so from their zero curves.
@dataclass(frozen=True)
class BondCurve:
    country: str  # ticker prefix, e.g. "US"
    source: str  # fetcher in infra.pipeline.bonds.SOURCES: "treasury" | "boe" | "bundesbank"
    name: str  # human label
    currency: str
    convention: str  # the yields' compounding/coupon convention, as the source publishes them
    tenors: tuple[int, ...] = (2, 3, 5, 7, 10, 20, 30)  # years
    history_start: str = "1990-01-02"  # never request before this (the source has nothing)
    # How par yields are obtained from the source, for a source that only publishes
    # another curve (the BoE: spot only) - a name in
    # infra.processing.bond_curves.PAR_METHODS, swappable without touching the fetcher.
    par_method: str | None = None
    enabled: bool = True
    # A confirmed bad print (the px step's peer-outlier rule, peers = neighbouring tenors)
    # is "NA" (dropped) or "roll" (last good value carried forward), CLAUDE.md 12.
    bad_print_policy: str = "NA"


def bond_ticker(country: str, tenor: int) -> str:
    return f"{country}_BOND_{tenor}y"


BOND_CURVES: dict[str, BondCurve] = {
    c.country: c
    for c in (
        # Daily Treasury Par Yield Curve Rates (the CMT curve): par yields on a
        # semi-annual bond-equivalent basis, 2 decimals. Verified 2026-09-30:
        # https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve
        BondCurve("US", "treasury", "US Treasury CMT par curve", "USD",
                  convention="par, semi-annual bond-equivalent (Treasury CMT)", history_start="1990-01-02"),
        # BoE nominal gilt curve (VRP spline): the BoE publishes SPOT (zero-coupon)
        # yields on a 0.5y grid to 40y, and par only for 5/10/20y (IADB IUDSNPY/IUDMNPY/
        # IUDLNPY) - so par is DERIVED here from the spot grid, one method for every
        # tenor (user decision 2026-09-30). It does not yet reproduce the BoE's own par
        # series exactly (TOFIX.md). Verified 2026-09-30:
        # https://www.bankofengland.co.uk/statistics/yield-curves
        BondCurve("UK", "boe", "UK gilt par curve (derived from BoE spot)", "GBP",
                  convention="par, semi-annual coupons, derived from BoE nominal spot curve",
                  history_start="2016-01-01", par_method="semiannual_from_spot"),
        # Bundesbank's daily SVENSSON PARAMETERS for German Federal securities (BBSIS
        # ZST B0..T2, 5 decimals, same day), evaluated here into the model's zero curve
        # (annually compounded, the Bundesbank's convention) and par derived from it with
        # SEMI-ANNUAL coupons, like every curve here (user decision 2026-09-30) - ~3bp
        # below the Bunds' native annual-coupon par at 2026 levels. With
        # ``annual_from_annual_spot`` it reproduces the Bundesbank's own published
        # (annual-coupon) par curve within +-0.5bp. Verified 2026-09-30, see
        # infra/api/bundesbank_client.py. Its API bot-challenges VPN addresses.
        BondCurve("DE", "bundesbank", "German Federal securities par curve (Bundesbank Svensson)", "EUR",
                  convention="par, semi-annual coupons, from the Bundesbank's Svensson parameters",
                  history_start="1997-08-01", par_method="semiannual_from_annual_spot"),
    )
}

# How many BUSINESS days back the scheduled run re-fetches (force_refetch=True) to catch
# upstream revisions. Global catch-all default; each cycle step can override its own.
DEFAULT_REVISION_WINDOW_DAYS = 3

# Relative (kind, rank) pairs offered per root (dashboard Expiry dropdown, defaults).
DEFAULT_RELATIVE_RANKS: tuple[tuple[str, int], ...] = (
    ("c", 0), ("c", 1), ("c", 2), ("c", 3), ("v", 0), ("v", 1),
)
DEFAULT_RELATIVE_TICKERS: list[str] = [
    f"{root}.{kind}.{rank}" for root in FUTURES_ROOTS for kind, rank in DEFAULT_RELATIVE_RANKS
]

DEFINITION_SNAPSHOT_DAYS = 30  # definition snapshot cadence used to discover contracts
VOLUME_EXTRA_CANDIDATES = 3  # calendar ranks beyond the requested v.N ranked by volume
# .v.N ranks by the trailing average of this many prior trading days' volume, not just
# the single prior day - a lone thin session (Sunday open, day before a holiday) can
# otherwise outrank a genuinely more liquid contract by a coin-flip margin and cause a
# one-day round-trip in the front contract. See infra/relative/rolls.py.
VOLUME_LOOKBACK_DAYS = 5

# Option parent symbols (Rule 2.3): parent symbol -> dataset.
OPTIONS_UNIVERSE: dict[str, str] = {
    # 3-Month SOFR options. `OQ.OPT` (the old assumed root) does not resolve; verified
    # against the real API 2026-09-21: SR3.OPT returns 3,358 real option contracts
    # (e.g. "SR3U6 C9762.5"), same convention as the futures root plus ".OPT".
    "SR3.OPT": "GLBX.MDP3",
}

# ------------------------------------------------------------------ FOMC meeting schedule
# Source of truth for infra.analytics.wirp (World Interest Rate Probability - implied
# Fed rate-move probabilities backed out of 30-Day Fed Funds futures, ZQ). A rate
# decision is conventionally effective the calendar day AFTER the meeting's last day
# (the day the FOMC statement is released) - ``end_date`` is that split day; see
# infra.analytics.wirp.solve_post_meeting_rate and CLAUDE.md section 11.
@dataclass(frozen=True)
class FOMCMeeting:
    start_date: str  # "YYYY-MM-DD", first day of the (usually 2-day) meeting
    end_date: str  # "YYYY-MM-DD", decision/announcement day
    has_projections: bool  # Summary of Economic Projections released alongside (informational)


# Verified against https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm,
# 2026-09-28. The Fed publishes next year's calendar around July of the prior year -
# re-check that page (and re-verify existing dates) when extending this list.
# 2025-08-22 was a "notation vote only" (no live 2-day meeting / rate decision in the
# normal sense) and is deliberately excluded - treating it as a regular meeting would
# corrupt the August 2025 month-to-meeting mapping.
FOMC_MEETINGS: tuple[FOMCMeeting, ...] = (
    FOMCMeeting("2025-01-28", "2025-01-29", False),
    FOMCMeeting("2025-03-18", "2025-03-19", True),
    FOMCMeeting("2025-05-06", "2025-05-07", False),
    FOMCMeeting("2025-06-17", "2025-06-18", True),
    FOMCMeeting("2025-07-29", "2025-07-30", False),
    FOMCMeeting("2025-09-16", "2025-09-17", True),
    FOMCMeeting("2025-10-28", "2025-10-29", False),
    FOMCMeeting("2025-12-09", "2025-12-10", True),
    FOMCMeeting("2026-01-27", "2026-01-28", False),
    FOMCMeeting("2026-03-17", "2026-03-18", True),
    FOMCMeeting("2026-04-28", "2026-04-29", False),
    FOMCMeeting("2026-06-16", "2026-06-17", True),
    FOMCMeeting("2026-07-28", "2026-07-29", False),
    FOMCMeeting("2026-09-15", "2026-09-16", True),
    FOMCMeeting("2026-10-27", "2026-10-28", False),
    FOMCMeeting("2026-12-08", "2026-12-09", True),
)

# ------------------------------------------------------------------ macro releases
# The nowcast's release universe (infra/models/nowcast), transcribed 2026-09-30 from the
# user's "Data Releases" table (OneNote, a photo read at full resolution - every cell
# legible, incl. the 130 vs 1305 ECDF half-lives). One row per release; the table's
# columns map 1:1 onto the fields below. The Bloomberg ticker is the KEY and a reference
# only: there is no Bloomberg data here - each release is rebuilt from a free official
# source series (``source`` + ``series_id``, which is what is STORED, as published,
# every vintage), then turned into the table's units by ``units``.
RELEASE_BLOCKS = (  # the table's subcategory columns, in its order
    "Activity_Housing", "Activity_Consumption", "Activity_Manufacturing", "Activity_OtherBusiness",
    "Activity_Trade", "Activity_Labor", "Activity_Gov",
    "Price_Wages", "Price_Commo", "Price_WholeSales", "Price_Retail", "Price_Expectation",
)


@dataclass(frozen=True)
class MacroRelease:
    ticker: str  # Bloomberg ticker (the table's key) - or our own id for a free proxy
    name: str
    frequency: str  # observation period: "Q" | "M" | "W"
    # Fetcher in infra.pipeline.releases.SOURCES; None = no free source (kept for the
    # record, skipped by the pipeline and the model).
    source: str | None
    series_id: str | None  # the source's own id - the stored raw ticker
    # Raw source series -> the table's units, computed per vintage (so a derived value's
    # release date is its inputs'): "level" as published | "diff" | "pct" (period % change)
    # | "saar" (period % change, annualized by compounding) | "yoy" (% change vs the same
    # period a year earlier).
    units: str
    # The table's staging flags: which scheduled estimates exist (Advance/Prelim/Final);
    # NoStage = one scheduled print, anything later is an ordinary revision.
    advanced: bool
    preliminary: bool
    final: bool
    short_history: bool  # the early stage(s) only exist for part of the history
    bbg_median: bool  # the table uses a Bloomberg consensus median (not available here)
    no_stage: bool
    sign: int  # +1: higher = stronger economy / higher prices; -1 the reverse (claims, unemployment)
    transform: str  # ";"-separated steps, infra.models.nowcast.transforms
    cat1: str  # "Activity" | "Price"
    cat2: str  # "Survey" | "Hard"
    blocks: tuple[str, ...]  # subset of RELEASE_BLOCKS
    proxy_for: tuple[str, ...] = ()  # a free proxy: the table rows it stands in for
    note: str = ""
    # source="calendar": regex matching this release's row name on the economic calendar
    # (after infra.processing.econ_calendar.normalize_report: lower case, "U.S."/"US"
    # dropped, punctuation to spaces). Names drift across the years - every observed
    # variant must match, and nothing else may (tests/test_econ_calendar.py).
    calendar_pattern: str | None = None
    # calendar value x calendar_scale = this release's own units (payrolls are quoted in
    # persons, the table's NFP change is in thousands -> 1e-3). For a FRED-sourced release
    # the calendar row is a CROSS-CHECK and the consensus (``consensus_mw``), never data.
    calendar_scale: float = 1.0

    @property
    def available(self) -> bool:
        return self.source is not None

    @property
    def stages(self) -> tuple[str, ...]:
        """Names of the scheduled estimates, in publication order ("first" when NoStage)."""
        named = tuple(n for n, on in (("advance", self.advanced), ("preliminary", self.preliminary),
                                      ("final", self.final)) if on)
        return named or ("first",)


_A, _P, _S, _H = "Activity", "Price", "Survey", "Hard"
_HOU, _CON, _MAN, _OB, _LAB = ("Activity_Housing", "Activity_Consumption", "Activity_Manufacturing",
                               "Activity_OtherBusiness", "Activity_Labor")
_E5 = "ecdf_no_demean_exp:1305"
_PMI = "deman_fix:50;ecdf_no_demean_exp:130"


def _r(ticker, name, freq, source, sid, units, adv, pre, fin, short, med, nostage, sign, transform,
       cat1, cat2, blocks, note="", calendar_pattern=None, calendar_scale=1.0):
    return MacroRelease(ticker, name, freq, source, sid, units, bool(adv), bool(pre), bool(fin), bool(short),
                        bool(med), bool(nostage), sign, transform, cat1, cat2, tuple(blocks), note=note,
                        calendar_pattern=calendar_pattern, calendar_scale=calendar_scale)


# Columns: ticker, name, freq, source, series id, units | Advanced, Preliminary, Final,
# ShortHistory, BbgMedian, NoStage | Sign, Transform | Cat1, Cat2, subcategories.
# FRED ids verified 2026-09-30 against FRED's own series metadata (scripts/update_releases.py
# --verify): title, frequency, units and seasonal adjustment all match the table's row.
_TABLE = (
    _r("GDP CQOQ Index", "US GDP chained 2012 dollars QoQ SAAR", "Q", "fred", "GDPC1", "saar",
       1, 1, 1, 0, 1, 0, 1, "rolling_ma:4;" + _E5, _A, _H, (_HOU, _CON, _MAN, _OB),
       note="the nowcast TARGET: enters the model in native units (QoQ % SAAR), not transformed. Built "
            "from the real GDP LEVEL (GDPC1, vintages since 1991-12) rather than BEA's published growth "
            "(A191RL1Q225SBEA, vintages only since 2014-09): verified 2026-09-30 to match the published "
            "growth within 0.05pp (its 1-decimal rounding) at every vintage checked"),
    _r("PCE CRCH Index", "US personal consumption expenditures chain price index", "M", "fred", "PCEPI", "pct",
       0, 0, 0, 0, 1, 1, 1, "rolling_ma:12;" + _E5, _P, _H, ("Price_Retail",),
       note="table name truncated ('US Personal Consumption Expend...'); read as the PCE price index "
            "MoM per the table's Price/Price_Retail categories - confirm"),
    _r("NFP TCH Index", "US employees on nonfarm payrolls, total change", "M", "fred", "PAYEMS", "diff",
       0, 0, 0, 0, 1, 1, 1, _E5, _A, _H, (_LAB,)),
    _r("USURTOT Index", "U-3 US unemployment rate", "M", "fred", "UNRATE", "level",
       0, 0, 0, 0, 1, 1, -1, "ecdf_exp:1305", _A, _H, (_LAB,)),
    _r("MWINCHNG Index", "Merchant wholesalers inventories MoM", "M", "fred", "I42IMSM144SCEN", "pct",
       0, 1, 1, 1, 1, 0, 1, "rolling_ma:3;" + _E5, _A, _H, (_CON,),
       note="from the Census Monthly Wholesale Trade release (FRED release 290), dated when the number is "
            "first published. WHLSLRIMSA (used until 2026-10-01) is the same total in the combined Manufacturing "
            "and Trade Inventories release, a median 6 days LATER - caught by the calendar cross-check"),
    _r("SBOITOTL Index", "NFIB small business optimism", "M", "calendar", "MW:SBOITOTL", "level",
       0, 0, 0, 0, 1, 1, 1, "deman_fix:100;" + _E5, _A, _S, (_OB,), note="NFIB: no official free history - archived MarketWatch calendar",
       calendar_pattern=r"^nfib\b"),
    _r("CPI YOY Index", "US CPI urban consumers YoY NSA", "M", "fred", "CPIAUCNS", "yoy",
       0, 0, 0, 0, 1, 1, 1, "ecdf_exp:1305", _P, _H, ("Price_Retail",)),
    _r("FDIUFDYO Index", "US PPI final demand YoY NSA", "M", "fred", "PPIFID", "yoy",
       0, 0, 0, 0, 1, 1, 1, _E5, _P, _H, ("Price_WholeSales",)),
    _r("INJCJC Index", "Initial jobless claims", "W", "fred", "ICSA", "level",
       0, 0, 0, 0, 1, 1, -1, "ecdf_exp:1305", _A, _H, (_LAB,)),
    _r("ADP CHNG Index", "ADP national employment report, change", "M", "fred", "ADPMNUSNERSA", "diff",
       0, 0, 0, 0, 1, 1, 1, _E5, _A, _H, (_LAB,)),
    _r("CONSSENT Index", "University of Michigan consumer sentiment", "M", "fred+prelims", "UMCSENT", "level",
       0, 1, 1, 0, 1, 0, 1, "ecdf_exp:1305", _A, _S, (_CON,),
       note="FRED (ALFRED) carries only the end-of-month FINAL; the mid-month PRELIMINARY comes from the "
            "archived economic calendar (infra.pipeline.releases.fred_with_calendar_prelims)"),
    _r("RSTAXAG% Index", "Adjusted retail sales less autos and gas MoM", "M", "fred", "MARTSSM44W72USS", "pct",
       0, 0, 0, 0, 1, 1, 1, "rolling_ma:3;" + _E5, _A, _H, (_CON,)),
    _r("IP CHNG Index", "US industrial production MoM", "M", "fred", "INDPRO", "pct",
       0, 0, 0, 0, 1, 1, 1, "rolling_ma:12;" + _E5, _A, _H, (_MAN,)),
    _r("NHSPSTOT Index", "US new privately owned housing starts", "M", "fred", "HOUST", "level",
       0, 0, 0, 0, 1, 1, 1, "ecdf_exp:1305", _A, _H, (_HOU,)),
    _r("OUTFGAF Index", "Philadelphia Fed business outlook, general activity", "M", "fred",
       "GACDFSA066MSFRBPHI", "level", 0, 0, 0, 0, 1, 1, 1, _E5, _A, _S, (_OB,)),
    _r("ETSLTOTL Index", "US existing home sales SAAR", "M", "calendar", "MW:ETSLTOTL", "level",
       0, 0, 0, 0, 1, 1, 1, "ecdf_exp:1305", _A, _H, (_HOU,),
       note="NAR: FRED keeps only 13 months - archived MarketWatch calendar (SAAR, units)",
       calendar_pattern=r"^existing home sales( \((annual rate|saar)\))?$"),
    _r("CFNAI Index", "Chicago Fed National Activity Index", "M", "fred", "CFNAI", "level",
       0, 0, 0, 0, 1, 1, 1, "rolling_ma:3;" + _E5, _A, _S, (_CON, _OB)),
    _r("NAPMPMI Index", "ISM manufacturing PMI", "M", "calendar", "MW:NAPMPMI", "level",
       0, 0, 0, 0, 1, 1, 1, _PMI, _A, _S, (_MAN,), note="ISM: proprietary, off FRED since 2016 - archived MarketWatch calendar",
       calendar_pattern=r"^ism( report on business)?( manufacturing)?( index| indext| pmi)?$"),
    _r("NAPMNMI Index", "ISM services PMI", "M", "calendar", "MW:NAPMNMI", "level",
       0, 0, 0, 0, 1, 1, 1, _PMI, _A, _S, (_OB,), note="ISM: proprietary - archived MarketWatch calendar",
       calendar_pattern=r"^ism( report on business)? (non ?manufacturi?ng|on manufacturing|services)( index| pmi)?$"),
    _r("MPMIUSMA Index", "S&P Global US manufacturing PMI", "M", "calendar", "MW:MPMIUSMA", "level",
       0, 1, 1, 0, 1, 0, 1, _PMI, _A, _S, (_MAN,), note="S&P Global (Markit before 2022): proprietary - archived MarketWatch calendar; flash = preliminary, final = final",
       calendar_pattern=r"^(?!.*(services|serivces|non ?manufacturing|composite|chicago|ism))(?=.*\bpmi\b)(?=.*(markit|market|arkit|s&p|flash|final|prelim|manufacturing)).*$|^(s&p|markit)( global)? (final|flash) manufacturing$"),
    _r("CHPMINDX Index", "MNI Chicago business barometer", "M", "calendar", "MW:CHPMINDX", "level",
       0, 0, 0, 0, 1, 1, 1, _PMI, _A, _S, (_OB,), note="MNI: proprietary - archived MarketWatch calendar",
       calendar_pattern=r"^chicago (pmi|business barometer|manufacturing pmi|purchasing managers)"),
    _r("DGNOCHNG Index", "US durable goods new orders MoM", "M", "fred", "DGORDER", "pct",
       0, 1, 1, 1, 1, 0, 1, "rolling_ma:12;ecdf_no_demean_exp:130", _A, _H, (_OB,)),
    _r("RCHSINDX Index", "Richmond Fed manufacturing survey, composite", "M", None, None, "level",
       0, 0, 0, 0, 1, 1, 1, "ecdf_no_demean_exp:130", _A, _S, (_MAN,),
       note="not on FRED; the Richmond Fed publishes it - no client yet (TOFIX.md)"),
    _r("CONCCONF Index", "Conference Board consumer confidence", "M", "calendar", "MW:CONCCONF", "level",
       0, 0, 0, 0, 1, 1, 1, "ecdf_exp:1305", _A, _S, (_CON,), note="Conference Board: proprietary - archived MarketWatch calendar",
       calendar_pattern=r"^(conference (board|bd) )?consumer confidence( index)?$"),
    _r("MPMIUSCA Index", "S&P Global US composite PMI", "M", "calendar", "MW:MPMIUSCA", "level",
       0, 1, 1, 0, 1, 0, 1, _PMI, _A, _S, (_OB,), note="S&P Global: proprietary - archived MarketWatch calendar",
       calendar_pattern=r"^(?!.*(chicago|ism))(?=.*\bpmi\b)(?=.*composite).*$"),
    _r("MPMIUSSA Index", "S&P Global US services PMI", "M", "calendar", "MW:MPMIUSSA", "level",
       0, 1, 1, 0, 1, 0, 1, _PMI, _A, _S, (_OB,), note="S&P Global (Markit before 2022): proprietary - archived MarketWatch calendar; flash = preliminary, final = final",
       calendar_pattern=r"^(?!.*(composite|chicago|ism))(?=.*\bpmi\b)(?=.*(services|serivces|non ?manufacturing)).*$"),
    _r("EMPRGBCI Index", "Empire State manufacturing survey, general conditions", "M", "fred",
       "GACDISA066MSFRBNY", "level", 0, 0, 0, 0, 1, 1, 1, "ecdf_no_demean_exp:130", _A, _S, (_MAN,),
       note="the SEASONALLY ADJUSTED headline - GACDINA066MNFRBNY (NSA) was used until 2026-10-01, caught by the "
            "calendar cross-check (0% match vs the published headline; scripts/validate_econ_calendar.py)"),
)

# The calendar row of each FRED-sourced release: (pattern, scale) - a CROSS-CHECK of the
# FRED data (scripts/validate_econ_calendar.py) and the release's consensus, never its
# data. Only where the calendar quotes the table's own units; left out where it doesn't:
# PPI (the calendar quotes MoM, the table is YoY NSA), retail sales (ex autos, not ex
# autos AND gas). Names: every variant 2009-2026 (scripts/backfill_econ_calendar.py
# --names), incl. the COVID-era claims variants and the 2019 shutdown's "(new date)".
_CALENDAR_CROSSCHECK: dict[str, tuple[str, float]] = {
    "GDP CQOQ Index": (r"^((2nd|3rd|second|third|advance) estimate )?(gdp|gross domestic product)( revision)?"
                       r"( \((real annual rate|revision|first revision|second revision)\))?$", 1.0),
    "PCE CRCH Index": (r"^pce (price )?(index|idx m m)$", 1.0),
    "NFP TCH Index": (r"^(nonfarm payrolls|employment report)$", 1e-3),
    "USURTOT Index": (r"^unemployment rate$", 1.0),
    "MWINCHNG Index": (r"^wholesale inventories$", 1.0),
    "CPI YOY Index": (r"^(cpi|consumer price index)( year over year| y y)$", 1.0),
    "INJCJC Index": (r"^(weekly )?(initial )?jobless claims( \((regular )?state program sa\))?$", 1.0),
    "ADP CHNG Index": (r"^adp (national )?(employment|jobs)( report)?$", 1.0),
    "CONSSENT Index": (r"^(u ?mich(igan)? )?(prelim(inary)? |final )?consumer (sentiment|survey)( index)?"
                       r"( \((final|preliminary|prelim|revised)\)| final| prelim(inary)?)?$", 1.0),
    "IP CHNG Index": (r"^industrial production( m m)?$", 1.0),
    "NHSPSTOT Index": (r"^housing starts( \((saar|annual rate)\))?$", 1e-3),
    "OUTFGAF Index": (r"^philly fed( manufacturing)?( index)?$|^philadelphia fed( s)? (manufacturing|business outlook) survey$",
                      1.0),
    "CFNAI Index": (r"^chicago (fed )?national (activity )?index$|^chicago (fed )?national activity$", 1.0),
    "DGNOCHNG Index": (r"^durable goods orders$", 1.0),
    "EMPRGBCI Index": (r"^empire state( manufacturing)?( index| survey)?$", 1.0),
}
_TABLE = tuple(replace(r, calendar_pattern=_CALENDAR_CROSSCHECK[r.ticker][0],
                       calendar_scale=_CALENDAR_CROSSCHECK[r.ticker][1]) if r.ticker in _CALENDAR_CROSSCHECK else r
               for r in _TABLE)
MACRO_RELEASES: dict[str, MacroRelease] = {r.ticker: r for r in _TABLE}


# Archived economic-calendar pages (infra/pipeline/econ_calendar.py). MarketWatch's U.S.
# economic calendar moved URL in April 2020 and the old URL froze (its later captures
# still show April 2020); both are listed, each with every query-string variant the
# archive holds (``?siteid=...``, ``?mod=...``). ``variants`` are in PRIORITY order: on a
# day with captures of several, the first variant's capture is used. Measured 2026-10-01:
# Apr 2020 - Sep 2026 every week has a capture (336/336); 2009 - Apr 2020 366 of 590
# weeks (62%; 2009-11 thin, 2012-19 25-47 weeks a year).
@dataclass(frozen=True)
class CalendarPage:
    key: str  # coverage key
    cdx_prefixes: tuple[str, ...]  # CDX prefix queries that list every variant
    variants: tuple[str, ...]  # regexes on the captured URL, priority order; others ignored
    history_start: str


CALENDAR_PAGES: dict[str, CalendarPage] = {
    "marketwatch": CalendarPage(
        "marketwatch",
        cdx_prefixes=("marketwatch.com/economy-politics/calendar", "marketwatch.com/tools/calendars/economic"),
        variants=(r"(?i)/economy-politics/calendar(?:\?|$|%0a)", r"(?i)/economy-politics/calendars/economic",
                  r"(?i)/tools/calendars/economic"),
        history_start="2009-01-01",
    ),
}

# ------------------------------------------------------------------ bulk series
# Full-granularity US inflation: every published series of a survey/table set, taken from
# the agency's own bulk flat file (free, no API limits). Neither agency serves VINTAGES,
# so each newly published file version is stored as a vintage stamped with the file's own
# HTTP Last-Modified day (the publication: 08:30 New York on release day, verified
# 2026-09-30 for all four files) - point-in-time history accumulates from the first
# snapshot on (infra/pipeline/bulk_series.py). Vintages before that: ALFRED, for the
# headline series it carries (MACRO_RELEASES).


@dataclass(frozen=True)
class BulkDataset:
    key: str  # coverage/catalog key
    name: str
    source: str  # fetcher in infra.pipeline.bulk_series.SOURCES: "bls" | "bea"
    file: str  # BLS: survey code (download.bls.gov/pub/time.series/<file>/); BEA: data file name
    store: str  # store directory under RAW_DATA_ROOT (one per source: ids are unique within it)
    tables: tuple[str, ...] = ()  # BEA only: keep the series that appear in these tables
    note: str = ""


BULK_DATASETS: dict[str, BulkDataset] = {d.key: d for d in (
    BulkDataset("cpi", "CPI - every item x area x index (CPI-U), SA and NSA", "bls", "cu", "BLS",
                note="~4,000 current series (cu.series also lists discontinued ones); monthly periods only (annual averages M13 and the semiannual "
                     "S01-S03 of the areas published half-yearly are not stored)"),
    BulkDataset("ppi_commodity", "PPI by commodity, incl. the final demand-intermediate demand tree",
                "bls", "wp", "BLS", note="~5,200 series; headline PPI final demand is WPUFD4"),
    BulkDataset("ppi_industry", "PPI by industry (NAICS)", "bls", "pc", "BLS"),
    BulkDataset("pce", "PCE by type of product: price indexes, nominal and real spending", "bea",
                "NipaDataM.txt", "BEA",
                tables=("U20404", "U20405", "U20406", "T20804", "T20805", "T20806"),
                note="~1,170 series, the NIPA 'underlying detail' tree (U2040x) plus the monthly "
                     "headline tables (T2080x). Price index DPCERG, core DPCCRG; U20405 nominal "
                     "spending gives the weights"),
)}

# ALFRED sub-series of CPI / PCE / PPI, fetched with every vintage by the macro-release
# pipeline (infra/pipeline/releases.py, same store RawData/Releases) - the POINT-IN-TIME
# history of the main inflation components BEFORE the bulk snapshots (BULK_DATASETS)
# started (Sep 2026). Not nowcast inputs, so kept out of MACRO_RELEASES. FRED id -> label;
# each verified 2026-09-30 to have ALFRED vintages (first vintage in the comment). SA
# versions throughout: seasonal adjustment is what gets revised (CPI every February).
INFLATION_ALFRED_SERIES: dict[str, str] = {
    # CPI-U, SA
    "CPIAUCSL": "CPI all items",  # 1972
    "CPILFESL": "CPI less food and energy",  # 1996
    "CPIUFDSL": "CPI food",  # 1996
    "CPIENGSL": "CPI energy",  # 1996
    "CPIHOSSL": "CPI housing",  # 2009
    "CPIAPPSL": "CPI apparel",  # 2009
    "CPITRNSL": "CPI transportation",  # 2009
    "CPIMEDSL": "CPI medical care",  # 2009
    "CPIRECSL": "CPI recreation",  # 2009
    "CPIEDUSL": "CPI education and communication",  # 2009
    "CUSR0000SACL1E": "CPI commodities less food and energy (core goods)",  # 2011
    "CUSR0000SASLE": "CPI services less energy services (core services)",  # 2011
    # NOT the market's "supercore" (core services less shelter): this one still holds
    # energy services - Jan 2024: 0.63% first print vs the ~0.85% supercore quoted then.
    # Supercore = CUSR0000SASLE less CUSR0000SAH1, recombined with the CPI weights.
    "CUSR0000SASL2RS": "CPI services less rent of shelter (incl. energy services)",  # 2011
    "CUSR0000SAH1": "CPI shelter",  # 2011
    "CUSR0000SEHA": "CPI rent of primary residence",  # 2011
    "CUSR0000SEHC": "CPI owners' equivalent rent of residences",  # 2011
    "CUSR0000SEHB": "CPI lodging away from home",  # 2011
    "CUSR0000SETA01": "CPI new vehicles",  # 2011
    "CUSR0000SETA02": "CPI used cars and trucks",  # 2011
    "CUSR0000SETD": "CPI motor vehicle maintenance and repair",  # 2011
    "CUSR0000SETG01": "CPI airline fares",  # 2013
    "CUSR0000SAM1": "CPI medical care commodities",  # 2011
    "CUSR0000SAM2": "CPI medical care services",  # 2011
    "CUSR0000SAS4": "CPI transportation services",  # 2011
    # PCE price indexes (PCEPI itself is in MACRO_RELEASES)
    "PCEPILFE": "PCE less food and energy",  # 2000
    "DGDSRG3M086SBEA": "PCE goods",  # 2013
    "DSERRG3M086SBEA": "PCE services",  # 2013
    "DFXARG3M086SBEA": "PCE food and beverages off-premises",  # 2013
    "DNRGRG3M086SBEA": "PCE energy goods and services",  # 2013
    "IA001260M": "PCE services excluding energy and housing (supercore)",  # 2023
    "IA001176M": "PCE excluding food, energy and housing",  # 2023
    # PPI final demand, SA (PPIFID itself is in MACRO_RELEASES)
    "PPIFES": "PPI final demand less foods and energy",  # 2014
    "PPIFDF": "PPI final demand foods",  # 2014
    "PPIFDE": "PPI final demand energy",  # 2014
    "PPIFDG": "PPI final demand goods",  # 2014
    "PPIFDS": "PPI final demand services",  # 2014
    "WPSFD49116": "PPI final demand less foods, energy and trade services",  # 2015
    "WPSFD49511": "PPI personal consumption less foods, energy and trade services",  # 2015
}
