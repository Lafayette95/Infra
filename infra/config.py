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
BMK_ROOT = DATABASE_ROOT / "Bmk"
# Full dated snapshots of the database, one per cycle run day (``_vintages/YYYY-MM-DD``) -
# the baseline each run's "no revisions" check compares against.
VINTAGE_ROOT = DATABASE_ROOT / "_vintages"
# Sidecar log of every value the pipeline TOUCHED (a bad print NA'd or rolled, ...): the
# data stores keep exactly what the vendor delivered; readers overlay this at read time.
# infra.storage.adjustment_store, CLAUDE.md section 12.
ADJUSTMENTS_DIR = DATABASE_ROOT / "_adjustments"

# ------------------------------------------------------------------ API settings
SCHEMA_OHLCV = "ohlcv-1m"
SCHEMA_OHLCV_1S = "ohlcv-1s"
SCHEMA_BBO_1M = "bbo-1m"
SCHEMA_BBO_1S = "bbo-1s"
SCHEMA_DEFINITION = "definition"
SCHEMA_STATISTICS = "statistics"
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
    # Currency value of a 1.00 move in the quoted price, per contract (daily cycle pnl /
    # DV01, CLAUDE.md 12). None = not yet verified against the exchange's contract specs.
    point_value: float | None = None
    currency: str | None = None


_CME, _EUREX, _ICE = "GLBX.MDP3", "XEUR.EOBI", "IFLL.IMPACT"
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
        FuturesRoot("ZT", _CME, "ZT.FUT", "US 2Y Note", "Bonds", point_value=2000.0, currency="USD"),
        FuturesRoot("ZF", _CME, "ZF.FUT", "US 5Y Note", "Bonds", point_value=1000.0, currency="USD"),
        FuturesRoot("ZN", _CME, "ZN.FUT", "US 10Y Note", "Bonds", point_value=1000.0, currency="USD"),
        FuturesRoot("TN", _CME, "TN.FUT", "US Ultra 10Y Note", "Bonds", point_value=1000.0, currency="USD"),
        FuturesRoot("ZB", _CME, "ZB.FUT", "US Classic Bond", "Bonds", point_value=1000.0, currency="USD"),
        FuturesRoot("UB", _CME, "UB.FUT", "US Ultra Bond", "Bonds", point_value=1000.0, currency="USD"),
        # ---- Bonds: Eurex (XEUR.EOBI only has data from 2025-03-10)
        FuturesRoot("FGBL", _EUREX, "FGBL.FUT", "Euro-Bund", "Bonds", point_value=1000.0, currency="EUR"),
        FuturesRoot("FGBM", _EUREX, "FGBM.FUT", "Euro-Bobl", "Bonds", point_value=1000.0, currency="EUR"),
        FuturesRoot("FGBS", _EUREX, "FGBS.FUT", "Euro-Schatz", "Bonds", point_value=1000.0, currency="EUR"),
        FuturesRoot("FBTP", _EUREX, "FBTP.FUT", "Euro-BTP", "Bonds", point_value=1000.0, currency="EUR"),
        # ---- Bonds: ICE Futures Europe (data from 2018-12-23)
        FuturesRoot("R", _ICE, "R.FUT", "UK Long Gilt", "Bonds", ticker_regex=_ICE_QUARTERLY),
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
