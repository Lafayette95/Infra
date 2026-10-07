"""Central configuration: paths, storage constants and the instrument universe.

Code lives in ~/Repos/Infra; the database lives separately under ~/Database.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from infra.reference.events import EventSeries, series_by_bbg

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- database paths
DATABASE_ROOT = Path(os.environ.get("INFRA_DATABASE_ROOT", "~/Database")).expanduser()
# Reference data: what instruments exist and their fixed characteristics, not prices
# (CLAUDE.md 18) - the futures contracts table, option-chain definitions, the Treasury
# securities / on-the-run map / delivery baskets.
REFERENCE_ROOT = DATABASE_ROOT / "Reference"
OHLCV_ROOT = DATABASE_ROOT / "ohlcv-1m"
FUTURES_DIR = OHLCV_ROOT / "Futures"
OPTIONS_DIR = OHLCV_ROOT / "Options"
COVERAGE_DIR = OHLCV_ROOT / "_coverage"  # which (key, date range) were already queried
# Cached option-chain `definition` snapshots, one file per parent and day (Rule 2.3).
DEFINITIONS_DIR = REFERENCE_ROOT / "Options" / "Definitions"

FUTURES_COVERAGE_FILE = COVERAGE_DIR / "futures.parquet"
# Which futures `definition` snapshots were already pulled (they build the contracts table).
FUTURES_DEFS_COVERAGE_FILE = REFERENCE_ROOT / "_coverage" / "futures_definitions.parquet"
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
# daily bars (Long Gilt: ICE's statistics schema is ~30x the price of these - CLAUDE.md 14)
OHLCV_1D_ROOT = DATABASE_ROOT / "ohlcv-1d"
OHLCV_1D_FUTURES_DIR = OHLCV_1D_ROOT / "Futures"
OHLCV_1D_FUTURES_COVERAGE_FILE = OHLCV_1D_ROOT / "_coverage" / "futures.parquet"
BBO_1S_ROOT = DATABASE_ROOT / "bbo-1s"
BBO_1S_FUTURES_DIR = BBO_1S_ROOT / "Futures"
BBO_1S_FUTURES_COVERAGE_FILE = BBO_1S_ROOT / "_coverage" / "futures.parquet"

# Master table of absolute futures contracts (root, ticker, expiry, ...).
FUTURES_CONTRACTS_FILE = REFERENCE_ROOT / "Futures" / "contracts.parquet"

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
DAILY_BOE_OIS_DIR = DAILY_ROOT / "BoeOIS"  # the BoE's SONIA OIS spot curve (2009 on)
DAILY_BOE_OIS_COVERAGE_FILE = DAILY_COVERAGE_DIR / "boe_ois.parquet"
# US Treasury prices per CUSIP (FedInvest END OF DAY, + accrued and yield) - CLAUDE.md 18.
# Every in-scope security on the page is stored (one free request per day returns them
# all); which bonds a consumer uses is a VIEW (on-the-run map, futures baskets).
DAILY_TREASURY_PRICES_DIR = DAILY_ROOT / "TreasuryPrices"
DAILY_TREASURY_PRICES_COVERAGE_FILE = DAILY_COVERAGE_DIR / "treasury_prices.parquet"
# German Federal securities (2026-10-07): per-ISIN clean / dirty price and yield from the
# Bundesbank's BBSSY (infra.pipeline.bunds); a day is claimed covered once BUND_PRICES_SETTLE_DAYS old
DAILY_BUND_PRICES_DIR = DAILY_ROOT / "BundPrices"
DAILY_BUND_PRICES_COVERAGE_FILE = DAILY_COVERAGE_DIR / "bund_prices.parquet"
BUND_PRICES_SETTLE_DAYS = 1
# When the Bundesbank's BBSSY prices are TAKEN (measured 2026-10-07, not published by the
# source): the on-the-run 10y Bund's daily yield change against the front Bund future's mid
# change at every 5 minutes, 2025-03..2026-10, fits best at 11:15 Frankfurt in summer
# (09:15 UTC, residual 0.33bp) and 11:20 in winter (10:20 UTC, 0.24bp), against 1-3bp an hour
# either side and 3.4bp at 16:15 London - a late-morning snapshot, not a close.
BUND_PRICES_LOCAL_TIME = ("11:15", "Europe/Berlin")
# German on/off-the-run map (infra.pipeline.bunds.otr_map, computed on demand from the
# issuance history): tenor -> (security types, maturity segment of the FIRST issuance);
# conventional securities only (no Green, no inflation-linked). 7y / 15y from 2020, 20y from 2026.
DE_OTR_TENORS: dict[str, tuple[tuple[str, ...], str]] = {
    "2y": (("Schatz",), "2 Y"), "5y": (("Bobl",), "5 Y"), "7y": (("Bund",), "7 Y"), "10y": (("Bund",), "10 Y"),
    "15y": (("Bund",), "15 Y"), "20y": (("Bund",), "20 Y"), "30y": (("Bund",), "30 Y"),
}
DE_OTR_DEPTH = 3
DAILY_TIPS_PRICES_DIR = DAILY_ROOT / "TipsPrices"
DAILY_TIPS_PRICES_COVERAGE_FILE = DAILY_COVERAGE_DIR / "tips_prices.parquet"
# An empty page this many days old is a holiday (covered); a newer one, or a page whose
# END OF DAY column isn't posted yet, is asked again on the next run.
TREASURY_PRICES_SETTLE_DAYS = 3

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
OIS_CURVES_DIR = DERIVED_ROOT / "OisCurves"
SWAP_SPREADS_DIR = DERIVED_ROOT / "SwapSpreads"
INFLATION_SWAP_CLOSES_DIR = DERIVED_ROOT / "InflationSwapCloses"
INFLATION_CURVES_DIR = DERIVED_ROOT / "InflationCurves"
XCCY_BASIS_CLOSES_DIR = DERIVED_ROOT / "XccyBasisCloses"
SWAPTION_RECORDS_DIR = DERIVED_ROOT / "SwaptionRecords"
SWAPTION_PRINTS_DIR = DERIVED_ROOT / "SwaptionPrints"
SWAPTION_VOLS_DIR = DERIVED_ROOT / "SwaptionVols"
SWAPTION_OI_DIR = DERIVED_ROOT / "SwaptionOI"
VRP_DIR = DERIVED_ROOT / "VolRiskPremium"
# On-the-run yield benchmark (CLAUDE.md 18): per day and tenor, the END OF DAY yield of the
# on-the-run CUSIP (issue-date convention, so it always has a price), stored under the SAME
# tickers as the CMT par curve (US_BOND_10y) - a consumer picks the series by ``source``
# (BOND_YIELD_SOURCES) through infra.pipeline.bond_yields.read_bond_yields.
OTR_YIELDS_DIR = DERIVED_ROOT / "OTRYields"
# Our own US Treasury zero curve (sub-project infra/models/curves, methodology there):
# per day x fit method - parameters, fit quality, zero / par grid; and per CUSIP - z-spread
# (+ leave-one-out), curve carry and rolldown. infra/pipeline/treasury_curves.py.
TREASURY_CURVES_DIR = DERIVED_ROOT / "TreasuryCurves"
TREASURY_RV_DIR = DERIVED_ROOT / "TreasuryRV"
# Our German Federal curve (infra.pipeline.bund_curves): the same fit and metrics as the US
# one on the Bundesbank's per-ISIN DIRTY prices (an 11:15 Frankfurt snapshot), settlement T+2,
# from BUND_CURVES_START - the first day with dirty prices AND the complete universe.
BUND_CURVES_DIR = DERIVED_ROOT / "BundCurves"
# Eurex German bond futures (infra.pipeline.eurex_basis): delivery baskets with conversion
# factors per day and listed contract, and the daily basis table (gross basis, implied repo,
# cheapest-to-deliver, futures DV01) at the Bundesbank's 11:15 Frankfurt price time
EUREX_BASKETS_DIR = REFERENCE_ROOT / "Bunds" / "FuturesBaskets"
EUREX_BASIS_DIR = DERIVED_ROOT / "EurexBasis"
EUREX_BASKETS_START = "2015-01-02"
EUREX_BASIS_START = "2025-03-20"   # the first day of stored Eurex futures quotes
BUND_RV_DIR = DERIVED_ROOT / "BundRV"
BUND_CURVES_START = "2022-06-01"
BUND_SETTLEMENT_DAYS = 2
# Chosen 2026-10-07 against the Bundesbank's own (Svensson) par curve on 6 days 2022-2026:
# the US knots (to 25y) made the spline's 30y 10-47bp off - only ~4 Bunds lie beyond 20y and
# none beyond 25y in the fit; knots to 15y bring it to ~5bp, the rest 1.3-2bp. Excluding only
# the on-the-run (rank 0, not the US's 0 and 1: few bonds per German tenor) keeps Svensson's
# 30y at 3.5bp (5.8 excluding both); Svensson is within 0.8-1.4bp to 20y.
BUND_CURVE_KNOTS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0)
BUND_FIT_EXCLUDE_RANKS = 1
@dataclass(frozen=True)
class EurexBondFuture:
    """A Eurex German government-bond future (infra.processing.eurex_baskets): remaining term
    of deliverables at delivery, the cap on their ORIGINAL term (None = none), the notional
    coupon of the conversion factor, the minimum issued volume. Verified 2026-10-07 against
    Eurex's deliverable-bonds file (every German contract listed then: basket and CFs)."""
    min_years: float
    max_years: float
    max_original_years: float | None
    notional_pct: float
    min_volume_m: float = 5000.0


EUREX_BOND_FUTURES: dict[str, EurexBondFuture] = {
    "FGBS": EurexBondFuture(1.75, 2.25, 11.0, 6.0),
    "FGBM": EurexBondFuture(4.5, 5.5, 11.0, 6.0),
    "FGBL": EurexBondFuture(8.5, 10.5, 11.0, 6.0),
    "FGBX": EurexBondFuture(24.0, 35.0, None, 4.0),
}
BUND_CURVE_BMK_METHOD = "svensson"   # the yield_curve bmk for DE_BOND_<t>y: Svensson, the closer and steadier fit
BUND_IRREGULAR_DAYS = 3    # Bundesbank accrued off a regular annual schedule by more than this: an irregular first coupon
TIPS_CURVES_DIR = DERIVED_ROOT / "TipsCurves"
TIPS_RV_DIR = DERIVED_ROOT / "TipsRV"
TIPS_FIT_MIN_YEARS = 1.5  # TIPS closer to maturity stay out of the real-curve fit (still get metrics): their
                          # yield is dominated by the next CPI prints' seasonality and carry - the Fed's GSW
                          # TIPS curve leaves out the last 18 months the same way
# Basis model runs per model spec (M2, M2T, ...): per day x contract and per day x contract x
# bond - written by scripts/run_basis.py --persist, read by the basis dashboard page.
BASIS_RUNS_DIR = DERIVED_ROOT / "BasisRuns"
CURVE_FIT_MIN_YEARS = 0.5      # bonds shorter than this aren't in the fit (still get metrics)
CURVE_FIT_EXCLUDE_RANKS = 2    # on-the-run and first off-the-run out of the fit (Fed GSW practice)
CURVE_HORIZON_DAYS = 91        # carry / rolldown horizon (calendar days)
CURVE_GRID_YEARS = (1, 2, 3, 5, 7, 10, 20, 30)
# Saved walk-forward runs of the statistical models (infra/models/stats, CLAUDE.md 25):
# one folder per run name holding params.parquet (one tidy params frame per refit) and
# predictions.parquet (the stitched out-of-sample rows), written by
# scripts/run_walk_forward.py through infra.storage.model_runs.
MODEL_RUNS_DIR = DERIVED_ROOT / "ModelRuns"
# Strategies (infra/strategies, root CLAUDE.md 27): per strategy, its firm signals and positions
# (relative and absolute tickers) and its plan vintages, written by infra.jobs.strategy_runs.
STRATEGIES_DIR = DATABASE_ROOT / "Strategies"
# The feature maker (infra.pipeline.features, root CLAUDE.md 31): how long a series' last value may
# be carried forward when a feature is aligned onto another timeline (a model's daily index),
# per series SOURCE (the id's prefix) - beyond it the value is stale and becomes NaN (user decision
# 2026-10-06: a limit per source type). Daily market data: a long weekend plus a holiday; monthly
# macro: a release cycle plus slack; quarterly: a quarter plus slack.
FEATURE_FFILL_LIMITS = {
    "fut": "5D", "settle": "5D", "stir": "5D", "bond": "5D", "otr": "5D", "bmk": "5D", "swap": "5D",
    "repo": "5D", "bar": "1h", "release": "100D", "surprise": "100D", "evt": "1D", "model": "5D",
    "derived": "5D",
}
BOND_YIELD_SOURCES = ("cmt", "otr", "curve")   # curve = our fitted spline's par yield (TREASURY_CURVES_DIR)
# Benchmark yield P&L persisted by the daily cycle's bmk_pnl step (infra/cycle/bmk_yields.py,
# root CLAUDE.md 12): per ticker and source, the day's PRICE-ACTION P&L in bp of a long
# position = -(yield change) x 100, stored under bmk "yield_<source>" in Bmk/Pnl. Sign: + = long
# duration (gains when yields fall), so P&L = position x pnl_per_dv01 for yields and futures
# alike (user decision 2026-10-05). "otr" is computed on the bond held the PREVIOUS day (an
# on-the-run switch is never a move); "cmt" / "curve" are constant-maturity points (no carry or
# rolldown: a total-return definition including carry comes later).
BMK_YIELD_TICKERS = tuple(f"US_BOND_{t}y" for t in (2, 3, 5, 7, 10, 20, 30))
BMK_YIELD_SOURCES = ("cmt", "otr", "curve")
BMK_YIELD_MAX_GAP_DAYS = 7          # a change spanning a longer gap (missing data) is not a day's P&L
BMK_YIELD_AGREE_BP = 5.0            # sources disagreeing on a day's move by more than this are listed
# Non-US benchmark yield P&L (2026-10-07): the official par curve per country (Daily/Bonds,
# BOND_CURVES), bmk ``yield_<source>`` - its own name, not ``yield_cmt``, since it is the
# central bank's fitted curve, not the US Treasury's CMT: country -> source key.
BMK_YIELD_OFFICIAL = {"UK": "boe", "DE": "bundesbank", "JP": "mof", "CA": "boc"}
BMK_YIELD_OFFICIAL_TENORS = (2, 3, 5, 7, 10, 20, 30)

# Repo rates and the NY Fed's Treasury securities lending (CLAUDE.md 19), both fetched by
# the px step. Free sources, verified 2026-10-02 (source facts in each infra/api client).
REPO_DIR = DAILY_ROOT / "Repo"
REPO_COVERAGE_FILE = DAILY_COVERAGE_DIR / "repo.parquet"
NYFED_REPO_RATES = ("SOFR", "TGCR", "BGCR")
NYFED_REPO_START = "2018-04-02"  # all three began that day
# OFR's U.S. Repo Markets release: per service, the term buckets (and "T" = Treasury
# collateral, all terms) stored. DVP (FICC cleared bilateral) has no collateral split but
# is almost all Treasuries, so its term buckets are the best free TERM general-collateral
# proxy. TRIV1 = tri-party excluding the Fed's own trades (its reverse repo facility).
# LE30 ends 2025-08-12, replaced by B27 + B830.
OFR_REPO_BUCKETS: dict[str, tuple[str, ...]] = {
    "DVP": ("OO", "B27", "B830", "LE30", "G30", "TOT"),
    "GCF": ("OO", "B27", "B830", "LE30", "G30", "TOT", "T"),
    "TRI": ("OO", "B27", "B830", "LE30", "G30", "TOT", "T"),
    "TRIV1": ("OO", "B27", "B830", "LE30", "G30", "TOT", "T"),
}
OFR_REPO_START = "2014-08-22"  # tri-party's first day; DVP and GCF start 2018-05-07
# The finals of this series mark how far OFR's final data reaches (all finals move
# together); days after it are re-requested every run until final.
OFR_FINAL_REFERENCE = ("DVP", "OO")
# The DTCC GCF Repo Index history - frozen, 2005-01-03..2024-12-31, fetched once.
DTCC_GCF_SPAN = ("2005-01-03", "2024-12-31")
# A day is claimed covered once this many days old (NY Fed rates can be revised the
# afternoon after; lending extensions post the morning after).
REPO_SETTLE_DAYS = 2
SEC_LENDING_DIR = DAILY_ROOT / "SecLending"
SEC_LENDING_COVERAGE_FILE = DAILY_COVERAGE_DIR / "sec_lending.parquet"
SEC_LENDING_START = "1999-01-04"  # first operation in the API: 1999-04-29
# The program's MINIMUM FEE (percent), (effective from, fee): bids can't go below it, so
# only the fee ABOVE it signals specialness (infra.processing.sec_lending.excess_fee).
# Inferred 2026-10-02 from the data itself - each regime's floor is the most common daily
# minimum, the switch the first day at the new level - and checked: 96.9% of days have
# their lowest fee exactly at this floor, none below it. Not yet matched to the NY Fed's
# announcements (TOFIX.md). The pre-2008 changes follow the FOMC (2001-09-18 after 9/11,
# 2003-06-26 the day after the cut to 1%, 2004-07-01 the day after the first hike) and the
# crisis (2007-08-21, 2008-10-08, 2008-12-18).
SEC_LENDING_MIN_FEE: tuple[tuple[str, float], ...] = (
    ("1999-04-29", 1.50), ("2001-09-18", 1.00), ("2003-06-26", 0.75), ("2004-07-01", 1.00),
    ("2007-08-21", 0.50), ("2008-10-08", 0.10), ("2008-12-18", 0.01), ("2009-04-08", 0.05),
)

# Financing models (CLAUDE.md 20): the rate a levered client pays to fund a Treasury,
# built in three swappable LAYERS (infra.analytics.financing): ``base`` (general-collateral
# path), ``basis`` (the client's spread over it) and ``specialness`` (per CUSIP, subtracted).
# A model is a named combination, so two can run side by side. v1 ASSUMPTIONS (user
# decisions 2026-10-02):
#   * ROLLING OVERNIGHT funding, not term repo locked to the end date - a term premium
#     then shows up in the futures' net basis, which is wanted;
#   * base = SOFR's path implied by SR1 futures, flat between FOMC decisions, plus the
#     year-end turn (median of the last ``year_end_turn_years`` year-ends);
#   * basis = SOFR's 75th percentile minus its median (hedge funds fund in DVP above the
#     dealers' median), a rolling ``basis_window``-fixing median held CONSTANT over the
#     term; its own small year-end premium (~3bp) is left out;
#   * specialness = per-CUSIP lifecycle profile (tenor x on/off-the-run rank x phase of the
#     issuance cycle) plus today's observed deviation (NY Fed lending fee above the
#     minimum) decaying with an estimated half-life, both estimated from the 5 years
#     before the as-of year (infra.analytics.specialness). Not modelled: squeezes of the
#     cheapest-to-deliver into futures delivery.
@dataclass(frozen=True)
class FinancingSpec:
    base: str  # infra.analytics.financing.BASE_MODELS
    basis: str  # BASIS_MODELS
    specialness: str  # SPECIALNESS_MODELS
    basis_window: int = 20  # published fixings in the basis's rolling median
    year_end_turn_years: int = 3  # recent year-ends the imposed turn is estimated from
    description: str = ""


FINANCING_MODELS: dict[str, FinancingSpec] = {
    "v1": FinancingSpec("sofr_futures", "sofr_p75", "lifecycle_decay",
                        description="rolling O/N: SR1-implied SOFR + SOFR p75 basis - lifecycle/decay specialness"),
}
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
# Where the Treasury history (OTR map, per-CUSIP prices) starts: FedInvest's first day
# (nothing before 2008-09-02; verified 2026-10-01). Backfilled to here 2026-10-02.
TREASURY_PRICES_START = "2008-09-02"
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
EUREX_CF_DIR = RAW_DATA_ROOT / "EUREX_CF"   # Eurex's deliverable-bonds CSV, archived raw per day (a file = coverage)
DE_AUCTIONS_DIR = RAW_DATA_ROOT / "DE_Auctions"  # the Finanzagentur issuance history, one row per (day, ISIN)
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
# CFTC cumulative report kinds archived (also: CREDITS, ...). FOREX added 2026-10-07 (user):
# FX forwards / swaps / NDFs / options, ~2.5MB a day - the short end of the cross-currency
# hedge (the RATES basis swaps are thin at 3m/6m); same column layout as RATES.
DTCC_REPORTS = ("RATES", "FOREX")
DTCC_FIRST_DAY = "2024-09-30"  # never request earlier days: DTCC no longer has them
DTCC_RETENTION_DAYS = 700  # look back this far for unarchived days (inside DTCC's ~730)
# CFTC Commitments of Traders, TRADERS IN FINANCIAL FUTURES (TFF): weekly positions by
# trader class (dealer / asset manager / leveraged funds / other / non-reportable) in every
# financial futures market, from 2006-06-13 - infra/pipeline/cftc_tff.py. Free Socrata API
# (publicreporting.cftc.gov), no key; ~47k rows per report, one request. Positions are as of
# TUESDAY, released FRIDAY 15:30 New York (a government shutdown delays releases - the 2013,
# 2018-19 and 2025 ones did). Reports: futures only, and futures + options combined.
CFTC_TFF_DIR = RAW_DATA_ROOT / "CFTC_TFF"
CFTC_TFF_STATE_FILE = RAW_DATA_ROOT / "_coverage" / "cftc_tff.parquet"
CFTC_TFF_REPORTS = {"futures": "gpe5-46if", "combined": "yw9f-hn96"}  # report -> Socrata dataset id
CFTC_TFF_RELEASE = ("15:30", "America/New_York", 3)  # local time, zone, days after the Tuesday
CFTC_TFF_REFETCH_WEEKS = 8  # on a new release, re-read this many weeks back (catches revisions)
# NY Fed PRIMARY DEALER STATISTICS (FR 2004): weekly positions, transactions, financing
# (repo / reverse repo by collateral and term) and fails of the primary dealers, ~2,300
# series ($ millions; "*" = suppressed for confidentiality -> NaN), from 1998-01-28 -
# infra/pipeline/primary_dealer.py. One free CSV holds every series' whole history (26 MB).
# As of WEDNESDAY (MBS settlement-class series: their own dates), released the THURSDAY of
# the following week, 16:15 New York. The reporting form changed at each "series break"
# (2001, 2013, 2015, 2022, 2024): a key's meaning can shift across one - the catalog holds
# the current break's descriptions (``RawData/_catalog/primary_dealer.parquet``).
PRIMARY_DEALER_DIR = RAW_DATA_ROOT / "PrimaryDealer"
PRIMARY_DEALER_STATE_FILE = RAW_DATA_ROOT / "_coverage" / "primary_dealer.parquet"
PRIMARY_DEALER_RELEASE = ("16:15", "America/New_York", 8)  # local time, zone, days after the Wednesday

# ------------------------------------------------------------------ API settings
SCHEMA_OHLCV = "ohlcv-1m"
SCHEMA_OHLCV_1S = "ohlcv-1s"
SCHEMA_OHLCV_1D = "ohlcv-1d"
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
        FuturesRoot("FGBX", _EUREX, "FGBX.FUT", "Euro-Buxl", "Bonds", point_value=1000.0, currency="EUR",
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        # ---- Bonds: ICE Futures Europe (data from 2018-12-23)
        FuturesRoot("R", _ICE, "R.FUT", "UK Long Gilt", "Bonds", ticker_regex=_ICE_QUARTERLY,
                    volume_extra_candidates=_FRONT_TWO,
                    volume_lookback_days=_BOND_LOOKBACK),
        # ---- Macro cross-asset (CME Globex), added 2026-10-07 for the positioning measures
        # (infra/analytics/positioning): daily settlements first, quotes later. NOT in the
        # daily cycle. Point values / currencies left unset (None = not verified against
        # CME's contract specs - cmegroup.com blocks this machine). Expiry cycles: index and
        # FX futures quarterly (serial FX months dropped); energy monthly; metals their
        # ACTIVE months only (GC Feb/Apr/Jun/Aug/Oct/Dec; SI, HG Mar/May/Jul/Sep/Dec) -
        # the inactive months trade little. v.0 among the front two, like bond futures.
        *(FuturesRoot(root, _CME, f"{root}.FUT", name, cat, expiry_months=months,
                      volume_extra_candidates=_FRONT_TWO, volume_lookback_days=_BOND_LOOKBACK)
          for root, name, cat, months in (
              ("ES", "E-mini S&P 500", "Equity", (3, 6, 9, 12)),
              ("NQ", "E-mini Nasdaq-100", "Equity", (3, 6, 9, 12)),
              ("RTY", "E-mini Russell 2000", "Equity", (3, 6, 9, 12)),
              ("NKD", "Nikkei 225 (USD)", "Equity", (3, 6, 9, 12)),
              ("6E", "Euro FX", "FX", (3, 6, 9, 12)),
              ("6J", "Japanese Yen", "FX", (3, 6, 9, 12)),
              ("6B", "British Pound", "FX", (3, 6, 9, 12)),
              ("6A", "Australian Dollar", "FX", (3, 6, 9, 12)),
              ("6C", "Canadian Dollar", "FX", (3, 6, 9, 12)),
              ("6S", "Swiss Franc", "FX", (3, 6, 9, 12)),
              ("6M", "Mexican Peso", "FX", (3, 6, 9, 12)),
              ("CL", "WTI Crude Oil", "Commodity", tuple(range(1, 13))),
              ("NG", "Henry Hub Natural Gas", "Commodity", tuple(range(1, 13))),
              ("GC", "Gold", "Commodity", (2, 4, 6, 8, 10, 12)),
              ("SI", "Silver", "Commodity", (3, 5, 7, 9, 12)),
              ("HG", "Copper", "Commodity", (3, 5, 7, 9, 12)),
          )),
    )
}
MACRO_ROOTS = ("ES", "NQ", "RTY", "NKD", "6E", "6J", "6B", "6A", "6C", "6S", "6M", "CL", "NG", "GC", "SI", "HG")

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
    # the fixed leg's conventions, for pricing (infra.analytics.swap_curve.swap_schedule):
    # accruals ACT/day_basis, a payment every fixed_freq_months beyond 1y (one payment up to
    # 1y), dates on ``calendar`` ("us" = SIFMA-like; "weekday" for the non-USD curves, whose
    # holiday calendars aren't encoded - TOFIX)
    day_basis: float = 360.0
    fixed_freq_months: int = 12
    calendar: str = "us"


@dataclass(frozen=True)
class InflationSwapSpec:
    """Zero-coupon inflation swaps snapped from the DTCC archive (infra.pipeline.
    inflation_swaps): the product, the nominal OIS curve the real rates are taken against,
    the closes, and whether package legs count (USD: yes - they trade at the market).

    Survey 2026-10-07 (USD CPI-U, 2 years): 169 fresh trades a day; spot-starting, no
    upfront, whole years, packages included, ~13 a day at 1y, 7 at 2y, 14 at 5y and 10y,
    7 at 30y. 55% of trades are seasoned (effective in the past), a third carry an upfront
    (their fixed rate is the ORIGINAL coupon) - both excluded by the par filter. The
    12-month 1st-to-1st no-upfront trades (~7,500) look like CPI fixings but their CPI-month
    mapping isn't decoded yet (TOFIX); EUR / UK swaps use standardized 15th-of-month dates
    (not mapped yet)."""
    curve: SwapCurveSpec
    currency: str
    ois_curve: str  # an OIS_CURVES name, for real rates
    closes: tuple[str, ...] = ("NY1530",)
    allow_packages: bool = True
    # the close's fallback window when fewer than 3 trades sit within +-30 min: inflation
    # swaps trade all day (9-12 New York busiest), so the OIS closes' +-60 found a print on
    # only 54-80% of days; +-240 finds one on 85-99% and the day-to-day noise doesn't rise
    # (10y daily-change sd 2.08bp at +-60, 1.89bp at +-240; 2025-10..2026-09)
    fallback_half_window_min: int = 240


SWAP_CURVES: dict[str, SwapCurveSpec] = {
    "USD": SwapCurveSpec("NA/Swap OIS USD", "SOFR", 2, (1, 2, 3, 5, 7, 10, 15, 20, 30)),
    "EUR": SwapCurveSpec("NA/Swap OIS EUR", "EuroSTR", 2, (1, 2, 3, 5, 7, 10, 15, 20, 30), calendar="weekday"),
    "GBP": SwapCurveSpec("NA/Swap OIS GBP", "SONIA", 0, (1, 2, 3, 5, 7, 10, 15, 20, 30), day_basis=365.0,
                         calendar="weekday"),
    # added 2026-10-07; CAD's fixed leg is SEMI-ANNUAL beyond 1y (reported MNTH x 6 - found:
    # an annual-only filter let 1.5 prints a day through instead of ~30)
    "JPY": SwapCurveSpec("NA/Swap OIS JPY", "TONA", 2, (1, 2, 3, 5, 7, 10, 15, 20, 30), day_basis=365.0,
                         calendar="weekday"),
    "CAD": SwapCurveSpec("NA/Swap OIS CAD", "CORRA", 1, (1, 2, 3, 5, 7, 10, 15, 20, 30),
                         fixed_frequencies=("MNTH", "YEAR", "EXPI"), day_basis=365.0, fixed_freq_months=6,
                         calendar="weekday"),
}

@dataclass(frozen=True)
class XccyBasisSpec:
    """Cross-currency OIS basis swaps snapped from the DTCC archive (infra.processing.
    dtcc_xccy, infra.pipeline.xccy_basis): the product (UPI FISN), the floating legs that
    make it OIS-vs-OIS (``underlier_pattern``, regex on ``UPI Underlier Name``), the tenors
    in MONTHS, the closes. The basis is the spread on the non-USD leg, in bp (market
    convention: foreign OIS + basis vs flat SOFR; negative = paying to swap into USD).

    Survey 2026-10-07 (2 years of RATES files): clean spot-starting OIS-vs-OIS prints a
    day - EUR 1y-10y 2-7 (3m/6m ~1), JPY 1y-10y 1.5-10, GBP 1-4, CAD 0.5-2 (1m-3m ~1)."""
    fisn: str
    currency: str  # the non-USD currency
    underlier_pattern: str
    tenors_months: tuple[int, ...] = (3, 6, 12, 24, 36, 60, 84, 120, 180, 240, 360)
    # an END-OF-DAY close from the day's prints: these trade in London (EUR, GBP) and Tokyo +
    # London (JPY) hours - at NY1530 +-240 min JPY had a close on 5-9% of days, EUR 34%. The
    # snap at 16:00 New York, the window reaching 18h BACK (to 02:00/03:00 UTC, Tokyo's
    # morning): nothing after the snap, so a close is known at its own timestamp; later prints
    # weigh more (the drift term). The basis moves ~1-2bp a day, so a day-wide window is fine.
    closes: tuple[str, ...] = ("NY1600",)
    fallback_half_window_min: int = 1080
    off_market_bp: float = 5.0  # a print further than max(this, 5 MADs) from its tenor's day median is dropped
    # a close more than ``suspect_bp`` from the median of its previous ``suspect_window``
    # closes is flagged ``suspect`` (kept; clean reads skip it): lone prints of the right
    # size but the WRONG sign (JPY 1y +33bp the day after -33bp) pass every same-day filter
    suspect_bp: float = 15.0
    suspect_window: int = 10


XCCY_BASIS: dict[str, XccyBasisSpec] = {
    "EURUSD": XccyBasisSpec("NA/Swap Flt Flt EUR USD", "EUR", r"EuroSTR.*SOFR|SOFR.*EuroSTR"),
    "USDJPY": XccyBasisSpec("NA/Swap Flt Flt JPY USD", "JPY", r"TONA.*SOFR|SOFR.*TONA"),
    "GBPUSD": XccyBasisSpec("NA/Swap Flt Flt GBP USD", "GBP", r"SONIA.*SOFR|SOFR.*SONIA"),
    "USDCAD": XccyBasisSpec("NA/Swap Flt Flt CAD USD", "CAD", r"CORRA.*SOFR|SOFR.*CORRA"),
}


INFLATION_SWAPS: dict[str, InflationSwapSpec] = {
    "USD_CPI": InflationSwapSpec(SwapCurveSpec("NA/Swap Infl Idx USD", "USA-CPI-U", 2, (1, 2, 3, 4, 5, 7, 10, 15, 20, 30),
                                               fixed_frequencies=("YEAR", "EXPI")), "USD", "USD_SOFR"),
}
# Spot-start tolerance: the effective date may land up to this many calendar days after
# trade + spot lag (holidays, which plain business days ignore); maturity may miss
# effective + N years by this many days (date rolling) and still count as tenor N.
SWAP_SPOT_TOLERANCE_DAYS = 2


@dataclass(frozen=True)
class OisCurveSpec:
    """An OIS discount curve bootstrapped from stored swap closes (infra.analytics.swap_curve,
    infra.pipeline.ois_curves): the close, the closes' method, the short end below the first
    swap pillar, and the fewest tenors a day needs."""
    currency: str
    close: str  # a SWAP_CLOSES name: the snap the curve is AT
    method: str = "pure"  # the swap closes' method ("pure" / "adjusted")
    # "sofr_path" (SR1-fitted overnight path, monthly nodes) / "esr_futures" (the ESR strip's
    # quarters, compounded) / "boe_ois" (the BoE's SONIA curve, monthly nodes) / "none" (flat
    # forward to the first swap pillar); short nodes stop 3 months before the first pillar
    short_end: str = "sofr_path"
    short_end_months: int = 6
    min_tenors: int = 6  # tenors with a close (fills not counted)
    fill_missing: bool = True  # fill an interior missing tenor by its last fly residual
    fill_max_age_days: int = 14  # ... observed at most this many calendar days before
    fill_ends: bool = False  # fill a missing first / last tenor too (neighbour + its last spread)


# SOFR's front end comes from the SR1-fitted overnight path (financing layer 1, CLAUDE.md 20):
# a flat forward to the 1y pillar misses a priced hiking / cutting path (2026-10-05: path
# 4.18% to 6 months, 4.70% from 6 months to 1y; flat-to-1y gave 4.45% throughout). The path
# and the 1y swap close agree: its 1y compounded rate 4.486% vs the 1y close 4.490%. Six
# months, not twelve, so the 6m-1y forward absorbs any gap between the two (the settlement
# is 15:00 New York, the close 15:30) instead of a kink right at the pillar.
# Missing interior tenors (15y absent on 15% of days, 20y 8%, 7y 5%) are FILLED, not
# interpolated over: bootstrapping across the gap bends the forwards with the curve's hump
# (106 forward segments jumped > 15bp and reversed next day, mostly on such days). The fill
# (infra.analytics.swap_curve.fill_missing_tenors) misses observed tenors by 0.5-0.6bp
# out of sample, unbiased, against 0.7-7bp biased for the bare line (2026-10-06).
# Foreign OIS curves (added 2026-10-07), from DTCC closes, all free:
# * EUR: the 16:15 London close; short end from the ESR strip (3M €STR futures, IMM quarters)
#   of the PREVIOUS settlement day (point in time - ESR's settlement time vs 16:15 London is
#   not verified, TOFIX), 9 months.
# * GBP: the 16:15 London close; short end from the Bank of England's own SONIA curve of the
#   previous day (published by the next morning), up to 3 months before the first pillar -
#   the 1y SONIA close is missing on ~60% of days, and SONIA futures are not fetched.
# * JPY (Tokyo 15:00) and CAD (Toronto 15:00): no short-end source (no TONA / CORRA futures
#   on Databento), so a flat forward to the first pillar.
# Thin markets (lone prints: CAD 47% of closes, EUR 41%, GBP 24%, JPY 9%), so: wider close
# windows (SWAP_CLOSES fallback_by_currency), a 4-tenor minimum, and FILLED end tenors and
# runs of missing tenors (fill_ends). Checked 2026-10-07 over 2024-09..2026-10: zero rates
# move 3.8-5.2bp a day (sd) at every tenor in EUR / GBP, 1.6-4.4bp in JPY, 5.4-7.5bp in CAD
# (USD 4.1-5.1bp); fills miss a hidden close by 0.7-2.7bp, unbiased; GBP vs the BoE's own
# curve (same day): +0.6..+2.2bp mean, sd 1.5-3.1bp; EUR's ESR strip 1y vs the 1y close
# -0.1bp mean. Days with a curve: GBP 96%, JPY 89%, EUR 86%, CAD 62%.
OIS_CURVES: dict[str, OisCurveSpec] = {
    "USD_SOFR": OisCurveSpec("USD", "NY1530"),
    "EUR_ESTR": OisCurveSpec("EUR", "LDN1615", short_end="esr_futures", short_end_months=9, min_tenors=4,
                             fill_ends=True),
    "GBP_SONIA": OisCurveSpec("GBP", "LDN1615", short_end="boe_ois", short_end_months=12, min_tenors=4,
                              fill_ends=True),
    "JPY_TONA": OisCurveSpec("JPY", "TKY1500", short_end="none", min_tenors=4, fill_ends=True),
    "CAD_CORRA": OisCurveSpec("CAD", "TOR1500", short_end="none", min_tenors=4, fill_ends=True),
}

# Swap spreads (Derived/SwapSpreads, infra.pipeline.swap_spreads; bmk P&L
# infra.cycle.bmk_swap_spreads), tickers US_SWSP_<t>y at the CMT tenors, two SOURCES side by
# side like the yield bmk's: "cmt" = the OIS curve's par swap rate minus the CMT par yield
# (plain difference, the market's quote convention; the bond-basis conversion is a column);
# "otr" = minus the on-the-run bond's PAR-PAR asset-swap spread over the OIS curve.
SWAP_SPREAD_CURVE = "USD_SOFR"
SWAP_SPREAD_TENORS = (2, 3, 5, 7, 10, 20, 30)
SWAP_SPREAD_SOURCES = ("cmt", "otr")
SWAP_SPREAD_OTR_RANKS = (0, 1)  # the 1-old too: the held-bond P&L diffs the previous day's bond across a roll
SWAP_SPREAD_MAX_GAP_DAYS = 7
SWAP_SPREAD_AGREE_BP = 3.0  # the two sources' daily moves disagreeing by more than this are listed (warn)


@dataclass(frozen=True)
class SwaptionSpec:
    """Swaptions read from the DTCC archive (infra.processing.dtcc_swaptions,
    infra.pipeline.swaptions): which records, the OIS curve their vols are implied on, the
    print filters and the standard surface points.

    Found in the 2-year survey (2026-10-06, USD): ~234 new trades a day; the report's
    Call/Put label does NOT fix payer vs receiver (for both labels the premium only fits
    the OUT-of-the-money reading), so a print's vol is implied as the OTM option - which is
    also why the open-interest ledger keeps no payer/receiver side (gamma is the same for
    both); ~10% of trades (26% of notional) are CAPPED (``250,000,000+``: counted at the
    floor, flagged); ~0.1% of notionals are garbage (1e14) - dropped above
    ``max_notional``; 43% are package legs (premium often 0 - no vol, still open interest);
    vols from expiries beyond ~2y come out implausibly high (220-260bp normal: a premium
    convention not yet understood), so the surface stops at ``surface_max_expiry_years``."""
    fisn_pattern: str  # regex on ``UPI FISN``
    curve: str  # an OIS_CURVES name
    max_notional: float = 20e9
    min_expiry_days: int = 7
    min_tenor_years: float = 0.4
    fresh_days: int = 1  # a NEWT TRAD counts as a new execution if disseminated within this many days of it
    correction_days: int = 10  # a print's day reads corrections disseminated this many days later (SWAP_CORRECTION_DAYS)
    atm_band_bp: float = 25.0
    surface_expiries: tuple = (("1m", 0.05, 0.14), ("3m", 0.17, 0.33), ("6m", 0.42, 0.58), ("1y", 0.83, 1.17),
                               ("2y", 1.75, 2.25))
    surface_tenors: tuple = (2, 5, 10, 30)
    surface_max_expiry_years: float = 2.25
    # each print's forward moved from the curve's snap to its trade time by the hedge
    # futures' quote move (infra.pipeline.swap_hedge, as the adjusted swap closes do);
    # a print without a quote keeps the snap forward, flagged ``forward_adjusted = False``
    adjust_forward: bool = True
    # surface quality (found 2026-10-06): identical-term Call+Put pairs (straddles) are left
    # out - in about half of them each leg reports the WHOLE straddle premium, which reads as
    # ~2x the vol (pair vol / same-day singles: median 1.05, 75th pct 1.95); a point needs
    # ``surface_min_prints`` prints and is flagged ``suspect`` when more than
    # ``suspect_bp`` from the median of its previous ``suspect_window`` values (point in
    # time). Clean 1m x 10y: 343 days, daily change median 3.9bp, p90 13bp (was 75bp).
    surface_min_prints: int = 2
    suspect_bp: float = 30.0
    suspect_window: int = 10


SWAPTIONS: dict[str, SwaptionSpec] = {
    "USD_SOFR": SwaptionSpec(r"^NA/O (Call|P) Epn OIS USD$", "USD_SOFR"),
}


@dataclass(frozen=True)
class VrpSpec:
    """Volatility risk premium = implied minus realised vol of the same rate
    (infra.analytics.vrp, infra.pipeline.vrp, store ``Derived/VolRiskPremium``).

    Implied: the swaption ATM surface (``SwaptionVols``, normal bp/yr) at ``swaption_points``,
    and the futures options' ATM vol at ``futures_horizon_days`` (``infra.pipeline.
    futures_iv``; lognormal x future = points/yr). Realised, in the same units:
    * EX ANTE (known on the day): trailing ``windows`` (trading days) and an EWMA
      (``ewma_lambda``) of the daily changes of the CONSTANT-MATURITY forward swap rate
      (start = day + expiry, the tenor's length) on each day's OIS curve, or of the futures'
      front contract (most open interest the day before);
    * EX POST (known only at the option's expiry): realised over the option's own life
      on its FIXED underlying (the forward swap with the option's start and end dates; the
      futures over the horizon) - ``life_end`` says when it became known."""
    swaption_points: tuple = (("1m", 1, 2), ("3m", 3, 2), ("1m", 1, 5), ("3m", 3, 5), ("1m", 1, 10), ("3m", 3, 10),
                              ("1m", 1, 30), ("3m", 3, 30))  # (surface expiry, months, tenor years)
    futures: tuple = ("ZN",)
    futures_horizon_days: int = 30
    windows: tuple = (21, 63)
    ewma_lambda: float = 0.94
    trading_days: int = 252
    min_obs: int = 15  # fewest daily changes behind a realised vol


VRP = VrpSpec()
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
    # per-currency override of the pure fallback window, ((currency, minutes), ...): the
    # non-USD OIS markets are thinner and trade all day (2026-10-07: EUR at LDN1615 +-60 had a
    # 5y close on 63% of days, 10y 57%, 30y 34%; a wider window doesn't raise day-to-day
    # noise - the inflation and cross-currency tests). USD keeps its calibrated windows.
    fallback_by_currency: tuple = ()

    def for_currency(self, currency: str) -> "SwapCloseSpec":
        m = dict(self.fallback_by_currency).get(currency)
        return self if m is None else replace(self, pure_fallback_half_window_min=m)


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
        fallback_by_currency=(("EUR", 240), ("GBP", 240)),
    ),
    # added 2026-10-07 for the foreign OIS curves: TONA trades in Tokyo hours (UTC 0-1 and
    # 4-5 hold ~65% of prints), so its close is Tokyo's 15:00 (06:00 UTC), +-6h = the whole UTC
    # morning; CORRA trades 12-19 UTC, its close Toronto's 15:00 with the widest window inside
    # the UTC day (+-225: 88-90% of days at 2-10y, 10y daily-change sd 4.4bp; a noon snap
    # covered 1y / 30y better but was noisier at 10y)
    "TKY1500": SwapCloseSpec("15:00", "Asia/Tokyo", ("JPY",), "Tokyo close (TONA OIS)",
                             source="DTCC print timing, 2026-10-07", fallback_by_currency=(("JPY", 360),)),
    "TOR1500": SwapCloseSpec("15:00", "America/Toronto", ("CAD",), "Toronto close (CORRA OIS)",
                             source="DTCC print timing, 2026-10-07", fallback_by_currency=(("CAD", 225),)),
}

# Futures snaps (CLAUDE.md 23): per snap instant and contract, the last two-sided bbo-1m
# quote at or before the instant (at most FUTURES_SNAP_TOLERANCE_MIN old), plus that day's
# settlement for comparison. The instants are SWAP_CLOSES' (one place for every benchmark
# time). BASIS_SNAP is the instant the futures basis is computed at: FedInvest's END OF DAY
# cash is a 15:30 New York snapshot - verified 2026-10-02, the CTD's daily gross-basis
# noise against futures at each minute has a sharp minimum exactly at 15:30 (ZN 0.49/32
# vs 2.06 at 15:00 and 1.68 at 16:00), so the street's 15:00 convention (settlement vs
# 15:00 cash) would mismatch our two legs by 30 minutes.
FUTURES_SNAPS_DIR = DERIVED_ROOT / "FuturesSnaps"
FUTURES_SNAPS: tuple[str, ...] = ("NY1500", "NY1530", "NY1600", "LDN1615")
FUTURES_SNAP_TOLERANCE_MIN = 5
BASIS_SNAP = "NY1530"

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
    "FGBX": DailyBackfillSpec(2, strict_ranks=1),  # added 2026-10-07 (settlements ~$0.005 since 2025-03)
    # ICE excluded for now: ~99% of the daily cycle's API cost (2026-09-28 cost check).
    "SO3": DailyBackfillSpec(0, enabled=False),
    "R": DailyBackfillSpec(0, enabled=False),
}


# Bad-print rule extensions for FUTURES settlements (infra.cycle.bad_prints, CLAUDE.md 12),
# added 2026-10-02 after the 2018-2025 SR1 backfill showed the base rule (calibrated on
# 2025-26, no shocks) blanking genuine moves: the 2020-03 emergency cut and crash days and
# the 2022-02-10 CPI print. Each part switches off or changes here:
#   * MARKET SCALING - on a day the whole market moves, a bigger off-peer move is
#     tolerated: both thresholds are divided by k = the median |z| that day across the
#     OTHER curves of the same currency (USD STIR judged against the Treasury futures, EUR
#     against the Bund complex), floored at ``market_scaling_floor`` so a quiet day is
#     judged exactly as before. Other curves, because a bad print on a short curve (ESR: 6
#     contracts) inflates its own curve's median enough to excuse itself (2026-09-11). Not
#     VIX: equity vol, and it FELL 7 points on 2020-03-10. Fewer than
#     ``market_scaling_min_instruments`` other instruments that day -> no scaling.
#   * EVENT EXEMPTIONS - on a policy decision's first settlement, the listed roots' front
#     contracts reprice alone BY DESIGN (2020-03-03: Treasuries moved only 2.4x normal), so
#     no market factor can excuse them; such candidates are logged, never treated.
#     ``exempt_events``: event name (infra.cycle.bad_prints.EXEMPTION_EVENTS) -> roots.
@dataclass(frozen=True)
class BadPrintRules:
    market_scaling: bool = True
    market_scaling_floor: float = 1.0
    market_scaling_min_instruments: int = 3
    exempt_events: tuple[tuple[str, tuple[str, ...]], ...] = (("fomc", ("ZQ", "SR1", "SR3")),)


BAD_PRINT_RULES = BadPrintRules()

# ------------------------------------------------------------------ cash-bond curves
# Daily constant-maturity PAR yields per sovereign, stored as absolute tickers
# ``<country>_BOND_<tenor>y`` (e.g. "US_BOND_10y"), in percent. Each curve has its own
# free official source (infra/api/<source>_client.py) - no Databento, no cost. EVERY
# curve is on one basis - par with SEMI-ANNUAL coupons (user decision 2026-09-30): the
# US as published (the Treasury's own basis), UK and DE derived so from their zero curves.
@dataclass(frozen=True)
class BondCurve:
    country: str  # ticker prefix, e.g. "US"
    source: str  # fetcher in infra.pipeline.bonds.SOURCES: "treasury" | "boe" | "bundesbank" | "mof" | "boc"
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
    # is "NA" (dropped) or "roll" (last good value carried forward), CLAUDE.md 12; "off" logs
    # candidates without treating them.
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
                  history_start="1979-01-02", par_method="semiannual_from_spot",  # the BoE archive's first year (was 2016)
                  # one peer-rule candidate 1979-2026, the 2y on 1992-09-16 - Black Wednesday (sterling's
                  # ERM exit), a real move; the BoE series is a fitted official curve. Logged, never treated.
                  bad_print_policy="off"),
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
        # Japan Ministry of Finance constant-maturity JGB yields (added 2026-10-07): semi-
        # annual compound, from the JSDA reference prices at the 15:00 Tokyo close, released
        # 09:30 Tokyo the next business day - the JGB counterpart of CMT (MoF doesn't say
        # "par" in so many words). Published, so no par derivation. 40y from 2007.
        BondCurve("JP", "mof", "JGB constant-maturity curve (Ministry of Finance)", "JPY",
                  convention="semi-annual compound, constant maturity (MoF), JSDA 15:00 Tokyo prices",
                  tenors=(2, 3, 5, 7, 10, 20, 30, 40), history_start="1990-01-04",
                  # the peer rule misfires on JGBs (checked 2026-10-07: all 8 candidates 1990-2026 were
                  # real moves - the 2003 VaR shock, March 2016 after negative rates, the Dec 2022 10y
                  # kink after the BoJ band change, April 2025): at near-zero rates a tenor's typical
                  # move is a fraction of a bp and z-scores explode; and MoF's series is itself a fitted
                  # official curve. Logged, never treated.
                  bad_print_policy="off"),
        # Bank of Canada BENCHMARK bond yields (added 2026-10-07): mid-market closing yields of
        # the benchmark GoC bond near each term, 2 decimals, from 2001 - not a fitted curve: a
        # benchmark switch (after the new bond's last auction) is a jump in the series; "long"
        # (currently the 2057 bond) is stored as the 30y. The Bank's fitted zero curve is
        # published weekly with a two-week lag - history only (TOFIX).
        BondCurve("CA", "boc", "Government of Canada benchmark bond yields (Bank of Canada)", "CAD",
                  convention="yield of the benchmark bond per term (a real bond, not a fitted par point), semi-annual "
                             "compounding (GoC bonds pay semi-annual coupons; market convention, not stated by the BoC), "
                             "mid-market close, 2 decimals",
                  tenors=(2, 3, 5, 7, 10, 30), history_start="2001-01-02",
                  # the peer rule's only candidates 2001-2026 were the 2y / 3y on 2021-10-27/28, the
                  # BoC's end of QE with earlier hikes (2y +21bp, held): a policy-day front-end move,
                  # and no BoC calendar is wired into the exemptions. Logged, never treated.
                  bad_print_policy="off"),
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
    # CBOT Treasury futures options are O-PREFIXED (verified 2026-10-02: OZN.OPT resolves to
    # 6,706 instruments; ZN.OPT is rejected 422) - unlike SR3.OPT.
    "OZN.OPT": "GLBX.MDP3",
}


@dataclass(frozen=True)
class FuturesOptionsIVSpec:
    """Which options feed a root's implied vol (infra/pipeline/futures_options_iv.py): the
    ``n_expiries`` nearest expiries >= ``min_days`` away, the ATM strike + ``n_strikes``
    each side, out-of-the-money side only; definitions snapshotted every ``snapshot_days``
    on a fixed grid (each day's selection uses the snapshots before AND after it, so a
    strike listed mid-interval is still found unless it also expired within it)."""
    parent: str
    root: str
    n_expiries: int = 2
    n_strikes: int = 2
    min_days: int = 5
    snapshot_days: int = 30


# ZN only as the first pass (user decision 2026-10-02): a 1-factor level-vol scaling for the
# basis models' add-on IV (infra/models/basis/CLAUDE.md).
FUTURES_OPTIONS_IV = {"ZN": FuturesOptionsIVSpec("OZN.OPT", "ZN")}

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
    scheduled: bool = True  # False for an unscheduled (emergency) meeting - never anticipated in advance


# Verified against https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (2021-2027)
# and .../fomchistorical{2018,2019,2020}.htm, 2026-10-02 (2025-2026 first verified
# 2026-09-28). ``end_date`` is the ANNOUNCEMENT day: a rate change takes effect the next
# day. The Fed publishes next year's calendar around July of the prior year - re-check
# that page (and re-verify existing dates) when extending this list.
# Excluded on purpose: 2025-08-22 was a "notation vote only" (no live meeting / rate
# decision) - treating it as a regular meeting would corrupt the August 2025
# month-to-meeting mapping; likewise conference calls without a rate decision. Included
# as unscheduled: 2020-03-03 (the "March 2" call, cut announced March 3) and Sunday
# 2020-03-15 (which replaced the scheduled March 17-18 meeting).
FOMC_MEETINGS: tuple[FOMCMeeting, ...] = (
    FOMCMeeting("2018-01-30", "2018-01-31", False),
    FOMCMeeting("2018-03-20", "2018-03-21", True),
    FOMCMeeting("2018-05-01", "2018-05-02", False),
    FOMCMeeting("2018-06-12", "2018-06-13", True),
    FOMCMeeting("2018-07-31", "2018-08-01", False),
    FOMCMeeting("2018-09-25", "2018-09-26", True),
    FOMCMeeting("2018-11-07", "2018-11-08", False),
    FOMCMeeting("2018-12-18", "2018-12-19", True),
    FOMCMeeting("2019-01-29", "2019-01-30", False),
    FOMCMeeting("2019-03-19", "2019-03-20", True),
    FOMCMeeting("2019-04-30", "2019-05-01", False),
    FOMCMeeting("2019-06-18", "2019-06-19", True),
    FOMCMeeting("2019-07-30", "2019-07-31", False),
    FOMCMeeting("2019-09-17", "2019-09-18", True),
    FOMCMeeting("2019-10-29", "2019-10-30", False),
    FOMCMeeting("2019-12-10", "2019-12-11", True),
    FOMCMeeting("2020-01-28", "2020-01-29", False),
    FOMCMeeting("2020-03-03", "2020-03-03", False, scheduled=False),
    FOMCMeeting("2020-03-15", "2020-03-15", False, scheduled=False),
    FOMCMeeting("2020-04-28", "2020-04-29", False),
    FOMCMeeting("2020-06-09", "2020-06-10", True),
    FOMCMeeting("2020-07-28", "2020-07-29", False),
    FOMCMeeting("2020-09-15", "2020-09-16", True),
    FOMCMeeting("2020-11-04", "2020-11-05", False),
    FOMCMeeting("2020-12-15", "2020-12-16", True),
    FOMCMeeting("2021-01-26", "2021-01-27", False),
    FOMCMeeting("2021-03-16", "2021-03-17", True),
    FOMCMeeting("2021-04-27", "2021-04-28", False),
    FOMCMeeting("2021-06-15", "2021-06-16", True),
    FOMCMeeting("2021-07-27", "2021-07-28", False),
    FOMCMeeting("2021-09-21", "2021-09-22", True),
    FOMCMeeting("2021-11-02", "2021-11-03", False),
    FOMCMeeting("2021-12-14", "2021-12-15", True),
    FOMCMeeting("2022-01-25", "2022-01-26", False),
    FOMCMeeting("2022-03-15", "2022-03-16", True),
    FOMCMeeting("2022-05-03", "2022-05-04", False),
    FOMCMeeting("2022-06-14", "2022-06-15", True),
    FOMCMeeting("2022-07-26", "2022-07-27", False),
    FOMCMeeting("2022-09-20", "2022-09-21", True),
    FOMCMeeting("2022-11-01", "2022-11-02", False),
    FOMCMeeting("2022-12-13", "2022-12-14", True),
    FOMCMeeting("2023-01-31", "2023-02-01", False),
    FOMCMeeting("2023-03-21", "2023-03-22", True),
    FOMCMeeting("2023-05-02", "2023-05-03", False),
    FOMCMeeting("2023-06-13", "2023-06-14", True),
    FOMCMeeting("2023-07-25", "2023-07-26", False),
    FOMCMeeting("2023-09-19", "2023-09-20", True),
    FOMCMeeting("2023-10-31", "2023-11-01", False),
    FOMCMeeting("2023-12-12", "2023-12-13", True),
    FOMCMeeting("2024-01-30", "2024-01-31", False),
    FOMCMeeting("2024-03-19", "2024-03-20", True),
    FOMCMeeting("2024-04-30", "2024-05-01", False),
    FOMCMeeting("2024-06-11", "2024-06-12", True),
    FOMCMeeting("2024-07-30", "2024-07-31", False),
    FOMCMeeting("2024-09-17", "2024-09-18", True),
    FOMCMeeting("2024-11-06", "2024-11-07", False),
    FOMCMeeting("2024-12-17", "2024-12-18", True),
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
    FOMCMeeting("2027-01-26", "2027-01-27", False),
    FOMCMeeting("2027-03-16", "2027-03-17", True),
    FOMCMeeting("2027-04-27", "2027-04-28", False),
    FOMCMeeting("2027-06-08", "2027-06-09", True),
    FOMCMeeting("2027-07-27", "2027-07-28", False),
    FOMCMeeting("2027-09-14", "2027-09-15", True),
    FOMCMeeting("2027-10-26", "2027-10-27", False),
    FOMCMeeting("2027-12-07", "2027-12-08", True),
)


@dataclass(frozen=True)
class CentralBankMeeting:
    """One policy DECISION of a central bank other than the Fed (the Fed: ``FOMC_MEETINGS``)."""
    decision: str  # "YYYY-MM-DD", the day the decision is announced
    published: str  # "YYYY-MM-DD", when this date was first published (point-in-time known_from)
    scheduled: bool = True  # False = unscheduled (known only on the day)
    time_local: str | None = None  # announcement time if it differs from the registry's (bank's zone)
    note: str = ""


def _cb(year_dates: dict[str, tuple[str, tuple[str, ...]]], time_by_date=lambda d: None):
    """{year: (published, (MM-DD, ...))} -> meetings."""
    return tuple(CentralBankMeeting(f"{y}-{md}", pub, True, time_by_date(f"{y}-{md}"))
                 for y, (pub, mds) in year_dates.items() for md in mds)


# ECB monetary policy decisions (day 2 of each 2-day meeting), verified 2026-10-05: each
# year's dates from the ECB's "indicative calendars ... reserve maintenance periods" press
# release (which lists the Governing Council's monetary policy meetings; ``published`` = that
# release's date: pr160914 for 2017-18, 11 Jul 2018, 9 Aug 2019, 10 Jun 2020, 23 Jul 2021,
# 18 Jul 2022 (pr220718), 15 Sep 2023 (pr230915), 19 Jul 2024 (pr240719_1), 24 Apr 2025
# (pr250424), 30 Jun 2026 (pr260630)), cross-checked against the monetary policy meeting
# ACCOUNTS ("Meeting of 21-22 July 2021" ...), the decision press releases (2019, 2022) and
# the ECB's own Governing Council calendar (2026-27). Times: 13:45 CET up to the 9 Jun 2022
# meeting, 14:15 from 21 Jul 2022 (ECB press release 27 Jun 2022); the registry carries 14:15.
# Unscheduled: 18 Mar 2020 (PEPP, an evening Governing Council meeting - time not verified)
# and 15 Jun 2022 (ad hoc meeting on fragmentation; no rate decision - kept as a policy event).
ECB_MEETINGS: tuple[CentralBankMeeting, ...] = _cb({
    "2018": ("2016-09-14", ("01-25", "03-08", "04-26", "06-14", "07-26", "09-13", "10-25", "12-13")),
    "2019": ("2018-07-11", ("01-24", "03-07", "04-10", "06-06", "07-25", "09-12", "10-24", "12-12")),
    "2020": ("2019-08-09", ("01-23", "03-12", "04-30", "06-04", "07-16", "09-10", "10-29", "12-10")),
    "2021": ("2020-06-10", ("01-21", "03-11", "04-22", "06-10", "07-22", "09-09", "10-28", "12-16")),
    "2022": ("2021-07-23", ("02-03", "03-10", "04-14", "06-09", "07-21", "09-08", "10-27", "12-15")),
    "2023": ("2022-07-18", ("02-02", "03-16", "05-04", "06-15", "07-27", "09-14", "10-26", "12-14")),
    "2024": ("2023-09-15", ("01-25", "03-07", "04-11", "06-06", "07-18", "09-12", "10-17", "12-12")),
    "2025": ("2024-07-19", ("01-30", "03-06", "04-17", "06-05", "07-24", "09-11", "10-30", "12-18")),
    "2026": ("2025-04-24", ("02-05", "03-19", "04-30", "06-11", "07-23", "09-10", "10-29", "12-17")),
    "2027": ("2026-06-30", ("02-04", "03-18", "04-29", "06-10", "07-22", "09-09", "10-28", "12-16")),
}, time_by_date=lambda d: "13:45" if d <= "2022-06-09" else None) + (
    CentralBankMeeting("2020-03-18", "2020-03-18", False, None, "PEPP announced after an evening meeting"),
    CentralBankMeeting("2022-06-15", "2022-06-15", False, None, "ad hoc meeting on fragmentation, no rate change"),
)

# Bank of England MPC announcements (12:00 UK), verified 2026-10-05 against the Bank's
# annual notices "Monetary Policy Committee dates for <year>" (``published`` = the notice's
# date: 19 Oct 2017 for 2018 - the updated 2018 notice that confirmed 1 November -, 13 Sep
# 2018, 19 Sep 2019, 17 Sep 2020, 23 Sep 2021, 22 Sep 2022, 21 Sep 2023, 19 Sep 2024, 18 Sep
# 2025; 2027 from the "upcoming MPC dates" page, last updated 21 Sep 2026 - taken as known
# from then). September 2022 moved from the 15th to the 22nd for the Queen's death (notice
# of 9 Sep 2022, "announced at 12pm on 22 September"). Unscheduled: 11 Mar 2020 (special
# meeting ending 10 Mar, cut to 0.25%, announced with the Bank's Covid package) and 19 Mar
# 2020 (special meeting, cut to 0.1%) - times not verified.
BOE_MEETINGS: tuple[CentralBankMeeting, ...] = _cb({
    "2018": ("2017-10-19", ("02-08", "03-22", "05-10", "06-21", "08-02", "09-13", "11-01", "12-20")),
    "2019": ("2018-09-13", ("02-07", "03-21", "05-02", "06-20", "08-01", "09-19", "11-07", "12-19")),
    "2020": ("2019-09-19", ("01-30", "03-26", "05-07", "06-18", "08-06", "09-17", "11-05", "12-17")),
    "2021": ("2020-09-17", ("02-04", "03-18", "05-06", "06-24", "08-05", "09-23", "11-04", "12-16")),
    "2022": ("2021-09-23", ("02-03", "03-17", "05-05", "06-16", "08-04", "11-03", "12-15")),
    "2023": ("2022-09-22", ("02-02", "03-23", "05-11", "06-22", "08-03", "09-21", "11-02", "12-14")),
    "2024": ("2023-09-21", ("02-01", "03-21", "05-09", "06-20", "08-01", "09-19", "11-07", "12-19")),
    "2025": ("2024-09-19", ("02-06", "03-20", "05-08", "06-19", "08-07", "09-18", "11-06", "12-18")),
    "2026": ("2025-09-18", ("02-05", "03-19", "04-30", "06-18", "07-30", "09-17", "11-05", "12-17")),
    "2027": ("2026-09-21", ("02-04", "03-18", "04-29", "06-17", "07-29", "09-16", "11-04", "12-16")),
}) + (
    CentralBankMeeting("2022-09-22", "2022-09-09", True, None, "moved from 15 Sep (national mourning)"),
    CentralBankMeeting("2020-03-11", "2020-03-11", False, None, "special meeting ending 10 Mar: cut to 0.25%"),
    CentralBankMeeting("2020-03-19", "2020-03-19", False, None, "special meeting: cut to 0.1%"),
)
CENTRAL_BANK_MEETINGS: dict[str, tuple[CentralBankMeeting, ...]] = {
    "EA_ECB_DECISION": ECB_MEETINGS, "GB_BOE_DECISION": BOE_MEETINGS,
}

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
    """A row of the user's release table: HOW the nowcast uses a series. WHAT the series is
    and where it comes from (source, stored id, units derivation, frequency, calendar name)
    lives in its registry entry, ``series`` (infra.reference.events) - the properties below
    read through to it, so ``release.source`` etc. keep working."""
    ticker: str  # Bloomberg ticker (the table's key) - or our own id for a free proxy
    name: str
    series: EventSeries  # its registry series (infra.reference.events.SERIES)
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

    # read-through to the registry series (one source of truth)
    @property
    def frequency(self) -> str:
        return self.series.frequency

    @property
    def source(self) -> str | None:
        return self.series.source

    @property
    def series_id(self) -> str | None:
        return self.series.store_id

    @property
    def units(self) -> str:
        return self.series.derive

    @property
    def calendar_pattern(self) -> str | None:
        return self.series.calendar_pattern

    @property
    def calendar_scale(self) -> float:
        return self.series.calendar_scale

    @property
    def available(self) -> bool:
        return self.series.available

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


def _r(ticker, name, adv, pre, fin, short, med, nostage, sign, transform, cat1, cat2, blocks, note=""):
    series = series_by_bbg(ticker)
    if series is None:
        raise KeyError(f"{ticker}: no registry series (infra.reference.events.SERIES)")
    return MacroRelease(ticker, name, series, bool(adv), bool(pre), bool(fin), bool(short), bool(med),
                        bool(nostage), sign, transform, cat1, cat2, tuple(blocks), note=note)


# Columns: ticker, name | Advanced, Preliminary, Final, ShortHistory, BbgMedian, NoStage |
# Sign, Transform | Cat1, Cat2, subcategories. Each row's series (source, stored id, units,
# frequency, calendar name) is its registry entry, by Bloomberg ticker.
_TABLE = (
    _r("GDP CQOQ Index", "US GDP chained 2012 dollars QoQ SAAR",
       1, 1, 1, 0, 1, 0, 1, "rolling_ma:4;" + _E5, _A, _H, (_HOU, _CON, _MAN, _OB),
       note="the nowcast TARGET: enters the model in native units (QoQ % SAAR), not transformed. Built from the real GDP LEVEL (GDPC1, vintages since 1991-12) rather than BEA's published growth (A191RL1Q225SBEA, vintages only since 2014-09): verified 2026-09-30 to match the published growth within 0.05pp (its 1-decimal rounding) at every vintage checked"),
    _r("PCE CRCH Index", "US personal consumption expenditures chain price index",
       0, 0, 0, 0, 1, 1, 1, "rolling_ma:12;" + _E5, _P, _H, ("Price_Retail",),
       note="table name truncated ('US Personal Consumption Expend...'); read as the PCE price index MoM per the table's Price/Price_Retail categories - confirm"),
    _r("NFP TCH Index", "US employees on nonfarm payrolls, total change",
       0, 0, 0, 0, 1, 1, 1, _E5, _A, _H, (_LAB,)),
    _r("USURTOT Index", "U-3 US unemployment rate",
       0, 0, 0, 0, 1, 1, -1, "ecdf_exp:1305", _A, _H, (_LAB,)),
    _r("MWINCHNG Index", "Merchant wholesalers inventories MoM",
       0, 1, 1, 1, 1, 0, 1, "rolling_ma:3;" + _E5, _A, _H, (_CON,),
       note='from the Census Monthly Wholesale Trade release (FRED release 290), dated when the number is first published. WHLSLRIMSA (used until 2026-10-01) is the same total in the combined Manufacturing and Trade Inventories release, a median 6 days LATER - caught by the calendar cross-check'),
    _r("SBOITOTL Index", "NFIB small business optimism",
       0, 0, 0, 0, 1, 1, 1, "deman_fix:100;" + _E5, _A, _S, (_OB,),
       note='NFIB: no official free history - archived MarketWatch calendar'),
    _r("CPI YOY Index", "US CPI urban consumers YoY NSA",
       0, 0, 0, 0, 1, 1, 1, "ecdf_exp:1305", _P, _H, ("Price_Retail",)),
    _r("FDIUFDYO Index", "US PPI final demand YoY NSA",
       0, 0, 0, 0, 1, 1, 1, _E5, _P, _H, ("Price_WholeSales",)),
    _r("INJCJC Index", "Initial jobless claims",
       0, 0, 0, 0, 1, 1, -1, "ecdf_exp:1305", _A, _H, (_LAB,)),
    _r("ADP CHNG Index", "ADP national employment report, change",
       0, 0, 0, 0, 1, 1, 1, _E5, _A, _H, (_LAB,)),
    _r("CONSSENT Index", "University of Michigan consumer sentiment",
       0, 1, 1, 0, 1, 0, 1, "ecdf_exp:1305", _A, _S, (_CON,),
       note='FRED (ALFRED) carries only the end-of-month FINAL; the mid-month PRELIMINARY comes from the archived economic calendar (infra.pipeline.releases.fred_with_calendar_prelims)'),
    _r("RSTAXAG% Index", "Adjusted retail sales less autos and gas MoM",
       0, 0, 0, 0, 1, 1, 1, "rolling_ma:3;" + _E5, _A, _H, (_CON,)),
    _r("IP CHNG Index", "US industrial production MoM",
       0, 0, 0, 0, 1, 1, 1, "rolling_ma:12;" + _E5, _A, _H, (_MAN,)),
    _r("NHSPSTOT Index", "US new privately owned housing starts",
       0, 0, 0, 0, 1, 1, 1, "ecdf_exp:1305", _A, _H, (_HOU,)),
    _r("OUTFGAF Index", "Philadelphia Fed business outlook, general activity",
       0, 0, 0, 0, 1, 1, 1, _E5, _A, _S, (_OB,)),
    _r("ETSLTOTL Index", "US existing home sales SAAR",
       0, 0, 0, 0, 1, 1, 1, "ecdf_exp:1305", _A, _H, (_HOU,),
       note='NAR: FRED keeps only 13 months - archived MarketWatch calendar (SAAR, units)'),
    _r("CFNAI Index", "Chicago Fed National Activity Index",
       0, 0, 0, 0, 1, 1, 1, "rolling_ma:3;" + _E5, _A, _S, (_CON, _OB)),
    _r("NAPMPMI Index", "ISM manufacturing PMI",
       0, 0, 0, 0, 1, 1, 1, _PMI, _A, _S, (_MAN,),
       note='ISM: proprietary, off FRED since 2016 - archived MarketWatch calendar'),
    _r("NAPMNMI Index", "ISM services PMI",
       0, 0, 0, 0, 1, 1, 1, _PMI, _A, _S, (_OB,),
       note='ISM: proprietary - archived MarketWatch calendar'),
    _r("MPMIUSMA Index", "S&P Global US manufacturing PMI",
       0, 1, 1, 0, 1, 0, 1, _PMI, _A, _S, (_MAN,),
       note='S&P Global (Markit before 2022): proprietary - archived MarketWatch calendar; flash = preliminary, final = final'),
    _r("CHPMINDX Index", "MNI Chicago business barometer",
       0, 0, 0, 0, 1, 1, 1, _PMI, _A, _S, (_OB,),
       note='MNI: proprietary - archived MarketWatch calendar'),
    _r("DGNOCHNG Index", "US durable goods new orders MoM",
       0, 1, 1, 1, 1, 0, 1, "rolling_ma:12;ecdf_no_demean_exp:130", _A, _H, (_OB,)),
    _r("RCHSINDX Index", "Richmond Fed manufacturing survey, composite",
       0, 0, 0, 0, 1, 1, 1, "ecdf_no_demean_exp:130", _A, _S, (_MAN,),
       note='not on FRED; the Richmond Fed publishes it - no client yet (TOFIX.md)'),
    _r("CONCCONF Index", "Conference Board consumer confidence",
       0, 0, 0, 0, 1, 1, 1, "ecdf_exp:1305", _A, _S, (_CON,),
       note='Conference Board: proprietary - archived MarketWatch calendar'),
    _r("MPMIUSCA Index", "S&P Global US composite PMI",
       0, 1, 1, 0, 1, 0, 1, _PMI, _A, _S, (_OB,),
       note='S&P Global: proprietary - archived MarketWatch calendar'),
    _r("MPMIUSSA Index", "S&P Global US services PMI",
       0, 1, 1, 0, 1, 0, 1, _PMI, _A, _S, (_OB,),
       note='S&P Global (Markit before 2022): proprietary - archived MarketWatch calendar; flash = preliminary, final = final'),
    _r("EMPRGBCI Index", "Empire State manufacturing survey, general conditions",
       0, 0, 0, 0, 1, 1, 1, "ecdf_no_demean_exp:130", _A, _S, (_MAN,),
       note='the SEASONALLY ADJUSTED headline - GACDINA066MNFRBNY (NSA) was used until 2026-10-01, caught by the calendar cross-check (0% match vs the published headline; scripts/validate_econ_calendar.py)'),
)

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
