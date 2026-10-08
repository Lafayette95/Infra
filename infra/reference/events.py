"""Economic-event registry: WHAT each scheduled event is - static, model-agnostic facts.

Two levels, because one release carries several numbers:

* ``EconEvent`` - a scheduled event: agency, country, kind (``release`` / ``auction`` /
  ``policy`` / ``treasury`` (security lifecycle) / ``futures`` (contract calendar)),
  frequency, usual LOCAL time and its IANA zone, estimate stages, schedule rule, and the
  identifiers that find it elsewhere (FRED release id).
* ``EventSeries`` - one number inside an event: Bloomberg ticker, FRED series id, units
  as published, seasonal adjustment.

WHEN events happen is DATA, not config (the release-calendar store, built from FRED's
release dates, the harvested economic calendar and agency schedules); HOW a model uses a
series stays in that model's own config (``infra.config.MACRO_RELEASES`` for the
nowcast: transform, sign, categories). ``tests/test_event_registry.py`` keeps the three
consistent (every nowcast release is a registry series with the same FRED id).

**Usual times** (``time_local`` in the event's own ``timezone``: the venue's wall clock,
the only form that stays right across DST - root CLAUDE.md 7) are MEASURED or VERIFIED,
not remembered: for US releases the modal time of each event's rows in the harvested
MarketWatch calendar, 2015 onward (2026-10-01; share of rows at that time in the comment);
for the others the source named in the entry. ``None`` = not verified, or a day-level event
(an issue date, a futures delivery day). They are converted to UTC instants at the FIRST
step that reads them (``infra.trading_calendar.snap_instants``, in
``infra.processing.release_calendar``); only UTC is ever stored or passed on.

**WHEN** an event happens is data (the release-calendar store), never here: dates come from
the sources (FRED, the economic calendar, Treasury), from config lists verified against
the official calendars (``infra.config.FOMC_MEETINGS``, ``ECB_MEETINGS``, ``BOE_MEETINGS``)
or are DERIVED from stored data (Treasury lifecycle from the auctions store, futures
calendars from the contracts table and CME's delivery rules) - see
``infra.pipeline.release_calendar.derived_schedules``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EconEvent:
    id: str
    name: str
    country: str  # ISO-2
    agency: str
    kind: str  # "release" | "auction" | "policy" | "treasury" | "futures"
    frequency: str  # "W" | "M" | "Q" | "8/yr" | "irregular"
    time_local: str | None  # "HH:MM" wall clock in ``timezone``; None = not verified / day-level
    stages: tuple[str, ...] = ("first",)  # scheduled estimates in publication order
    # Where FUTURE dates come from: "fred_release" (FRED's release-dates API, ~3 months),
    # "treasury_announcement" (auctions: announced ~1 week ahead), "rule" (``rule`` below,
    # validated), "agency_schedule" (the publisher's own schedule - not fetched yet).
    # Past dates come from FRED and/or the harvested economic calendar either way.
    schedule: str | None = None
    fred_release_id: int | None = None
    note: str = ""
    # A VALIDATED date rule (infra.processing.schedule_rules syntax) the release calendar
    # may project forward, the stage it dates ("" = the event's only one), and the evidence:
    # its hit rate against the harvested release days. scripts/validate_schedule_rules.py
    # re-scores every rule; one that drops below MIN_RULE_HIT should be revisited.
    rule: str | None = None
    rule_stage: str = ""
    rule_evidence: str = ""
    timezone: str = "America/New_York"  # IANA zone of ``time_local``


MIN_RULE_HIT = 0.90  # a rule is only kept (and projected) if it matched at least this often


@dataclass(frozen=True)
class EventSeries:
    id: str
    event: str  # EconEvent.id
    name: str
    units: str  # as PUBLISHED by the source (not as any model transforms it)
    seasonal_adjustment: str  # "SA" | "NSA" | "SAAR" | "n/a"
    bbg_ticker: str | None = None
    # WHERE the series comes from and how it is stored (set from ``_SOURCES`` below):
    store_id: str | None = None  # the stored raw ticker: a FRED id, or "MW:<ticker>" (calendar)
    note: str = ""
    frequency: str = "M"  # observation period: "Q" | "M" | "W"
    # Fetcher in infra.pipeline.releases.SOURCES ("fred", "calendar", "fred+prelims"); None =
    # no free source (kept for the record, skipped by the pipeline and the models).
    source: str | None = None
    # Stored raw series -> the series' own units, computed per vintage (so a derived value's
    # release date is its inputs'): "level" as published | "diff" | "pct" (period % change)
    # | "saar" (period % change, annualized by compounding) | "yoy" (vs a year earlier).
    derive: str = "level"
    # Its row name on the economic calendar (after infra.processing.econ_calendar.
    # normalize_report). For a calendar-sourced series it IS the data source; for a FRED-
    # sourced one a cross-check and the consensus (never data). None = no calendar row in
    # the series' own units (PPI YoY, retail ex autos AND gas). Every observed variant must
    # match, and nothing else (tests/test_econ_calendar.py::NAMES).
    calendar_pattern: str | None = None
    calendar_scale: float = 1.0  # calendar value x scale = the series' own units

    @property
    def fred_series_id(self) -> str | None:
        return self.store_id if (self.source or "").startswith("fred") else None

    @property
    def available(self) -> bool:
        return self.source is not None


# Futures roots with a contract calendar here: (root, country, exchange, exchange time zone).
# Kept here (not read from infra.config, which imports this module).
_CME_TREASURY_ROOTS = ("ZT", "ZF", "ZN", "TN", "ZB", "UB")
_FUTURES_ROOTS = (
    *((r, "US", "CME", "America/Chicago") for r in ("SR3", "SR1", "ZQ", *_CME_TREASURY_ROOTS)),
    ("ESR", "EA", "CME", "America/Chicago"),
    ("SO3", "GB", "ICE Futures Europe", "Europe/London"), ("R", "GB", "ICE Futures Europe", "Europe/London"),
    *((r, "DE", "Eurex", "Europe/Berlin") for r in ("FGBL", "FGBM", "FGBS", "FBTP")),
)

_E = EconEvent
EVENTS: dict[str, EconEvent] = {e.id: e for e in (
    # ---------------------------------------------------------------- US releases
    _E("US_GDP", "Gross Domestic Product", "US", "BEA", "release", "Q", "08:30",  # 100% of 78
       ("advance", "second", "third"), "fred_release", 53),
    _E("US_PERSONAL_INCOME", "Personal Income and Outlays", "US", "BEA", "release", "M", "08:30",  # 95% of 61
       schedule="fred_release", fred_release_id=54),
    _E("US_EMPLOYMENT_SITUATION", "Employment Situation", "US", "BLS", "release", "M", "08:30",  # 100% of 137
       schedule="fred_release", fred_release_id=50),
    _E("US_JOBLESS_CLAIMS", "Unemployment Insurance Weekly Claims", "US", "DOL", "release", "W", "08:30",  # 100% of 560
       schedule="fred_release", fred_release_id=180),
    _E("US_ADP", "ADP National Employment Report", "US", "ADP", "release", "M", "08:15",  # 99% of 133
       schedule="fred_release", fred_release_id=194),
    _E("US_CPI", "Consumer Price Index", "US", "BLS", "release", "M", "08:30",  # 100% of 46
       schedule="fred_release", fred_release_id=10),
    _E("US_PPI", "Producer Price Index", "US", "BLS", "release", "M", "08:30",
       schedule="fred_release", fred_release_id=46, note="time not measured: no table series on the calendar"),
    _E("US_RETAIL_SALES", "Advance Monthly Retail Trade", "US", "Census", "release", "M", "08:30",
       schedule="fred_release", fred_release_id=9,
       note="time from the calendar's 'Retail sales' rows; the table's ex autos & gas has no row of its own"),
    _E("US_WHOLESALE_TRADE", "Monthly Wholesale Trade", "US", "Census", "release", "M", "10:00",  # 100% of 119
       ("preliminary", "final"), "fred_release", 290,
       note="preliminary = the Advance Economic Indicators report (8:30); FRED's release 290 is the full report"),
    _E("US_INDUSTRIAL_PRODUCTION", "G.17 Industrial Production and Capacity Utilization", "US", "Federal Reserve",
       "release", "M", "09:15", schedule="fred_release", fred_release_id=13),  # 99% of 125
    _E("US_HOUSING_STARTS", "New Residential Construction", "US", "Census", "release", "M", "08:30",  # 100% of 145
       schedule="fred_release", fred_release_id=27),
    _E("US_EXISTING_HOME_SALES", "Existing-Home Sales", "US", "NAR", "release", "M", "10:00",  # 100% of 115
       schedule="agency_schedule"),
    _E("US_DURABLE_GOODS", "Advance Report on Durable Goods", "US", "Census", "release", "M", "08:30",  # 100% of 136
       ("advance", "final"), "fred_release", 95, note="final = the full M3 report (Manufacturers' Shipments, "
       "Inventories, and Orders), ~1 week later"),
    _E("US_CFNAI", "Chicago Fed National Activity Index", "US", "Chicago Fed", "release", "M", "08:30",  # 100% of 76
       schedule="fred_release", fred_release_id=219),
    _E("US_EMPIRE_STATE", "Empire State Manufacturing Survey", "US", "NY Fed", "release", "M", "08:30",  # 100% of 95
       schedule="fred_release", fred_release_id=321),
    _E("US_PHILLY_FED_MFG", "Manufacturing Business Outlook Survey", "US", "Philadelphia Fed", "release", "M",
       "08:30", schedule="fred_release", fred_release_id=351),  # 87% of 67 (10:00 x9)
    _E("US_RICHMOND_FED_MFG", "Fifth District Survey of Manufacturing Activity", "US", "Richmond Fed", "release",
       "M", None, schedule="agency_schedule", note="no source yet (TOFIX.md); time unverified"),
    _E("US_UMICH_SENTIMENT", "Surveys of Consumers", "US", "University of Michigan", "release", "M", "10:00",  # 98% of 269
       ("preliminary", "final"), "fred_release", 91, note="FRED's release dates are the FINALS only"),
    _E("US_CONFERENCE_BOARD_CONFIDENCE", "Consumer Confidence Survey", "US", "Conference Board", "release", "M",
       "10:00", schedule="rule"),  # 100% of 140
    _E("US_NFIB", "NFIB Small Business Economic Trends", "US", "NFIB", "release", "M", "06:00",  # 96% of 134
       schedule="rule"),
    _E("US_ISM_MANUFACTURING", "ISM Manufacturing Report On Business", "US", "ISM", "release", "M", "10:00",  # 100% of 135
       schedule="rule"),
    _E("US_ISM_SERVICES", "ISM Services Report On Business", "US", "ISM", "release", "M", "10:00",  # 100% of 137
       schedule="rule"),
    _E("US_SPGLOBAL_MANUFACTURING_PMI", "S&P Global US Manufacturing PMI", "US", "S&P Global", "release", "M",
       "09:45", ("flash", "final"), "agency_schedule",  # 100% of 218
       note="Markit before 2022; flash ~ the 3rd/4th week of the month, final the 1st business day after"),
    _E("US_SPGLOBAL_SERVICES_PMI", "S&P Global US Services PMI", "US", "S&P Global", "release", "M", "09:45",
       ("flash", "final"), "agency_schedule",  # 99% of 231
       note="the composite PMI is published with it - but is NOT on the MarketWatch calendar (no source)"),
    _E("US_CHICAGO_PMI", "MNI Chicago Business Barometer", "US", "MNI", "release", "M", "09:45",  # 100% of 134
       schedule="rule"),
    # ---------------------------------------------------------------- US policy
    _E("US_FOMC_DECISION", "FOMC statement", "US", "Federal Reserve", "policy", "8/yr", "14:00",
       schedule="agency_schedule", note="dates: infra.config.FOMC_MEETINGS (verified against the Fed's calendar)"),
    # ---------------------------------------------------------------- other central banks
    # Dates: infra.config.ECB_MEETINGS / BOE_MEETINGS, verified 2026-10-05 against each
    # bank's own published calendars (sources there). The time here is the CURRENT one;
    # a meeting with a different time carries its own (ECB before 21 Jul 2022: 13:45).
    _E("EA_ECB_DECISION", "ECB monetary policy decisions", "EA", "European Central Bank", "policy", "8/yr",
       "14:15", schedule="agency_schedule", timezone="Europe/Berlin",
       note="14:15 CET from 21 Jul 2022, 13:45 before (ECB press release 27 Jun 2022, 'New times for "
            "ECB's monetary policy decisions and press conference'); decided on day 2 of a 2-day meeting"),
    _E("GB_BOE_DECISION", "Bank of England MPC announcement", "GB", "Bank of England", "policy", "8/yr",
       "12:00", schedule="agency_schedule", timezone="Europe/London",
       note="12:00 UK (e.g. BoE notice 9 Sep 2022: 'announced at 12pm on 22 September')"),
    # ---------------------------------------------------------------- US Treasury debt management
    # Verified 2026-10-05: financing estimates at 3:00 PM on the Monday, the refunding
    # policy statement at 8:30 AM the following Wednesday (home.treasury.gov, "most recent
    # quarterly refunding documents"). Dated from the auctions store: the refunding
    # auctions' (Feb/May/Aug/Nov 3y/10y/30y) announcement date IS the statement's date
    # (every quarter 2023-2026, e.g. 2025-07-30 for August 2025).
    _E("US_TSY_REFUNDING", "Treasury Quarterly Refunding statement", "US", "US Treasury", "policy", "Q", "08:30",
       schedule="derived", note="derived: announcement date of the refunding auctions"),
    _E("US_TSY_BORROWING_ESTIMATES", "Treasury marketable borrowing estimates", "US", "US Treasury", "policy", "Q",
       "15:00", schedule="derived", note="derived: the Monday two days before the refunding statement"),
    # ---------------------------------------------------------------- US Treasury auctions
    # Nominal coupons only for now (user, 2026-10-01: 2y-30y notes and bonds, no TIPS/FRNs/
    # bills). Time = close of competitive bidding, verified 2026-10-01 on TreasuryDirect's
    # upcoming-auctions feed ("closingTimeCompetitive": 01:00 PM for every coupon).
    *(_E(f"US_TSY_AUCTION_{t}Y", f"US Treasury {t}-year {'bond' if t >= 20 else 'note'} auction", "US",
         "US Treasury", "auction", "M", "13:00",
         schedule="treasury_announcement",
         note="reopenings carry their REMAINING term (e.g. '9-Year 10-Month' = a 10y reopening): match on "
              "the original security term")
      for t in (2, 3, 5, 7, 10, 20, 30)),
    # ---------------------------------------------------------------- German Federal auctions
    # The Finanzagentur's auctions (2026-10-07). Bidding 08:00-11:30 Frankfurt, verified on its
    # auction-process page 2026-10-07 (the current rule; older years not verified). Dates from
    # the issuance plan (annual outlook -> quarterly updates -> the live calendar,
    # infra.pipeline.de_issuance) and the held auctions (issuance history, 1999 on);
    # syndications are not auctions and are left out. Stage = new_issue / reopening.
    *(_E(f"DE_AUCTION_{t}Y", f"German Federal {t}-year {'Schatz' if t == 2 else 'Bobl' if t == 5 else 'Bund'} auction",
         "DE", "Finanzagentur", "auction", "M", "11:30", schedule="agency_schedule", timezone="Europe/Berlin",
         note="Bunds by the ISIN's maturity segment; a multi-tenor placeholder is DE_AUCTION_LONG")
      for t in (2, 5, 7, 10, 15, 20, 30)),
    _E("DE_AUCTION_LONG", "German Federal long-end auction, line decided later (15/20/30y)", "DE", "Finanzagentur",
       "auction", "M", "11:30", schedule="agency_schedule", timezone="Europe/Berlin",
       note="a plan line naming several tenors and no ISIN; the held auction carries its own tenor"),
    _E("DE_AUCTION_GREEN", "German Federal Green bond auction", "DE", "Finanzagentur", "auction", "M", "11:30",
       schedule="agency_schedule", timezone="Europe/Berlin"),
    _E("DE_AUCTION_ILB", "German Federal inflation-linked auction", "DE", "Finanzagentur", "auction", "irregular",
       "11:30", schedule="agency_schedule", timezone="Europe/Berlin"),
    _E("DE_AUCTION_BUBILL", "German Treasury discount paper (Bubill) auction", "DE", "Finanzagentur", "auction", "W",
       "11:30", schedule="agency_schedule", timezone="Europe/Berlin"),
    # ---------------------------------------------------------------- Japanese Government Bond auctions
    # The MoF's auctions (2026-10-08): day-level (time None) - the bidding deadline isn't on the
    # MoF's English pages, so it isn't verified. Dates from the results workbooks (coupon JGBs
    # 1979 on, T-bills FY2008 on, Liquidity Enhancement 2006 on) and the monthly auction
    # calendars (2023 on, infra.pipeline.jgb_auctions). Stage = new_issue / reopening (held).
    *(_E(f"JP_AUCTION_{t}Y", f"Japan {t}-year JGB auction", "JP", "Ministry of Finance", "auction", "M", None,
         schedule="agency_schedule", timezone="Asia/Tokyo") for t in (2, 5, 10, 20, 30, 40)),
    _E("JP_AUCTION_TBILL", "Japan Treasury Discount Bill auction", "JP", "Ministry of Finance", "auction", "W", None,
       schedule="agency_schedule", timezone="Asia/Tokyo"),
    _E("JP_AUCTION_LIQ", "Japan Liquidity Enhancement Auction (off-the-run reopenings)", "JP", "Ministry of Finance",
       "auction", "M", None, schedule="agency_schedule", timezone="Asia/Tokyo"),
    _E("JP_AUCTION_ILB", "Japan 10-year inflation-indexed JGB auction", "JP", "Ministry of Finance", "auction",
       "irregular", None, schedule="agency_schedule", timezone="Asia/Tokyo"),
    _E("JP_AUCTION_GX", "Japan Climate Transition (GX) bond auction", "JP", "Ministry of Finance", "auction",
       "irregular", None, schedule="agency_schedule", timezone="Asia/Tokyo"),
    _E("JP_AUCTION_OTHER", "Japan other JGB auction (4/6-year, 15-year floating, 3-year discount; discontinued)",
       "JP", "Ministry of Finance", "auction", "irregular", None, schedule="agency_schedule", timezone="Asia/Tokyo"),
    # ---------------------------------------------------------------- Government of Canada auctions
    # The Bank of Canada's auctions (2026-10-08). Bidding deadline 12:00 Ottawa for bonds and
    # 10:30 for T-bills, from the results data itself (each auction's own deadline - 12:30 /
    # 10:30 for bonds in early years - is used for held auctions). Dates from the Valet results
    # (1998 on) and the quarterly bond auction schedule (infra.pipeline.goc_auctions).
    *(_E(f"CA_AUCTION_{t}Y", f"Government of Canada {t}-year bond auction", "CA", "Bank of Canada", "auction", "M",
         "12:00", schedule="agency_schedule", timezone="America/Toronto") for t in (2, 3, 5, 7, 10, 30)),
    _E("CA_AUCTION_RRB", "Government of Canada Real Return Bond auction (none since 2022)", "CA", "Bank of Canada",
       "auction", "Q", "12:00", schedule="agency_schedule", timezone="America/Toronto"),
    _E("CA_AUCTION_ULTRA", "Government of Canada ultra-long bond auction (2014-2022)", "CA", "Bank of Canada",
       "auction", "irregular", "12:00", schedule="agency_schedule", timezone="America/Toronto"),
    _E("CA_AUCTION_TBILL", "Government of Canada Treasury bill auction", "CA", "Bank of Canada", "auction", "W",
       "10:30", schedule="agency_schedule", timezone="America/Toronto"),
    _E("CA_AUCTION_OTHER", "Government of Canada bond auction, other term", "CA", "Bank of Canada", "auction",
       "irregular", "12:00", schedule="agency_schedule", timezone="America/Toronto"),
    # ---------------------------------------------------------------- US Treasury lifecycle
    # Day-level events (no time), derived from the auctions store: each auction's issue
    # (settlement) date - stage "new_issue" / "reopening" - and the day a new issue becomes
    # on the run (its issue date: the project's default "issue" convention, CLAUDE.md 18;
    # the market's "auction" convention is the new-issue auction itself).
    *(_E(f"US_TSY_ISSUE_{t}Y", f"US Treasury {t}-year issue (settlement)", "US", "US Treasury", "treasury", "M",
         None, schedule="derived", note="derived: auctions store issue_date; known from the announcement")
      for t in (2, 3, 5, 7, 10, 20, 30)),
    *(_E(f"US_TSY_OTR_ROLL_{t}Y", f"US Treasury {t}-year on-the-run roll", "US", "US Treasury", "treasury",
         "irregular", None, schedule="derived",
         note="derived: issue date of each new original issue; known from its announcement")
      for t in (2, 3, 5, 7, 10, 20, 30)),
    # ---------------------------------------------------------------- futures contract calendars
    # Day-level (no time), one occurrence per contract (the release-calendar row's ``stage``
    # holds the contract). Derived: last trading day = the contract's expiry in the contracts
    # table (Databento definitions) for every root; the delivery days of the CME Treasury
    # roots from CME's rules (infra.analytics.futures_basis.delivery_window); the v.0
    # (volume) roll = the first day ``<root>.v.0`` maps to a new contract.
    *(_E(f"FUT_{r}_LAST_TRADE", f"{r} futures last trading day", c, x, "futures", "irregular", None,
         schedule="derived", timezone=tz, note="derived: contracts table expiry; known from listing")
      for r, c, x, tz in _FUTURES_ROOTS),
    *(_E(f"FUT_{r}_{k}", f"{r} futures {k.lower().replace('_', ' ')} day", "US", "CME", "futures", "Q", None,
         schedule="derived", timezone="America/Chicago",
         note="derived: CME delivery rules (first notice = the business day before the first delivery day)")
      for r in _CME_TREASURY_ROOTS for k in ("FIRST_NOTICE", "FIRST_DELIVERY", "LAST_DELIVERY")),
    *(_E(f"FUT_{r}_ROLL_V0", f"{r} futures volume roll (v.0 changes contract)", "US", "CME", "futures", "Q",
         None, schedule="derived", timezone="America/Chicago",
         note="derived: stored daily relative series; the ranking uses prior days' volume, so it is known "
              "before the day")
      for r in _CME_TREASURY_ROOTS),
)}

# Validated date rules (2026-10-01, against the harvested MarketWatch CONFIRMED release
# days - unconfirmed schedule rows excluded -, months since 2018; misses inspected,
# systematic ones became overrides):
_RULES = {
    "US_ISM_MANUFACTURING": ("nth_business_day:1;jan=nth_business_day:2", "",
                             "100% of 99 months; January is the 2nd business day every year"),
    "US_ISM_SERVICES": ("nth_business_day:3;jan=nth_business_day:4", "", "98% of 100 months; January = 4th"),
    "US_SPGLOBAL_MANUFACTURING_PMI": ("nth_business_day:1", "final", "100% of 62 months (final only - the flash "
                                      "has no rule: best 68%)"),
    "US_SPGLOBAL_SERVICES_PMI": ("nth_business_day:3", "final", "100% of 70 months (final only)"),
    "US_NFIB": ("nth_weekday:2:TUE", "", "98.9% of 95 months"),
    "US_EMPIRE_STATE": ("day_or_next_business_day:15", "", "100% of 99 months"),
    "US_CONFERENCE_BOARD_CONFIDENCE": ("last_weekday:TUE;dec=none", "", "100% of 95 months; December moves "
                                       "earlier around Christmas irregularly - no rule"),
    "US_CHICAGO_PMI": ("market:last_business_day;nov=none;dec=none", "", "97.6% of 85 months; market calendar (no "
                       "release on Good Friday); Nov/Dec avoid the holidays irregularly - no rule"),
    "US_PHILLY_FED_MFG": ("nth_weekday:3:THU", "", "98.7% of 77 months; scattered one-offs (e.g. Juneteenth 2025)"),
}
EVENTS = {k: (replace(e, rule=_RULES[k][0], rule_stage=_RULES[k][1], rule_evidence=_RULES[k][2])
              if k in _RULES else e) for k, e in EVENTS.items()}

_S = EventSeries
SERIES: dict[str, EventSeries] = {s.id: s for s in (
    _S("US_GDP_QOQ_SAAR", "US_GDP", "Real GDP, % change annualized", "% q/q SAAR (derived from the level)", "SAAR",
       "GDP CQOQ Index", "GDPC1", note="FRED's level series; BEA's published growth = A191RL1Q225SBEA"),
    _S("US_PCE_PRICE_INDEX", "US_PERSONAL_INCOME", "PCE chain-type price index", "index 2017=100", "SA",
       "PCE CRCH Index", "PCEPI"),
    _S("US_NFP_LEVEL", "US_EMPLOYMENT_SITUATION", "All employees, total nonfarm", "thousands of persons", "SA",
       "NFP TCH Index", "PAYEMS", note="the table's NFP TCH is its monthly CHANGE"),
    _S("US_UNEMPLOYMENT_RATE", "US_EMPLOYMENT_SITUATION", "U-3 unemployment rate", "%", "SA", "USURTOT Index",
       "UNRATE"),
    _S("US_INITIAL_CLAIMS", "US_JOBLESS_CLAIMS", "Initial claims", "persons", "SA", "INJCJC Index", "ICSA"),
    _S("US_ADP_LEVEL", "US_ADP", "ADP total nonfarm private payroll employment", "persons", "SA", "ADP CHNG Index",
       "ADPMNUSNERSA", note="the table's ADP CHNG is its monthly CHANGE; ALFRED vintages from 2022-08 (relaunch)"),
    _S("US_CPI_NSA", "US_CPI", "CPI-U all items", "index 1982-84=100", "NSA", "CPI YOY Index", "CPIAUCNS"),
    _S("US_PPI_FINAL_DEMAND_NSA", "US_PPI", "PPI final demand", "index Nov 2009=100", "NSA", "FDIUFDYO Index",
       "PPIFID"),
    _S("US_RETAIL_EX_AUTOS_GAS", "US_RETAIL_SALES", "Retail trade and food services ex motor vehicles & parts and "
       "gasoline stations", "millions of $", "SA", "RSTAXAG% Index", "MARTSSM44W72USS"),
    _S("US_WHOLESALE_INVENTORIES", "US_WHOLESALE_TRADE", "Merchant wholesalers' inventories", "millions of $", "SA",
       "MWINCHNG Index", "I42IMSM144SCEN"),
    _S("US_INDUSTRIAL_PRODUCTION", "US_INDUSTRIAL_PRODUCTION", "Industrial production, total index",
       "index 2017=100", "SA", "IP CHNG Index", "INDPRO"),
    _S("US_HOUSING_STARTS", "US_HOUSING_STARTS", "Housing starts, total", "thousands of units, annual rate", "SAAR",
       "NHSPSTOT Index", "HOUST"),
    _S("US_EXISTING_HOME_SALES", "US_EXISTING_HOME_SALES", "Existing home sales", "units, annual rate", "SAAR",
       "ETSLTOTL Index", note="FRED keeps only 13 months (EXHOSLUSM495S); history from the archived calendar"),
    _S("US_DURABLE_GOODS_ORDERS", "US_DURABLE_GOODS", "New orders, durable goods", "millions of $", "SA",
       "DGNOCHNG Index", "DGORDER"),
    _S("US_CFNAI", "US_CFNAI", "Chicago Fed National Activity Index", "index (0 = trend growth)", "n/a",
       "CFNAI Index", "CFNAI"),
    _S("US_EMPIRE_GENERAL", "US_EMPIRE_STATE", "Current general business conditions, diffusion index", "index",
       "SA", "EMPRGBCI Index", "GACDISA066MSFRBNY"),
    _S("US_PHILLY_GENERAL", "US_PHILLY_FED_MFG", "Current general activity, diffusion index", "index", "SA",
       "OUTFGAF Index", "GACDFSA066MSFRBPHI"),
    _S("US_RICHMOND_COMPOSITE", "US_RICHMOND_FED_MFG", "Manufacturing composite index", "index", "SA",
       "RCHSINDX Index"),
    _S("US_UMICH_SENTIMENT", "US_UMICH_SENTIMENT", "Index of consumer sentiment", "index 1966Q1=100", "NSA",
       "CONSSENT Index", "UMCSENT"),
    _S("US_CB_CONFIDENCE", "US_CONFERENCE_BOARD_CONFIDENCE", "Consumer confidence index", "index 1985=100", "SA",
       "CONCCONF Index"),
    _S("US_NFIB_OPTIMISM", "US_NFIB", "Small business optimism index", "index 1986=100", "SA", "SBOITOTL Index"),
    _S("US_ISM_MANUFACTURING_PMI", "US_ISM_MANUFACTURING", "ISM manufacturing PMI", "diffusion index", "SA",
       "NAPMPMI Index"),
    _S("US_ISM_SERVICES_PMI", "US_ISM_SERVICES", "ISM services PMI", "diffusion index", "SA", "NAPMNMI Index"),
    _S("US_SPGLOBAL_MANUFACTURING_PMI", "US_SPGLOBAL_MANUFACTURING_PMI", "S&P Global US manufacturing PMI",
       "diffusion index", "SA", "MPMIUSMA Index"),
    _S("US_SPGLOBAL_SERVICES_PMI", "US_SPGLOBAL_SERVICES_PMI", "S&P Global US services business activity index",
       "diffusion index", "SA", "MPMIUSSA Index"),
    _S("US_SPGLOBAL_COMPOSITE_PMI", "US_SPGLOBAL_SERVICES_PMI", "S&P Global US composite output index",
       "diffusion index", "SA", "MPMIUSCA Index", note="not on the MarketWatch calendar: no source yet"),
    _S("US_CHICAGO_PMI", "US_CHICAGO_PMI", "MNI Chicago business barometer", "diffusion index", "SA",
       "CHPMINDX Index"),
)}


# Where each series comes from: (frequency, source, store_id, derive, calendar_pattern,
# calendar_scale). FRED ids verified 2026-09-30 against FRED's own series metadata
# (scripts/update_releases.py --verify); each choice and its evidence is in
# infra/models/CLAUDE.md section 2. Calendar patterns cover every name variant 2009-2026
# (scripts/backfill_econ_calendar.py --names), incl. the 2019/2025 shutdown markers.
_SOURCES: dict[str, tuple] = {
    "US_GDP_QOQ_SAAR": ("Q", 'fred', 'GDPC1', "saar", r"^((2nd|3rd|second|third|advance) estimate )?(gdp|gross domestic product)( revision)?( \((real annual rate|revision|first revision|second revision)\))?$", 1.0),
    "US_PCE_PRICE_INDEX": ("M", 'fred', 'PCEPI', "pct", r"^pce (price )?(index|idx m m)$", 1.0),
    "US_NFP_LEVEL": ("M", 'fred', 'PAYEMS', "diff", r"^(nonfarm payrolls|employment report)$", 0.001),
    "US_UNEMPLOYMENT_RATE": ("M", 'fred', 'UNRATE', "level", r"^unemployment rate$", 1.0),
    "US_WHOLESALE_INVENTORIES": ("M", 'fred', 'I42IMSM144SCEN', "pct", r"^wholesale inventories$", 1.0),
    "US_NFIB_OPTIMISM": ("M", 'calendar', 'MW:SBOITOTL', "level", r"^nfib\b", 1.0),
    "US_CPI_NSA": ("M", 'fred', 'CPIAUCNS', "yoy", r"^(cpi|consumer price index)( year over year| y y)$", 1.0),
    "US_PPI_FINAL_DEMAND_NSA": ("M", 'fred', 'PPIFID', "yoy", None, 1.0),
    "US_INITIAL_CLAIMS": ("W", 'fred', 'ICSA', "level", r"^(weekly )?(initial )?jobless claims( \((regular )?state program sa\))?$", 1.0),
    "US_ADP_LEVEL": ("M", 'fred', 'ADPMNUSNERSA', "diff", r"^adp (national )?(employment|jobs)( report)?$", 1.0),
    "US_UMICH_SENTIMENT": ("M", 'fred+prelims', 'UMCSENT', "level", r"^(u ?mich(igan)? )?(prelim(inary)? |final )?consumer (sentiment|survey)( index)?( \((final|preliminary|prelim|revised)\)| final| prelim(inary)?)?$", 1.0),
    "US_RETAIL_EX_AUTOS_GAS": ("M", 'fred', 'MARTSSM44W72USS', "pct", None, 1.0),
    "US_INDUSTRIAL_PRODUCTION": ("M", 'fred', 'INDPRO', "pct", r"^industrial production( m m)?$", 1.0),
    "US_HOUSING_STARTS": ("M", 'fred', 'HOUST', "level", r"^housing starts( \((saar|annual rate)\))?$", 0.001),
    "US_PHILLY_GENERAL": ("M", 'fred', 'GACDFSA066MSFRBPHI', "level", r"^philly fed( manufacturing)?( index)?$|^philadelphia fed( s)? (manufacturing|business outlook) survey$", 1.0),
    "US_EXISTING_HOME_SALES": ("M", 'calendar', 'MW:ETSLTOTL', "level", r"^existing home sales( \((annual rate|saar)\))?$", 1.0),
    "US_CFNAI": ("M", 'fred', 'CFNAI', "level", r"^chicago (fed )?national (activity )?index$|^chicago (fed )?national activity$", 1.0),
    "US_ISM_MANUFACTURING_PMI": ("M", 'calendar', 'MW:NAPMPMI', "level", r"^ism( report on business)?( manufacturing)?( index| indext| pmi)?$", 1.0),
    "US_ISM_SERVICES_PMI": ("M", 'calendar', 'MW:NAPMNMI', "level", r"^ism( report on business)? (non ?manufacturi?ng|on manufacturing|services)( index| pmi)?$", 1.0),
    "US_SPGLOBAL_MANUFACTURING_PMI": ("M", 'calendar', 'MW:MPMIUSMA', "level", r"^(?!.*(services|serivces|non ?manufacturing|composite|chicago|ism))(?=.*\bpmi\b)(?=.*(markit|market|arkit|s&p|flash|final|prelim|manufacturing)).*$|^(s&p|markit)( global)? (final|flash) manufacturing$", 1.0),
    "US_CHICAGO_PMI": ("M", 'calendar', 'MW:CHPMINDX', "level", r"^chicago (pmi|business barometer|manufacturing pmi|purchasing managers)", 1.0),
    "US_DURABLE_GOODS_ORDERS": ("M", 'fred', 'DGORDER', "pct", r"^durable goods orders$", 1.0),
    "US_RICHMOND_COMPOSITE": ("M", None, None, "level", None, 1.0),
    "US_CB_CONFIDENCE": ("M", 'calendar', 'MW:CONCCONF', "level", r"^(conference (board|bd) )?consumer confidence( index)?$", 1.0),
    "US_SPGLOBAL_COMPOSITE_PMI": ("M", 'calendar', 'MW:MPMIUSCA', "level", r"^(?!.*(chicago|ism))(?=.*\bpmi\b)(?=.*composite).*$", 1.0),
    "US_SPGLOBAL_SERVICES_PMI": ("M", 'calendar', 'MW:MPMIUSSA', "level", r"^(?!.*(composite|chicago|ism))(?=.*\bpmi\b)(?=.*(services|serivces|non ?manufacturing)).*$", 1.0),
    "US_EMPIRE_GENERAL": ("M", 'fred', 'GACDISA066MSFRBNY', "level", r"^empire state( manufacturing)?( index| survey)?$", 1.0),
}
SERIES = {k: (replace(s_, frequency=_SOURCES[k][0], source=_SOURCES[k][1], store_id=_SOURCES[k][2],
                      derive=_SOURCES[k][3], calendar_pattern=_SOURCES[k][4], calendar_scale=_SOURCES[k][5])
              if k in _SOURCES else s_) for k, s_ in SERIES.items()}


def event_of(series_id: str) -> EconEvent:
    return EVENTS[SERIES[series_id].event]


def series_by_bbg(ticker: str) -> EventSeries | None:
    return next((s for s in SERIES.values() if s.bbg_ticker == ticker), None)
