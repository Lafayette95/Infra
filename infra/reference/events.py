"""Economic-event registry: WHAT each scheduled event is - static, model-agnostic facts.

Two levels, because one release carries several numbers:

* ``EconEvent`` - a scheduled publication: agency, country, kind (``release`` /
  ``auction`` / ``policy``), frequency, usual time (New York wall clock), estimate
  stages, schedule rule, and the identifiers that find it elsewhere (FRED release id).
* ``EventSeries`` - one number inside an event: Bloomberg ticker, FRED series id, units
  as published, seasonal adjustment.

WHEN events happen is DATA, not config (the release-calendar store, built from FRED's
release dates, the harvested economic calendar and agency schedules); HOW a model uses a
series stays in that model's own config (``infra.config.MACRO_RELEASES`` for the
nowcast: transform, sign, categories). ``tests/test_event_registry.py`` keeps the three
consistent (every nowcast release is a registry series with the same FRED id).

**Usual times** (``time_et``, the New York wall-clock time of the release; tz-aware
conversion happens at the pipeline layer, never here) are MEASURED, not remembered:
the modal time of each event's rows in the harvested MarketWatch calendar, 2015 onward
(2026-10-01; share of rows at that time in the comment). ``None`` = not yet verified.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EconEvent:
    id: str
    name: str
    country: str  # ISO-2
    agency: str
    kind: str  # "release" | "auction" | "policy"
    frequency: str  # "W" | "M" | "Q" | "8/yr" | "irregular"
    time_et: str | None  # "HH:MM", New York wall clock; None = not verified
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


MIN_RULE_HIT = 0.90  # a rule is only kept (and projected) if it matched at least this often


@dataclass(frozen=True)
class EventSeries:
    id: str
    event: str  # EconEvent.id
    name: str
    units: str  # as PUBLISHED by the source (not as any model transforms it)
    seasonal_adjustment: str  # "SA" | "NSA" | "SAAR" | "n/a"
    bbg_ticker: str | None = None
    fred_series_id: str | None = None
    note: str = ""


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


def event_of(series_id: str) -> EconEvent:
    return EVENTS[SERIES[series_id].event]


def series_by_bbg(ticker: str) -> EventSeries | None:
    return next((s for s in SERIES.values() if s.bbg_ticker == ticker), None)
