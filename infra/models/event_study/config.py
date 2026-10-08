"""Event-study parameters (a spec per study, toggled by name, infra/models/CLAUDE.md 0) and
families of studies generated from rules. Specs say HOW to study; the event code says WHICH
window (``infra.processing.event_windows``); the instruments say ON WHAT.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product


@dataclass(frozen=True)
class ConditionSpec:
    """Condition the windows on a third series (infra.models.event_study.conditions): its
    point-in-time feature at each window's start, partitioned into -1 / 0 / +1."""
    series: str                                 # a series id (infra.pipeline.series_panel), e.g. "fut:ZN.v.0"
    feature: str | None = None                  # the feature GRAMMAR along its timeline (infra.processing.features,
                                                # e.g. "chg:20", "chg:5|norm:vol:60"); None -> the legacy `steps`
    steps: tuple[str, ...] = ()                 # LEGACY: prep steps ("diff:20", "ewm_z:20") = literal grammar aliases
    period_diff: bool = False                   # vintage series: the latest period's change vs the previous
    partition: str = "rolling_tercile:252"      # PARTITIONERS rule, W in timeline rows
    lag_steps: int = 1                          # grid steps BEFORE the window start, on top of availability
    history: str = "1095D"                      # timeline read this far before the panel (partition warm-up)
    min_bucket_obs: int = 10
    did_t_min: float | None = 1.5               # |t| of the bucket's event-minus-placebo effect vs the rest's
    vs_rest_t_min: float | None = None          # |t| of the bucket's events vs the other buckets' (Welch)


@dataclass(frozen=True)
class EventStudySpec:
    name: str = "custom"
    description: str = ""
    code: str = ""                              # the event code (window definition)
    instruments: tuple[str, ...] = ("ZN.v.0",)
    source: str = "FUTURE_BPS_BBO"              # infra.pipeline.event_pnl.PNL_SOURCES
    # step 1
    max_pair_gap: int = 22                      # trading days between paired date+time events
    ignore_as_of: bool = False                  # list events as known NOW, not as of the read date
    # fit sample
    window: str | None = None                   # look back (a Timedelta, "1825D"); None = all history
    min_obs: int = 20                           # fewer completed events -> not tested (fails)
    # tests (None switches a test off); each is per instrument
    t_min: float | None = 1.5                   # |t| of the mean event move
    ev_abs_min: float | None = 2.0              # |mean| in the source's units (bp)
    ev_vol_min: float | None = None             # |mean| / placebo sd (same window, non-event days)
    hit_min: float | None = None                # share of events moving the mean's way
    placebo_t_min: float | None = None          # |t| of (event mean - placebo mean), Welch
    stable_halves: bool = False                 # both halves of the sample: same sign as the whole
    year_share_min: float | None = None         # share of years (>= 2 events) with the same sign
    trim: int = 1                               # events dropped at EACH tail for the trimmed mean
    condition: ConditionSpec | None = None      # a third series' regime (None = unconditional)
    frequency: str = "intraday"                 # intraday | daily: must match the code's cycle and the source
    # how the statistics become a decision (infra.models.fit_modes): fitted = the mean's sign where the tests
    # pass; exante = the STATED sign, no tests; prior = the fitted sign, 0 where it isn't the stated one
    fit_mode: str = "fitted"
    expected_sign: tuple[tuple[str, float], ...] = ()   # (instrument | "instrument|+1" bucket | "*", +1 / -1)
    # seasonality: only windows whose END falls in these calendar months are event windows (the others
    # are kept illegal, reason "outside_months"; the placebo baseline still spans every month)
    months: tuple[int, ...] = ()

    def __post_init__(self):
        from infra.models.fit_modes import check_fit_mode
        check_frequency(self.frequency, cycle=self.code.rsplit(";;", 1)[-1] if self.code else None,
                        source=self.source, what=f"event study {self.name!r}")
        check_fit_mode(self.fit_mode, has_direction=bool(self.expected_sign), what=f"event study {self.name!r}")

    def stated_sign(self, key: str) -> float:
        """The stated sign of a table row (an instrument, or ``instrument|<bucket>``): its own entry,
        else its instrument's, else ``*``; 0 if none."""
        d = dict(self.expected_sign)
        for k in (key, key.split("|", 1)[0], "*"):
            if k in d:
                return float(d[k])
        return 0.0


FREQUENCIES = ("intraday", "daily")


def check_frequency(frequency: str, *, cycle: str | None = None, source: str | None = None, what: str = "spec"):
    """A spec's declared frequency must match its cycle's and its P&L source's (root CLAUDE.md 29):
    a daily study can't run on the 15-minute grid or on bbo P&L, and vice versa."""
    from infra.pipeline.event_pnl import PNL_SOURCES
    from infra.reference.event_grid import resolve_cycle
    if frequency not in FREQUENCIES:
        raise ValueError(f"{what}: frequency {frequency!r} (one of {FREQUENCIES})")
    if cycle:
        cf = resolve_cycle(cycle).frequency
        if cf != frequency:
            raise ValueError(f"{what}: {frequency} but its cycle {cycle!r} is {cf}")
    if source and source in PNL_SOURCES and PNL_SOURCES[source].frequency != frequency:
        raise ValueError(f"{what}: {frequency} but its P&L source {source!r} is {PNL_SOURCES[source].frequency}")


def registry(frequency: str, specs) -> dict:
    """A frequency's registry: rejects a spec of the other frequency (the guard)."""
    out = {}
    for s in specs:
        if s.frequency != frequency:
            raise ValueError(f"{s.name!r} is {s.frequency}: it can't go in the {frequency} registry")
        if s.name in out:
            raise ValueError(f"duplicate name {s.name!r}")
        out[s.name] = s
    return out


EVENT_STUDIES_INTRADAY: dict[str, EventStudySpec] = registry("intraday", (
    EventStudySpec("nfp_morning", "NFP: from 15 minutes before to 2 hours after the print",
                   code="US_EMPLOYMENT_SITUATION;;;&0_0_&_-1__&0_0_&_8;;DEFAULT_CYCLE",
                   instruments=("ZT.v.0", "ZF.v.0", "ZN.v.0", "ZB.v.0")),
    EventStudySpec("auction_3y_to_7y", "The user's example: 30 min before the 3y auction -> 30 min after the "
                   "grid opens the day after the following 7y auction",
                   code="US_TSY_AUCTION_3Y__US_TSY_AUCTION_7Y;GRID_START;;&0_0_&_-2__&1_1_%0_2;;DEFAULT_CYCLE",
                   instruments=("ZT.v.0", "ZF.v.0", "ZN.v.0")),
    EventStudySpec("nfp_morning_by_trend", "nfp_morning conditioned on ZN's 20-day trend into the print "
                   "(change of the back-adjusted settlement over 20 days, rolling terciles over 2 years)",
                   code="US_EMPLOYMENT_SITUATION;;;&0_0_&_-1__&0_0_&_8;;DEFAULT_CYCLE",
                   instruments=("ZT.v.0", "ZN.v.0", "ZB.v.0"),
                   condition=ConditionSpec("fut:ZN.v.0", feature="chg:20", partition="rolling_tercile:504")),
    EventStudySpec("refunding_day", "Refunding statement: 08:30 -> grid end",
                   code="US_TSY_REFUNDING;GRID_END;;&0_0_&_0__&0_0_%0_0;;DEFAULT_CYCLE",
                   instruments=("ZN.v.0", "ZB.v.0", "UB.v.0")),
))

# DAILY studies (cycle DAILY_SETTLE, one point a day at the 14:00 CT settlement; daily P&L
# sources). A leg's time is "%0" (the day's only point): an event's own clock time ("&", e.g.
# 08:30 ET) lies off the daily grid and is illegal. Default instrument: the 10y yield.
DAILY = dict(frequency="daily")
EVENT_STUDIES_DAILY: dict[str, EventStudySpec] = registry("daily", (
    EventStudySpec("nfp_day", "NFP day, settlement to settlement, on the CMT par yields (bp, + = rally)",
                   code="US_EMPLOYMENT_SITUATION;GRID_START;;&0_-1_%0_0__&0_0_%0_0;;DAILY_SETTLE",
                   instruments=("US_BOND_2y", "US_BOND_5y", "US_BOND_10y", "US_BOND_30y"), source="YIELD_BPS_CMT",
                   **DAILY),
    EventStudySpec("nfp_day_futures", "the same day on the futures (settlement bp): the futures' view of it",
                   code="US_EMPLOYMENT_SITUATION;GRID_START;;&0_-1_%0_0__&0_0_%0_0;;DAILY_SETTLE",
                   instruments=("ZT.v.0", "ZF.v.0", "ZN.v.0", "TN.v.0", "ZB.v.0", "UB.v.0"),
                   source="FUTURE_BPS_SETTLE", **DAILY),
))

EVENT_STUDIES: dict[str, EventStudySpec] = {**EVENT_STUDIES_INTRADAY, **EVENT_STUDIES_DAILY}
assert len(EVENT_STUDIES) == len(EVENT_STUDIES_INTRADAY) + len(EVENT_STUDIES_DAILY), "a name in both registries"


FAMILY_PREFIX = "family:"


def family_member_name(family: str, i: int) -> str:
    """The spec name of a family's i-th code: ``family:<name>#<i>`` (a family's codes are
    ordinary single-code runs; their config stores only this name)."""
    return f"{FAMILY_PREFIX}{family}#{i}"


def get_event_study_spec(spec: EventStudySpec | str | None = None, **overrides) -> EventStudySpec:
    if isinstance(spec, str) and spec.startswith(FAMILY_PREFIX):
        fam_name, _, idx = spec[len(FAMILY_PREFIX):].rpartition("#")
        fam = EVENT_FAMILIES[fam_name]
        base = replace(fam.study, name=spec, code=fam.codes()[int(idx)])
    else:
        base = EVENT_STUDIES["nfp_morning"] if spec is None else (EVENT_STUDIES[spec] if isinstance(spec, str)
                                                                  else spec)
    return replace(base, **overrides) if overrides else base


# --------------------------------------------------------------------------- families
@dataclass(frozen=True)
class LegRule:
    """The values a family tries for one leg (cartesian product)."""
    refs: tuple[int, ...] = (0,)
    day_lags: tuple[int, ...] = (0,)
    times: tuple[str, ...] = ("&",)             # "&" (the event's own time) or "%j"
    step_lags: tuple[int, ...] = (0,)

    def legs(self) -> list[str]:
        return [f"&{r}_{d}_{t}_{s}" for r, d, t, s in product(self.refs, self.day_lags, self.times, self.step_lags)]


def leg_values(rule: LegRule | tuple[LegRule, ...]) -> list[str]:
    """A family leg: one rule (its cartesian product) or several (the UNION of their products, in
    order, duplicates dropped) - so e.g. lags can apply to the event's own time but not to the
    grid's close, where any lag leaves the grid and the code could never be legal (2026-10-05:
    16 of nfp_intraday's 40 codes were such dead codes)."""
    rules = (rule,) if isinstance(rule, LegRule) else tuple(rule)
    return list(dict.fromkeys(leg for r in rules for leg in r.legs()))


@dataclass(frozen=True)
class FamilySpec:
    name: str
    dt_refs: tuple[str, ...]
    time_refs: tuple[str, ...] = ()
    start: LegRule | tuple[LegRule, ...] = field(default_factory=LegRule)   # one rule, or several (union)
    end: LegRule | tuple[LegRule, ...] = field(default_factory=LegRule)
    cycle: str = "DEFAULT_CYCLE"
    study: EventStudySpec = field(default_factory=EventStudySpec)   # instruments, source, thresholds
    fdr_q: float = 0.10                         # Benjamini-Hochberg across the family's tests
    description: str = ""
    frequency: str = "intraday"                 # must match the cycle and the study's source / frequency

    def __post_init__(self):
        check_frequency(self.frequency, cycle=self.cycle, source=self.study.source, what=f"family {self.name!r}")
        if self.study.frequency != self.frequency:
            raise ValueError(f"family {self.name!r} is {self.frequency} but its study spec is {self.study.frequency}")

    def codes(self) -> list[str]:
        refs = "__".join(self.dt_refs) + ";" + "__".join(self.time_refs)
        return [f"{refs};;{s}__{e};;{self.cycle}" for s, e in product(leg_values(self.start), leg_values(self.end))]


EVENT_FAMILIES_INTRADAY: dict[str, FamilySpec] = registry("intraday", (
    FamilySpec("nfp_intraday", ("US_EMPLOYMENT_SITUATION",), ("GRID_START", "GRID_END"),
               start=LegRule(times=("&",), step_lags=(-8, -4, -1, 0)),
               end=(LegRule(times=("&",), step_lags=(1, 2, 4, 8, 0)), LegRule(times=("%1",), step_lags=(0,))),
               study=EventStudySpec(instruments=("ZT.v.0", "ZN.v.0")),
               description="NFP: entries 2h/1h/15m before and at the print, exits 15m..2h after it and at the "
                           "grid's close"),
    FamilySpec("nfp_intraday_struct", ("US_EMPLOYMENT_SITUATION",), ("GRID_START", "GRID_END"),
               start=LegRule(times=("&",), step_lags=(-8, -4, -1, 0)),
               end=(LegRule(times=("&",), step_lags=(1, 2, 4, 8, 0)), LegRule(times=("%1",), step_lags=(0,))),
               study=EventStudySpec(instruments=("DUR__TY", "CURVE__FV__WN", "FLY__FV__UXY__WN", "FRONT__TU__H",
                                                 "MICRO__TY__FV__H", "MICRO__US__WN__H"),
                                    source="STRUCT_BPS_BBO", ev_abs_min=None),
               description="the NFP family on the curve structures (bp per unit of structure; no absolute-size "
                           "test: a fly moving 0.8bp a day can't clear a 2bp floor set for single futures)"),
))

def _cc(c: str) -> tuple[str, ...]:
    b = f"{c}_BOND"
    return (f"{b}_10y", f"CURVE__{b}_5y__{b}_30y", f"FLY__{b}_5y__{b}_10y__{b}_30y")


# calendar-seasonality instrument sets: (tag, daily P&L source, instruments, label)
_CAL_SETS = (
    ("otr", "YIELD_BPS_OTR",
     ("US_BOND_2y", "US_BOND_5y", "US_BOND_10y", "US_BOND_30y", "CURVE__US_BOND_2y__US_BOND_10y",
      "CURVE__US_BOND_5y__US_BOND_30y", "FLY__US_BOND_2y__US_BOND_5y__US_BOND_10y",
      "FLY__US_BOND_5y__US_BOND_10y__US_BOND_30y"),
     "US on-the-run yields, curves, flies (from 2008)"),
    ("ldn", "SERIES_BPS", tuple(f"bmk:{b}@LDN1615:{i}" for b, c in (("yield_cmt", "US"), ("yield_boe", "UK"))
                                for i in _cc(c)),
     "US and UK at 16:15 London (synchronized, from 2016): 10y, 5s30s, 5s10s30s"),
)

EVENT_FAMILIES_DAILY: dict[str, FamilySpec] = registry("daily", (
    FamilySpec("nfp_days", ("US_EMPLOYMENT_SITUATION",), ("GRID_START",),
               start=LegRule(day_lags=(-3, -2, -1), times=("%0",)), end=LegRule(day_lags=(0, 1, 2), times=("%0",)),
               cycle="DAILY_SETTLE", frequency="daily",
               study=EventStudySpec(instruments=("US_BOND_2y", "US_BOND_5y", "US_BOND_10y", "US_BOND_30y"),
                                    source="YIELD_BPS_CMT", **DAILY),
               description="NFP over days: from 1-3 settlements before to 0-2 after the release, CMT yields"),
    FamilySpec("nfp_days_struct", ("US_EMPLOYMENT_SITUATION",), ("GRID_START",),
               start=LegRule(day_lags=(-3, -2, -1), times=("%0",)), end=LegRule(day_lags=(0, 1, 2), times=("%0",)),
               cycle="DAILY_SETTLE", frequency="daily",
               study=EventStudySpec(instruments=("DUR__TY", "CURVE__FV__WN", "FLY__FV__UXY__WN", "FRONT__TU__H",
                                                 "MICRO__TY__FV__H", "MICRO__US__WN__H"),
                                    source="STRUCT_BPS_SETTLE", ev_abs_min=None, **DAILY),
               description="the same day windows on the curve structures (settlement bp), tradeable via layers"),
    # ------------------------------------------------ calendar seasonality (half-months, turn of the month)
    *(FamilySpec(f"cal_{half}_{tag}", refs, ("GRID_START",),
                 start=LegRule(refs=(0,), day_lags=(-1, 0, 1), times=("%0",)),
                 end=LegRule(refs=(1,), day_lags=(-1, 0, 1), times=("%0",)),
                 cycle="DAILY_SETTLE", frequency="daily",
                 study=EventStudySpec(instruments=insts, source=src, ev_abs_min=None, **DAILY),
                 description=f"{what}, +-1 business day at each end; {label}")
      for half, refs, what in (("half1", ("CAL_MONTH_END", "CAL_MID_MONTH"), "month end -> mid-month (15th)"),
                               ("half2", ("CAL_MID_MONTH", "CAL_MONTH_END"), "mid-month -> month end"))
      for tag, src, insts, label in _CAL_SETS),
    *(FamilySpec(f"cal_turn_{tag}", ("CAL_MONTH_END",), ("GRID_START",),
                 start=LegRule(day_lags=(-5, -3, -1, 0), times=("%0",)),
                 end=LegRule(day_lags=(0, 1, 2, 3), times=("%0",)),
                 cycle="DAILY_SETTLE", frequency="daily",
                 study=EventStudySpec(instruments=insts, source=src, ev_abs_min=None, **DAILY),
                 description=f"the turn of the month: from 5 / 3 / 1 / 0 business days before month end to 0-3 "
                             f"after (index extension, coupon reinvestment); {label}")
      for tag, src, insts, label in _CAL_SETS),
))

EVENT_FAMILIES: dict[str, FamilySpec] = {**EVENT_FAMILIES_INTRADAY, **EVENT_FAMILIES_DAILY}
