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
    steps: tuple[str, ...] = ()                 # feature: prep steps along its timeline ("diff", "ewm_z:20")
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


EVENT_STUDIES: dict[str, EventStudySpec] = {s.name: s for s in (
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
                   condition=ConditionSpec("fut:ZN.v.0", steps=("diff:20",), partition="rolling_tercile:504")),
    EventStudySpec("refunding_day", "Refunding statement: 08:30 -> grid end",
                   code="US_TSY_REFUNDING;GRID_END;;&0_0_&_0__&0_0_%0_0;;DEFAULT_CYCLE",
                   instruments=("ZN.v.0", "ZB.v.0", "UB.v.0")),
)}


def get_event_study_spec(spec: EventStudySpec | str | None = None, **overrides) -> EventStudySpec:
    base = EVENT_STUDIES["nfp_morning"] if spec is None else (EVENT_STUDIES[spec] if isinstance(spec, str) else spec)
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


@dataclass(frozen=True)
class FamilySpec:
    name: str
    dt_refs: tuple[str, ...]
    time_refs: tuple[str, ...] = ()
    start: LegRule = field(default_factory=LegRule)
    end: LegRule = field(default_factory=LegRule)
    cycle: str = "DEFAULT_CYCLE"
    study: EventStudySpec = field(default_factory=EventStudySpec)   # instruments, source, thresholds
    fdr_q: float = 0.10                         # Benjamini-Hochberg across the family's tests
    description: str = ""

    def codes(self) -> list[str]:
        refs = "__".join(self.dt_refs) + ";" + "__".join(self.time_refs)
        return [f"{refs};;{s}__{e};;{self.cycle}" for s, e in product(self.start.legs(), self.end.legs())]


EVENT_FAMILIES: dict[str, FamilySpec] = {f.name: f for f in (
    FamilySpec("nfp_intraday", ("US_EMPLOYMENT_SITUATION",), ("GRID_START", "GRID_END"),
               start=LegRule(times=("&",), step_lags=(-8, -4, -1, 0)),
               end=LegRule(times=("&", "%1"), step_lags=(1, 2, 4, 8, 0)),
               study=EventStudySpec(instruments=("ZT.v.0", "ZN.v.0")),
               description="NFP: entries 2h/1h/15m before and at the print, exits 15m..2h after it and at the "
                           "grid's close"),
)}
