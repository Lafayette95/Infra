"""Event-study windows: parse an event code, resolve it against event occurrences on a
trading grid, and build the placebo windows. Pure (frames in, frames out).

**The event code** (``parse_code`` / ``format_code``)::

    DT_REFS;TIME_REFS;;START_LEG__END_LEG;;CYCLE
    US_TSY_AUCTION_3Y__US_TSY_AUCTION_7Y;GRID_START;;&0_0_&_-2__&1_1_%0_2;;DEFAULT_CYCLE

* ``DT_REFS``: date+time events (``infra.reference.events`` ids, case-insensitive),
  separated by ``__``, indexed from 0; ``ID:stage`` keeps only occurrences of that stage
  (``US_GDP:advance``, ``US_TSY_AUCTION_10Y:new_issue``).
* ``TIME_REFS``: time-only events (``infra.reference.event_grid.TIME_EVENTS``), ``__``-
  separated, indexed from 0; may be empty.
* A LEG ``&i_D_T_L``: the day of date+time event ``i``'s occurrence, moved ``D`` trading days
  of the cycle's calendar; at time ``T`` = ``&`` (event i's own clock time) or ``%j``
  (time-only event j); moved ``L`` grid steps.
* ``CYCLE``: a cycle or alias (``DEFAULT_CYCLE``).

**Pairing** (several date+time events): each occurrence of event 0 (the anchor) pairs with
the FIRST occurrence of each other event at or after it, within ``max_gap`` trading days;
none -> the window is illegal (``no_pair``).

**Legality and snapping** (the user's interval convention, 2026-10-05): grid points run
``start`` .. ``end`` inclusive and a window is the half-open (start point, end point]: it
starts AFTER its start point, so starting at the first point (06:00) is fine, and it ends
AT its end point, so ending at the last point (19:00) is fine. A leg's reference time must
lie in [first point, last point] of its day - outside is illegal (``outside_grid``; 19:01 is
not snapped back to 19:00). Inside, it snaps to the grid point at or before it; the step lag
then moves it, and leaving the day's points is illegal (``lag_outside_grid``) - a lag never
rolls into another day. ``&`` needs a verified event time (``no_event_time`` for a day-level
event); a leg's day must be a trading day when its day lag is 0 (``not_a_trading_day``); the
end point must come after the start point (``end_not_after_start``).

**Placebo windows**: the same (day gap, start slot, end slot) pattern on trading days that
are not event days - the unconditional baseline the event's move is judged against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from infra.reference.event_grid import TIME_EVENTS, CycleSpec, resolve_cycle
from infra.reference.events import EVENTS
from infra.trading_calendar import local_wallclock, snap_instants

SEP_SECTION, SEP_REFS, SEP_LIST, SEP_FIELD = ";;", ";", "__", "_"
TIME_PRIORITY = {"source": 0, "registry": 1, "unknown": 2}
EXCLUDED_SOURCES = ("marketwatch_unconfirmed",)  # a schedule that moved or was cancelled, not an event
PROJECTION_SOURCES = ("rule",)                   # used only where no other source has the day


@dataclass(frozen=True)
class Leg:
    ref: int            # date+time event index
    day_lag: int        # trading days
    time_ref: int | None  # None = the event's own time ("&"); else the time-only event index ("%j")
    step_lag: int       # grid steps

    def encode(self) -> str:
        t = "&" if self.time_ref is None else f"%{self.time_ref}"
        return SEP_FIELD.join([f"&{self.ref}", str(self.day_lag), t, str(self.step_lag)])


@dataclass(frozen=True)
class EventCode:
    dt_refs: tuple[tuple[str, str | None], ...]   # (event id, stage or None)
    time_refs: tuple[str, ...]
    start: Leg
    end: Leg
    cycle: str

    def encode(self) -> str:
        dt = SEP_LIST.join(e if s is None else f"{e}:{s}" for e, s in self.dt_refs)
        return (f"{dt}{SEP_REFS}{SEP_LIST.join(self.time_refs)}{SEP_SECTION}"
                f"{self.start.encode()}{SEP_LIST}{self.end.encode()}{SEP_SECTION}{self.cycle}")


def _parse_leg(text: str, n_dt: int, n_time: int) -> Leg:
    f = text.split(SEP_FIELD)
    if len(f) != 4 or not f[0].startswith("&"):
        raise ValueError(f"bad leg {text!r}: expected &i_DAYLAG_(&|%j)_STEPLAG")
    ref = int(f[0][1:])
    if not 0 <= ref < n_dt:
        raise ValueError(f"leg {text!r}: date+time event &{ref} not listed (0..{n_dt - 1})")
    if f[2] == "&" or f[2] == f"&{ref}":
        time_ref = None
    elif f[2].startswith("%"):
        time_ref = int(f[2][1:])
        if not 0 <= time_ref < n_time:
            raise ValueError(f"leg {text!r}: time-only event %{time_ref} not listed (0..{n_time - 1})")
    else:
        raise ValueError(f"leg {text!r}: time must be '&' (the event's own) or '%j' (a time-only event)")
    return Leg(ref, int(f[1]), time_ref, int(f[3]))


def parse_code(code: str) -> EventCode:
    """Event code -> ``EventCode``; ids upper-cased and checked against the registries."""
    # split from the RIGHT: with no time-only events the refs end in ';' ("ID;;;LEGS;;CYCLE")
    head, sep, cycle = code.strip().rpartition(SEP_SECTION)
    refs, sep2, legs = head.rpartition(SEP_SECTION)
    if not sep or not sep2:
        raise ValueError(f"bad event code {code!r}: expected REFS;;LEGS;;CYCLE")
    if SEP_REFS not in refs:
        raise ValueError(f"bad event code {code!r}: refs need 'DT_REFS;TIME_REFS' (TIME_REFS may be empty)")
    dt_text, time_text = refs.split(SEP_REFS, 1)
    dt_refs = []
    for item in [x for x in dt_text.split(SEP_LIST) if x]:
        eid, _, stage = item.partition(":")
        eid = eid.upper()
        if eid not in EVENTS:
            raise ValueError(f"unknown date+time event {eid!r} (infra.reference.events)")
        dt_refs.append((eid, stage or None))
    time_refs = tuple(x.upper() for x in time_text.split(SEP_LIST) if x)
    for t in time_refs:
        if t not in TIME_EVENTS:
            raise ValueError(f"unknown time-only event {t!r} (infra.reference.event_grid.TIME_EVENTS)")
    if not dt_refs:
        raise ValueError(f"event code {code!r} references no date+time event")
    leg_texts = legs.split(SEP_LIST)
    if len(leg_texts) != 2:
        raise ValueError(f"bad legs {legs!r}: expected START__END")
    resolve_cycle(cycle)
    return EventCode(tuple(dt_refs), time_refs, _parse_leg(leg_texts[0], len(dt_refs), len(time_refs)),
                     _parse_leg(leg_texts[1], len(dt_refs), len(time_refs)), cycle)


def normalize_code(code: str) -> str:
    return parse_code(code).encode()


# --------------------------------------------------------------------------- occurrences
def consolidate(calendar: pd.DataFrame, *, as_of=None) -> pd.DataFrame:
    """Release-calendar rows (several sources per event day) -> ONE occurrence per (event,
    local day): the row with the best time (``source`` > ``registry`` > ``unknown``), the
    earliest ``known_from`` of all its sources, a non-empty stage when any source has one.
    Unconfirmed schedule rows are dropped; rule projections are used only for a day no other
    source lists (in practice: future days)."""
    if calendar is None or calendar.empty:
        return pd.DataFrame(columns=["event", "timestamp", "day", "stage", "time_source", "known_from"])
    cal = calendar[~calendar["source"].isin(EXCLUDED_SOURCES)].copy()
    tz = cal["event"].map(lambda e: EVENTS[e].timezone if e in EVENTS else "UTC")
    day = pd.Series(pd.NaT, index=cal.index, dtype="datetime64[ns]")
    for zone, idx in cal.groupby(tz).groups.items():
        day.loc[idx] = local_wallclock(cal.loc[idx, "timestamp"], zone).normalize()
    cal["day"] = day
    cal["proj"] = cal["source"].isin(PROJECTION_SOURCES)
    cal["prio"] = cal["time_source"].map(TIME_PRIORITY).fillna(3)
    has_real = cal.groupby(["event", "day"])["proj"].transform(lambda s: (~s).any())
    cal = cal[~(cal["proj"] & has_real)]
    cal = cal.sort_values(["event", "day", "prio", "timestamp"])
    best = cal.groupby(["event", "day"], as_index=False).first()
    best["known_from"] = cal.groupby(["event", "day"])["known_from"].min().to_numpy()
    stages = cal[cal["stage"].fillna("").astype(str) != ""].groupby(["event", "day"])["stage"].first()
    best = best.set_index(["event", "day"])
    best["stage"] = stages.reindex(best.index).fillna(best["stage"].fillna("")).astype(str)
    out = best.reset_index()[["event", "timestamp", "day", "stage", "time_source", "known_from"]]
    if as_of is not None:
        out = out[out["known_from"] <= pd.Timestamp(as_of)]
    return out.sort_values(["event", "timestamp"]).reset_index(drop=True)


# --------------------------------------------------------------------------- grid
@dataclass
class Grid:
    cycle: CycleSpec
    days: pd.DatetimeIndex          # trading days (local dates of the grid's zone)
    first: pd.DatetimeIndex         # UTC instant of each day's first point

    @property
    def step(self) -> pd.Timedelta:
        return pd.Timedelta(minutes=self.cycle.step_minutes)

    @property
    def n(self) -> int:
        return self.cycle.points_per_day

    def point(self, day_pos, slot) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.first[np.asarray(day_pos)] + np.asarray(slot) * self.step)

    def instants(self) -> pd.DatetimeIndex:
        """Every grid point, sorted (day by day)."""
        k = np.arange(self.n)
        return pd.DatetimeIndex((self.first.to_numpy()[:, None] + k[None, :] * self.step.to_timedelta64()).ravel())


def local_days(points, cycle: CycleSpec) -> pd.DatetimeIndex:
    """The grid days (local dates in the cycle's zone) that grid points fall on."""
    return pd.DatetimeIndex(local_wallclock(points, cycle.timezone).normalize().unique()).sort_values()


def make_grid(cycle: CycleSpec, days: pd.DatetimeIndex) -> Grid:
    days = pd.DatetimeIndex(days).normalize().unique().sort_values()
    return Grid(cycle, days, snap_instants(days, cycle.start, cycle.timezone))


def _day_pos(grid: Grid, base: pd.Timestamp, lag: int) -> int | None:
    """Trading-day position of ``base`` moved by ``lag`` trading days (None = not a grid day)."""
    i = grid.days.searchsorted(base)
    on = i < len(grid.days) and grid.days[i] == base
    if lag == 0:
        return i if on else None
    pos = (i + lag if on else (i + lag - 1 if lag > 0 else i + lag))
    return pos if 0 <= pos < len(grid.days) else None


# --------------------------------------------------------------------------- resolution
WINDOW_COLUMNS = ["anchor", "start", "end", "start_day", "end_day", "legal", "reason", "known_from",
                  "gap_days", "start_slot", "end_slot", "stages"]


def resolve(code: EventCode, occurrences: pd.DataFrame, grid: Grid, *, max_gap: int = 10) -> pd.DataFrame:
    """One row per anchor occurrence (``consolidate``'s output, all events of the code):
    its window's start / end grid points (UTC), legality and reason, and ``known_from`` =
    the latest of the involved occurrences' (the window is known once all of them are)."""
    cyc = grid.cycle
    occ = {}
    for i, (eid, stage) in enumerate(code.dt_refs):
        o = occurrences[occurrences["event"] == eid]
        if stage is not None:
            o = o[o["stage"].str.lower() == stage.lower()]
        occ[i] = o.sort_values("timestamp").reset_index(drop=True)
    rows = []
    for a in occ[0].itertuples(index=False):
        paired, reason = {0: a}, ""
        a_pos = grid.days.searchsorted(pd.Timestamp(local_wallclock([a.timestamp], cyc.timezone)[0]).normalize())
        for k in range(1, len(code.dt_refs)):
            later = occ[k][occ[k]["timestamp"] >= a.timestamp]
            if later.empty:
                reason = "no_pair"
                break
            b = later.iloc[0]
            b_pos = grid.days.searchsorted(pd.Timestamp(local_wallclock([b["timestamp"]], cyc.timezone)[0]).normalize())
            if b_pos - a_pos > max_gap:
                reason = "no_pair"
                break
            paired[k] = b
        legs = {}
        if not reason:
            for name, leg in (("start", code.start), ("end", code.end)):
                legs[name], reason = _resolve_leg(leg, paired[leg.ref], code, grid)
                if reason:
                    break
        row = {"anchor": a.timestamp, "start": pd.NaT, "end": pd.NaT, "start_day": pd.NaT, "end_day": pd.NaT,
               "legal": False, "reason": reason, "gap_days": np.nan, "start_slot": np.nan, "end_slot": np.nan,
               "known_from": max(pd.Timestamp(p.known_from if hasattr(p, "known_from") else p["known_from"])
                                 for p in paired.values()),
               "stages": "|".join(str(p.stage if hasattr(p, "stage") else p["stage"]) for p in paired.values())}
        if not reason:
            (sp, ss), (ep, es) = legs["start"], legs["end"]
            s, e = grid.point([sp], [ss])[0], grid.point([ep], [es])[0]
            row.update(start=s, end=e, start_day=grid.days[sp], end_day=grid.days[ep], gap_days=ep - sp,
                       start_slot=ss, end_slot=es, legal=e > s, reason="" if e > s else "end_not_after_start")
        rows.append(row)
    out = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    out["legal"] = out["legal"].astype(bool)   # object dtype would make ~legal a bitwise int flip
    return out


def _resolve_leg(leg: Leg, occ, code: EventCode, grid: Grid):
    cyc = grid.cycle
    ts = pd.Timestamp(occ.timestamp if hasattr(occ, "timestamp") else occ["timestamp"])
    eid = code.dt_refs[leg.ref][0]
    base = pd.Timestamp(local_wallclock([ts], cyc.timezone)[0]).normalize()
    pos = _day_pos(grid, base, leg.day_lag)
    if pos is None:
        return None, "not_a_trading_day"
    day = grid.days[pos]
    first = grid.first[pos]
    last = first + (grid.n - 1) * grid.step
    if leg.time_ref is None:
        tsrc = occ.time_source if hasattr(occ, "time_source") else occ["time_source"]
        if tsrc == "unknown":
            return None, "no_event_time"
        tz = EVENTS[eid].timezone
        clock = pd.Timestamp(local_wallclock([ts], tz)[0])
        raw = snap_instants([day], clock.strftime("%H:%M"), tz)[0] + pd.Timedelta(seconds=clock.second)
    else:
        te = TIME_EVENTS[code.time_refs[leg.time_ref]]
        if te.grid == "start":
            raw = first
        elif te.grid == "end":
            raw = last
        else:
            raw = snap_instants([day], te.time_local, te.timezone)[0]
    if raw < first or raw > last:
        return None, "outside_grid"
    slot = int((raw - first) // grid.step) + leg.step_lag
    if not 0 <= slot < grid.n:
        return None, "lag_outside_grid"
    return (pos, slot), ""


def placebo(windows: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    """For each window pattern (day gap, start slot, end slot) of the LEGAL windows: the same
    window on every trading day that is not an event day (no legal window spans it). Columns
    as ``resolve`` plus ``pattern``; ``known_from`` = the window's start day."""
    legal = windows[windows["legal"]]
    if legal.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS + ["pattern"])
    busy = np.zeros(len(grid.days), dtype=bool)
    for r in legal.itertuples(index=False):
        i0, i1 = grid.days.searchsorted(r.start_day), grid.days.searchsorted(r.end_day)
        busy[i0:i1 + 1] = True
    out = []
    for (gap, s_slot, e_slot), _ in legal.groupby(["gap_days", "start_slot", "end_slot"]):
        gap, s_slot, e_slot = int(gap), int(s_slot), int(e_slot)
        pos = np.arange(len(grid.days) - gap)
        ok = ~busy[pos] & ~busy[pos + gap]
        pos = pos[ok]
        if not len(pos):
            continue
        s = grid.point(pos, np.full(len(pos), s_slot))
        e = grid.point(pos + gap, np.full(len(pos), e_slot))
        out.append(pd.DataFrame({"anchor": s, "start": s, "end": e, "start_day": grid.days[pos],
                                 "end_day": grid.days[pos + gap], "legal": True, "reason": "",
                                 "known_from": grid.days[pos], "gap_days": gap, "start_slot": s_slot,
                                 "end_slot": e_slot, "stages": "", "pattern": f"{gap}:{s_slot}:{e_slot}"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=WINDOW_COLUMNS + ["pattern"])


def pattern_of(windows: pd.DataFrame) -> pd.Series:
    return (windows["gap_days"].astype("Int64").astype(str) + ":" + windows["start_slot"].astype("Int64").astype(str)
            + ":" + windows["end_slot"].astype("Int64").astype(str))
