"""Time-only events and trading grids (cycles) for event studies - static reference, like
the event registry (``infra.reference.events``), which holds the DATE+time events.

* ``TIME_EVENTS``: a time of day in a venue's own zone, with no date of its own (a futures
  open, a benchmark close). An event-study leg puts it on a day taken from a date+time
  event (``infra.models.event_study``). ``GRID_START`` / ``GRID_END`` resolve to the
  cycle's own first / last point, so they follow the grid if the grid changes.
* ``CYCLES``: the traded grid - a step, a first and a last point (local wall clock + IANA
  zone), and the calendar of days it trades. Grid points run from ``start`` to ``end``
  INCLUSIVE; a step's P&L is the move over (previous point, point], stamped at the point,
  and the first point of a day's step reaches back to the previous trading day's last point
  (the overnight move). The grid is NOT continuous: hours outside it are not traded.
* ``CYCLE_ALIASES``: indirection (``DEFAULT_CYCLE`` -> ``15MIN_NO_OVERNIGHT``), so the
  default grid can move in one line - studies then need re-running (their outputs are tied
  to the grid they ran on, which every result records).

Local times are converted to UTC instants at the first step that reads them
(``infra.trading_calendar.snap_instants``); only UTC travels further (root CLAUDE.md 7).
"""
from __future__ import annotations

from dataclasses import dataclass

from infra.config import SWAP_CLOSES

NEW_YORK, CHICAGO = "America/New_York", "America/Chicago"


@dataclass(frozen=True)
class TimeEvent:
    id: str
    time_local: str | None  # "HH:MM" in ``timezone``; None = from the cycle (``grid``)
    timezone: str | None
    note: str = ""
    grid: str | None = None  # "start" | "end": the cycle's own first / last point


@dataclass(frozen=True)
class CycleSpec:
    name: str
    step_minutes: int
    start: str  # "HH:MM", first grid point (local)
    end: str  # "HH:MM", last grid point (local), inclusive
    timezone: str
    calendar: str = "market"  # trading days (infra.processing.schedule_rules.business_days)
    note: str = ""

    @property
    def points_per_day(self) -> int:
        h0, m0 = map(int, self.start.split(":"))
        h1, m1 = map(int, self.end.split(":"))
        span = (h1 * 60 + m1) - (h0 * 60 + m0)
        if span <= 0 or span % self.step_minutes:
            raise ValueError(f"cycle {self.name}: {self.start}-{self.end} is not a whole number of "
                             f"{self.step_minutes}-minute steps")
        return span // self.step_minutes + 1


TIME_EVENTS: dict[str, TimeEvent] = {e.id: e for e in (
    TimeEvent("GRID_START", None, None, "the cycle's first point", grid="start"),
    TimeEvent("GRID_END", None, None, "the cycle's last point", grid="end"),
    # CME Treasury (and SOFR) futures on Globex: 5:00 p.m. - 4:00 p.m. CT, Sunday-Friday
    # (cmegroup.com, verified 2026-10-05; the session crosses midnight - root CLAUDE.md 6e)
    TimeEvent("CME_GLOBEX_OPEN", "17:00", CHICAGO, "Globex session open (the evening before its trade date)"),
    TimeEvent("CME_GLOBEX_CLOSE", "16:00", CHICAGO, "Globex session close"),
    TimeEvent("CME_TSY_SETTLE", "14:00", CHICAGO, "CME Treasury / SOFR futures settlement (root CLAUDE.md 16)"),
    # The benchmark snaps (infra.config.SWAP_CLOSES - one place for every benchmark time)
    *(TimeEvent(name, spec.local_time, spec.timezone, f"benchmark snap {name}") for name, spec in SWAP_CLOSES.items()),
)}


CYCLES: dict[str, CycleSpec] = {c.name: c for c in (
    CycleSpec("15MIN_NO_OVERNIGHT", 15, "06:00", "19:00", NEW_YORK,
              note="New York day, no overnight: 06:00-19:00 ET every 15 minutes"),
    CycleSpec("1MIN_NO_OVERNIGHT", 1, "06:00", "19:00", NEW_YORK, note="the same hours on the 1-minute grid"),
)}

CYCLE_ALIASES: dict[str, str] = {"DEFAULT_CYCLE": "15MIN_NO_OVERNIGHT"}


def resolve_cycle(name: str) -> CycleSpec:
    """A cycle by name, following aliases (``DEFAULT_CYCLE`` -> ``15MIN_NO_OVERNIGHT``)."""
    seen = set()
    while name in CYCLE_ALIASES:
        if name in seen:
            raise ValueError(f"cycle alias loop at {name!r}")
        seen.add(name)
        name = CYCLE_ALIASES[name]
    if name not in CYCLES:
        raise KeyError(f"unknown cycle {name!r}; known: {sorted(CYCLES)} (aliases {sorted(CYCLE_ALIASES)})")
    return CYCLES[name]
