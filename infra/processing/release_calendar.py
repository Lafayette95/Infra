"""Release-calendar rows: WHEN scheduled events happen. Pure (no I/O).

One row per (event, release instant, source):

* ``timestamp`` - the release instant, tz-naive UTC (CLAUDE.md 7): the release DAY from
  the source, at the event's usual New York time (``EconEvent.time_et``) unless the source
  gives its own time; converted once, by ``infra.trading_calendar.snap_instants``;
* ``event`` - ``infra.reference.events`` id; ``source`` - where the date came from;
* ``stage`` - the estimate stage when the source says it (``flash``, ``final``,
  ``preliminary``, ``second`` ...), "" otherwise;
* ``time_source`` - ``"registry"`` (the usual time), ``"source"`` (the source's own) or
  ``"unknown"``;
* ``known_from`` - the first day this release was KNOWN to be scheduled then. A date
  observed only after it happened (FRED's history) is known from its own day; a future
  date from the day it was first seen;
* ``last_seen`` - the last day a fetch still listed it. A scheduled date that stops being
  listed (a shutdown delay, a moved release) keeps its old ``last_seen`` while later
  fetches move on - so "what did we expect on day D" (``as_of``) = everything known by D
  that had happened, or that no later fetch made by D had dropped.
"""
from __future__ import annotations

import pandas as pd

from infra.trading_calendar import snap_instants

COLUMNS = ["timestamp", "event", "source", "stage", "time_source", "known_from", "last_seen"]
KEYS = ["timestamp", "event", "source"]
NEW_YORK = "America/New_York"


def empty() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
    for c in ("timestamp", "known_from", "last_seen"):
        df[c] = df[c].astype("datetime64[ms]")
    return df


def schedule_rows(event, days: pd.DatetimeIndex, *, source: str, observed: pd.Timestamp) -> pd.DataFrame:
    """Rows for ``event`` on ``days`` as observed on day ``observed``: a day already past
    is known from itself, a future one from ``observed``. Events with no verified time
    (``time_et`` None) sit at 00:00 New York with ``time_source="unknown"``."""
    days = pd.DatetimeIndex(days).normalize()
    if days.empty:
        return empty()
    time_et = event.time_et or "00:00"
    instants = snap_instants(days, time_et, NEW_YORK)
    observed = pd.Timestamp(observed).normalize()
    return pd.DataFrame({
        "timestamp": instants.astype("datetime64[ms]"),
        "event": event.id, "source": source, "stage": "",
        "time_source": "registry" if event.time_et else "unknown",
        "known_from": pd.DatetimeIndex([min(d, observed) for d in days]).astype("datetime64[ms]"),
        "last_seen": pd.DatetimeIndex([observed] * len(days)).astype("datetime64[ms]"),
    })


def from_econ_calendar(rows: pd.DataFrame, patterns: dict[str, list[str]], events: dict) -> pd.DataFrame:
    """Release-calendar rows from harvested economic-calendar rows (infra.processing.
    econ_calendar): one per (event, release day, stage) whose row name matches one of the
    event's ``patterns``, at the row's OWN time when it has one (``time_source="source"``),
    source ``marketwatch`` - or ``marketwatch_unconfirmed`` for a past day that never showed
    an actual (a schedule that moved or was cancelled, not a release),
    else the registry's. ``known_from``: the release day for a row seen after its release
    (no earlier knowledge is claimed), the capture day for one seen before it;
    ``last_seen``: the capture day."""
    from infra.processing import econ_calendar as ec

    if rows.empty:
        return empty()
    out = []
    # a row whose day has passed (vs the latest capture held) but which never showed an
    # actual was a SCHEDULE observation never confirmed - moved or cancelled (the 2025
    # shutdown's CPI dates, found 2026-10-01): kept as its own source, not as a release
    frontier = pd.to_datetime(rows["capture"]).max().normalize() if len(rows) else pd.NaT
    for event_id, pats in patterns.items():
        mine = rows[rows["report"].map(lambda r: any(ec.matches(r, p) for p in pats)).astype(bool)]
        confirmed = set(pd.to_datetime(mine.loc[mine["actual"].notna(), "timestamp"]).dt.normalize())
        for r in mine.itertuples():
            t = ec.parse_time(r.time)
            day = pd.Timestamp(r.timestamp).normalize()
            local = t or events[event_id].time_et or "00:00"
            instant = snap_instants([day], local, NEW_YORK)[0]
            capture = pd.Timestamp(r.capture).normalize()
            unconfirmed = day not in confirmed and pd.notna(frontier) and day < frontier - pd.Timedelta(days=1)
            out.append((instant, event_id, "marketwatch_unconfirmed" if unconfirmed else "marketwatch",
                        ec.stage_of(r.report),
                        "source" if t else ("registry" if events[event_id].time_et else "unknown"),
                        min(day, capture), capture))
    if not out:
        return empty()
    df = pd.DataFrame(out, columns=COLUMNS)
    for c in ("timestamp", "known_from", "last_seen"):
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ms]")
    return df.groupby(KEYS, as_index=False).agg(stage=("stage", "max"), time_source=("time_source", "first"),
                                                 known_from=("known_from", "min"), last_seen=("last_seen", "max"))[COLUMNS]


def merge_observation(stored: pd.DataFrame, observed_rows: pd.DataFrame) -> pd.DataFrame:
    """Fold one fetch into what is stored: a row seen again keeps its EARLIEST
    ``known_from`` and gets the new ``last_seen``; new rows are added; rows not seen this
    time are left as they were (their ``last_seen`` stops advancing)."""
    if stored.empty:
        return observed_rows.reset_index(drop=True)
    both = pd.concat([stored[COLUMNS], observed_rows[COLUMNS]], ignore_index=True)
    both["stage"] = both["stage"].fillna("") if "stage" in both else ""
    out = both.groupby(KEYS, sort=True, as_index=False).agg(
        stage=("stage", "max"), time_source=("time_source", "last"), known_from=("known_from", "min"),
        last_seen=("last_seen", "max"))
    return out[COLUMNS]


def as_of(rows: pd.DataFrame, day) -> pd.DataFrame:
    """The calendar as known at the end of ``day``: every release known by then that had
    either already happened or was not yet DROPPED. Dropped = a later fetch of the same
    (event, source), made by ``day``, no longer listed it - "not fetched since" is not
    "dropped" (a source fetched daily lists every still-scheduled date each time)."""
    day = pd.Timestamp(day).normalize()
    known = rows[rows["known_from"] <= day].copy()
    if known.empty:
        return known.reset_index(drop=True)
    # latest fetch of each (event, source) made by `day`, as far as the rows can tell
    seen_by_day = known["last_seen"].where(known["last_seen"] <= day, day)
    latest_fetch = seen_by_day.groupby([known["event"], known["source"]]).transform("max")
    happened = known["timestamp"] < day + pd.Timedelta(days=1)
    dropped = latest_fetch > known["last_seen"]
    return known[happened | ~dropped].sort_values("timestamp").reset_index(drop=True)
