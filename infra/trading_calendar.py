"""Exchange trading-day bucketing (pure function; the config lives in infra.config).

This is a DATA-SEMANTICS module, not a display one (CLAUDE.md section 6e vs 7): it
defines which UTC timestamps share a "trading day" for this asset class - consumed by
infra.relative (roll-day / volume-ranking assignment) and infra.processing.resample
(the "1D" bucket). It never depends on, and must never be confused with, the DISPLAY
timezone a viewer picks in infra.dashboard.timezones (section 7) - that only changes
how an instant already assigned to a trading day is rendered, never which day it is
assigned to.
"""
from __future__ import annotations

import pandas as pd

from infra.config import TRADING_HOURS


def trading_day(index: pd.DatetimeIndex, dataset: str) -> pd.DatetimeIndex:
    """UTC tz-naive timestamps -> the exchange trading-day date each one belongs to.

    Returns a tz-naive, midnight-normalized DatetimeIndex - a grouping LABEL, not a
    literal UTC instant (the same convention ``infra.relative.rolls.day_index`` already
    uses for "day"). Raises for a dataset with no configured session rather than
    silently falling back to the UTC calendar day, since that would misclassify roll
    days and daily bars.
    """
    try:
        session = TRADING_HOURS[dataset]
    except KeyError as exc:
        raise KeyError(
            f"No trading session configured for dataset {dataset!r} - add it to "
            "infra.config.TRADING_HOURS."
        ) from exc
    local = index.tz_localize("UTC").tz_convert(session.timezone)
    day = local.normalize().tz_localize(None)
    if not session.crosses_midnight:
        return day
    close = pd.Timestamp(session.close_time).time()
    bump = local.time >= close
    return day + pd.to_timedelta(bump.astype(int), unit="D")


def snap_instants(days, local_time: str, timezone: str) -> pd.DatetimeIndex:
    """A wall-clock time in a venue's zone (config's form, e.g. ``SwapCloseSpec``) -> the
    UTC instant it falls at on each of ``days`` (local calendar dates), tz-naive UTC.

    The one place a configured local time becomes UTC (CLAUDE.md 7): call it as soon as
    the config is read, never carry the local time further. DST is the zone's own on each
    day, so a 16:15 London snap is 15:15 UTC in summer and 16:15 UTC in winter. Raises
    if the time doesn't exist that day (a spring-forward gap) rather than guessing.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(days)).normalize()
    local = (dates + pd.Timedelta(local_time + ":00")).tz_localize(timezone, nonexistent="raise",
                                                                     ambiguous="raise")
    return local.tz_convert("UTC").tz_localize(None)


def local_wallclock(instants, timezone: str) -> pd.DatetimeIndex:
    """UTC tz-naive instants -> their wall-clock reading in ``timezone``, tz-naive: a LABEL
    (which local day, which local time of day), like ``trading_day`` - never written back
    over a stored timestamp. Event studies use it to put an event's own clock time on
    another day (NFP's 08:30 on the next trading day) and to name the local day an event
    falls on; the instants they compute come from ``snap_instants``."""
    idx = pd.DatetimeIndex(pd.to_datetime(instants))
    return idx.tz_localize("UTC").tz_convert(timezone).tz_localize(None)
