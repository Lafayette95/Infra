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
