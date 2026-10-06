"""Conditioning event-study windows on a third series: its point-in-time FEATURE at each
window's start, and the regime (-1 / 0 / +1) the feature is in. Pure.

1. ``state_timeline``: a series' rows (``infra.pipeline.series_panel.read_available``:
   ``label, available_at, value``) -> its STATE at each availability instant: the value of the
   latest label known by then (a revision of an older label changes nothing; a vintage series'
   new print does). ``period_diff=True``: the latest label's value minus the previous label's,
   as known at that instant (a print's change vs the prior period).
2. ``feature``: the stateless prep steps (``infra.models.prep``: ``diff``, ``logdiff``,
   ``ewm_z:hl`` ...) applied along that timeline - trailing, so a row only uses earlier rows.
3. ``partition``: a swappable rule (``PARTITIONERS``) turning each timeline row's feature into
   -1 / 0 / +1 using only the trailing ``W`` rows up to and INCLUDING it (point in time):
   ``rolling_tercile:W`` (below the 1/3 quantile, above the 2/3), ``zscore:W:k`` (beyond
   +-k trailing standard deviations), ``sign``, ``fixed:a:b`` (below a, above b). Too little
   history -> NaN (no regime).
4. ``at_windows``: each window's feature / regime = the timeline row available at or before
   ``start - lag`` (``lag``: a safety margin on top of availability, e.g. one grid step).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


from infra.processing.features import state_timeline  # noqa: E402,F401  (moved 2026-10-06)


def feature(timeline: pd.DataFrame, steps=()) -> pd.Series:
    """The prep steps along the timeline (index = ``available_at``)."""
    from infra.models.prep import run_stateless
    s = pd.DataFrame({"x": timeline["value"].to_numpy(dtype="float64")},
                     index=pd.DatetimeIndex(timeline["available_at"]))
    steps = tuple(x for x in steps if not x.startswith("resample"))
    return run_stateless(s, steps)["x"] if steps else s["x"]


# The partitions live in the central feature maker (infra.processing.features, 2026-10-06); these
# names stay for existing callers.
from infra.processing.features import PARTITIONS as _PARTITIONS  # noqa: E402
from infra.processing.features import partition  # noqa: E402,F401

PARTITIONERS: dict[str, Callable] = {k: _PARTITIONS[k] for k in ("rolling_tercile", "zscore", "sign", "fixed")}


def at_windows(starts, feat: pd.Series, buckets: np.ndarray, lag: pd.Timedelta) -> pd.DataFrame:
    """Per window start: ``cond_value``, ``cond_bucket`` and ``cond_asof`` (the availability
    instant of the row used) - the latest row available at or before ``start - lag``."""
    starts = pd.DatetimeIndex(pd.to_datetime(starts))
    avail = pd.DatetimeIndex(feat.index)
    pos = avail.searchsorted(starts - lag, side="right") - 1
    ok = (pos >= 0) & starts.notna()
    val, bkt = np.full(len(starts), np.nan), np.full(len(starts), np.nan)
    asof = pd.Series(pd.NaT, index=range(len(starts)), dtype="datetime64[ns]")
    if ok.any():
        val[ok] = feat.to_numpy(dtype="float64")[pos[ok]]
        bkt[ok] = np.asarray(buckets, dtype="float64")[pos[ok]]
        asof[np.flatnonzero(ok)] = avail[pos[ok]].to_numpy()
    return pd.DataFrame({"cond_value": val, "cond_bucket": bkt, "cond_asof": asof.to_numpy()})
