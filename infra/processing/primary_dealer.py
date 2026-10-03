"""NY Fed Primary Dealer Statistics CSV -> a long frame. Pure, no I/O.

One row per (as-of date, series): ``value`` in $ millions (float; a suppressed ``*`` ->
NaN with ``suppressed`` True). Point in time: ``known_from`` = the release,
``PRIMARY_DEALER_RELEASE`` (16:15 New York, the Thursday after the reporting week) - for
the Wednesday-dated series exactly the published schedule; the MBS settlement-class
series (dated other days) take the first release at least that lag after their date
(conservative: never earlier than the true release).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from infra.trading_calendar import snap_instants

KEYS = ["timestamp", "series"]


def release_day(dates: pd.Series, lag_days: int) -> pd.Series:
    """The release DAY for each as-of date: the first Thursday at least ``lag_days`` after
    it (a Wednesday + 8 = the next week's Thursday)."""
    d = pd.to_datetime(dates) + pd.Timedelta(days=lag_days)
    return d + pd.to_timedelta((3 - d.dt.dayofweek) % 7, unit="D")


def parse_primary_dealer(text: str, release: tuple[str, str, int]) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = ["timestamp", "series", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
    df["series"] = df["series"].str.strip()
    raw = df["value"].str.strip()
    df["suppressed"] = raw.eq("*")
    df["value"] = pd.to_numeric(raw.str.replace(",", ""), errors="coerce").astype("float64")
    local_time, zone, lag = release
    days = release_day(df["timestamp"], lag)
    uniq = days.drop_duplicates()
    instant = pd.Series(snap_instants(uniq, local_time, zone), index=uniq.to_numpy())
    df["known_from"] = days.map(instant).astype("datetime64[ms]")
    df = df.drop_duplicates(KEYS, keep="last").sort_values(KEYS, ignore_index=True)
    return df[[*KEYS, "value", "suppressed", "known_from"]]


def changed_rows(new: pd.DataFrame, stored: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """The rows of ``new`` that are new or differ from ``stored`` (value or suppression),
    and how many of them REVISE a stored row."""
    if stored is None or stored.empty:
        return new, 0
    m = new.merge(stored[[*KEYS, "value", "suppressed"]], on=KEYS, how="left", suffixes=("", "_old"), indicator=True)
    fresh = m["_merge"].eq("left_only")
    same = (np.isclose(m["value"], m["value_old"], rtol=0, atol=1e-9) | (m["value"].isna() & m["value_old"].isna())) \
        & m["suppressed"].eq(m["suppressed_old"].astype("boolean").fillna(False))
    keep = fresh | ~same
    return new.loc[keep.to_numpy()].reset_index(drop=True), int((keep & ~fresh).sum())
