"""Macro-release vintages: storage encoding and point-in-time views. Pure pandas, no I/O.

The raw store holds every published value of every source series, one row per
``(timestamp, ticker, period)``:

* ``timestamp`` - the PUBLICATION day of the value (the source's real-time start), so
  the store is partitioned by when things were known - a point-in-time read ("what was
  published by D") prunes by partition like every other store (CLAUDE.md 3, ``as_of``);
* ``ticker`` - the source's own series id (``PAYEMS``), exactly as published;
* ``period`` - the observation period's first day (``2026-08-01`` = August, a quarter's
  first month for quarterly data, the week-ending day for weekly data - as the source
  dates it);
* ``value`` - float64, as published. Deliberately NOT the x10000 fixed-point of CLAUDE.md
  6b: that rule is for prices, and macro values span ~10 orders of magnitude (initial
  claims peaked at 6.1 million in 2020, which x10000 overflows int32).

A row exists only where the published value CHANGED (``drop_unchanged`` - and ALFRED itself
only starts a new real-time period when the value changes). So a scheduled estimate that
repeats the previous one leaves NO row: verified 2026-09-30, Q2 2026 GDP's second estimate
(2026-08-27, unchanged at 1.5%) is absent. The k-th row is the k-th DISTINCT value, not
necessarily the k-th scheduled estimate - labelling rows as Advance/Second/Third needs the
source's release calendar (TOFIX.md). The model never needs the labels: it works on
point-in-time snapshots, and an unchanged re-publication carries no news.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = ["timestamp", "ticker", "period", "value"]
RAW_KEYS = ["timestamp", "ticker", "period"]
SERIES_KEYS = ["ticker", "period"]


def empty_raw() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ms]"), "ticker": pd.Series(dtype="object"),
        "period": pd.Series(dtype="datetime64[ms]"), "value": pd.Series(dtype="float64"),
    })


def from_vintages(vintages: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """A fetcher's ``realtime_start, date, value`` frame -> raw rows for ``ticker``."""
    if vintages.empty:
        return empty_raw()
    return pd.DataFrame({
        "timestamp": vintages["realtime_start"].astype("datetime64[ms]").to_numpy(),
        "ticker": ticker,
        "period": vintages["date"].astype("datetime64[ms]").to_numpy(),
        "value": vintages["value"].astype("float64").to_numpy(),
    })


def from_snapshot(values: pd.DataFrame, published) -> pd.DataFrame:
    """A bulk file's ``series_id, date, value`` frame (the latest revised history, no
    vintages - infra.api.bls_client / bea_client) -> raw rows, every one stamped with the
    file's PUBLICATION day. ``drop_unchanged`` then keeps only what that publication changed,
    so successive snapshots build the vintage history the source itself doesn't keep."""
    if values.empty:
        return empty_raw()
    return pd.DataFrame({
        "timestamp": pd.Timestamp(published).normalize(),
        "ticker": values["series_id"].astype(str).to_numpy(),
        "period": values["date"].astype("datetime64[ms]").to_numpy(),
        "value": values["value"].astype("float64").to_numpy(),
    }).astype({"timestamp": "datetime64[ms]"})


def encode_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Flat, disk-ready (CLAUDE.md 6a): no index, ms timestamps, plain string ticker."""
    out = df[RAW_COLUMNS].copy()
    out["timestamp"] = out["timestamp"].astype("datetime64[ms]")
    out["period"] = out["period"].astype("datetime64[ms]")
    out["ticker"] = out["ticker"].astype(str)
    out["value"] = out["value"].astype("float64")
    return out.reset_index(drop=True)


def decode_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df[RAW_COLUMNS].copy()
    out["ticker"] = out["ticker"].astype("category")
    return out.sort_values(["ticker", "period", "timestamp"]).reset_index(drop=True)


def drop_unchanged(incoming: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    """Incoming rows that are genuinely NEW information: a value that differs from the
    one already published for that period as of that day (from ``stored`` or earlier
    incoming rows). Drops re-fetched duplicates and the source's clipped "still current"
    rows at a window's edge (infra.api.fred_client). An incoming row at the SAME key as a
    stored row but a different value is kept - it overwrites, and the raw step's
    no-revisions check reports it (a published vintage should never change)."""
    if incoming.empty:
        return incoming
    inc = incoming.assign(_src=1)
    both = pd.concat([stored[RAW_COLUMNS].assign(_src=0), inc], ignore_index=True) if len(stored) else inc
    both["ticker"] = both["ticker"].astype(str)
    both = both.sort_values(["ticker", "period", "timestamp", "_src"], kind="stable").reset_index(drop=True)
    prev = both.groupby(SERIES_KEYS, sort=False)["value"].shift(1)
    new = both["_src"].eq(1) & ~np.isclose(both["value"], prev, rtol=0.0, atol=1e-9, equal_nan=False)
    return both.loc[new, RAW_COLUMNS].reset_index(drop=True)


def snapshot(raw: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Each ``(ticker, period)``'s value as published by the END of day ``as_of``
    (inclusive; None = latest): ``ticker, period, value, timestamp`` (its publication day)."""
    df = raw if as_of is None else raw[raw["timestamp"] <= pd.Timestamp(as_of)]
    if df.empty:
        return empty_raw()[["ticker", "period", "value", "timestamp"]]
    last = df.sort_values("timestamp", kind="stable").groupby(SERIES_KEYS, observed=True, sort=True).tail(1)
    return last[["ticker", "period", "value", "timestamp"]].sort_values(SERIES_KEYS).reset_index(drop=True)


def estimate(raw: pd.DataFrame, k: int) -> pd.DataFrame:
    """The k-th DISTINCT published value (0 = first print) of each ``(ticker, period)``
    that has one: ``ticker, period, value, timestamp``. Not the k-th scheduled estimate
    when an estimate repeated the previous value (module doc)."""
    df = raw.sort_values([*SERIES_KEYS, "timestamp"], kind="stable")
    nth = df.groupby(SERIES_KEYS, observed=True, sort=True).nth(k)
    return nth[["ticker", "period", "value", "timestamp"]].reset_index(drop=True)


# How many periods back each unit's derivation reaches, per observation frequency.
_LAGS = {"level": {"Q": 0, "M": 0, "W": 0}, "diff": {"Q": 1, "M": 1, "W": 1},
         "pct": {"Q": 1, "M": 1, "W": 1}, "saar": {"Q": 1, "M": 1}, "yoy": {"Q": 4, "M": 12, "W": 52}}
_PER_YEAR = {"Q": 4, "M": 12}


def derive_units(series: pd.DataFrame, units: str, frequency: str) -> pd.DataFrame:
    """One ticker's point-in-time ``period, value, timestamp`` (a ``snapshot``) converted
    into the release table's units. A derived value is published once ALL its inputs are:
    its ``timestamp`` is the latest of theirs (e.g. May's payroll change needs May's AND
    April's level as published). Periods must be consecutive for ``diff``/``pct``/``yoy``;
    a gap yields NaN (dropped), never a change across the gap."""
    if units not in _LAGS or frequency not in _LAGS[units]:
        raise ValueError(f"unknown units {units!r} for frequency {frequency!r}")
    s = series.sort_values("period").reset_index(drop=True)
    lag = _LAGS[units][frequency]
    if lag == 0:
        return s[["period", "value", "timestamp"]]
    step = {"Q": pd.DateOffset(months=3), "M": pd.DateOffset(months=1), "W": pd.DateOffset(weeks=1)}[frequency]
    prior = s.set_index("period")[["value", "timestamp"]]
    back = prior.reindex(s["period"] - step * lag)
    v0, t0 = back["value"].to_numpy(), back["timestamp"].to_numpy()
    v1 = s["value"].to_numpy()
    if units == "diff":
        val = v1 - v0
    elif units == "saar":
        val = 100.0 * ((v1 / v0) ** _PER_YEAR[frequency] - 1.0)
    else:
        val = 100.0 * (v1 / v0 - 1.0)
    out = pd.DataFrame({"period": s["period"], "value": val,
                        "timestamp": np.maximum(s["timestamp"].to_numpy(), t0)})
    return out.dropna(subset=["value"]).reset_index(drop=True)
