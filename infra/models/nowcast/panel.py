"""Data preparation: raw release vintages -> the model's monthly panel, AS OF a day.

Pure (no I/O): takes the decoded raw vintage rows (``infra.pipeline.releases.
read_releases_from_disk``) and the release table, returns what the model consumes. Each
step is its own function so a single release can be inspected at any stage:

    raw vintages --snapshot(as_of)--> source series as published by ``as_of``
      --derive_units--> the table's units (``MacroRelease.units``: diff / pct / yoy)
      --transform + sign--> the table's transform (the TARGET skips this: native units)
      --to_monthly--> one monthly grid (quarterly at the quarter's 3rd month, weekly as
                      the mean of a COMPLETE month's weeks)

Everything is point-in-time: nothing published after ``as_of`` is ever read, and every
transform is trailing (transforms.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from infra.config import MacroRelease
from infra.models.nowcast import transforms
from infra.models.nowcast.spec import ModelSpec
from infra.processing import releases as pr

# A release needs this many monthly observations in the sample to enter the model.
MIN_SERIES_OBS = 24


@dataclass
class Panel:
    as_of: pd.Timestamp | None
    data: pd.DataFrame  # months x release tickers: model units (transformed; target native)
    native: pd.DataFrame  # months x release tickers: the table's units, untransformed
    published: pd.DataFrame  # months x release tickers: publication day of each value
    frequency: dict[str, str]  # release ticker -> "Q" | "M" | "W"
    dropped: dict[str, str]  # release ticker -> why it is not in the panel


def release_native(raw: pd.DataFrame, release: MacroRelease, as_of=None) -> pd.DataFrame:
    """One release's ``period, value, timestamp`` in the table's units, as published by
    ``as_of``, at its native frequency."""
    snap = pr.snapshot(raw[raw["ticker"] == release.series_id], as_of)
    return pr.derive_units(snap, release.units, release.frequency)


def model_transform(release: MacroRelease, spec: ModelSpec) -> str:
    """The transform the model applies: the table's own, ``gauss`` appended when the
    spec asks for it; none for the target (it stays in GDP units)."""
    if release.ticker == spec.target or not spec.use_transforms:
        return ""
    if spec.gaussianize and "ecdf" in release.transform:
        return release.transform + ";gauss"
    return release.transform


def transformed(native: pd.DataFrame, release: MacroRelease, spec: ModelSpec) -> pd.Series:
    s = native.set_index("period")["value"]
    out = transforms.apply(s, model_transform(release, spec)) if model_transform(release, spec) else s
    return out if release.ticker == spec.target else release.sign * out


def to_monthly(s: pd.Series, frequency: str, *, how: str = "mean") -> pd.Series:
    """Native-frequency series (index = the source's period date) -> month-start index.

    ``Q``: placed at the quarter's THIRD month (the source dates a quarter by its first).
    ``W``: the source dates a week by its last day; a month's value is the mean (``how``)
    of the weeks ending in it, and exists only once EVERY such week is published - a
    partial month would otherwise change as weeks arrive, turning each new week into a
    revision of the month rather than news."""
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    if frequency == "M":
        return s.set_axis(s.index.to_period("M").to_timestamp())
    if frequency == "Q":
        return s.set_axis(s.index.to_period("Q").to_timestamp(how="end").to_period("M").to_timestamp())
    if frequency != "W":
        raise ValueError(f"unknown frequency {frequency!r}")
    months = s.index.to_period("M").to_timestamp()
    grouped = s.groupby(months)
    agg = grouped.agg(how)
    weekday = s.index[0].dayofweek  # the source's week-ending weekday
    expected = pd.Series({m: _count_weekday(m, weekday) for m in agg.index})
    return agg[grouped.size() == expected]


def _count_weekday(month_start: pd.Timestamp, weekday: int) -> int:
    days = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0), freq="D")
    return int((days.dayofweek == weekday).sum())


def exclude_months(data: pd.DataFrame, ranges) -> pd.DataFrame:
    """``data`` with every row inside the inclusive month ``ranges`` set to missing."""
    out = data.copy()
    for start, end in ranges:
        out.loc[pd.Timestamp(start):pd.Timestamp(end)] = float("nan")
    return out


def build_panel(
    raw: pd.DataFrame,
    releases: dict[str, MacroRelease],
    spec: ModelSpec,
    as_of=None,
    *,
    end=None,
) -> Panel:
    """The monthly panel of every available release as published by ``as_of``, from
    ``spec.sample_start`` through the month of ``end`` (default: ``as_of``'s quarter end,
    so the current quarter's target month is on the grid even before any data for it)."""
    as_of = None if as_of is None else pd.Timestamp(as_of)
    last = pd.Timestamp(end) if end is not None else (as_of or raw["timestamp"].max())
    months = pd.date_range(pd.Timestamp(spec.sample_start), last.to_period("Q").to_timestamp(how="end"),
                           freq="MS")
    cols, native_cols, pub_cols, freq, dropped = {}, {}, {}, {}, {}
    for ticker, rel in releases.items():
        if not rel.available:
            dropped[ticker] = "no free source"
            continue
        nat = release_native(raw, rel, as_of)
        if nat.empty:
            dropped[ticker] = "nothing published by as_of"
            continue
        m_data = to_monthly(transformed(nat, rel, spec), rel.frequency).reindex(months)
        if m_data.notna().sum() < MIN_SERIES_OBS:
            dropped[ticker] = f"fewer than {MIN_SERIES_OBS} observations in the sample"
            continue
        cols[ticker] = m_data
        native_cols[ticker] = to_monthly(nat.set_index("period")["value"], rel.frequency).reindex(months)
        # a weekly month's publication day is its last week's
        pub_cols[ticker] = to_monthly(nat.set_index("period")["timestamp"], rel.frequency, how="max").reindex(months)
        freq[ticker] = rel.frequency
    if spec.target not in cols:
        raise ValueError(f"target {spec.target} not in the panel as of {as_of}: {dropped.get(spec.target)}")
    order = [spec.target] + [t for t in cols if t != spec.target]
    data = exclude_months(pd.DataFrame(cols, index=months)[order], spec.exclude)
    return Panel(as_of, data, pd.DataFrame(native_cols, index=months)[order],
                 pd.DataFrame(pub_cols, index=months)[order], {t: freq[t] for t in order}, dropped)
