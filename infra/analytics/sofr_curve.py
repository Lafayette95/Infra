"""Overnight SOFR path implied by SR1 futures (financing model layer 1, CLAUDE.md 20).
Pure functions, no I/O.

The path is PIECEWISE FLAT between FOMC decisions: one level per interval between
consecutive rate-change dates (each meeting's announcement day + 1), plus a YEAR-END TURN
(the jump SOFR makes on the last business day of a year, a balance-sheet effect futures
can't separate from December's level, so it is estimated from history and imposed).
Ordinary month- and quarter-end spikes are left inside the levels (they average out over
a financing horizon; measured 2026-10-02: no premium on ordinary month-ends).

Each SR1 contract settles on the ARITHMETIC average of daily SOFR over every CALENDAR day
of its month, a non-business day taking the previous business day's rate. Days whose
fixing is already published are known exactly; the levels are fitted by least squares on
the remaining days. Linear, so it solves in one step.

Business days approximate SIFMA's US calendar: federal holidays plus Good Friday.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar

_ONE_DAY = pd.Timedelta(days=1)


def business_days(start, end) -> pd.DatetimeIndex:
    """SOFR fixing days in ``[start, end]`` (approximation of the SIFMA calendar)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    holidays = USFederalHolidayCalendar().holidays(start, end)
    good_friday = pd.DatetimeIndex(GoodFriday.dates(start, end))
    return pd.bdate_range(start, end).difference(holidays).difference(good_friday)


def governing_days(days: pd.DatetimeIndex, fixing_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For each calendar day, the business day whose fixing applies (itself, or the last
    one before it)."""
    fixing_days = pd.DatetimeIndex(fixing_days).sort_values()
    idx = fixing_days.searchsorted(days, side="right") - 1
    if (idx < 0).any():
        raise ValueError("calendar days before the first fixing day")
    return fixing_days[idx]


def year_end_days(fixing_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last fixing day of each year in ``fixing_days``."""
    s = pd.Series(fixing_days, index=fixing_days)
    return pd.DatetimeIndex(s.groupby(s.index.year).max().to_numpy())


def year_end_turn_bp(fixings: pd.Series, before, *, last_n: int = 3, lookback_days: int = 5, min_years: int = 1,
                     default_bp: float = 0.0) -> float:
    """Median year-end jump in bp: SOFR on each year's last fixing day minus the median of
    the ``lookback_days`` fixings before it, over the ``last_n`` year-ends strictly before
    ``before`` (point in time; recent ones only, because the jump follows the reserve
    regime: +56bp 2018, ~0 2020-22, +9 / +16bp 2024 / 2025). ``default_bp`` when fewer
    than ``min_years`` are available."""
    fixings = fixings.sort_index()
    fixings = fixings[fixings.index < pd.Timestamp(before)]
    jumps = []
    for ye in year_end_days(fixings.index):
        if ye.month != 12 or ye.day < 24:  # the series ends mid-year: not a year-end
            continue
        prior = fixings[fixings.index < ye].tail(lookback_days)
        if len(prior) == lookback_days:
            jumps.append((fixings[ye] - prior.median()) * 100)
    jumps = jumps[-last_n:]
    return float(np.median(jumps)) if len(jumps) >= min_years else float(default_bp)


@dataclass(frozen=True)
class SofrPath:
    """A fitted overnight path. ``daily``: calendar day -> rate (%), realized where known.
    ``levels``: one row per interval (``start``, ``end`` inclusive calendar days, ``level``).
    ``residuals_bp``: fitted minus settlement-implied monthly average, per contract month."""
    as_of: pd.Timestamp
    daily: pd.Series
    known_through: pd.Timestamp
    levels: pd.DataFrame
    residuals_bp: pd.Series
    year_end_turn_bp: float

    def compounded(self, start, end) -> float:
        """Annualized rate (%, simple ACT/360) of rolling overnight from ``start`` to ``end``
        (calendar days in ``[start, end)``), daily compounded."""
        days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() - _ONE_DAY)
        if len(days) == 0:
            return float("nan")
        missing = days.difference(self.daily.index)
        if len(missing):
            raise ValueError(f"path does not cover {missing[0].date()}..{missing[-1].date()}")
        growth = np.prod(1.0 + self.daily.loc[days].to_numpy() / 100.0 / 360.0)
        return float((growth - 1.0) * 360.0 / len(days) * 100.0)


def fit_sofr_path(as_of, futures: pd.DataFrame, fixings: pd.Series, change_dates, *,
                  year_end_turn: float = 0.0) -> SofrPath:
    """Fit the path from SR1 settlements.

    ``futures``: one row per contract, ``month`` (first day of the delivery month) and
    ``price``. ``fixings``: published SOFR by fixing day (%), only those published by
    ``as_of``. ``change_dates``: days a new policy rate takes EFFECT (announcement + 1),
    only those known by ``as_of``. ``year_end_turn``: bp added on each year's last fixing
    day not yet published."""
    as_of = pd.Timestamp(as_of).normalize()
    futures = futures.sort_values("month").reset_index(drop=True)
    fixings = fixings.sort_index()
    first = pd.Timestamp(futures["month"].iloc[0])
    last = (pd.Timestamp(futures["month"].iloc[-1]) + pd.offsets.MonthEnd(0)).normalize()
    known_through = fixings.index.max()
    fixing_days = business_days(min(first, fixings.index.min()) - pd.Timedelta(days=10), last)
    fixing_days = fixing_days.union(fixings.index)
    calendar = pd.date_range(first, last)
    gov = governing_days(calendar, fixing_days)
    known = gov <= known_through
    rates = pd.Series(np.nan, index=calendar)
    rates[known] = fixings.reindex(gov[known]).to_numpy()
    if rates[known].isna().any():
        raise ValueError("a published fixing day has no fixing")
    # intervals over the unknown days, cut at each change date
    unknown_days = calendar[~known]
    cuts = sorted(pd.Timestamp(c).normalize() for c in change_dates)
    interval = np.searchsorted(np.array(cuts, dtype="datetime64[ns]"), gov[~known].to_numpy(), side="right")
    turn = np.isin(gov[~known], year_end_days(fixing_days)) * (year_end_turn / 100.0)
    # one equation per contract: sum over its calendar days = n_days x (100 - price)
    labels = np.unique(interval)
    col = {k: j for j, k in enumerate(labels)}
    A, b, months = [], [], []
    for _, row in futures.iterrows():
        m0 = pd.Timestamp(row["month"])
        in_month = (calendar >= m0) & (calendar <= (m0 + pd.offsets.MonthEnd(0)))
        target = in_month.sum() * (100.0 - float(row["price"]))
        target -= rates[in_month & known].sum()
        sel = in_month[~known]
        if not sel.any():
            continue  # fully fixed already: nothing to fit
        a = np.zeros(len(labels))
        np.add.at(a, [col[k] for k in interval[sel]], 1.0)
        A.append(a)
        b.append(target - turn[sel].sum())
        months.append(m0)
    A, b = np.array(A), np.array(b)
    used = A.sum(axis=0) > 0
    x = np.full(len(labels), np.nan)
    x[used] = np.linalg.lstsq(A[:, used], b, rcond=None)[0]
    lv = x[[col[k] for k in interval]]
    rates[~known] = lv + turn
    resid = (A[:, used] @ x[used] - b) / A.sum(axis=1) * 100.0
    bounds = pd.DataFrame({"interval": interval, "day": unknown_days}).groupby("interval")["day"].agg(["min", "max"])
    levels = pd.DataFrame({"start": bounds["min"].to_numpy(), "end": bounds["max"].to_numpy(),
                           "level": x[[col[k] for k in bounds.index]]})
    return SofrPath(as_of=as_of, daily=rates, known_through=known_through, levels=levels,
                    residuals_bp=pd.Series(resid, index=pd.DatetimeIndex(months)), year_end_turn_bp=year_end_turn)
