"""Release-date RULES (``EconEvent.schedule``) - generate and validate. Pure (no I/O).

A rule says which day of each month an event comes out, e.g. ``first_business_day``,
``nth_weekday:2:TUE``, ``last_weekday:TUE``. Rules are never trusted on recall: each is
scored against the release days actually observed (the harvested economic calendar,
infra.pipeline.release_calendar), by year (``score``), and only a rule that keeps matching
recent history may project dates forward (``infra.pipeline.release_calendar``).

Business days: weekdays that are not US federal holidays (pandas'
``USFederalHolidayCalendar``: New Year's, MLK, Presidents', Memorial, Juneteenth (from
2021), Independence, Labor, Columbus, Veterans, Thanksgiving, Christmas). Agencies'
own closures (e.g. a one-off presidential day of mourning) are not in it - such days show
up as misses, which is what the per-year score is for.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USFederalHolidayCalendar,
                                    nearest_workday, sunday_to_monday)

_WEEKDAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


class _MarketCalendar(AbstractHolidayCalendar):
    """NYSE-style: federal holidays minus Columbus and Veterans Day, plus Good Friday (and
    Juneteenth from 2022, the year the exchange first closed for it). New Year's Day on a
    SATURDAY is not observed (markets open on Friday 31 December - 2010, 2021; pandas'
    federal rule observes it on the Friday): found 2026-10-05 when CME's Dec-2021 Treasury
    futures last trading days (stored expiries) disagreed with the rule by one day."""
    rules = [r for r in USFederalHolidayCalendar.rules
             if r.name not in ("Columbus Day", "Veterans Day", "Juneteenth National Independence Day",
                               "New Year's Day")] + [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        GoodFriday, Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday)]


def business_days(start, end, calendar: str = "federal") -> pd.DatetimeIndex:
    """Weekdays minus the ``calendar``'s holidays: ``federal`` (agencies) or ``market``
    (NYSE-style - private publishers like MNI follow it: no Chicago PMI on Good Friday)."""
    cal = USFederalHolidayCalendar() if calendar == "federal" else _MarketCalendar()
    holidays = cal.holidays(pd.Timestamp(start) - pd.Timedelta(days=7), pd.Timestamp(end) + pd.Timedelta(days=7))
    days = pd.bdate_range(start, end)
    return days[~days.isin(holidays)]


def _months(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), end, freq="MS")


def _in_month(days: pd.DatetimeIndex, month: pd.Timestamp) -> pd.DatetimeIndex:
    return days[(days >= month) & (days < month + pd.offsets.MonthBegin(1))]


_MONTH_KEYS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                              "nov", "dec"], start=1)}


def parse_rule(rule: str) -> tuple[str, str, dict[int, str | None]]:
    """``"[market:]BASE[;mon=RULE|none ...]"`` -> (calendar, base rule, month overrides).
    E.g. ``"nth_business_day:1;jan=nth_business_day:2"`` (ISM manufacturing),
    ``"last_weekday:TUE;dec=none"`` (no reliable December date), ``"market:last_business_day"``."""
    parts = rule.split(";")
    base, calendar = parts[0], "federal"
    if base.startswith("market:"):
        calendar, base = "market", base[len("market:"):]
    overrides = {}
    for part in parts[1:]:
        mon, _, r = part.partition("=")
        overrides[_MONTH_KEYS[mon.strip().lower()]] = None if r.strip() == "none" else r.strip()
    return calendar, base, overrides


def rule_dates(rule: str, start, end) -> pd.DatetimeIndex:
    """Every date ``rule`` (``parse_rule`` syntax) gives in ``[start, end]``."""
    calendar, base, overrides = parse_rule(rule)
    out = []
    for month in _months(start, end):
        r = overrides.get(month.month, base)
        if r is None:
            continue
        out += list(_base_dates(r, max(pd.Timestamp(start), month),
                                min(pd.Timestamp(end), month + pd.offsets.MonthEnd(0)), calendar))
    return pd.DatetimeIndex(out)


def _base_dates(rule: str, start, end, calendar: str) -> pd.DatetimeIndex:
    """Every date ``rule`` gives in ``[start, end]``. Rules:

    * ``nth_business_day:N`` (``first_business_day`` = N=1, ``third_business_day`` = N=3)
    * ``last_business_day``
    * ``nth_weekday:N:DAY`` (``nth_weekday:2:TUE`` = the 2nd Tuesday)
    * ``last_weekday:DAY``
    * ``nth_weekday_after:N:DAY:D`` - the N-th DAY on or after day-of-month D
    * ``business_day_before_last:N`` - N business days before the month's last one
    * ``day_or_next_business_day:D`` - day-of-month D, or the next business day
    Prefix ``market:`` to count business days on the market calendar; append month
    overrides ``;jan=RULE`` or ``;dec=none`` (``parse_rule``).
    """
    rule = {"first_business_day": "nth_business_day:1", "second_business_day": "nth_business_day:2",
            "third_business_day": "nth_business_day:3"}.get(rule, rule)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    bdays = business_days(start - pd.Timedelta(days=40), end + pd.Timedelta(days=40), calendar)
    name, *args = rule.split(":")
    out = []
    for month in _months(start, end):
        days = pd.date_range(month, month + pd.offsets.MonthEnd(0))
        mb = _in_month(bdays, month)
        if name == "nth_business_day":
            pick = mb[int(args[0]) - 1] if len(mb) >= int(args[0]) else None
        elif name == "last_business_day":
            pick = mb[-1] if len(mb) else None
        elif name == "business_day_before_last":
            pick = mb[-1 - int(args[0])] if len(mb) > int(args[0]) else None
        elif name == "nth_weekday":
            wd = days[days.dayofweek == _WEEKDAYS[args[1]]]
            pick = wd[int(args[0]) - 1]
        elif name == "last_weekday":
            pick = days[days.dayofweek == _WEEKDAYS[args[0]]][-1]
        elif name == "day_or_next_business_day":
            after = mb[mb.day >= int(args[0])]
            pick = after[0] if len(after) else None
        elif name == "nth_weekday_after":
            wd = days[(days.dayofweek == _WEEKDAYS[args[1]]) & (days.day >= int(args[2]))]
            pick = wd[int(args[0]) - 1] if len(wd) >= int(args[0]) else None
        else:
            raise ValueError(f"unknown rule {rule!r}")
        if pick is not None and start <= pick <= end:
            out.append(pick)
    return pd.DatetimeIndex(out)


def score(rule: str, observed: pd.DatetimeIndex) -> pd.DataFrame:
    """Per year: of the months with an observed release day, the share where the rule
    gives EXACTLY that day (``hit``), and how many were observed (``n``). Months the rule
    deliberately abstains on (``;dec=none``) are left out, not counted as misses. A release
    that comes out once a month is assumed - the observed day of each month is its first."""
    observed = pd.DatetimeIndex(observed).normalize().unique().sort_values()
    if observed.empty:
        return pd.DataFrame(columns=["year", "n", "hit"])
    first = pd.Series(observed, index=observed.to_period("M")).groupby(level=0).min()
    _, _, overrides = parse_rule(rule)
    first = first[[overrides.get(m.month, "") is not None for m in first.index]]  # months the rule abstains on
    if first.empty:
        return pd.DataFrame(columns=["year", "n", "hit"])
    predicted = pd.Series(rule_dates(rule, observed.min().to_period("M").to_timestamp(), observed.max()))
    predicted.index = pd.DatetimeIndex(predicted).to_period("M")
    hit = first.index.map(lambda m: m in predicted.index and predicted[m] == first[m])
    df = pd.DataFrame({"year": first.index.year, "hit": np.asarray(hit, dtype=float)})
    return df.groupby("year").agg(n=("hit", "size"), hit=("hit", "mean")).reset_index()


CANDIDATES = ("nth_business_day:1", "nth_business_day:2", "nth_business_day:3", "last_business_day",
              "market:last_business_day", *(f"business_day_before_last:{n}" for n in range(1, 9)),
              *(f"day_or_next_business_day:{d}" for d in (14, 15, 16)),
              *(f"nth_weekday:{n}:{d}" for n in (1, 2, 3, 4) for d in ("TUE", "WED", "THU", "FRI")),
              *(f"last_weekday:{d}" for d in ("TUE", "WED", "THU", "FRI")))


def best_rule(observed: pd.DatetimeIndex, *, since: int, candidates=CANDIDATES) -> tuple[str, float, int]:
    """The candidate with the highest hit rate over years >= ``since``: (rule, hit, n)."""
    best = (None, -1.0, 0)
    for rule in candidates:
        s = score(rule, observed)
        s = s[s["year"] >= since]
        if s.empty:
            continue
        n = int(s["n"].sum())
        hit = float((s["hit"] * s["n"]).sum() / n)
        if hit > best[1]:
            best = (rule, hit, n)
    return best


def is_rule(text: str | None) -> bool:
    return bool(text) and re.match(r"^(market:)?(nth_business_day|first_business_day|second_business_day|"
                                   r"third_business_day|last_business_day|business_day_before_last|nth_weekday|"
                                   r"last_weekday|nth_weekday_after|day_or_next_business_day)\b", text) is not None
