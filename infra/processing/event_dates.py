"""Event dates DERIVED from data already stored, or from verified config lists - release-
calendar rows (infra.processing.release_calendar.COLUMNS) for the registry events that no
publisher's calendar dates for us. Pure: frames in, rows out; the reads are in
``infra.pipeline.release_calendar.derived_schedules``.

* Central banks (``central_bank_rows``, ``fomc_rows``): decision days from config lists
  verified against each bank's own calendar, at the registry's time (or the meeting's own:
  the ECB announced at 13:45 before 21 Jul 2022), known from the day the schedule was
  published; an unscheduled meeting (stage ``unscheduled``) is known only on its day.
* US Treasury (``treasury_rows``): from the auctions store (nominal 2y-30y coupons):
  each auction's issue date (``US_TSY_ISSUE_<t>Y``, stage ``new_issue`` / ``reopening``),
  the on-the-run roll (``US_TSY_OTR_ROLL_<t>Y``, the issue date of each NEW issue),
  the quarterly refunding statement (the refunding auctions' announcement date) and the
  borrowing estimates (the Monday two days earlier), from ``REFUNDING_FROM``.
* Futures (``futures_rows``): the last trading day of every contract (the contracts
  table's expiry), the CME Treasury roots' first notice / first delivery / last delivery
  days (CME's rules, ``infra.analytics.futures_basis.delivery_window``), and the days the
  volume-ranked ``<root>.v.0`` switches contract (``roll_rows``). Stage = the contract.
"""
from __future__ import annotations

import pandas as pd

from infra.analytics.futures_basis import delivery_window
from infra.processing import release_calendar as rc

CB_SOURCE = "cb_calendar"
FOMC_SOURCE = "fomc_calendar"
TREASURY_SOURCE = "auctions_derived"
CONTRACTS_SOURCE = "contracts"
ROLL_SOURCE = "relative_v0"
CALENDAR_SOURCE = "calendar"
REFUNDING_MONTHS = (2, 5, 8, 11)
# The Monday 15:00 estimates / Wednesday 08:30 statement pattern is verified for 2016 (May:
# estimates Mon 2 May, statement Wed 4 May 08:30) and 2026 (Feb: Mon 2 Feb 15:00, Wed 4 Feb
# 08:30); earlier years are not emitted until verified (TOFIX.md).
REFUNDING_FROM = 2016
REFUNDING_TENORS = (3, 10, 30)


def _day_level_unscheduled(rows: pd.DataFrame, scheduled: list[bool]) -> pd.DataFrame:
    """An unscheduled decision's time is not the usual one (and not verified): it sits at
    00:00 in the bank's zone, ``time_source="unknown"``."""
    rows["time_source"] = [ts if s else "unknown" for ts, s in zip(rows["time_source"], scheduled)]
    return rows


def central_bank_rows(event, meetings, observed) -> pd.DataFrame:
    """Decision rows of one bank: ``meetings`` = ``infra.config.CentralBankMeeting``s."""
    if not meetings:
        return rc.empty()
    scheduled = [m.scheduled for m in meetings]
    rows = rc.event_rows(
        event, [m.decision for m in meetings], source=CB_SOURCE, observed=observed,
        known_from=[m.published if m.scheduled else m.decision for m in meetings],
        stage=["" if m.scheduled else "unscheduled" for m in meetings],
        times=[m.time_local if m.scheduled else "00:00" for m in meetings])
    # a meeting's own (verified) time is still the registry-level fact, not a source's
    rows["time_source"] = ["registry" if s else "unknown" for s in scheduled]
    return rows


def fomc_rows(event, meetings, observed) -> pd.DataFrame:
    """FOMC decisions (``infra.config.FOMC_MEETINGS``, ``end_date`` = announcement day).
    The Fed publishes a year's calendar around July-August of the year before; with no
    stored publication date, a scheduled meeting is taken as known from 31 December of the
    prior year (LATER than the truth: conservative, never look-ahead). Unscheduled: known
    on the day, time not verified."""
    if not meetings:
        return rc.empty()
    days = [m.end_date for m in meetings]
    scheduled = [m.scheduled for m in meetings]
    rows = rc.event_rows(
        event, days, source=FOMC_SOURCE, observed=observed,
        known_from=[f"{int(d[:4]) - 1}-12-31" if m.scheduled else d for d, m in zip(days, meetings)],
        stage=["" if m.scheduled else "unscheduled" for m in meetings],
        times=[None if m.scheduled else "00:00" for m in meetings])
    return _day_level_unscheduled(rows, scheduled)


def treasury_rows(coupons: pd.DataFrame, events: dict, observed) -> pd.DataFrame:
    """``coupons`` = ``infra.processing.tsy_auctions.nominal_coupons`` of the auctions
    store (columns announcemt_date, auction_date, issue_date, reopening, tenor_years)."""
    if coupons is None or coupons.empty:
        return rc.empty()
    a = coupons.copy()
    for c in ("announcemt_date", "auction_date", "issue_date"):
        a[c] = pd.to_datetime(a[c])
    a = a.dropna(subset=["auction_date", "issue_date"])
    a["announced"] = a["announcemt_date"].fillna(a["auction_date"])
    reopen = a["reopening"].astype(str).str.lower().eq("yes")
    frames = []
    for t, g in a.groupby(a["tenor_years"].astype(int)):
        r = reopen.loc[g.index]
        frames.append(rc.event_rows(events[f"US_TSY_ISSUE_{t}Y"], g["issue_date"], source=TREASURY_SOURCE,
                                    observed=observed, known_from=g["announced"].tolist(),
                                    stage=["reopening" if x else "new_issue" for x in r]))
        new = g[~r]
        frames.append(rc.event_rows(events[f"US_TSY_OTR_ROLL_{t}Y"], new["issue_date"], source=TREASURY_SOURCE,
                                    observed=observed, known_from=new["announced"].tolist()))
    # quarterly refunding: the refunding auctions (Feb/May/Aug/Nov 3y/10y/30y) are announced
    # in the refunding statement itself - their announcement date IS the statement's date
    ref = a[a["auction_date"].dt.month.isin(REFUNDING_MONTHS) & a["tenor_years"].astype(int).isin(REFUNDING_TENORS)]
    ref = ref.dropna(subset=["announcemt_date"])
    ref = ref[ref["auction_date"].dt.year >= REFUNDING_FROM]
    days = ref.groupby(ref["auction_date"].dt.to_period("M"))["announcemt_date"].min()
    if len(days):
        stmt = pd.DatetimeIndex(days.to_numpy())
        frames.append(rc.event_rows(events["US_TSY_REFUNDING"], stmt, source=TREASURY_SOURCE, observed=observed,
                                    known_from=list(stmt)))
        est = stmt - pd.Timedelta(days=2)  # the Monday before the Wednesday statement
        frames.append(rc.event_rows(events["US_TSY_BORROWING_ESTIMATES"], est, source=TREASURY_SOURCE,
                                    observed=observed, known_from=list(est)))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else rc.empty()


def futures_rows(contracts: pd.DataFrame, events: dict, business_days: pd.DatetimeIndex, observed,
                 *, cme_treasury_roots=("ZT", "ZF", "ZN", "TN", "ZB", "UB")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows, check). ``contracts``: root, ticker, expiry, activation. Each contract is
    known from its listing (``activation``; the expiry day itself when missing). ``check``:
    per CME Treasury contract, the stored expiry vs CME's last-trading rule (``match``)."""
    if contracts is None or contracts.empty:
        return rc.empty(), pd.DataFrame()
    c = contracts.dropna(subset=["expiry"]).copy()
    c["expiry"] = pd.to_datetime(c["expiry"]).dt.normalize()
    c["known"] = pd.to_datetime(c["activation"]).dt.normalize().fillna(c["expiry"])
    frames, checks = [], []
    for root, g in c.groupby("root"):
        ev = events.get(f"FUT_{root}_LAST_TRADE")
        if ev is None:
            continue
        frames.append(rc.event_rows(ev, g["expiry"], source=CONTRACTS_SOURCE, observed=observed,
                                    known_from=g["known"].tolist(), stage=g["ticker"].astype(str).tolist()))
        if root not in cme_treasury_roots:
            continue
        rows = {k: [] for k in ("FIRST_NOTICE", "FIRST_DELIVERY", "LAST_DELIVERY")}
        for r in g.itertuples(index=False):
            w = delivery_window(root, r.expiry, business_days)
            before = business_days[business_days < w["first_delivery"]]
            rows["FIRST_NOTICE"].append(before[-1])
            rows["FIRST_DELIVERY"].append(w["first_delivery"])
            rows["LAST_DELIVERY"].append(w["last_delivery"])
            checks.append({"root": root, "ticker": r.ticker, "expiry": r.expiry, "rule_last_trading": w["last_trading"],
                           "match": w["last_trading"] == r.expiry})
        for k, days in rows.items():
            frames.append(rc.event_rows(events[f"FUT_{root}_{k}"], days, source=CONTRACTS_SOURCE, observed=observed,
                                        known_from=g["known"].tolist(), stage=g["ticker"].astype(str).tolist()))
    frames = [f for f in frames if not f.empty]
    return (pd.concat(frames, ignore_index=True) if frames else rc.empty()), pd.DataFrame(checks)


def roll_rows(relative: pd.DataFrame, events: dict, observed) -> pd.DataFrame:
    """``relative``: stored daily relative series (``timestamp, ticker, contract``) of
    ``<root>.v.0`` tickers -> a row on each day the mapped contract changes (stage = the new
    contract). The ranking uses only prior days' volume, so the switch is known before the
    day; the row claims it from the day itself."""
    if relative is None or relative.empty:
        return rc.empty()
    frames = []
    for ticker, g in relative.sort_values("timestamp").groupby("ticker"):
        root = str(ticker).split(".")[0]
        ev = events.get(f"FUT_{root}_ROLL_V0")
        if ev is None:
            continue
        con = g["contract"].astype(str)
        switch = g[con.ne(con.shift()) & con.shift().notna()]
        if switch.empty:
            continue
        days = pd.to_datetime(switch["timestamp"]).dt.normalize()
        frames.append(rc.event_rows(ev, days, source=ROLL_SOURCE, observed=observed, known_from=list(days),
                                    stage=switch["contract"].astype(str).tolist()))
    return pd.concat(frames, ignore_index=True) if frames else rc.empty()


def calendar_rows(events: dict, rules: dict[str, str], start, end, observed) -> pd.DataFrame:
    """Calendar anchors (``infra.reference.events.CALENDAR_RULES``) on every date their rule gives in
    ``[start, end]``, day-level, source ``calendar``. Known from 31 December of the year before
    (deterministic, but market holidays are fixed about a year ahead: conservative)."""
    from infra.processing.schedule_rules import rule_dates
    frames = []
    for event_id, rule in rules.items():
        days = rule_dates(rule, start, end)
        if len(days):
            frames.append(rc.event_rows(events[event_id], days, source=CALENDAR_SOURCE, observed=observed,
                                        known_from=[f"{d.year - 1}-12-31" for d in days]))
    return pd.concat(frames, ignore_index=True) if frames else rc.empty()
