"""Release calendar (WHEN events happen): fetch / store / read, point-in-time.

Sources: FRED's release dates (``fetch_fred_schedules``, every event with a
``fred_release_id``: past dates since the series began plus the scheduled ones, ~3 months
ahead, no times); the harvested economic calendar (``calendar_schedules``, local: dates
AND times, stages where named - the only date source for ISM, the PMIs, MNI, the
Conference Board, NFIB, NAR); Treasury auctions (infra.pipeline.tsy_auctions, exact
close times). Agency schedules come next.

Store ``~/Database/RawData/ReleaseCalendar`` (flat, hive year/quarter on the release
instant). No coverage manifest: a fetch is ONE observation of the whole schedule, folded
in by ``merge_observation`` (re-fetching is the point - it is how a moved date is seen).
NOT yet in the daily cycle.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import fred_client
from infra.config import MACRO_RELEASES, RELEASE_CALENDAR_DIR
from infra.processing import release_calendar as rc
from infra.reference.events import EVENTS, EconEvent, series_by_bbg
from infra.storage import parquet_store

log = logging.getLogger(__name__)
FRED_SOURCE = "fred_release_dates"
CALENDAR_SOURCE = "marketwatch"
RULE_SOURCE = "rule"
NAR_SOURCE = "nar"
TREASURY_SCHEDULE_SOURCE = "treasury_schedule"
RULE_HORIZON_MONTHS = 12


def calendar_patterns(releases=MACRO_RELEASES) -> dict[str, list[str]]:
    """event id -> the calendar-name patterns of its series (they live on the nowcast's
    release table for now; the registry takes them over in the planned migration)."""
    out: dict[str, list[str]] = {}
    for ticker, rel in releases.items():
        s = series_by_bbg(ticker)
        if s is not None and rel.calendar_pattern:
            out.setdefault(s.event, []).append(rel.calendar_pattern)
    return out


def calendar_schedules(*, events: dict[str, EconEvent] = EVENTS, calendar_root=None) -> pd.DataFrame:
    """LOCAL: release-calendar rows from the harvested economic calendar (dates AND times,
    with the stage where the row names it) - the only date source for ISM, the PMIs,
    MNI Chicago, the Conference Board, NFIB and NAR."""
    from infra.pipeline.econ_calendar import read_calendar_from_disk

    rows = read_calendar_from_disk(**({"root": calendar_root} if calendar_root else {}))
    return rc.from_econ_calendar(rows, calendar_patterns(), events)


def rule_schedules(events: dict[str, EconEvent] = EVENTS, *, observed=None,
                   horizon_months: int = RULE_HORIZON_MONTHS) -> pd.DataFrame:
    """LOCAL: dates PROJECTED by each event's validated ``rule`` from ``observed`` through
    ``horizon_months`` ahead, at the registry's usual time. Source ``"rule"``, known from
    the day projected - a projection, never an announcement: where a dated source (FRED,
    the calendar, Treasury) lists the same release, read that one."""
    from infra.processing import schedule_rules as sr

    observed = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if observed is None else pd.Timestamp(observed)
    end = observed + pd.DateOffset(months=horizon_months)
    frames = []
    for e in events.values():
        if not e.rule:
            continue
        rows = rc.schedule_rows(e, sr.rule_dates(e.rule, observed, end), source=RULE_SOURCE, observed=observed)
        frames.append(rows.assign(stage=e.rule_stage))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else rc.empty()


def nar_schedule(*, observed=None, fetch=None) -> pd.DataFrame:
    """NETWORK: NAR's announced NEXT Existing-Home Sales release (infra.api.nar_client) as
    one release-calendar row at NAR's own time, known from the day it was read."""
    from infra.api import nar_client
    from infra.trading_calendar import snap_instants

    observed = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if observed is None else pd.Timestamp(observed)
    nxt = nar_client.fetch_next_existing_home_sales(**({"fetch": fetch} if fetch else {}))
    if nxt is None:
        log.warning("NAR: next Existing-Home Sales release not found on the page")
        return rc.empty()
    return pd.DataFrame({
        "timestamp": [snap_instants([nxt["day"]], nxt["time_et"], rc.NEW_YORK)[0]], "event": "US_EXISTING_HOME_SALES",
        "source": NAR_SOURCE, "stage": "", "time_source": "source",
        "known_from": [min(nxt["day"], observed)], "last_seen": [observed],
    }).astype({"timestamp": "datetime64[ms]", "known_from": "datetime64[ms]", "last_seen": "datetime64[ms]"})


def treasury_schedule(*, observed=None, fetch=None) -> pd.DataFrame:
    """NETWORK: Treasury's tentative auction schedule (infra.api.treasury_schedule_client)
    - nominal 2y-30y coupon auctions as release-calendar rows at the registry's 13:00
    close, known from the schedule's publication (its refunding date), not from today."""
    from infra.api import treasury_schedule_client as tsc
    from infra.processing.tsy_auctions import NOMINAL_COUPON_TENORS

    observed = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if observed is None else pd.Timestamp(observed)
    sched = tsc.fetch_schedule(**({"fetch": fetch} if fetch else {}))
    if sched.empty:
        return rc.empty()
    years = sched["term"].str.extract(r"^(\d+)-Year$")[0].astype(float)
    nc = sched[sched["security_type"].isin(["NOTE", "BOND"]) & ~sched["tips"] & ~sched["frn"]
               & years.isin(NOMINAL_COUPON_TENORS)].assign(years=years)
    frames = [rc.schedule_rows(EVENTS[f"US_TSY_AUCTION_{int(y)}Y"], pd.DatetimeIndex(g["auction_date"]),
                               source=TREASURY_SCHEDULE_SOURCE, observed=observed).assign(
                  known_from=pd.Timestamp(min(g["published"].min(), observed)))
              for y, g in nc.groupby("years")]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return rc.empty()
    out = pd.concat(frames, ignore_index=True)
    out["known_from"] = out["known_from"].astype("datetime64[ms]")
    return out


def validate_rules(events: dict[str, EconEvent] = EVENTS, *, since: int = 2018, root: Path = RELEASE_CALENDAR_DIR):
    """Every registry rule re-scored against the release days the economic calendar
    OBSERVED (its stage only): ``event, rule, n, hit`` over years >= ``since``."""
    from infra.processing import schedule_rules as sr

    cal = read_release_calendar(root=root)
    cal = cal[(cal["source"] == CALENDAR_SOURCE) & (cal["timestamp"] <= pd.Timestamp.now())]
    out = []
    for e in events.values():
        if not e.rule:
            continue
        d = cal[cal["event"] == e.id]
        d = d[d["stage"] == e.rule_stage] if e.rule_stage else d
        s = sr.score(e.rule, pd.DatetimeIndex(d["timestamp"]).normalize())
        s = s[s["year"] >= since]
        n = int(s["n"].sum()) if len(s) else 0
        out.append((e.id, e.rule, n, float((s["hit"] * s["n"]).sum() / n) if n else float("nan")))
    return pd.DataFrame(out, columns=["event", "rule", "n", "hit"])


def crosscheck_dates(*, root: Path = RELEASE_CALENDAR_DIR) -> pd.DataFrame:
    """For every event dated by BOTH FRED and the calendar, over the span both cover: the
    share of calendar release days FRED also has (date AGREEMENT - expect ~100% unless the
    calendar lists stages FRED doesn't), and of FRED days the calendar also has (COVERAGE:
    the calendar misses unarchived weeks)."""
    cal = read_release_calendar(root=root)
    out = []
    for event, grp in cal.groupby("event"):
        days = {s: set(g["timestamp"].dt.normalize()) for s, g in grp.groupby("source")}
        if FRED_SOURCE not in days or CALENDAR_SOURCE not in days:
            continue
        c, f = days[CALENDAR_SOURCE], days[FRED_SOURCE]
        lo, hi = max(min(c), min(f)), min(max(c), max(f))  # the span BOTH sources cover
        c_span = {d for d in c if lo <= d <= hi}
        f_span = {d for d in f if lo <= d <= hi}
        out.append((event, len(c_span), len(c_span & f) / len(c_span) if c_span else None,
                    len(f_span), len(f_span & c) / len(f_span) if f_span else None))
    return pd.DataFrame(out, columns=["event", "calendar_days", "on_fred", "fred_days", "on_calendar"])


def fetch_fred_schedules(events: dict[str, EconEvent] = EVENTS, *, observed=None,
                         fetch=fred_client.fetch_release_dates) -> pd.DataFrame:
    """NETWORK ONLY: one observation of every FRED-dated event's schedule."""
    observed = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if observed is None else pd.Timestamp(observed)
    frames = [rc.schedule_rows(e, fetch(e.fred_release_id), source=FRED_SOURCE, observed=observed)
              for e in events.values() if e.fred_release_id is not None]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else rc.empty()


def read_release_calendar(as_of=None, *, events: list[str] | None = None, start=None, end=None,
                          root: Path = RELEASE_CALENDAR_DIR) -> pd.DataFrame:
    """The calendar (optionally only ``events``, release instants in ``[start, end)``) as
    known at the end of day ``as_of`` (None = as known now). No network."""
    raw = parquet_store.read_partitioned(root, start=start, end=end,
                                         equals_in=None if events is None else {"event": events})
    if raw is None or raw.empty:
        return rc.empty()
    rows = raw[rc.COLUMNS]
    return rows.sort_values("timestamp").reset_index(drop=True) if as_of is None else rc.as_of(rows, as_of)


def store_observation(observed_rows: pd.DataFrame, *, root: Path = RELEASE_CALENDAR_DIR) -> int:
    """FILES ONLY: fold one observation into the store."""
    if observed_rows.empty:
        return 0
    stored = parquet_store.read_partitioned(root)
    stored = rc.empty() if stored is None else stored[rc.COLUMNS]
    merged = rc.merge_observation(stored, observed_rows)
    for c in ("event", "source", "stage", "time_source"):
        merged[c] = merged[c].fillna("").astype(str)
    parquet_store.write_partitioned(merged, root, rc.KEYS)
    return len(merged)


def refresh_release_calendar(*, root: Path = RELEASE_CALENDAR_DIR, observed=None, fetch=None,
                             with_calendar: bool = True, calendar_root=None, with_rules: bool = True,
                             with_agencies: bool = True, with_fred: bool = True, nar_fetch=None,
                             treasury_fetch=None, errors: dict | None = None) -> int:
    """Parent: observe every source's schedule now (FRED's release dates; the harvested
    economic calendar, local; the validated rules' projections) and fold it in."""
    errors = {} if errors is None else errors
    rows = rc.empty()
    if with_fred:
        try:
            rows = fetch_fred_schedules(observed=observed, **({"fetch": fetch} if fetch else {}))
        except Exception as exc:  # one source never blocks the others
            errors["fred"] = f"{type(exc).__name__}: {exc}"
            log.warning("FRED release dates failed: %s", exc)
    for name, on, build in (("calendar", with_calendar, lambda: calendar_schedules(calendar_root=calendar_root)),
                             ("rules", with_rules, lambda: rule_schedules(observed=observed))):
        if not on:
            continue
        try:  # one source never blocks the others
            rows = pd.concat([rows, build()], ignore_index=True)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            log.warning("release calendar %s failed: %s", name, exc)
    if with_agencies:
        try:
            rows = pd.concat([rows, nar_schedule(observed=observed, fetch=nar_fetch)], ignore_index=True)
        except Exception as exc:  # one publisher's page never blocks the rest of the calendar
            errors["nar"] = f"{type(exc).__name__}: {exc}"
            log.warning("NAR schedule failed: %s", exc)
        try:
            rows = pd.concat([rows, treasury_schedule(observed=observed, fetch=treasury_fetch)], ignore_index=True)
        except Exception as exc:
            errors["treasury_schedule"] = f"{type(exc).__name__}: {exc}"
            log.warning("Treasury tentative auction schedule failed: %s", exc)
    n = store_observation(rows, root=root)
    log.info("release calendar: %d rows observed, %d stored", len(rows), n)
    return n
