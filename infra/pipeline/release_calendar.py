"""Release calendar (WHEN events happen): fetch / store / read, point-in-time.

Sources: FRED's release dates (``fetch_fred_schedules``, every event with a
``fred_release_id``: past dates since the series began plus the scheduled ones, ~3 months
ahead, no times); the harvested economic calendar (``calendar_schedules``, local: dates
AND times, stages where named - the only date source for ISM, the PMIs, MNI, the
Conference Board, NFIB, NAR); Treasury auctions (infra.pipeline.tsy_auctions, exact
close times); and DERIVED dates (``derived_schedules``: central-bank decisions from verified
config lists, the Treasury lifecycle from the auctions store, futures contract calendars).

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
from infra.config import RELEASE_CALENDAR_DIR
from infra.processing import release_calendar as rc
from infra.reference.events import EVENTS, SERIES, EconEvent
from infra.storage import parquet_store

log = logging.getLogger(__name__)
FRED_SOURCE = "fred_release_dates"
CALENDAR_SOURCE = "marketwatch"
RULE_SOURCE = "rule"
NAR_SOURCE = "nar"
TREASURY_SCHEDULE_SOURCE = "treasury_schedule"
RULE_HORIZON_MONTHS = 12


def calendar_patterns(series: dict | None = None) -> dict[str, list[str]]:
    """event id -> the calendar-name patterns of its registry series."""
    series = SERIES if series is None else series
    out: dict[str, list[str]] = {}
    for s_ in series.values():
        if s_.calendar_pattern:
            out.setdefault(s_.event, []).append(s_.calendar_pattern)
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


def german_auctions(*, observed=None, plan_root: Path | None = None, auctions_root: Path | None = None,
                    errors: dict | None = None) -> pd.DataFrame:
    """German Federal auctions (``infra.pipeline.de_issuance``): refresh the archived plan
    (NETWORK: the calendar page and its outlook workbooks), then rows from the archive - the
    plan replayed point in time and the held auctions. A fetch failure still yields the rows
    already archived."""
    from infra.config import DE_AUCTIONS_DIR, DE_ISSUANCE_PLAN_DIR
    from infra.pipeline import de_issuance

    errors = {} if errors is None else errors
    plan_root, auctions_root = plan_root or DE_ISSUANCE_PLAN_DIR, auctions_root or DE_AUCTIONS_DIR
    try:
        de_issuance.update(root=plan_root)
    except Exception as exc:
        errors["de_plan_fetch"] = f"{type(exc).__name__}: {exc}"
        log.warning("German issuance plan refresh failed: %s", exc)
    try:
        return de_issuance.release_rows(observed=observed, root=plan_root, auctions_root=auctions_root)
    except Exception as exc:
        errors["de_auctions"] = f"{type(exc).__name__}: {exc}"
        log.warning("German auction rows failed: %s", exc)
        return rc.empty()


def japanese_auctions(*, observed=None, files_root: Path | None = None, auctions_root: Path | None = None,
                      errors: dict | None = None) -> pd.DataFrame:
    """JGB auctions (``infra.pipeline.jgb_auctions``): refresh the archived workbooks and
    calendar pages (NETWORK), then rows from the archive - the plan replayed point in time and
    the held auctions. A fetch failure still yields the rows already archived."""
    from infra.config import JP_AUCTION_FILES_DIR, JP_AUCTIONS_DIR
    from infra.pipeline import jgb_auctions

    errors = {} if errors is None else errors
    files_root, auctions_root = files_root or JP_AUCTION_FILES_DIR, auctions_root or JP_AUCTIONS_DIR
    try:
        fetch_errors: dict = {}
        jgb_auctions.update(root=files_root, auctions_root=auctions_root, errors=fetch_errors)
        errors.update(fetch_errors)
    except Exception as exc:
        errors["jp_auctions_fetch"] = f"{type(exc).__name__}: {exc}"
        log.warning("JGB auction refresh failed: %s", exc)
    try:
        return jgb_auctions.release_rows(observed=observed, root=files_root, auctions_root=auctions_root)
    except Exception as exc:
        errors["jp_auctions"] = f"{type(exc).__name__}: {exc}"
        log.warning("JGB auction rows failed: %s", exc)
        return rc.empty()


def canadian_auctions(*, observed=None, roots: dict | None = None, errors: dict | None = None) -> pd.DataFrame:
    """Government of Canada auctions (``infra.pipeline.goc_auctions``): refresh the Valet
    results, outstanding securities and the bond schedule (NETWORK), then rows from disk.
    ``roots``: ``auctions_root`` / ``outstanding_root`` / ``plan_root`` (default config)."""
    from infra.pipeline import goc_auctions

    errors = {} if errors is None else errors
    roots = roots or {}
    try:
        fetch_errors: dict = {}
        goc_auctions.update(**roots, errors=fetch_errors)
        errors.update(fetch_errors)
    except Exception as exc:
        errors["ca_auctions_fetch"] = f"{type(exc).__name__}: {exc}"
        log.warning("Canadian auction refresh failed: %s", exc)
    try:
        kw = {k: v for k, v in roots.items() if k in ("auctions_root",)}
        if "plan_root" in roots:
            kw["root"] = roots["plan_root"]
        return goc_auctions.release_rows(observed=observed, **kw)
    except Exception as exc:
        errors["ca_auctions"] = f"{type(exc).__name__}: {exc}"
        log.warning("Canadian auction rows failed: %s", exc)
        return rc.empty()


CALENDAR_FROM = "2000-01-01"        # calendar anchors: generated from here ...
CALENDAR_YEARS_AHEAD = 2            # ... to this far past the observation day


def derived_schedules(*, observed=None, auctions_root: Path | None = None, contracts_file: Path | None = None,
                      daily_root: Path | None = None, roll_start="2015-01-01", with_rolls: bool = True,
                      errors: dict | None = None) -> pd.DataFrame:
    """LOCAL (disk + config, no network): the registry events no publisher dates for us
    (``infra.processing.event_dates``) - FOMC / ECB / BoE decisions from the verified config
    lists, the Treasury lifecycle and refunding dates from the auctions store, the futures
    contract calendars from the contracts table and CME's rules, and the v.0 roll days from
    the stored daily relative series. Each part fails alone (``errors``)."""
    from infra.config import (CENTRAL_BANK_MEETINGS, DAILY_FUTURES_DIR, FOMC_MEETINGS, FUTURES_CONTRACTS_FILE,
                              TSY_AUCTIONS_DIR)
    from infra.processing import event_dates as ed

    errors = {} if errors is None else errors
    observed = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if observed is None else pd.Timestamp(observed)
    frames = []

    def part(name, build):
        try:
            frames.append(build())
        except Exception as exc:  # one derivation never blocks the others
            errors[name] = f"{type(exc).__name__}: {exc}"
            log.warning("derived event dates %s failed: %s", name, exc)

    part("fomc", lambda: ed.fomc_rows(EVENTS["US_FOMC_DECISION"], FOMC_MEETINGS, observed))
    from infra.reference.events import CALENDAR_RULES
    part("calendar", lambda: ed.calendar_rows(EVENTS, CALENDAR_RULES, CALENDAR_FROM,
                                              observed + pd.DateOffset(years=CALENDAR_YEARS_AHEAD), observed))
    for event_id, meetings in CENTRAL_BANK_MEETINGS.items():
        part(event_id, lambda e=event_id, m=meetings: ed.central_bank_rows(EVENTS[e], m, observed))

    def treasury():
        from infra.pipeline.tsy_auctions import read_auctions
        from infra.processing.tsy_auctions import nominal_coupons
        auctions = read_auctions(root=auctions_root or TSY_AUCTIONS_DIR)
        return ed.treasury_rows(nominal_coupons(auctions) if len(auctions) else auctions, EVENTS, observed)
    part("treasury", treasury)

    def futures():
        from infra.processing.schedule_rules import business_days
        f = contracts_file or FUTURES_CONTRACTS_FILE
        if not Path(f).exists():
            return rc.empty()
        contracts = pd.read_parquet(f)
        rows, check = ed.futures_rows(contracts, EVENTS, business_days("2000-01-01", "2045-12-31", "market"),
                                         observed)
        if len(check) and not check["match"].all():
            bad = check[~check["match"]]
            errors["futures_rule_mismatch"] = (f"{len(bad)} CME Treasury contract(s) whose stored expiry differs "
                                               f"from CME's last-trading rule: "
                                               + ", ".join(f"{r.ticker} {r.expiry.date()}" for r in bad.head(5).itertuples()))
        return rows
    part("futures", futures)

    if with_rolls:
        def rolls():
            from infra.relative.symbology import parse_relative
            from infra.pipeline.relative_daily import load_relative_daily
            roots = [e.id.split("_")[1] for e in EVENTS.values() if e.id.endswith("_ROLL_V0")]
            kw = {"fetch_missing": False}
            if daily_root is not None:
                kw["daily_root"] = daily_root
            if contracts_file is not None:
                kw["contracts_file"] = contracts_file
            rel = load_relative_daily([parse_relative(f"{r}.v.0") for r in roots], roll_start,
                                      observed + pd.Timedelta(days=1), **kw)
            return ed.roll_rows(rel, EVENTS, observed)
        part("rolls", rolls)
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else rc.empty()


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
                             treasury_fetch=None, with_derived: bool = True, auctions_root: Path | None = None,
                             contracts_file: Path | None = None, daily_root: Path | None = None,
                             de_plan_root: Path | None = None, de_auctions_root: Path | None = None,
                             jp_files_root: Path | None = None, jp_auctions_root: Path | None = None,
                             ca_roots: dict | None = None, errors: dict | None = None) -> int:
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
        rows = pd.concat([rows, german_auctions(observed=observed, plan_root=de_plan_root,
                                                auctions_root=de_auctions_root, errors=errors)], ignore_index=True)
        rows = pd.concat([rows, japanese_auctions(observed=observed, files_root=jp_files_root,
                                                  auctions_root=jp_auctions_root, errors=errors)], ignore_index=True)
        rows = pd.concat([rows, canadian_auctions(observed=observed, roots=ca_roots, errors=errors)], ignore_index=True)
    if with_derived:
        derived_errors: dict = {}
        rows = pd.concat([rows, derived_schedules(observed=observed, auctions_root=auctions_root,
                                                  contracts_file=contracts_file, daily_root=daily_root,
                                                  errors=derived_errors)], ignore_index=True)
        errors.update({f"derived:{k}": v for k, v in derived_errors.items()})
    n = store_observation(rows, root=root)
    log.info("release calendar: %d rows observed, %d stored", len(rows), n)
    return n
