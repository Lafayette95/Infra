"""German Federal issuance plans - archive / read / point-in-time state (root CLAUDE.md 17;
pure parsing ``infra.processing.de_issuance``, client ``infra.api.finanzagentur_client``).

Archive ``DE_ISSUANCE_PLAN_DIR`` (a file on disk is its own coverage, Rule 2.1):

* ``outlook/<name>__<Last-Modified UTC>.xlsx`` - every VERSION of each annual / quarterly
  outlook workbook. Versions, not names: the Finanzagentur replaces a file in place when the
  plan changes (the 2026 annual plan, published 2025-12-18, was replaced on 2026-02-04), so a
  version is known from its own ``Last-Modified``, never from the quarter's publication day.
* ``live/<UTC time>__<site|wayback>.html`` - the calendar page whenever its "Upcoming
  Issues" table changed, plus the Wayback Machine's captures (21, 2022-2026).

``plan_state(as_of)`` is the plan as known at ``as_of``: the vintages replayed in order, each
replacing the plan from its own coverage start (an outlook: its first quarter; a live
snapshot: its own day) - so a quarterly update replaces the annual plan from that quarter
on, and the live table the update. ``release_rows`` turns the replay into release-calendar
rows (source ``de_plan``: each replay step an observation, so a line a later vintage drops
falls out point in time) plus the held auctions (``de_auctions``, known from their own day).
"""
from __future__ import annotations

import email.utils
import logging
from pathlib import Path

import pandas as pd

from infra.api import finanzagentur_client as fa
from infra.config import DE_AUCTIONS_DIR, DE_ISSUANCE_PLAN_DIR
from infra.processing import de_issuance as di

log = logging.getLogger(__name__)
PLAN_SOURCE, HELD_SOURCE = "de_plan", "de_auctions"
STAMP = "%Y%m%dT%H%M%SZ"
VINTAGE_COLUMNS = di.PLAN_COLUMNS + ["vintage", "layer", "known_from", "cover_from"]
# network hooks; tests stub them
FETCH_PAGE = fa.fetch_calendar_page
LAST_MODIFIED = fa.outlook_last_modified
FETCH_OUTLOOK = fa.fetch_outlook


def _utc(http_date: str | None, fallback: pd.Timestamp) -> pd.Timestamp:
    if not http_date:
        return fallback
    return pd.Timestamp(email.utils.parsedate_to_datetime(http_date)).tz_convert("UTC").tz_localize(None)


def _archived(root: Path, sub: str) -> list[Path]:
    d = Path(root) / sub
    return sorted(d.glob("*")) if d.exists() else []


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)


# ------------------------------------------------------------------ archive
def archive_outlooks(names, *, root: Path = DE_ISSUANCE_PLAN_DIR, now=None) -> list[Path]:
    """Every listed workbook whose current ``Last-Modified`` isn't archived yet: fetched and
    stored as a new version. One HEAD per name otherwise. A missing name is skipped."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    have = {p.name for p in _archived(root, "outlook")}
    new = []
    for name in sorted(set(names)):
        try:
            lm = _utc(LAST_MODIFIED(name), now)
        except Exception as exc:  # a 404 for a candidate name
            log.debug("outlook %s: %s", name, exc)
            continue
        fname = f"{name[:-5]}__{lm.strftime(STAMP)}.xlsx"
        if fname in have:
            continue
        body, lm2 = FETCH_OUTLOOK(name)
        fname = f"{name[:-5]}__{_utc(lm2, lm).strftime(STAMP)}.xlsx"
        _write(Path(root) / "outlook" / fname, body)
        new.append(Path(root) / "outlook" / fname)
        log.info("issuance outlook: new version %s", fname)
    return new


def _latest_live(root: Path) -> pd.DataFrame | None:
    files = _archived(root, "live")
    return di.parse_upcoming(files[-1].read_text("utf-8", "replace")) if files else None


def update(*, root: Path = DE_ISSUANCE_PLAN_DIR, now=None) -> dict:
    """The daily refresh: the calendar page (archived when its table changed), then every
    outlook it links or that is already archived (a replaced file is a new version)."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    page = FETCH_PAGE()
    rows = di.parse_upcoming(page)
    prev = _latest_live(root)
    live_new = bool(len(rows)) and (prev is None or not rows.reset_index(drop=True).equals(prev.reset_index(drop=True)))
    if live_new:
        _write(Path(root) / "live" / f"{now.strftime(STAMP)}__site.html", page.encode("utf-8"))
    names = set(fa.outlook_links(page)) | {p.name.split("__")[0] + ".xlsx" for p in _archived(root, "outlook")}
    new = archive_outlooks(names, root=root, now=now)
    return {"live_rows": len(rows), "live_changed": live_new, "new_outlooks": [p.name for p in new]}


def backfill(years=(2024, 2025, 2026), *, wayback: bool = True, root: Path = DE_ISSUANCE_PLAN_DIR) -> dict:
    """One-off history: every name the outlooks have used in ``years`` (only the CURRENT
    version of each is on the server), and the Wayback Machine's captures of the calendar page
    (each archived at its capture time)."""
    names = [n for y in years for n in fa.outlook_candidates(y)]
    new = archive_outlooks(names, root=root)
    n_way = 0
    if wayback:
        from infra.api import wayback_client as wb
        have = {p.name.split("__")[0] for p in _archived(root, "live")}
        caps = wb.list_captures(fa.CALENDAR_URL.split("://")[1], pd.Timestamp("2015-01-01"), pd.Timestamp.now())
        for cap in caps.drop_duplicates("digest").itertuples():
            stamp = pd.Timestamp(cap.timestamp).strftime(STAMP)
            if stamp in have:
                continue
            body = wb.fetch_capture(cap.timestamp, cap.original)
            if body[:2] == b"\x1f\x8b":
                import gzip
                body = gzip.decompress(body)
            if di.parse_upcoming(body.decode("utf-8", "replace")).empty:
                log.info("wayback capture %s: no upcoming-issues table, skipped", stamp)
                continue
            _write(Path(root) / "live" / f"{stamp}__wayback.html", body)
            n_way += 1
    return {"new_outlooks": [p.name for p in new], "wayback_snapshots": n_way}


# ------------------------------------------------------------------ vintages and state
def _quarter_start(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).to_period("Q").start_time


def vintages(*, root: Path = DE_ISSUANCE_PLAN_DIR) -> pd.DataFrame:
    """Every archived version as plan rows with ``vintage`` (file), ``layer``
    (``outlook`` / ``live``), ``known_from`` (UTC) and ``cover_from`` (the first auction
    day it speaks for: an outlook's first quarter, a live snapshot's own day)."""
    frames = []
    for p in _archived(root, "outlook"):
        df, _ = di.parse_outlook(p.read_bytes())
        if df.empty:
            continue
        known = pd.Timestamp(p.stem.split("__")[1].rstrip("Z"))
        frames.append(df.assign(vintage=p.name, layer="outlook", known_from=known,
                                cover_from=_quarter_start(df["auction_date"].min())))
    for p in _archived(root, "live"):
        df = di.parse_upcoming(p.read_text("utf-8", "replace"))
        if df.empty:
            continue
        known = pd.Timestamp(p.stem.split("__")[0].rstrip("Z"))
        frames.append(df.assign(vintage=p.name, layer="live", known_from=known, cover_from=known.normalize()))
    if not frames:
        return pd.DataFrame(columns=VINTAGE_COLUMNS)
    return pd.concat(frames, ignore_index=True)[VINTAGE_COLUMNS].sort_values(["known_from", "auction_date"]).reset_index(drop=True)


def replay(v: pd.DataFrame):
    """Yield ``(known_from, plan)`` after each vintage, in order: the vintage replaces the
    plan for auction days on or after its ``cover_from`` (never before its own day - the past
    isn't rewritten by a later file), for the security CLASSES it lists only - the live table
    shows no Green or inflation-linked lines, which the outlooks plan."""
    plan = pd.DataFrame(columns=v.columns)
    for vint, g in v.groupby("vintage", sort=False):
        t = g["known_from"].iloc[0]
        lo = max(g["cover_from"].iloc[0], t.normalize())
        classes = set(g["security"].map(di.security_class))
        replaced = (plan["auction_date"] >= lo) & plan["security"].map(di.security_class).isin(classes)
        plan = pd.concat([plan[~replaced], g[g["auction_date"] >= lo]], ignore_index=True)
        yield t, plan


def plan_state(as_of=None, *, root: Path = DE_ISSUANCE_PLAN_DIR) -> pd.DataFrame:
    """The plan as known at instant ``as_of`` (UTC; None = now): auctions from that day on,
    each with the vintage it comes from."""
    v = vintages(root=root).sort_values("known_from", kind="stable")
    as_of = pd.Timestamp.now(tz="UTC").tz_localize(None) if as_of is None else pd.Timestamp(as_of)
    v = v[v["known_from"] <= as_of]
    state = pd.DataFrame(columns=VINTAGE_COLUMNS)
    for _, state in replay(v):
        pass
    return state[state["auction_date"] >= as_of.normalize()].sort_values("auction_date").reset_index(drop=True)


def _segments(auctions_root: Path) -> dict:
    from infra.pipeline.bunds import read_auctions
    a = read_auctions(root=auctions_root)
    return dict(zip(a["isin"], a["segment"])) if len(a) else {}


def release_rows(*, observed=None, root: Path = DE_ISSUANCE_PLAN_DIR, auctions_root: Path = DE_AUCTIONS_DIR) -> pd.DataFrame:
    """Release-calendar rows (``infra.processing.release_calendar`` columns): the plan replay
    as observations (``de_plan``, each line known from the vintage that first listed it, last
    seen at the last vintage still listing it) and the held auctions (``de_auctions``)."""
    from infra.processing import release_calendar as rc
    from infra.reference.events import EVENTS
    observed = pd.Timestamp.now(tz="UTC").tz_localize(None) if observed is None else pd.Timestamp(observed)
    seg = _segments(auctions_root)
    frames = []
    v = vintages(root=root).sort_values("known_from", kind="stable")
    for t, plan in replay(v):
        if plan.empty:
            continue
        p = plan.assign(event=[di.event_id(s, tm, i, seg) for s, tm, i in zip(plan["security"], plan["term"], plan["isin"])])
        for (ev, kind), g in p.groupby(["event", p["issue_kind"].fillna("")]):
            r = rc.event_rows(EVENTS[ev], g["auction_date"], source=PLAN_SOURCE, known_from=t.normalize(),
                              observed=t.normalize(), stage=kind)
            frames.append(r.assign(last_seen=t.normalize()))
    from infra.pipeline.bunds import read_auctions
    a = read_auctions(root=auctions_root)
    a = a[a["process"].isin(["Auc", "M-A"])]
    if len(a):
        a = a.assign(event=[di.event_id(ty, sg, i, seg) for ty, sg, i in zip(a["type"], a["segment"], a["isin"])],
                     stage=a["issue_kind"].map({"N": "new_issue", "R": "reopening"}).fillna(""))
        for (ev, kind), g in a.groupby(["event", "stage"]):
            days = pd.DatetimeIndex(sorted(set(pd.to_datetime(g["timestamp"]).dt.normalize())))
            frames.append(rc.event_rows(EVENTS[ev], days, source=HELD_SOURCE, known_from=days, observed=observed,
                                        stage=kind))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return rc.empty()
    # one row per (instant, event, source): earliest known_from, latest last_seen
    return rc.merge_observation(frames[0], pd.concat(frames[1:], ignore_index=True)) if len(frames) > 1 else frames[0]
