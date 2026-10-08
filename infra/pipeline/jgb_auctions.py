"""Japanese Government Bond auctions - archive / store / point-in-time plan (root CLAUDE.md
17; pure parsing ``infra.processing.jgb_auctions``, client ``infra.api.mof_client``).

* Archive ``JP_AUCTION_FILES_DIR``: the MoF's three results workbooks and every auction-
  calendar page (``<yymm>e``) and alteration notice (``<yymm>ae``), each VERSION by its
  ``Last-Modified`` (a file on disk is its own coverage; one HEAD per file a run).
* Auctions ``JP_AUCTIONS_DIR`` (keys ``timestamp`` = auction day, ``security_id``): rebuilt
  from the latest workbooks. ``read_securities(as_of)`` is a VIEW of it (one row per bond /
  bill from its first auction) - the JGB universe.
* Plan: each month's calendar (the JAPANESE pages, published ahead - the English ones only
  once the month starts), known from the END of the month three months before (the MoF
  announces it on the 24th-28th of that month; ``announced_by``) or its first archived
  version if earlier, replaced by an alteration notice (English, ``<yymm>ae``) from the
  notice's own day (its "before" table is the original calendar).
  ``release_rows`` replays it as release-calendar observations (``jp_plan``) next to the
  held auctions (``jp_auctions``, known from their own day). Calendar pages exist from 2023.
"""
from __future__ import annotations

import email.utils
import logging
from pathlib import Path

import pandas as pd

from infra.api import mof_client as mc
from infra.config import JP_AUCTION_FILES_DIR, JP_AUCTIONS_DIR
from infra.processing import jgb_auctions as ja
from infra.storage import parquet_store

log = logging.getLogger(__name__)
PLAN_SOURCE, HELD_SOURCE = "jp_plan", "jp_auctions"
KEYS = ["timestamp", "security_id"]
STAMP = "%Y%m%dT%H%M%SZ"
REFRESH_MONTHS = 2   # calendar pages from this many months back are re-checked each run
# network hooks; tests stub them
FETCH = mc.fetch_url
LAST_MODIFIED = mc.last_modified
_ONE_DAY = pd.Timedelta(days=1)


def _stamp(http_date: str | None) -> str:
    if not http_date:
        return pd.Timestamp.now(tz="UTC").strftime(STAMP)
    return pd.Timestamp(email.utils.parsedate_to_datetime(http_date)).tz_convert("UTC").strftime(STAMP)


def _versions(root: Path, sub: str, stem: str) -> list[Path]:
    d = Path(root) / sub
    return sorted(d.glob(f"{stem}__*")) if d.exists() else []


def _archive(url: str, sub: str, stem: str, ext: str, root: Path) -> Path | None:
    """The file's current version, fetched only if its Last-Modified isn't archived."""
    have = {p.name for p in _versions(root, sub, stem)}
    if f"{stem}__{_stamp(LAST_MODIFIED(url))}.{ext}" in have:
        return None
    body, lm = FETCH(url)
    path = Path(root) / sub / f"{stem}__{_stamp(lm)}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    log.info("JGB auctions: new %s", path.name)
    return path


# ------------------------------------------------------------------ fetch / store
def update(*, root: Path = JP_AUCTION_FILES_DIR, auctions_root: Path = JP_AUCTIONS_DIR, now=None,
           errors: dict | None = None) -> dict:
    """The daily refresh: the results workbooks and the calendar pages (every page not yet
    archived, and the last ``REFRESH_MONTHS`` months onward again), then the auctions store
    rebuilt if a workbook changed. Each file fails alone (``errors``)."""
    errors = {} if errors is None else errors
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    new_results = []
    for stem, rel in mc.RESULT_FILES.items():
        try:
            p = _archive(mc.AUCTION_BASE + rel, "results", stem, "xls", root)
            new_results += [p] if p else []
        except Exception as exc:
            errors[f"jp_{stem}"] = f"{type(exc).__name__}: {exc}"
    new_pages = 0
    recent = (now - pd.DateOffset(months=REFRESH_MONTHS)).strftime("%y%m")
    # month calendars from the Japanese site (published ahead), alteration notices from the English
    for index_url, base, keep in ((mc.CALENDAR_INDEX_JA, mc.AUCTION_BASE_JA, lambda c: c.isdigit()),
                                  (mc.CALENDAR_INDEX, mc.AUCTION_BASE, lambda c: c.endswith("ae"))):
        try:
            index, _ = FETCH(index_url)
            codes = [c for c in mc.calendar_links(index.decode("utf-8", "replace")) if keep(c)]
        except Exception as exc:
            errors[f"jp_calendar_index_{'ja' if base == mc.AUCTION_BASE_JA else 'en'}"] = f"{type(exc).__name__}: {exc}"
            continue
        for code in codes:
            if _versions(root, "calendar", code) and code[:4] < recent:
                continue
            try:
                new_pages += _archive(base + f"calendar/{code}.htm", "calendar", code, "html", root) is not None
            except Exception as exc:
                errors[f"jp_calendar_{code}"] = f"{type(exc).__name__}: {exc}"
    rows = store_auctions(root=root, auctions_root=auctions_root) if new_results or not parquet_store.has_data(auctions_root) else 0
    return {"new_results": [p.name for p in new_results], "new_pages": new_pages, "auction_rows": rows}


def _latest(root: Path, sub: str, stem: str) -> Path | None:
    v = _versions(root, sub, stem)
    return v[-1] if v else None


def parse_archive(*, root: Path = JP_AUCTION_FILES_DIR) -> pd.DataFrame:
    """Every auction in the latest archived workbooks."""
    parsers = {"jgb": ja.parse_results, "tbill": ja.parse_tbills, "liquidity": ja.parse_liquidity}
    frames = [parsers[stem](p.read_bytes()) for stem in parsers if (p := _latest(root, "results", stem))]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ja.AUCTION_COLUMNS)


def store_auctions(*, root: Path = JP_AUCTION_FILES_DIR, auctions_root: Path = JP_AUCTIONS_DIR) -> int:
    df = parse_archive(root=root)
    if df.empty:
        return 0
    df = df.drop_duplicates(KEYS, keep="last")   # a Liquidity Enhancement day appears once
    out = df.assign(**{c: pd.to_datetime(df[c]).astype("datetime64[ms]") for c in ("timestamp", "issue_date", "maturity_date")},
                    issue_number=df["issue_number"].astype("Int64"), tenor_months=df["tenor_months"].astype("float"))
    parquet_store.write_partitioned(out, auctions_root, KEYS)
    return len(out)


def read_auctions(start=None, end=None, *, root: Path = JP_AUCTIONS_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end)) \
        if parquet_store.has_data(root) else None
    if df is None or df.empty:
        return pd.DataFrame(columns=ja.AUCTION_COLUMNS)
    for c in ("security_id", "kind"):
        df[c] = df[c].astype(str)
    return df[ja.AUCTION_COLUMNS].sort_values(KEYS).reset_index(drop=True)


def read_securities(as_of=None, *, root: Path = JP_AUCTIONS_DIR) -> pd.DataFrame:
    """The JGB / T-bill universe: one row per security first auctioned by ``as_of``."""
    a = read_auctions(None, None if as_of is None else pd.Timestamp(as_of).normalize() + _ONE_DAY, root=root)
    return ja.securities(a)


# ------------------------------------------------------------------ plan
def plan_vintages(*, root: Path = JP_AUCTION_FILES_DIR) -> list[tuple[pd.Timestamp, str, pd.DataFrame]]:
    """``(known_from, month code, rows)`` for every month's calendar (the Japanese
    ``<yymm>`` pages): the original - known from the end of the month three months before,
    or the page's first archived version if earlier; for a month with an alteration notice its
    "before" table, known from the date the notice gives - and each alteration from the
    notice's own day, in time order."""
    d = Path(root) / "calendar"
    codes = sorted({p.name.split("__")[0] for p in d.glob("*")}) if d.exists() else []
    out = []
    for code in [c for c in codes if c.isdigit()]:
        first = _versions(root, "calendar", code)[0]
        seen = pd.Timestamp(first.name.split("__")[1].split(".")[0].rstrip("Z")).normalize()
        known = min(ja.announced_by(code), seen)
        alt = _latest(root, "calendar", code + "ae")
        if alt is not None:
            notice, announced, before, after = ja.parse_alteration(alt.read_text("utf-8", "replace"), code + "ae")
            out.append((announced if announced is not None else known, code, before))
            out.append((notice, code, after))
        else:
            page = _latest(root, "calendar", code)
            out.append((known, code, ja.parse_month(page.read_text("utf-8", "replace"), code)))
    return sorted(out, key=lambda v: v[0])


def plan_state(as_of=None, *, root: Path = JP_AUCTION_FILES_DIR) -> pd.DataFrame:
    """The auction calendar as known at the end of day ``as_of`` (None = now), auctions from
    that day on."""
    as_of = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of).normalize()
    months = {}
    for t, code, rows in plan_vintages(root=root):
        if t <= as_of:
            months[code] = rows
    frames = [r for r in months.values() if len(r)]
    if not frames:
        return pd.DataFrame(columns=ja.PLAN_COLUMNS)
    plan = pd.concat(frames, ignore_index=True)
    return plan[plan["auction_date"] >= as_of].sort_values("auction_date").reset_index(drop=True)


def release_rows(*, observed=None, root: Path = JP_AUCTION_FILES_DIR, auctions_root: Path = JP_AUCTIONS_DIR) -> pd.DataFrame:
    """Release-calendar rows: the plan replayed (each vintage one observation of the whole
    known calendar, so a line an alteration drops falls out point in time) and the held
    auctions (stage new_issue / reopening by the security's first auction)."""
    from infra.processing import release_calendar as rc
    from infra.reference.events import EVENTS
    observed = pd.Timestamp.now().normalize() if observed is None else pd.Timestamp(observed)
    frames, months = [], {}
    for t, code, rows in plan_vintages(root=root):
        months[code] = rows
        plan = pd.concat([r for r in months.values() if len(r)], ignore_index=True) if months else None
        if plan is None or plan.empty:
            continue
        ev = [ja.event_id(k, m) for k, m in zip(plan["kind"], plan["tenor_months"])]
        for e, g in plan.assign(event=ev).groupby("event"):
            frames.append(rc.event_rows(EVENTS[e], g["auction_date"], source=PLAN_SOURCE, known_from=t, observed=t))
    a = read_auctions(root=auctions_root)
    if len(a):
        first = a.groupby("security_id")["timestamp"].transform("min")
        a = a.assign(event=[ja.event_id(k, m) for k, m in zip(a["kind"], a["tenor_months"])],
                     stage=(a["timestamp"] == first).map({True: "new_issue", False: "reopening"}).where(a["kind"] != "LIQ", ""))
        for (e, st), g in a.groupby(["event", "stage"]):
            days = pd.DatetimeIndex(sorted(set(pd.to_datetime(g["timestamp"]).dt.normalize())))
            frames.append(rc.event_rows(EVENTS[e], days, source=HELD_SOURCE, known_from=days, observed=observed, stage=st))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return rc.empty()
    return rc.merge_observation(frames[0], pd.concat(frames[1:], ignore_index=True)) if len(frames) > 1 else frames[0]
