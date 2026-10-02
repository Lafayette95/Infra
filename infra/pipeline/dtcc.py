"""DTCC swap-trade archive (CFTC Part 43 public dissemination): plan / fetch / store / read
+ a ``load`` parent, over ``Database/RawData/DTCC``.

* **Raw files, byte-for-byte:** each day's cumulative zip is kept exactly as DTCC published
  it, at ``DTCC/CFTC_<KIND>/<year>/CFTC_CUMULATIVE_<KIND>_YYYY_MM_DD.zip``. DTCC only keeps
  a rolling ~2 years, so the archive - not a parsed extract - is the record: any later
  extraction (swap curves, package trades, other currencies) re-reads it.
* **The file on disk IS the coverage** (Rule 2.1), no manifest: a file is only ever written
  whole, after its zip is verified (``store_dtcc_raw``), so its existence means the day is
  archived, and a manifest could only drift from the disk. An archived day is never
  requested again; DTCC publishes each day once, complete (``infra.api.dtcc_client``).
* **Days are UTC days** - DTCC's own: file D holds what was disseminated over UTC day D, so
  a New York afternoon is in file D but a New York evening (20:00-24:00 EDT) in file D+1.
* A day is requested only once it has ended in UTC and only from ``DTCC_FIRST_DAY`` on
  (earlier days are gone from DTCC).
"""
from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.api import dtcc_client
from infra.config import DTCC_DIR, DTCC_FIRST_DAY
from infra.coverage.intervals import to_utc_day

log = logging.getLogger(__name__)

# fn(kind, day) -> zip bytes, or None when DTCC has no file for the day (network only)
FETCH: Callable[[str, pd.Timestamp], bytes | None] = dtcc_client.fetch_cumulative
PAUSE_S = 0.5  # between consecutive requests: the CDN answers 503 to bursts


def path_for(kind: str, day, *, root: Path = DTCC_DIR) -> Path:
    day = pd.Timestamp(day)
    return root / f"CFTC_{kind}" / f"{day:%Y}" / dtcc_client.file_name(kind, day)


def archived_days(kind: str, *, root: Path = DTCC_DIR) -> set[pd.Timestamp]:
    """Every day whose file is on disk. No network."""
    prefix = f"CFTC_CUMULATIVE_{kind}_"
    days = set()
    for f in (root / f"CFTC_{kind}").glob(f"*/{prefix}*.zip"):
        days.add(pd.Timestamp(f.name[len(prefix):-4].replace("_", "-")))
    return days


def last_complete_day(now=None) -> pd.Timestamp:
    """The latest UTC day that has ended - the latest day DTCC can have published."""
    return to_utc_day(pd.Timestamp.now(tz="UTC") if now is None else now) - pd.Timedelta(days=1)


def plan_dtcc_update(kind: str, start, end, *, now=None, root: Path = DTCC_DIR) -> list[pd.Timestamp]:
    """Days in ``[start, end]`` (inclusive) not archived yet, clamped to the days DTCC can
    have: from ``DTCC_FIRST_DAY`` through the last UTC day that has ended. No network."""
    start = max(pd.Timestamp(start).normalize(), pd.Timestamp(DTCC_FIRST_DAY))
    end = min(pd.Timestamp(end).normalize(), last_complete_day(now))
    if end < start:
        return []
    have = archived_days(kind, root=root)
    return [d for d in pd.date_range(start, end) if d not in have]


def validate_zip(content: bytes) -> str:
    """The name of the one CSV in a published zip; raises DtccError if the bytes aren't a
    sound zip holding exactly that."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = z.namelist()
            bad = z.testzip()
    except zipfile.BadZipFile as exc:
        raise dtcc_client.DtccError(f"corrupt zip: {exc}") from exc
    if bad is not None:
        raise dtcc_client.DtccError(f"corrupt member {bad}")
    if len(names) != 1 or not names[0].lower().endswith(".csv"):
        raise dtcc_client.DtccError(f"expected one CSV, got {names}")
    return names[0]


def store_dtcc_raw(kind: str, day, content: bytes, *, root: Path = DTCC_DIR) -> Path:
    """FILES ONLY: verify the zip, then write it whole (temp file + rename), so a file on
    disk is always a complete, verified one."""
    validate_zip(content)
    out = path_for(kind, day, root=root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(content)
    os.replace(tmp, out)
    return out


def fetch_and_store_dtcc(kind: str, days, *, root: Path = DTCC_DIR, fetch=None, sleep=None) -> dict:
    """Fetch and archive each day, one request at a time. Per-day failures are collected,
    not raised, so one bad day never costs the others. Returns ``{"archived": [...],
    "unpublished": [...], "errors": {day: message}}`` - ``unpublished`` = DTCC has no file."""
    fetch = FETCH if fetch is None else fetch
    sleep = time.sleep if sleep is None else sleep
    out = {"archived": [], "unpublished": [], "errors": {}}
    for i, day in enumerate(days):
        if i:
            sleep(PAUSE_S)
        try:
            content = fetch(kind, day)
            if content is None:
                out["unpublished"].append(day)
                continue
            store_dtcc_raw(kind, day, content, root=root)
            out["archived"].append(day)
        except Exception as exc:
            out["errors"][day] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            log.warning("dtcc %s %s failed: %s", kind, pd.Timestamp(day).date(), out["errors"][day])
    if out["archived"]:
        log.info("dtcc %s: archived %d day(s)", kind, len(out["archived"]))
    return out


def read_dtcc_day(kind: str, day, *, columns=None, root: Path = DTCC_DIR) -> pd.DataFrame:
    """One archived day's CSV as published (all columns as text unless pandas infers
    otherwise). ``columns`` restricts the read to those that EXIST in the file - a column
    DTCC drops or renames is simply absent, for the caller to handle. Empty if the day
    isn't archived. No network."""
    path = path_for(kind, day, root=root)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    wanted = None if columns is None else set(columns)
    return pd.read_csv(path, usecols=None if wanted is None else (lambda c: c in wanted), low_memory=False)


def load_dtcc_day(kind: str, day, *, columns=None, fetch_missing: bool = True, root: Path = DTCC_DIR,
                  fetch=None) -> pd.DataFrame:
    """Parent: archive the day if it's missing (and DTCC can have it), then read it."""
    if fetch_missing:
        missing = plan_dtcc_update(kind, day, day, root=root)
        if missing:
            fetch_and_store_dtcc(kind, missing, root=root, fetch=fetch)
    return read_dtcc_day(kind, day, columns=columns, root=root)
