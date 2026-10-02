"""Treasury futures delivery baskets (CLAUDE.md 18): archive CME's daily conversion-factor
files raw, and build / store / read the basket reference store from them.

* ``RawData/CME_TCF/<year>/TCF_YYYYMMDD.csv``: each file byte-for-byte as published; the
  file on disk IS the coverage (like the DTCC archive) - a day never asked again.
* ``Reference/Treasuries/FuturesBaskets``: parsed rows, keys (timestamp = the file's day,
  root, contract, cusip, source). ``source = "cme"`` rows are rebuilt from the archive;
  computed history (``"computed"``) is added for days CME's files don't cover.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.api import cme_ftp_client
from infra.config import (
    CME_TCF_DIR,
    TREASURY_BASKET_HISTORY,
    TREASURY_BASKET_RULES,
    TREASURY_BASKETS_DIR,
    TREASURY_SECURITIES_DIR,
    TSY_AUCTIONS_DIR,
)
from infra.pipeline.treasury_ref import read_securities
from infra.pipeline.tsy_auctions import read_auctions
from infra.processing import futures_baskets as fb
from infra.storage import parquet_store

log = logging.getLogger(__name__)
# network hooks (stubbed suite-wide in tests/conftest.py)
LIST: Callable[[], list[str]] = cme_ftp_client.list_tcf_files
FETCH: Callable[[str], bytes] = cme_ftp_client.fetch_tcf_file
PAUSE_S = 0.2


def _day(name: str) -> pd.Timestamp:
    return pd.Timestamp(name[4:12])


def path_for(name: str, *, root: Path = CME_TCF_DIR) -> Path:
    return root / f"{_day(name):%Y}" / name


def archived_files(*, root: Path = CME_TCF_DIR) -> dict[pd.Timestamp, Path]:
    """``{file day: path}`` of every archived file. No network."""
    return {_day(p.name): p for p in root.glob("*/TCF_*.csv")}


def archive_tcf(*, root: Path = CME_TCF_DIR, listing=None, fetch=None, sleep=None) -> dict:
    """Download every listed file not yet archived (one at a time). Per-file failures
    collected. Returns ``{"archived": [names], "errors": {name: msg}}``."""
    listing = (LIST if listing is None else listing)()
    fetch, sleep = (FETCH if fetch is None else fetch), (time.sleep if sleep is None else sleep)
    have = archived_files(root=root)
    todo = [n for n in listing if _day(n) not in have]
    out = {"archived": [], "errors": {}}
    for i, name in enumerate(todo):
        if i:
            sleep(PAUSE_S)
        try:
            content = fetch(name)
            if not content.startswith(b"Exch,Period,PFCode,CUSIP"):
                raise ValueError(f"unexpected header {content[:60]!r}")
            dest = path_for(name, root=root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(content)
            os.replace(tmp, dest)
            out["archived"].append(name)
        except Exception as exc:
            out["errors"][name] = f"{type(exc).__name__}: {exc}"
            log.warning("CME TCF %s failed: %s", name, out["errors"][name])
    return out


def build_cme_baskets(*, archive_root: Path = CME_TCF_DIR, root: Path = TREASURY_BASKETS_DIR) -> int:
    """Rebuild every ``source = "cme"`` row from the archive (files are small: ~2s for all)."""
    frames = [fb.parse_tcf(p.read_bytes(), day) for day, p in sorted(archived_files(root=archive_root).items())]
    frames = [f for f in frames if not f.empty]
    parquet_store.delete_where(root, lambda part: part["source"] == "cme")
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    parquet_store.write_partitioned(df, root, fb.BASKET_KEYS)
    log.info("CME baskets: %d rows from %d files", len(df), len(frames))
    return len(df)


def read_baskets(start, end, *, root_symbol: str | None = None, contract: str | None = None,
                 source: str | None = None, root: Path = TREASURY_BASKETS_DIR) -> pd.DataFrame:
    """Basket rows with ``timestamp`` in ``[start, end)``, optionally filtered. No network."""
    eq = {k: [v] for k, v in (("root", root_symbol), ("contract", contract), ("source", source)) if v is not None}
    df = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end), equals_in=eq or None)
    if df is None or df.empty:
        return pd.DataFrame(columns=fb.BASKET_COLUMNS)
    return df[fb.BASKET_COLUMNS].sort_values(["timestamp", "root", "contract", "cusip"]).reset_index(drop=True)


def build_computed_baskets(start=None, end=None, *, root: Path = TREASURY_BASKETS_DIR,
                           archive_root: Path = CME_TCF_DIR, securities_root: Path = TREASURY_SECURITIES_DIR,
                           auctions_root: Path = TSY_AUCTIONS_DIR) -> int:
    """``source = "computed"`` rows for business days in ``[start, end]`` that have NO CME
    file (default: from each root's TREASURY_BASKET_HISTORY to the first archived file),
    for the listed contracts of each root in TREASURY_BASKET_HISTORY. Replaces those days'
    computed rows."""
    have = archived_files(root=archive_root)
    first_file = min(have) if have else pd.Timestamp.now().normalize()
    start = pd.Timestamp(start or min(TREASURY_BASKET_HISTORY.values())).normalize()
    end = pd.Timestamp(end).normalize() if end is not None else first_file - pd.Timedelta(days=1)
    days = [d for d in pd.bdate_range(start, end) if d not in have]
    if not days:
        return 0
    sec, auc = read_securities(root=securities_root), read_auctions(nominal_only=False, root=auctions_root)
    frames = []
    for day in days:
        terms = fb.issued_terms(auc, day)
        for r, since in TREASURY_BASKET_HISTORY.items():
            if day < pd.Timestamp(since):
                continue
            for dm in fb.listed_contracts(r, day):
                frames.append(fb.computed_basket(sec, r, dm, TREASURY_BASKET_RULES[r], day, terms))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    lo, hi = days[0], days[-1] + pd.Timedelta(days=1)
    parquet_store.delete_where(root, lambda part: (part["source"] == "computed")
                               & pd.to_datetime(part["timestamp"]).between(lo, hi, inclusive="left"))
    parquet_store.write_partitioned(df, root, fb.BASKET_KEYS)
    log.info("computed baskets %s..%s: %d rows", lo.date(), days[-1].date(), len(df))
    return len(df)
