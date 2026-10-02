"""Treasury auction tails from archived auction recaps: list / harvest / store / read.

``CANDIDATES`` says where each source's recaps live in the archive (CDX prefixes plus a
URL filter applied on the archive's side). ``harvest_tails`` fetches every candidate not
yet processed, extracts and checks its numbers (infra.processing.auction_tails), matches
it to its auction (tenor + the official high yield, from infra.pipeline.tsy_auctions) and
stores one row per (auction, source).

* **Store** ``~/Database/RawData/TsyAuctionTails`` (hive on the auction close). A row is
  KNOWN from the auction close: the WI at the close is market information then, whenever
  the article appeared.
* **Processed-URL manifest** ``RawData/_coverage/tsy_auction_tails.parquet``: every URL
  handled, with its outcome and the parser version. A PASSED article is never fetched
  again (Rule 2.1); any other outcome is retried once ``PARSER_VERSION`` moves on - so a
  parser gap can't permanently write articles off (found 2026-10-01: version 1 missed the
  2011-16 phrasings and "processed" 145 of 180 articles with nothing).
* Polite like the calendar harvest: one fetch per PAUSE_S, failures logged and skipped.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from infra.api import wayback_client
from infra.config import TSY_TAILS_COVERAGE_FILE, TSY_TAILS_DIR
from infra.pipeline.tsy_auctions import read_auctions
from infra.processing import auction_tails as at
from infra.processing.econ_calendar import decode_html
from infra.storage import parquet_store

log = logging.getLogger(__name__)
PAUSE_S = 3.0
# A capture smaller than this is a stub, not an article: real archived recaps measured
# 27-33KB (ZeroHedge) and 42-249KB (ForexLive), while the degraded archive of 2026-10-01
# redirected to another capture of the same URL that was 7.8KB with no article text.
# Treated as a FETCH ERROR (retried next run), never as an article that "has no high yield".
MIN_PAGE_BYTES = 12_000
FLUSH_EVERY = 20
_AUCTION_WORDS = r".*(auction|sells|tail|stop).*"
# source -> [(CDX prefix, archive-side URL regex)]
CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "zerohedge": [("zerohedge.com/markets/", _AUCTION_WORDS), ("zerohedge.com/news/", _AUCTION_WORDS)],
    # /news/!/<slug>-YYYYMMDD from ~2017; before that /<id>/all/<slug> (2010-2016)
    "forexlive": [("forexlive.com/news/", r".*us-sells-.*(year|yr).*"),
                  ("forexlive.com/", r".*/all/us-sells-.*(year|yr).*")],
}
MANIFEST_COLUMNS = ["source", "url_key", "processed", "outcome", "parser_version"]


def url_key(url: str) -> str:
    return re.sub(r"^https?://(www\.)?|:80(?=/)", "", url.split("?")[0].lower()).rstrip("/")


def list_candidates(source: str, *, list_fn=None, start="2009-01-01", end=None) -> pd.DataFrame:
    """One row per candidate article (its FIRST capture): ``timestamp, original, url_key``
    - tenor-bearing auction recaps only (no TIPS/FRN/bills)."""
    list_fn = list_fn or wayback_client.list_captures
    end = end or pd.Timestamp.now().normalize()
    frames = [list_fn(prefix, pd.Timestamp(start), pd.Timestamp(end), url_regex=rx, first_per_url=True)
              for prefix, rx in CANDIDATES[source]]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df.assign(url_key=pd.Series(dtype=str))
    df["url_key"] = df["original"].map(url_key)
    slug = df["url_key"].str.rsplit("/", n=1).str[-1]
    df = df[slug.map(at.tenor).notna() & ~slug.map(lambda x: bool(at.NON_US.search(x)))]
    return df.sort_values("timestamp").drop_duplicates("url_key").reset_index(drop=True)


def read_manifest(path: Path = TSY_TAILS_COVERAGE_FILE) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)


def _record(path: Path, rows: list[tuple]) -> None:
    if not rows:
        return
    df = pd.concat([read_manifest(path), pd.DataFrame(rows, columns=MANIFEST_COLUMNS)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.drop_duplicates(["source", "url_key"], keep="last").to_parquet(tmp, index=False)
    tmp.replace(path)


def process_article(source: str, page: str, url: str, capture: pd.Timestamp, auctions: pd.DataFrame):
    """(row or None, outcome) for one article page."""
    title, body = at.page_text(page)
    nums = at.extract(source, title, body)
    tenor = at.tenor(title) or at.tenor(url.rsplit("/", 1)[-1])
    near = at.slug_date(url) or capture
    a = at.match(tenor, nums["high_yield_reported"], near, auctions, decimals=nums["hy_decimals"])
    if a is None:
        return None, "no auction match" if np.isfinite(nums["high_yield_reported"]) else "no high yield"
    if at.NON_US.search(title):
        return None, "not a US auction"
    if not np.isfinite(nums["wi_yield"]):
        return None, "no WI"
    row = at.check({"timestamp": a["timestamp"], "cusip": a["cusip"], "source": source, "url": url,
                    "capture": capture, "tenor_years": tenor, **nums, "high_yield_official": a["high_yield"]})
    return row, "passed" if row["passed"] else "failed check"


def store_tails(rows: list[dict], *, root: Path = TSY_TAILS_DIR) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=at.COLUMNS)
    df["timestamp"] = df["timestamp"].astype("datetime64[ms]")
    df["capture"] = pd.to_datetime(df["capture"]).astype("datetime64[ms]")
    # several articles on one auction from one source: a passing one wins, then the latest
    df = df.sort_values(["passed", "capture"]).drop_duplicates(at.KEYS, keep="last")
    parquet_store.write_partitioned(df, root, at.KEYS)
    return len(df)


def harvest_tails(sources=at.SOURCES, *, root: Path = TSY_TAILS_DIR, manifest: Path = TSY_TAILS_COVERAGE_FILE,
                  list_fn=None, fetch_fn=None, auctions: pd.DataFrame | None = None, pause_s: float = PAUSE_S,
                  limit: int | None = None, budget_s: float | None = None) -> dict:
    """Fetch, parse, match and store every not-yet-processed candidate of ``sources``
    (at most ``limit`` per source - the daily cycle caps it so a backlog never makes the
    run slow; the newest candidates first then, the rest left for later runs), and stop
    cleanly once ``budget_s`` seconds have gone - what is done is stored, the rest waits."""
    started = time.monotonic()
    fetch_fn = fetch_fn or wayback_client.fetch_capture
    auctions = read_auctions() if auctions is None else auctions
    auctions = auctions[auctions["high_yield"].notna()]
    stats = {}
    for source in sources:
        if budget_s is not None and time.monotonic() - started > budget_s:
            stats[source] = {"stopped: time budget": None}
            continue
        # done = passed, or handled by THIS parser version (an improved parser retries the rest)
        m = read_manifest(manifest).query("source == @source")
        version = m["parser_version"] if "parser_version" in m else pd.Series(0, index=m.index)
        done = set(m.loc[(m["outcome"] == "passed") | (version.fillna(0) >= at.PARSER_VERSION), "url_key"])
        todo = list_candidates(source, list_fn=list_fn)
        todo = todo[~todo["url_key"].isin(done)]
        if limit is not None:
            todo = todo.sort_values("timestamp", ascending=False).head(limit)
        rows, seen, outcomes = [], [], {}
        for i, cap in enumerate(todo.itertuples()):
            if budget_s is not None and time.monotonic() - started > budget_s:
                outcomes["stopped: time budget"] = len(todo) - i
                break
            try:
                raw = fetch_fn(cap.timestamp, cap.original)
                if len(raw) < MIN_PAGE_BYTES:
                    raise ValueError(f"stub page ({len(raw)} bytes)")
                row, outcome = process_article(source, decode_html(raw), cap.original, cap.timestamp, auctions)
            except Exception as exc:
                row, outcome = None, f"error: {type(exc).__name__}"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if row is not None:
                rows.append(row)
            if not outcome.startswith("error"):  # a failed fetch is retried next run, never marked done
                seen.append((source, cap.url_key, pd.Timestamp.now(), outcome, at.PARSER_VERSION))
            if (i + 1) % FLUSH_EVERY == 0:
                store_tails(rows, root=root)
                _record(manifest, seen)
                rows, seen = [], []
                log.info("tails %s: %d/%d %s", source, i + 1, len(todo), outcomes)
            if pause_s:
                time.sleep(pause_s)
        store_tails(rows, root=root)
        _record(manifest, seen)
        stats[source] = outcomes
    return stats


def read_tails(as_of=None, *, root: Path = TSY_TAILS_DIR) -> pd.DataFrame:
    """Every stored (auction, source) tail row; with ``as_of`` (an instant, UTC) only
    auctions closed by then. No network."""
    raw = parquet_store.read_partitioned(root, end=None if as_of is None else pd.Timestamp(as_of) + pd.Timedelta(milliseconds=1))
    if raw is None or raw.empty:
        return pd.DataFrame(columns=at.COLUMNS)
    return raw[at.COLUMNS].sort_values(at.KEYS).reset_index(drop=True)


def best_tails(as_of=None, *, root: Path = TSY_TAILS_DIR) -> pd.DataFrame:
    """One checked tail per auction (``at.best``: source priority, with the number of
    sources that passed and their spread)."""
    return at.best(read_tails(as_of, root=root))
