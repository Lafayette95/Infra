"""CFTC Traders in Financial Futures: plan / fetch / store / read (CLAUDE.md 24).

Every financial futures market, both reports (``CFTC_TFF_REPORTS``), one store
(``CFTC_TFF_DIR``, keys ``timestamp`` = the Tuesday the positions are as of, ``report``,
``market_code``). Rule 2.1: a report is requested only when its ``Last-Modified`` moved
past the one stored in ``CFTC_TFF_STATE_FILE`` - then everything since the latest stored
week minus ``CFTC_TFF_REFETCH_WEEKS`` (catches CFTC's revisions), or the whole history on
the first run. Reading is point in time (``as_of``: rows released by then).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.api import cftc_client
from infra.config import (CFTC_TFF_DIR, CFTC_TFF_REFETCH_WEEKS, CFTC_TFF_RELEASE, CFTC_TFF_REPORTS,
                          CFTC_TFF_STATE_FILE)
from infra.processing.cftc_tff import KEYS, parse_tff
from infra.storage import parquet_store

# network hooks, module-level so tests stub them (tests/conftest.py)
LAST_MODIFIED = cftc_client.last_modified
FETCH = cftc_client.fetch_tff


def read_state(state_file: Path = CFTC_TFF_STATE_FILE) -> pd.DataFrame:
    if not Path(state_file).exists():
        return pd.DataFrame(columns=["report", "last_modified", "latest", "fetched_at"])
    return pd.read_parquet(state_file)


def _write_state(state: pd.DataFrame, state_file: Path) -> None:
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(state_file).with_suffix(".tmp")
    state.to_parquet(tmp, index=False)
    tmp.replace(state_file)


def read_tff(start=None, end=None, *, markets=None, report: str = "futures", as_of=None,
             root: Path = CFTC_TFF_DIR) -> pd.DataFrame:
    """Stored rows of one report in ``[start, end]`` (report days, inclusive), optionally
    only ``markets`` (CFTC contract market codes, e.g. ``"043602"`` = 10y note) and only
    what had been RELEASED by ``as_of`` (CLAUDE.md 3's point-in-time cutoff; None = all)."""
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end) + pd.Timedelta(days=1))
    if df is None or df.empty:
        return pd.DataFrame(columns=[*KEYS, "known_from"])
    df = df[df["report"].astype(str) == report]
    if markets is not None:
        df = df[df["market_code"].astype(str).isin({str(m) for m in markets})]
    if as_of is not None:
        df = df[df["known_from"] <= pd.Timestamp(as_of)]
    return df.sort_values(KEYS, ignore_index=True)


def update_tff(*, reports: dict[str, str] = CFTC_TFF_REPORTS, root: Path = CFTC_TFF_DIR,
               state_file: Path = CFTC_TFF_STATE_FILE, force: bool = False, now=None) -> dict:
    """Fetch and store each report whose source changed (module doc). Never raises for one
    report's failure: ``{report: {"status", "rows", "latest", "error"}}``."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    state = read_state(state_file).set_index("report")
    out = {}
    for report, dataset in reports.items():
        try:
            modified = LAST_MODIFIED(dataset)
            seen = state["last_modified"].get(report) if report in state.index else None
            if not force and seen is not None and modified is not None and pd.Timestamp(modified) <= pd.Timestamp(seen):
                out[report] = {"status": "unchanged", "rows": 0, "latest": state["latest"].get(report), "error": None}
                continue
            latest = state["latest"].get(report) if report in state.index else None
            since = None if latest is None or pd.isna(latest) else pd.Timestamp(latest) - pd.Timedelta(weeks=CFTC_TFF_REFETCH_WEEKS)
            df = parse_tff(FETCH(dataset, since), report, CFTC_TFF_RELEASE)
            if not df.empty:
                parquet_store.write_partitioned(df, root, KEYS)
            new_latest = df["timestamp"].max() if not df.empty else latest
            state.loc[report, ["last_modified", "latest", "fetched_at"]] = [modified, new_latest, now]
            out[report] = {"status": "fetched", "rows": len(df), "latest": new_latest, "error": None}
        except Exception as exc:  # one report's failure must not stop the other
            out[report] = {"status": "failed", "rows": 0, "latest": None, "error": f"{type(exc).__name__}: {exc}"}
    _write_state(state.reset_index(names="report"), state_file)
    return out
