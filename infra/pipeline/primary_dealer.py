"""NY Fed Primary Dealer Statistics: plan / fetch / store / read (CLAUDE.md 24).

Every series, one store (``PRIMARY_DEALER_DIR``, keys ``timestamp`` = the as-of date,
``series``). The source is ONE CSV with the whole history, so a fetch always re-reads
everything - and only rows that are new or REVISED are written (the count of revisions is
reported). Rule 2.1: requested only when a new week is due (the latest stored Wednesday's
next release has passed) or the store is empty; ``force`` re-reads regardless. The catalog
(descriptions of the current series break) is refreshed with each fetch.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.api import nyfed_client
from infra.config import (BULK_CATALOG_DIR, PRIMARY_DEALER_DIR, PRIMARY_DEALER_RELEASE,
                          PRIMARY_DEALER_STATE_FILE)
from infra.processing.primary_dealer import KEYS, changed_rows, parse_primary_dealer, release_day
from infra.storage import parquet_store
from infra.trading_calendar import snap_instants

# network hooks, module-level so tests stub them (tests/conftest.py)
FETCH = nyfed_client.fetch_primary_dealer_csv
CATALOG_FETCH = nyfed_client.fetch_primary_dealer_catalog
CATALOG_FILE = BULK_CATALOG_DIR / "primary_dealer.parquet"


def read_primary_dealer(series=None, start=None, end=None, *, as_of=None, root: Path = PRIMARY_DEALER_DIR) -> pd.DataFrame:
    """Stored rows in ``[start, end]`` (as-of dates, inclusive), optionally only ``series``
    (keys like ``PDPOSGST-TOT``) and only what had been RELEASED by ``as_of``."""
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end) + pd.Timedelta(days=1))
    if df is None or df.empty:
        return pd.DataFrame(columns=[*KEYS, "value", "suppressed", "known_from"])
    df["series"] = df["series"].astype(str)
    if series is not None:
        df = df[df["series"].isin(set(series))]
    if as_of is not None:
        df = df[df["known_from"] <= pd.Timestamp(as_of)]
    return df.sort_values(KEYS, ignore_index=True)


def read_catalog(catalog_file: Path = CATALOG_FILE) -> pd.DataFrame:
    return pd.read_parquet(catalog_file) if Path(catalog_file).exists() else pd.DataFrame(columns=["series", "description"])


def latest_stored(root: Path = PRIMARY_DEALER_DIR) -> pd.Timestamp | None:
    df = parquet_store.read_partitioned(root, columns=["timestamp"]) if parquet_store.has_data(root) else None
    return None if df is None or df.empty else pd.Timestamp(df["timestamp"].max())


def next_release(latest: pd.Timestamp) -> pd.Timestamp:
    """When the week after ``latest`` (a Wednesday) is released, UTC."""
    local_time, zone, lag = PRIMARY_DEALER_RELEASE
    day = release_day(pd.Series([pd.Timestamp(latest) + pd.Timedelta(days=7)]), lag).iloc[0]
    return snap_instants([day], local_time, zone)[0]


def is_due(*, root: Path = PRIMARY_DEALER_DIR, now=None) -> bool:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    latest = latest_stored(root)
    return latest is None or now >= next_release(latest)


def update_primary_dealer(*, root: Path = PRIMARY_DEALER_DIR, catalog_file: Path = CATALOG_FILE,
                          state_file: Path = PRIMARY_DEALER_STATE_FILE, force: bool = False, now=None) -> dict:
    """Fetch and store if due (module doc). Never raises:
    ``{"status", "rows", "revised", "latest", "error"}``."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    if not force and not is_due(root=root, now=now):
        return {"status": "not due", "rows": 0, "revised": 0, "latest": latest_stored(root), "error": None}
    try:
        new = parse_primary_dealer(FETCH(), PRIMARY_DEALER_RELEASE)
        stored = parquet_store.read_partitioned(root) if parquet_store.has_data(root) else None
        if stored is not None:
            stored["series"] = stored["series"].astype(str)
        write, revised = changed_rows(new, stored)
        if not write.empty:
            parquet_store.write_partitioned(write, root, KEYS)
        try:
            series, breaks = CATALOG_FETCH()
            cat = pd.DataFrame([{"series": s.get("keyid"), "series_break": s.get("seriesbreak"),
                                 "description": s.get("description")} for s in series])
            if not cat.empty:
                Path(catalog_file).parent.mkdir(parents=True, exist_ok=True)
                cat.to_parquet(catalog_file, index=False)
        except Exception:  # the catalog is a convenience; the data is what matters
            pass
        latest = pd.Timestamp(new["timestamp"].max()) if not new.empty else None
        Path(state_file).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"latest": latest, "fetched_at": now, "rows_written": len(write), "revised": revised}]) \
            .to_parquet(state_file, index=False)
        return {"status": "fetched", "rows": len(write), "revised": revised, "latest": latest, "error": None}
    except Exception as exc:
        return {"status": "failed", "rows": 0, "revised": 0, "latest": None, "error": f"{type(exc).__name__}: {exc}"}
