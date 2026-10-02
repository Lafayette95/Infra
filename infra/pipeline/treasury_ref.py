"""US Treasury reference data (CLAUDE.md 18): build / store / read the static security
table under ``Reference/Treasuries/Securities``. No network - it is a VIEW of the raw
auctions store (``RawData/TsyAuctions``, fetched by infra.pipeline.tsy_auctions), rebuilt
whole on every run (~3s): an announced auction's coupon appears only once it is held, so a
security's row can change, never just append.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.config import TREASURY_SECURITIES_DIR, TSY_AUCTIONS_DIR
from infra.pipeline.tsy_auctions import read_auctions
from infra.processing import treasury_ref as tr
from infra.storage import parquet_store

log = logging.getLogger(__name__)


def build_securities(*, auctions_root: Path = TSY_AUCTIONS_DIR, root: Path = TREASURY_SECURITIES_DIR) -> int:
    """Rebuild the security table from the raw auctions and store it (upsert by CUSIP)."""
    sec = tr.securities(read_auctions(nominal_only=False, root=auctions_root))
    if sec.empty:
        return 0
    parquet_store.write_partitioned(sec, root, tr.SECURITY_KEYS)
    log.info("treasury securities: %d", len(sec))
    return len(sec)


def read_securities(as_of=None, *, security_types=None, root: Path = TREASURY_SECURITIES_DIR) -> pd.DataFrame:
    """The securities ANNOUNCED by the end of day ``as_of`` (None = all), optionally only
    ``security_types`` (e.g. ``("Note", "Bond")``). No network."""
    end = None if as_of is None else pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
    df = parquet_store.read_partitioned(root, end=end,
                                        equals_in={"security_type": list(security_types)} if security_types else None)
    if df is None or df.empty:
        return pd.DataFrame(columns=tr.SECURITY_COLUMNS)
    return df[tr.SECURITY_COLUMNS].sort_values(["timestamp", "cusip"]).reset_index(drop=True)


def load_securities(as_of=None, *, security_types=None, rebuild: bool = True, auctions_root: Path = TSY_AUCTIONS_DIR,
                    root: Path = TREASURY_SECURITIES_DIR) -> pd.DataFrame:
    """Parent: rebuild from the raw auctions on disk (no fetch), then read."""
    if rebuild:
        build_securities(auctions_root=auctions_root, root=root)
    return read_securities(as_of, security_types=security_types, root=root)
