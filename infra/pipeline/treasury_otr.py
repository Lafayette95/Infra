"""US Treasury on/off-the-run map (CLAUDE.md 18): build / store / read over
``Reference/Treasuries/OTR``. No network - computed from the reference table
(infra.pipeline.treasury_ref), which must be built first. A rebuilt range REPLACES its
days' rows (an issue announced late, or a revised date, must not leave a stale rank).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.config import (
    TREASURY_OTR_CONVENTIONS,
    TREASURY_OTR_DEFAULT_CONVENTION,
    TREASURY_OTR_DEPTH,
    TREASURY_OTR_DIR,
    TREASURY_OTR_TENORS,
    TREASURY_PRICES_START,
    TREASURY_SECURITIES_DIR,
)
from infra.pipeline.treasury_ref import read_securities
from infra.processing.treasury_otr import OTR_COLUMNS, OTR_KEYS, otr_map
from infra.storage import parquet_store

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)


def build_otr(start=TREASURY_PRICES_START, end=None, *, root: Path = TREASURY_OTR_DIR,
              securities_root: Path = TREASURY_SECURITIES_DIR) -> int:
    """Compute and store the map for business days in ``[start, end]`` (default: through
    today), replacing those days. Uses only securities announced by each day's end
    (point in time), via the reference table's announcement stamp."""
    end = pd.Timestamp.now().normalize() if end is None else pd.Timestamp(end).normalize()
    days = pd.bdate_range(pd.Timestamp(start).normalize(), end)
    if days.empty:
        return 0
    sec = read_securities(end, root=securities_root)
    df = otr_map(sec, days, TREASURY_OTR_TENORS, TREASURY_OTR_DEPTH, TREASURY_OTR_CONVENTIONS)
    parquet_store.delete_where(root, lambda part: pd.to_datetime(part["timestamp"]).between(
        days[0], days[-1] + _ONE_DAY, inclusive="left"))
    if not df.empty:
        parquet_store.write_partitioned(df, root, OTR_KEYS)
    log.info("treasury OTR map %s..%s: %d rows", days[0].date(), days[-1].date(), len(df))
    return len(df)


def read_otr(start, end, *, tenor: str | None = None, rank: int | None = None,
             convention: str = TREASURY_OTR_DEFAULT_CONVENTION, root: Path = TREASURY_OTR_DIR) -> pd.DataFrame:
    """Stored map rows with ``timestamp`` in ``[start, end)`` for one convention (default
    "issue"), optionally one tenor and/or rank. No network."""
    eq = {"convention": [convention]}
    if tenor is not None:
        eq["tenor"] = [tenor]
    if rank is not None:
        eq["rank"] = [rank]
    df = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end), equals_in=eq)
    if df is None or df.empty:
        return pd.DataFrame(columns=OTR_COLUMNS)
    return df[OTR_COLUMNS].sort_values(OTR_KEYS).reset_index(drop=True)


def otr_cusip(day, tenor: str, rank: int = 0, *, convention: str = TREASURY_OTR_DEFAULT_CONVENTION,
              root: Path = TREASURY_OTR_DIR) -> str | None:
    """The CUSIP at ``rank`` for ``tenor`` on ``day`` (None if none). No network."""
    day = pd.Timestamp(day).normalize()
    df = read_otr(day, day + _ONE_DAY, tenor=tenor, rank=rank, convention=convention, root=root)
    return None if df.empty else str(df["cusip"].iloc[0])
