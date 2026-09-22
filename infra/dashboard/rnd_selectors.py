"""Options for the RND page's Underlying/Expiry/Day dropdowns (no Dash imports;
unit-testable). Reads only what's already cached under Database/Daily/Options - this
page is display-only (CLAUDE.md section 9/8), never triggers a fetch; populate data
first via scripts/update_daily_options.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.config import DAILY_OPTIONS_DIR
from infra.processing.statistics import decode_expiry_column
from infra.storage import parquet_store


def underlying_options(root: Path = DAILY_OPTIONS_DIR) -> list[str]:
    """Distinct underlyings with daily settlement data already on disk, sorted."""
    return parquet_store.list_values(root, "underlying")


def expiry_options(underlying: str, *, root: Path = DAILY_OPTIONS_DIR) -> list[str]:
    """Distinct option expiries on disk for one underlying (most have exactly one;
    serial/mid-curve underlyings can have several - see CLAUDE.md section 9)."""
    raw = parquet_store.read_partitioned(root, equals_in={"underlying": [underlying]}, columns=["expiry"])
    if raw is None or raw.empty:
        return []
    expiries = decode_expiry_column(raw["expiry"])
    return [d.strftime("%Y-%m-%d") for d in sorted(expiries.unique())]


def day_options(underlying: str, expiry: str, *, root: Path = DAILY_OPTIONS_DIR) -> list[str]:
    """Trading days on disk for one (underlying, expiry), excluding the expiry day
    itself (T=0, extract_rnd rejects it - see CLAUDE.md section 9's practical-range note)."""
    raw = parquet_store.read_partitioned(
        root, equals_in={"underlying": [underlying]}, columns=["timestamp", "expiry"]
    )
    if raw is None or raw.empty:
        return []
    expiry_ts = pd.Timestamp(expiry)
    raw = raw[decode_expiry_column(raw["expiry"]) == expiry_ts]
    days = sorted(d for d in pd.to_datetime(raw["timestamp"].unique()) if d < expiry_ts)
    return [d.strftime("%Y-%m-%d") for d in days]
