"""Options pipeline implementing Rule 2.3:

1. pull the tiny ``definition`` schema (cached on disk),
2. filter it locally by strike/expiry,
3. query pricing ONLY for the isolated instrument ids that are not yet on disk.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import databento_client as api
from infra.config import (
    DEFINITIONS_DIR,
    MAX_COST_USD,
    OPTIONS_COVERAGE_FILE,
    OPTIONS_DIR,
    OPTIONS_UNIVERSE,
)
from infra.coverage.intervals import Interval, find_missing_ranges, to_utc_day
from infra.processing import definitions as defs
from infra.processing import transforms as tf
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)


def _definitions_path(parent: str, day: pd.Timestamp, directory: Path) -> Path:
    return directory / f"{parent}_{day:%Y-%m-%d}.parquet"


def load_definitions(
    parent: str,
    day,
    *,
    dataset: str | None = None,
    fetch_missing: bool = True,
    directory: Path = DEFINITIONS_DIR,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """Normalised option definitions for ``parent`` on ``day`` (cache-first, Rule 2.1)."""
    day = to_utc_day(day)
    path = _definitions_path(parent, day, directory)
    if path.exists():
        return pd.read_parquet(path)
    if not fetch_missing:
        return defs.normalize_definitions(pd.DataFrame())
    dataset = dataset or OPTIONS_UNIVERSE[parent]
    raw = api.fetch_definitions(dataset, [parent], day, max_cost_usd=max_cost_usd, client=client)
    out = defs.normalize_definitions(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, engine="pyarrow", index=False, compression="zstd", compression_level=5)
    return out


def plan_options_update(
    instrument_ids: list[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    coverage_file: Path = OPTIONS_COVERAGE_FILE,
) -> dict[Interval, list[int]]:
    """Group instrument ids by the identical missing range they need. No API."""
    requested = (to_utc_day(start), to_utc_day(end))
    plan: dict[Interval, list[int]] = {}
    for instrument_id in instrument_ids:
        covered = coverage_store.read_covered(coverage_file, str(instrument_id))
        for gap in find_missing_ranges(requested, covered):
            plan.setdefault(gap, []).append(instrument_id)
    return plan


def fetch_and_store_options(
    plan: dict[Interval, list[int]],
    definitions: pd.DataFrame,
    *,
    dataset: str,
    root: Path = OPTIONS_DIR,
    coverage_file: Path = OPTIONS_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> int:
    """Query bars for exactly the planned ids/ranges, save, record coverage."""
    rows = 0
    for (range_start, range_end), ids in plan.items():
        raw = api.fetch_ohlcv_by_instrument_ids(
            dataset, ids, range_start, range_end, max_cost_usd=max_cost_usd, client=client
        )
        clean = tf.clean_options_ohlcv(raw, definitions)
        parquet_store.write_partitioned(tf.encode_options(clean), root, tf.OPTIONS_KEYS)
        for instrument_id in ids:
            coverage_store.record_covered(coverage_file, str(instrument_id), [(range_start, range_end)])
        rows += len(clean)
    return rows


def read_options_from_disk(
    contracts: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    root: Path = OPTIONS_DIR,
) -> pd.DataFrame:
    """Read decoded option bars, restricted to the given (filtered) contracts."""
    raw = parquet_store.read_partitioned(
        root, start=start, end=end,
        equals_in={"underlying": contracts["underlying"].unique().tolist()},
    )
    if raw is None or raw.empty:
        return tf.decode_options(tf.encode_options(tf._empty_options()))
    df = tf.decode_options(raw[tf.OPTIONS_COLUMNS])
    keys = contracts[["underlying", "option_type", "strike", "expiry"]].astype(
        {"underlying": "str", "option_type": "str"}
    )
    df = df.astype({"underlying": "str", "option_type": "str"}).merge(
        keys, on=["underlying", "option_type", "strike", "expiry"], how="inner"
    )
    return df.sort_values(tf.OPTIONS_KEYS).reset_index(drop=True)


def load_options(
    parent: str,
    definitions_day,
    start,
    end,
    *,
    underlyings: list[str] | None = None,
    option_types: list[str] | None = None,
    strike_range: tuple[float, float] | None = None,
    expiry_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    fetch_missing: bool = True,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """Parent: definitions -> local filter -> fetch only missing ids/ranges -> read."""
    start, end = to_utc_day(start), to_utc_day(end)
    dataset = OPTIONS_UNIVERSE[parent]
    all_defs = load_definitions(
        parent, definitions_day, dataset=dataset, fetch_missing=fetch_missing,
        max_cost_usd=max_cost_usd, client=client,
    )
    contracts = defs.filter_definitions(
        all_defs, underlyings=underlyings, option_types=option_types,
        strike_range=strike_range, expiry_range=expiry_range,
    )
    if contracts.empty:
        return tf.decode_options(tf.encode_options(tf._empty_options()))
    if fetch_missing:
        plan = plan_options_update(contracts["instrument_id"].tolist(), start, end)
        fetch_and_store_options(
            plan, contracts, dataset=dataset, max_cost_usd=max_cost_usd, client=client
        )
    return read_options_from_disk(contracts, start, end)
