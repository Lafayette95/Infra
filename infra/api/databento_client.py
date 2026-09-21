"""The ONLY module that talks to Databento. No caching or parquet logic lives here.

Cost guardrails enforced at this boundary:
  * Rule 2.2 - futures symbols must be continuous (``SR3.c.0``); wildcards rejected.
  * Every request is priced with the free ``metadata.get_cost`` first and refused
    if it exceeds ``max_cost_usd``.
"""
from __future__ import annotations

import os
import re
from typing import Sequence

import databento as db
import pandas as pd
from dotenv import load_dotenv

from infra.config import (
    API_KEY_ENV,
    MAX_COST_USD,
    PROJECT_ROOT,
    SCHEMA_DEFINITION,
    SCHEMA_OHLCV,
)

_CONTINUOUS_RE = re.compile(r"^[A-Z0-9]+\.[cvn]\.\d+$")


class CostLimitExceeded(RuntimeError):
    """Estimated request cost is above the configured budget guardrail."""


def get_client(api_key: str | None = None) -> db.Historical:
    """Build a historical client from an explicit key or ``DATABENTO_API_KEY``."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {API_KEY_ENV} in the environment or in .env")
    return db.Historical(key)


def validate_continuous_symbol(symbol: str) -> str:
    """Reject wildcards/empty values; accept only continuous contracts (Rule 2.2)."""
    if not symbol or "*" in symbol or not _CONTINUOUS_RE.match(symbol):
        raise ValueError(
            f"{symbol!r} is not a continuous symbol like 'SR3.c.0'; wildcards and "
            "empty symbols are banned (Rule 2.2)."
        )
    return symbol


def estimate_cost(
    dataset: str,
    schema: str,
    symbols: Sequence[str | int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    stype_in: str,
    client: db.Historical | None = None,
) -> float:
    """Price a request in USD without downloading anything (free endpoint)."""
    client = client or get_client()
    return float(client.metadata.get_cost(
        dataset=dataset,
        symbols=list(symbols),
        schema=schema,
        stype_in=stype_in,
        start=_utc(start),
        end=_utc(end),
    ))


def _get_range(
    dataset: str,
    schema: str,
    symbols: Sequence[str | int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    stype_in: str,
    max_cost_usd: float,
    client: db.Historical | None,
) -> pd.DataFrame:
    client = client or get_client()
    cost = estimate_cost(dataset, schema, symbols, start, end, stype_in, client)
    if cost > max_cost_usd:
        raise CostLimitExceeded(
            f"{dataset} {schema} {list(symbols)[:3]}... {start:%F}->{end:%F} would cost "
            f"${cost:.2f} > limit ${max_cost_usd:.2f}"
        )
    store = client.timeseries.get_range(
        dataset=dataset,
        symbols=list(symbols),
        schema=schema,
        stype_in=stype_in,
        start=_utc(start),
        end=_utc(end),
    )
    return store.to_df(price_type="float", pretty_ts=True, map_symbols=True)


def fetch_futures_ohlcv(
    dataset: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_cost_usd: float = MAX_COST_USD,
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """Raw ``ohlcv-1m`` bars for ONE continuous futures symbol over ``[start, end)``."""
    validate_continuous_symbol(symbol)
    return _get_range(dataset, SCHEMA_OHLCV, [symbol], start, end, "continuous", max_cost_usd, client)


def fetch_definitions(
    dataset: str,
    parent_symbols: Sequence[str],
    day: pd.Timestamp,
    *,
    max_cost_usd: float = MAX_COST_USD,
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """Instrument ``definition`` rows for parent symbols (e.g. ``OQ.OPT``) on ONE day."""
    day = pd.Timestamp(day).normalize()
    return _get_range(
        dataset, SCHEMA_DEFINITION, parent_symbols, day, day + pd.Timedelta(days=1),
        "parent", max_cost_usd, client,
    )


def fetch_ohlcv_by_instrument_ids(
    dataset: str,
    instrument_ids: Sequence[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_cost_usd: float = MAX_COST_USD,
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """Raw ``ohlcv-1m`` bars for an isolated array of instrument ids (Rule 2.3 step 3)."""
    if not len(instrument_ids):
        raise ValueError("instrument_ids is empty; refusing an unbounded options query.")
    return _get_range(
        dataset, SCHEMA_OHLCV, [int(i) for i in instrument_ids], start, end,
        "instrument_id", max_cost_usd, client,
    )


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
