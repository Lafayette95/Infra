"""The ONLY module that talks to Databento. No caching or parquet logic lives here.

Cost guardrails enforced at this boundary:
  * Rule 2.2 - futures are queried by ABSOLUTE raw symbol (``SRZ4``) only; wildcards
    and relative tickers (``SR3.c.0``) are rejected here and resolved locally instead.
  * Every request is priced with the free ``metadata.get_cost`` first and refused
    if it exceeds ``max_cost_usd``.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Sequence, TypeVar

import databento as db
import pandas as pd
from dotenv import load_dotenv

from infra.config import (
    API_KEY_ENV,
    MAX_COST_USD,
    PROJECT_ROOT,
    SCHEMA_BBO_1M,
    SCHEMA_DEFINITION,
    SCHEMA_OHLCV,
    SCHEMA_STATISTICS,
)
from infra.relative.symbology import is_relative


log = logging.getLogger(__name__)
T = TypeVar("T")


class CostLimitExceeded(RuntimeError):
    """Estimated request cost is above the configured budget guardrail."""


class ApiTimeout(RuntimeError):
    """Databento didn't answer within the wall-clock deadline, on every attempt."""


# Wall-clock deadlines WE enforce around every Databento call. The client's own
# requests timeout is per socket read, and on 2026-09-28 a backfill still hung for 40
# minutes blocked in an SSL read on an established connection (0% CPU, stack-sampled) -
# fatal for an unattended scheduled job. A call that overruns is abandoned (its daemon
# thread is left to die with the process) and retried on a fresh call; only hangs are
# retried - real API errors (4xx, the cost guard) are raised immediately.
DATA_DEADLINE_S = 300.0  # timeseries.get_range (payloads here are small; a year of 1m bars fits easily)
META_DEADLINE_S = 60.0  # metadata calls (cost estimates, dataset ranges)
API_ATTEMPTS = 3
RETRY_PAUSE_S = 5.0


def is_transient(exc: BaseException) -> bool:
    """Worth retrying: a server-side (5xx) error, rate limiting (429), or a dropped
    connection. Never a client error (4xx - e.g. 422 range unavailable) or the cost guard:
    those are deterministic and retrying just repeats them. Added 2026-09-29 after the first
    scheduled run lost 7 contracts to transient failures that succeeded on a manual retry.

    A response stream that BREAKS partway ("Read timed out", "Response ended
    prematurely") reaches us as a plain ``BentoError("Error streaming response: ...")`` -
    also transient (added 2026-10-02: three bond-futures backfill requests failed this way
    on their first and only attempt, ZFZ9 quotes, ZFZ5 and ZNH9 statistics)."""
    import aiohttp
    import requests
    from databento.common.error import BentoClientError, BentoError, BentoServerError
    if isinstance(exc, BentoServerError):
        return True
    if isinstance(exc, BentoClientError):
        return getattr(exc, "http_status", None) == 429
    if isinstance(exc, BentoError) and str(exc).startswith("Error streaming response"):
        return True
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, aiohttp.ClientConnectionError,
                            ConnectionError, TimeoutError))


def call_with_deadline(
    fn: Callable[[], T],
    *,
    what: str,
    deadline_s: float,
    attempts: int = API_ATTEMPTS,
    pause_s: float = RETRY_PAUSE_S,
) -> T:
    """``fn()`` with a hard wall-clock deadline per attempt, retried (up to ``attempts``) on
    a hang or a transient failure (``is_transient``); raises ``ApiTimeout`` if every attempt
    hangs. A non-transient exception raised BY ``fn`` propagates at once."""
    for attempt in range(1, attempts + 1):
        box: dict = {}

        def target() -> None:
            try:
                box["value"] = fn()
            except BaseException as exc:  # handed back to the caller's thread below
                box["error"] = exc

        worker = threading.Thread(target=target, daemon=True, name=f"databento:{what}")
        worker.start()
        worker.join(deadline_s)
        if not worker.is_alive():
            if "error" not in box:
                return box["value"]
            exc = box["error"]
            if not is_transient(exc) or attempt == attempts:
                raise exc
            log.warning("%s: transient %s: %s (attempt %d/%d)", what, type(exc).__name__,
                        str(exc).splitlines()[0][:160], attempt, attempts)
            time.sleep(pause_s)
            continue
        log.warning("%s: no response within %.0fs (attempt %d/%d)", what, deadline_s, attempt, attempts)
        if attempt < attempts:
            time.sleep(pause_s)
    raise ApiTimeout(f"{what}: no response within {deadline_s:.0f}s on {attempts} attempts")


def get_client(api_key: str | None = None) -> db.Historical:
    """Build a historical client from an explicit key or ``DATABENTO_API_KEY``."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {API_KEY_ENV} in the environment or in .env")
    return db.Historical(key)


def validate_absolute_symbol(symbol: str) -> str:
    """Accept a single absolute contract symbol; reject wildcards and relative tickers."""
    if not symbol or not symbol.strip() or "*" in symbol or is_relative(symbol):
        raise ValueError(
            f"{symbol!r} is not an absolute contract symbol like 'SRZ4'; wildcards, empty "
            "and relative tickers (SR3.c.0) are banned at the API boundary (Rule 2.2)."
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
    return float(call_with_deadline(
        lambda: client.metadata.get_cost(
            dataset=dataset,
            symbols=list(symbols),
            schema=schema,
            stype_in=stype_in,
            start=_utc(start),
            end=_utc(end),
        ),
        what=f"get_cost {dataset} {schema} {list(symbols)[:2]}", deadline_s=META_DEADLINE_S,
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
    store = call_with_deadline(
        lambda: client.timeseries.get_range(
            dataset=dataset,
            symbols=list(symbols),
            schema=schema,
            stype_in=stype_in,
            start=_utc(start),
            end=_utc(end),
        ),
        what=f"get_range {dataset} {schema} {list(symbols)[:2]}", deadline_s=DATA_DEADLINE_S,
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
    schema: str = SCHEMA_OHLCV,
) -> pd.DataFrame:
    """Raw OHLCV bars (``ohlcv-1m`` by default, or e.g. ``ohlcv-1s``) for ONE absolute
    contract (raw symbol) over ``[start, end)``."""
    validate_absolute_symbol(symbol)
    return _get_range(dataset, schema, [symbol], start, end, "raw_symbol", max_cost_usd, client)


def fetch_futures_bbo(
    dataset: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_cost_usd: float = MAX_COST_USD,
    client: db.Historical | None = None,
    schema: str = SCHEMA_BBO_1M,
) -> pd.DataFrame:
    """Raw ``bbo-1m`` records for ONE absolute contract over ``[start, end)``: the top of
    book (``bid_px_00``/``ask_px_00``/sizes) SAMPLED every minute, indexed by ``ts_recv``
    on the exact minute - verified 2026-09-30 (``ts_event``/``price`` there are the last
    trade, which can be much older)."""
    validate_absolute_symbol(symbol)
    return _get_range(dataset, schema, [symbol], start, end, "raw_symbol", max_cost_usd, client)


def fetch_definitions(
    dataset: str,
    parent_symbols: Sequence[str],
    day: pd.Timestamp,
    *,
    max_cost_usd: float = MAX_COST_USD,
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """Instrument ``definition`` rows for parent symbols (e.g. ``SR3.OPT``) on ONE day."""
    day = pd.Timestamp(day).normalize()
    return _get_range(
        dataset, SCHEMA_DEFINITION, parent_symbols, day, day + pd.Timedelta(days=1),
        "parent", max_cost_usd, client,
    )


# Databento caps a single request at 2,000 symbols (verified empirically 2026-09-21: a
# 3,358-instrument SR3 option chain request failed with data_exceeded_maximum_number_of_symbols).
# Kept comfortably under that, not at the exact limit.
MAX_SYMBOLS_PER_REQUEST = 1_900


def _batched(items: Sequence, size: int) -> list[list]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def fetch_ohlcv_by_instrument_ids(
    dataset: str,
    instrument_ids: Sequence[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_cost_usd: float = MAX_COST_USD,
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """Raw ``ohlcv-1m`` bars for an isolated array of instrument ids (Rule 2.3 step 3).

    Batches at ``MAX_SYMBOLS_PER_REQUEST`` - a large enough option chain can exceed
    Databento's 2,000-symbol-per-request cap even after Rule 2.3 filtering.
    """
    if not len(instrument_ids):
        raise ValueError("instrument_ids is empty; refusing an unbounded options query.")
    ids = [int(i) for i in instrument_ids]
    frames = [
        _get_range(dataset, SCHEMA_OHLCV, batch, start, end, "instrument_id", max_cost_usd, client)
        for batch in _batched(ids, MAX_SYMBOLS_PER_REQUEST)
    ]
    return pd.concat(frames)  # ids is non-empty (checked above), so frames always has >=1 entry


def fetch_statistics_by_instrument_ids(
    dataset: str,
    instrument_ids: Sequence[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_cost_usd: float = MAX_COST_USD,
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """Raw ``statistics`` rows for an isolated array of instrument ids (Rule 2.3 step 3,
    applied to settlement price / open interest instead of pricing bars). Batches at
    ``MAX_SYMBOLS_PER_REQUEST`` for the same reason as ``fetch_ohlcv_by_instrument_ids``.
    """
    if not len(instrument_ids):
        raise ValueError("instrument_ids is empty; refusing an unbounded options query.")
    ids = [int(i) for i in instrument_ids]
    frames = [
        _get_range(dataset, SCHEMA_STATISTICS, batch, start, end, "instrument_id", max_cost_usd, client)
        for batch in _batched(ids, MAX_SYMBOLS_PER_REQUEST)
    ]
    return pd.concat(frames)  # ids is non-empty (checked above), so frames always has >=1 entry


def fetch_statistics(
    dataset: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_cost_usd: float = MAX_COST_USD,
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """Raw ``statistics`` rows (settlement price, open interest, ...) for ONE absolute
    contract over ``[start, end)``. Long format: one row per (stat_type, update)."""
    validate_absolute_symbol(symbol)
    return _get_range(dataset, SCHEMA_STATISTICS, [symbol], start, end, "raw_symbol", max_cost_usd, client)


def fetch_statistics_bulk(
    dataset: str,
    symbols: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_cost_usd: float = MAX_COST_USD,
    client: db.Historical | None = None,
) -> pd.DataFrame:
    """Raw ``statistics`` rows for MANY absolute contracts over ``[start, end)``, with the
    ``symbol`` column naming each row's contract; batched under the per-request symbol cap.
    One request per batch instead of one per contract: the per-contract path's latency is
    the bottleneck of a long backfill (TOFIX "Daily statistics requests are slow")."""
    for s in symbols:
        validate_absolute_symbol(s)
    frames = [
        _get_range(dataset, SCHEMA_STATISTICS, batch, start, end, "raw_symbol", max_cost_usd, client)
        for batch in _batched(list(symbols), MAX_SYMBOLS_PER_REQUEST)
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames) if frames else pd.DataFrame()


_AVAILABLE_END_TTL_S = 300.0
_available_end_cache: dict[tuple[str, str], tuple[float, pd.Timestamp]] = {}


def available_end(dataset: str, schema: str, client: db.Historical | None = None) -> pd.Timestamp:
    """The latest instant ``schema`` can be queried for on ``dataset`` (tz-naive UTC), from
    the free ``metadata.get_dataset_range``. Any request ending after it is rejected
    (422). Verified 2026-09-28 at 19:34 UTC: GLBX advertised 11:34:56 - exactly 8h behind
    now - and requests ending anywhere past it were rejected (``dataset_unavailable_range``),
    even though another 422's message claimed availability "up to 19:20"; that message is
    not the queryable bound, this is. XEUR advertised the previous day's 22:00. Note also
    that omitting ``end`` does NOT mean "up to available": Databento forward-fills ``end``
    from ``start``'s precision (a date = that one day). Cached for a few minutes: free,
    but otherwise called once per contract."""
    key = (dataset, schema)
    hit = _available_end_cache.get(key)
    if hit and time.monotonic() - hit[0] < _AVAILABLE_END_TTL_S:
        return hit[1]
    client = client or get_client()
    rng = call_with_deadline(lambda: client.metadata.get_dataset_range(dataset=dataset),
                             what=f"get_dataset_range {dataset}", deadline_s=META_DEADLINE_S)
    raw = rng.get("schema", {}).get(schema, rng)["end"]
    end = pd.Timestamp(raw).tz_convert("UTC").tz_localize(None)
    _available_end_cache[key] = (time.monotonic(), end)
    return end


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
