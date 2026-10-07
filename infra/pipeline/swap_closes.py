"""Benchmark swap closes (CLAUDE.md 16): read the archived DTCC reports, compute every
configured close (``SWAP_CLOSES``) for every currency it covers, store them under
``Derived/SwapCloses``. Reads disk only - the archive is filled by the cycle's ``dtcc``
raw source (infra.pipeline.dtcc).

* A day D's trades come from file D (every snap window sits inside one UTC day -
  asserted); their corrections and cancellations from files D..D+SWAP_CORRECTION_DAYS,
  as far as they exist (and only those published by ``as_of``, if given).
* A recomputed day REPLACES that day's rows (a tenor with no print any more must not keep
  a stale close), like the daily WIRP.
* Two methods: ``pure`` always; ``adjusted`` (futures-adjusted) for currencies with hedges
  (``SWAP_HEDGES``) on days whose hedge quotes and ratio are on disk
  (infra.pipeline.swap_hedge).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.config import (
    DTCC_DIR,
    SWAP_CLOSE_WEIGHTING,
    SWAP_CLOSES,
    SWAP_CLOSES_DIR,
    SWAP_CORRECTION_DAYS,
    SWAP_CURVES,
    SWAP_HEDGES,
    SwapCloseSpec,
)
from infra.pipeline import dtcc
from infra.processing import dtcc_trades as dt
from infra.pipeline.swap_hedge import HedgeBook, build_hedge_book, futures_moves
from infra.processing.swap_closes import CLOSE_COLUMNS, adjusted_closes, pure_closes
from infra.storage import parquet_store
from infra.trading_calendar import snap_instants

log = logging.getLogger(__name__)
KEYS = ["timestamp", "close", "currency", "tenor", "method"]
REPORT = "RATES"
_ONE_DAY = pd.Timedelta(days=1)


def _file(d: pd.Timestamp, dtcc_root: Path, cache: dict | None) -> pd.DataFrame:
    """One archived file, normalized - from ``cache`` if a backfill already parsed it."""
    if cache is not None and d in cache:
        return cache[d]
    raw = dtcc.read_dtcc_day(REPORT, d, columns=list(dt.RAW_COLUMNS), root=dtcc_root)
    out = dt.normalize(raw) if not raw.empty else raw
    if cache is not None:
        cache[d] = out
    return out


def day_events(day, *, as_of=None, dtcc_root: Path = DTCC_DIR, cache: dict | None = None) -> pd.DataFrame:
    """Normalized records from file ``day`` and the correction files after it. No network."""
    day = pd.Timestamp(day).normalize()
    last = day + pd.Timedelta(days=SWAP_CORRECTION_DAYS)
    if as_of is not None:
        last = min(last, pd.Timestamp(as_of).normalize())
    frames = [f for f in (_file(d, dtcc_root, cache) for d in pd.date_range(day, last)) if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def par_trades_on(day, currency: str, events: pd.DataFrame, *, as_of=None) -> pd.DataFrame:
    """One currency's clean par trades EXECUTED on UTC day ``day``."""
    spec = SWAP_CURVES[currency]
    if events.empty:
        return pd.DataFrame(columns=dt.TRADE_COLUMNS)
    current = dt.current_trades(dt.trade_events(dt.product_rows(events, spec)), as_of)
    trades = dt.par_trades(current, spec, currency)
    day = pd.Timestamp(day).normalize()
    return trades[(trades["executed"] >= day) & (trades["executed"] < day + _ONE_DAY)].reset_index(drop=True)


def _window_inside_day(instant: pd.Timestamp, spec: SwapCloseSpec, day: pd.Timestamp) -> None:
    half = pd.Timedelta(minutes=max(spec.pure_fallback_half_window_min, spec.adjusted_half_window_min))
    if not (day <= instant - half and instant + half < day + _ONE_DAY):
        raise ValueError(f"snap {instant} +-{half} crosses UTC day {day.date()}: it would need two DTCC files")


def compute_closes(day, *, closes: dict[str, SwapCloseSpec] = SWAP_CLOSES, as_of=None,
                   dtcc_root: Path = DTCC_DIR, cache: dict | None = None,
                   book: HedgeBook | None = None, methods: tuple[str, ...] = ("pure", "adjusted")) -> pd.DataFrame:
    """Every close in ``closes`` x currency on ``day``, both methods where possible. No
    network, no writes. ``book`` (hedges) is built for the day if not given. ``methods``:
    ``("pure",)`` skips the futures-adjusted closes and their hedge book (which reads
    intraday ``bbo-1m`` quotes) - the daily cycle's ``derived`` step runs pure only."""
    day = pd.Timestamp(day).normalize()
    hedged = sorted({c for spec in closes.values() for c in spec.currencies if c in SWAP_HEDGES}) \
        if "adjusted" in methods else []
    if book is None and hedged:
        book = build_hedge_book(day, day, currencies=hedged)
    events = day_events(day, as_of=as_of, dtcc_root=dtcc_root, cache=cache)
    trades = {ccy: par_trades_on(day, ccy, events, as_of=as_of)
              for ccy in sorted({c for spec in closes.values() for c in spec.currencies})}
    out = []
    for name, spec in closes.items():
        instant = snap_instants([day], spec.local_time, spec.timezone)[0]
        for ccy in spec.currencies:
            cs = spec.for_currency(ccy)
            _window_inside_day(instant, cs, day)
            out.append(pure_closes(trades[ccy], instant, cs, SWAP_CLOSE_WEIGHTING, close_name=name, currency=ccy))
            if ccy in hedged and len(trades[ccy]):
                moved = trades[ccy].assign(futures_move_bp=futures_moves(trades[ccy], instant, ccy, day, book))
                out.append(adjusted_closes(moved, instant, spec, SWAP_CLOSE_WEIGHTING, close_name=name, currency=ccy))
    out = [o for o in out if not o.empty]
    return pd.concat(out, ignore_index=True)[CLOSE_COLUMNS] if out else pd.DataFrame(columns=CLOSE_COLUMNS)


def store_closes(day, closes: pd.DataFrame, *, root: Path = SWAP_CLOSES_DIR) -> int:
    """Replace ``day``'s rows with ``closes`` (FILES ONLY)."""
    day = pd.Timestamp(day).normalize()
    parquet_store.delete_where(root, lambda part: pd.to_datetime(part["timestamp"]).between(
        day, day + _ONE_DAY, inclusive="left"))
    if closes.empty:
        return 0
    out = closes.astype({"timestamp": "datetime64[ms]", "tenor": "int32", "n_trades": "int32",
                         "half_window_min": "int32"})
    parquet_store.write_partitioned(out, root, KEYS)
    return len(out)


def backfill_swap_closes(start, end, *, as_of=None, root: Path = SWAP_CLOSES_DIR, dtcc_root: Path = DTCC_DIR,
                         hedge_paths: dict | None = None) -> dict:
    """Compute and store every day in ``[start, end]`` (inclusive). Days with nothing
    archived are skipped (their rows are left as they are)."""
    rows, days, cache = 0, 0, {}
    have = dtcc.archived_days(REPORT, root=dtcc_root)
    book = build_hedge_book(start, end, **(hedge_paths or {}))
    for day in pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()):
        for d in [d for d in cache if d < day]:  # each file parsed once across the whole run
            del cache[d]
        if day not in have:
            continue
        rows += store_closes(day, compute_closes(day, as_of=as_of, dtcc_root=dtcc_root, cache=cache, book=book),
                             root=root)
        days += 1
    log.info("swap closes: %d day(s), %d row(s)", days, rows)
    return {"days": days, "rows": rows}


def read_swap_closes(start, end, *, close: str | None = None, currency: str | None = None, method: str | None = None,
                     root: Path = SWAP_CLOSES_DIR) -> pd.DataFrame:
    """Stored closes with ``timestamp`` in ``[start, end)``, optionally filtered. No compute.
    ``method="best"``: per snap and tenor the ADJUSTED close where one exists, else the PURE
    one (rows keep their own ``method``)."""
    if method == "best":
        both = read_swap_closes(start, end, close=close, currency=currency, root=root)
        if both.empty:
            return both
        both = both.assign(_rank=both["method"].astype(str).map({"adjusted": 0, "pure": 1}))
        both = both.dropna(subset=["_rank"]).sort_values("_rank")
        best = both.drop_duplicates(["timestamp", "close", "currency", "tenor"], keep="first").drop(columns="_rank")
        return best.sort_values(KEYS).reset_index(drop=True)
    eq = {k: [v] for k, v in (("close", close), ("currency", currency), ("method", method)) if v is not None}
    df = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end), equals_in=eq or None)
    if df is None or df.empty:
        return pd.DataFrame(columns=CLOSE_COLUMNS)
    return df[CLOSE_COLUMNS].sort_values(KEYS).reset_index(drop=True)
