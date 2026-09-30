"""WIRP (Fed rate-probability) data layer: reads already-cached ZQ settlements (CLOSE)
or 1-minute bars (LIVE) and builds the per-meeting schedule with infra.analytics.wirp.
Never fetches - populate ZQ data first (scripts/update_futures.py for LIVE, the daily
cycle or scripts/update_daily.py for CLOSE). See CLAUDE.md sections 11 and 12.

Used by BOTH the dashboard page (infra/dashboard/wirp_*) and the daily cycle
(infra.cycle.derived) - which is why it lives here and not under infra/dashboard: the
dependency direction is dashboard -> pipeline, never the reverse (CLAUDE.md section 3).
It started as infra/dashboard/wirp_selectors.py.

Every read function takes its storage root/contracts-file as an explicit keyword
default (the real config paths) rather than reaching for the config constant inside
the function body - the same pattern infra.pipeline.daily_options.load_daily_options
was fixed to use after an early test once leaked fake rows into the real ~/Database
(see TOFIX.md's `infra.pipeline.options.load_options` entry for the failure mode this
avoids); this module uses it from the start.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import wirp
from infra.config import ADJUSTMENTS_DIR, DAILY_FUTURES_DIR, FOMC_MEETINGS, FUTURES_CONTRACTS_FILE, FUTURES_DIR
from infra.processing import statistics as stats
from infra.processing import transforms as tf
from infra.storage import adjustment_store, contract_store, parquet_store

ROOT = "ZQ"
_ANCHOR_LOOKBACK_MONTHS = 6  # FOMC meetings are never more than a couple months apart


def zq_contracts(contracts_file: Path = FUTURES_CONTRACTS_FILE) -> pd.DataFrame:
    """Known ZQ absolute contracts (root/ticker/expiry), sorted by expiry."""
    return contract_store.read_contracts(contracts_file, ROOT)


def contract_for_month(month: pd.Period, contracts: pd.DataFrame) -> str | None:
    """The ZQ absolute ticker whose expiry falls inside ``month`` (None if not cached).
    Matched by the contract's real ``expiry`` date, never by parsing the ticker's month
    code - CLAUDE.md section 5's symbology caution (Eurex raw symbols aren't even
    CME-style, so this project never relies on ticker parsing for contract identity)."""
    hit = contracts[(contracts["expiry"] >= month.start_time) & (contracts["expiry"] <= month.end_time)]
    return None if hit.empty else str(hit.iloc[0]["ticker"])


def month_contract_map(months: list[pd.Period], contracts: pd.DataFrame) -> dict[pd.Period, str]:
    """{month: ticker} for months with a cached ZQ contract; others simply omitted."""
    out = {}
    for month in months:
        ticker = contract_for_month(month, contracts)
        if ticker is not None:
            out[month] = ticker
    return out


def _latest_rates(
    month_tickers: dict[pd.Period, str], root: Path, columns: list[str], decode, price_column: str,
    *, as_of: pd.Timestamp | None = None, adjust=None,
) -> tuple[pd.Series, pd.Timestamp | None]:
    """Shared read for close_rates/live_rates: the LATEST cached price per ticker AS
    OF ``as_of`` (default: no cutoff, i.e. the latest row in the whole dataset),
    converted to that ticker's implied average rate.

    This ``as_of`` cutoff is what makes the function usable for BOTH live dashboard
    use (the default) and a point-in-time historical backfill (an explicit ``as_of``) -
    see CLAUDE.md section 3's "point-in-time cutoff" convention, and
    infra.pipeline.wirp.backfill_schedule for the worked example. Without
    it, asking "what did this look like on day X" would silently read whatever the
    LATEST cached row happens to be regardless of X - look-ahead bias, not history.
    """
    if not month_tickers:
        return pd.Series(dtype="float64"), None
    end = as_of + pd.Timedelta(days=1) if as_of is not None else None  # read_partitioned's end is exclusive
    raw = parquet_store.read_partitioned(root, end=end, equals_in={"ticker": list(month_tickers.values())})
    if raw is None or raw.empty:
        return pd.Series(dtype="float64"), None
    df = decode(raw[columns])
    if adjust is not None:
        df = adjust(df)
    df = df.dropna(subset=[price_column])
    if df.empty:
        return pd.Series(dtype="float64"), None
    latest = df.sort_values("timestamp").groupby("ticker", observed=True).last()
    latest_as_of = df["timestamp"].max()
    ticker_to_month = {t: m for m, t in month_tickers.items()}
    rates = {
        ticker_to_month[ticker]: wirp.implied_rate(row[price_column])
        for ticker, row in latest.iterrows() if ticker in ticker_to_month
    }
    return pd.Series(rates).sort_index(), latest_as_of


def close_rates(
    month_tickers: dict[pd.Period, str], *, root: Path = DAILY_FUTURES_DIR, as_of: pd.Timestamp | None = None,
    adjustments_dir: Path | None = None,
) -> tuple[pd.Series, pd.Timestamp | None]:
    """CLOSE mode: latest cached OFFICIAL SETTLEMENT price per contract, as of
    ``as_of`` (default: no cutoff - the latest row on disk), with the adjustments log
    overlaid (a bad print NA'd or rolled - CLAUDE.md 12). An NA'd latest settlement
    falls back to the contract's previous one, exactly as any missing settlement does."""
    from infra.pipeline.daily import STORE

    def adjust(df: pd.DataFrame) -> pd.DataFrame:
        adj = adjustment_store.read(ADJUSTMENTS_DIR if adjustments_dir is None else adjustments_dir,
                                    store=STORE, keys=list(month_tickers.values()))
        return adjustment_store.apply(df, adj, key_column="ticker")

    return _latest_rates(month_tickers, root, stats.DAILY_COLUMNS, stats.decode_daily, "settlement_price",
                         as_of=as_of, adjust=adjust)


def live_rates(
    month_tickers: dict[pd.Period, str], *, root: Path = FUTURES_DIR, as_of: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.Timestamp | None]:
    """LIVE mode: latest cached 1-MINUTE bar close per contract, as of ``as_of``
    (default: no cutoff - the latest row on disk). Not a real-time feed (this project
    only ever calls Databento's Historical API) - it's the freshest bar already on
    disk; ``as_of`` (the return value, not the cutoff parameter of the same name - see
    infra.dashboard.wirp_charts) is always surfaced next to it so the page never
    implies it's more current than it is."""
    return _latest_rates(month_tickers, root, tf.FUTURES_COLUMNS, tf.decode_futures, "close", as_of=as_of)


def find_anchor(
    before_month: pd.Period,
    contracts: pd.DataFrame,
    rate_fn,
    meeting_months: set[pd.Period],
    *,
    max_lookback: int = _ANCHOR_LOOKBACK_MONTHS,
) -> tuple[pd.Period, float] | None:
    """Walk backward from ``before_month`` for the nearest FLAT month with a cached ZQ
    contract - its price directly gives the prevailing rate with no day-weighting
    needed, since nothing changes mid-flat-month. This is the self-consistent "current
    rate" anchor for the chain (no external EFFR feed - CLAUDE.md section 11).

    ``meeting_months`` must cover EVERY month that ever had an FOMC meeting (past or
    future, not just upcoming ones): a past meeting month's contract, even read long
    after settlement, is itself a day-weighted blend of two rates, not a flat one - so
    it would be wrong to anchor on it even though its own meeting has already happened.
    """
    month = before_month - 1
    for _ in range(max_lookback):
        if month not in meeting_months:
            ticker = contract_for_month(month, contracts)
            if ticker is not None:
                rates, _ = rate_fn({month: ticker})
                if month in rates.index and pd.notna(rates[month]):
                    return month, float(rates[month])
        month -= 1
    return None


def build_schedule(
    mode: str,
    *,
    today: pd.Timestamp | None = None,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    close_root: Path = DAILY_FUTURES_DIR,
    live_root: Path = FUTURES_DIR,
    adjustments_dir: Path | None = None,
    settlement_rates_fn=None,
    toggle_rates_fn=None,
) -> tuple[pd.DataFrame, dict]:
    """Main entry point: (long-format schedule, meta dict) for ``mode`` ("close" or
    "live"). Empty schedule + a human ``meta["status"]`` message when data isn't cached
    yet, rather than raising - this page must degrade gracefully (CLAUDE.md section 10).

    ``settlement_rates_fn`` / ``toggle_rates_fn`` (``{month: ticker} -> (rates, as_of)``)
    replace the default disk reads - how ``intraday_schedules`` prices each grid time
    from prices preloaded ONCE (``PanelRates``), still through this one function."""
    if mode not in ("close", "live"):
        raise ValueError(f"mode must be 'close' or 'live', got {mode!r}")
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    today = today.normalize()

    all_meeting_months = {pd.Period(m.end_date, freq="M") for m in FOMC_MEETINGS}
    upcoming = [m for m in FOMC_MEETINGS if pd.Timestamp(m.end_date) > today]
    empty_meta = {"mode": mode, "as_of": None, "anchor_month": None, "anchor_rate": None}
    if not upcoming:
        return pd.DataFrame(), {**empty_meta, "status": "No upcoming FOMC meetings configured - "
                                                          "update infra.config.FOMC_MEETINGS."}

    upcoming_dates = [pd.Timestamp(m.end_date) for m in upcoming]
    earliest_upcoming_month = min(pd.Period(d, freq="M") for d in upcoming_dates)
    last_known_month = max(all_meeting_months)  # last month FOMC_MEETINGS actually covers

    contracts = zq_contracts(contracts_file)
    # The anchor, and any already-past meeting the chain runs through, represent KNOWN,
    # settled fact - not a live-evolving prediction - so they always read from the
    # official settlement, regardless of the LIVE/CLOSE toggle. Only genuinely upcoming
    # meetings (and the reference months used to price them) follow the toggle: those
    # are real forecasts a viewer might want to watch move intraday. Both are cut off
    # at ``today`` (CLAUDE.md section 3's "point-in-time cutoff" convention) - without
    # it, calling this with a PAST ``today`` (a backfill) would silently read whatever
    # the latest cached row happens to be now, not what was known as of that day.
    settlement_fn = settlement_rates_fn or (
        lambda mt: close_rates(mt, root=close_root, as_of=today, adjustments_dir=adjustments_dir))
    toggle_fn = toggle_rates_fn or (
        settlement_fn if mode == "close" else (lambda mt: live_rates(mt, root=live_root, as_of=today)))

    anchor = find_anchor(earliest_upcoming_month, contracts, settlement_fn, all_meeting_months)
    if anchor is None:
        return pd.DataFrame(), {
            **empty_meta,
            "status": "No flat (non-meeting) ZQ settlement cached before the first upcoming "
                      "meeting to anchor the current rate - fetch an earlier month.",
        }
    anchor_month, anchor_rate = anchor

    # CHAIN meetings = every FOMC meeting after the anchor month, not just the upcoming
    # ones. The anchor's flat month can sit several meetings back (find_anchor walks
    # past any meeting month while searching for a flat one) - an already-past meeting
    # in that gap (e.g. today is late September and September itself just met) still
    # moved the prevailing rate, and skipping it would silently leave the anchor's
    # PRE-hike/cut rate feeding straight into the first upcoming meeting instead of the
    # real current rate. All of them are priced so the chain is correct; only the
    # upcoming subset is returned for display.
    upcoming_set = set(upcoming_dates)
    chain_meetings = [m for m in FOMC_MEETINGS if pd.Period(m.end_date, freq="M") > anchor_month]
    chain_dates = [pd.Timestamp(m.end_date) for m in chain_meetings]
    chain_months = {pd.Period(d, freq="M") for d in chain_dates}
    past_chain_months = {pd.Period(d, freq="M") for d in chain_dates if d not in upcoming_set}
    upcoming_months = {pd.Period(d, freq="M") for d in upcoming_dates}

    # infra.analytics.wirp.meeting_schedule prefers reading the FLAT month right after
    # a meeting (no day-weighting needed there) over day-weighting the meeting's own
    # month - so its rate needs to be fetched too, not just the meeting months
    # themselves, or that preferred path can never be taken (see CLAUDE.md section 11).
    # Capped at ``last_known_month``: a month past FOMC_MEETINGS's last entry is
    # UNKNOWN, not confirmed flat - the Fed's calendar page simply doesn't extend that
    # far yet, it doesn't say there's no meeting there. Treating it as flat would risk
    # reintroducing the exact bug this preference was built to avoid (a real meeting
    # hiding in a month we wrongly assumed was flat). Each next-month reference is
    # settlement or toggle depending on whether the meeting it BACKS is past or upcoming.
    next_month_source = {
        m + 1: m for m in chain_months
        if (m + 1) not in chain_months and (m + 1) <= last_known_month
    }
    settlement_months = sorted(past_chain_months | {
        nm for nm, src in next_month_source.items() if src in past_chain_months
    })
    toggle_months = sorted(upcoming_months | {
        nm for nm, src in next_month_source.items() if src in upcoming_months
    })
    needed_months = sorted(set(settlement_months) | set(toggle_months))

    month_tickers = month_contract_map(needed_months, contracts)
    if not month_tickers:
        return pd.DataFrame(), {
            **empty_meta, "anchor_month": str(anchor_month), "anchor_rate": anchor_rate,
            "status": "No cached ZQ contracts for any upcoming meeting month yet - run "
                      "scripts/update_futures.py / scripts/update_daily.py for ZQ first.",
        }

    settlement_tickers = {m: t for m, t in month_tickers.items() if m in settlement_months}
    toggle_tickers = {m: t for m, t in month_tickers.items() if m in toggle_months}
    settlement_rates, _ = settlement_fn(settlement_tickers) if settlement_tickers else (pd.Series(dtype="float64"), None)
    toggle_rates, as_of = toggle_fn(toggle_tickers) if toggle_tickers else (pd.Series(dtype="float64"), None)
    rates = pd.concat([settlement_rates, toggle_rates]).sort_index()

    full_schedule = wirp.meeting_schedule(rates, chain_dates, anchor_rate)
    # Already-past meetings were priced only to chain the rate correctly - trim them
    # before returning; they're realized history now, not an upcoming prediction.
    schedule = full_schedule[full_schedule["meeting_date"].isin(upcoming_set)].reset_index(drop=True)

    priced = schedule["meeting_date"].nunique() if not schedule.empty else 0
    meta = {
        "mode": mode, "as_of": as_of, "anchor_month": str(anchor_month), "anchor_rate": anchor_rate,
        "status": (
            f"{priced} of {len(upcoming)} upcoming meetings priced from cached ZQ data"
            if priced else "Cached ZQ contracts don't cover any upcoming meeting month yet."
        ),
    }
    return schedule, meta


def available_days(
    mode: str,
    *,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    close_root: Path = DAILY_FUTURES_DIR,
    live_root: Path = FUTURES_DIR,
) -> list[pd.Timestamp]:
    """Distinct trading days with ANY cached ZQ price data for ``mode`` - the valid
    ``today`` values ``backfill_schedule`` can meaningfully iterate over."""
    contracts = zq_contracts(contracts_file)
    tickers = contracts["ticker"].tolist()
    if not tickers:
        return []
    root = close_root if mode == "close" else live_root
    raw = parquet_store.read_partitioned(root, equals_in={"ticker": tickers}, columns=["timestamp"])
    if raw is None or raw.empty:
        return []
    return sorted(pd.to_datetime(raw["timestamp"]).dt.normalize().unique())


def backfill_schedule(
    mode: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    close_root: Path = DAILY_FUTURES_DIR,
    live_root: Path = FUTURES_DIR,
    adjustments_dir: Path | None = None,
) -> pd.DataFrame:
    """A historical WIRP time series: re-runs ``build_schedule`` for every cached
    trading day (optionally bounded to ``[start, end)``) and stacks the results - one
    row per (as-of day, meeting, outcome level), ``infra.analytics.wirp.
    meeting_schedule``'s own long format plus an ``as_of`` column.

    Deliberately just a loop over ``build_schedule`` - no separate backfill-only
    pricing logic - so a backfilled day is computed exactly the way the live dashboard
    would have shown it THAT day (CLAUDE.md section 3's cutoff convention makes this
    possible: ``build_schedule``'s own ``today``/``as_of`` plumbing is what stops a
    backfill from leaking in today's actual latest price). No API calls; read-only
    against whatever's already cached (``scripts/backfill_wirp.py`` is the CLI wrapper).
    """
    days = available_days(mode, contracts_file=contracts_file, close_root=close_root, live_root=live_root)
    if start is not None:
        days = [d for d in days if d >= pd.Timestamp(start)]
    if end is not None:
        days = [d for d in days if d < pd.Timestamp(end)]

    rows = []
    for day in days:
        schedule, meta = build_schedule(
            mode, today=day, contracts_file=contracts_file, close_root=close_root, live_root=live_root,
            adjustments_dir=adjustments_dir,
        )
        if schedule.empty:
            continue
        schedule = schedule.copy()
        schedule.insert(0, "as_of", day)
        schedule["anchor_month"] = meta["anchor_month"]
        schedule["anchor_rate"] = meta["anchor_rate"]
        rows.append(schedule)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------------- intraday
class PanelRates:
    """Prices preloaded ONCE for many point-in-time lookups (intraday grids, 1-second
    event windows) - the in-memory equivalent of ``_latest_rates``: for each ticker, the
    latest price KNOWN at or before ``as_of``, as its implied rate.

    ``frame``: ``timestamp, ticker, <price_column>``; ``known_after``: how long after
    its timestamp a row is known - 0 for a quote sample (the book AT that instant), one
    bar-length for an OHLCV bar (stamped at its START, complete one bar later)."""

    def __init__(self, frame: pd.DataFrame, price_column: str, known_after: pd.Timedelta = pd.Timedelta(0)):
        df = frame.dropna(subset=[price_column])
        self._by_ticker = {}
        for ticker, g in df.groupby(df["ticker"].astype(str), observed=True):
            g = g.sort_values("timestamp")
            known = (pd.to_datetime(g["timestamp"]) + known_after).to_numpy(dtype="datetime64[ns]")
            self._by_ticker[ticker] = (known, g[price_column].to_numpy(dtype=float))
        self.last_used: dict[str, pd.Timestamp] = {}  # ticker -> known-time of the price used

    def known_times(self) -> np.ndarray:
        arrays = [k for k, _ in self._by_ticker.values()]
        return np.sort(np.concatenate(arrays)) if arrays else np.array([], dtype="datetime64[ns]")

    def rates(self, month_tickers: dict[pd.Period, str], as_of: pd.Timestamp) -> tuple[pd.Series, pd.Timestamp | None]:
        cutoff = np.datetime64(pd.Timestamp(as_of), "ns")
        out, self.last_used = {}, {}
        for month, ticker in month_tickers.items():
            if ticker not in self._by_ticker:
                continue
            known, price = self._by_ticker[ticker]
            i = np.searchsorted(known, cutoff, side="right") - 1
            if i < 0:
                continue
            out[month] = wirp.implied_rate(price[i])
            self.last_used[ticker] = pd.Timestamp(known[i])
        latest = max(self.last_used.values()) if self.last_used else None
        return pd.Series(out, dtype="float64").sort_index(), latest


# Intraday price sources for WIRP's upcoming months: store root, reader, price column,
# and how long after its timestamp a row is known (see PanelRates).
def _intraday_source(source: str):
    from infra.config import BBO_1S_FUTURES_DIR, BBO_FUTURES_DIR, OHLCV_1S_FUTURES_DIR
    from infra.pipeline import bbo
    from infra.pipeline.futures import read_futures_from_disk
    sources = {
        "bbo-1m": (BBO_FUTURES_DIR, bbo.read_bbo_from_disk, "mid", pd.Timedelta(0)),
        "bbo-1s": (BBO_1S_FUTURES_DIR, bbo.read_bbo_from_disk, "mid", pd.Timedelta(0)),
        "ohlcv-1m": (FUTURES_DIR, read_futures_from_disk, "close", pd.Timedelta(minutes=1)),
        "ohlcv-1s": (OHLCV_1S_FUTURES_DIR, read_futures_from_disk, "close", pd.Timedelta(seconds=1)),
    }
    if source not in sources:
        raise ValueError(f"unknown intraday source {source!r}; one of {sorted(sources)}")
    return sources[source]


INTRADAY_SOURCES = ("bbo-1m", "bbo-1s", "ohlcv-1m", "ohlcv-1s")
_SETTLEMENT_LOOKBACK = pd.Timedelta(days=200)  # anchor months sit up to ~6 months back


def settlement_cutoff(at: pd.Timestamp) -> pd.Timestamp:
    """The last trading day whose settlement is KNOWN at instant ``at`` (UTC): the one
    before ``at``'s own CME trading day (infra.trading_calendar) - that session's own
    settlement is only published after it closes."""
    from infra.trading_calendar import trading_day
    return trading_day(pd.DatetimeIndex([pd.Timestamp(at)]), "GLBX.MDP3")[0] - pd.Timedelta(days=1)


def intraday_schedules(
    start,
    end,
    *,
    grid: str | None = None,
    source: str = "bbo-1m",
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    close_root: Path = DAILY_FUTURES_DIR,
    intraday_root: Path | None = None,
    adjustments_dir: Path | None = None,
) -> pd.DataFrame:
    """WIRP at every ``grid`` time in ``[start, end)`` (UTC) - ``build_schedule``, point
    in time at each instant: upcoming months from ``source`` prices KNOWN by then
    (``bbo-1m`` quote mids by default; ``bbo-1s``/``ohlcv-1s`` for small event windows
    - pass e.g. ``grid="1s"``), settled months from settlements published before then
    (``settlement_cutoff``). Grid times with no ``source`` update in the preceding grid
    interval (weekends, the daily halt) are skipped. One row per (grid time, meeting,
    outcome) - the daily WIRP's columns plus ``price_as_of`` / ``oldest_price_as_of``
    (the newest and STALEST upcoming-month price used: a thin contract's quote can lag).

    Meeting inclusion stays day-granular, as in ``build_schedule``: a meeting counts as
    upcoming until its decision DAY begins (UTC), so on an FOMC day itself that
    meeting is already treated as past (TOFIX.md)."""
    from infra.config import WIRP_INTRADAY_GRID
    from infra.pipeline.daily import read_daily_from_disk
    grid = grid or WIRP_INTRADAY_GRID
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    root, read, price_column, known_after = _intraday_source(source)
    root = intraday_root or root
    tickers = zq_contracts(contracts_file)["ticker"].astype(str).tolist()
    if not tickers:
        return pd.DataFrame()

    live = PanelRates(read(tickers, start - pd.Timedelta(days=7), end, root=root), price_column, known_after)
    settled = read_daily_from_disk(tickers, start - _SETTLEMENT_LOOKBACK, end, root=close_root,
                                   adjustments_dir=adjustments_dir)
    close = PanelRates(settled, "settlement_price")

    step = pd.Timedelta(grid)
    known = live.known_times()
    frames = []
    for at in pd.date_range(start, end, freq=grid, inclusive="left"):
        lo, hi = np.datetime64(at - step, "ns"), np.datetime64(at, "ns")
        if np.searchsorted(known, hi, side="right") == np.searchsorted(known, lo, side="right"):
            continue  # nothing new in the interval: market closed
        cutoff = settlement_cutoff(at)
        schedule, meta = build_schedule(
            "live", today=at, contracts_file=contracts_file,
            settlement_rates_fn=lambda mt, c=cutoff: close.rates(mt, c),
            toggle_rates_fn=lambda mt, a=at: live.rates(mt, a),
        )
        if schedule.empty:
            continue
        used = live.last_used
        frames.append(schedule.assign(
            timestamp=at, anchor_month=meta["anchor_month"], anchor_rate=meta["anchor_rate"],
            price_as_of=meta["as_of"], oldest_price_as_of=min(used.values()) if used else None, source=source))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ("timestamp", "meeting_date", "price_as_of", "oldest_price_as_of"):
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ms]")
    df["outcome_step"] = df["outcome_step"].astype("int32")
    return df


WIRP_1S_KEYS = ["timestamp", "meeting_date", "outcome_step", "source"]


def store_wirp_1s(start, end, *, source: str = "bbo-1s", root: Path | None = None, **kw) -> pd.DataFrame:
    """Ad hoc: compute WIRP every second in ``[start, end)`` (UTC) from ``source``
    (``bbo-1s`` quote mids by default, or ``ohlcv-1s``) and SAVE it to
    ``Derived/WIRP_1s``, replacing that window's rows for that source (a re-run never
    leaves stale seconds). Reads only what is already on disk - load the 1-second data
    first (``infra.pipeline.bbo.load_bbo_1s`` / ``infra.pipeline.ohlcv_1s.load_ohlcv_1s``).
    ``kw`` go to ``intraday_schedules`` (store roots). Returns the rows saved."""
    from infra.config import WIRP_1S_DIR
    root = WIRP_1S_DIR if root is None else root
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    df = intraday_schedules(start, end, grid="1s", source=source, **kw)
    parquet_store.delete_where(root, lambda part: (part["source"] == source)
                               & pd.to_datetime(part["timestamp"]).between(start, end, inclusive="left"))
    if not df.empty:
        parquet_store.write_partitioned(df, root, WIRP_1S_KEYS)
    return df


def read_wirp_1s(start, end, *, source: str | None = None, root: Path | None = None) -> pd.DataFrame:
    """Saved 1-second WIRP rows in ``[start, end)`` (optionally one ``source``). No compute."""
    from infra.config import WIRP_1S_DIR
    root = WIRP_1S_DIR if root is None else root
    df = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end),
                                        equals_in={"source": [source]} if source else None)
    return pd.DataFrame() if df is None else df.sort_values(WIRP_1S_KEYS).reset_index(drop=True)

