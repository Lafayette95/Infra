"""Roll rules: which ABSOLUTE contract is the Nth relative one on each date.

Pure functions (no I/O). A mapping is a frame indexed by UTC day whose integer columns
are ranks (0 = front) and whose values are absolute tickers (``None`` if unavailable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def day_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Daily UTC index over ``[start, end)`` at ms resolution (matches stored timestamps)."""
    idx = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(),
                        freq="D", inclusive="left")
    return idx.astype("datetime64[ms]")


def _eligible_contracts(contracts: pd.DataFrame, expiry_months: tuple[int, ...]) -> pd.DataFrame:
    """Outrights on the ranked expiry cycle, sorted by expiry then ticker."""
    c = contracts[contracts["expiry"].dt.month.isin(expiry_months)]
    return c.drop_duplicates(subset=["ticker", "expiry"]).sort_values(["expiry", "ticker"]).reset_index(drop=True)


def calendar_mapping(
    contracts: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    max_rank: int,
    expiry_months: tuple[int, ...] = (3, 6, 9, 12),
    roll_offset_days: int = 0,
) -> pd.DataFrame:
    """Rank c.N: the Nth contract by expiry that has not yet expired on each date.

    A contract stays front until ``roll_offset_days`` before its expiry day, inclusive
    of the expiry day itself when the offset is 0.
    """
    c = _eligible_contracts(contracts, expiry_months)
    columns = list(range(max_rank + 1))
    if c.empty:
        return pd.DataFrame(None, index=dates, columns=columns, dtype=object)
    expiry = c["expiry"].dt.normalize().to_numpy("datetime64[D]")
    tickers = c["ticker"].to_numpy(dtype=object)
    threshold = dates.to_numpy("datetime64[D]") + np.timedelta64(roll_offset_days, "D")
    first = np.searchsorted(expiry, threshold, side="left")  # first contract with expiry >= threshold
    out = {}
    for rank in columns:
        idx = first + rank
        picked = tickers[np.minimum(idx, len(tickers) - 1)]
        out[rank] = np.where(idx < len(tickers), picked, None)
    return pd.DataFrame(out, index=dates, dtype=object)


def volume_mapping(
    daily_volume: pd.DataFrame,
    contracts: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    max_rank: int,
    expiry_months: tuple[int, ...] = (3, 6, 9, 12),
    roll_offset_days: int = 0,
    lookback_days: int = 1,
) -> pd.DataFrame:
    """Rank v.N: the Nth unexpired contract by trailing average volume (no look-ahead).

    ``daily_volume`` is indexed by UTC day with one column per candidate ticker.
    ``lookback_days`` sets how many PRIOR trading days are averaged (1 = prior day only,
    matching the naive rule). A single thin session - a Sunday open, a day before a
    holiday - can otherwise outrank a genuinely more liquid contract by a coin-flip
    margin and cause a one-day round-trip in the front contract; averaging over several
    days damps that noise. Ties and days with no prior volume fall back to calendar
    (expiry) order.
    """
    c = _eligible_contracts(contracts, expiry_months)
    c = c[c["ticker"].isin(daily_volume.columns)].reset_index(drop=True)
    columns = list(range(max_rank + 1))
    if c.empty:
        return pd.DataFrame(None, index=dates, columns=columns, dtype=object)

    tickers = c["ticker"].to_numpy(dtype=object)
    expiry = c["expiry"].dt.normalize().to_numpy("datetime64[D]")
    threshold = dates.to_numpy("datetime64[D]") + np.timedelta64(roll_offset_days, "D")
    alive = expiry[None, :] >= threshold[:, None]  # (dates, contracts)

    # shift(1) first so the average only ever sees days strictly before the ranked day;
    # rolling() then never introduces look-ahead, it just smooths what's already past.
    prior = daily_volume.reindex(columns=tickers).sort_index().shift(1)
    prior = prior.rolling(window=lookback_days, min_periods=1).mean()
    prior = prior.reindex(dates, method="ffill").fillna(0.0).to_numpy()

    out = np.full((len(dates), max_rank + 1), None, dtype=object)
    for i in range(len(dates)):
        candidates = np.flatnonzero(alive[i])
        if candidates.size == 0:
            continue
        # highest prior volume first; ties -> earlier expiry (candidates are expiry-sorted)
        order = candidates[np.argsort(-prior[i, candidates], kind="stable")]
        picked = tickers[order[: max_rank + 1]]
        out[i, : len(picked)] = picked
    return pd.DataFrame(out, index=dates, columns=columns, dtype=object)


def contract_runs(mapping: pd.DataFrame, rank: int) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """``[(ticker, first_day, last_day + 1d)]`` - each UNBROKEN run of days ``rank`` sits
    in one contract. Unlike a per-ticker min..max window this never merges two runs of
    the same raw symbol: CME reuses symbols every decade (``ZNH5`` = March 2015 AND
    March 2025), so a min..max across both would span ten years."""
    col = mapping[rank]
    runs, current, first, prev = [], None, None, None
    for day, ticker in col.items():
        if ticker != current or (prev is not None and day - prev > pd.Timedelta(days=7)):
            if current is not None:
                runs.append((current, first, prev + pd.Timedelta(days=1)))
            current, first = ticker, day
        prev = day
    if current is not None:
        runs.append((current, first, prev + pd.Timedelta(days=1)))
    return [(t, a, b) for t, a, b in runs if t is not None]


def roll_switches(mapping: pd.DataFrame, rank: int = 0) -> list[tuple[pd.Timestamp, str, str]]:
    """``[(day, from_ticker, to_ticker)]`` - every day ``rank`` changes contract."""
    col = mapping[rank].dropna()
    prev = col.shift(1)
    changed = col[(col != prev) & prev.notna()]
    return [(day, prev[day], ticker) for day, ticker in changed.items()]


def around_switch_windows(
    mapping: pd.DataFrame, rank: int = 0, *, pad: pd.Timedelta = pd.Timedelta(days=7),
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """The OTHER contract's window around each switch of ``rank``: the incoming one for
    ``pad`` before the switch day, the outgoing one for ``pad`` after it - so both legs
    of a roll are on hand around it, and only there."""
    out = []
    for day, old, new in roll_switches(mapping, rank):
        out.append((new, day - pad, day))
        out.append((old, day, day + pad))
    return out
