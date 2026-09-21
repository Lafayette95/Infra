"""Single entry point for the dashboard/scripts: relative AND absolute tickers.

``load_series`` returns bars with a ``ticker`` (what was asked for) and a ``contract``
(the absolute contract actually used). Only absolute contracts ever touch the API.
"""
from __future__ import annotations

import pandas as pd

from infra.config import FUTURES_CONTRACTS_FILE, FUTURES_ROOTS, MAX_COST_USD
from infra.coverage.intervals import Interval, to_utc_day
from infra.pipeline import futures as fut
from infra.pipeline import relative as rel
from infra.relative.symbology import RelativeSpec, parse_relative
from infra.storage import contract_store


def split_tickers(tickers: list[str]) -> tuple[list[RelativeSpec], list[str]]:
    """(relative specs, absolute tickers) from a mixed list."""
    specs, absolute = [], []
    for ticker in tickers:
        spec = parse_relative(ticker)
        (specs if spec else absolute).append(spec or ticker)
    return specs, absolute


def dataset_for_absolute(ticker: str) -> str:
    """Dataset serving an absolute contract, via the contracts table."""
    root = contract_store.dataset_lookup(FUTURES_CONTRACTS_FILE).get(ticker)
    if root is None:
        raise KeyError(f"{ticker!r} is not in the contracts table; load its root's definitions first.")
    return FUTURES_ROOTS[root].dataset


def load_series(
    tickers: list[str],
    start,
    end,
    *,
    fetch_missing: bool = True,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """Bars for relative (``SR3.c.0``) and/or absolute (``SRZ4``) tickers."""
    specs, absolute = split_tickers(tickers)
    frames = []
    if specs:
        frames.append(rel.load_relative_futures(
            specs, start, end, fetch_missing=fetch_missing, max_cost_usd=max_cost_usd, client=client))
    for ticker in absolute:
        dataset = dataset_for_absolute(ticker) if fetch_missing else None
        df = fut.load_futures([ticker], start, end, dataset=dataset, fetch_missing=fetch_missing,
                              max_cost_usd=max_cost_usd, client=client)
        frames.append(df.assign(contract=df["ticker"].astype(str)))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return fut.read_futures_from_disk([], to_utc_day(start), to_utc_day(start)).assign(
            contract=pd.Series(dtype="str"))
    return pd.concat(frames, ignore_index=True)


def plan_series(tickers: list[str], start, end) -> dict[str, list[Interval]]:
    """What is still missing on disk for ``tickers`` (no API)."""
    specs, absolute = split_tickers(tickers)
    plan = rel.plan_relative_update(specs, start, end) if specs else {}
    for ticker in absolute:
        gaps = fut.plan_futures_update(ticker, start, end)
        if gaps:
            plan[ticker] = gaps
    return plan
