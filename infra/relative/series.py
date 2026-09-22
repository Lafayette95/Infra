"""Turn stored absolute bars into a relative series using a roll mapping."""
from __future__ import annotations

import pandas as pd

from infra.trading_calendar import trading_day


def daily_volume(bars: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """TRADING-day x ticker volume pivot from absolute bars (input to ``volume_mapping``).

    ``dataset`` selects the exchange's trading-day boundary (infra.trading_calendar,
    CLAUDE.md section 6e) - required, not defaulted, so a caller can never silently
    fall back to the wrong exchange's session hours.
    """
    day = trading_day(pd.DatetimeIndex(bars["timestamp"]), dataset)
    flat = bars.assign(ticker=bars["ticker"].astype(str), day=day)
    return flat.groupby(["day", "ticker"])["volume"].sum().unstack("ticker")


def apply_mapping(bars: pd.DataFrame, mapping: pd.DataFrame, rank: int, label: str, dataset: str) -> pd.DataFrame:
    """Keep, for each bar, only rows of the contract that is ``rank`` on that bar's
    TRADING day (infra.trading_calendar, CLAUDE.md section 6e).

    Returns the input columns with ``ticker`` set to the relative ``label`` and a new
    ``contract`` column holding the absolute ticker actually used. Prices are unadjusted
    across rolls.
    """
    out_cols = list(bars.columns) + ["contract"]
    if bars.empty or rank not in mapping.columns:
        return pd.DataFrame({c: pd.Series(dtype=bars[c].dtype) if c in bars else pd.Series(dtype="str")
                             for c in out_cols})
    day = trading_day(pd.DatetimeIndex(bars["timestamp"]), dataset)
    chosen = mapping[rank].reindex(day).to_numpy()
    keep = bars["ticker"].astype(str).to_numpy() == chosen
    out = bars.loc[keep].copy()
    out["contract"] = out["ticker"].astype(str)
    out["ticker"] = pd.Categorical([label] * len(out))
    return out.reset_index(drop=True)
