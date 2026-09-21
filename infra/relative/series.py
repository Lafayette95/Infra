"""Turn stored absolute bars into a relative series using a roll mapping."""
from __future__ import annotations

import pandas as pd


def daily_volume(bars: pd.DataFrame) -> pd.DataFrame:
    """UTC-day x ticker volume pivot from absolute bars (input to ``volume_mapping``)."""
    flat = bars.assign(ticker=bars["ticker"].astype(str), day=bars["timestamp"].dt.normalize())
    return flat.groupby(["day", "ticker"])["volume"].sum().unstack("ticker")


def apply_mapping(bars: pd.DataFrame, mapping: pd.DataFrame, rank: int, label: str) -> pd.DataFrame:
    """Keep, for each bar, only rows of the contract that is ``rank`` on that bar's day.

    Returns the input columns with ``ticker`` set to the relative ``label`` and a new
    ``contract`` column holding the absolute ticker actually used. Prices are unadjusted
    across rolls. The roll day is the UTC calendar day of the bar.
    """
    out_cols = list(bars.columns) + ["contract"]
    if bars.empty or rank not in mapping.columns:
        return pd.DataFrame({c: pd.Series(dtype=bars[c].dtype) if c in bars else pd.Series(dtype="str")
                             for c in out_cols})
    chosen = mapping[rank].reindex(bars["timestamp"].dt.normalize()).to_numpy()
    keep = bars["ticker"].astype(str).to_numpy() == chosen
    out = bars.loc[keep].copy()
    out["contract"] = out["ticker"].astype(str)
    out["ticker"] = pd.Categorical([label] * len(out))
    return out.reset_index(drop=True)
