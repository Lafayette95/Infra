"""Turn stored absolute rows into a relative series using a roll mapping."""
from __future__ import annotations

import pandas as pd

from infra.trading_calendar import trading_day


def volume_pivot(daily_rows: pd.DataFrame) -> pd.DataFrame:
    """TRADING-day x ticker volume pivot (input to ``volume_mapping``) from daily
    statistics rows, whose ``timestamp`` already IS the trading day (CLAUDE.md 8) and
    whose ``volume`` is the exchange's cleared volume. Days without a published volume
    are left out, so ``volume_mapping`` carries the last known ones forward."""
    if daily_rows.empty or "volume" not in daily_rows:
        return pd.DataFrame(dtype="float64")
    rows = daily_rows.dropna(subset=["volume"])
    flat = rows.assign(ticker=rows["ticker"].astype(str), volume=rows["volume"].astype("float64"))
    return flat.pivot_table(index="timestamp", columns="ticker", values="volume", aggfunc="last")


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
