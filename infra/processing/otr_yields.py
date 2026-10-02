"""On-the-run yield benchmark (CLAUDE.md 18). Pure functions, no I/O.

The yield of a day's on-the-run issue for each tenor, under the CMT curve's ticker for
that tenor (``US_BOND_10y``), with the bond behind it - a series that is a real, tradable
bond each day (unlike CMT's fitted par point), and switches bond when a new issue is
issued.
"""
from __future__ import annotations

import pandas as pd

OTR_YIELD_COLUMNS = ["timestamp", "ticker", "yield", "cusip", "coupon", "maturity_date", "price_eod"]
OTR_YIELD_KEYS = ["timestamp", "ticker"]
_SCALE = 10_000


def ticker_for(tenor: str, country: str = "US") -> str:
    """``"10y"`` -> ``"US_BOND_10y"`` - the CMT curve's naming (CLAUDE.md 13)."""
    return f"{country}_BOND_{tenor}"


def otr_yields(otr_map: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Rank-0 rows of the map joined to that day's prices. A day whose on-the-run bond has
    no END OF DAY yield gets no row (never a stale or substituted value)."""
    if otr_map.empty or prices.empty:
        return pd.DataFrame(columns=OTR_YIELD_COLUMNS)
    m = otr_map[otr_map["rank"] == 0]
    px = prices[["timestamp", "cusip", "yield_eod", "price_eod"]].assign(cusip=lambda d: d["cusip"].astype(str))
    out = m.assign(cusip=m["cusip"].astype(str)).merge(px, on=["timestamp", "cusip"], how="inner")
    out = out.dropna(subset=["yield_eod"])
    if out.empty:
        return pd.DataFrame(columns=OTR_YIELD_COLUMNS)
    out = out.assign(ticker=out["tenor"].map(ticker_for), **{"yield": out["yield_eod"]})
    return out[OTR_YIELD_COLUMNS].sort_values(OTR_YIELD_KEYS).reset_index(drop=True)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df[OTR_YIELD_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    out["maturity_date"] = pd.to_datetime(out["maturity_date"]).astype("datetime64[ms]")
    for c in ("yield", "coupon", "price_eod"):
        out[c] = (out[c].astype("float64") * _SCALE).round().astype("Int32")
    return out


def decode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("yield", "coupon", "price_eod"):
        out[c] = out[c].astype("float64") / _SCALE
    out["ticker"] = out["ticker"].astype("category")
    return out
