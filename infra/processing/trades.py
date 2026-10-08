"""Tick trades with their aggressor side (pure: no I/O).

One row per fill, as Databento's ``trades`` schema delivers it (``infra.api.databento_client.
fetch_futures_trades``), flattened for the store:

==============  ==================================================================
``timestamp``   matching-engine time (``ts_event``), tz-naive UTC, NANOSECONDS
``ticker``      absolute contract
``sequence``    the exchange packet's sequence number (several fills of one sweep
                share it, a microsecond apart)
``fill``        the fill's position within its (ticker, timestamp, sequence) - with
                the three above, the row's key
``ts_recv``     when Databento received it (point in time: known from here)
``price``       x 10000 fixed point (int32, root CLAUDE.md 6b; ZN's 1/64 ticks are
                rounded by at most 0.00005 points, every tick stays distinct)
``size``        contracts
``side``        the AGGRESSOR: ``B`` buyer-initiated, ``A`` seller-initiated, ``N`` no
                aggressor (e.g. the outright legs CME prints for a spread trade)
``flags``       Databento's record flags (int)
==============  ==================================================================

``signed_bars`` aggregates them into 1-minute (or any) bars: buy / sell / no-aggressor
volume and counts, a volume-weighted price per side, signed volume and the imbalance ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_SCALE = 10_000
TRADE_KEYS = ["timestamp", "ticker", "sequence", "fill"]
TRADE_COLUMNS = TRADE_KEYS + ["ts_recv", "price", "size", "side", "flags"]
SIDES = ("B", "A", "N")


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype="str"),
        "sequence": pd.Series(dtype="int64"), "fill": pd.Series(dtype="int16"),
        "ts_recv": pd.Series(dtype="datetime64[ns]"), "price": pd.Series(dtype="float64"),
        "size": pd.Series(dtype="int32"), "side": pd.Series(dtype="str"), "flags": pd.Series(dtype="int16"),
    })


def clean_trades(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Raw ``trades`` records for ONE contract -> flat rows (decoded price, float)."""
    if raw is None or raw.empty:
        return empty_trades()
    df = raw.reset_index() if "ts_recv" not in raw.columns else raw.copy()
    if "action" in df:
        df = df[df["action"].astype(str) == "T"]
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["ts_event"], utc=True).dt.tz_localize(None).astype("datetime64[ns]"),
        "ticker": ticker,
        "sequence": df["sequence"].astype("int64"),
        "ts_recv": pd.to_datetime(df["ts_recv"], utc=True).dt.tz_localize(None).astype("datetime64[ns]"),
        "price": df["price"].astype("float64"),
        "size": df["size"].astype("int32"),
        "side": df["side"].astype(str),
        "flags": np.asarray([int(f) for f in df["flags"]], dtype="int16"),
    })
    out = out.sort_values(["timestamp", "sequence", "ts_recv"], kind="stable")
    out["fill"] = out.groupby(["timestamp", "sequence"]).cumcount().astype("int16")
    return out[TRADE_COLUMNS].reset_index(drop=True)


def encode_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price"] = np.round(out["price"] * PRICE_SCALE).astype("int32")
    # plain strings, not category: a partition file rewritten with an appended batch would
    # otherwise mix dictionary and string types across files, and the dataset can't merge
    # them (found 2026-10-08 on the first backfill)
    out["side"] = out["side"].astype(str)
    out["ticker"] = out["ticker"].astype(str)
    return out


def decode_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price"] = out["price"].astype("float64") / PRICE_SCALE
    for c in ("ticker", "side"):
        out[c] = out[c].astype(str)
    return out


def signed_bars(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Per (bar START, ticker): ``buy_volume`` / ``sell_volume`` / ``none_volume`` and
    their trade counts, ``vwap_buy`` / ``vwap_sell`` / ``vwap``, ``signed_volume`` (buy -
    sell) and ``imbalance`` = signed / (buy + sell) in [-1, 1] (NaN without aggressor
    volume). A bar is stamped at its START (complete one ``freq`` later, like the ohlcv
    stores). Bars with no trade are absent."""
    cols = ["timestamp", "ticker", "buy_volume", "sell_volume", "none_volume", "n_buy", "n_sell", "n_none",
            "vwap_buy", "vwap_sell", "vwap", "signed_volume", "imbalance"]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    t = trades.assign(bar=trades["timestamp"].dt.floor(freq), notional=trades["price"] * trades["size"])
    g = t.groupby(["bar", "ticker"])
    out = pd.DataFrame(index=g.size().index)
    for side, name in (("B", "buy"), ("A", "sell"), ("N", "none")):
        s = t[t["side"] == side].groupby(["bar", "ticker"])
        out[f"{name}_volume"] = s["size"].sum()
        out[f"n_{name}"] = s.size()
        if name != "none":
            out[f"vwap_{name}"] = s["notional"].sum() / s["size"].sum()
    for c in ("buy_volume", "sell_volume", "none_volume", "n_buy", "n_sell", "n_none"):
        out[c] = out[c].fillna(0).astype("int64")
    out["vwap"] = g["notional"].sum() / g["size"].sum()
    out["signed_volume"] = out["buy_volume"] - out["sell_volume"]
    aggr = out["buy_volume"] + out["sell_volume"]
    out["imbalance"] = (out["signed_volume"] / aggr).where(aggr > 0)
    out = out.reset_index().rename(columns={"bar": "timestamp"})
    return out[cols].sort_values(["ticker", "timestamp"]).reset_index(drop=True)
