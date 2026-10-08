"""Back-adjusted continuous futures prices from a relative series (pure, no I/O).

A relative series (``ZN.v.0``, infra.pipeline.relative_daily) jumps at every roll, since
each row is whichever contract held that rank that day. A trend or volatility measured on
it would read the roll gap (the calendar spread) as a market move. The fix used here is
the standard ADDITIVE back-adjustment: each day's change is measured on ONE contract, the
one held that day, against that same contract's previous settlement:

    change_t = P_t(c_t) - P_{t-1}(c_t)

and the continuous price is the cumulative sum of those changes, anchored so its LAST
value equals the latest real settlement (so recent levels are real prices, older ones are
shifted by the sum of the roll gaps since). This is the same "settlement change of the
same contract both days" the daily cycle's pnl uses (infra.cycle.bmk).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def same_contract_changes(relative: pd.DataFrame, absolute: pd.DataFrame,
                          price: str = "settlement_price") -> pd.DataFrame:
    """Per relative ticker and day: the day's change measured on the contract held that day.

    ``relative``: rows ``timestamp, ticker, contract, <price>`` (a relative series).
    ``absolute``: rows ``timestamp, ticker, <price>`` of absolute contracts, which must
    include each contract's settlement on the day BEFORE it entered the relative series
    (else that roll day's change is NaN: unknown, never guessed).

    Returns ``timestamp, ticker, contract, price, prior_price, change``. The prior day is
    the relative series' own previous row, so a day with no settlement (holiday, OI-only
    row) is skipped rather than diffed across.
    """
    rel = relative.dropna(subset=[price]).sort_values(["ticker", "timestamp"])
    rel = rel.assign(ticker=rel["ticker"].astype(str), contract=rel["contract"].astype(str))
    rel["prior_day"] = rel.groupby("ticker")["timestamp"].shift(1)
    abs_px = (absolute.dropna(subset=[price])
              .assign(ticker=lambda d: d["ticker"].astype(str))
              .set_index(["timestamp", "ticker"])[price])
    abs_px = abs_px[~abs_px.index.duplicated(keep="last")]
    keys = pd.MultiIndex.from_arrays([rel["prior_day"], rel["contract"]])
    prior = abs_px.reindex(keys).to_numpy()
    out = pd.DataFrame({
        "timestamp": rel["timestamp"].to_numpy(),
        "ticker": rel["ticker"].to_numpy(),
        "contract": rel["contract"].to_numpy(),
        "price": rel[price].to_numpy(dtype="float64"),
        "prior_price": prior.astype("float64"),
    })
    out["change"] = out["price"] - out["prior_price"]
    return out.reset_index(drop=True)


def back_adjusted(changes: pd.DataFrame) -> pd.DataFrame:
    """Wide ``timestamp x ticker`` additive back-adjusted prices from
    ``same_contract_changes`` output. A ticker's first day has no change (its level is the
    anchor's start); a missing change (unknown roll prior) counts as 0 and is reported by
    ``unknown_changes``, so the level stays defined."""
    wide_chg = changes.pivot_table(index="timestamp", columns="ticker", values="change", aggfunc="last")
    wide_px = changes.pivot_table(index="timestamp", columns="ticker", values="price", aggfunc="last")
    out = {}
    for ticker in wide_px.columns:
        px = wide_px[ticker].dropna()
        chg = wide_chg[ticker].reindex(px.index).fillna(0.0)
        chg.iloc[0] = 0.0
        level = chg.cumsum()
        out[ticker] = level - level.iloc[-1] + px.iloc[-1]
    return pd.DataFrame(out).sort_index()


def unknown_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """Rows (after each ticker's first) whose change couldn't be measured: the held
    contract had no settlement on the prior day. Mostly roll days with a gap in the store."""
    first = changes.groupby("ticker")["timestamp"].transform("min")
    return changes.loc[changes["change"].isna() & (changes["timestamp"] > first)].reset_index(drop=True)


def log_returns(changes: pd.DataFrame) -> pd.DataFrame:
    """Wide ``timestamp x ticker`` same-contract LOG returns, log(P_t(c_t) / P_{t-1}(c_t)).

    The right return for anything quoted as a price that compounds (equity index, FX,
    commodities): a log difference of the ADDITIVELY back-adjusted level divides each day's
    move by a shifted level, not the real price - found 2026-10-07, crude oil's
    back-adjusted series went negative early in 2010-2026 (16 years of contango roll gaps),
    and every macro future's volatility came out wrong (NQ 14% vs 20%, crude 64% vs 43%).
    NaN where either price is not positive (crude, 2020-04-20) or the prior is unknown."""
    ok = (changes["price"] > 0) & (changes["prior_price"] > 0)
    ret = np.log(changes["price"].where(ok) / changes["prior_price"].where(ok))
    return (changes.assign(ret=ret)
            .pivot_table(index="timestamp", columns="ticker", values="ret", aggfunc="last", dropna=False)
            .sort_index())

