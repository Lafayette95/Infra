"""Hedges for futures-adjusted swap closes (CLAUDE.md 16): per bond-futures root, the
contract hedged with each day (its ``.v.0``), the hedge ratio known that day, and the
futures move behind each print. Reads disk only: daily settlements (with volume),
cash-bond par yields, ``bbo-1m`` quotes and the contracts table must already be stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from infra.config import (
    BBO_FUTURES_DIR,
    DAILY_BONDS_DIR,
    DAILY_FUTURES_DIR,
    FUTURES_CONTRACTS_FILE,
    FUTURES_ROOTS,
    SWAP_HEDGE_CMT,
    SWAP_HEDGE_QUOTE_RATIO,
    SWAP_HEDGE_RATIO_DAYS,
    SWAP_HEDGES,
)
from infra.pipeline.bbo import read_bbo_from_disk
from infra.pipeline.bonds import read_bonds_from_disk
from infra.pipeline.daily import read_daily_from_disk
from infra.pipeline.relative import volume_ranked_mapping
from infra.processing.swap_hedge import hedge_ratios, mid_at, same_contract_changes
from infra.relative.symbology import RelativeSpec
from infra.storage import contract_store

_ONE_DAY = pd.Timedelta(days=1)
# history read before the first day, so its ratio has a full window (business days + slack)
_RATIO_HISTORY = pd.Timedelta(days=int(SWAP_HEDGE_RATIO_DAYS * 1.6) + 30)


@dataclass
class HedgeBook:
    """Per root: ``contract`` (UTC day -> v.0 ticker) and ``ratio`` (day -> bp per point,
    known before that day)."""
    contract: dict[str, pd.Series] = field(default_factory=dict)
    ratio: dict[str, pd.Series] = field(default_factory=dict)
    bbo_root: Path = BBO_FUTURES_DIR

    def contract_on(self, root: str, day) -> str | None:
        c = self.contract.get(root)
        t = None if c is None else c.get(pd.Timestamp(day).normalize())
        return None if t is None or pd.isna(t) else str(t)

    def ratio_on(self, root: str, day) -> float:
        r = self.ratio.get(root)
        if r is None or r.empty:
            return np.nan
        r = r[r.index <= pd.Timestamp(day).normalize()]
        return float(r.iloc[-1]) if len(r) else np.nan


def hedge_roots(currencies=None) -> list[str]:
    return sorted({r for ccy in (currencies or SWAP_HEDGES) for r in SWAP_HEDGES.get(ccy, {}).values()})


def hedge_contracts(start, end, *, roots=None, daily_root: Path = DAILY_FUTURES_DIR,
                    contracts_file: Path = FUTURES_CONTRACTS_FILE) -> dict[str, pd.Series]:
    """Per hedge root, the contract hedged with each day of ``[start, end]`` - its
    ``.v.0`` (CLAUDE.md 5), the one the book reads quotes for. Shared by the book and by
    the cycle's quote fetch (``infra.cycle.intraday``), so the two never disagree."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    out = {}
    for root in roots or hedge_roots():
        contracts = contract_store.read_contracts(contracts_file, root)
        if contracts.empty:
            continue
        if root in SWAP_HEDGE_QUOTE_RATIO:
            front = quoted_front(list(contracts["ticker"].astype(str)), start, end)
            if len(front):
                out[root] = front
            continue
        m = volume_ranked_mapping([RelativeSpec(root, "v", 0)], FUTURES_ROOTS[root], contracts, start, end,
                                  fetch_missing=False, daily_root=daily_root)
        out[root] = m[0].dropna()
    return out


def quoted_front(tickers, start, end, *, bbo_root: Path = BBO_FUTURES_DIR) -> pd.Series:
    """UTC day -> the contract with the most two-sided ``bbo-1m`` samples that day (the
    liquid front, for roots with no stored cleared volume). Reads stored quotes only."""
    q = read_bbo_from_disk(list(tickers), pd.Timestamp(start), pd.Timestamp(end), root=bbo_root).dropna(subset=["bid", "ask"])
    if q.empty:
        return pd.Series(dtype=object)
    q = q.assign(day=q["timestamp"].dt.normalize(), ticker=q["ticker"].astype(str))
    n = q.groupby(["day", "ticker"]).size().rename("n").reset_index()
    return n.sort_values(["day", "n"]).groupby("day").tail(1).set_index("day")["ticker"]


def snap_mids(tickers, start, end, local_time: str, timezone: str, *, bbo_root: Path = BBO_FUTURES_DIR) -> pd.DataFrame:
    """Day x ticker: each contract's mid at ``local_time`` that day (the latest sample at or
    before the instant, within 5 minutes)."""
    from infra.trading_calendar import snap_instants
    q = read_bbo_from_disk(list(tickers), pd.Timestamp(start), pd.Timestamp(end), root=bbo_root).dropna(subset=["mid"])
    if q.empty:
        return pd.DataFrame()
    q = q.assign(day=q["timestamp"].dt.normalize(), ticker=q["ticker"].astype(str))
    days = pd.DatetimeIndex(sorted(q["day"].unique()))
    inst = pd.Series(snap_instants(days, local_time, timezone), index=days)
    q = q[(q["timestamp"] <= q["day"].map(inst)) & (q["timestamp"] > q["day"].map(inst) - pd.Timedelta(minutes=5))]
    return q.sort_values("timestamp").groupby(["day", "ticker"])["mid"].last().unstack("ticker")


def build_hedge_book(start, end, *, currencies=None, daily_root: Path = DAILY_FUTURES_DIR,
                     bonds_root: Path = DAILY_BONDS_DIR, bbo_root: Path = BBO_FUTURES_DIR,
                     contracts_file: Path = FUTURES_CONTRACTS_FILE) -> HedgeBook:
    """Everything the adjustment needs over ``[start, end]`` for every hedge root of
    ``currencies`` (default: all in SWAP_HEDGES). No network."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    roots = hedge_roots(currencies)
    book = HedgeBook(bbo_root=bbo_root)
    history = start - _RATIO_HISTORY
    yields = read_bonds_from_disk(sorted({SWAP_HEDGE_CMT[r] for r in roots}), history, end, root=bonds_root)
    yields = yields.pivot(index="timestamp", columns="ticker", values="par_yield") if len(yields) else pd.DataFrame()
    mapped = hedge_contracts(history, end - _ONE_DAY, roots=roots, daily_root=daily_root, contracts_file=contracts_file)
    for root in roots:
        if root not in mapped:
            continue
        contract = mapped[root]
        book.contract[root] = contract
        cmt = SWAP_HEDGE_CMT[root]
        if root in SWAP_HEDGE_QUOTE_RATIO:
            st = snap_mids(sorted(set(contract)), history, end, *SWAP_HEDGE_QUOTE_RATIO[root], bbo_root=bbo_root)
            if st.empty or cmt not in yields:
                continue
        else:
            st = read_daily_from_disk(sorted(set(contract)), history, end, root=daily_root)
            if st.empty or cmt not in yields:
                continue
            st = st.pivot(index="timestamp", columns="ticker", values="settlement_price")
        common = st.index.intersection(yields.index[yields[cmt].notna()])
        st = st.loc[common]
        dy_bp = yields.loc[common, cmt].diff() * 100.0
        book.ratio[root] = hedge_ratios(same_contract_changes(st, contract), dy_bp, SWAP_HEDGE_RATIO_DAYS)
    return book


def futures_moves(trades: pd.DataFrame, instant: pd.Timestamp, currency: str, day, book: HedgeBook) -> pd.Series:
    """bp each print's rate should move to stand at ``instant``: hedge ratio x (hedge mid
    at the snap - mid at the print). NaN where the tenor has no hedge, or no quote/ratio."""
    out = pd.Series(np.nan, index=trades.index)
    hedges = SWAP_HEDGES.get(currency, {})
    day = pd.Timestamp(day).normalize()
    for root in sorted({hedges[t] for t in trades["tenor"].unique() if t in hedges}):
        ticker, ratio = book.contract_on(root, day), book.ratio_on(root, day)
        if ticker is None or np.isnan(ratio):
            continue
        quotes = read_bbo_from_disk([ticker], day, day + _ONE_DAY, root=book.bbo_root)
        if quotes.empty:
            continue
        sel = trades["tenor"].map(hedges).eq(root)
        snap_mid = mid_at(quotes, [instant])[0]
        out[sel] = ratio * (snap_mid - mid_at(quotes, trades.loc[sel, "executed"]))
    return out
