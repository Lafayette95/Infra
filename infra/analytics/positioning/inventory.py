"""Entry-price inventory: where the open positions in a futures market were put on (pure).

Two books, built day by day from the root's TOTAL open interest (all contracts) and a price:

* positions OPENED BY BUYERS (aggressive buyers whose trades raised open interest: new longs
  initiated by the long side) and positions OPENED BY SELLERS (new shorts initiated);
* a day with open interest UP: the increase is booked at the day's price, in the book of that
  day's aggressor side (net signed volume from ticks; the sign of the price change as a proxy
  where there are no ticks - the open-interest proxy's assumption, ``flow.oi_flow_proxy``);
* a day with open interest DOWN: positions are closed by that day's aggressor side covering the
  OTHER side's book (aggressive buying = shorts covering: the sellers' book shrinks), run off
  PROPORTIONALLY across its entry prices; any excess comes off the other book.

Each book is a set of (entry price, size) lots; outputs per day: size, average entry price,
the share of the book UNDERWATER at the day's price (buyers: entry above price; sellers: below)
and the distance of the price from the average entry in units of daily vol.

What it can't know: WHO holds the positions, and whether an aggressive buyer opened a long or
closed a short beyond what the open-interest change implies (the day's net only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_books(oi: pd.Series, price: pd.Series, side: pd.Series, *, max_lots: int = 500) -> pd.DataFrame:
    """Daily books from ``oi`` (total open interest), ``price`` (the day's VWAP or
    settlement) and ``side`` (+1 buyers aggressive, -1 sellers, 0 neither), all on one daily
    index. Lots older than ``max_lots`` days are merged into their neighbour (memory bound)."""
    d = pd.concat({"oi": oi, "price": price, "side": side}, axis=1).dropna()
    books = {+1: [], -1: []}            # lists of [price, size]
    rows = []
    prev_oi = None
    for t, (o, p, s) in d.iterrows():
        if prev_oi is not None:
            delta = o - prev_oi
            if delta > 0 and s != 0:
                books[int(np.sign(s))].append([p, delta])
            elif delta < 0:
                need = -delta
                # aggressive buyers close shorts (the sellers' book), aggressive sellers close longs
                order = [-1, +1] if s > 0 else [+1, -1]
                for b in order:
                    size = sum(l[1] for l in books[b])
                    if size <= 0 or need <= 0:
                        continue
                    take = min(need, size)
                    keep = 1 - take / size
                    books[b] = [[lp, ls * keep] for lp, ls in books[b] if ls * keep > 1e-9]
                    need -= take
            for b in (+1, -1):
                if len(books[b]) > max_lots:
                    (p0, s0), (p1, s1) = books[b][0], books[b][1]
                    books[b] = [[(p0 * s0 + p1 * s1) / (s0 + s1), s0 + s1]] + books[b][2:]
        prev_oi = o
        row = {"timestamp": t, "price": p, "oi": o}
        for b, name in ((+1, "buyers"), (-1, "sellers")):
            lots = np.array(books[b]) if books[b] else np.zeros((0, 2))
            size = lots[:, 1].sum() if len(lots) else 0.0
            row[f"{name}_size"] = size
            row[f"{name}_entry"] = (lots[:, 0] * lots[:, 1]).sum() / size if size > 0 else np.nan
            under = lots[:, 0] > p if b > 0 else lots[:, 0] < p
            row[f"{name}_underwater"] = lots[under, 1].sum() / size if size > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("timestamp")


def pain_features(books: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    """Per day: ``net_underwater`` = buyers' share underwater - sellers' (+ = the longs hurt
    more), ``buyers_gap`` / ``sellers_gap`` = (price - average entry) / daily price vol (buyers
    profit when > 0, sellers when < 0), ``net_gap`` = buyers_gap + sellers_gap (+ = the price
    sits above both books' entries: shorts hurt, longs fine), and ``balance`` = buyers' share
    of the two books' sizes."""
    v = vol.reindex(books.index)
    out = pd.DataFrame(index=books.index)
    out["net_underwater"] = books["buyers_underwater"] - books["sellers_underwater"]
    out["buyers_gap"] = (books["price"] - books["buyers_entry"]) / v
    out["sellers_gap"] = (books["price"] - books["sellers_entry"]) / v
    out["net_gap"] = out["buyers_gap"] + out["sellers_gap"]
    tot = books["buyers_size"] + books["sellers_size"]
    out["balance"] = books["buyers_size"] / tot.where(tot > 0)
    return out
