"""Entry-price inventory of open positions (positioning CLAUDE.md 12): where the open
positions were put on, by the side that opened them, and whether the pain predicts.

  A) ZN 2026-04..10: books from TICK aggressor flow vs from the price-sign PROXY, same start
  B) every root, proxy books 2010/2014-2026: pain features vs the next 5 / 20 days' move
     (in daily vols), pooled across roots, both halves separately.

Price = back-adjusted continuous settlement (``fut:``): differences across rolls are the
same-contract chain's, so an entry before a roll compares correctly with a price after it.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/study_entry_prices.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from infra.analytics.positioning.flow import daily_flow  # noqa: E402
from infra.analytics.positioning.inventory import build_books, pain_features  # noqa: E402
from infra.config import FUTURES_CONTRACTS_FILE, MACRO_ROOTS  # noqa: E402
from infra.pipeline import daily as dl  # noqa: E402
from infra.pipeline.series_panel import read_panel  # noqa: E402
from infra.pipeline.trades import read_signed_bars  # noqa: E402
from infra.storage import contract_store  # noqa: E402

BONDS = ("ZT", "ZF", "ZN", "TN", "ZB", "UB")
FEATURES = ("net_underwater", "net_gap", "buyers_gap", "sellers_gap", "balance")


def root_inputs(root: str, start, end) -> pd.DataFrame:
    tickers = contract_store.read_contracts(FUTURES_CONTRACTS_FILE, root)["ticker"].astype(str).unique().tolist()
    rows = dl.read_daily_from_disk(sorted(tickers), pd.Timestamp(start), pd.Timestamp(end))
    oi = rows.groupby("timestamp")["open_interest"].sum(min_count=1).astype(float)
    px = read_panel([f"fut:{root}.v.0"], start, end).iloc[:, 0]
    d = pd.DataFrame({"oi": oi, "price": px}).dropna()
    d["chg"] = d["price"].diff()
    d["vol"] = d["chg"].rolling(63, min_periods=40).std()
    return d


def part_a():
    start, end = pd.Timestamp("2026-04-07"), pd.Timestamp("2026-10-08")
    d = root_inputs("ZN", start, end)
    flow = daily_flow(read_signed_bars(["ZNM6", "ZNU6", "ZNZ6", "ZNH7"], start, end))
    tick_side = np.sign(flow["net"]).reindex(d.index)
    proxy_side = np.sign(d["chg"])
    agree = (tick_side == proxy_side)[tick_side.notna() & (proxy_side != 0)].mean()
    bt = build_books(d["oi"], d["price"], tick_side.fillna(0))
    bp = build_books(d["oi"], d["price"], proxy_side.fillna(0))
    ft, fp = pain_features(bt, d["vol"]), pain_features(bp, d["vol"])
    warm = ft.index >= "2026-05-07"
    print("== A) ZN 2026-04..10: tick-side books vs price-sign proxy books (both start empty on 04-07)")
    print(f"   daily side agreement (tick net flow sign == price change sign): {agree:.0%}")
    for f in FEATURES:
        print(f"   {f:15s} corr(tick, proxy) {ft.loc[warm, f].corr(fp.loc[warm, f]):+.2f}   "
              f"means tick {ft.loc[warm, f].mean():+.2f} / proxy {fp.loc[warm, f].mean():+.2f}")


def part_b():
    print("\n== B) proxy books, every root: pain feature vs the next h days' move (in daily vols)")
    print("   (+ = the market went UP afterwards; per root, sampled every h days, after a 1-year warm-up)")
    res = []
    for r in BONDS + MACRO_ROOTS:
        d = root_inputs(r, "2010-07-10", "2026-10-08")
        if len(d) < 600:
            continue
        b = build_books(d["oi"], d["price"], np.sign(d["chg"]).fillna(0))
        f = pain_features(b, d["vol"]).iloc[252:]
        for h in (5, 20):
            fwd = (d["price"].shift(-h) - d["price"]) / (d["vol"] * np.sqrt(h))
            j = f.join(fwd.rename("fwd")).dropna().iloc[::h]
            for half, m in (("first half", j.index < j.index[len(j) // 2]), ("second half", j.index >= j.index[len(j) // 2])):
                for feat in FEATURES:
                    res.append({"root": r, "h": h, "half": half, "feature": feat,
                                "r": j.loc[m, feat].corr(j.loc[m, "fwd"]), "n": int(m.sum())})
    df = pd.DataFrame(res)
    for h in (5, 20):
        x = df[df.h == h]
        tab = x.groupby(["feature", "half"]).agg(mean_r=("r", "mean"), pos=("r", lambda s: (s > 0).mean()),
                                                 n_roots=("r", "size"), n=("n", "median"))
        tab["t_pool"] = tab["mean_r"] * np.sqrt(tab["n"] * tab["n_roots"] / 4)   # roots ~ 4x correlated
        print(f"\n   h = {h} days")
        print(tab.round(3).to_string())
    by = df[(df.h == 20)].groupby(["feature", "root"])["r"].mean().unstack(0).round(2)
    print("\n   per root, h = 20, both halves averaged:\n" + by.to_string())


if __name__ == "__main__":
    part_a()
    part_b()
