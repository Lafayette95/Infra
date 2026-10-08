"""Multi-day aggressor-flow "programs" in one futures root (positioning CLAUDE.md 6).

Pre-registered tests (2026-10-08), run on a DISCOVERY and a VALIDATION period:
  1. persistence: daily net flow vs the next 1..5 days'
  2. front-loading: day t's net flow vs day t+1's flow per session, and its price change
  3. program signature: persistence split by the day's evenness (steady vs bursty)
  4. CTA: the CTA model's position change known at day t's close vs day t+1's net flow
  5. the open-interest proxy: vs the real daily flow (here), then its own persistence over
     the whole daily history

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/study_flow_programs.py --root ZN --start 2026-04-07 --end 2026-10-08 --split 2026-08-01
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from infra.analytics.positioning.flow import SESSIONS, daily_flow, oi_flow_proxy, session_of  # noqa: E402
from infra.config import FUTURES_ROOTS  # noqa: E402
from infra.pipeline import daily as dl  # noqa: E402
from infra.pipeline.relative_daily import load_relative_daily  # noqa: E402
from infra.pipeline.series_panel import read_panel  # noqa: E402
from infra.pipeline.trades import read_signed_bars  # noqa: E402
from infra.relative.symbology import parse_relative  # noqa: E402
from infra.trading_calendar import trading_day  # noqa: E402

CTA_ASSET = {"ZT": "US2Y", "ZF": "US5Y", "ZN": "US10Y", "ZB": "US20Y", "UB": "US30Y"}


def corr_t(x, y):
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 8:
        return np.nan, np.nan, len(d)
    r = d.iloc[:, 0].corr(d.iloc[:, 1])
    return r, r * np.sqrt((len(d) - 2) / max(1e-12, 1 - r * r)), len(d)


def fmt(r, t, n):
    return f"{r:+.2f} (t {t:+.1f}, n {n})" if np.isfinite(r) else f"  n/a (n {n})"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="ZN")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--split", required=True, help="first day of the validation period")
    a = p.parse_args()
    start, end, split = pd.Timestamp(a.start), pd.Timestamp(a.end), pd.Timestamp(a.split)
    cfg = FUTURES_ROOTS[a.root]

    rel = load_relative_daily([parse_relative(f"{a.root}.v.0")], start - pd.Timedelta(days=10), end,
                              fetch_missing=False).dropna(subset=["settlement_price"])
    contracts = sorted(rel["contract"].astype(str).unique())
    bars = read_signed_bars([c for c in contracts] + [t for t in _later(a.root, contracts)], start, end)
    d = daily_flow(bars, cfg.dataset)

    # prices: front contract's session-end vwaps (intraday) and same-contract settlement change
    front = rel.set_index("timestamp")["contract"].astype(str)
    fb = bars.assign(day=trading_day(pd.DatetimeIndex(bars["timestamp"]), cfg.dataset))
    fb = fb[fb["ticker"] == fb["day"].map(front)]
    fb["session"] = session_of(pd.DatetimeIndex(fb["timestamp"]))
    tick = 1 / 64
    for name, *_ in SESSIONS:
        s = fb[fb["session"] == name].groupby("day")
        d[f"ret_{name}"] = (s["vwap"].last() - s["vwap"].first()) / tick
    settle = read_panel([f"fut:{a.root}.v.0"], start - pd.Timedelta(days=10), end).iloc[:, 0]
    d["ret_day"] = (settle.diff() / tick).reindex(d.index)

    nxt = d.shift(-1)
    periods = {"discovery": d.index < split, "validation": d.index >= split}
    print(f"{a.root}: {len(d)} trading days ({start.date()}..{end.date()}), split {split.date()}; "
          f"daily net flow sd {d['net'].std():,.0f} contracts, mean |imbalance| {d['imbalance'].abs().mean():.3f}")

    for name, m in periods.items():
        x = d[m]
        n1 = nxt[m]
        print(f"\n== {name} ({m.sum()} days)")
        print("  1. persistence: net(t) vs net(t+k):  " + "  ".join(
            f"k={k} {fmt(*corr_t(x['net'], d['net'].shift(-k)[m]))}" for k in (1, 2, 3, 5)))
        print("  2. net(t) vs day t+1 flow by session: " + "  ".join(
            f"{s} {fmt(*corr_t(x['net'], n1[f'net_{s}']))}" for s, *_ in SESSIONS))
        print("     net(t) vs day t+1 price move (ticks; + = continues): " + "  ".join(
            f"{s} {fmt(*corr_t(x['net'], n1[f'ret_{s}']))}" for s, *_ in SESSIONS)
            + f"  day {fmt(*corr_t(x['net'], n1['ret_day']))}")
        even = x["evenness"] >= x["evenness"].median()
        print(f"  3. persistence by evenness: steady days {fmt(*corr_t(x['net'][even], n1['net'][even]))}   "
              f"bursty days {fmt(*corr_t(x['net'][~even], n1['net'][~even]))}")

    # 4. CTA model: position change known at day t's close vs day t+1's flow
    if a.root in CTA_ASSET:
        from infra.models.cta.config import get_universe
        from infra.models.cta.inputs import universe_prices
        from infra.models.cta.model import walk_forward
        from infra.models.cta.prep import prepare
        uni = get_universe("ubs_us_bonds")
        px = universe_prices(uni, "2014-12-01", end)
        wf = walk_forward("ubs2022_cal", prepare(px, universe=uni), start - pd.Timedelta(days=10), end)
        pos = wf[wf["asset"] == CTA_ASSET[a.root]].set_index("timestamp")["position"]
        dpos = pos.diff().reindex(d.index)
        print("\n== 4. CTA model position change (t, known at the close) vs day t+1 net flow "
              "(+ = CTAs' modelled buying shows as aggressive buying next day)")
        for name, m in periods.items():
            print(f"  {name}: {fmt(*corr_t(dpos[m], nxt['net'][m]))}   same day: {fmt(*corr_t(dpos[m], d['net'][m]))}")

    # 5. open-interest proxy
    oi_rows = dl.read_daily_from_disk(sorted(set(_all_contracts(a.root))), pd.Timestamp("2014-12-01"), end)
    oi = oi_rows.groupby("timestamp")["open_interest"].sum(min_count=1).astype(float)
    full_settle = read_panel([f"fut:{a.root}.v.0"], "2014-12-01", end).iloc[:, 0]
    daily = pd.DataFrame({"oi": oi, "price_change": full_settle.diff()}).dropna()
    proxy = oi_flow_proxy(daily)
    print("\n== 5. OI proxy (OI change x price-change sign) vs real daily net flow: "
          f"{fmt(*corr_t(proxy.reindex(d.index), d['net']))}")
    pr = proxy.dropna()
    print("   its own persistence over the whole daily history "
          f"({pr.index.min().date()}..{pr.index.max().date()}): " + "  ".join(
              f"k={k} {fmt(*corr_t(pr, pr.shift(-k)))}" for k in (1, 2, 5)))
    return 0


def _all_contracts(root: str) -> list[str]:
    from infra.config import FUTURES_CONTRACTS_FILE
    from infra.storage import contract_store
    return contract_store.read_contracts(FUTURES_CONTRACTS_FILE, root)["ticker"].astype(str).tolist()


def _later(root: str, contracts: list[str]) -> list[str]:
    """Contracts after the last front one (the next roll's target trades before it leads)."""
    from infra.config import FUTURES_CONTRACTS_FILE
    from infra.storage import contract_store
    c = contract_store.read_contracts(FUTURES_CONTRACTS_FILE, root)
    last = c[c["ticker"].astype(str).isin(contracts)]["expiry"].max()
    return c[pd.to_datetime(c["expiry"]) > pd.to_datetime(last)].sort_values("expiry")["ticker"].astype(str).head(2).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
