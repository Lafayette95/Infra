"""Persistence of the open-interest flow proxy (positioning CLAUDE.md 6-7): robustness,
generality across roots, which participants, and whether it matters for price.

proxy(t) = (total open interest, all contracts)(t) - (t-1), x sign(same-contract settlement
change on t): + = positions added in the direction prices moved.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/study_oi_programs.py                       # ZN robustness + all roots + CFTC
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

from infra.analytics.positioning.flow import oi_flow_proxy  # noqa: E402
from infra.config import FUTURES_CONTRACTS_FILE, MACRO_ROOTS  # noqa: E402
from infra.pipeline import daily as dl  # noqa: E402
from infra.pipeline.relative_daily import load_relative_daily  # noqa: E402
from infra.pipeline.series_panel import read_panel  # noqa: E402
from infra.relative.symbology import parse_relative  # noqa: E402
from infra.storage import contract_store  # noqa: E402

BONDS = ("ZT", "ZF", "ZN", "TN", "ZB", "UB")
# CFTC TFF contract market codes (futures only report)
TFF_CODES = {"ZT": "042601", "ZF": "044601", "ZN": "043602", "ZB": "020601", "UB": "020604", "TN": "043607",
             "ES": "13874A", "NQ": "209742", "6E": "099741", "6J": "097741", "6B": "096742", "6A": "232741",
             "6C": "090741", "6S": "092741", "6M": "095741"}


def corr_t(x, y, method="pearson"):
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 20:
        return np.nan, np.nan, len(d)
    r = d.iloc[:, 0].corr(d.iloc[:, 1], method=method)
    return r, r * np.sqrt((len(d) - 2) / max(1e-12, 1 - r * r)), len(d)


def fmt(r, t, n):
    return f"{r:+.3f} (t {t:+.1f}, n {n})" if np.isfinite(r) else f"n/a (n {n})"


def root_daily(root: str, start, end) -> pd.DataFrame:
    """Per trading day: total OI over all contracts, same-contract settlement change of the
    front (v.0), the front contract, and the proxy."""
    tickers = contract_store.read_contracts(FUTURES_CONTRACTS_FILE, root)["ticker"].astype(str).unique().tolist()
    rows = dl.read_daily_from_disk(sorted(tickers), pd.Timestamp(start), pd.Timestamp(end))
    oi = rows.groupby("timestamp")["open_interest"].sum(min_count=1).astype(float)
    px = read_panel([f"fut:{root}.v.0"], start, end).iloc[:, 0]
    rel = load_relative_daily([parse_relative(f"{root}.v.0")], pd.Timestamp(start), pd.Timestamp(end),
                              fetch_missing=False).dropna(subset=["settlement_price"])
    front = rel.set_index("timestamp")["contract"].astype(str)
    d = pd.DataFrame({"oi": oi, "price_change": px.diff(), "front": front}).dropna(subset=["oi", "price_change"])
    d["proxy"] = oi_flow_proxy(d)
    d["doi"] = d["oi"].diff()
    d["sign"] = np.sign(d["price_change"])
    return d


def roll_mask(d: pd.DataFrame, days: int = 7) -> pd.Series:
    """True within +-``days`` calendar days of a front-contract change, and in the last 10
    calendar days before it (OI runs off through expiry)."""
    sw = d.index[d["front"] != d["front"].shift(1)][1:]
    m = pd.Series(False, index=d.index)
    for s in sw:
        m |= (d.index >= s - pd.Timedelta(days=days + 10)) & (d.index <= s + pd.Timedelta(days=days))
    return m


def event_mask(index: pd.DatetimeIndex) -> pd.Series:
    from infra.pipeline.release_calendar import read_release_calendar
    ev = read_release_calendar(events=["US_EMPLOYMENT_SITUATION", "US_CPI", "US_FOMC_DECISION"])
    weeks = set(pd.DatetimeIndex(ev["timestamp"]).to_period("W-FRI"))
    return pd.Series([p in weeks for p in index.to_period("W-FRI")], index=index)


def persistence(s: pd.Series, k: int = 1, method: str = "pearson"):
    return corr_t(s, s.shift(-k), method)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2010-07-12")
    p.add_argument("--end", default=pd.Timestamp.now().normalize().strftime("%Y-%m-%d"))
    a = p.parse_args()

    zn = root_daily("ZN", "2014-12-01", a.end)
    print("== 1. ZN robustness (2014-12..)")
    print(f"  proxy lag-1 persistence: {fmt(*persistence(zn['proxy']))}   rank: {fmt(*persistence(zn['proxy'], 1, 'spearman'))}")
    print(f"  parts: OI change alone {fmt(*persistence(zn['doi']))}   price sign alone {fmt(*persistence(zn['sign']))}"
          f"   |OI change| {fmt(*persistence(zn['doi'].abs()))}")
    rm, em = roll_mask(zn), event_mask(zn.index)
    print(f"  excluding roll windows ({rm.mean():.0%} of days): {fmt(*persistence(zn['proxy'].where(~rm)))}")
    print(f"  excluding payroll / CPI / FOMC weeks ({em.mean():.0%}): {fmt(*persistence(zn['proxy'].where(~em)))}")
    print(f"  excluding both: {fmt(*persistence(zn['proxy'].where(~rm & ~em)))}")
    print("  by year: " + "  ".join(f"{y}:{persistence(g)[0]:+.2f}" for y, g in zn['proxy'].groupby(zn.index.year)))
    print(f"  price relevance: proxy(t) vs next-day price change {fmt(*corr_t(zn['proxy'], zn['price_change'].shift(-1)))}"
          f"   vs next 5 days {fmt(*corr_t(zn['proxy'], zn['price_change'].rolling(5).sum().shift(-5)))}")

    print("\n== 2. every root: lag-1 persistence of the proxy (rank), and excluding rolls")
    rows = []
    for r in BONDS + MACRO_ROOTS:
        try:
            d = root_daily(r, a.start, a.end)
        except Exception as e:
            print(f"  {r}: skipped ({type(e).__name__})")
            continue
        rows.append({"root": r, "days": len(d), "from": d.index.min().date(),
                     "lag1": persistence(d["proxy"])[0], "lag1_rank": persistence(d["proxy"], 1, "spearman")[0],
                     "lag1_ex_roll": persistence(d["proxy"].where(~roll_mask(d)))[0],
                     "lag2": persistence(d["proxy"], 2)[0],
                     "next_day_price": corr_t(d["proxy"], d["price_change"].shift(-1))[0]})
    df = pd.DataFrame(rows).set_index("root")
    print(df.round(3).to_string())
    se = 1 / np.sqrt(df["days"].median())
    print(f"  (one root's standard error ~{se:.3f}; mean lag1 across roots {df['lag1'].mean():+.3f}, "
          f"positive in {(df['lag1'] > 0).sum()} of {len(df)})")

    print("\n== 3. who: weekly CFTC net position changes vs the weekly sum of the proxy (Tue-to-Tue)")
    from infra.pipeline.cftc_tff import read_tff
    tff = read_tff(start="2010-01-01", markets=list(TFF_CODES.values()))
    out = []
    for r, code in TFF_CODES.items():
        g = tff[tff["market_code"].astype(str) == code].groupby("timestamp").last().sort_index()
        if g.empty:
            continue
        net = pd.DataFrame({k: g[f"{v}_positions_long{s}"].astype(float) - g[f"{v}_positions_short{s2}"].astype(float)
                            for k, v, s, s2 in (("lev", "lev_money", "", ""), ("am", "asset_mgr", "", ""),
                                                ("dealer", "dealer", "_all", "_all"))})
        chg = net.diff()
        try:
            d = root_daily(r, a.start, a.end)
        except Exception:
            continue
        wk = d["proxy"].groupby(d.index.to_period("W-TUE")).sum()
        wk.index = wk.index.to_timestamp(how="end").normalize()
        chg.index = chg.index.to_period("W-TUE").to_timestamp(how="end").normalize()
        j = chg.join(wk.rename("proxy"), how="inner")
        out.append({"root": r, **{f"{k}_vs_proxy": corr_t(j[k], j["proxy"])[0] for k in ("lev", "am", "dealer")},
                    **{f"{k}_persist": persistence(chg[k])[0] for k in ("lev", "am", "dealer")}, "weeks": len(j)})
    print(pd.DataFrame(out).set_index("root").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
