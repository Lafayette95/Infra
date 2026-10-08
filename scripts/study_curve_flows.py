"""Known curve flows in US Treasuries, from data on disk (positioning CLAUDE.md 8).

  b) AUCTION CYCLE (Lou, Yan & Zhang 2013): the auctioned tenor cheapens against its
     neighbours into the auction and recovers after. Per auction, the P&L of a LONG
     position in the auctioned tenor against its on-the-run neighbours (bp, bmk OTR yield
     P&L: a fly long the belly, or a two-leg curve where the tenor sits at an end):
     before (closes D-5 -> D-1), auction day (D-1 -> D, the 13:00 result inside), after
     (D -> D+5).
  c) MONTH-END INDEX EXTENSION: index trackers buy duration at month-end, richening the long
     end - 5s30s / 2s10s flatten into the last business days and may reverse after;
     refunding months (Feb, May, Aug, Nov) extend most.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/study_curve_flows.py --start 2008-09-02
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

from infra.pipeline.series_panel import read_panel  # noqa: E402
from infra.pipeline.tsy_auctions import read_auctions  # noqa: E402

B = "US_BOND_{}y"
# a LONG position in the auctioned tenor against its neighbours, as a bmk structure id and the
# sign that makes "+" = the auctioned tenor richened (outperformed)
AUCTION_STRUCTURES = {
    2: (f"CURVE__{B.format(2)}__{B.format(3)}", +1.0),    # long the front leg (2y) vs 3y
    3: (f"FLY__{B.format(2)}__{B.format(3)}__{B.format(5)}", +1.0),
    5: (f"FLY__{B.format(3)}__{B.format(5)}__{B.format(7)}", +1.0),
    7: (f"FLY__{B.format(5)}__{B.format(7)}__{B.format(10)}", +1.0),
    10: (f"FLY__{B.format(7)}__{B.format(10)}__{B.format(30)}", +1.0),   # not 7/10/20: OTR 20y only from 2020
    20: (f"FLY__{B.format(10)}__{B.format(20)}__{B.format(30)}", +1.0),   # 2020-06 on
    30: (f"CURVE__{B.format(10)}__{B.format(30)}", -1.0),  # long the 10y front leg: flip for long 30y
}


def t_of(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan, len(x)


def fmt(m, t, n):
    return f"{m:+.2f}bp (t {t:+.1f}, n {n})"


def auction_cycle(start, end):
    a = read_auctions(start=start, end=end)
    a = a[a["tenor_years"].isin(list(AUCTION_STRUCTURES))]
    ids = {k: f"bmk:otr:{s}" for k, (s, _) in AUCTION_STRUCTURES.items()}
    pnl = read_panel(list(ids.values()), start, end)
    print("== b) auction cycle: P&L of LONG the auctioned tenor vs its neighbours (+ = it richened)")
    print(f"   {'tenor':>5} {'before (D-5..D-1)':>26} {'auction day':>24} {'after (D..D+5)':>24}")
    rows = []
    for tenor, (s, sign) in AUCTION_STRUCTURES.items():
        series = pnl[ids[tenor]] * sign
        days = series.index
        pre, day, post = [], [], []
        for d in pd.DatetimeIndex(a.loc[a["tenor_years"] == tenor, "auction_date"]).normalize():
            i = days.searchsorted(d)
            if i >= len(days) or days[i] != d or i < 5 or i + 5 >= len(days):
                continue
            w = series.iloc[i - 4:i + 6]
            if w.isna().any():                          # a complete window or none (missing != 0)
                continue
            pre.append(series.iloc[i - 4:i].sum())      # closes D-5 -> D-1
            day.append(series.iloc[i])                  # D-1 -> D
            post.append(series.iloc[i + 1:i + 6].sum())  # D -> D+5
        pre, day, post = pd.Series(pre), pd.Series(day), pd.Series(post)
        print(f"   {tenor:>4}y {fmt(*t_of(pre)):>26} {fmt(*t_of(day)):>24} {fmt(*t_of(post)):>24}")
        rows.append({"tenor": tenor, "pre": pre.mean(), "day": day.mean(), "post": post.mean(), "n": len(pre)})
    return pd.DataFrame(rows)


def month_end(start, end, last_days: int = 3, first_days: int = 2):
    ids = {"5s30s steepener": f"bmk:otr:CURVE__{B.format(5)}__{B.format(30)}",
           "2s10s steepener": f"bmk:otr:CURVE__{B.format(2)}__{B.format(10)}",
           "10s30s steepener": f"bmk:otr:CURVE__{B.format(10)}__{B.format(30)}",
           "30y long": f"bmk:otr:{B.format(30)}"}
    pnl = read_panel(list(ids.values()), start, end)
    days = pnl.dropna(how="all").index
    s = pd.Series(days, index=days)
    month = days.to_period("M")
    rank_from_end = s.groupby(month).rank(ascending=False, method="first")
    rank_from_start = s.groupby(month).rank(method="first")
    last = rank_from_end <= last_days
    first = rank_from_start <= first_days
    refunding = days.month.isin([2, 5, 8, 11])
    print(f"\n== c) month-end extension: mean daily P&L (bp) on the last {last_days} business days vs the rest, "
          f"and the first {first_days} of the next month")
    print("   (steepener < 0 at month-end = the long end richened; '30y long' > 0 = 30y yield fell)")
    for name, sid in ids.items():
        x = pnl[sid]
        print(f"   {name:16s} last {last_days}: {fmt(*t_of(x[last]))}  other days: {fmt(*t_of(x[~last & ~first]))}  "
              f"first {first_days}: {fmt(*t_of(x[first]))}")
        print(f"   {'':16s} refunding months, last {last_days}: {fmt(*t_of(x[last & refunding]))}   "
              f"other months: {fmt(*t_of(x[last & ~refunding]))}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2008-09-02")
    p.add_argument("--end", default=pd.Timestamp.now().normalize().strftime("%Y-%m-%d"))
    a = p.parse_args()
    auction_cycle(a.start, a.end)
    month_end(a.start, a.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
