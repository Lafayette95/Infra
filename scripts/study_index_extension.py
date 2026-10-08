"""Month-end Treasury index extension from the index's rules (positioning CLAUDE.md 11), and
whether it sizes the month-end rally (section 8: the 30y rallies into the last 3 business
days). Data on disk only: securities table, auctions (amounts), FedInvest END OF DAY prices.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/study_index_extension.py --start 2009-01-01
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

from infra.analytics.positioning.index_extension import extension  # noqa: E402
from infra.pipeline.series_panel import read_panel  # noqa: E402
from infra.pipeline.treasury_prices import read_prices  # noqa: E402
from infra.pipeline.treasury_ref import read_securities  # noqa: E402
from infra.pipeline.tsy_auctions import read_auctions  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2009-01-01")
    p.add_argument("--end", default="2026-10-01")
    a = p.parse_args()
    sec = read_securities()
    auc = read_auctions(start="1979-01-01")
    amounts = pd.DataFrame({"cusip": auc["cusip"].astype(str), "issue_date": pd.to_datetime(auc["issue_date"]),
                            "amount": pd.to_numeric(auc["total_accepted"], errors="coerce")}).dropna()
    prices = read_prices(pd.Timestamp(a.start) - pd.Timedelta(days=40), pd.Timestamp(a.end) + pd.Timedelta(days=1))
    prices["cusip"] = prices["cusip"].astype(str)
    days = pd.DatetimeIndex(sorted(prices["timestamp"].unique()))
    month_ends = days.to_series().groupby(days.to_period("M")).max()
    rows = []
    for prior, me in zip(month_ends.iloc[:-1], month_ends.iloc[1:]):
        if me < pd.Timestamp(a.start):
            continue
        rows.append(extension(sec, amounts, prices[prices["timestamp"] == me], me, prior))
    ext = pd.DataFrame(rows).set_index("month_end")
    pd.set_option("display.width", 200)
    print(ext[["duration_old", "duration_new", "extension", "entering", "leaving", "n_new", "mv_new_bn"]].tail(14).round(3).to_string())
    e = ext["extension"]
    print(f"\nextension, {e.index.min().date()}..{e.index.max().date()} ({len(e)} months): mean {e.mean():.3f}y, "
          f"sd {e.std():.3f}, min {e.min():.3f}, max {e.max():.3f}")
    print("by calendar month (refunding = Feb/May/Aug/Nov): " + "  ".join(
        f"{m}:{v:.3f}" for m, v in e.groupby(e.index.month).mean().round(3).items()))

    # does a bigger extension come with a bigger month-end rally? (last 3 business days, bp)
    ids = {"30y": "bmk:otr:US_BOND_30y", "10y": "bmk:otr:US_BOND_10y", "5s30s steepener": "bmk:otr:CURVE__US_BOND_5y__US_BOND_30y"}
    pnl = read_panel(list(ids.values()), a.start, a.end)
    d = pnl.index
    last3 = pnl.groupby(d.to_period("M")).apply(lambda g: g.tail(3).sum())
    last3.index = last3.index.to_timestamp(how="end").normalize()
    ext.index = ext.index.to_period("M").to_timestamp(how="end").normalize()
    j = last3.join(ext["extension"], how="inner").dropna()
    print(f"\nmonth-end rally (sum of the last 3 business days' long P&L, bp) vs the extension, {len(j)} months:")
    for name, sid in ids.items():
        r = j[sid].corr(j["extension"])
        b = np.polyfit(j["extension"], j[sid], 1)[0]
        print(f"   {name:16s} corr {r:+.2f} (t {r * np.sqrt((len(j) - 2) / (1 - r * r)):+.1f})   "
              f"{b / 10:+.2f}bp per 0.1y of extension")
    out = Path("/private/tmp/claude-501/-Users-cristianzaharia-Repos-Infra/66a613c5-7837-4af6-8a86-14b5c497730d/scratchpad/index_extension.csv")
    ext.to_csv(out)
    print(f"\n(table saved to {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
