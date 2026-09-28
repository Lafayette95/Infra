"""Backfill a historical WIRP (Fed rate-probability) time series.

No API calls - reads whatever's already cached (populate it first via
scripts/update_futures.py for LIVE, scripts/update_daily.py for CLOSE). Re-runs
infra.dashboard.wirp_selectors.build_schedule for every cached trading day, so a
backfilled day is computed exactly the way the live dashboard would have shown it that
day - see CLAUDE.md section 3's point-in-time cutoff convention.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/backfill_wirp.py --mode close
    $PY scripts/backfill_wirp.py --mode close --start 2026-09-01 --out wirp_backfill.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.dashboard.wirp_selectors import backfill_schedule  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["close", "live"], default="close")
    parser.add_argument("--start", default=None, help="UTC date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="UTC date, exclusive (YYYY-MM-DD)")
    parser.add_argument("--out", default=None, help="CSV path to write; prints a summary if omitted")
    args = parser.parse_args()

    out = backfill_schedule(args.mode, start=args.start, end=args.end)
    if out.empty:
        print("No cached data to backfill.")
        return 1

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Wrote {len(out):,} rows ({out['as_of'].nunique()} days) to {args.out}")
    else:
        modal = out.loc[out.groupby(["as_of", "meeting_date"])["probability"].idxmax()]
        print(modal[["as_of", "meeting_date", "method", "outcome_bps", "probability"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
