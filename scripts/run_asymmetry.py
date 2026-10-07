"""Asymmetric reaction measures (positioning from how the market moves). Disk only.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/run_asymmetry.py --list
    $PY scripts/run_asymmetry.py --build ust_daily ust_intraday     # compute + store (replaces each spec)
    $PY scripts/run_asymmetry.py --show ust_daily                   # latest value per measure x instrument

Methodology and results: infra/analytics/positioning/CLAUDE.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.analytics.positioning.config import ASYMMETRY_SPECS  # noqa: E402
from infra.pipeline.positioning import build, read_asymmetry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--build", nargs="*", default=None, help="specs to compute and store (none = all)")
    parser.add_argument("--show", default=None, help="spec whose latest values to print")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 220)
    if args.list:
        for name, spec in ASYMMETRY_SPECS.items():
            print(f"{name:20s} {spec.frequency:9s} {spec.description}")
    if args.build is not None:
        for name in args.build or list(ASYMMETRY_SPECS):
            df = build(name, args.start, args.end)
            print(f"{name}: {len(df)} rows, {df['timestamp'].min().date()} .. {df['timestamp'].max().date()}")
    if args.show:
        df = read_asymmetry(args.show)
        last = df.sort_values("timestamp").groupby(["family", "measure", "instrument"]).tail(1)
        print(last.pivot_table(index=["family", "measure"], columns="instrument", values="value").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
