"""Fit our US Treasury curve (spline + Svensson) over a range and store it with the
per-CUSIP z-spread / carry / rolldown (infra/pipeline/treasury_curves.py). Disk only.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/build_treasury_curves.py --start 2008-09-02 --end 2026-10-01
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.pipeline.treasury_curves import build_curves  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--chunk-years", type=int, default=1, help="build and store a year at a time (bounded memory)")
    a = p.parse_args()
    t0 = time.time()
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    lo = start
    while lo <= end:
        hi = min(pd.Timestamp(year=lo.year + a.chunk_years - 1, month=12, day=31), end)
        out = build_curves(lo, hi)
        print(f"{lo.date()}..{hi.date()}: {out['days']} days, {out['bond_rows']} bond rows ({time.time() - t0:.0f}s)", flush=True)
        lo = hi + pd.Timedelta(days=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
