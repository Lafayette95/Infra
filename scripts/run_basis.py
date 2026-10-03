"""Run a Treasury futures basis model over stored days and print its validation bench
(infra/models/basis/CLAUDE.md). Reads disk only.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/run_basis.py --model M0 --start 2026-09-01 --end 2026-09-30
    $PY scripts/run_basis.py --model M0 --start 2019-01-02 --end 2026-09-30 --out /tmp/m0.parquet
    $PY scripts/run_basis.py --start 2019-01-02 --end 2026-09-30 --scores-from /tmp/m0.parquet --realised /tmp/realised.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.models.basis import validate as v  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="M0")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--out", help="save the per-contract predictions (parquet; bonds next to it)")
    p.add_argument("--scores-from", help="score saved predictions (an --out file) instead of running")
    p.add_argument("--realised", help="realised CTDs (parquet): read if it exists, else computed and saved there")
    a = p.parse_args()
    t0 = time.time()
    if a.scores_from:
        res = {"contracts": pd.read_parquet(a.scores_from),
               "bonds": pd.read_parquet(a.scores_from.replace(".parquet", "_bonds.parquet"))}
    else:
        res = v.run(a.model, a.start, a.end, log_every=50)
        print(f"{len(res['contracts'])} contract-days in {time.time() - t0:.0f}s")
        if a.out:
            res["contracts"].to_parquet(a.out)
            res["bonds"].to_parquet(a.out.replace(".parquet", "_bonds.parquet"))
    c = res["contracts"]
    pd.set_option("display.width", 200)
    print("\n== 1. option value (32nds): observed = fair - market\n", v.option_value_bench(c).round(2).to_string())
    if a.realised and Path(a.realised).exists():
        realised = pd.read_parquet(a.realised)
    else:
        realised = v.realised_ctd(c)
        if a.realised:
            realised.to_parquet(a.realised)
    print("\n== 2. CTD calibration vs realised CTD at the last trading day\n",
          v.ctd_calibration(res["bonds"], realised, c).round(3).to_string())
    print("\n== 3. tracking (daily changes)\n", v.tracking_bench(c).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
