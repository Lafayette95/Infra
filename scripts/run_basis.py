"""Run a Treasury futures basis model over stored days and print its validation bench
(infra/models/basis/CLAUDE.md). Reads disk only.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/run_basis.py --model M0 --start 2026-09-01 --end 2026-09-30
    $PY scripts/run_basis.py --model M0 --start 2019-01-02 --end 2026-09-30 --out /tmp/m0.parquet
    $PY scripts/run_basis.py --start 2019-01-02 --end 2026-09-30 --scores-from /tmp/m0.parquet --realised /tmp/realised.parquet
    # parallel + checkpointed (infra/models/basis/parallel.py): a month per task, resumable
    $PY scripts/run_basis.py --model M2 --start 2019-01-02 --end 2026-09-30 --every 14 --workers 2 \
        --out-dir /tmp/m2_sample --set level_betas=true --realised /tmp/realised.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.models.basis import validate as v  # noqa: E402


def _parse_set(item: str) -> tuple[str, object]:
    """``field=value`` -> (field, typed value): true/false, int, float, none, else str."""
    k, v = item.split("=", 1)
    low = v.lower()
    if low in ("true", "false"):
        return k, low == "true"
    if low == "none":
        return k, None
    for cast in (int, float):
        try:
            return k, cast(v)
        except ValueError:
            pass
    return k, v


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="M0")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--out", help="save the per-contract predictions (parquet; bonds next to it)")
    p.add_argument("--scores-from", help="score saved predictions (an --out file) instead of running")
    p.add_argument("--realised", help="realised CTDs (parquet): read if it exists, else computed and saved there")
    p.add_argument("--workers", type=int, help="run in parallel (a month per process) - needs --out-dir")
    p.add_argument("--out-dir", help="checkpoint directory: one file pair per month, finished months skipped")
    p.add_argument("--every", type=int, default=1, help="every Nth business day (a sample), with --out-dir")
    p.add_argument("--persist", action="store_true",
                   help="store the run's contracts and bonds in Derived/BasisRuns/<model> (the dashboard reads it)")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="override a spec field, e.g. --set level_betas=true (repeatable)")
    a = p.parse_args()
    t0 = time.time()
    if a.out_dir:
        from infra.models.basis.parallel import resolve_spec, run_parallel
        spec = resolve_spec(a.model, dict(_parse_set(x) for x in a.set))
        res = run_parallel(spec, a.start, a.end, a.out_dir, every=a.every, workers=a.workers or 1)
        print(f"{len(res['contracts'])} contract-days in {time.time() - t0:.0f}s")
        if a.persist:
            from infra.storage import basis_runs
            basis_runs.save(spec.name, res["contracts"], res["bonds"])
            print(f"persisted to Derived/BasisRuns/{spec.name}")
            return 0
    elif a.scores_from:
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
