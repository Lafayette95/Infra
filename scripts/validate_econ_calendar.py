"""Cross-check the harvested economic calendar against FRED (infra.pipeline.econ_calendar.
crosscheck): for every FRED-sourced release with a calendar pattern, each calendar ACTUAL
vs FRED's value for that period as published on the same day. A match validates the
calendar's parsing, period, units and release day at once - the quality guardrail for the
calendar-ONLY releases (ISM, PMIs, ...), which have nothing to be checked against.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/validate_econ_calendar.py                      # every release, all history
    $PY scripts/validate_econ_calendar.py --since 2020-01-01 --show 10

Exit code 1 if any release's match rate (over comparable prints) is below --min-match.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.config import MACRO_RELEASES  # noqa: E402
from infra.pipeline.econ_calendar import crosscheck  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default=None)
    parser.add_argument("--min-match", type=float, default=0.95)
    parser.add_argument("--show", type=int, default=5, help="non-matching prints to list per release")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    worst = 1.0
    for t, r in MACRO_RELEASES.items():
        if not (r.source or "").startswith("fred") or not r.calendar_pattern:
            continue
        df = crosscheck(r, since=args.since)
        counts = df["status"].value_counts()
        comparable = counts.get("match", 0) + counts.get("mismatch", 0) + counts.get("not_on_fred", 0)
        rate = counts.get("match", 0) / comparable if comparable else float("nan")
        worst = min(worst, rate) if comparable else worst
        print(f"{t:16s} {r.series_id:20s} prints {len(df):4d}  comparable {comparable:4d}  match {rate:6.1%}  "
              + "  ".join(f"{k} {v}" for k, v in counts.items() if k != "match"))
        bad = df[df["status"].isin(["mismatch", "not_on_fred"])]
        if args.show and len(bad):
            print(bad.head(args.show).to_string(index=False))
    return 1 if worst < args.min_match else 0


if __name__ == "__main__":
    raise SystemExit(main())
