"""Re-score every event's date rule (infra.reference.events, ``EconEvent.rule``) against
the release days the harvested economic calendar observed - the check that a rule the
release calendar projects forward still holds. Exit 1 if any falls below MIN_RULE_HIT.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/validate_schedule_rules.py [--since 2018]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.pipeline.release_calendar import validate_rules  # noqa: E402
from infra.reference.events import MIN_RULE_HIT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", type=int, default=2018)
    args = parser.parse_args()
    df = validate_rules(since=args.since)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.1%}"))
    bad = df[df["hit"] < MIN_RULE_HIT]
    if len(bad):
        print(f"\nbelow {MIN_RULE_HIT:.0%}: {', '.join(bad['event'])}")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
