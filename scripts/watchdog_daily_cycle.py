"""One tick of the daily cycle's missed-run watchdog (infra.cycle.watchdog) - launchd runs
it hourly (deploy/launchd/com.infra.daily-cycle-watchdog.plist). Alerts via macOS
Notification Center, once per missed slot.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/watchdog_daily_cycle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.cycle.flows import SCHEDULE_CRON  # noqa: E402
from infra.cycle.paths import CyclePaths  # noqa: E402
from infra.cycle.watchdog import check, run  # noqa: E402

STATE = Path.home() / "Library" / "Logs" / "infra" / "watchdog_state.json"

if __name__ == "__main__":
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    alert = run(now, SCHEDULE_CRON, CyclePaths.default(), STATE)
    ok, slot = check(now, SCHEDULE_CRON, CyclePaths.default())
    print(f"{now:%Y-%m-%d %H:%M} UTC - last due slot {slot:%a %Y-%m-%d %H:%M} UTC: "
          f"{'OK' if ok else 'MISSING'}" + (f" - ALERTED: {alert}" if alert else ""))
