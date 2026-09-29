"""Long-running process that runs the daily cycle on its schedule via Prefect
(Tue-Sat 10:00 UTC, for the prior trading day - infra.cycle.flows.SCHEDULE_CRON). Needs a Prefect API to register
with: start one with `prefect server start` (UI at http://127.0.0.1:4200), then:

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    PREFECT_API_URL=http://127.0.0.1:4200/api $PY scripts/serve_daily_cycle.py

Both processes must stay up for the schedule to fire (e.g. under launchd).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect.schedules import Cron  # noqa: E402

from infra.cycle.flows import SCHEDULE_CRON, SCHEDULE_TZ, daily_cycle_flow  # noqa: E402

if __name__ == "__main__":
    daily_cycle_flow.serve(name="daily-cycle-scheduled", schedule=Cron(SCHEDULE_CRON, timezone=SCHEDULE_TZ))
