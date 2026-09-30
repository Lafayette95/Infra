"""Missed-run watchdog for the scheduled daily cycle (CLAUDE.md section 12).

Ground truth is the VINTAGE on disk, not Prefect: a vintage is written only after a fully
successful cycle, so "no vintage for the last due slot" covers every way a day can be lost
- the scheduler never creating the run (2026-09-29: Prefect's scheduler crashed silently
on an incompatible SQLAlchemy), the Mac asleep, the Prefect server down, a run that started
but failed. Runs from its own launchd job (hourly), independent of Prefect, so it still
works when Prefect is what's broken. Prefect's API is only consulted, best-effort, to say
WHY a day is missing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
from cronsim import CronSim

from infra.cycle import vintage
from infra.cycle.paths import CyclePaths

GRACE = pd.Timedelta(hours=2)  # a run normally takes minutes; API stalls can add ~15-20
DEPLOYMENT = "daily-cycle-scheduled"


def last_due_slot(now: pd.Timestamp, cron: str, grace: pd.Timedelta = GRACE, tz: str = "UTC") -> pd.Timestamp:
    """The latest scheduled slot (returned as naive UTC) whose run should have FINISHED by
    ``now`` (naive UTC): the last fire time at or before ``now - grace`` of ``cron`` read
    in timezone ``tz`` - DST-aware, like the scheduler itself."""
    cutoff = (pd.Timestamp(now) - grace).to_pydatetime().replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz))
    start = cutoff - pd.Timedelta(days=8).to_pytimedelta()
    last = None
    for fire in CronSim(cron, start):
        if fire > cutoff:
            break
        last = fire
    if last is None:
        raise ValueError(f"cron {cron!r} has no fire time in the 8 days before {cutoff}")
    return pd.Timestamp(last).tz_convert("UTC").tz_localize(None)


def check(now: pd.Timestamp, cron: str, paths: CyclePaths, tz: str = "UTC") -> tuple[bool, pd.Timestamp]:
    """``(ok, slot)``: ok when a vintage exists for the last due slot's day (or later -
    a manual catch-up run later that day counts)."""
    slot = last_due_slot(now, cron, tz=tz)
    ok = any(day >= slot.normalize() for day in vintage.list_vintages(paths))
    return ok, slot


def diagnose(slot: pd.Timestamp, api_url: str = "http://127.0.0.1:4200/api") -> str:
    """Best-effort reason a slot has no vintage, from Prefect's API - never raises."""
    import requests
    try:
        body = {"flow_runs": {"expected_start_time": {"after_": slot.isoformat() + "Z"}},
                "deployments": {"name": {"any_": [DEPLOYMENT]}}, "sort": "EXPECTED_START_TIME_DESC", "limit": 1}
        runs = requests.post(f"{api_url}/flow_runs/filter", json=body, timeout=10).json()
    except Exception:
        return "the Prefect server is not reachable"
    if not runs:
        return "no run was created for it (scheduler or serve agent not working)"
    state = runs[0].get("state_name") or runs[0].get("state_type")
    return f"its run ended {state}" if state else "its run has no state"


def notify(title: str, message: str) -> None:
    """macOS Notification Center (osascript); silently skipped elsewhere."""
    exe = shutil.which("osascript")
    if exe:
        text = message.replace('"', "'")
        subprocess.run([exe, "-e", f'display notification "{text}" with title "{title}"'], check=False)


def run(now: pd.Timestamp, cron: str, paths: CyclePaths, state_file: Path, tz: str = "UTC") -> str | None:
    """One watchdog tick. Returns the alert text if it alerted, else None. Alerts ONCE per
    missed slot (remembered in ``state_file``), not every hour until it's fixed."""
    ok, slot = check(now, cron, paths, tz=tz)
    if ok:
        return None
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get("alerted_slot") == slot.isoformat():
        return None
    text = f"No successful daily cycle for the {slot:%a %Y-%m-%d %H:%M} UTC run - {diagnose(slot)}."
    notify("Infra daily cycle", text)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"alerted_slot": slot.isoformat(),
                                      "at": datetime.now(timezone.utc).isoformat()}))
    return text
