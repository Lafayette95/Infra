"""Missed-run watchdog (infra.cycle.watchdog): slot arithmetic, vintage ground truth,
alert-once. No Prefect, no notifications (both stubbed)."""
from __future__ import annotations

import pandas as pd

from infra.cycle import vintage, watchdog
from infra.cycle.paths import CyclePaths

D = pd.Timestamp
CRON = "0 10 * * 2-6"  # the real schedule: Tue-Sat 10:00 UTC


def test_last_due_slot_respects_grace_and_skips_sunday_monday():
    assert watchdog.last_due_slot(D("2026-09-29 11:00"), CRON) == D("2026-09-26 10:00")  # Tue, run not due yet
    assert watchdog.last_due_slot(D("2026-09-29 12:30"), CRON) == D("2026-09-29 10:00")  # Tue, due
    assert watchdog.last_due_slot(D("2026-09-28 15:00"), CRON) == D("2026-09-26 10:00")  # Mon -> Sat's run


def test_check_uses_the_vintage_on_disk(tmp_path):
    paths = CyclePaths.under(tmp_path / "db")
    now = D("2026-09-29 13:00")
    assert watchdog.check(now, CRON, paths) == (False, D("2026-09-29 10:00"))
    vintage.snapshot(D("2026-09-29"), paths)
    assert watchdog.check(now, CRON, paths)[0] is True


def test_alerts_once_per_missed_slot(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(watchdog, "notify", lambda title, text: sent.append(text))
    monkeypatch.setattr(watchdog, "diagnose", lambda slot, **k: "no run was created for it")
    paths, state = CyclePaths.under(tmp_path / "db"), tmp_path / "state.json"
    now = D("2026-09-29 13:00")
    assert "Tue 2026-09-29 10:00" in watchdog.run(now, CRON, paths, state)
    assert watchdog.run(now + pd.Timedelta(hours=1), CRON, paths, state) is None  # already alerted
    assert len(sent) == 1
    assert watchdog.run(D("2026-09-30 13:00"), CRON, paths, state) is not None  # next missed slot alerts again


def test_no_alert_when_the_day_succeeded(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "notify", lambda *a: (_ for _ in ()).throw(AssertionError("no alert")))
    paths = CyclePaths.under(tmp_path / "db")
    vintage.snapshot(D("2026-09-29"), paths)
    assert watchdog.run(D("2026-09-29 13:00"), CRON, paths, tmp_path / "s.json") is None


def test_a_local_time_schedule_is_due_at_the_right_utc_instant_either_side_of_dst():
    from infra.cycle.watchdog import last_due_slot
    cron, tz = "0 6 * * 2-6", "America/New_York"
    # EDT (UTC-4): Wednesday 2026-09-30 06:00 local = 10:00 UTC
    assert last_due_slot(pd.Timestamp("2026-09-30 13:00"), cron, tz=tz) == pd.Timestamp("2026-09-30 10:00")
    # EST (UTC-5), after 2026-11-01: Wednesday 2026-11-04 06:00 local = 11:00 UTC
    assert last_due_slot(pd.Timestamp("2026-11-04 12:30"), cron, tz=tz) == pd.Timestamp("2026-11-03 11:00")
    assert last_due_slot(pd.Timestamp("2026-11-04 13:00"), cron, tz=tz) == pd.Timestamp("2026-11-04 11:00")
