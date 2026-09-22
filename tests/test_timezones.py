"""Display-only timezone conversion (infra.dashboard.timezones). No Dash needed."""
from __future__ import annotations

import pandas as pd

from infra.dashboard.timezones import DEFAULT_TIMEZONE, DISPLAY_TIMEZONES, to_display_index


def test_utc_is_a_no_op_same_object():
    idx = pd.date_range("2026-09-15", periods=3, freq="h")
    assert to_display_index(idx, "UTC") is idx
    assert to_display_index(idx, "") is idx


def test_conversion_shifts_wall_clock_and_stays_tz_naive():
    idx = pd.date_range("2026-09-15 12:00", periods=2, freq="h")  # UTC, no DST at play
    out = to_display_index(idx, "America/New_York")
    assert out.tz is None  # still tz-naive: a display value, never re-treated as UTC
    assert list(out) == [pd.Timestamp("2026-09-15 08:00"), pd.Timestamp("2026-09-15 09:00")]


def test_conversion_respects_dst_transitions():
    # US DST starts 2026-03-08: EST (UTC-5) before, EDT (UTC-4) after.
    idx = pd.DatetimeIndex(["2026-03-07 12:00", "2026-03-09 12:00"])
    out = to_display_index(idx, "America/New_York")
    assert list(out) == [pd.Timestamp("2026-03-07 07:00"), pd.Timestamp("2026-03-09 08:00")]


def test_never_mutates_input_index():
    idx = pd.date_range("2026-09-15", periods=3, freq="h")
    before = idx.copy()
    to_display_index(idx, "Europe/London")
    assert idx.equals(before)


def test_every_display_timezone_resolves_and_default_is_utc():
    idx = pd.date_range("2026-09-15", periods=2, freq="h")
    for tz in DISPLAY_TIMEZONES.values():
        to_display_index(idx, tz)  # must not raise for any configured zone
    assert DEFAULT_TIMEZONE == "UTC" and DISPLAY_TIMEZONES["UTC"] == "UTC"
