"""The event registry (infra/reference/events.py) is internally consistent and agrees with
the nowcast's release table: same series, same FRED ids."""
from __future__ import annotations

import re

from infra.config import FOMC_MEETINGS, MACRO_RELEASES
from infra.reference.events import EVENTS, SERIES, event_of, series_by_bbg


def test_ids_and_links():
    assert all(e.id == k for k, e in EVENTS.items()) and all(s.id == k for k, s in SERIES.items())
    assert all(s.event in EVENTS for s in SERIES.values())
    assert all(e.kind in {"release", "auction", "policy", "treasury", "futures"} for e in EVENTS.values())
    assert all(e.time_local is None or re.fullmatch(r"\d{2}:\d{2}", e.time_local) for e in EVENTS.values())
    from zoneinfo import ZoneInfo
    assert all(ZoneInfo(e.timezone) for e in EVENTS.values())


def test_every_nowcast_release_is_a_registry_series_with_the_same_fred_id():
    for ticker, rel in MACRO_RELEASES.items():
        s = series_by_bbg(ticker)
        assert s is not None, ticker
        if (rel.source or "").startswith("fred"):
            assert s.fred_series_id == rel.series_id, ticker
        assert event_of(s.id).frequency == ("W" if rel.frequency == "W" else rel.frequency), ticker


def test_staged_releases_agree_with_the_table():
    """A table row with a Preliminary/Advance flag sits on an event with more than one stage."""
    for ticker, rel in MACRO_RELEASES.items():
        if rel.advanced or rel.preliminary:
            assert len(event_of(series_by_bbg(ticker).id).stages) > 1, ticker


def test_treasury_coupon_auctions_and_fomc():
    assert {f"US_TSY_AUCTION_{t}Y" for t in (2, 3, 5, 7, 10, 20, 30)} <= set(EVENTS)
    assert EVENTS["US_FOMC_DECISION"].time_local == "14:00" and len(FOMC_MEETINGS) > 0


def test_identifiers_live_only_on_the_registry():
    """Each nowcast release links to THE registry entry (no copy), and every calendar-
    sourced or FRED-sourced series is fully specified there."""
    for ticker, rel in MACRO_RELEASES.items():
        assert rel.series is SERIES[rel.series.id], ticker
    for s in SERIES.values():
        if s.source is not None:
            assert s.store_id and s.derive in {"level", "diff", "pct", "saar", "yoy"} and s.frequency in "WMQ", s.id
        if s.source == "calendar":
            assert s.calendar_pattern and s.store_id.startswith("MW:"), s.id
