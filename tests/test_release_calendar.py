"""Release calendar: release instants in UTC, point-in-time knowledge (known_from /
last_seen), a moved release, and the FRED fetch -> store -> read loop. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.pipeline import release_calendar as prc
from infra.processing import release_calendar as rc
from infra.reference.events import EVENTS

D = pd.Timestamp
NFP = EVENTS["US_EMPLOYMENT_SITUATION"]


def test_release_instants_are_utc_across_dst():
    rows = rc.schedule_rows(NFP, pd.DatetimeIndex(["2026-07-02", "2026-12-04"]), source="t", observed=D("2026-10-01"))
    assert list(rows["timestamp"]) == [D("2026-07-02 12:30"), D("2026-12-04 13:30")]  # 08:30 New York, EDT / EST


def test_history_is_known_from_its_own_day_the_future_from_when_seen():
    rows = rc.schedule_rows(NFP, pd.DatetimeIndex(["2026-09-04", "2026-11-06"]), source="t", observed=D("2026-10-01"))
    assert list(rows["known_from"]) == [D("2026-09-04"), D("2026-10-01")]


def test_a_moved_release_point_in_time():
    """Scheduled for Oct 3 (seen Sep 1..Sep 30); a shutdown moves it to Oct 24 (seen from Oct 1)."""
    first = rc.schedule_rows(NFP, pd.DatetimeIndex(["2025-10-03"]), source="t", observed=D("2025-09-01"))
    stored = first
    for day in pd.date_range("2025-09-02", "2025-09-30"):
        stored = rc.merge_observation(stored, rc.schedule_rows(NFP, pd.DatetimeIndex(["2025-10-03"]), source="t",
                                                               observed=day))
    stored = rc.merge_observation(stored, rc.schedule_rows(NFP, pd.DatetimeIndex(["2025-10-24"]), source="t",
                                                           observed=D("2025-10-01")))
    on = lambda day: [t.date() for t in rc.as_of(stored, day)["timestamp"]]  # noqa: E731
    assert on("2025-09-15") == [D("2025-10-03").date()]
    assert on("2025-10-02") == [D("2025-10-24").date()]  # Oct 3 no longer listed: not expected any more
    assert rc.as_of(stored, "2025-09-15")["known_from"].iloc[0] == D("2025-09-01")  # earliest sighting kept


def test_fetch_store_read(tmp_path):
    dates = {50: pd.DatetimeIndex(["2026-09-04", "2026-10-02", "2026-11-06"])}
    fetch = lambda rid: dates.get(rid, pd.DatetimeIndex([]))  # noqa: E731
    prc.refresh_release_calendar(root=tmp_path, observed=D("2026-10-01"), fetch=fetch, with_calendar=False,
                                 with_rules=False, with_agencies=False)
    prc.refresh_release_calendar(root=tmp_path, observed=D("2026-10-02"), fetch=fetch, with_calendar=False,
                                 with_rules=False, with_agencies=False)
    cal = prc.read_release_calendar(root=tmp_path, events=["US_EMPLOYMENT_SITUATION"])
    assert len(cal) == 3 and (cal["last_seen"] == D("2026-10-02")).all()
    # as of Sep 5 only Sep 4 is known: Oct/Nov were first seen on Oct 1 (no look-ahead)
    assert len(prc.read_release_calendar("2026-09-05", root=tmp_path)) == 1


def test_calendar_rows_carry_their_own_time_and_stage():
    from infra.processing import econ_calendar as ec

    rows = ec.parse_page(
        "<table><tr><th>Time (ET)</th><th>Report</th><th>Period</th><th>Actual</th><th>Median Forecast</th>"
        "<th>Previous</th></tr><tr><td>MONDAY, SEPT. 23</td></tr>"
        "<tr><td>9:45 am</td><td>S&amp;P flash U.S. manufacturing PMI</td><td>Sept.</td><td>47.0</td><td>48.5</td>"
        "<td>47.9</td></tr><tr><td>TUESDAY, OCT. 1</td></tr>"
        "<tr><td>10:00 am</td><td>ISM manufacturing</td><td>Sept.</td><td></td><td>47.5</td><td>47.2</td></tr></table>",
        D("2024-09-24 18:00"))
    pats = {"US_SPGLOBAL_MANUFACTURING_PMI": [r"\bpmi\b(?!.*services)"], "US_ISM_MANUFACTURING": [r"^ism manufacturing$"]}
    out = rc.from_econ_calendar(rows, pats, EVENTS).set_index("event")
    flash = out.loc["US_SPGLOBAL_MANUFACTURING_PMI"]
    assert flash["timestamp"] == D("2024-09-23 13:45") and flash["stage"] == "flash" and flash["time_source"] == "source"
    assert flash["known_from"] == D("2024-09-23")  # seen after its release: known from that day, no earlier
    ism = out.loc["US_ISM_MANUFACTURING"]
    assert ism["timestamp"] == D("2024-10-01 14:00") and ism["known_from"] == D("2024-09-24")  # next week's table


def test_nar_next_release():
    page = ("<p>Existing-Home Sales for September 2026 will be released on Tuesday, October 13, 2026 at "
            "10:00 a.m. Eastern.</p>")
    rows = prc.nar_schedule(observed=D("2026-10-01"), fetch=lambda: page)
    assert list(rows["timestamp"]) == [D("2026-10-13 14:00")]  # 10:00 New York (EDT)
    assert rows["known_from"].iloc[0] == D("2026-10-01") and rows["source"].iloc[0] == "nar"
    assert prc.nar_schedule(observed=D("2026-10-01"), fetch=lambda: "<p>nothing</p>").empty


def test_a_schedule_never_confirmed_is_not_a_release():
    """The 2025 shutdown: CPI was listed for Oct 15 (pre-release capture, no actual) and
    actually came out Oct 24 - the Oct 15 row is 'marketwatch_unconfirmed', not a release."""
    from infra.processing import econ_calendar as ec

    rows = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-10-15", "2025-10-24"]), "report": ["CPI year over year"] * 2,
        "period": ["Sept.", "Sept."], "time": ["8:30 am"] * 2, "actual": [np.nan, 3.0], "forecast": [3.1, 3.1],
        "previous": [2.9, 2.9], "actual_raw": ["", "3.0%"], "forecast_raw": ["3.1%"] * 2, "previous_raw": ["2.9%"] * 2,
        "capture": pd.to_datetime(["2025-10-09", "2025-10-30"]), "page": ["p"] * 2})
    out = rc.from_econ_calendar(ec.encode_rows(rows), {"US_CPI": [r"^cpi year over year$"]}, EVENTS)
    got = dict(zip(out["timestamp"].dt.date.astype(str), out["source"]))
    assert got == {"2025-10-15": "marketwatch_unconfirmed", "2025-10-24": "marketwatch"}
