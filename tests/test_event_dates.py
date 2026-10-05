"""Derived event dates (infra/processing/event_dates.py, infra.pipeline.release_calendar.
derived_schedules) and the verified central-bank lists (infra.config)."""
from __future__ import annotations

import pandas as pd

from infra.config import BOE_MEETINGS, CENTRAL_BANK_MEETINGS, ECB_MEETINGS, FOMC_MEETINGS
from infra.processing import event_dates as ed
from infra.processing import release_calendar as rc
from infra.processing.schedule_rules import business_days
from infra.reference.events import EVENTS

D = pd.Timestamp
OBS = D("2026-10-05")


def test_central_bank_lists_are_consistent():
    for event_id, meetings in CENTRAL_BANK_MEETINGS.items():
        assert event_id in EVENTS
        days = pd.to_datetime([m.decision for m in meetings])
        assert not days.duplicated().any(), event_id
        assert all(m.published <= m.decision for m in meetings), event_id
        sched = days[[m.scheduled for m in meetings]]
        assert set(sched.year) >= set(range(2018, 2028)), event_id
    assert {"2020-03-18", "2022-06-15"} <= {m.decision for m in ECB_MEETINGS if not m.scheduled}
    assert "2022-09-15" not in {m.decision for m in BOE_MEETINGS}  # postponed to the 22nd


def test_central_bank_instants_are_local_times_converted_once():
    rows = ed.central_bank_rows(EVENTS["EA_ECB_DECISION"], ECB_MEETINGS, OBS).set_index("timestamp")
    # 13:45 CET (UTC+1) before 21 Jul 2022, 14:15 after; CEST in summer
    assert D("2022-06-09 11:45") in rows.index      # 13:45 CEST
    assert D("2022-07-21 12:15") in rows.index      # 14:15 CEST
    assert D("2025-01-30 13:15") in rows.index      # 14:15 CET
    pepp = rows[rows["stage"] == "unscheduled"]
    assert (pepp["time_source"] == "unknown").all() and len(pepp) == 2
    boe = ed.central_bank_rows(EVENTS["GB_BOE_DECISION"], BOE_MEETINGS, OBS).set_index("timestamp")
    assert D("2025-02-06 12:00") in boe.index and D("2025-06-19 11:00") in boe.index  # GMT / BST
    moved = boe.loc[D("2022-09-22 11:00")]
    assert moved["known_from"] == D("2022-09-09")


def test_fomc_rows_are_never_known_early():
    rows = ed.fomc_rows(EVENTS["US_FOMC_DECISION"], FOMC_MEETINGS, OBS)
    assert (rows["known_from"] <= rows["timestamp"]).all()
    unsched = rows[rows["stage"] == "unscheduled"]
    assert set(unsched["timestamp"].dt.normalize()) == {D("2020-03-03"), D("2020-03-15")}
    assert (unsched["known_from"] == unsched["timestamp"].dt.normalize()).all()


def _coupons():
    return pd.DataFrame({
        "announcemt_date": ["2025-07-30", "2025-07-30", "2025-08-21", "2025-09-04"],
        "auction_date": ["2025-08-05", "2025-08-06", "2025-08-26", "2025-09-10"],
        "issue_date": ["2025-08-15", "2025-08-15", "2025-08-31", "2025-09-15"],
        "reopening": ["No", "No", "No", "Yes"],
        "tenor_years": [3.0, 10.0, 2.0, 10.0],
    })


def test_treasury_lifecycle_rows():
    rows = ed.treasury_rows(_coupons(), EVENTS, OBS)
    issue10 = rows[rows["event"] == "US_TSY_ISSUE_10Y"].sort_values("timestamp")
    assert issue10["stage"].tolist() == ["new_issue", "reopening"]
    assert (issue10["known_from"] < issue10["timestamp"]).all()  # known from the announcement
    roll10 = rows[rows["event"] == "US_TSY_OTR_ROLL_10Y"]
    assert roll10["timestamp"].dt.normalize().tolist() == [D("2025-08-15")]  # a reopening is no roll
    ref = rows[rows["event"] == "US_TSY_REFUNDING"]
    assert ref["timestamp"].tolist() == [D("2025-07-30 12:30")]  # 08:30 New York (EDT)
    est = rows[rows["event"] == "US_TSY_BORROWING_ESTIMATES"]
    assert est["timestamp"].tolist() == [D("2025-07-28 19:00")]  # Monday 15:00 EDT


def test_futures_rows_follow_cme_rules_and_flag_a_mismatch():
    contracts = pd.DataFrame({"root": ["ZN", "ZN", "SR3"], "ticker": ["ZNZ1", "ZNZ1x", "SR3Z5"],
                              "expiry": [D("2021-12-21"), D("2021-12-20"), D("2026-03-17")],
                              "activation": [D("2021-03-22"), D("2021-03-22"), D("2019-12-13")]})
    rows, check = ed.futures_rows(contracts, EVENTS, business_days("2000-01-01", "2030-12-31", "market"), OBS)
    # Dec 2021: markets open Fri 31 Dec (New Year's Day on a Saturday): last trade = 7 business days earlier
    assert check.set_index("ticker")["match"].to_dict() == {"ZNZ1": True, "ZNZ1x": False}
    fn = rows[(rows["event"] == "FUT_ZN_FIRST_NOTICE") & (rows["stage"] == "ZNZ1")]
    assert fn["timestamp"].dt.tz_localize(None).iloc[0] == D("2021-11-30 06:00")  # 00:00 Chicago
    assert (rows["known_from"] == D("2021-03-22")).sum() == 8
    assert rows[rows["event"] == "FUT_SR3_LAST_TRADE"]["stage"].tolist() == ["SR3Z5"]


def test_v0_roll_rows():
    rel = pd.DataFrame({"timestamp": pd.bdate_range("2025-02-24", periods=5), "ticker": "ZN.v.0",
                        "contract": ["ZNH5", "ZNH5", "ZNM5", "ZNM5", "ZNM5"]})
    rows = ed.roll_rows(rel, EVENTS, OBS)
    assert rows["stage"].tolist() == ["ZNM5"] and rows["event"].tolist() == ["FUT_ZN_ROLL_V0"]
    assert rows["timestamp"].dt.normalize().tolist() == [D("2025-02-26")]


def test_rows_have_the_calendar_schema():
    rows = ed.treasury_rows(_coupons(), EVENTS, OBS)
    assert list(rows.columns) == rc.COLUMNS
