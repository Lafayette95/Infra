"""The raw step's reference-data sources (release calendar, Treasury auctions, auction
tails): they run, store, and a broken publisher only ever WARNS - never fails the step."""
from __future__ import annotations

import pandas as pd

from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.raw import RAW_STEP
from infra.cycle.runner import execute_step
from infra.pipeline import release_calendar as prc
from infra.pipeline import tsy_auctions as pa

D = pd.Timestamp
REC = {"cusip": "A10", "auction_date": "2026-09-10", "announcemt_date": "2026-09-04", "issue_date": "2026-09-15",
       "maturity_date": "2036-08-15", "security_type": "Note", "security_term": "10-Year",
       "original_security_term": "10-Year", "reopening": "No", "inflation_index_security": "No", "floating_rate": "No",
       "closing_time_comp": "01:00 PM", "offering_amt": "39000000000", "high_yield": "4.217",
       "bid_to_cover_ratio": "2.4", "int_rate": "4.25"}


def _ctx(tmp_path):
    return StepContext(D("2026-09-08"), D("2026-09-12"), D("2026-09-12"), CyclePaths.under(tmp_path))


def test_reference_sources_run_and_store(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.cycle.raw_reference.AUCTIONS_FETCH", lambda since: pd.DataFrame([REC]))
    monkeypatch.setattr("infra.cycle.raw_reference.FRED_DATES_FETCH",
                        lambda rid: pd.DatetimeIndex(["2026-09-04", "2026-10-02"]) if rid == 50 else pd.DatetimeIndex([]))
    monkeypatch.setattr("infra.cycle.raw_reference.NAR_FETCH", lambda: (
        "<p>Existing-Home Sales for September 2026 will be released on Tuesday, October 13, 2026 at 10:00 a.m. "
        "Eastern.</p>"))
    out = execute_step(RAW_STEP, _ctx(tmp_path))
    assert out.status == "ok", out
    paths = CyclePaths.under(tmp_path)
    assert len(pa.read_auctions(root=paths.tsy_auctions_dir)) == 1
    cal = prc.read_release_calendar(root=paths.release_calendar_dir)
    assert {"fred_release_dates", "fiscal_data", "nar", "rule"} <= set(cal["source"])
    checks = {c.name: c for c in out.checks}
    assert checks["calendar_refreshed"].passed and checks["auctions_fetch_ok"].passed


def test_a_broken_publisher_only_warns(tmp_path, monkeypatch):
    def down(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr("infra.cycle.raw_reference.AUCTIONS_FETCH", down)
    monkeypatch.setattr("infra.cycle.raw_reference.NAR_FETCH", down)
    out = execute_step(RAW_STEP, _ctx(tmp_path))
    assert out.status == "ok"  # warnings only: the day's vintage is not lost
    checks = {c.name: c for c in out.checks}
    assert not checks["auctions_fetch_ok"].passed and checks["auctions_fetch_ok"].severity.value == "warn"
    assert not checks["calendar_refreshed"].passed and "nar" in checks["calendar_refreshed"].message


def test_out_of_range_auction_results_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.cycle.raw_reference.AUCTIONS_FETCH",
                        lambda since: pd.DataFrame([{**REC, "high_yield": "42.17"}]))
    out = execute_step(RAW_STEP, _ctx(tmp_path))
    sane = {c.name: c for c in out.checks}["auctions_sane"]
    assert not sane.passed and out.status == "ok"
