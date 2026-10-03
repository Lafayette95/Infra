"""CFTC Traders in Financial Futures and NY Fed Primary Dealer Statistics: parsing, the
point-in-time cutoff, Rule 2.1 (nothing re-fetched until the source moves), revisions, and
the raw-step checks. Fakes only - no network."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.config import CFTC_TFF_RELEASE, PRIMARY_DEALER_RELEASE
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.raw import RAW_STEP
from infra.cycle.runner import execute_step
from infra.pipeline import cftc_tff, primary_dealer
from infra.processing.cftc_tff import parse_tff, position_identity_gaps
from infra.processing.primary_dealer import changed_rows, parse_primary_dealer, release_day

D = pd.Timestamp


def _tff_row(day, code="043602", oi=100, lev_long=10):
    return {"id": f"{day}{code}F", "report_date_as_yyyy_mm_dd": f"{day}T00:00:00.000",
            "cftc_contract_market_code": code, "market_and_exchange_names": "UST 10Y NOTE - CBT",
            "commodity_name": "T-NOTES", "open_interest_all": str(oi), "lev_money_positions_long": str(lev_long),
            "tot_rept_positions_long_all": str(oi - 5), "nonrept_positions_long_all": "5",
            "tot_rept_positions_short": str(oi - 7), "nonrept_positions_short_all": "7",
            "pct_of_oi_lev_money_long": "10.0", "yyyy_report_week_ww": "x"}


def test_tff_parse_types_and_release_instant():
    df = parse_tff([_tff_row("2026-09-29")], "futures", CFTC_TFF_RELEASE)
    r = df.iloc[0]
    assert r["market_code"] == "043602"  # a code keeps its leading zero
    assert df["open_interest_all"].dtype == "Int64" and df["pct_of_oi_lev_money_long"].dtype == "float64"
    assert r["known_from"] == D("2026-10-02 19:30")  # Friday 15:30 New York (EDT), as the source's Last-Modified
    assert "id" not in df.columns and "yyyy_report_week_ww" not in df.columns


def test_tff_identity_tolerates_one_contract_of_rounding():
    rows = [_tff_row("2026-09-29"), {**_tff_row("2026-09-29", code="1"), "nonrept_positions_long_all": "6"},
            {**_tff_row("2026-09-29", code="2"), "nonrept_positions_long_all": "8"}]
    gaps = position_identity_gaps(parse_tff(rows, "futures", CFTC_TFF_RELEASE))
    assert list(gaps["market_code"]) == ["2"]


def test_tff_fetches_only_when_the_source_moved_and_reads_point_in_time(tmp_path, monkeypatch):
    calls = []
    modified = {"v": D("2026-09-26 19:30")}
    monkeypatch.setattr(cftc_tff, "LAST_MODIFIED", lambda ds: modified["v"])
    monkeypatch.setattr(cftc_tff, "FETCH", lambda ds, since=None: calls.append(since) or
                        [_tff_row("2026-09-22"), _tff_row("2026-09-29", lev_long=20)])
    kw = dict(reports={"futures": "x"}, root=tmp_path / "s", state_file=tmp_path / "st.parquet")
    assert cftc_tff.update_tff(**kw)["futures"]["status"] == "fetched" and calls == [None]  # first run: everything
    assert cftc_tff.update_tff(**kw)["futures"]["status"] == "unchanged" and len(calls) == 1  # Rule 2.1
    modified["v"] = D("2026-10-03 19:30")
    cftc_tff.update_tff(**kw)
    assert calls[-1] == D("2026-09-29") - pd.Timedelta(weeks=8)  # re-reads recent weeks for revisions
    pit = cftc_tff.read_tff(as_of=D("2026-10-02 19:00"), root=tmp_path / "s")
    assert list(pit["timestamp"]) == [D("2026-09-22")]  # the 09-29 week wasn't released yet


def test_tff_one_failing_report_does_not_stop_the_other(tmp_path, monkeypatch):
    monkeypatch.setattr(cftc_tff, "LAST_MODIFIED", lambda ds: D("2026-10-02"))

    def fetch(ds, since=None):
        if ds == "bad":
            raise RuntimeError("HTTP 502")
        return [_tff_row("2026-09-29")]

    monkeypatch.setattr(cftc_tff, "FETCH", fetch)
    out = cftc_tff.update_tff(reports={"futures": "ok", "combined": "bad"}, root=tmp_path / "s",
                              state_file=tmp_path / "st.parquet")
    assert out["futures"]["status"] == "fetched" and out["combined"]["status"] == "failed"


_PD = '''"As Of Date","Time Series","Value (millions)"
"2026-09-16","PDPOSGST-TOT","454557"
"2026-09-23","PDPOSGST-TOT","470836"
"2026-09-23","PDSECRET","*"
'''


def test_primary_dealer_parse_suppression_and_release():
    df = parse_primary_dealer(_PD, PRIMARY_DEALER_RELEASE).set_index(["timestamp", "series"])
    s = df.loc[(D("2026-09-23"), "PDSECRET")]
    assert s["suppressed"] and pd.isna(s["value"])
    assert df.loc[(D("2026-09-23"), "PDPOSGST-TOT"), "known_from"] == D("2026-10-01 20:15")  # Thu 16:15 EDT
    # an off-Wednesday (MBS settlement-class) date takes the next release at least 8 days on
    assert release_day(pd.Series([D("2026-09-24")]), 8).iloc[0] == D("2026-10-08")


def test_primary_dealer_writes_only_new_or_revised_rows():
    old = parse_primary_dealer(_PD, PRIMARY_DEALER_RELEASE)
    new = parse_primary_dealer(_PD.replace("454557", "454600") + '"2026-09-30","PDPOSGST-TOT","1"\n',
                               PRIMARY_DEALER_RELEASE)
    rows, revised = changed_rows(new, old)
    assert revised == 1 and len(rows) == 2


def test_primary_dealer_is_fetched_only_when_a_week_is_due(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(primary_dealer, "FETCH", lambda: calls.append(1) or _PD)
    kw = dict(root=tmp_path / "s", catalog_file=tmp_path / "c.parquet", state_file=tmp_path / "st.parquet")
    assert primary_dealer.update_primary_dealer(now=D("2026-10-02"), **kw)["status"] == "fetched"
    assert primary_dealer.update_primary_dealer(now=D("2026-10-08 20:00"), **kw)["status"] == "not due"
    assert primary_dealer.update_primary_dealer(now=D("2026-10-08 20:16"), **kw)["status"] == "fetched"
    assert len(calls) == 2


def _checks(ctx):
    outcome = execute_step(RAW_STEP, ctx)
    return {c.name: c for c in outcome.checks if c.name.startswith(("tff", "primary_dealer"))}


def test_raw_step_stores_both_and_its_checks_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(cftc_tff, "LAST_MODIFIED", lambda ds: D("2026-10-02 19:30"))
    monkeypatch.setattr(cftc_tff, "FETCH", lambda ds, since=None: [_tff_row("2026-09-22"), _tff_row("2026-09-29")])
    monkeypatch.setattr(primary_dealer, "FETCH", lambda: _PD)
    now = D("2026-10-02 22:00")
    for src in ("cftc_tff", "primary_dealer"):
        monkeypatch.setitem(RAW_STEP_SOURCES(), src, _with_now(RAW_STEP_SOURCES()[src], now))
    ctx = StepContext(D("2026-09-30"), D("2026-10-01"), D("2026-10-02"), CyclePaths.under(tmp_path),
                      options={"raw_sources": {k: RAW_STEP_SOURCES()[k] for k in ("cftc_tff", "primary_dealer")}})
    checks = _checks(ctx)
    assert set(checks) == {"tff_fetch_ok", "tff_fresh", "tff_sane", "primary_dealer_fetch_ok",
                           "primary_dealer_fresh", "primary_dealer_sane"}
    assert all(c.passed for c in checks.values()), {k: c.message for k, c in checks.items()}


def test_raw_step_flags_a_stale_tff_but_only_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(cftc_tff, "LAST_MODIFIED", lambda ds: D("2026-09-19 19:30"))
    monkeypatch.setattr(cftc_tff, "FETCH", lambda ds, since=None: [_tff_row("2026-09-15")])
    now = D("2026-10-02 22:00")
    src = _with_now(RAW_STEP_SOURCES()["cftc_tff"], now)
    ctx = StepContext(D("2026-09-30"), D("2026-10-01"), D("2026-10-02"), CyclePaths.under(tmp_path),
                      options={"raw_sources": {"cftc_tff": src}})
    c = _checks(ctx)["tff_fresh"]
    assert not c.passed and c.severity.value == "warn"


def RAW_STEP_SOURCES():
    from infra.cycle.raw import RAW_SOURCES
    return RAW_SOURCES


def _with_now(fn, now):
    return lambda start, end, **kw: fn(start, end, now=now, **kw)


@pytest.mark.parametrize("name", ["cftc_tff", "primary_dealer"])
def test_both_sources_registered_in_the_raw_step(name):
    assert name in RAW_STEP_SOURCES()
