"""US Treasury on/off-the-run map (infra.processing.treasury_otr, infra.pipeline.treasury_otr):
ranks, the two switch conventions, maturity, reopenings. Synthetic reference rows."""
from __future__ import annotations

import pandas as pd

from infra.pipeline import treasury_otr as pot
from infra.processing.treasury_otr import otr_map

TENORS = {"10y": ("Note", "10-Year")}


def _sec(cusip, auction, issue, maturity, term="10-Year", type_="Note"):
    return {"timestamp": pd.Timestamp(auction) - pd.Timedelta(days=7), "cusip": cusip, "security_type": type_,
            "original_term": term, "coupon": 4.0, "auction_date": pd.Timestamp(auction),
            "issue_date": pd.Timestamp(issue), "maturity_date": pd.Timestamp(maturity)}


SEC = pd.DataFrame([_sec("A", "2026-02-11", "2026-02-17", "2036-02-15"),
                    _sec("B", "2026-05-12", "2026-05-15", "2036-05-15"),
                    _sec("C", "2026-08-12", "2026-08-17", "2036-08-15"),
                    _sec("X", "2026-01-27", "2026-01-31", "2028-01-31", term="2-Year")])  # other tenor


def _rank(df, day, rank):
    r = df[(df["timestamp"] == pd.Timestamp(day)) & (df["rank"] == rank)]
    return r["cusip"].tolist()


def test_ranks_newest_first_by_issue_date():
    df = otr_map(SEC, pd.bdate_range("2026-08-10", "2026-08-18"), TENORS, 3, conventions=("issue",))
    assert _rank(df, "2026-08-14", 0) == ["B"] and _rank(df, "2026-08-14", 1) == ["A"]
    assert _rank(df, "2026-08-17", 0) == ["C"] and _rank(df, "2026-08-17", 2) == ["A"]
    assert _rank(df, "2026-08-17", 3) == []  # nothing older
    assert "X" not in set(df["cusip"])


def test_auction_convention_switches_at_the_auction():
    df = otr_map(SEC, pd.bdate_range("2026-08-10", "2026-08-18"), TENORS, 1, conventions=("auction",))
    assert _rank(df, "2026-08-12", 0) == ["C"] and _rank(df, "2026-08-11", 0) == ["B"]


def test_a_matured_issue_drops_out():
    old = pd.DataFrame([_sec("OLD", "2016-02-10", "2016-02-16", "2026-02-15"), _sec("A", "2026-02-11", "2026-02-17", "2036-02-15")])
    df = otr_map(old, pd.bdate_range("2026-02-13", "2026-02-18"), TENORS, 1, conventions=("issue",))
    assert _rank(df, "2026-02-13", 0) == ["OLD"]
    assert _rank(df, "2026-02-16", 0) == [] and _rank(df, "2026-02-17", 0) == ["A"]  # matured Sun 15th
    assert _rank(df, "2026-02-17", 1) == []


def test_build_replaces_days_and_reads_back(tmp_path, monkeypatch):
    monkeypatch.setattr(pot, "read_securities", lambda end, root=None: SEC)
    monkeypatch.setattr(pot, "TREASURY_OTR_TENORS", TENORS)
    pot.build_otr("2026-08-10", "2026-08-18", root=tmp_path)
    n1 = len(pot.read_otr("2026-08-10", "2026-08-19", root=tmp_path))
    pot.build_otr("2026-08-10", "2026-08-18", root=tmp_path)  # same days again: replaced, not duplicated
    assert len(pot.read_otr("2026-08-10", "2026-08-19", root=tmp_path)) == n1
    assert pot.otr_cusip("2026-08-17", "10y", root=tmp_path) == "C"
    assert pot.otr_cusip("2026-08-17", "10y", convention="auction", root=tmp_path) == "C"
    assert pot.otr_cusip("2026-08-13", "10y", convention="auction", root=tmp_path) == "C"
    assert pot.otr_cusip("2026-08-13", "10y", root=tmp_path) == "B"
