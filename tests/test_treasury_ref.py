"""Treasury reference data (infra.processing.treasury_ref, infra.pipeline.treasury_ref):
one row per CUSIP from its original issue, scope (no TIPS / floaters), term parsing,
point-in-time reads. Synthetic auction rows in the raw store's layout; no network."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from infra.pipeline import treasury_ref as ptr
from infra.processing import treasury_ref as tr


def _auction(cusip, *, type_="Note", term="10-Year", reopening="No", coupon="4.625", announced="2026-08-05",
             auction="2026-08-12", issue="2026-08-17", maturity="2036-08-15", tips="No", frn="No"):
    r = {"cusip": cusip, "security_type": type_, "original_security_term": term, "security_term": term,
         "reopening": reopening, "int_rate": coupon, "announcemt_date": announced, "auction_date": auction,
         "issue_date": issue, "dated_date": "2026-08-15", "maturity_date": maturity,
         "first_int_payment_date": "2027-02-15", "int_payment_frequency": "Semi-Annual",
         "cash_management_bill_cmb": "No", "inflation_index_security": tips, "floating_rate": frn}
    return {"timestamp": pd.Timestamp(auction) + pd.Timedelta(hours=17), "cusip": cusip, "security_type": type_,
            "reopening": reopening, "inflation_index_security": tips, "floating_rate": frn,
            "raw_json": json.dumps(r)}


def test_term_months():
    assert tr.term_months("10-Year") == 120 and tr.term_months("9-Year 10-Month") == 118
    assert tr.term_months("26-Week") == 6.0 and np.isnan(tr.term_months("??"))


def test_one_row_per_cusip_from_the_original_issue():
    df = pd.DataFrame([_auction("A"), _auction("A", reopening="Yes", term="9-Year 11-Month", auction="2026-09-09",
                                                 announced="2026-09-03")])
    s = tr.securities(df)
    assert len(s) == 1
    row = s.iloc[0]
    assert row["original_term"] == "10-Year" and row["term_months"] == 120 and row["coupon"] == 4.625
    assert row["timestamp"] == pd.Timestamp("2026-08-05") and row["coupons_per_year"] == 2


def test_tips_and_floaters_are_out_of_scope():
    df = pd.DataFrame([_auction("A"), _auction("T", tips="Yes"), _auction("F", frn="Yes")])
    assert tr.securities(df)["cusip"].tolist() == ["A"]


def test_bills_have_no_coupon_fields():
    s = tr.securities(pd.DataFrame([_auction("B", type_="Bill", term="26-Week", coupon="null")])).iloc[0]
    assert np.isnan(s["coupon"]) and pd.isna(s["first_coupon_date"]) and s["coupons_per_year"] == 0


def test_store_and_point_in_time_read(tmp_path, monkeypatch):
    df = pd.DataFrame([_auction("OLD", announced="2026-05-06", auction="2026-05-12"), _auction("NEW")])
    monkeypatch.setattr(ptr, "read_auctions", lambda **kw: df)
    assert ptr.build_securities(root=tmp_path) == 2
    assert ptr.read_securities("2026-08-04", root=tmp_path)["cusip"].tolist() == ["OLD"]
    assert ptr.read_securities("2026-08-05", root=tmp_path)["cusip"].tolist() == ["OLD", "NEW"]
    assert ptr.read_securities(security_types=("Bill",), root=tmp_path).empty
