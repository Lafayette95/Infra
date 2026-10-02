"""Treasury prices per CUSIP (infra.processing.treasury_prices, infra.pipeline.treasury_prices):
page parsing, bond maths (accrued, street yield, bill BEY), complete-day coverage. No
network: a synthetic page in FedInvest's layout and fake fetchers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.api.fedinvest_client import COLUMNS, parse_page
from infra.pipeline import treasury_prices as ptp
from infra.processing import treasury_prices as tp

PAGE = """<table><tr><th>CUSIP</th><th>SECURITY TYPE</th><th>RATE</th><th>MATURITY DATE</th><th>CALL DATE</th>
<th>BUY</th><th>SELL</th><th>END OF DAY</th></tr>
<tr><td>91282CRF0</td><td>MARKET BASED NOTE</td><td>4.625%</td><td>08/15/2036</td><td></td>
<td>99.500000</td><td>99.480000</td><td>99.540696</td></tr>
<tr><td>912797SA6</td><td>MARKET BASED BILL</td><td>0.000%</td><td>12/31/2026</td><td></td>
<td>0.000000</td><td>98.900000</td><td>98.910000</td></tr>
<tr><td>9128TIPS1</td><td>TIPS</td><td>2.375%</td><td>07/15/2036</td><td></td>
<td>99.000000</td><td>99.000000</td><td>99.000000</td></tr></table>"""

SEC = pd.DataFrame({"cusip": ["91282CRF0", "912797SA6"], "security_type": ["Note", "Bill"],
                    "coupon": [4.625, np.nan], "maturity_date": pd.to_datetime(["2036-08-15", "2026-12-31"]),
                    "dated_date": pd.to_datetime(["2026-08-15", None]),
                    "first_coupon_date": pd.to_datetime(["2027-02-15", None]), "coupons_per_year": [2, 0]})


def test_parse_page_keeps_only_price_rows():
    page = parse_page(PAGE)
    assert list(page.columns) == COLUMNS and len(page) == 3 and page.loc[0, "cusip"] == "91282CRF0"


def test_par_bond_yields_its_coupon_and_accrues_from_the_last_coupon():
    # settles on a coupon date at par -> yield = coupon, no accrued
    acc, y = tp.accrued_and_yield(100.0, 4.0, "2036-02-15", pd.Timestamp("2026-08-15"), 2)
    assert acc == 0.0 and abs(y - 4.0) < 1e-9
    acc, _ = tp.accrued_and_yield(100.0, 4.0, "2036-02-15", pd.Timestamp("2026-11-15"), 2)
    assert abs(acc - 2.0 * 92 / 184) < 1e-12  # 92 of the 184 days Aug 15 -> Feb 15


def test_bill_yield_regimes_meet_at_six_months():
    y = [tp.bill_yield(98.0, m, "2026-12-02") for m in ("2027-06-01", "2027-06-02", "2027-06-03", "2027-06-04")]
    steps = np.diff(y)  # 181->182 (simple), 182->183 (switch to the quadratic), 183->184 (quadratic)
    assert abs(steps[1] - steps[0]) < 1e-3 and abs(steps[1] - steps[2]) < 1e-3  # no jump at the switch
    assert 4.0 < y[1] < 4.2


def test_prices_with_yields_scope_and_columns():
    out = tp.prices_with_yields(parse_page(PAGE), "2026-09-29", SEC)
    assert out["cusip"].tolist() == ["91282CRF0", "912797SA6"]  # TIPS dropped
    note, bill = out.iloc[0], out.iloc[1]
    assert np.isnan(bill["price_buy"]) and bill["accrued"] == 0.0 and 4.0 < bill["yield_eod"] < 5.0
    assert 4.6 < note["yield_eod"] < 4.8 and note["accrued"] > 0


def test_a_day_is_covered_only_when_its_end_of_day_prices_are_posted(tmp_path):
    early = PAGE.replace("99.540696", "0.000000").replace("98.910000", "0.000000")
    kw = dict(root=tmp_path / "p", coverage_file=tmp_path / "c.parquet", now=pd.Timestamp("2026-09-30 02:00"))
    assert ptp.store_prices_day("2026-09-29", parse_page(early), SEC, **kw) == "pending"
    assert ptp.plan_prices_update("2026-09-29", "2026-09-29", coverage_file=kw["coverage_file"],
                                  now=kw["now"]) == [pd.Timestamp("2026-09-29")]
    assert ptp.store_prices_day("2026-09-29", parse_page(PAGE), SEC, **kw) == "stored"
    assert ptp.plan_prices_update("2026-09-29", "2026-09-29", coverage_file=kw["coverage_file"], now=kw["now"]) == []
    assert ptp.read_prices("2026-09-29", "2026-09-30", root=kw["root"])["yield_eod"].notna().all()


def test_empty_page_is_a_holiday_only_once_old_enough(tmp_path):
    empty = pd.DataFrame(columns=COLUMNS)
    kw = dict(root=tmp_path / "p", coverage_file=tmp_path / "c.parquet")
    assert ptp.store_prices_day("2025-12-25", empty, SEC, now=pd.Timestamp("2025-12-26"), **kw) == "pending"
    assert ptp.store_prices_day("2025-12-25", empty, SEC, now=pd.Timestamp("2026-01-05"), **kw) == "holiday"


def test_weekends_are_never_planned(tmp_path):
    days = ptp.plan_prices_update("2026-09-25", "2026-09-29", coverage_file=tmp_path / "c.parquet",
                                  now=pd.Timestamp("2026-10-01"))
    assert [d.day_name() for d in days] == ["Friday", "Monday", "Tuesday"]
