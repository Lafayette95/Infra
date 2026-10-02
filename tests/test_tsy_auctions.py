"""Treasury auctions: parsing, nominal-coupon selection (reopenings by ORIGINAL term),
close instants in UTC, point-in-time results, coverage only for held auctions, and the
release-calendar rows. No network."""
from __future__ import annotations

import pandas as pd

from infra.pipeline import release_calendar as prc
from infra.pipeline import tsy_auctions as pa
from infra.processing import tsy_auctions as ta

D = pd.Timestamp
NULL = "null"


def _rec(cusip, auction, announced, term, original, stype="Note", hy=NULL, close="01:00 PM", tips="No", frn="No"):
    return {"cusip": cusip, "auction_date": auction, "announcemt_date": announced, "issue_date": auction,
            "maturity_date": "2036-08-15", "security_type": stype, "security_term": term,
            "original_security_term": original, "reopening": "Yes" if term != original else "No",
            "inflation_index_security": tips, "floating_rate": frn, "closing_time_comp": close,
            "offering_amt": "39000000000", "high_yield": hy, "bid_to_cover_ratio": "2.4" if hy != NULL else NULL,
            "int_rate": "4.25"}


RECORDS = pd.DataFrame([
    _rec("A10", "2026-09-10", "2026-09-04", "10-Year", "10-Year", hy="4.217"),
    _rec("B10", "2026-10-07", "2026-10-01", "9-Year 10-Month", "10-Year"),  # announced, not held
    _rec("TIP", "2026-09-18", "2026-09-11", "10-Year", "10-Year", hy="1.80", tips="Yes"),
    _rec("BIL", "2026-09-08", "2026-09-03", "13-Week", "13-Week", stype="Bill", hy="3.9", close="11:30 AM"),
])


def test_parse_select_and_close_instant():
    df = ta.parse(RECORDS)
    nc = ta.nominal_coupons(df)
    assert set(nc["cusip"]) == {"A10", "B10"}  # the reopening counts as a 10y; TIPS and bills don't
    assert nc.set_index("cusip").loc["A10", "timestamp"] == D("2026-09-10 17:00")  # 13:00 New York (EDT)
    assert list(ta.held(nc)) == [True, False]


def test_results_are_point_in_time():
    df = ta.nominal_coupons(ta.parse(RECORDS))
    before = ta.as_of(df, D("2026-09-10 16:59")).set_index("cusip")
    after = ta.as_of(df, D("2026-09-10 17:00")).set_index("cusip")
    assert pd.isna(before.loc["A10", "high_yield"]) and after.loc["A10", "high_yield"] == 4.217
    assert "B10" not in after.index  # announced only on Oct 1


def test_coverage_stops_at_the_first_unheld_auction_and_feeds_the_calendar(tmp_path):
    calls = []

    def fetch(since):
        calls.append(since)
        return RECORDS

    pa.update_auctions(root=tmp_path / "a", coverage_file=tmp_path / "c.parquet", fetch=fetch,
                       calendar_root=tmp_path / "cal", observed=D("2026-10-01"))
    assert pa.plan_auctions_update(coverage_file=tmp_path / "c.parquet") == D("2026-10-07")  # B10 not held yet
    cal = prc.read_release_calendar(root=tmp_path / "cal")
    assert set(cal["event"]) == {"US_TSY_AUCTION_10Y"}
    b10 = cal[cal["timestamp"] == D("2026-10-07 17:00")]
    assert b10["known_from"].iloc[0] == D("2026-10-01")  # known from its announcement
    assert len(pa.read_auctions(root=tmp_path / "a")) == 2


SCHEDULE_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<AuctionCalendar><AuctionCalendarName>Aug2026 Refunding Auction Calendar</AuctionCalendarName>
<StartDate>2026-08-05</StartDate><EndDate>2027-01-30</EndDate>
<AuctionCalendarDate><SecurityTermWeekYear>10-Year</SecurityTermWeekYear><SecurityType>NOTE</SecurityType>
<ReOpeningIndicator>Y</ReOpeningIndicator><TIPS>N</TIPS><FloatingRate>N</FloatingRate>
<AnnouncementDate>2026-11-04</AnnouncementDate><AuctionDate>2026-11-10</AuctionDate>
<SettlementDate>2026-11-16</SettlementDate></AuctionCalendarDate>
<AuctionCalendarDate><SecurityTermWeekYear>10-Year</SecurityTermWeekYear><SecurityType>NOTE</SecurityType>
<ReOpeningIndicator>N</ReOpeningIndicator><TIPS>Y</TIPS><FloatingRate>N</FloatingRate>
<AnnouncementDate>2026-11-12</AnnouncementDate><AuctionDate>2026-11-19</AuctionDate>
<SettlementDate>2026-11-30</SettlementDate></AuctionCalendarDate>
<AuctionCalendarDate><SecurityTermWeekYear>13-Week</SecurityTermWeekYear><SecurityType>BILL</SecurityType>
<TIPS>N</TIPS><FloatingRate>N</FloatingRate><AnnouncementDate>2026-11-05</AnnouncementDate>
<AuctionDate>2026-11-09</AuctionDate><SettlementDate>2026-11-12</SettlementDate></AuctionCalendarDate>
</AuctionCalendar>"""


def test_tentative_schedule_rows():
    rows = prc.treasury_schedule(observed=D("2026-10-02"), fetch=lambda: SCHEDULE_XML)
    assert list(rows["event"]) == ["US_TSY_AUCTION_10Y"]  # the TIPS and the bill are not nominal coupons
    assert rows["timestamp"].iloc[0] == D("2026-11-10 18:00")  # 13:00 New York (EST)
    assert rows["known_from"].iloc[0] == D("2026-08-05")  # known since the refunding, not since today
    assert rows["source"].iloc[0] == "treasury_schedule"
