"""JGB auctions (infra.processing.jgb_auctions, infra.pipeline.jgb_auctions): the calendar
parsers (English, Japanese, alteration notices), the event mapping, the announcement rule and
the point-in-time plan."""
from __future__ import annotations

import pandas as pd

from infra.pipeline import jgb_auctions as pj
from infra.processing import jgb_auctions as ja

D = pd.Timestamp

EN_MONTH = """<table><tr><th>Auction Date</th><th>Issue</th></tr>
<tr><td>Oct. 6, 2026</td><td class="sente">10-year<span>(384)</span></td></tr>
<tr><td>Oct. 8, 2026</td><td>Treasury Discount Bills (6-month)<span>(1411)</span></td></tr>
<tr><td>Oct. 22, 2026</td><td>Liquidity Enhancement Auction(remaining maturities of 5-11 years)</td></tr>
</table>"""

JA_MONTH = """<table><tr><th>入札予定日</th><th>入札対象国債等</th></tr>
<tr><td>12月3日（木）</td><td>10年利付国債</td></tr>
<tr><td>12月4日（金）</td><td>国庫短期証券（3ヶ月）</td></tr>
<tr><td>12月15日（火）</td><td>流動性供給入札（残存期間5年超11年以下）</td></tr>
<tr><td>12月18日（金）</td><td>10年物価連動国債</td></tr>
</table>"""

ALTERATION = """<p>December 7, 2023 Ministry of Finance</p>
<span>Before the alteration (Announced on Nov. 28, 2023)</span><span>After the alteration</span>
<table><tr><th>Auction Date</th><th>Issue</th></tr><tr><td>Feb.29</td><td>2 - Year</td></tr></table>
<table><tr><th>Auction Date</th><th>Issue</th></tr><tr><td>Feb.27</td><td>2 - Year</td></tr></table>"""


def test_month_pages_both_languages():
    en = ja.parse_month(EN_MONTH, "2610e")
    assert list(en["auction_date"]) == [D("2026-10-06"), D("2026-10-08"), D("2026-10-22")]
    assert list(en["kind"]) == ["JGB", "TB", "LIQ"] and list(en["issue_number"].astype(float)[:2]) == [384, 1411]
    jp = ja.parse_month(JA_MONTH, "2612")
    assert list(jp["auction_date"].dt.day) == [3, 4, 15, 18]
    assert list(zip(jp["kind"], jp["tenor_months"].fillna(0))) == [("JGB", 120), ("TB", 3), ("LIQ", 0), ("ILB", 120)]


def test_alteration_notice_keeps_the_leap_day():
    notice, announced, before, after = ja.parse_alteration(ALTERATION, "2402ae")
    assert (notice, announced) == (D("2023-12-07"), D("2023-11-28"))
    assert list(before["auction_date"]) == [D("2024-02-29")]      # no 1900 (not a leap year) on the way
    assert list(after["auction_date"]) == [D("2024-02-27")]


def test_events_and_announcement_rule():
    assert ja.event_id("JGB", 120) == "JP_AUCTION_10Y" and ja.event_id("JGB", 72) == "JP_AUCTION_OTHER"
    assert ja.event_id("TB", 3) == "JP_AUCTION_TBILL" and ja.event_id("LIQ", None) == "JP_AUCTION_LIQ"
    # announced on the 24th-28th of the month three months before; known from that month's end
    assert ja.announced_by("2612") == D("2026-09-30") and ja.announced_by("2402") == D("2023-11-30")


def _put(root, code, stamp, page):
    p = root / "calendar" / f"{code}__{stamp}.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page, "utf-8")


def test_plan_point_in_time(tmp_path):
    _put(tmp_path, "2402", "20240201T000000Z", ALTERATION.replace("Feb.27", "Feb.29"))   # the month page
    _put(tmp_path, "2402ae", "20231207T000000Z", ALTERATION)
    # before the notice: the original Feb 29; after it: Feb 27; before the announcement: nothing
    assert list(pj.plan_state("2023-12-01", root=tmp_path)["auction_date"]) == [D("2024-02-29")]
    assert list(pj.plan_state("2023-12-08", root=tmp_path)["auction_date"]) == [D("2024-02-27")]
    assert pj.plan_state("2023-11-27", root=tmp_path).empty
    rows = pj.release_rows(observed=D("2024-03-01"), root=tmp_path, auctions_root=tmp_path / "none")
    from infra.processing import release_calendar as rc
    # day-level events sit at 00:00 Tokyo (15:00 UTC the day before)
    seen = lambda day: [(t + pd.Timedelta(hours=9)).date() for t in rc.as_of(rows, day)["timestamp"]]  # noqa: E731
    assert seen("2023-12-01") == [D("2024-02-29").date()]
    assert seen("2024-03-05") == [D("2024-02-27").date()]          # the moved date never comes back
