"""Economic-calendar history: page parsing across layouts, year inference, value parsing,
name patterns, vintages from actual/previous, merge priority, the harvest loop, and the
"calendar" release source. No network: fake list/fetch functions throughout."""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from infra.config import CALENDAR_PAGES, MACRO_RELEASES
from infra.pipeline import econ_calendar as pc
from infra.processing import econ_calendar as ec

D = pd.Timestamp


def _table(header, body):
    tr = lambda cells, tag="td": "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"  # noqa: E731
    return "<table>" + tr(header, "th") + "".join(tr(r) for r in body) + "</table>"


# The three layouts seen 2010-2026 (structure as archived; values as published).
PAGE_2010 = _table(["time (et)", "report", "period", "Actual", "Consensus<br>forecast", "previous"], [
    ["MONDAY, March 29"], ["8:30 am", "Personal incomes", "Feb.", "", "0.1%", "0.1%"],
    ["Wednesday, March 31"], ["9:45 am", "Chicago PMI", "March", "58.8%", "61.7%", "62.6%"],
    ["Thursday, April 1"], ["10 am", "ISM", "March", "59.6%", "57.4%", "56.5%"],
    ["10 am", "Existing-home sales", "Feb.", "5.02 mln", "4.93 mln", "5.05 mln"],
])
PAGE_2024 = _table(["Time (ET)", "Report", "Period", "Actual", "Median Forecast", "Previous"], [
    ["TUESDAY, SEPT. 3", "", "", "", "", ""],
    ["9:45 am", "S&P final U.S. manufacturing PMI", "Aug.", "47.9", "--", "48.0"],
    ["10:00 am", "ISM manufacturing", "Aug.", "47.2%", "47.9%", "46.8%"],
    ["8:30 am", "Initial jobless claims", "Aug. 31", "227,000", "225,000", "232,000"],
    ["TUESDAY, SEPT 10", "", "", "", "", ""],
    ["6:00 am", "NFIB optimism index", "Aug.", "", "", "93.7"],
])
PAGE_2026 = _table(["", "Time (ET)", "Report", "Period", "Actual", "Forecast", "Previous"], [
    ["", "Tuesday, Sep. 1"],
    ["", "9:45 AM", "US Manufacturing PMI", "Aug.", "53.9", "53.5", "53.9"],
    ["", "Add to My CalendarPMI, Mfg"],
    ["", "10:00 AM", "ISM Report On Business Manufacturing PMI", "Aug.", "54.6", "55.3", "55.6"],
    ["", "10:00 AM", "Job Openings & Labor Turnover Survey", "Jul.", "7.3M", "7.3M", "7.4M"],
])


@pytest.mark.parametrize("page, capture, expect", [
    (PAGE_2010, "2010-04-02 20:00", [(D("2010-04-01"), "ISM", "March", 59.6, 57.4, 56.5),
                                     (D("2010-04-01"), "Existing-home sales", "Feb.", 5.02e6, 4.93e6, 5.05e6)]),
    (PAGE_2024, "2024-09-05 16:37", [(D("2024-09-03"), "ISM manufacturing", "Aug.", 47.2, 47.9, 46.8)]),
    (PAGE_2026, "2026-09-02 20:49", [(D("2026-09-01"), "ISM Report On Business Manufacturing PMI", "Aug.",
                                      54.6, 55.3, 55.6)]),
])
def test_parse_page_each_layout(page, capture, expect):
    rows = ec.parse_page(page, D(capture))
    for day, report, period, actual, forecast, previous in expect:
        r = rows[(rows["report"] == report)].iloc[0]
        assert (r["timestamp"], r["period"]) == (day, period)
        assert (r["actual"], r["forecast"], r["previous"]) == pytest.approx((actual, forecast, previous))
    assert not rows["report"].str.startswith("Add to My Calendar").any()


def test_a_stale_page_is_dated_by_its_own_week_not_the_capture():
    # the old URL froze in April 2020; an October 2020 capture still shows that week
    page = _table(["time (et)", "report", "period", "ACTUAL", "MEDIAN forecast", "previous"],
                  [["MONDAY, APRIL 20"], ["8:30 am", "Chicago Fed national activity index", "March", "-4.19", "--", "-0.06"]])
    rows = ec.parse_page(page, D("2020-10-31 00:51"))
    assert rows["timestamp"].iloc[0] == D("2020-04-20")


def test_next_week_table_can_be_after_the_capture():
    # a page captured on New Year's Eve shows next week's Monday, 2026-01-05
    assert ec.infer_day("monday", 1, 5, D("2025-12-31 18:00")) == D("2026-01-05")


def test_gzip_pages_are_decoded():
    assert ec.decode_html(gzip.compress(PAGE_2024.encode())) == PAGE_2024


@pytest.mark.parametrize("raw, value", [
    ("47.2%", 47.2), ("-$78.8B", -7.88e10), ("5.02 mln", 5.02e6), ("4.43 miln", 4.43e6), ("3.86 million", 3.86e6),
    ("7.3M", 7.3e6), ("205K", 2.05e5), ("308,000", 308000.0), ("2.1% (Q4)", 2.1), ("-90B", -9e10),
    ("--", np.nan), ("", np.nan), ("N/A", np.nan), ("-", np.nan),
])
def test_parse_value(raw, value):
    out = ec.parse_value(raw)
    assert (np.isnan(out) and np.isnan(value)) or out == pytest.approx(value)


# Every name observed 2010-2026 for the calendar-sourced releases, and lookalikes that
# must NOT match. Extend this list whenever ``backfill_econ_calendar.py --names`` shows a
# new variant.
NAMES = {
    "NAPMPMI Index": ["ISM", "ISM manufacturing index", "ISM manufacturing", "ISM Manufacturing",
                      "ISM Report On Business Manufacturing PMI", "ISM manufacturing indext"],
    "NAPMNMI Index": ["ISM nonmanufacturing index", "ISM non-manufacturing", "ISM nonmanufacturng", "ISM on-manufacturing",
                      "ISM services", "ISM services index", "ISM Report On Business Services PMI",
                      "ISM nonmanufacturing index (new date)"],
    "MPMIUSMA Index": ["Markit manufacturing PMI", "Markit manufacturing PMI (final)", "Markit manufacturing PMI (flash)",
                       "S&P final U.S. manufacturing PMI", "US Manufacturing PMI", "Markit PMI", "Markit flash PMI",
                       "Market PMI", "M arkit PMI", 'Markit "flash" PMI', "Markit PMI (flash )", "Markit PMI, final",
                       "IHS Markit manufacturing PMI (final)", "IS&P U.S. manufacturing PMI", "Flash Markit PMI",
                       "S&P Global (Markit) manufacturing PMI (flash)", "Markit manufacturing PMI (preliminary)",
                       'S&P "flash" U.S. manufacturing PMI', "Flash manufacturing PMI", "Markit PMI flash reading",
                       "S&P final U.S. manufacturing"],
    "MPMIUSSA Index": ["Markit services PMI", "Markit services PMI (flash)", "S&P final U.S. services PMI",
                       "US Services PMI", "Markit nonmanufacturing PMI", "S&P Global (Markit) serivces PMI (final)",
                       "Markit PMI services, flash", "IHS Markit services PMI (final)", "Markit services PMI (new date)"],
    "CHPMINDX Index": ["Chicago PMI", "Chicago Business Barometer (PMI)", "Chicago manufacturing PMI",
                       "Chicago purchasing managers' index",
                       "Chicago Business Barometer - ISM-Chicago Business Survey - Chicago PMI"],
    "CONCCONF Index": ["Consumer confidence", "Consumer confidence index", "Conference Bd - Consumer Confidence"],
    "SBOITOTL Index": ["NFIB small-business index", "NFIB optimism index", "NFIB Index of Small Business Optimism",
                       "NFIB small-buisness index", "NFIB index"],
    "ETSLTOTL Index": ["Existing-home sales", "Existing home sales", "Existing home sales (annual rate)",
                       "Existing Home Sales", "Existing home sales (SAAR)"],
    "CFNAI Index": ["Chicago Fed national activity index", "Chicago Fed national activity", "Chicago Fed national index",
                    "Chicago national activity index"],
    "NFP TCH Index": ["Nonfarm payrolls", "Employment Report"],
    "INJCJC Index": ["Weekly jobless claims", "Initial jobless claims", "Jobless claims",
                     "Initial jobless claims (regular state program, SA)", "Initial jobless claims [delayed due to shutdown]",
                     "Initial jobless claims/delayed*"],
    "CONSSENT Index": ["Consumer sentiment", "Consumer sentiment index (final)", "UMich consumer sentiment index",
                       "Consumer sentiment index (preliminary)", "U. Michigan Prelim Consumer Survey",
                       "U. Michigan Final Consumer Survey", "U Mich consumer sentiment (final)",
                       "Consumer sentiment (revised)"],
    "GDP CQOQ Index": ["GDP", "*GDP", "2nd estimate GDP", "GDP (first revision)", "GDP (delayed report)",
                       "Gross domestic product (real annual rate)"],
    "EMPRGBCI Index": ["Empire state index", "Empire State Manufacturing Survey"],
    "OUTFGAF Index": ["Philly Fed", "Philadelphia Fed Business Outlook Survey", "Philadelphia Fed’s manufacturing survey"],
    None: ["Pending home sales index", "New home sales", "Empire state index (3-month average)",
           "Chicago national activity index (3-month average)", "Home builder confidence index",
           "Continuing jobless claims", "Initial jobless claims (total, NSA)", "Chicago Fed's Charles Evans speaks",
           "Core PCE price index", "Retail sales ex-autos", "Core PCE index (delayed report)", "PCE Price Idx, Y/Y%",
           "Core CPI year over year", "Durable-goods minus transportation"],
}


def test_every_known_name_maps_to_exactly_its_release():
    cal = {t: r.calendar_pattern for t, r in MACRO_RELEASES.items() if r.calendar_pattern}
    for ticker, names in NAMES.items():
        for name in names:
            hits = [t for t, pat in cal.items() if ec.matches(name, pat)]
            assert hits == ([ticker] if ticker else []), (name, hits)


def test_month_period():
    assert ec.month_period("Aug.", D("2024-09-03")) == D("2024-08-01")
    assert ec.month_period("Dec.", D("2025-01-05")) == D("2024-12-01")
    assert ec.month_period("Sept", D("2024-09-23")) == D("2024-09-01")  # a flash: same month
    assert ec.month_period("Aug. 31", D("2024-09-05")) is None  # a weekly period, not August
    assert ec.month_period("Q1", D("2024-04-25")) is None


def _rows(records):
    df = pd.DataFrame(records, columns=["timestamp", "report", "period", "actual", "previous", "capture"])
    df["time"] = df["forecast"] = ""
    df["forecast"] = np.nan
    df["actual_raw"] = df["forecast_raw"] = df["previous_raw"] = df["page"] = ""
    return ec.encode_rows(df[ec.ROW_COLUMNS])


def test_vintages_from_actuals_and_previous():
    pat = MACRO_RELEASES["CONCCONF Index"].calendar_pattern
    rows = _rows([
        (D("2024-07-30"), "Consumer confidence", "July", 100.3, 100.4, D("2024-07-31")),
        (D("2024-08-27"), "Consumer confidence", "Aug.", 103.3, 101.9, D("2024-08-28")),  # July revised
        (D("2024-09-03"), "ISM manufacturing", "Aug.", 47.2, 46.8, D("2024-09-05")),  # another release
    ])
    got = {(r.realtime_start, r.date): r.value for r in ec.release_vintages(rows, pat).itertuples()}
    assert got == {
        (D("2024-07-30"), D("2024-07-01")): 100.3, (D("2024-07-30"), D("2024-06-01")): 100.4,
        (D("2024-08-27"), D("2024-08-01")): 103.3, (D("2024-08-27"), D("2024-07-01")): 101.9,
    }


def test_flash_final_releases_ignore_previous():
    """A final row's "previous" is the same month's flash (2024-09-03: final 47.9, previous
    48.0 = the August flash; July's final was 49.6) - never a revision of the prior month."""
    pat = MACRO_RELEASES["MPMIUSMA Index"].calendar_pattern
    rows = _rows([
        (D("2024-08-22"), "S&P flash U.S. manufacturing PMI", "Aug.", 48.0, 49.6, D("2024-08-23")),
        (D("2024-09-03"), "S&P final U.S. manufacturing PMI", "Aug.", 47.9, 48.0, D("2024-09-05")),
    ])
    got = {(r.realtime_start, r.date): r.value for r in ec.release_vintages(rows, pat, use_previous=False).itertuples()}
    assert got == {(D("2024-08-22"), D("2024-08-01")): 48.0, (D("2024-09-03"), D("2024-08-01")): 47.9}
    assert MACRO_RELEASES["MPMIUSMA Index"].preliminary  # what switches previous off in the pipeline


def test_merge_prefers_post_release_rows_then_the_latest_capture():
    pre = _rows([(D("2020-04-28"), "Consumer confidence index", "April", np.nan, 120.0, D("2020-10-31"))])
    post = _rows([(D("2020-04-28"), "Consumer confidence index", "April", 86.9, 118.8, D("2020-05-01"))])
    merged = pc.merge_rows(post, pre)  # the stale pre-release page arrives LATER
    assert merged["actual"].iloc[0] == 86.9 and merged["previous"].iloc[0] == 118.8
    later = _rows([(D("2020-04-28"), "Consumer confidence index", "April", 87.1, 118.8, D("2020-05-02"))])
    assert pc.merge_rows(post, later)["actual"].iloc[0] == 87.1


def test_select_captures_one_per_day_best_variant_latest():
    caps = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-05-01 09:00", "2020-05-01 21:00", "2020-05-01 23:00", "2020-05-02 10:00"]),
        "original": ["new", "new", "old", "old"], "priority": [0, 0, 1, 1]})
    out = pc.select_captures(caps)
    assert list(out["timestamp"]) == list(pd.to_datetime(["2020-05-01 21:00", "2020-05-02 10:00"]))


def test_harvest_is_resumable_and_feeds_the_release_store(tmp_path):
    root, cov = tmp_path / "cal", tmp_path / "cov.parquet"
    page = CALENDAR_PAGES["marketwatch"]
    pages = {"marketwatch": page}
    captures = {D("2024-09-05 16:00"): PAGE_2024, D("2024-09-12 16:00"): PAGE_2024}
    listed, fetched = [], []

    def list_fn(prefix, start, end):
        listed.append((prefix, start, end))
        ts = [t for t in captures if start <= t < end] if prefix == page.cdx_prefixes[0] else []
        return pd.DataFrame({"timestamp": pd.to_datetime(ts),
                             "original": ["https://www.marketwatch.com/economy-politics/calendar"] * len(ts),
                             "digest": ["x"] * len(ts)})

    def fetch_fn(ts, original):
        fetched.append(ts)
        return captures[ts].encode()

    stats = pc.harvest("marketwatch", "2024-09-01", "2024-09-20", root=root, coverage_file=cov, list_fn=list_fn,
                       fetch_fn=fetch_fn, pause_s=0, pages=pages)
    assert stats["pages"] == 2 and not stats["failed"]
    pc.harvest("marketwatch", "2024-09-01", "2024-09-20", root=root, coverage_file=cov, list_fn=list_fn,
               fetch_fn=fetch_fn, pause_s=0, pages=pages)
    assert len(fetched) == 2  # Rule 2.1: covered capture days are never re-fetched
    rows = pc.read_calendar_from_disk(root=root)
    assert set(rows["report"]) >= {"ISM manufacturing", "S&P final U.S. manufacturing PMI"}
    # the harvest didn't start at the page's history start, so the source claims nothing ...
    df, covered = pc.calendar_vintages("MW:NAPMPMI", D("2009-01-01"), D("2024-09-20"), root=root, coverage_file=cov)
    assert covered == []
    assert (df["value"] == 47.2).any()  # ... but the vintages are there


def test_calendar_source_covers_up_to_the_harvest_frontier(tmp_path):
    cov = tmp_path / "cov.parquet"
    from infra.storage import coverage_store
    coverage_store.record_covered(cov, "marketwatch", [(D("2009-01-01"), D("2024-09-20"))])
    _, covered = pc.calendar_vintages("MW:NAPMPMI", D("2024-09-01"), D("2024-10-01"), root=tmp_path / "none",
                                      coverage_file=cov)
    assert covered == [(D("2024-09-01"), D("2024-09-20"))]


def test_recent_capture_days_are_reprocessed_until_settled(tmp_path, monkeypatch):
    """The archive indexes captures late: a day younger than SETTLE_DAYS is processed but
    never claimed covered, so the next run looks at it again."""
    root, cov = tmp_path / "cal", tmp_path / "cov.parquet"
    page = CALENDAR_PAGES["marketwatch"]
    monkeypatch.setattr(pc, "settled_end", lambda now=None: D("2024-09-10"))
    calls = []

    def list_fn(prefix, start, end):
        calls.append((start, end))
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ms]"), "original": pd.Series(dtype=str),
                             "digest": pd.Series(dtype=str)})

    for _ in range(2):
        pc.harvest("marketwatch", "2024-09-01", "2024-09-20", root=root, coverage_file=cov, list_fn=list_fn,
                   fetch_fn=None, pause_s=0, pages={"marketwatch": page})
    from infra.storage import coverage_store
    assert coverage_store.read_covered(cov, "marketwatch") == [(D("2024-09-01"), D("2024-09-10"))]
    assert calls[-2:] == [(D("2024-09-10"), D("2024-09-20"))] * 2  # 2nd run: only the unsettled days


# ------------------------------------------------------------------ periods, typos
def test_quarter_and_week_periods():
    assert ec.quarter_period("2Q", D("2025-07-30")) == D("2025-04-01")
    assert ec.quarter_period("Q4", D("2026-01-29")) == D("2025-10-01")
    assert ec.week_period("4/25", D("2020-04-30")) == D("2020-04-25")  # a Saturday already
    assert ec.week_period("Oct. 1", D("2010-10-07")) == D("2010-10-02")  # a Friday label -> its Saturday
    assert ec.week_period("June 13 week", D("2020-06-25")) == D("2020-06-13")
    assert ec.period_of("Aug. 31", D("2024-09-05"), "M") is None


def test_resolution_carries_the_unit():
    assert ec.resolution("2.5 million") == pytest.approx(5e4)
    assert ec.resolution("0.2%") == pytest.approx(0.05)
    assert ec.resolution("368,000") == pytest.approx(0.5)


def _series(values, report="Housing starts", start="2011-01-01"):
    """Consecutive monthly rows: actual_t, previous_t = actual_{t-1}, consensus ~ actual."""
    recs = []
    for i, (actual, raw) in enumerate(values):
        day = pd.Timestamp(start) + pd.DateOffset(months=i + 1) + pd.Timedelta(days=16)
        period = (pd.Timestamp(start) + pd.DateOffset(months=i)).strftime("%b.")
        prev = values[i - 1][0] if i else np.nan
        recs.append((day, report, period, actual, prev, D(day) + pd.Timedelta(days=1), raw))
    df = pd.DataFrame(recs, columns=["timestamp", "report", "period", "actual", "previous", "capture", "actual_raw"])
    df["forecast"] = [v for v, _ in values]
    df["time"] = df["forecast_raw"] = df["previous_raw"] = df["page"] = ""
    return ec.encode_rows(df[ec.ROW_COLUMNS])


def test_a_unit_slip_is_flagged_and_its_period_restated_later():
    vals = [(600e3, "600,000"), (610e3, "610,000"), (6.28e6, "6,280,000"), (630e3, "630,000"), (640e3, "640,000")]
    rows = _series(vals)
    # the slip's own row: its consensus is the true ~628k, and the next row restates 628k
    rows.loc[2, "forecast"] = 628e3
    rows.loc[3, "previous"] = 628e3
    pat = MACRO_RELEASES["NHSPSTOT Index"].calendar_pattern
    sus = ec.suspect_prints(rows, pat)
    assert list(sus["actual"]) == [6.28e6]
    v = ec.release_vintages(rows, pat, scale=1e-3)
    oct_ = v[v["date"] == sus["period"].iloc[0]]
    assert list(oct_["value"]) == [628.0]  # the slip is dropped; the restatement stands, dated later


def test_a_genuine_small_or_huge_print_is_not_a_typo():
    pat = MACRO_RELEASES["NHSPSTOT Index"].calendar_pattern
    # a genuine 10x shock, confirmed by its restatement (COVID-style)
    shock = _series([(280e3, "280,000"), (3.3e6, "3,300,000"), (3.0e6, "3,000,000")])
    shock.loc[1, "forecast"] = 1.5e6
    assert ec.suspect_prints(shock, pat).empty
    # a genuine tiny value (a +20k payroll month), restated at about itself
    tiny = _series([(300e3, "300,000"), (20e3, "20,000"), (200e3, "200,000")])
    assert ec.suspect_prints(tiny, pat).empty


def test_calendar_prelims_and_consensus(tmp_path):
    root = tmp_path / "cal"
    rows = _rows([
        (D("2024-09-13"), "Consumer sentiment (prelim)", "Sept.", 69.0, 67.9, D("2024-09-14")),
        (D("2024-09-27"), "Consumer sentiment (final)", "Sept.", 70.1, 69.0, D("2024-09-28")),
        (D("2010-02-12"), "Consumer sentiment", "Feb.", 73.7, 74.4, D("2010-02-13")),  # unlabelled, mid-month
        (D("2010-02-26"), "Consumer sentiment", "Feb.", 73.6, 73.7, D("2010-02-27")),  # unlabelled, end-month
    ])
    rows["forecast"] = [68.5, 69.3, 75.0, 73.8]
    pc.store_rows(rows, root=root)
    pre = pc.calendar_prelims("UMCSENT", D("2030-01-01"), root=root)
    assert sorted(zip(pre["realtime_start"], pre["value"])) == [(D("2010-02-12"), 73.7), (D("2024-09-13"), 69.0)]
    c = pc.consensus(MACRO_RELEASES["CONSSENT Index"], root=root)
    assert list(c["surprise"].round(2)) == [-1.3, -0.2, 0.5, 0.8]
    assert len(pc.consensus(MACRO_RELEASES["CONSSENT Index"], "2024-09-20", root=root)) == 3  # point in time
