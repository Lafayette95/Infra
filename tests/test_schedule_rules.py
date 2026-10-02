"""Date rules: generation (overrides, calendars, abstention), scoring, and the release
calendar's projection of validated rules."""
from __future__ import annotations

import pandas as pd

from infra.pipeline import release_calendar as prc
from infra.processing import schedule_rules as sr
from infra.reference.events import EVENTS

D = pd.Timestamp


def test_ism_first_business_day_with_its_january_exception():
    rule = EVENTS["US_ISM_MANUFACTURING"].rule
    got = sr.rule_dates(rule, "2026-01-01", "2026-03-31")
    assert list(got) == [D("2026-01-05"), D("2026-02-02"), D("2026-03-02")]  # Jan: 2nd business day


def test_market_calendar_skips_good_friday():
    assert sr.rule_dates("last_business_day", "2024-03-01", "2024-03-31")[0] == D("2024-03-29")  # federal: Good Friday open
    assert sr.rule_dates("market:last_business_day", "2024-03-01", "2024-03-31")[0] == D("2024-03-28")


def test_abstained_months_produce_no_date_and_no_miss():
    rule = "last_weekday:TUE;dec=none"
    assert list(sr.rule_dates(rule, "2025-11-01", "2025-12-31")) == [D("2025-11-25")]
    observed = pd.DatetimeIndex(["2025-11-25", "2025-12-23"])  # December's actual date is irregular
    s = sr.score(rule, observed)
    assert s["n"].sum() == 1 and s["hit"].iloc[0] == 1.0


def test_day_or_next_business_day():
    # 2025-11-15 is a Saturday -> Monday the 17th
    assert sr.rule_dates("day_or_next_business_day:15", "2025-11-01", "2025-11-30")[0] == D("2025-11-17")


def test_best_rule_recovers_the_true_rule():
    truth = sr.rule_dates("nth_weekday:2:TUE", "2019-01-01", "2026-09-30")
    rule, hit, n = sr.best_rule(truth, since=2020)
    assert rule == "nth_weekday:2:TUE" and hit == 1.0


def test_rule_projection_rows():
    rows = prc.rule_schedules(observed=D("2026-10-02"), horizon_months=3)
    ism = rows[rows["event"] == "US_ISM_MANUFACTURING"]
    assert list(ism["timestamp"]) == [D("2026-11-02 15:00"), D("2026-12-01 15:00")]  # 10:00 New York (EST)
    assert (ism["source"] == "rule").all() and (ism["known_from"] == D("2026-10-02")).all()
    sp = rows[rows["event"] == "US_SPGLOBAL_MANUFACTURING_PMI"]
    assert (sp["stage"] == "final").all()
