"""Offline tests for infra/analytics/wirp.py (Fed rate-probability extraction). No API."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.analytics.wirp import (
    implied_rate, level_probabilities, meeting_schedule, modal_outcome,
    post_meeting_rate_from_flat_month, solve_post_meeting_rate,
)

D = pd.Timestamp
P = lambda s: pd.Period(s, freq="M")  # noqa: E731


# ---------------------------------------------------------------------- implied_rate
def test_implied_rate_is_100_minus_price():
    assert implied_rate(95.75) == pytest.approx(4.25)


# -------------------------------------------------------- solve_post_meeting_rate
def test_solve_post_meeting_rate_recovers_exact_round_trip():
    """Construct the month average FORWARD from known pre/post rates (the day-weighted
    formula itself), then solve BACKWARD and recover the known post-meeting rate
    exactly - a closed-form self-consistency check, like test_analytics.py's
    put-call-parity test."""
    pre, post, d, N = 5.0, 5.25, 10, 31
    avg = (pre * d + post * (N - d)) / N
    assert solve_post_meeting_rate(pre, avg, d, N) == pytest.approx(post, abs=1e-10)


def test_solve_post_meeting_rate_matches_published_worked_example():
    """Keasler & Goff (2007), J. Economics & Finance Education 6(2), p.11: pre=5.0%,
    meeting on the 10th of a 31-day month, month avg (their own rounded intermediate)
    5.169% -> post ~5.25%. Loose tolerance since the paper rounds to 3 decimals."""
    assert solve_post_meeting_rate(5.0, 5.169, 10, 31) == pytest.approx(5.25, abs=1e-3)


def test_solve_post_meeting_rate_rejects_meeting_on_last_day():
    with pytest.raises(ValueError):
        solve_post_meeting_rate(5.0, 5.1, 31, 31)


# ------------------------------------------------------------------ level_probabilities
def test_level_probabilities_exact_multiple_is_a_single_level():
    assert level_probabilities(25.0) == pytest.approx({1: 1.0})
    assert level_probabilities(0.0) == pytest.approx({0: 1.0})
    assert level_probabilities(-50.0) == pytest.approx({-2: 1.0})


def test_level_probabilities_between_levels_sums_to_one_and_matches_expected_value():
    levels = level_probabilities(32.5, step_bps=25.0)
    assert levels == pytest.approx({1: 0.7, 2: 0.3})
    assert sum(levels.values()) == pytest.approx(1.0)
    expected_change = sum(level * 25.0 * p for level, p in levels.items())
    assert expected_change == pytest.approx(32.5)


def test_level_probabilities_cut_direction_uses_negative_levels():
    """-32.5bp sits between -1 step (-25bp) and -2 steps (-50bp), closer to -1 - so -1
    gets the larger weight (0.7), mirroring the +32.5bp case's weighting on +1 (0.7)."""
    levels = level_probabilities(-32.5, step_bps=25.0)
    assert levels == pytest.approx({-2: 0.3, -1: 0.7})


# ------------------------------------------------------- post_meeting_rate_from_flat_month
def test_post_meeting_rate_from_flat_month_is_a_direct_pass_through():
    assert post_meeting_rate_from_flat_month(4.35) == 4.35


# --------------------------------------------------------------------- meeting_schedule
def test_meeting_schedule_prefers_the_flat_next_month_over_day_weighting():
    """Regression test for a real bug found 2026-09-28: day-weighting a LATE-month
    meeting (Oct 27-28, day 28 of 31 - amplification ~10x) is wildly noise-amplified.
    When the following month (Nov) is flat and cached, it must be used instead, and
    must NOT match what plain day-weighting of October would have given."""
    anchor = 3.63
    meeting_date = D("2026-10-28")
    november_avg = 4.035  # flat month, directly read - no day-weighting
    october_avg = 3.89  # would day-weight to a nonsensical ~6.3% via the fallback path

    rates = pd.Series({P("2026-10"): october_avg, P("2026-11"): november_avg})
    schedule = meeting_schedule(rates, [meeting_date], anchor)

    row = schedule.loc[schedule["probability"].idxmax()]
    assert row["method"] == "next_month_flat"
    assert row["implied_rate"] == pytest.approx(november_avg)
    assert row["implied_rate"] < 5.0  # sanity: nowhere near the day-weighted ~6.3%


def test_meeting_schedule_falls_back_to_day_weighting_when_next_month_uncached():
    anchor = 4.00
    meeting_date = D("2026-01-28")
    avg = (anchor * 28 + 4.25 * (31 - 28)) / 31  # constructed for an exact +25bp
    rates = pd.Series({P("2026-01"): avg})  # February intentionally absent

    schedule = meeting_schedule(rates, [meeting_date], anchor)

    row = schedule.loc[schedule["probability"].idxmax()]
    assert row["method"] == "day_weighted"
    assert row["implied_rate"] == pytest.approx(4.25, abs=1e-9)


def test_meeting_schedule_never_uses_a_next_month_that_itself_has_a_meeting():
    """Back-to-back meeting months never occur in the real FOMC schedule, but the
    fallback to day-weighting must still trigger correctly if it ever did."""
    anchor = 4.00
    m1_date, m2_date = D("2026-01-28"), D("2026-02-18")  # adjacent months, both meetings
    m1_avg = (anchor * 28 + 4.25 * (31 - 28)) / 31  # exact +25bp, day-weighted
    m2_avg = 100.0  # deliberately absurd - must NOT be read as February's "flat" rate
    rates = pd.Series({P("2026-01"): m1_avg, P("2026-02"): m2_avg})

    schedule = meeting_schedule(rates, [m1_date, m2_date], anchor)
    jan = schedule[schedule["month"] == "2026-01"]
    assert jan.loc[jan["probability"].idxmax(), "method"] == "day_weighted"
    assert jan.loc[jan["probability"].idxmax(), "implied_rate"] == pytest.approx(4.25, abs=1e-9)


def test_meeting_schedule_chains_and_recovers_a_known_path():
    """Two meeting months, each constructed so its contract average day-weights to a
    KNOWN exact post-meeting rate (their own next months are deliberately absent, so
    the day-weighted fallback is what's exercised) - meeting_schedule must chain
    meeting 2's pre_rate from meeting 1's solved post_rate, not from the anchor again."""
    anchor = 4.00
    m1_date, m1_pre, m1_post = D("2026-01-28"), anchor, 4.25  # +25bp
    m1_avg = (m1_pre * 28 + m1_post * (31 - 28)) / 31
    m2_date, m2_post = D("2026-03-18"), 4.50  # +25bp again, chained from m1's post
    m2_avg = (m1_post * 18 + m2_post * (31 - 18)) / 31

    rates = pd.Series({P("2026-01"): m1_avg, P("2026-03"): m2_avg})
    schedule = meeting_schedule(rates, [m2_date, m1_date], anchor)  # unsorted input

    jan = schedule[schedule["month"] == "2026-01"]
    mar = schedule[schedule["month"] == "2026-03"]
    assert jan["implied_rate"].iloc[0] == pytest.approx(m1_post, abs=1e-9)
    assert jan["probability"].iloc[0] == pytest.approx(1.0)
    assert jan["method"].iloc[0] == "day_weighted"
    assert mar["pre_rate"].iloc[0] == pytest.approx(m1_post, abs=1e-9)  # chained, not anchor
    assert mar["implied_rate"].iloc[0] == pytest.approx(m2_post, abs=1e-9)


def test_meeting_schedule_skips_months_without_cached_contract():
    anchor = 4.00
    priced_date, priced_post = D("2026-03-18"), 4.25
    priced_avg = (anchor * 18 + priced_post * (31 - 18)) / 31
    rates = pd.Series({P("2026-03"): priced_avg})  # January intentionally missing

    schedule = meeting_schedule(rates, [D("2026-01-28"), D("2026-03-18")], anchor)

    assert set(schedule["month"]) == {"2026-03"}
    # the priced meeting still chains from the ANCHOR (the skipped Jan meeting never
    # produced a solved rate to chain from)
    assert schedule["pre_rate"].iloc[0] == pytest.approx(anchor)
    assert schedule["implied_rate"].iloc[0] == pytest.approx(priced_post, abs=1e-9)


def test_meeting_schedule_empty_when_no_months_cached():
    schedule = meeting_schedule(pd.Series(dtype="float64"), [D("2026-01-28")], 4.0)
    assert schedule.empty


# ----------------------------------------------------------------------- modal_outcome
def test_modal_outcome_picks_highest_probability_row_per_meeting():
    schedule = pd.DataFrame({
        "meeting_date": [D("2026-01-28"), D("2026-01-28"), D("2026-03-18")],
        "month": ["2026-01", "2026-01", "2026-03"],
        "pre_rate": [4.0, 4.0, 4.25],
        "implied_rate": [4.25, 4.5, 4.25],
        "change_bps": [21.0, 21.0, 0.0],
        "outcome_step": [0, 1, 0],
        "outcome_bps": [0.0, 25.0, 0.0],
        "outcome_rate": [4.0, 4.25, 4.25],
        "probability": [0.16, 0.84, 1.0],
    })
    modal = modal_outcome(schedule)
    assert len(modal) == 2
    jan_row = modal[modal["month"] == "2026-01"].iloc[0]
    assert jan_row["probability"] == pytest.approx(0.84)
    assert jan_row["outcome_bps"] == pytest.approx(25.0)


def test_modal_outcome_empty_in_empty_out():
    assert modal_outcome(pd.DataFrame()).empty
