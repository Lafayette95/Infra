"""World Interest Rate Probability (WIRP) analytics: implied probability of a Fed
rate hike/cut at each upcoming FOMC meeting, backed out of 30-Day Fed Funds futures
(ZQ) prices. See CLAUDE.md section 11 for methodology, sources and known limitations.

Pure functions only - no API calls, no pipeline/storage imports, no FOMC-calendar
config (meeting dates are passed in as plain timestamps by the caller, matching
infra.analytics.rnd/forward's convention of taking already-loaded data). Everything
here works in RATE space (percent), not futures price - callers convert
``100 - price`` at the boundary (see infra.dashboard.wirp_selectors).
"""
from __future__ import annotations

import math
from typing import Sequence

import pandas as pd


def implied_rate(price: float) -> float:
    """Average EFFR (%) a 30-Day Fed Funds futures price implies for its contract
    month - Robertson & Thornton (1997): the contract quotes 100 minus the expected
    average daily rate for the delivery month."""
    return 100.0 - price


def solve_post_meeting_rate(
    pre_rate: float, month_avg_rate: float, meeting_day: int, days_in_month: int,
) -> float:
    """The market-implied post-meeting rate, correcting for the meeting not falling on
    the 1st of the month (Geraty 2000, via Keasler & Goff, "Using Fed Funds Futures to
    Predict a Federal Reserve Rate Hike", J. Economics & Finance Education 6(2), 2007 -
    see CLAUDE.md section 11).

    A contract month's average rate is a day-weighted blend of the OLD rate (days
    1..meeting_day, inclusive - calendar days, not business days, per the same source)
    and the NEW rate (meeting_day+1..days_in_month, i.e. effective the day after the
    decision):

        month_avg = (pre_rate * meeting_day + post_rate * (days_in_month - meeting_day)) / days_in_month

    Solved here for the one unknown, ``post_rate``.
    """
    days_after = days_in_month - meeting_day
    if days_after <= 0:
        raise ValueError(
            f"meeting_day={meeting_day} leaves no days after it in a {days_in_month}-day "
            "month - the day-weighting formula needs at least one post-meeting day"
        )
    return (days_in_month * month_avg_rate - meeting_day * pre_rate) / days_after


def level_probabilities(change_bps: float, step_bps: float = 25.0, *, eps: float = 1e-9) -> dict[int, float]:
    """Decompose a continuous implied change (bps) into probabilities over the two
    nearest discrete step levels (multiples of ``step_bps``, signed: positive = hike,
    negative = cut, 0 = hold) - the standard assumption that the Fed only ever moves in
    whole steps (Keasler & Goff 2007; CME's own FedWatch methodology states the same:
    "the size of a rate change is always in multiples of 25bp"). A change of, say,
    +32.5bp is read as "70% chance of +25bp, 30% chance of +50bp" - weighted so the
    probability-weighted average change matches the market-implied value exactly.

    This is a per-meeting MARGINAL decomposition, not CME's full cross-meeting
    probability tree (which also enforces a consistent joint path across meetings). It
    is exact for a single meeting's own implied change; see CLAUDE.md section 11 for
    what that does and doesn't capture.
    """
    steps = change_bps / step_bps
    floor_steps = math.floor(steps + eps)  # nudge values landing exactly on an integer
    frac = steps - floor_steps
    if frac < eps:
        return {floor_steps: 1.0}
    return {floor_steps: 1.0 - frac, floor_steps + 1: frac}


def meeting_schedule(
    contract_rates: pd.Series,
    meeting_dates: Sequence[pd.Timestamp],
    anchor_rate: float,
    *,
    step_bps: float = 25.0,
) -> pd.DataFrame:
    """One row per (meeting, discrete outcome level) - long format, for a probability-
    distribution chart per meeting. Chains meeting to meeting: each meeting's solved
    post-meeting rate becomes the next meeting's pre-meeting rate (self-consistent, no
    external EFFR feed needed - matches infra.analytics.forward's "no separate rate
    curve" approach).

    ``contract_rates``: Series indexed by ``pd.Period(freq="M")``, values = the average
    implied rate (%) for that month (``implied_rate`` applied to that month's ZQ
    contract price). ``meeting_dates``: FOMC decision/announcement days (the last day
    of each meeting - CLAUDE.md section 11's day-weighting convention), any order.
    ``anchor_rate``: the prevailing rate (%) immediately before the first meeting here -
    typically derived from the nearest FLAT (no-meeting) month's own contract, so it
    needs no day-weighting itself (see infra.dashboard.wirp_selectors.find_anchor).

    A meeting whose month has no entry in ``contract_rates`` is silently skipped (no
    cached contract yet) and does not break the chain - the next priced meeting still
    chains from the last SOLVED rate, not the skipped one's.
    """
    rows = []
    pre_rate = anchor_rate
    for meeting_date in sorted(pd.Timestamp(d) for d in meeting_dates):
        month = pd.Period(meeting_date, freq="M")
        if month not in contract_rates.index or pd.isna(contract_rates[month]):
            continue
        avg = float(contract_rates[month])
        meeting_day, days_in_month = meeting_date.day, month.days_in_month
        post_rate = solve_post_meeting_rate(pre_rate, avg, meeting_day, days_in_month)
        change_bps = (post_rate - pre_rate) * 100.0
        for level, prob in level_probabilities(change_bps, step_bps).items():
            rows.append({
                "meeting_date": meeting_date,
                "month": str(month),
                "pre_rate": pre_rate,
                "implied_rate": post_rate,
                "change_bps": change_bps,
                "outcome_step": level,
                "outcome_bps": level * step_bps,
                "outcome_rate": pre_rate + level * step_bps / 100.0,
                "probability": prob,
            })
        pre_rate = post_rate  # chain forward regardless of how many levels were split
    return pd.DataFrame(rows, columns=[
        "meeting_date", "month", "pre_rate", "implied_rate", "change_bps",
        "outcome_step", "outcome_bps", "outcome_rate", "probability",
    ])


def modal_outcome(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per meeting: its highest-probability discrete outcome (summary tiles) -
    mirrors infra.analytics.rnd.percentiles's role of reducing a distribution to a
    headline number. Empty in, empty out."""
    if schedule.empty:
        return schedule
    idx = schedule.groupby("meeting_date")["probability"].idxmax()
    return schedule.loc[idx].sort_values("meeting_date").reset_index(drop=True)
