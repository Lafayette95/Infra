"""Expected richness drift of a Treasury to a future date, from event profiles - pure, no
I/O (infra/models/basis/CLAUDE.md, add-on MS).

``profiles``: the event-profile interface (``infra.analytics.event_study``:
``event_type, tenor, rel_day, mean_bp`` - here the means NET of specialness carry,
``infra.analytics.event_netting``), read as a step function of business days from the
event. Per bond (``bond_drift_bp``), the expected change of its curve residual from
``today`` to ``target``, by one rule each (they would otherwise overlap):

* ``new_issue`` - a bond within the profile's window of its ISSUE: P(age at target) -
  P(age today). For the monthly tenors (2/3/5/7y) that window already covers the bond's
  loss of on-the-run status ~20 business days in.
* ``otr_roll`` - otherwise, the CURRENT on-the-run bond whose successor issues before
  ``target``: P(k at target) - P(k today), k = business days from the successor's issue,
  clipped to the profile's window (flat outside it).
* otherwise 0. Reopenings and auction concession are left out of v1 (short-lived, or
  inside the above).
A positive drift = the bond is expected to CHEAPEN (its yield rises relative to the curve).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def profile_lookup(profiles: pd.DataFrame, event_type: str, tenor: str) -> pd.Series | None:
    """``mean_bp`` by rel_day for one (event type, tenor), or None if not estimated."""
    p = profiles[(profiles["event_type"] == event_type) & (profiles["tenor"] == tenor)]
    return None if p.empty else p.set_index("rel_day")["mean_bp"].sort_index()


def _at(p: pd.Series, k: int) -> float:
    """The profile at business day ``k``, flat beyond its window."""
    k = int(np.clip(k, p.index.min(), p.index.max()))
    return float(p.get(k, np.interp(k, p.index.to_numpy(), p.to_numpy())))


def bond_drift_bp(state: dict, today, target, bdays: pd.DatetimeIndex, profiles: pd.DataFrame) -> tuple[float, str]:
    """(drift bp, rule) for one bond. ``state``: ``tenor`` (e.g. "10y"), ``issue_date``,
    ``is_otr`` (bool), ``successor_issue`` (date or NaT). ``bdays``: the business-day
    calendar the profiles were measured on (or one like it)."""
    pos = lambda d: int(bdays.searchsorted(pd.Timestamp(d)))
    t0, t1 = pos(today), pos(target)
    if t1 <= t0:
        return 0.0, "none"
    new = profile_lookup(profiles, "new_issue", state["tenor"])
    if new is not None and pd.notna(state.get("issue_date")):
        age = t0 - pos(state["issue_date"])
        if 0 <= age <= int(new.index.max()):
            return _at(new, age + (t1 - t0)) - _at(new, age), "new_issue"
    roll = profile_lookup(profiles, "otr_roll", state["tenor"])
    if roll is not None and state.get("is_otr") and pd.notna(state.get("successor_issue")):
        s = pos(state["successor_issue"])
        if s <= t1:
            return _at(roll, t1 - s) - _at(roll, t0 - s), "otr_roll"
    return 0.0, "none"


AGE_BUCKETS = (0, 20, 40, 60, 90, 130, 190, 260, 400, 600)  # business days since issue


def aging_profile(daily: pd.DataFrame, *, buckets=AGE_BUCKETS) -> pd.DataFrame:
    """The AGING profile per tenor, in the event-profile interface's shape (``event_type``
    = "aging", ``rel_day`` = age in business days 0..max, ``mean_bp`` = the expected
    CUMULATIVE residual change since issue). ``daily``: ``tenor, age_bd, d1_bp`` - each
    bond-day's residual change (net of specialness carry), its age. Within each age bucket
    the mean daily change is constant, so the profile is piecewise linear in age.
    Found 2026-10-03: the event rules (``bond_drift_bp``) gave a new 10y +2.0bp and the
    1-old 10y 0 (outside both rules), flipping TN's CTD - Brier 0.050 -> 0.150; the realised
    pair moved -0.11 / +0.05bp. Aging applies to EVERY bond, so only the RELATIVE drift of
    two deliverables matters, as it should."""
    rows = []
    for tenor, g in daily.groupby("tenor"):
        cum, ages = 0.0, []
        for lo, hi in zip(buckets[:-1], buckets[1:]):
            m = g.loc[(g["age_bd"] > lo) & (g["age_bd"] <= hi), "d1_bp"].mean()
            m = 0.0 if not np.isfinite(m) else float(m)
            for a in range(lo + 1, hi + 1):
                cum += m
                ages.append((a, cum))
        rows += [("aging", tenor, 0, 0.0)] + [("aging", tenor, a, v) for a, v in ages]
    return pd.DataFrame(rows, columns=["event_type", "tenor", "rel_day", "mean_bp"])


def aging_drift_bp(state: dict, today, target, bdays: pd.DatetimeIndex, profiles: pd.DataFrame) -> tuple[float, str]:
    """(drift bp, "aging") for one bond: P(age at target) - P(age today), flat beyond the
    profile's last age; (0, "none") without a profile for its tenor or an issue date."""
    p = profile_lookup(profiles, "aging", state.get("tenor"))
    if p is None or pd.isna(state.get("issue_date")):
        return 0.0, "none"
    pos = lambda d: int(bdays.searchsorted(pd.Timestamp(d)))
    age0 = pos(today) - pos(state["issue_date"])
    age1 = age0 + max(pos(target) - pos(today), 0)
    if age0 < 0:
        return 0.0, "none"
    return _at(p, age1) - _at(p, age0), "aging"
