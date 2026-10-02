"""Per-CUSIP specialness over a financing term (financing model layer 3, CLAUDE.md 20).
Pure functions, no I/O.

Specialness on day d (bp below general collateral) is modelled as

    S(d) = mu(tenor, rank, phase(d)) + (s_now - mu(tenor, rank, phase(now))) * 2^(-bd(now, d) / half_life)

* ``mu`` - the LIFECYCLE profile: mean specialness by tenor, on/off-the-run rank and phase
  of the issuance cycle (rank 0: share of its on-the-run life elapsed; rank 1: business
  days since it lost on-the-run status; rank 2+: one value). Measured 2026-10-02: the 10y
  and 20y peak early and fade well before the next new issue (reopenings add supply);
  the 2y and 5y peak late and keep a few days' tail once 1-old; 3y/7y/30y barely move.
  The rank changes along the path (an on-the-run bond becomes 1-old when its successor
  is issued), so known events move the expectation, not just decay.
* ``s_now`` - today's OBSERVED specialness: the NY Fed lending fee above the minimum fee
  (0 if the bond wasn't borrowed: not special beyond the floor). Its deviation from the
  profile decays with ``half_life`` business days (measured ~2-3).
The term value is the average of S over the term's calendar days (a weekend takes the
previous business day's value) - rolling overnight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from infra.analytics.sofr_curve import business_days as _business_days

RANK0_BUCKETS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001)  # share of on-the-run life elapsed
RANK1_BUCKETS = (0, 1, 3, 5, 10, 20, 60, 10**6)  # business days since losing on-the-run status
MAX_RANK = 5  # deeper off-the-runs: profile 0


def phase_bucket(rank: int, frac: float | None = None, days_since_loss: int | None = None) -> int:
    """The profile bucket of one bond-day (rank 2+ has a single bucket, 0)."""
    if rank == 0:
        return int(np.searchsorted(RANK0_BUCKETS, min(max(frac, 0.0), 1.0), side="right") - 1)
    if rank == 1:
        return int(np.searchsorted(RANK1_BUCKETS, max(days_since_loss, 0), side="right") - 1)
    return 0


def phase_buckets(panel: pd.DataFrame) -> pd.Series:
    """Vectorised ``phase_bucket`` over ``rank``, ``frac``, ``days_since_loss`` columns."""
    out = pd.Series(0, index=panel.index, dtype="int64")
    r0, r1 = panel["rank"] == 0, panel["rank"] == 1
    out[r0] = np.searchsorted(RANK0_BUCKETS, panel.loc[r0, "frac"].clip(0, 1), side="right") - 1
    out[r1] = np.searchsorted(RANK1_BUCKETS, panel.loc[r1, "days_since_loss"].clip(lower=0), side="right") - 1
    return out


def lifecycle_profile(panel: pd.DataFrame) -> pd.Series:
    """Mean specialness (bp) by (tenor, rank, bucket). ``panel``: one row per bond-day with
    ``tenor``, ``rank``, ``bucket``, ``sp`` (bp, 0 where not borrowed)."""
    return panel.groupby(["tenor", "rank", "bucket"])["sp"].mean()


def half_life(panel: pd.DataFrame, profile: pd.Series, *, min_rho: float = 0.05) -> float:
    """Business-day half-life of deviations from the profile: pooled AR(1) on consecutive
    days of the same bond (ranks 0-1, where specialness lives)."""
    p = panel[panel["rank"] <= 1].copy()
    mu = profile.reindex(pd.MultiIndex.from_frame(p[["tenor", "rank", "bucket"]])).fillna(0.0).to_numpy()
    p["dev"] = p["sp"].to_numpy() - mu
    p = p.sort_values(["cusip", "timestamp"])
    lag = p.groupby("cusip")["dev"].shift()
    ok = lag.notna()
    x, y = lag[ok].to_numpy(), p.loc[ok, "dev"].to_numpy()
    rho = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else 0.0
    rho = min(max(rho, min_rho), 0.999)
    return float(np.log(0.5) / np.log(rho))


@dataclass(frozen=True)
class BondState:
    """What layer 3 needs about one bond as of today: its tenor (None if not a tracked
    on/off-the-run coupon), rank today, the business days of its on-the-run life
    (``otr_start``, ``otr_end`` = its successor's issue date, known or expected) and its
    observed specialness today (bp)."""
    cusip: str
    tenor: str | None
    rank: int | None
    otr_start: pd.Timestamp | None
    otr_end: pd.Timestamp | None
    observed_bp: float


def expected_path(state: BondState, as_of, days: pd.DatetimeIndex, profile: pd.Series, half_life_bd: float,
                  business_days: pd.DatetimeIndex) -> pd.Series:
    """S(d) in bp for each business day in ``days`` (all after ``as_of``)."""
    as_of = pd.Timestamp(as_of).normalize()
    bd_index = pd.DatetimeIndex(business_days)

    def bd_between(a, b) -> int:
        return int(bd_index.searchsorted(b) - bd_index.searchsorted(a))

    def mu(day) -> float:
        if state.tenor is None or state.rank is None or state.rank > MAX_RANK:
            return 0.0
        rank = state.rank
        if rank == 0 and state.otr_end is not None and day >= state.otr_end:
            rank, since = 1, bd_between(state.otr_end, day)
        elif rank == 1 and state.otr_end is not None:
            since = bd_between(state.otr_end, day)
        else:
            since = None
        if rank == 0:
            life = max(bd_between(state.otr_start, state.otr_end), 1)
            bucket = phase_bucket(0, frac=bd_between(state.otr_start, day) / life)
        else:
            bucket = phase_bucket(rank, days_since_loss=since if since is not None else 10**6)
        return float(profile.get((state.tenor, rank, bucket), 0.0))

    dev = state.observed_bp - mu(as_of)
    decay = np.log(2) / max(half_life_bd, 1e-6)
    return pd.Series([mu(d) + dev * np.exp(-decay * bd_between(as_of, d)) for d in days], index=days, dtype="float64")


def term_specialness_bp(path: pd.Series, start, end, fixing_days: pd.DatetimeIndex) -> float:
    """Average of ``path`` (bp, by business day) over calendar days ``[start, end)``, a
    non-business day taking the previous business day's value."""
    cal = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() - pd.Timedelta(days=1))
    if len(cal) == 0:
        return float("nan")
    fd = pd.DatetimeIndex(fixing_days).sort_values()
    gov = fd[np.clip(fd.searchsorted(cal, side="right") - 1, 0, None)]
    return float(path.reindex(gov).ffill().bfill().mean())


def term_specialness(state: BondState, as_of, start, end, profile: pd.Series, half_life_bd: float) -> float:
    """Expected average specialness (bp) of one bond over calendar days ``[start, end)``,
    from its state at the end of ``as_of`` (whose observation governs the days up to the
    next business day)."""
    as_of, end = pd.Timestamp(as_of).normalize(), pd.Timestamp(end).normalize()
    bd = _business_days(as_of - pd.Timedelta(days=10), end + pd.Timedelta(days=10))
    days = bd[(bd >= bd[bd <= as_of][-1]) & (bd < end)]
    return term_specialness_bp(expected_path(state, as_of, days, profile, half_life_bd, bd), start, end, bd)

