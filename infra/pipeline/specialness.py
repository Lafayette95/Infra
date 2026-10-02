"""Specialness inputs for financing layer 3 (CLAUDE.md 20): the estimation panel, the
point-in-time profile and half-life, and each bond's state on a day. Reads disk only.

* Observed specialness of a bond on a day = the NY Fed lending fee above the minimum fee
  (``infra.processing.sec_lending.excess_fee``), par-weighted over the day's operations,
  in bp; a tracked bond not borrowed that day counts as 0 (not special beyond the floor -
  assumes the Fed held it, true for nearly every coupon issue it rolls at auction).
* Tracked bonds = the OTR map's ranks 0..MAX_RANK (issue convention), from 2008-09-02.
  On-the-run life = its issue date to its successor's (the next original issue of the
  same security type and term): known once the successor is announced, else expected
  from the tenor's median cycle length.
* The profile and half-life are estimated from the ``ESTIMATION_YEARS`` calendar years
  BEFORE ``as_of``'s year (point in time; cached per year).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import specialness as sa
from infra.analytics.sofr_curve import business_days
from infra.config import (
    SEC_LENDING_DIR,
    SEC_LENDING_MIN_FEE,
    TREASURY_OTR_DIR,
    TREASURY_OTR_TENORS,
    TREASURY_SECURITIES_DIR,
)
from infra.pipeline.sec_lending import read_sec_lending
from infra.pipeline.treasury_otr import read_otr
from infra.pipeline.treasury_ref import read_securities
from infra.processing.sec_lending import excess_fee

ESTIMATION_YEARS = 5
PANEL_START = pd.Timestamp("2008-09-02")  # the OTR map's first day
DEFAULT_HALF_LIFE_BD = 2.5
_ONE_DAY = pd.Timedelta(days=1)


def observed_specialness(start, end, *, root: Path = SEC_LENDING_DIR) -> pd.Series:
    """Excess lending fee (bp), par-weighted per (timestamp, cusip), for Treasuries lent
    in ``[start, end)``."""
    df = read_sec_lending(start, end, root=root)
    df = df[df["par_accepted"] > 0]
    if df.empty:
        return pd.Series(dtype="float64", index=pd.MultiIndex.from_arrays([[], []], names=["timestamp", "cusip"]))
    df = df.assign(sp=excess_fee(df, SEC_LENDING_MIN_FEE) * 100, w=df["par_accepted"])
    g = df.assign(spw=df["sp"] * df["w"]).groupby(["timestamp", "cusip"])[["spw", "w"]].sum()
    return (g["spw"] / g["w"]).rename("sp")


def successors(securities: pd.DataFrame) -> pd.DataFrame:
    """Per tracked original issue: ``cusip``, ``tenor``, ``issue_date`` and its successor's
    ``next_issue`` (NaT if none announced yet)."""
    out = []
    for tenor, (stype, term) in TREASURY_OTR_TENORS.items():
        s = securities[(securities["security_type"] == stype) & (securities["original_term"] == term)]
        s = s.dropna(subset=["issue_date"]).sort_values("issue_date")
        out.append(pd.DataFrame({"cusip": s["cusip"].astype(str).to_numpy(), "tenor": tenor,
                                 "issue_date": pd.to_datetime(s["issue_date"]).to_numpy(),
                                 "next_issue": pd.to_datetime(s["issue_date"]).shift(-1).to_numpy()}))
    return pd.concat(out, ignore_index=True)


def median_cycle_days(succ: pd.DataFrame) -> dict[str, float]:
    """Median calendar days between consecutive original issues, per tenor."""
    d = succ.dropna(subset=["next_issue"])
    return ((d["next_issue"] - d["issue_date"]).dt.days.groupby(d["tenor"]).median()).to_dict()


def build_panel(start, end, *, otr_root: Path = TREASURY_OTR_DIR, lending_root: Path = SEC_LENDING_DIR,
                securities_root: Path = TREASURY_SECURITIES_DIR) -> pd.DataFrame:
    """One row per tracked bond-day in ``[start, end)``: ``timestamp``, ``cusip``,
    ``tenor``, ``rank``, ``frac``, ``days_since_loss``, ``bucket``, ``sp`` (bp)."""
    start, end = max(pd.Timestamp(start), PANEL_START), pd.Timestamp(end)
    otr = read_otr(start, end, root=otr_root)
    otr = otr[otr["rank"] <= sa.MAX_RANK]
    if otr.empty:
        return pd.DataFrame(columns=["timestamp", "cusip", "tenor", "rank", "frac", "days_since_loss", "bucket", "sp"])
    succ = successors(read_securities(root=securities_root)).set_index("cusip")
    p = otr[["timestamp", "cusip", "tenor", "rank"]].copy()
    p["cusip"] = p["cusip"].astype(str)
    p = p.join(succ[["issue_date", "next_issue"]], on="cusip")
    bd = business_days(p["issue_date"].min() - pd.Timedelta(days=7), p["timestamp"].max() + pd.Timedelta(days=400))
    pos = lambda s: bd.searchsorted(pd.DatetimeIndex(s))  # noqa: E731
    t, iss = pos(p["timestamp"]), pos(p["issue_date"])
    nxt = np.where(p["next_issue"].notna(), pos(p["next_issue"].fillna(p["timestamp"])), -1)
    life = np.where(nxt > iss, nxt - iss, np.nan)
    p["frac"] = (t - iss) / life
    p["days_since_loss"] = np.where(nxt >= 0, t - nxt, -1)
    p["bucket"] = sa.phase_buckets(p.fillna({"frac": 0.0}))
    sp = observed_specialness(start, end, root=lending_root)
    p = p.join(sp, on=["timestamp", "cusip"])
    p["sp"] = p["sp"].fillna(0.0)
    return p.reset_index(drop=True)


@dataclass(frozen=True)
class SpecialnessModel:
    profile: pd.Series
    half_life_bd: float
    estimated_from: tuple[pd.Timestamp, pd.Timestamp]


@lru_cache(maxsize=32)
def _estimate_year(year: int, years: int, otr_root: str, lending_root: str, securities_root: str) -> SpecialnessModel:
    end = pd.Timestamp(year=year, month=1, day=1)
    start = end - pd.DateOffset(years=years)
    panel = build_panel(start, end, otr_root=Path(otr_root), lending_root=Path(lending_root),
                        securities_root=Path(securities_root))
    if panel.empty:
        return SpecialnessModel(pd.Series(dtype="float64"), DEFAULT_HALF_LIFE_BD, (start, end))
    profile = sa.lifecycle_profile(panel)
    return SpecialnessModel(profile, sa.half_life(panel, profile), (max(start, PANEL_START), end))


def estimate(as_of, *, years: int = ESTIMATION_YEARS, otr_root: Path = TREASURY_OTR_DIR,
             lending_root: Path = SEC_LENDING_DIR, securities_root: Path = TREASURY_SECURITIES_DIR) -> SpecialnessModel:
    """Profile and half-life from the ``years`` calendar years before ``as_of``'s year."""
    return _estimate_year(pd.Timestamp(as_of).year, years, str(otr_root), str(lending_root), str(securities_root))


def bond_states(as_of, *, otr_root: Path = TREASURY_OTR_DIR, lending_root: Path = SEC_LENDING_DIR,
                securities_root: Path = TREASURY_SECURITIES_DIR) -> dict[str, sa.BondState]:
    """State of every tracked bond, plus every bond lent on the latest lending day <=
    ``as_of`` (untracked ones get no profile, only their decaying observation)."""
    as_of = pd.Timestamp(as_of).normalize()
    otr = read_otr(as_of - pd.Timedelta(days=10), as_of + _ONE_DAY, root=otr_root)
    otr = otr[otr["timestamp"] == otr["timestamp"].max()] if len(otr) else otr
    sec = read_securities(as_of, root=securities_root)  # announced by as_of: point in time
    succ = successors(sec).set_index("cusip")
    cycle = median_cycle_days(successors(sec))
    sp = observed_specialness(as_of - pd.Timedelta(days=10), as_of + _ONE_DAY, root=lending_root)
    latest = sp.index.get_level_values("timestamp").max() if len(sp) else None
    today = sp.xs(latest, level="timestamp") if latest is not None else pd.Series(dtype="float64")
    states = {}
    for _, row in otr[otr["rank"] <= sa.MAX_RANK].iterrows():
        c = str(row["cusip"])
        issue = pd.Timestamp(succ.loc[c, "issue_date"]) if c in succ.index else pd.Timestamp(row["issue_date"])
        nxt = succ.loc[c, "next_issue"] if c in succ.index else pd.NaT
        if pd.isna(nxt):
            nxt = issue + pd.Timedelta(days=cycle.get(row["tenor"], 91))  # not announced yet: expected
        states[c] = sa.BondState(c, row["tenor"], int(row["rank"]), issue, pd.Timestamp(nxt), float(today.get(c, 0.0)))
    for c, v in today.items():
        if c not in states:
            states[c] = sa.BondState(c, None, None, None, None, float(v))
    return states


def term_specialness_bp(as_of, cusip: str, start, end, *, model: SpecialnessModel | None = None,
                        states: dict[str, sa.BondState] | None = None, **roots) -> float:
    """Expected average specialness (bp) of ``cusip`` over calendar days ``[start, end)``,
    as known at the end of ``as_of``. 0 for a bond neither tracked nor recently lent."""
    as_of = pd.Timestamp(as_of).normalize()
    model = estimate(as_of, **roots) if model is None else model
    states = bond_states(as_of, **roots) if states is None else states
    state = states.get(str(cusip))
    if state is None:
        return 0.0
    return sa.term_specialness(state, as_of, start, end, model.profile, model.half_life_bd)
