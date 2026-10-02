"""Financing rates of Treasury positions (CLAUDE.md 20): read what a model needs AS OF a
day, then evaluate it with ``infra.analytics.financing``. Reads disk only, never fetches.

Point in time (CLAUDE.md 3): everything is what was known at the end of ``as_of`` -
SR1 settlements of the latest settlement day <= ``as_of``, SOFR fixings published by then
(day D is published on D+1, so fixing days < ``as_of``), FOMC change dates for scheduled
meetings plus unscheduled ones already announced.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.analytics import financing as fa
from infra.analytics.sofr_curve import SofrPath, business_days, fit_sofr_path, year_end_turn_bp
from infra.config import (
    DAILY_FUTURES_DIR,
    FINANCING_MODELS,
    FOMC_MEETINGS,
    FUTURES_CONTRACTS_FILE,
    REPO_DIR,
)
from infra.pipeline import specialness as ps
from infra.pipeline.daily import read_daily_from_disk
from infra.pipeline.repo import read_repo
from infra.storage import contract_store

_ONE_DAY = pd.Timedelta(days=1)
SR1_ROOT = "SR1"
FIXING_HISTORY_DAYS = 3 * 366 + 60  # enough for the year-end turn's recent year-ends
SETTLEMENT_LOOKBACK_DAYS = 10


def change_dates(as_of, meetings=FOMC_MEETINGS) -> list[pd.Timestamp]:
    """Days a decided policy rate takes effect (announcement + 1) for every meeting known
    by ``as_of``: all scheduled ones (published a year ahead), unscheduled ones only once
    announced."""
    as_of = pd.Timestamp(as_of).normalize()
    return [pd.Timestamp(m.end_date) + _ONE_DAY for m in meetings
            if m.scheduled or pd.Timestamp(m.end_date) <= as_of]


def published_sofr(as_of, *, days: int = FIXING_HISTORY_DAYS, root: Path = REPO_DIR) -> pd.DataFrame:
    """NY Fed SOFR rows published by the end of ``as_of`` (fixing days before it)."""
    as_of = pd.Timestamp(as_of).normalize()
    return read_repo(as_of - pd.Timedelta(days=days), as_of, series=["SOFR"], root=root)


def sr1_settlements(as_of, *, futures_root: Path = DAILY_FUTURES_DIR,
                    contracts_file: Path = FUTURES_CONTRACTS_FILE) -> pd.DataFrame:
    """``month``, ``price``, ``ticker`` for every SR1 contract settled on the latest
    settlement day <= ``as_of`` (that day in ``settled_on``)."""
    as_of = pd.Timestamp(as_of).normalize()
    contracts = contract_store.read_contracts(contracts_file, SR1_ROOT)
    contracts = contracts[pd.to_datetime(contracts["expiry"]) >= as_of - pd.Timedelta(days=SETTLEMENT_LOOKBACK_DAYS)]
    if contracts.empty:
        return pd.DataFrame(columns=["month", "price", "ticker", "settled_on"])
    px = read_daily_from_disk(list(contracts["ticker"].astype(str)), as_of - pd.Timedelta(days=SETTLEMENT_LOOKBACK_DAYS),
                              as_of + _ONE_DAY, root=futures_root)
    px = px.dropna(subset=["settlement_price"])
    if px.empty:
        return pd.DataFrame(columns=["month", "price", "ticker", "settled_on"])
    day = px["timestamp"].max()
    px = px[px["timestamp"] == day]
    expiry = dict(zip(contracts["ticker"].astype(str), pd.to_datetime(contracts["expiry"])))
    month = px["ticker"].astype(str).map(expiry).dt.to_period("M").dt.start_time
    return pd.DataFrame({"month": month, "price": px["settlement_price"].astype(float), "ticker": px["ticker"].astype(str),
                         "settled_on": day}).sort_values("month").reset_index(drop=True)


def sofr_path(as_of, *, year_end_turn_years: int = 3, futures_root: Path = DAILY_FUTURES_DIR,
              contracts_file: Path = FUTURES_CONTRACTS_FILE, repo_root: Path = REPO_DIR,
              fixings: pd.DataFrame | None = None) -> SofrPath:
    """Layer 1's fitted overnight path as of ``as_of``."""
    as_of = pd.Timestamp(as_of).normalize()
    fut = sr1_settlements(as_of, futures_root=futures_root, contracts_file=contracts_file)
    if fut.empty:
        raise ValueError(f"no SR1 settlements on or before {as_of.date()}")
    fx = published_sofr(as_of, root=repo_root) if fixings is None else fixings
    series = fx.set_index("timestamp")["rate"].astype(float)
    turn = year_end_turn_bp(series, as_of, last_n=year_end_turn_years)
    return fit_sofr_path(as_of, fut[["month", "price"]], series, change_dates(as_of), year_end_turn=turn)


def financing_inputs(as_of, model: str = "v1", **paths) -> fa.FinancingInputs:
    """Read what ``model``'s layers need, as of ``as_of``."""
    spec = FINANCING_MODELS[model]
    need = set().union(*(fa.NEEDS[n] for n in (spec.base, spec.basis, spec.specialness)))
    as_of = pd.Timestamp(as_of).normalize()
    fixings = published_sofr(as_of, root=paths.get("repo_root", REPO_DIR)) if need & {"sofr_fixings", "sofr_path"} \
        else None
    path = sofr_path(as_of, year_end_turn_years=spec.year_end_turn_years, fixings=fixings, **paths) \
        if "sofr_path" in need else None
    special = "specialness" in need
    return fa.FinancingInputs(as_of=as_of, sofr_path=path, sofr_fixings=fixings,
                              specialness_model=ps.estimate(as_of) if special else None,
                              bond_states=ps.bond_states(as_of) if special else None)


def financing_rate(as_of, start, end, *, cusip: str | None = None, model: str = "v1",
                   inputs: fa.FinancingInputs | None = None, **paths) -> fa.FinancingQuote:
    """The model's financing rate (%, ACT/360) for rolling overnight over ``[start, end)``,
    as known at the end of ``as_of``. Pass ``inputs`` to reuse one day's reads."""
    inputs = financing_inputs(as_of, model, **paths) if inputs is None else inputs
    return fa.financing_rate(inputs, model, FINANCING_MODELS[model], start, end, cusip)


def next_business_day(day) -> pd.Timestamp:
    """The SOFR business day after ``day`` (a T+1 settlement)."""
    day = pd.Timestamp(day).normalize()
    return business_days(day + _ONE_DAY, day + pd.Timedelta(days=10))[0]
