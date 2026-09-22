"""Exchange trading-day bucketing (infra.trading_calendar). No Dash/API needed."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.config import FUTURES_ROOTS, TRADING_HOURS
from infra.trading_calendar import trading_day

D = pd.Timestamp


def test_cme_same_day_before_close():
    # 20:00 UTC = 15:00 CT (CDT, UTC-5 in September) - before the 16:00 CT close
    idx = pd.DatetimeIndex(["2026-09-21 20:00"])
    assert list(trading_day(idx, "GLBX.MDP3")) == [D("2026-09-21")]


def test_cme_bumps_to_next_day_after_close():
    # 23:00 UTC = 18:00 CT - after the 16:00 CT close, so this is tomorrow's trade date
    idx = pd.DatetimeIndex(["2026-09-21 23:00"])
    assert list(trading_day(idx, "GLBX.MDP3")) == [D("2026-09-22")]


def test_cme_sunday_reopen_is_monday_trade_date():
    """CME's own documented example: Sunday evening's session is trade date Monday."""
    # 2026-09-20 is a Sunday; 22:30 UTC = 17:30 CT, just after the 17:00 CT reopen.
    idx = pd.DatetimeIndex(["2026-09-20 22:30"])
    assert list(trading_day(idx, "GLBX.MDP3")) == [D("2026-09-21")]  # Monday


def test_cme_respects_dst_transition():
    # US DST starts 2026-03-08: 22:00 UTC is 16:00 CT (CST) on Mar 7, but 17:00 CT
    # (CDT) on Mar 9 - the wall-clock close time is what matters, not a fixed UTC offset.
    idx = pd.DatetimeIndex(["2026-03-07 22:00", "2026-03-09 22:00"])
    assert list(trading_day(idx, "GLBX.MDP3")) == [D("2026-03-08"), D("2026-03-10")]


def test_eurex_and_ice_never_cross_midnight():
    # Session hours (02:10-22:00 CET, 01:00-21:00 London) are entirely inside one local
    # calendar day; a bar near either edge still lands on that same local date.
    eurex = pd.DatetimeIndex(["2026-09-21 00:30", "2026-09-21 20:30"])  # ~02:30/22:30 CEST
    assert list(trading_day(eurex, "XEUR.EOBI")) == [D("2026-09-21")] * 2
    ice = pd.DatetimeIndex(["2026-09-21 00:15", "2026-09-21 19:45"])  # ~01:15/20:45 BST
    assert list(trading_day(ice, "IFLL.IMPACT")) == [D("2026-09-21")] * 2


def test_empty_index():
    out = trading_day(pd.DatetimeIndex([]), "GLBX.MDP3")
    assert len(out) == 0


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        trading_day(pd.DatetimeIndex(["2026-09-21"]), "NOPE.DATASET")


def test_every_configured_root_has_a_trading_session():
    for cfg in FUTURES_ROOTS.values():
        assert cfg.dataset in TRADING_HOURS, f"{cfg.root} ({cfg.dataset}) has no TradingSession"


def test_every_trading_session_resolves_without_error():
    idx = pd.DatetimeIndex(["2026-09-21 12:00"])
    for dataset in TRADING_HOURS:
        trading_day(idx, dataset)  # must not raise
