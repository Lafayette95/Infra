"""Synchronized benchmark P&L (infra.processing.sync_pnl, infra.cycle.bmk_sync): the held
contract across a roll, per-DV01 scaling, and moving a yield to the sync instant."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.processing import sync_pnl as sp

D = pd.Timestamp


def test_futures_pnl_follows_the_contract_held_over_the_interval():
    days = pd.bdate_range("2026-09-01", periods=3)
    marks = pd.DataFrame({"SEP": [100.0, 100.5, 101.0], "DEC": [99.0, 99.4, 100.2]}, index=days)
    held = pd.Series(["SEP", "DEC", "DEC"], index=days)          # rolls to DEC on day 2
    dv01 = pd.DataFrame({"SEP": [0.1, 0.1, 0.1], "DEC": [0.1, 0.1, 0.1]}, index=days)
    out = sp.held_futures_pnl(marks, held, dv01, ticker="FUT_X", bmk="fut@T", currency="EUR", point_value=1000.0)
    assert list(out["cusip"]) == ["SEP", "DEC"]                  # day 2 is still SEP's move (held from day 1)
    assert out["pnl_per_dv01"].round(9).tolist() == [5.0, 8.0]   # 0.5 / 0.1, 0.8 / 0.1
    assert out["pnl"].tolist() == pytest.approx([500.0, 800.0])


def test_a_missing_dv01_leaves_pnl_but_no_per_dv01():
    days = pd.bdate_range("2026-09-01", periods=2)
    out = sp.held_futures_pnl(pd.DataFrame({"C": [100.0, 101.0]}, index=days), pd.Series(["C", "C"], index=days),
                              pd.DataFrame(), ticker="FUT_X", bmk="fut@T", currency="USD", point_value=1000.0)
    assert out["pnl"].iloc[0] == pytest.approx(1000.0) and np.isnan(out["pnl_per_dv01"].iloc[0])


def test_moved_yield_uses_the_futures_move_between_the_two_instants():
    # Bund future +0.6 points from 11:15 to 17:15 at -9.5bp per point: the yield falls 5.7bp
    assert sp.moved_yield(3.20, -9.5, 121.1, 120.5) == pytest.approx(3.20 - 0.057)
    assert np.isnan(sp.moved_yield(3.20, np.nan, 121.1, 120.5))


def test_every_sync_family_is_registered_in_bmk_pnl():
    from infra.cycle.bmk import PNL_STEP
    assert {"sync_pnl_present", "sync_pnl_sane"} <= {c.name for c in PNL_STEP.checks}


def test_rolling_beta_is_point_in_time_and_moved_price():
    from infra.processing import sync_pnl as sp
    days = pd.bdate_range("2026-01-01", periods=80)
    x = pd.Series(np.sin(np.arange(80)), index=days)
    y = 2.0 * x
    y.iloc[-1] = 1000.0                                   # a shock on the last day ...
    b = sp.rolling_beta(y, x, window=60, min_obs=40)
    assert b.iloc[-1] == pytest.approx(2.0)               # ... never in that day's own ratio
    assert days[39] not in b.index and days[40] in b.index
    assert sp.moved_price(115.0, 1.1, 112.5, 112.0) == pytest.approx(115.55)
    assert np.isnan(sp.moved_price(115.0, np.nan, 112.5, 112.0))
