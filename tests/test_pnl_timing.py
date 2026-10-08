"""Lag-aware joins (infra.pipeline.pnl_timing): mark times per bmk / issuer, each P&L row
assigned to the latest decision at or before its interval START, and per-issuer availability."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.pipeline import pnl_timing as pt
from infra.pipeline.series_panel import available_at

D = pd.Timestamp


def test_mark_specs():
    assert pt.mark_spec("yield_mof", "JP_BOND_10y") == ("15:00", "Asia/Tokyo")
    assert pt.mark_spec("yield_otr", "DE_BOND_10y") == ("11:15", "Europe/Berlin")
    assert pt.mark_spec("fut@LDN1615", "FUT_ZN") == ("16:15", "Europe/London")
    assert pt.mark_spec("futures_price", "FGBL SI 20261208 PS") == ("17:15", "Europe/Berlin")
    assert pt.mark_spec("futures_price", "ZNZ6") == ("14:00", "America/Chicago")
    with pytest.raises(KeyError):
        pt.mark_spec("yield_mof", "US_BOND_10y")


def test_a_row_belongs_to_the_decision_before_its_interval_started():
    # JGB rows: Tokyo close to Tokyo close (06:00 UTC); decisions at the US close (19:30 UTC)
    rows = pd.DataFrame({"timestamp": [D("2026-09-30"), D("2026-10-01")],
                         "prev_mark_at": [D("2026-09-29 06:00"), D("2026-09-30 06:00")],
                         "mark_at": [D("2026-09-30 06:00"), D("2026-10-01 06:00")],
                         "pnl_per_dv01": [2.0, -3.0], "pnl": [2.0, -3.0]})
    dec = pd.Series([1.0, -1.0], index=[D("2026-09-28 19:30"), D("2026-09-29 19:30")])
    e = pt.earned(rows, dec)
    # the 09-30 row started at 09-29 06:00, BEFORE the 09-29 decision: it is the 09-28 position's
    assert list(e["decision_at"]) == [D("2026-09-28 19:30"), D("2026-09-29 19:30")]
    assert list(e["earned"]) == [2.0, 3.0]


def test_rows_before_the_first_decision_earn_nothing():
    rows = pd.DataFrame({"timestamp": [D("2026-09-29")], "prev_mark_at": [D("2026-09-28 06:00")],
                         "mark_at": [D("2026-09-29 06:00")], "pnl_per_dv01": [5.0], "pnl": [5.0]})
    e = pt.earned(rows, pd.Series([1.0], index=[D("2026-09-29 19:30")]))
    assert e["position"].iloc[0] == 0.0 and pd.isna(e["decision_at"].iloc[0])


def test_issuer_specific_availability():
    fri = [D("2026-10-02")]
    assert available_at("bmk:mof:JP_BOND_10y", fri)[0] == D("2026-10-05 00:30")      # Monday 09:30 Tokyo
    assert available_at("bmk:fut@LDN1615:FUT_FGBL", fri)[0] == D("2026-10-02 15:16")  # at the snap (BST)
    assert available_at("bmk:ois@LDN1615:EUR_OIS_10y", fri)[0] == D("2026-10-02 19:15")
