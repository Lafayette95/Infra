"""DTCC swaptions: trade keys across the 2025-11 id format change, the open-interest
ledger (point in time), executions with corrections, normal-vol maths. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.analytics import swap_curve as sc
from infra.analytics import swaptions as sw
from infra.processing import dtcc_swaptions as ds

D = pd.Timestamp


def test_trade_key_links_new_format_ids_by_base_and_keeps_namespaces_apart():
    ids = pd.Series([1174107000, 4896008322000000301, 4896008322000000512, 4896008322], dtype="Int64")
    k = ds.trade_key(ids)
    assert list(k) == ["S1174107000", "L4896008322", "L4896008322", "S4896008322"]  # base != the old id 4896008322


def _rec(trade_id, action, event, day, **terms):
    base = {"file_day": D(day), "diss_id": 0, "trade_id": trade_id, "action": action, "event": event,
            "event_ts": D(day) + pd.Timedelta(hours=12), "executed": D("2026-01-05 15:00"), "expiry": D("2026-12-15"),
            "maturity": D("2036-12-17"), "strike": 4.0, "notional": 100e6, "capped": False, "premium": 1e6,
            "label": "Call", "package": False, "block": False, "platform": "BILT"}
    base.update(terms)
    return base


def test_ledger_partial_unwind_termination_cancel_point_in_time():
    rec = pd.DataFrame([
        _rec("S1", "NEWT", "TRAD", "2026-01-05"),
        _rec("S1", "MODI", "TRAD", "2026-02-02", notional=60e6),   # partial unwind
        _rec("S1", "TERM", "ETRM", "2026-03-02"),
        _rec("S2", "NEWT", "TRAD", "2026-01-05"),
        _rec("S2", "EROR", None, "2026-01-07", notional=None),
        _rec("S3", "MODI", "TRAD", "2026-01-06"),                   # executed before the archive
    ])
    v = ds.ledger_versions(rec)
    def oi(day):
        o = ds.open_at(v, day, max_notional=20e9)
        return dict(zip(o["trade_id"], o["notional"]))
    assert oi("2026-01-05") == {"S1": 100e6, "S2": 100e6}
    assert oi("2026-01-07") == {"S1": 100e6, "S3": 100e6}      # S2 cancelled; S3 seen via its MODI
    assert oi("2026-02-02") == {"S1": 60e6, "S3": 100e6}
    assert oi("2026-03-02") == {"S3": 100e6}
    assert oi("2026-12-15") == {}                               # expired
    assert set(v.loc[v["trade_id"] == "S3", "origin"]) == {"lifecycle"}


def test_executions_apply_corrections_known_by_as_of_only():
    rec = pd.DataFrame([
        _rec("S1", "NEWT", "TRAD", "2026-01-05", premium=1e6),
        _rec("S1", "CORR", None, "2026-01-08", premium=2e6),
        _rec("S2", "NEWT", "TRAD", "2026-01-05"),
        _rec("S2", "EROR", None, "2026-01-06"),
        _rec("S3", "NEWT", "TRAD", "2026-01-20"),                  # disseminated 15 days after execution: not fresh
    ])
    e = ds.executions(rec, as_of="2026-01-06", fresh_days=1)
    assert dict(zip(e["trade_id"], e["premium"])) == {"S1": 1e6}
    e = ds.executions(rec, as_of="2026-01-31", fresh_days=1)
    assert dict(zip(e["trade_id"], e["premium"])) == {"S1": 2e6}


def test_normal_vol_roundtrip_gamma_symmetry_and_forward():
    for payer in (True, False):
        p = sw.normal_price(0.04, 0.0425, 100.0, 1.0, payer)
        assert sw.implied_normal_vol(p, 0.04, 0.0425, 1.0, payer) == pytest.approx(100.0, abs=1e-6)
    assert np.isnan(sw.implied_normal_vol(0.001, 0.04, 0.03, 1.0, True))  # below intrinsic
    # a flat 4% (cc) curve: the forward swap rate ~ the annual ACT/360 equivalent of 4% cc
    t = np.array([1.0, 40.0])
    curve = sc.OisCurve(t, np.exp(-0.04 * t))
    f, a = sw.forward_annuity(curve, D("2026-01-05"), D("2027-01-07"), D("2037-01-07"))
    assert f == pytest.approx((np.exp(0.04) - 1) * 360 / 365.25, abs=2e-4) and 7 < a < 9
