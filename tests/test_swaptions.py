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


def test_hedge_tenor_maps_a_tail_to_the_nearest_hedged_swap():
    from infra.pipeline.swaptions import hedge_tenor
    assert [hedge_tenor(x) for x in (1.0, 2.2, 4.0, 9.6, 12.0, 25.0, 30.0)] == [1, 2, 5, 10, 10, 30, 30]


def test_forward_moved_to_the_print_time(monkeypatch):
    """F(print) = F(snap) - the hedge-implied rate move from the print to the snap; the
    moneyness, the OTM side and the vol follow the moved forward."""
    from infra.config import SWAPTIONS
    from infra.pipeline import swap_hedge, swaptions as sp
    day = D("2026-09-15")
    t = np.array([1.0, 40.0])
    curve = sc.OisCurve(t, np.exp(-0.04 * t))
    ex = pd.DataFrame([{"trade_id": "S1", "executed": D("2026-09-15 14:00"), "expiry": D("2026-12-15"),
                        "maturity": D("2036-12-17"), "strike": 4.10, "notional": 100e6, "capped": False,
                        "premium": 1.5e6, "label": "Call", "package": False, "platform": "BILT"}])
    monkeypatch.setattr(swap_hedge, "futures_moves", lambda tr, inst, ccy, d, book: pd.Series(8.0, index=tr.index))
    adj = sp.price_prints(ex, {day: curve}, SWAPTIONS["USD_SOFR"], book=object())
    raw = sp.price_prints(ex, {day: curve}, SWAPTIONS["USD_SOFR"], book=None)
    assert bool(adj["forward_adjusted"].iloc[0]) and not bool(raw["forward_adjusted"].iloc[0])
    assert adj["forward"].iloc[0] == pytest.approx(raw["forward"].iloc[0] - 0.08)  # 8bp lower at the print
    assert adj["moneyness_bp"].iloc[0] == pytest.approx(raw["moneyness_bp"].iloc[0] + 8.0)


def _print(ts, vol, label="Call", strike=4.0, notional=100e6):
    return {"timestamp": D(ts), "trade_id": f"S{ts}{label}{vol}", "expiry_date": D("2026-12-15"), "maturity": D("2036-12-17"),
            "t_years": 0.08, "tenor_years": 10.0, "strike": strike, "forward": 4.0, "moneyness_bp": 0.0, "vol_bp": vol,
            "notional": notional, "premium": 1e6, "label": label, "minutes_from_snap": 10.0, "forward_adjusted": True,
            "straddle_pair": False}


def test_surface_leaves_straddle_pairs_out_and_flags_a_suspect_point():
    from infra.config import SWAPTIONS
    from infra.pipeline import swaptions as sp
    rows = []
    for k, day in enumerate(pd.bdate_range("2026-09-01", periods=6)):
        rows += [_print(f"{day.date()} 14:00", 80.0 + k * 0.1), _print(f"{day.date()} 15:00", 80.2 + k * 0.1)]
    last = pd.bdate_range("2026-09-01", periods=7)[-1]
    rows += [_print(f"{last.date()} 14:00", 160.0), _print(f"{last.date()} 15:00", 161.0)]  # a jump: suspect
    pair = [_print("2026-09-03 14:30", 170.0, "Call", 4.1), _print("2026-09-03 14:30", 170.0, "Put", 4.1)]
    for p in pair:
        p["straddle_pair"] = True
    v = sp.atm_surface(pd.DataFrame(rows + pair), SWAPTIONS["USD_SOFR"])
    v = v[(v["expiry"] == "1m") & (v["tenor"] == 10)].set_index("timestamp")
    assert v.loc[D("2026-09-03"), "vol_bp"] < 81  # the straddle pair's doubled vol is not in the median
    assert bool(v.loc[last, "suspect"]) and not v["suspect"].iloc[:-1].any()
