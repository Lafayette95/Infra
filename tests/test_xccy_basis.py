"""Cross-currency basis prints: units by notation, the non-USD leg, the USD-leg sign
reference, the suspect flags. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.processing import dtcc_xccy as dx

D = pd.Timestamp


def _row(i, spread1, notation1="3", cur1="EUR", tenor_years=5, spread2=None, notation2="3", at="2026-09-15T12:00:00Z"):
    eff = D("2026-09-17")
    return {"Dissemination Identifier": str(1000 + i), "Original Dissemination Identifier": None, "Action type": "NEWT",
            "Event type": "TRAD", "Event timestamp": at, "Execution Timestamp": at, "Effective Date": str(eff.date()),
            "Expiration Date": str((eff + pd.DateOffset(years=tenor_years)).date()), "Notional amount-Leg 1": "100,000,000",
            "Notional currency-Leg 1": cur1, "Notional currency-Leg 2": "USD" if cur1 != "USD" else "EUR",
            "Spread-Leg 1": spread1, "Spread-Leg 2": spread2, "Spread notation-Leg 1": notation1,
            "Spread notation-Leg 2": notation2, "Cleared": "N", "Platform identifier": "BILT",
            "Block trade election indicator": "False", "Package indicator": "False", "Non-standardized term indicator": "False",
            "UPI FISN": "NA/Swap Flt Flt EUR USD", "UPI Underlier Name": "EUR-EuroSTR-OIS Compound vs USD-SOFR-OIS Compound"}


def test_units_by_notation_and_percent_mislabels():
    df = dx.normalize(pd.DataFrame([_row(1, "-0.000300"), _row(2, "-3", notation1="4"), _row(3, "-0.03")]))
    assert df["spread_leg1"].round(6).tolist() == [-3.0, -3.0, -3.0]  # decimal, bp, a "decimal" that is a percent


def test_basis_trades_resolve_the_usd_leg_sign_against_the_day():
    from infra.config import XCCY_BASIS
    rows = [_row(i, "-0.000300", tenor_years=5) for i in range(3)]
    rows.append(_row(9, "-0.000310", cur1="USD", tenor_years=5))  # -3.1bp reported on a USD-labelled leg
    t = dx.basis_trades(dx.normalize(pd.DataFrame(rows)), XCCY_BASIS["EURUSD"])
    assert len(t) == 4 and t["basis_bp"].round(2).tolist().count(-3.1) == 1  # kept as -3.1 (closer than +3.1)


def test_suspect_flag_uses_previous_closes_only():
    from infra.pipeline import xccy_basis as xb
    days = pd.bdate_range("2026-09-01", periods=8)
    v = [-35.0, -34.5, -35.2, -34.8, -35.1, -34.9, 33.0, -35.0]
    df = pd.DataFrame({"timestamp": days, "close": "NY1600", "pair": "USDJPY", "tenor": 12, "method": "pure", "basis_bp": v})
    f = xb.flag_suspect(df)
    assert f["suspect"].tolist() == [False] * 6 + [True, False]


def test_xccy_metric_registered():
    from infra.cycle import derived
    assert "xccy_basis_closes" in derived.DERIVED_METRICS
