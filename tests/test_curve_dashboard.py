"""The bond-curve page (/curve): its figure and table on a synthetic view - no stores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.dashboard import curve_charts
from infra.dashboard.curve_callbacks import bond_table


def _view():
    m = np.array([1.9, 2.0, 4.9, 5.0, 9.8, 29.8])
    b = pd.DataFrame({"cusip": list("ABCDEF"), "maturity_years": m, "ytm": 4 + m / 30, "zspread_bp": [0, -1, 0.5, -1, 2, 1],
                      "zspread_loo_bp": [0, -1.2, 0.5, -1.1, 2.2, 1], "carry_curve_bp": 1.0, "rolldown_bp": -0.5,
                      "in_fit": [True, False, True, False, True, False], "coupon": 4.0,
                      "maturity_date": pd.Timestamp("2026-10-01") + pd.to_timedelta(m * 365, unit="D"),
                      "issue_date": pd.Timestamp("2024-01-01"), "original_term": "x", "security_type": "Note",
                      "tenor": [None, "2y", None, "5y", None, "30y"], "rank": [np.nan, 0, np.nan, 0, np.nan, 1],
                      "status": ["other", "on-the-run", "other", "on-the-run", "other", "old"],
                      "baskets": ["ZTZ6 (CF 0.9)", "ZTZ6 (CF 0.9)", "", "", "", "UBZ6 (CF 0.7)"],
                      "ctd_of": ["ZTZ6 60%", "", "", "", "", "UBZ6 51%"]})
    line = pd.DataFrame({"maturity_years": np.arange(0.5, 30.1, 0.5), "par_yield": 4.0})
    sectors = pd.DataFrame({"root": ["ZT", "UB"], "contract": ["ZTZ6", "UBZ6"], "min": [1.7, 25.0], "max": [2.0, 30.0], "size": [2, 1]})
    return {"bonds": b, "line": line, "sectors": sectors, "ctds": pd.DataFrame(), "fit": {"n_fit": 3, "rmse_bp": 1.0}}


def test_curve_figure_has_line_points_and_ctds():
    f = curve_charts.curve_figure(_view())
    names = [t.name for t in f.data]
    assert "Fitted par curve" in names and "CTD" in names and "on-the-run" in names


def test_table_filters_by_basket_sector():
    v = _view()
    rows = bond_table(v["bonds"], "ZTZ6", v["sectors"]).children[1].children
    assert len(rows) == 2 and len(bond_table(v["bonds"], "all", v["sectors"]).children[1].children) == 6


def test_empty_view_is_a_message_not_a_crash():
    assert curve_charts.curve_figure({"bonds": pd.DataFrame()}).layout.annotations
