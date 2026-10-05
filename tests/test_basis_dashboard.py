"""The basis page (/basis): its store, figures and bond table on synthetic data - no
model run, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.dashboard import basis_charts
from infra.dashboard.basis_callbacks import bond_table
from infra.storage import basis_runs

D = pd.Timestamp


def _run():
    days = pd.bdate_range("2026-09-01", periods=4)
    c = pd.DataFrame({"day": days, "root": "ZN", "contract": "ZNZ6", "ctd": "A", "delivery_kind": "last",
                      "delivery": D("2026-12-31"), "option_value_model_32": [1.0, 1.1, 1.2, 1.3],
                      "option_value_obs_32": [0.5, 0.6, 0.4, 0.7], "quality_macro_32": 0.6, "quality_spread_32": 0.2,
                      "wildcard_32": 0.2, "eom_32": [0.0, 0.1, 0.2, 0.3]})
    b = pd.concat([pd.DataFrame({"day": days, "root": "ZN", "contract": "ZNZ6", "cusip": cu, "delivery_kind": "last",
                                 "coupon": cpn, "maturity": D("2033-11-15"), "cf": 0.85, "price": 99.0, "fwd": 99.2,
                                 "implied_futures": f, "gross_basis": 0.3, "carry": 0.1, "net_basis": 0.2, "irr": 4.1,
                                 "repo": 4.2, "prob": p}) for cu, cpn, f, p in (("A", 4.5, 112.0, 0.8), ("B", 4.0, 112.1, 0.2))])
    return c, b


def test_store_round_trip_and_rerun_replaces_days(tmp_path):
    c, b = _run()
    basis_runs.save("M2T", c, b, root=tmp_path)
    basis_runs.save("M2T", c.iloc[:1].assign(option_value_model_32=9.9), b[b["day"] == c["day"].iloc[0]], root=tmp_path)
    back = basis_runs.read("M2T", "contracts", root=tmp_path)
    assert len(back) == 4 and back["option_value_model_32"].iloc[0] == 9.9
    assert basis_runs.models(root=tmp_path) == ["M2T"]
    assert len(basis_runs.read("M2T", "bonds", contracts=["ZNZ6"], root=tmp_path)) == 8


def test_figures_and_table_render():
    c, b = _run()
    assert len(basis_charts.optionality_figure(c, part="quality").data) == 2  # macro + spread
    assert len(basis_charts.optionality_figure(c, part="timing").data) == 4  # wild card + EOM + model + observed
    sw = c.assign(delivery_kind=["first", "first", "last", "last"])
    assert basis_charts._switch_days(sw) == [sw["day"].iloc[2]]
    assert len(basis_charts.probability_figure(b).data) == 2
    assert len(basis_charts.net_basis_figure(b, c).data) == 3  # two bonds + the model option value
    t = bond_table(b[b["day"] == b["day"].iloc[0]], pd.DataFrame())
    rows = t.children[1].children
    assert len(rows) == 2 and rows[0].children[0].children.endswith("(A)")  # most likely CTD first


def test_page_registers():
    from infra.dashboard.app import create_app
    import dash
    create_app()
    assert any(p["path"] == "/basis" for p in dash.page_registry.values())
