"""OTR yield benchmark and the shared bond-yield reader (infra.processing.otr_yields,
infra.pipeline.bond_yields): the same ticker from either source, the bond behind each OTR
row, no row without a price, day replacement. Synthetic stores in temp dirs."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.pipeline import bond_yields as pby
from infra.processing import otr_yields as oy


def _map(rows):
    return pd.DataFrame([{"timestamp": pd.Timestamp(d), "tenor": t, "rank": 0, "convention": "issue", "cusip": c,
                          "issue_date": pd.Timestamp("2026-08-17"), "auction_date": pd.Timestamp("2026-08-12"),
                          "maturity_date": pd.Timestamp("2036-08-15"), "coupon": 4.625} for d, t, c in rows])


def _prices(rows):
    return pd.DataFrame([{"timestamp": pd.Timestamp(d), "cusip": c, "yield_eod": y, "price_eod": 99.5}
                         for d, c, y in rows])


def test_otr_yields_join_rank_zero_to_prices_under_cmt_tickers():
    out = oy.otr_yields(_map([("2026-09-29", "10y", "C"), ("2026-09-30", "10y", "C")]),
                        _prices([("2026-09-29", "C", 4.70)]))  # no price on the 30th
    assert out["ticker"].tolist() == ["US_BOND_10y"] and out["yield"].tolist() == [4.70]
    assert out["cusip"].iloc[0] == "C"


def test_reader_switches_source_by_name_only(tmp_path, monkeypatch):
    monkeypatch.setattr(pby, "read_otr", lambda s, e, rank=0, root=None: _map([("2026-09-29", "10y", "C")]))
    monkeypatch.setattr(pby, "read_prices", lambda s, e, root=None: _prices([("2026-09-29", "C", 4.70)]))
    assert pby.build_otr_yields("2026-09-29", "2026-09-29", root=tmp_path) == 1
    monkeypatch.setattr(pby, "read_bonds_from_disk", lambda t, s, e, root=None: pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-09-29")], "ticker": ["US_BOND_10y"], "par_yield": [4.68]}))
    otr = pby.read_bond_yields(["US_BOND_10y"], "2026-09-29", "2026-09-30", source="otr", otr_root=tmp_path)
    cmt = pby.read_bond_yields(["US_BOND_10y"], "2026-09-29", "2026-09-30", source="cmt")
    assert list(otr.columns) == list(cmt.columns) == pby.YIELD_COLUMNS
    assert otr["yield"].tolist() == [4.70] and cmt["yield"].tolist() == [4.68]
    detailed = pby.read_bond_yields(["US_BOND_10y"], "2026-09-29", "2026-09-30", source="otr", details=True,
                                    otr_root=tmp_path)
    assert detailed["cusip"].tolist() == ["C"] and detailed["price_eod"].tolist() == [99.5]
    with pytest.raises(ValueError):
        pby.read_bond_yields(["US_BOND_10y"], "2026-09-29", "2026-09-30", source="bvals")


def test_rebuilding_a_day_replaces_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pby, "read_otr", lambda s, e, rank=0, root=None: _map([("2026-09-29", "10y", "C")]))
    monkeypatch.setattr(pby, "read_prices", lambda s, e, root=None: _prices([("2026-09-29", "C", 4.70)]))
    pby.build_otr_yields("2026-09-29", "2026-09-29", root=tmp_path)
    monkeypatch.setattr(pby, "read_prices", lambda s, e, root=None: _prices([]))  # price withdrawn
    pby.build_otr_yields("2026-09-29", "2026-09-29", root=tmp_path)
    assert pby.read_otr_yields(["US_BOND_10y"], "2026-09-29", "2026-09-30", root=tmp_path).empty
