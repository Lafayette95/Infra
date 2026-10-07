"""Eurex German bond futures: conversion factors (checked against Eurex's own published
values), baskets, contract naming, implied repo, and the bmk DV01 hook. No network."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.config import EUREX_BOND_FUTURES
from infra.processing import eurex_baskets as eb

D = pd.Timestamp


@pytest.mark.parametrize("root, coupon, maturity, contract_month, commencement, eurex_cf", [
    ("FGBL", 2.6, "2035-08-15", (2026, 12), None, 0.774902),            # regular
    ("FGBM", 0.0, "2031-08-15", (2026, 12), None, 0.761347),            # zero coupon
    ("FGBL", 3.0, "2036-08-15", (2026, 12), "2026-07-10", 0.784157),    # long first coupon (new 10y)
    ("FGBS", 2.7, "2028-09-13", (2026, 12), "2026-07-16", 0.946091),    # long first coupon (new Schatz)
    ("FGBX", 2.9, "2056-08-15", (2027, 3), None, 0.811555),             # Buxl: 4% notional coupon
])
def test_conversion_factor_matches_eurex(root, coupon, maturity, contract_month, commencement, eurex_cf):
    delivery = eb.delivery_day(*contract_month)
    cf = eb.conversion_factor(coupon, D(maturity), delivery, EUREX_BOND_FUTURES[root].notional_pct,
                              commencement=None if commencement is None else D(commencement))
    assert cf == pytest.approx(eurex_cf, abs=5e-7)


def test_contracts_are_named_like_databento_and_roll_after_the_last_trading_day():
    c = eb.listed_contracts("FGBL", D("2026-12-08"))
    assert [x[0] for x in c] == ["FGBL SI 20261208 PS", "FGBL SI 20270308 PS", "FGBL SI 20270608 PS"]
    assert eb.listed_contracts("FGBL", D("2026-12-09"))[0][0] == "FGBL SI 20270308 PS"
    assert eb.delivery_day(2027, 3) == D("2027-03-10") and eb.delivery_day(2026, 10) == D("2026-10-12")  # Sat -> Mon


def test_basket_rules_remaining_and_original_term_volume_and_issue_day():
    sec = pd.DataFrame({
        "isin": ["NEW10", "OLD30", "SMALL", "GREEN", "LATE"],
        "type": ["Bund", "Bund", "Bund", "Green", "Bund"],
        "maturity_date": [D("2036-08-15"), D("2036-01-04"), D("2036-02-15"), D("2035-08-15"), D("2036-11-15")],
        "issue_date": [D("2026-07-10"), D("2005-01-05"), D("2026-01-09"), D("2025-06-01"), D("2026-12-01")],
        "bill": False, "inflation_linked": False})
    vol = pd.Series({"NEW10": 9000, "OLD30": 20000, "SMALL": 3000, "GREEN": 6000, "LATE": 9000})
    b = eb.basket(sec, vol, eb.delivery_day(2026, 12), EUREX_BOND_FUTURES["FGBL"], as_of=D("2026-11-20"))
    # the old 30y has 9 years left but a 31-year original term; SMALL is under 5bn; LATE isn't issued yet
    assert sorted(b["isin"]) == ["GREEN", "NEW10"]


def test_implied_repo_and_accrued_at_delivery():
    acc = eb.accrued_at(2.6, D("2035-08-15"), D("2026-12-10"))
    assert acc == pytest.approx(2.6 * (D("2026-12-10") - D("2026-08-15")).days / 365, abs=1e-12)
    # buy at 100 dirty, deliver 90 days later for 100.5 invoice: 2% a year ACT/360
    assert eb.implied_repo(100.0, D("2026-09-11"), D("2026-12-10"), 100.5, 1.0, 0.0, 0.0) == pytest.approx(2.0)


def test_bmk_registers_the_eurex_dv01_hook_for_the_german_roots():
    from infra.cycle import bmk
    assert bmk.EUREX_DV01_ROOTS == ("FGBS", "FGBM", "FGBL", "FGBX") and callable(bmk.EUREX_FUTURES_DV01)
