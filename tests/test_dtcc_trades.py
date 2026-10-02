"""DTCC report parsing (infra.processing.dtcc_trades): typed columns, trade identity across
corrections and cancellations, and the plain spot par-swap filter. Synthetic rows in the
real report layout; no files."""
from __future__ import annotations

import pandas as pd

from infra.config import SWAP_CURVES
from infra.processing import dtcc_trades as dt

USD = SWAP_CURVES["USD"]


def _row(diss_id, action="NEWT", event="TRAD", orig=None, executed="2026-09-29T18:40:00Z", event_ts=None,
         effective="2026-10-01", maturity="2036-10-01", rate="0.048795", notional="100,000,000",
         package="False", cleared="I", freq="YEAR", fisn="NA/Swap OIS USD", underlier="USD-SOFR-OIS Compound"):
    return {"Dissemination Identifier": diss_id, "Original Dissemination Identifier": orig, "Action type": action,
            "Event type": event, "Event timestamp": event_ts or executed, "Execution Timestamp": executed,
            "Effective Date": effective, "Expiration Date": maturity, "Fixed rate-Leg 1": rate,
            "Fixed rate-Leg 2": None, "Notional amount-Leg 1": notional, "Notional currency-Leg 1": "USD",
            "Cleared": cleared, "Platform identifier": "TWSF", "Block trade election indicator": "False",
            "Package indicator": package, "Non-standardized term indicator": "False",
            "Fixed rate payment frequency period-Leg 1": freq, "Fixed rate payment frequency period-Leg 2": None,
            "UPI FISN": fisn, "UPI Underlier Name": underlier}


def _par(rows, as_of=None):
    df = dt.normalize(pd.DataFrame(rows))
    return dt.par_trades(dt.current_trades(dt.trade_events(dt.product_rows(df, USD)), as_of), USD, "USD")


def test_normalize_types_rates_in_percent_and_capped_notionals():
    df = dt.normalize(pd.DataFrame([_row("1", notional="250,000,000+"), _row("2", rate=None)]))
    assert df.loc[0, "rate"] == 4.8795 and df.loc[0, "notional"] == 250e6 and bool(df.loc[0, "notional_capped"])
    assert df.loc[0, "executed"] == pd.Timestamp("2026-09-29 18:40")  # tz-naive UTC
    assert pd.isna(df.loc[1, "rate"])


def test_rate_on_leg_two_is_used_when_leg_one_is_empty():
    row = _row("1", rate=None)
    row["Fixed rate-Leg 2"] = "0.0475"
    assert dt.normalize(pd.DataFrame([row])).loc[0, "rate"] == 4.75


def test_a_plain_spot_ten_year_is_kept_with_its_tenor():
    out = _par([_row("1")])
    assert out[["tenor", "rate"]].values.tolist() == [[10, 4.8795]]


def test_a_cancelled_trade_is_dropped_and_a_corrected_one_replaced():
    rows = [_row("1"), _row("9", action="EROR", event=None, orig="1", event_ts="2026-09-29T19:00:00Z"),
            _row("2", rate="0.0490"),
            _row("8", action="CORR", event=None, orig="2", rate="0.04881", event_ts="2026-09-29T19:05:00Z")]
    out = _par(rows)
    assert out["trade_id"].tolist() == ["2"] and out["rate"].tolist() == [4.881]


def test_corrections_respect_the_as_of_cutoff():
    rows = [_row("1"), _row("9", action="EROR", event=None, orig="1", event_ts="2026-10-02T12:00:00Z")]
    assert len(_par(rows, as_of="2026-09-30")) == 1  # the cancellation wasn't known yet
    assert len(_par(rows)) == 0


def test_lifecycle_and_clearing_records_are_not_trades():
    rows = [_row("1", event="CLRG"), _row("2", event="COMP"), _row("3", action="MODI", orig="4"),
            _row("5", action="CORR", event=None, orig="77")]  # a correction of a trade we never saw executed
    assert _par(rows).empty


def test_packages_forward_starts_odd_dates_uncleared_and_other_products_are_excluded():
    rows = [_row("1", package="True"),
            _row("2", effective="2026-12-16", maturity="2036-12-16"),  # IMM-dated forward start
            _row("3", maturity="2036-07-15"),  # broken date
            _row("4", cleared="N"),
            _row("5", freq="MNTH"),
            _row("6", fisn="NA/Swap Fxd Flt USD", underlier="USD-SOFR CME Term"),
            _row("7")]
    assert _par(rows)["trade_id"].tolist() == ["7"]


def test_spot_allows_a_holiday_but_not_a_week():
    ok = _row("1", effective="2026-10-02", maturity="2036-10-02")  # spot + 1 day (a holiday)
    late = _row("2", effective="2026-10-08", maturity="2036-10-08")
    assert _par([ok, late])["trade_id"].tolist() == ["1"]


def test_an_upfront_fee_swap_is_not_a_par_trade():
    row = _row("1", rate="0.035")
    row["Other payment amount"] = "1225890.00"
    assert _par([row, _row("2")])["trade_id"].tolist() == ["2"]


def test_an_off_market_coupon_is_dropped_against_its_tenors_market():
    # eight par prints around 4.88%, one 2.50% coupon in the middle of them
    rows = [_row(str(i), executed=f"2026-09-29T19:{10 + 2 * i:02d}:00Z", rate=f"{0.0488 + 0.00001 * i:.6f}")
            for i in range(8)]
    rows.append(_row("x", executed="2026-09-29T19:17:00Z", rate="0.025"))
    out = _par(rows)
    assert "x" not in out["trade_id"].tolist() and len(out) == 8


def test_a_thin_tenor_is_judged_against_its_neighbours():
    rows = [_row(f"a{i}", maturity="2031-10-01", rate="0.0480") for i in range(3)]  # 5y
    rows += [_row(f"b{i}", rate="0.0490") for i in range(3)]  # 10y
    rows += [_row("lone_ok", maturity="2033-10-01", rate="0.0484"),  # 7y, near the 5y-10y line
             _row("lone_bad", maturity="2033-10-01", rate="3.823")]  # 7y at 382.3%
    kept = set(_par(rows)["trade_id"])
    assert "lone_ok" in kept and "lone_bad" not in kept


def test_excluded_platforms_are_not_market_prints():
    bilateral = _row("1")
    bilateral["Platform identifier"] = "BILT"
    assert _par([bilateral, _row("2")])["trade_id"].tolist() == ["2"]
