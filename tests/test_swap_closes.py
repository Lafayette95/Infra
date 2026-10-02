"""Benchmark swap closes (infra.processing.swap_closes): window selection and fallback,
time weighting, outlier rejection, and no invented values. Synthetic trades; no files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.config import SWAP_CLOSE_WEIGHTING, SwapCloseSpec
from infra.processing.swap_closes import estimate, pure_closes, weighted_median

SNAP = pd.Timestamp("2026-09-29 19:30")  # 15:30 New York, as UTC
# explicit windows: these tests exercise the window MECHANISM, not the calibrated defaults
SPEC = SwapCloseSpec("15:30", "America/New_York", ("USD",), "test", "test",
                     pure_half_window_min=15, pure_fallback_half_window_min=30, adjusted_half_window_min=90)


def _trades(rows):
    """rows: (minutes from the snap, tenor, rate %)."""
    return pd.DataFrame({"executed": [SNAP + pd.Timedelta(minutes=m) for m, _, _ in rows],
                         "tenor": [t for _, t, _ in rows], "rate": [r for _, _, r in rows]})


def _closes(rows):
    return pure_closes(_trades(rows), SNAP, SPEC, SWAP_CLOSE_WEIGHTING, close_name="NY1530", currency="USD")


def test_weighted_median():
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])) == 2.0
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([10.0, 1.0, 1.0])) == 1.0


def test_only_prints_inside_the_window_count():
    out = _closes([(-10, 10, 4.80), (5, 10, 4.81), (12, 10, 4.82), (-60, 10, 5.50), (45, 10, 3.00)])
    row = out.iloc[0]
    assert row["half_window_min"] == 15 and row["n_trades"] == 3 and row["rate"] == 4.81


def test_the_window_widens_when_too_few_prints():
    out = _closes([(-2, 10, 4.80), (25, 10, 4.83), (-28, 10, 4.79)])  # one inside +-15, three inside +-30
    assert out.iloc[0]["half_window_min"] == 30 and out.iloc[0]["n_trades"] == 3


def test_a_tenor_with_nothing_near_the_snap_gets_no_row():
    out = _closes([(-2, 10, 4.80), (-50, 2, 4.70)])
    assert out["tenor"].tolist() == [10]


def test_prints_nearer_the_snap_count_more():
    # two prints at the snap on one side, three far out on the other: the near ones win
    out = _closes([(0, 10, 4.80), (1, 10, 4.80), (-29, 10, 4.85), (-28, 10, 4.85), (29, 10, 4.85)])
    assert out.iloc[0]["half_window_min"] == 15 or out.iloc[0]["rate"] == 4.80


def test_an_off_market_print_is_rejected():
    rows = [(m, 10, 4.80 + 0.001 * i) for i, m in enumerate([-10, -5, 0, 5, 10])] + [(1, 10, 4.95)]
    row = _closes(rows).iloc[0]
    assert row["n_trades"] == 5 and abs(row["rate"] - 4.802) < 1e-9


def test_a_lone_print_never_claims_less_than_the_print_noise():
    est = estimate(np.array([4.8]), np.array([0.0]), SWAP_CLOSE_WEIGHTING)
    assert est["n_trades"] == 1 and est["se_bp"] >= 1.2533 * SWAP_CLOSE_WEIGHTING.trade_noise_bp - 1e-12


# ------------------------------------------------------------------ pipeline (fake archive)
def _report_row(diss_id, executed, rate, *, action="NEWT", event="TRAD", orig=None, event_ts=None,
                maturity="2036-10-01"):
    return {"Dissemination Identifier": diss_id, "Original Dissemination Identifier": orig, "Action type": action,
            "Event type": event, "Event timestamp": event_ts or executed, "Execution Timestamp": executed,
            "Effective Date": "2026-10-01", "Expiration Date": maturity, "Fixed rate-Leg 1": rate,
            "Fixed rate-Leg 2": None, "Notional amount-Leg 1": "100,000,000", "Notional currency-Leg 1": "USD",
            "Cleared": "I", "Platform identifier": "TWSF", "Block trade election indicator": "False",
            "Package indicator": "False", "Non-standardized term indicator": "False",
            "Fixed rate payment frequency period-Leg 1": "YEAR", "Fixed rate payment frequency period-Leg 2": None,
            "UPI FISN": "NA/Swap OIS USD", "UPI Underlier Name": "USD-SOFR-OIS Compound"}


def _archive(root, day, rows):
    import io
    import zipfile

    from infra.pipeline import dtcc
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", pd.DataFrame(rows).to_csv(index=False))
    dtcc.store_dtcc_raw("RATES", day, buf.getvalue(), root=root)


def test_pipeline_applies_a_later_files_cancellation_and_replaces_the_day(tmp_path):
    from infra.config import SWAP_CLOSES
    from infra.pipeline import swap_closes as psc
    dtcc_root, root = tmp_path / "DTCC", tmp_path / "SwapCloses"
    # 15:30 New York on 2026-09-29 = 19:30 UTC
    _archive(dtcc_root, "2026-09-29", [_report_row("1", "2026-09-29T19:28:00Z", "0.0480"),
                                       _report_row("2", "2026-09-29T19:31:00Z", "0.0482"),
                                       _report_row("3", "2026-09-29T19:33:00Z", "0.0481")])
    _archive(dtcc_root, "2026-09-30", [_report_row("9", "2026-09-29T19:33:00Z", "0.0481", action="EROR",
                                                   event=None, orig="3", event_ts="2026-09-30T08:00:00Z")])
    closes = {"NY1530": SWAP_CLOSES["NY1530"]}
    from infra.pipeline.swap_hedge import HedgeBook
    before = psc.compute_closes("2026-09-29", closes=closes, as_of="2026-09-29 23:59", dtcc_root=dtcc_root,
                                book=HedgeBook())
    after = psc.compute_closes("2026-09-29", closes=closes, dtcc_root=dtcc_root, book=HedgeBook())
    assert before.iloc[0]["n_trades"] == 3 and after.iloc[0]["n_trades"] == 2  # cancelled next day

    psc.store_closes("2026-09-29", after, root=root)
    psc.store_closes("2026-09-29", after.iloc[0:0], root=root)  # recomputed with nothing: day emptied
    assert psc.read_swap_closes("2026-09-29", "2026-09-30", root=root).empty


def test_backfill_stores_every_archived_day_once(tmp_path):
    from infra.pipeline import swap_closes as psc
    dtcc_root, root = tmp_path / "DTCC", tmp_path / "SwapCloses"
    for day, ts in [("2026-09-28", "2026-09-28T19:30:00Z"), ("2026-09-29", "2026-09-29T19:30:00Z")]:
        _archive(dtcc_root, day, [_report_row(day, ts, "0.0480")])
    empty = tmp_path / "empty"
    out = psc.backfill_swap_closes("2026-09-27", "2026-09-30", root=root, dtcc_root=dtcc_root,
                                   hedge_paths=dict(daily_root=empty / "d", bonds_root=empty / "b", bbo_root=empty / "q",
                                                    contracts_file=empty / "c.parquet"))
    assert out["days"] == 2
    stored = psc.read_swap_closes("2026-09-28", "2026-09-30", close="NY1530", root=root)
    assert stored["timestamp"].tolist() == [pd.Timestamp("2026-09-28 19:30"), pd.Timestamp("2026-09-29 19:30")]
    assert set(stored["tenor"]) == {10}


def test_every_configured_close_window_fits_one_utc_file():
    from infra.config import SWAP_CLOSES
    from infra.pipeline.swap_closes import _window_inside_day
    from infra.trading_calendar import snap_instants
    for day in pd.date_range("2026-01-01", "2026-12-31"):
        for spec in SWAP_CLOSES.values():
            _window_inside_day(snap_instants([day], spec.local_time, spec.timezone)[0], spec, day)



# ------------------------------------------------------------------ futures-adjusted
def test_hedge_ratio_is_known_before_its_day_and_never_spans_a_roll():
    from infra.processing.swap_hedge import hedge_ratios, same_contract_changes
    days = pd.date_range("2026-01-01", periods=6)
    st = pd.DataFrame({"A": [100.0, 101.0, 100.5, 101.5, np.nan, np.nan],
                       "B": [np.nan, np.nan, np.nan, 99.0, 98.0, 99.0]}, index=days)
    contract = pd.Series(["A", "A", "A", "A", "B", "B"], index=days)
    dp = same_contract_changes(st, contract)
    # the roll day (index 4) moves to B, measured B against B's own prior settlement (99 -> 98),
    # never B against A's (101.5)
    assert dp.tolist() == [1.0, -0.5, 1.0, -1.0, 1.0]
    dy = -10.0 * dp  # 10bp per point
    ratio = hedge_ratios(dp, dy, window=2)
    assert ratio.index[0] == dp.index[2] and np.allclose(ratio, -10.0)  # first full window, used the day after


def test_mid_at_uses_the_quote_known_at_the_time():
    from infra.processing.swap_hedge import mid_at
    q = pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-29 19:00", "2026-09-29 19:01"]), "mid": [110.0, 110.5]})
    out = mid_at(q, pd.to_datetime(["2026-09-29 18:59:30", "2026-09-29 19:00:59", "2026-09-29 19:05:00"]))
    assert np.isnan(out[0]) and out[1] == 110.0 and out[2] == 110.5


def test_adjusted_close_moves_prints_to_the_snap():
    from infra.processing.swap_closes import adjusted_closes
    # the market rallied 5bp over the hour before the snap; prints taken along the way
    # are all 4.80 + drift, and the futures moves bring each back to the snap's level
    rows = [(-60, 10, 4.85, -5.0), (-30, 10, 4.825, -2.5), (-5, 10, 4.8, 0.0), (20, 10, 4.80, 0.0)]
    t = pd.DataFrame({"executed": [SNAP + pd.Timedelta(minutes=m) for m, *_ in rows], "tenor": [r[1] for r in rows],
                      "rate": [r[2] for r in rows], "futures_move_bp": [r[3] for r in rows]})
    out = adjusted_closes(t, SNAP, SPEC, SWAP_CLOSE_WEIGHTING, close_name="NY1530", currency="USD").iloc[0]
    assert out["method"] == "adjusted" and out["n_trades"] == 4 and abs(out["rate"] - 4.80) < 1e-9
    assert out["half_window_min"] == SPEC.adjusted_half_window_min
