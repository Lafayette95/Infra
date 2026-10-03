"""Near-the-money option selection for implied vol (Rule 2.3's local filter) and the
snapshot grid - pure, no network."""
from __future__ import annotations

import pandas as pd

from infra.pipeline.futures_options_iv import snapshot_days
from infra.processing.option_selection import fetch_ranges, select_near_the_money, strike_scale

D = pd.Timestamp


def _defs(scale=1.0):
    rows, i = [], 0
    for expiry, und in ((D("2024-10-25"), "ZNZ4"), (D("2024-11-22"), "ZNZ4"), (D("2024-12-27"), "ZNH5")):
        for k in [110 + 0.5 * j for j in range(13)]:  # 110 .. 116
            for t in "CP":
                i += 1
                rows.append((i, und, t, k * scale, expiry))
    return pd.DataFrame(rows, columns=["instrument_id", "underlying", "option_type", "strike", "expiry"])


def _px(day, znz4=113.1, znh5=113.4):
    return pd.DataFrame([(D(day), "ZNZ4", znz4), (D(day), "ZNH5", znh5)], columns=["timestamp", "ticker", "price"])


def test_strike_scale_is_inferred():
    assert strike_scale(pd.Series([11300.0]), pd.Series([113.1])) == 100.0
    assert strike_scale(pd.Series([113.0]), pd.Series([113.1])) == 1.0


def test_two_nearest_expiries_atm_plus_two_each_side_otm_only():
    sel = select_near_the_money(_defs(), _px("2024-10-01"))
    assert sorted(sel["expiry"].unique()) == [D("2024-10-25"), D("2024-11-22")]
    one = sel[sel["expiry"] == D("2024-10-25")]
    assert one["atm_strike"].iloc[0] == 113.0
    assert sorted(one.loc[one.option_type == "P", "strike"]) == [112.0, 112.5, 113.0]
    assert sorted(one.loc[one.option_type == "C", "strike"]) == [113.0, 113.5, 114.0]


def test_an_expiry_too_close_is_skipped_and_scale_does_not_matter():
    sel = select_near_the_money(_defs(scale=100.0), _px("2024-10-22"))  # 3 days before the October expiry
    assert sorted(sel["expiry"].unique()) == [D("2024-11-22"), D("2024-12-27")]
    assert set(sel.loc[sel.expiry == D("2024-12-27"), "underlying"]) == {"ZNH5"}
    assert sel["atm_strike"].iloc[0] == 11300.0


def test_each_instrument_gets_one_contiguous_range():
    px = pd.concat([_px("2024-10-01"), _px("2024-10-02", znz4=113.6)])
    r = fetch_ranges(select_near_the_money(_defs(), px)).set_index("instrument_id")
    assert (r["start"] <= r["end"]).all() and r["days"].max() == 2


def test_snapshot_grid_brackets_the_range_and_never_reaches_the_future():
    days = snapshot_days("2024-10-01", "2024-12-31", 30, today="2026-10-02")
    assert days[0] <= D("2024-10-01") and days[-1] > D("2024-12-31")
    assert all(b - a == pd.Timedelta(days=30) for a, b in zip(days, days[1:]))
    assert snapshot_days("2026-09-01", "2026-10-01", 30, today="2026-10-03")[-1] == D("2026-10-01")
