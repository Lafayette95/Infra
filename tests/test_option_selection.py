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


def test_reused_instrument_ids_only_count_inside_their_validity_window():
    """CME reuses instrument ids: statistics outside an option's window belong to the id's
    previous owner and must not land under the option's key (found 2026-10-03)."""
    from infra.processing.statistics import clean_daily_option_statistics
    defs = pd.DataFrame({"instrument_id": [7, 7], "underlying": ["ZNM0", "ZNU0"], "option_type": ["P", "C"],
                         "strike": [137.75, 140.0], "expiry": [D("2020-04-24"), D("2020-08-21")],
                         "valid_from": [D("2020-04-17"), D("2020-06-01")], "valid_to": [D("2020-04-24"), D("2020-08-21")]})
    raw = pd.DataFrame({"instrument_id": [7, 7, 7], "stat_type": [3, 3, 3], "price": [114.0, 0.25, 1.5],
                        "quantity": [0, 0, 0],
                        "ts_ref": pd.to_datetime(["2020-03-09", "2020-04-20", "2020-06-15"], utc=True),
                        "ts_recv": pd.to_datetime(["2020-03-09 21:00", "2020-04-20 21:00", "2020-06-15 21:00"], utc=True),
                        "update_action": [1, 1, 1], "stat_flags": [3, 3, 3]})
    out = clean_daily_option_statistics(raw, defs, "GLBX.MDP3")
    assert list(zip(out["timestamp"], out["underlying"], out["settlement_price"])) == \
        [(D("2020-04-20"), "ZNM0", 0.25), (D("2020-06-15"), "ZNU0", 1.5)]


def test_selection_respects_validity_windows():
    defs = _defs().assign(valid_from=D("2024-10-02"), valid_to=D("2024-12-31"))
    assert select_near_the_money(defs, _px("2024-10-01")).empty
    assert not select_near_the_money(defs, _px("2024-10-02")).empty


def test_new_definition_snapshots_carry_each_options_own_life():
    from infra.processing.definitions import normalize_definitions
    raw = pd.DataFrame({"instrument_id": [7], "underlying": ["SR3U6"], "instrument_class": ["P"], "strike_price": [9600.0],
                        "expiration": pd.to_datetime(["2026-09-11 21:00"], utc=True),
                        "activation": pd.to_datetime(["2026-06-15 22:00"], utc=True)})
    d = normalize_definitions(raw).iloc[0]
    assert d["valid_from"] == D("2026-06-15") and d["valid_to"] == D("2026-09-11")
    old = normalize_definitions(raw.drop(columns="activation"))
    assert "valid_from" not in old.columns  # a cached snapshot without the field: no window, nothing dropped
