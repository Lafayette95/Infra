"""Offline tests: WIRP dashboard page (selectors, chart builder). No API."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.dashboard import wirp_charts
from infra.dashboard.wirp_selectors import (
    build_schedule, close_rates, contract_for_month, find_anchor, live_rates, month_contract_map,
)
from infra.processing import statistics as stats
from infra.processing import transforms as tf
from infra.storage import contract_store, parquet_store

D = pd.Timestamp
P = lambda s: pd.Period(s, freq="M")  # noqa: E731


def _write_contracts(path, rows: list[tuple[str, str]]) -> None:
    """rows: (ticker, expiry) pairs, all root ZQ."""
    df = pd.DataFrame({
        "root": ["ZQ"] * len(rows),
        "ticker": [r[0] for r in rows],
        "instrument_id": range(len(rows)),
        "expiry": [D(r[1]) for r in rows],
        "activation": [D(r[1]) - pd.Timedelta(days=90) for r in rows],
    })
    contract_store.write_contracts(path, df)


def _write_daily(root, ticker: str, day: str, settlement_price: float) -> None:
    df = pd.DataFrame({
        "timestamp": [D(day)], "ticker": [ticker],
        "settlement_price": [settlement_price], "open_interest": pd.array([None], dtype="Int64"),
    })
    parquet_store.write_partitioned(stats.encode_daily(df), root, stats.DAILY_KEYS)


def _write_bar(root, ticker: str, ts: str, close_price: float) -> None:
    df = pd.DataFrame({
        "timestamp": [D(ts)], "ticker": [ticker],
        "open": [close_price], "high": [close_price], "low": [close_price], "close": [close_price],
        "volume": [100], "open_interest": pd.array([None], dtype="Int64"),
    })
    parquet_store.write_partitioned(tf.encode_futures(df), root, tf.FUTURES_KEYS)


# ------------------------------------------------------------------------- selectors
def test_contract_for_month_matches_by_expiry_not_ticker_parsing(tmp_path):
    contracts_file = tmp_path / "contracts.parquet"
    _write_contracts(contracts_file, [("ZQF6", "2026-01-30"), ("ZQG6", "2026-02-27")])
    contracts = contract_store.read_contracts(contracts_file, "ZQ")
    assert contract_for_month(P("2026-01"), contracts) == "ZQF6"
    assert contract_for_month(P("2026-03"), contracts) is None


def test_month_contract_map_omits_uncached_months(tmp_path):
    contracts_file = tmp_path / "contracts.parquet"
    _write_contracts(contracts_file, [("ZQF6", "2026-01-30")])
    contracts = contract_store.read_contracts(contracts_file, "ZQ")
    out = month_contract_map([P("2026-01"), P("2026-02")], contracts)
    assert out == {P("2026-01"): "ZQF6"}


def test_close_rates_reads_latest_settlement_and_implies_rate(tmp_path):
    root = tmp_path / "Daily"
    _write_daily(root, "ZQF6", "2026-01-05", 95.75)
    _write_daily(root, "ZQF6", "2026-01-06", 95.80)  # latest must win
    rates, as_of = close_rates({P("2026-01"): "ZQF6"}, root=root)
    assert rates[P("2026-01")] == pytest.approx(100.0 - 95.80)
    assert as_of == D("2026-01-06")


def test_live_rates_reads_latest_1m_bar_close(tmp_path):
    root = tmp_path / "ohlcv"
    _write_bar(root, "ZQF6", "2026-01-05 12:00", 95.70)
    _write_bar(root, "ZQF6", "2026-01-05 12:01", 95.72)  # latest must win
    rates, as_of = live_rates({P("2026-01"): "ZQF6"}, root=root)
    assert rates[P("2026-01")] == pytest.approx(100.0 - 95.72)
    assert as_of == D("2026-01-05 12:01")


def test_close_rates_empty_when_nothing_cached(tmp_path):
    rates, as_of = close_rates({P("2026-01"): "ZQF6"}, root=tmp_path / "Daily")
    assert rates.empty and as_of is None


def test_find_anchor_walks_back_to_nearest_flat_month(tmp_path):
    root = tmp_path / "Daily"
    contracts_file = tmp_path / "contracts.parquet"
    _write_contracts(contracts_file, [("ZQZ5", "2025-12-31"), ("ZQF6", "2026-01-30")])
    _write_daily(root, "ZQZ5", "2025-12-15", 95.50)  # Dec: flat in this fixture's meeting set
    contracts = contract_store.read_contracts(contracts_file, "ZQ")
    anchor = find_anchor(
        P("2026-01"), contracts, lambda mt: close_rates(mt, root=root), meeting_months={P("2026-01")},
    )
    assert anchor == (P("2025-12"), pytest.approx(100.0 - 95.50))


def test_find_anchor_skips_a_past_meeting_month_even_though_its_not_upcoming(tmp_path):
    """A month that itself had a meeting is never flat, even read long after the fact -
    its own contract price is a day-weighted blend, not a single prevailing rate."""
    root = tmp_path / "Daily"
    contracts_file = tmp_path / "contracts.parquet"
    _write_contracts(contracts_file, [
        ("ZQX5", "2025-11-28"), ("ZQZ5", "2025-12-31"), ("ZQF6", "2026-01-30"),
    ])
    _write_daily(root, "ZQX5", "2025-11-10", 95.60)  # Nov: genuinely flat
    _write_daily(root, "ZQZ5", "2025-12-15", 95.50)  # Dec: had a meeting - must be skipped
    contracts = contract_store.read_contracts(contracts_file, "ZQ")
    anchor = find_anchor(
        P("2026-01"), contracts, lambda mt: close_rates(mt, root=root),
        meeting_months={P("2025-12"), P("2026-01")},
    )
    assert anchor == (P("2025-11"), pytest.approx(100.0 - 95.60))


def test_find_anchor_returns_none_when_nothing_cached(tmp_path):
    contracts_file = tmp_path / "contracts.parquet"
    _write_contracts(contracts_file, [])
    contracts = contract_store.read_contracts(contracts_file, "ZQ")
    anchor = find_anchor(
        P("2026-01"), contracts, lambda mt: (pd.Series(dtype="float64"), None), meeting_months=set(),
    )
    assert anchor is None


# ------------------------------------------------------------------------ end-to-end
def test_build_schedule_end_to_end_with_synthetic_data(tmp_path):
    """Uses the REAL infra.config.FOMC_MEETINGS schedule (Dec 2025 meeting ends
    2025-12-10) so this also checks the config data itself lines up with the pipeline."""
    contracts_file = tmp_path / "contracts.parquet"
    daily_root = tmp_path / "Daily"
    _write_contracts(contracts_file, [("ZQX5", "2025-11-28"), ("ZQZ5", "2025-12-31")])
    _write_daily(daily_root, "ZQX5", "2025-11-10", 100.0 - 4.33)  # Nov: flat anchor

    # December 2025's FOMC meeting ends 2025-12-10 (31-day month); construct the
    # contract's average EXACTLY from a full 25bp cut so the outcome is unambiguous.
    days_in_month, meeting_day = 31, 10
    pre, post = 4.33, 4.08
    avg = (pre * meeting_day + post * (days_in_month - meeting_day)) / days_in_month
    _write_daily(daily_root, "ZQZ5", "2025-12-01", 100.0 - avg)

    schedule, meta = build_schedule(
        "close", today=D("2025-11-15"), contracts_file=contracts_file, close_root=daily_root,
    )

    assert not schedule.empty
    dec = schedule[schedule["month"] == "2025-12"]
    # Constructed as (near-)exactly a 25bp cut, but the settlement price is stored at
    # fixed-point (×10000, CLAUDE.md 6b) - the round trip can leave a tiny residual
    # second level, so check the DOMINANT outcome rather than requiring a single row.
    dominant = dec.loc[dec["probability"].idxmax()]
    assert dominant["probability"] > 0.999
    assert dominant["outcome_bps"] == pytest.approx(-25.0)
    assert dominant["implied_rate"] == pytest.approx(post, abs=1e-3)
    assert meta["mode"] == "close"
    assert meta["anchor_month"] == "2025-11"
    assert meta["anchor_rate"] == pytest.approx(4.33)
    assert meta["as_of"] == D("2025-12-01")


def test_build_schedule_chains_through_an_already_past_meeting(tmp_path):
    """Regression test for a real bug a user caught 2026-09-28: 'today' sits right
    after September's meeting already happened (real schedule: Sep 15-16 2026) but
    before October's (Oct 27-28). The anchor walks back PAST September to the nearest
    flat month (August) - September's own hike/cut must still be chained through when
    pricing October, not silently dropped just because September itself is no longer
    "upcoming"."""
    contracts_file = tmp_path / "contracts.parquet"
    daily_root = tmp_path / "Daily"
    _write_contracts(contracts_file, [
        ("ZQQ6", "2026-08-31"), ("ZQU6", "2026-09-30"),
        ("ZQX6", "2026-11-30"), ("ZQZ6", "2026-12-31"),
    ])
    _write_daily(daily_root, "ZQQ6", "2026-08-15", 100.0 - 4.00)  # Aug: flat anchor

    # September meets 2026-09-16 (30-day month); construct EXACTLY a +25bp hike.
    sep_avg = (4.00 * 16 + 4.25 * (30 - 16)) / 30
    _write_daily(daily_root, "ZQU6", "2026-09-20", 100.0 - sep_avg)

    # November is flat (no meeting) - read directly as October's post-meeting rate:
    # another +25bp on top of September's solved 4.25%, i.e. 4.50%.
    _write_daily(daily_root, "ZQX6", "2026-11-01", 100.0 - 4.50)

    # December meets 2026-12-09 (31-day month, Jan 2027 uncached/beyond config so this
    # falls back to day-weighting its own month) - constructed as an exact hold.
    _write_daily(daily_root, "ZQZ6", "2026-12-01", 100.0 - 4.50)

    schedule, meta = build_schedule(
        "close", today=D("2026-09-20"), contracts_file=contracts_file, close_root=daily_root,
    )

    assert meta["anchor_month"] == "2026-08"
    assert meta["anchor_rate"] == pytest.approx(4.00)
    assert set(schedule["month"]) == {"2026-10", "2026-12"}  # September itself not shown - it's past

    oct_row = schedule[schedule["month"] == "2026-10"].loc[lambda d: d["probability"].idxmax()]
    assert oct_row["method"] == "next_month_flat"
    # The crux of the fix: October's PRE-rate must be September's solved ~4.25%, not
    # August's un-chained anchor of 4.00% - a bug would show pre_rate == 4.00 here.
    # (Loose-ish tolerance: settlement prices round-trip through fixed-point ×10000
    # storage, CLAUDE.md 6b, leaving a tiny residual.)
    assert oct_row["pre_rate"] == pytest.approx(4.25, abs=1e-3)
    assert oct_row["implied_rate"] == pytest.approx(4.50, abs=1e-3)
    assert oct_row["outcome_bps"] == pytest.approx(25.0, abs=1.0)
    assert oct_row["probability"] > 0.99

    dec_row = schedule[schedule["month"] == "2026-12"].loc[lambda d: d["probability"].idxmax()]
    assert dec_row["method"] == "day_weighted"
    assert dec_row["pre_rate"] == pytest.approx(4.50, abs=1e-3)
    assert dec_row["outcome_bps"] == pytest.approx(0.0, abs=1.0)


def test_build_schedule_never_treats_a_month_past_fomc_meetings_as_flat(tmp_path):
    """The LAST meeting in infra.config.FOMC_MEETINGS (2026-12-09) must fall back to
    day-weighting its own month, never read January 2027 as if it were confirmed
    flat - FOMC_MEETINGS simply doesn't cover 2027 yet, it doesn't say there's no
    meeting there. Regression test for the real bug this project found and fixed
    2026-09-28 (a wrongly-flat late-month read produced a +275bp nonsense outcome)."""
    contracts_file = tmp_path / "contracts.parquet"
    daily_root = tmp_path / "Daily"
    _write_contracts(contracts_file, [
        ("ZQX6", "2026-11-30"), ("ZQZ6", "2026-12-31"), ("ZQF7", "2027-01-29"),
    ])
    _write_daily(daily_root, "ZQX6", "2026-11-10", 100.0 - 4.00)  # Nov: flat anchor
    # Dec 2026 meeting ends 2026-12-09 (31-day month); construct EXACTLY a +25bp hike.
    days_in_month, meeting_day, pre, post = 31, 9, 4.00, 4.25
    dec_avg = (pre * meeting_day + post * (days_in_month - meeting_day)) / days_in_month
    _write_daily(daily_root, "ZQZ6", "2026-12-01", 100.0 - dec_avg)
    # January 2027 is deliberately an absurd value - if it were ever read as "flat"
    # this would leak straight into the result.
    _write_daily(daily_root, "ZQF7", "2027-01-05", 100.0 - 99.0)

    schedule, _ = build_schedule(
        "close", today=D("2026-11-15"), contracts_file=contracts_file, close_root=daily_root,
    )
    dec = schedule[schedule["month"] == "2026-12"]
    assert not dec.empty
    dominant = dec.loc[dec["probability"].idxmax()]
    assert dominant["method"] == "day_weighted"
    assert dominant["implied_rate"] == pytest.approx(post, abs=1e-3)


def test_build_schedule_empty_when_nothing_cached(tmp_path):
    schedule, meta = build_schedule(
        "close", today=D("2025-11-15"),
        contracts_file=tmp_path / "contracts.parquet", close_root=tmp_path / "Daily",
    )
    assert schedule.empty
    # find_anchor now runs first (needed to know the chain), so with nothing cached at
    # all it's the anchor lookup that reports the gap, not the meeting-month fetch.
    assert "anchor" in meta["status"].lower()


def test_build_schedule_rejects_unknown_mode():
    with pytest.raises(ValueError):
        build_schedule("realtime")


# ---------------------------------------------------------------------------- charts
def _fake_schedule() -> pd.DataFrame:
    return pd.DataFrame({
        "meeting_date": [D("2026-01-28")] * 2,
        "month": ["2026-01"] * 2,
        "pre_rate": [4.33] * 2,
        "implied_rate": [4.20, 4.20],
        "change_bps": [-13.0, -13.0],
        "method": ["next_month_flat", "next_month_flat"],
        "outcome_step": [0, -1],
        "outcome_bps": [0.0, -25.0],
        "outcome_rate": [4.33, 4.08],
        "probability": [0.48, 0.52],
    })


def _meta() -> dict:
    return {"mode": "close", "as_of": D("2026-01-05"), "anchor_month": "2025-12", "anchor_rate": 4.33}


def test_probability_figure_builds_one_trace_per_outcome_level():
    fig = wirp_charts.probability_figure(_fake_schedule(), _meta(), "light")
    assert len(fig.data) == 2
    assert fig.layout.barmode == "stack"
    assert fig.layout.showlegend is True


def test_probability_figure_empty_input_shows_status_message():
    fig = wirp_charts.probability_figure(pd.DataFrame(), {"status": "no data yet"}, "light")
    assert fig.layout.annotations
    assert "no data yet" in fig.layout.annotations[0].text
