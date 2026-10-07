"""Intraday data and intraday WIRP: bbo-1m/bbo-1s quotes, the ohlcv-1s reader, point-in-time
PanelRates and intraday_schedules. No network: the API fetchers are stubbed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.api import databento_client as api
from infra.pipeline import bbo, ohlcv_1s
from infra.pipeline.wirp import PanelRates, build_schedule, intraday_schedules, settlement_cutoff
from infra.processing import statistics as stats
from infra.processing import transforms as tf
from infra.storage import contract_store, parquet_store

D = pd.Timestamp


# --------------------------------------------------------------------- bbo transforms
def _raw_bbo(rows):
    """rows: (ts_recv, bid, ask) for ZQX6, as Databento's bbo-1m frame (ts_recv index)."""
    df = pd.DataFrame({
        "ts_recv": pd.to_datetime([r[0] for r in rows], utc=True),
        "bid_px_00": [r[1] for r in rows], "ask_px_00": [r[2] for r in rows],
        "bid_sz_00": [10] * len(rows), "ask_sz_00": [12] * len(rows), "symbol": ["ZQX6"] * len(rows),
        "price": [95.95] * len(rows),
    })
    return df.set_index("ts_recv")


def test_bbo_keeps_one_sided_quotes_and_drops_empty_or_crossed_books():
    clean = tf.clean_futures_bbo(_raw_bbo([
        ("2026-09-29 13:00", 95.945, 95.95),
        ("2026-09-29 13:01", np.nan, 95.95),  # one-sided: kept, mid NaN
        ("2026-09-29 13:02", np.nan, np.nan),  # empty: dropped
        ("2026-09-29 13:03", 95.96, 95.95),  # crossed: dropped
    ]))
    back = tf.decode_futures_bbo(tf.encode_futures_bbo(clean))
    assert list(back["timestamp"].dt.strftime("%H:%M")) == ["13:00", "13:01"]
    assert back["mid"].iloc[0] == pytest.approx(95.9475) and np.isnan(back["mid"].iloc[1])


# --------------------------------------------------------------- caching (Rule 2.1)
def _stub(monkeypatch, name, calls):
    def fake(dataset, symbol, start, end, **kw):
        calls.append((kw.get("schema"), symbol, start, end))
        minutes = pd.date_range(start, end, freq="1min", inclusive="left")[:3]
        if "bbo" in (kw.get("schema") or ""):
            return _raw_bbo([(m, 95.945, 95.95) for m in minutes])
        return pd.DataFrame({"ts_event": pd.to_datetime(minutes, utc=True), "open": 95.9, "high": 95.9,
                             "low": 95.9, "close": 95.9, "volume": 5, "symbol": symbol}).set_index("ts_event")
    monkeypatch.setattr(api, name, fake)


def test_bbo_1m_is_fetched_once_per_day(monkeypatch, tmp_path):
    calls = []
    _stub(monkeypatch, "fetch_futures_bbo", calls)
    kw = dict(dataset="GLBX.MDP3", root=tmp_path / "bbo", coverage_file=tmp_path / "cov.parquet")
    bbo.load_bbo(["ZQX6"], "2025-03-03", "2025-03-05", **kw)
    bbo.load_bbo(["ZQX6"], "2025-03-03 10:00", "2025-03-04 12:00", **kw)
    assert [c[2:] for c in calls] == [(D("2025-03-03"), D("2025-03-05"))]


@pytest.mark.parametrize("loader,fetcher,schema", [
    (bbo.load_bbo_1s, "fetch_futures_bbo", "bbo-1s"),
    (ohlcv_1s.load_ohlcv_1s, "fetch_futures_ohlcv", "ohlcv-1s"),
])
def test_1s_readers_store_what_they_fetch_and_only_pay_for_the_exact_uncovered_window(
        monkeypatch, tmp_path, loader, fetcher, schema):
    calls = []
    _stub(monkeypatch, fetcher, calls)
    kw = dict(dataset="GLBX.MDP3", root=tmp_path / "s", coverage_file=tmp_path / "cov.parquet")
    first = loader(["ZQX6"], "2025-03-19 18:00", "2025-03-19 19:00", **kw)
    loader(["ZQX6"], "2025-03-19 18:30", "2025-03-19 19:30", **kw)  # half already on disk
    assert calls == [(schema, "ZQX6", D("2025-03-19 18:00"), D("2025-03-19 19:00")),
                     (schema, "ZQX6", D("2025-03-19 19:00"), D("2025-03-19 19:30"))]
    assert not first.empty and parquet_store.has_data(tmp_path / "s")


# ------------------------------------------------------------------ point in time
def test_a_quote_is_known_at_its_timestamp_a_bar_only_once_complete():
    frame = pd.DataFrame({"timestamp": [D("2026-09-29 14:00")], "ticker": ["ZQX6"], "px": [95.9]})
    month = {pd.Period("2026-11", freq="M"): "ZQX6"}
    quote, bar = PanelRates(frame, "px"), PanelRates(frame, "px", pd.Timedelta(minutes=1))
    assert len(quote.rates(month, D("2026-09-29 14:00"))[0]) == 1
    assert len(bar.rates(month, D("2026-09-29 14:00"))[0]) == 0  # the 14:00 bar ends 14:01
    assert len(bar.rates(month, D("2026-09-29 14:01"))[0]) == 1


def test_settlement_cutoff_is_the_trading_day_before_the_instants_own():
    assert settlement_cutoff(D("2025-01-07 15:00")) == D("2025-01-06")  # Tue session, Mon settled
    assert settlement_cutoff(D("2025-01-07 23:30")) == D("2025-01-07")  # after 17:00 CT: Wed's session
    assert settlement_cutoff(D("2025-01-06 15:00")) == D("2025-01-05")  # Mon: Fri's settlement is <= Sun


# --------------------------------------------------------------------- intraday WIRP
PRICES = {"ZQQ6": 96.37, "ZQU6": 96.20, "ZQV6": 96.08, "ZQX6": 95.95, "ZQZ6": 95.80}


def _fixture(tmp_path):
    contracts = tmp_path / "contracts.parquet"
    expiries = {"ZQQ6": "2026-08-31", "ZQU6": "2026-09-30", "ZQV6": "2026-10-30",
                "ZQX6": "2026-11-30", "ZQZ6": "2026-12-31"}
    contract_store.write_contracts(contracts, pd.DataFrame({
        "root": "ZQ", "ticker": list(expiries), "instrument_id": range(len(expiries)),
        "expiry": [D(e) for e in expiries.values()], "activation": D("2026-01-01")}))
    daily = tmp_path / "Daily"
    rows = [(D(d), t, p) for d in ("2026-09-21", "2026-09-22", "2026-09-23") for t, p in PRICES.items()]
    df = pd.DataFrame(rows, columns=["timestamp", "ticker", "settlement_price"])
    df["open_interest"] = pd.array([None] * len(df), dtype="Int64")
    parquet_store.write_partitioned(stats.encode_daily(df), daily, stats.DAILY_KEYS)
    quotes = tmp_path / "bbo"
    minutes = pd.date_range("2026-09-23 14:00", "2026-09-23 14:59", freq="1min")
    q = pd.DataFrame([(m, t, p - 0.0025, p + 0.0025) for m in minutes for t, p in PRICES.items()
                      if t in ("ZQV6", "ZQX6", "ZQZ6")], columns=["timestamp", "ticker", "bid", "ask"])
    # a 5bp move in NOVEMBER - the flat month the October meeting is priced from (not ZQV6 itself)
    q.loc[(q["ticker"] == "ZQX6") & (q["timestamp"] >= "2026-09-23 14:30"), ["bid", "ask"]] -= 0.05
    q["bid_size"], q["ask_size"] = pd.array([1] * len(q), dtype="Int32"), pd.array([1] * len(q), dtype="Int32")
    parquet_store.write_partitioned(tf.encode_futures_bbo(q), quotes, tf.BBO_KEYS)
    return dict(contracts_file=contracts, close_root=daily, intraday_root=quotes, adjustments_dir=tmp_path / "adj")


def test_intraday_wirp_matches_the_daily_one_when_quotes_equal_settlements_and_never_looks_ahead(tmp_path):
    kw = _fixture(tmp_path)
    out = intraday_schedules("2026-09-23 13:00", "2026-09-23 15:30", grid="15min", **kw)
    grid = sorted(out["timestamp"].unique())
    assert [D(t).strftime("%H:%M") for t in grid] == ["14:00", "14:15", "14:30", "14:45", "15:00"]  # only while quoting
    daily, _ = build_schedule("close", today=D("2026-09-23"), contracts_file=kw["contracts_file"],
                              close_root=kw["close_root"], adjustments_dir=kw["adjustments_dir"])
    at_1415 = out[out["timestamp"] == "2026-09-23 14:15"].reset_index(drop=True)
    cols = ["meeting_date", "outcome_bps", "probability"]
    pd.testing.assert_frame_equal(at_1415[cols], daily[cols], check_dtype=False)
    october = out[out["meeting_date"] == "2026-10-28"].groupby("timestamp")["change_bps"].first()
    assert october[D("2026-09-23 14:15")] == pytest.approx(october[D("2026-09-23 14:00")])
    assert october[D("2026-09-23 14:30")] > october[D("2026-09-23 14:15")] + 1  # the move shows at 14:30, not before


def test_intraday_wirp_records_the_stalest_price_it_used(tmp_path):
    out = intraday_schedules("2026-09-23 14:15", "2026-09-23 14:30", grid="15min", **_fixture(tmp_path))
    row = out.iloc[0]
    assert row["price_as_of"] == D("2026-09-23 14:15") and row["oldest_price_as_of"] == D("2026-09-23 14:15")


def test_1s_wirp_is_saved_and_a_rerun_replaces_its_window(tmp_path):
    from infra.pipeline.wirp import read_wirp_1s, store_wirp_1s
    kw, root = _fixture(tmp_path), tmp_path / "WIRP_1s"
    saved = store_wirp_1s("2026-09-23 14:10", "2026-09-23 14:12", source="bbo-1m", root=root, **kw)
    store_wirp_1s("2026-09-23 14:10", "2026-09-23 14:12", source="bbo-1m", root=root, **kw)  # re-run
    back = read_wirp_1s("2026-09-23 14:00", "2026-09-23 15:00", root=root)
    assert len(back) == len(saved) > 0 and set(back["source"]) == {"bbo-1m"}



def test_foreign_bond_futures_universe_and_daily_bar_store(tmp_path):
    """Eurex bonds + Long Gilt quotes, the Long Gilt's daily bars in their OWN store and
    coverage (added 2026-10-07)."""
    from types import SimpleNamespace
    from infra.config import FUTURES_ROOTS, SCHEMA_OHLCV_1D
    from infra.cycle import intraday as it
    assert set(it.FOREIGN_BOND_FUTURES) == {"FGBL", "FGBM", "FGBS", "FGBX", "FBTP", "R"} and "FGBX" in FUTURES_ROOTS
    ip = it.IntradayPaths.under(tmp_path)
    m = {"R   FMZ0026!": SimpleNamespace(ticker="R   FMZ0026!", first=pd.Timestamp("2026-09-01"),
                                       last=pd.Timestamp("2026-09-30"), dataset="IFLL.IMPACT")}
    plan = it.plan_intraday_px(m, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-30"), ipaths=ip,
                               schemas=(SCHEMA_OHLCV_1D,))
    assert list(plan) == [(SCHEMA_OHLCV_1D, "R   FMZ0026!")]
    assert "ohlcv-1d" in str(ip.ohlcv_1d_coverage)
