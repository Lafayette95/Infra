"""FTSE-Tradeweb closing prices (infra.processing.tradeweb_prices, infra.pipeline.tradeweb_prices):
parsing, storage round trip, the history backfill's retry-then-split, and the daily update -
all with a stub session (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.pipeline import tradeweb_prices as tw
from infra.processing import tradeweb_prices as tp

D = pd.Timestamp
HEAD = '"Gilt Name","Close of Business Date","ISIN","Type","Coupon","Maturity","Clean Price","Dirty Price","Yield","Mod Duration","Accrued Interest"\n'


def _csv(rows):
    return HEAD + "".join(",".join(f'"{v}"' for v in r) + "\n" for r in rows)


GILT = ["UKT 4.125 01/27", "10/7/2026", "GB00BL6C7720", "Conventional", "4.125", "1/29/2027", "99.995", "100.790856",
        "4.075293", "0.305732", "0.795856"]
BILL = ["UKTB  10/26", "10/7/2026", "GB00BSGQFR99", "Bills", "N/A", "10/12/2026", "99.957062", "N/A", "3.919758",
        "0.010954", "N/A"]
BTP = ["BTPS 3.85 07/34", "10/7/2026", "IT0005584856", "Conventional", "3.85", "7/1/2034", "102.1", "103.0",
       "3.5", "6.6", "0.9"]


def test_parse_and_round_trip():
    df = tp.parse_export(_csv([GILT, BILL, BTP, BTP]))                 # a duplicated line kept once
    assert list(df["isin"]) == ["GB00BL6C7720", "GB00BSGQFR99", "IT0005584856"]
    assert list(df["country"]) == ["GB", "GB", "IT"] and df["name"].iloc[1] == "UKTB 10/26"
    assert pd.isna(df["coupon"].iloc[1]) and pd.isna(df["price_dirty"].iloc[1])
    back = tp.decode(tp.encode(df))
    assert back["yield"].iloc[0] == pytest.approx(4.0753, abs=1e-4)    # stored to 0.01bp
    assert back["price_clean"].iloc[1] == pytest.approx(99.9571, abs=1e-4)


class FlakySession:
    """Fails a multi-day range for one ISIN on every try; any single-half range works."""

    def __init__(self, fail_ranges_longer_than_days=40):
        self.calls, self.limit = [], fail_ranges_longer_than_days

    def export(self, lo, hi, *, isin="", **_):
        self.calls.append((pd.Timestamp(lo), pd.Timestamp(hi)))
        if (pd.Timestamp(hi) - pd.Timestamp(lo)).days > self.limit:
            raise RuntimeError("InSite export: the site returned an application error")
        days = pd.bdate_range(lo, hi)
        return _csv([[GILT[0], f"{d.month}/{d.day}/{d.year}", isin, *GILT[3:]] for d in days])


def test_history_retries_then_splits(tmp_path, monkeypatch):
    s = FlakySession()
    monkeypatch.setattr(tw, "SESSION", s)
    monkeypatch.setattr(tw, "ERROR_PAUSE_S", 0)
    r = tw.backfill_isin("GB00BL6C7720", "2026-01-05", "2026-03-31", raw_root=tmp_path / "raw", root=tmp_path / "px")
    assert r["failed_days"] == [] and r["rows"] == len(pd.bdate_range("2026-01-05", "2026-03-31"))
    full = [c for c in s.calls if c == (D("2026-01-05"), D("2026-03-31"))]
    assert len(full) == 1 + tw.ERROR_RETRIES                              # retried before splitting
    n = len(s.calls)
    tw.backfill_isin("GB00BL6C7720", "2026-01-05", "2026-03-31", raw_root=tmp_path / "raw", root=tmp_path / "px")
    assert len(s.calls) == n                                              # archived chunks never asked again
    px = tw.read_prices(root=tmp_path / "px")
    assert px["timestamp"].is_unique and px["isin"].unique().tolist() == ["GB00BL6C7720"]


def test_daily_update_stores_and_extends_the_universe(tmp_path, monkeypatch):
    class Daily:
        def export(self, lo, hi, **_):
            return _csv([GILT, BTP])

    monkeypatch.setattr(tw, "SESSION", Daily())
    r = tw.update_daily(now=D("2026-10-08 15:45"), raw_root=tmp_path / "raw", root=tmp_path / "px")
    assert r["rows"] == 2 and (tmp_path / "raw" / "daily" / "2026-10-08.csv").exists()
    u = tw.read_universe(raw_root=tmp_path / "raw")
    assert set(u["isin"]) == {"GB00BL6C7720", "IT0005584856"}


def test_discovery_never_reads_an_error_as_no_securities(tmp_path, monkeypatch):
    class Grid:
        def grid_rows(self, start, end, **_):
            if pd.Timestamp(start) == D("2022-11-01"):
                raise RuntimeError("InSite search: the site returned an application error")
            return [["UKT 5 03/18", "1/2/2018", "GB00B1VWPC84", "Conventional", "5.000", "3/7/2018"]]

    monkeypatch.setattr(tw, "SESSION", Grid())
    monkeypatch.setattr(tw, "ERROR_PAUSE_S", 0)
    r = tw.discover([D("2018-01-02"), D("2022-11-01")], security_types=("Conventional",), raw_root=tmp_path)
    assert r["failed_days"] == [D("2022-11-01")] and r["universe"] == 1
    assert list(tw.read_universe(raw_root=tmp_path)["isin"]) == ["GB00B1VWPC84"]      # saved as it went
