"""Montréal Exchange futures (infra.processing.mx_futures, infra.pipeline.mx_futures): parsing,
the held contract (point in time), the row-cap split and the continuous change."""
from __future__ import annotations

import pandas as pd

from infra.pipeline import mx_futures as mx
from infra.processing import mx_futures as mf

D = pd.Timestamp
HEAD = ('"Date","Symbol","Class Symbol","Root Symbol","Underlying Symbol","Strike Price","Expiry Date","Call/Put",'
        '"Ins. Type","Bid Price","Ask Price","Bid Size","Ask Size","Last Price","Volume","Last Close Price","Net Change",'
        '"Open Price","High Price","Low Price","Total Value","Nb. Trades","Settlement Price","Open Interest",'
        '"Implied Volatility","Accrued Financing Value","Accrued Financing Date","CORRA Rate"\n')


def row(day, tk, settle, oi, vol=1000, bid=0.0):
    return (f'{day},{tk},,CGB,,,2026-12-18,,1,{bid},{bid},0,0,{settle},{vol},0,0,{settle},{settle},{settle},0,0,'
            f'{settle},{oi},,,,\n')


def test_parse_zero_is_missing_and_front_is_point_in_time():
    text = HEAD + row("2026-10-05", "CGBZ26", 115.5, 900) + row("2026-10-05", "CGBH27", 115.2, 1000) \
        + row("2026-10-06", "CGBZ26", 115.6, 950) + row("2026-10-06", "CGBH27", 115.3, 800)
    df = mf.parse(text, "CGB")
    assert df["bid"].isna().all()                        # 0.00 = no quote, never a price
    held = mf.front(df)
    # 10-06 holds the contract with the larger open interest on 10-05 (CGBH27), not on 10-06
    assert held.to_dict() == {D("2026-10-06"): "CGBH27"}


def test_downloads_at_the_row_cap_are_split(monkeypatch):
    calls = []

    def fetch(symbol, lo, hi):
        calls.append((D(lo), D(hi)))
        lines = [row(d.date(), f"CGB{k}", 115.0, 10) for d in pd.bdate_range(lo, hi) for k in range(8)]
        return HEAD + "".join(lines[:499])                 # the site truncates silently at ~500 rows
    monkeypatch.setattr(mx, "FETCH", fetch)
    texts = mx._fetch("CGB", D("2026-01-01"), D("2026-03-31"))
    days = pd.concat([mf.parse(t, "CGB") for t in texts])["timestamp"]
    assert len(calls) > 1 and set(days) == set(pd.bdate_range("2026-01-01", "2026-03-31"))


def test_front_series_moves_on_the_held_contract(tmp_path, monkeypatch):
    text = HEAD + row("2026-10-05", "CGBZ26", 115.5, 900) + row("2026-10-06", "CGBZ26", 115.8, 950) \
        + row("2026-10-07", "CGBZ26", 115.6, 950)
    monkeypatch.setattr(mx, "FETCH", lambda s, lo, hi: text)
    mx.update(["CGB"], now=D("2026-10-08"), raw_root=tmp_path / "raw", root=tmp_path / "st")
    f = mx.front_series("CGB", root=tmp_path / "st")
    assert [round(c, 4) for c in f["change"]] == [0.3, -0.2]
