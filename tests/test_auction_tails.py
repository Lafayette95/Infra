"""Auction tails: extraction from both sources' styles, matching by tenor + official high
yield, the two checks, the best-tail view, and the harvest loop (each URL once). No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.pipeline import auction_tails as pt
from infra.processing import auction_tails as at

D = pd.Timestamp


@pytest.fixture(autouse=True)
def _short_test_pages(monkeypatch):
    """The synthetic pages here are a few hundred bytes; only the stub test uses the real minimum."""
    monkeypatch.setattr(pt, "MIN_PAGE_BYTES", 0)


AUCTIONS = pd.DataFrame({
    "timestamp": pd.to_datetime(["2021-11-10 18:00", "2022-02-22 18:00", "2017-01-25 18:00", "2023-11-09 18:00"]),
    "cusip": ["30Y", "2Y", "5Y", "30Y-23"], "tenor_years": [30.0, 2.0, 5.0, 30.0],
    "high_yield": [1.940, 1.553, 1.988, 4.769]})
ZH_30Y = ("<title>We Have A Tantrum: 30 Year Treasury Auction Is Catastrophic</title><p>Stopping at a high yield "
          "of 1.940%, the auction was slightly below last month's 2.049%, but because while the When Issued traded "
          "at 1.888% the auction tailed by 5.2bps, the biggest tail on record.</p>")
ZH_2Y = ("<title>Record Low Dealers In Stellar 2Y Treasury Auction</title><p>starting with the high yield which at "
         "1.553 (the highest since Dec 2019), stopped through the When Issued 1.559% by 0.6bps. The bid</p>")
FXL_5Y = "<title>US sells 5-year notes at 1.988% vs 1.980 WI bid</title><p>Highlights</p>"


def test_extract_both_styles():
    x = at.extract("zerohedge", *at.page_text(ZH_30Y))
    assert (x["high_yield_reported"], x["wi_yield"], x["stated_tail_bp"]) == (1.940, 1.888, 5.2)
    assert at.extract("zerohedge", *at.page_text(ZH_2Y))["stated_tail_bp"] == -0.6
    fx = at.extract("forexlive", *at.page_text(FXL_5Y))
    assert (fx["high_yield_reported"], fx["wi_yield"]) == (1.988, 1.980)


@pytest.mark.parametrize("body, expect", [  # 2011-2012 phrasings (archived ZeroHedge recaps)
    ("The 30 year priced at 3.75%, a huge 11 bps tail to the When Issued which was trading at 3.64%, the Bid",
     (3.75, 3.64, 11.0, 2)),
    ("as expected: the closing yield of 0.90% was inside the When Issued of 0.905%. The Bid", (0.90, 0.905, None, 2)),
    ("First, the tail was a notable 3 bps with the When Issued trading at 2.24%, ahead of the auction pricing a "
     "disappointing 2.27%, well above", (2.27, 2.24, 3.0, 2)),
])
def test_extract_older_phrasings(body, expect):
    x = at.extract("zerohedge", "", body)
    hy, wi, stated, dec = expect
    assert (x["high_yield_reported"], x["wi_yield"], x["hy_decimals"]) == (hy, wi, dec)
    assert (np.isnan(x["stated_tail_bp"]) if stated is None else x["stated_tail_bp"] == stated)


def test_two_decimal_reports_check_at_their_own_precision():
    row = {"high_yield_reported": 0.90, "wi_yield": 0.905, "stated_tail_bp": np.nan, "hy_decimals": 2,
           "high_yield_official": 0.904}
    assert at.check(row)["passed"]  # 0.904 rounds to the reported 0.90
    assert not at.check({**row, "wi_yield": 0.70})["passed"]  # a 20bp "tail": a mis-read number


def test_non_us_auctions_are_not_candidates():
    assert at.NON_US.search("Contagion Shakes Euro Core As 10 Year German Bund Auction Complete And Utter Disaster")
    assert not at.NON_US.search("Horrible 30 Year Bond Auction Prices With Unprecedented 11 bps Tail")


def test_process_checks_and_matches():
    row, outcome = pt.process_article("zerohedge", ZH_30Y, "https://www.zerohedge.com/markets/30y", D("2021-11-10"),
                                      AUCTIONS)
    assert outcome == "passed" and row["cusip"] == "30Y" and row["tail_bp"] == 5.2
    # a ForexLive post first archived years later still matches: the date is in its URL
    row, outcome = pt.process_article("forexlive", FXL_5Y,
                                      "https://www.forexlive.com/news/!/us-sells-5-year-notes-20170125",
                                      D("2024-07-18"), AUCTIONS)
    assert outcome == "passed" and row["tail_bp"] == 0.8
    # a stated tail that contradicts high yield - WI fails the check
    bad = ZH_30Y.replace("tailed by 5.2bps", "tailed by 2.0bps")
    row, outcome = pt.process_article("zerohedge", bad, "u", D("2021-11-10"), AUCTIONS)
    assert outcome == "failed check" and not row["passed"]
    # a reported high yield no auction had: no match
    assert pt.process_article("zerohedge", ZH_30Y.replace("1.940%", "1.941%"), "u", D("2021-11-10"),
                              AUCTIONS)[1] == "no auction match"


def test_best_prefers_priority_and_reports_spread():
    rows = pd.DataFrame([
        {"timestamp": D("2017-01-25 18:00"), "cusip": "5Y", "source": "forexlive", "tail_bp": 0.8, "passed": True},
        {"timestamp": D("2017-01-25 18:00"), "cusip": "5Y", "source": "zerohedge", "tail_bp": 0.7, "passed": True},
        {"timestamp": D("2021-11-10 18:00"), "cusip": "30Y", "source": "zerohedge", "tail_bp": 9.0, "passed": False},
    ])
    b = at.best(rows)
    assert len(b) == 1 and b["source"].iloc[0] == "zerohedge" and b["n_sources"].iloc[0] == 2
    assert np.isclose(b["spread_bp"].iloc[0], 0.1)


def test_harvest_processes_each_url_once(tmp_path):
    pages = {"https://www.zerohedge.com/markets/catastrophic-30-year-auction": ZH_30Y,
             "https://www.zerohedge.com/markets/stellar-2y-auction": ZH_2Y}

    def list_fn(prefix, start, end, url_regex=None, first_per_url=False):
        urls = [u for u in pages if "zerohedge.com/markets/" in prefix]
        return pd.DataFrame({"timestamp": pd.to_datetime(["2021-11-10 19:00", "2022-02-22 19:00"][:len(urls)]),
                             "original": urls, "digest": ["x"] * len(urls)})

    fetched = []

    def fetch_fn(ts, url):
        fetched.append(url)
        return pages[url].encode()

    kw = dict(root=tmp_path / "t", manifest=tmp_path / "m.parquet", list_fn=list_fn, fetch_fn=fetch_fn,
              auctions=AUCTIONS, pause_s=0)
    stats = pt.harvest_tails(("zerohedge",), **kw)
    pt.harvest_tails(("zerohedge",), **kw)
    assert stats["zerohedge"] == {"passed": 2} and len(fetched) == 2
    best = pt.best_tails(root=tmp_path / "t")
    assert sorted(best["tail_bp"]) == [-0.6, 5.2]
    assert len(pt.best_tails("2022-01-01", root=tmp_path / "t")) == 1  # point in time: by auction close


def test_an_older_parser_versions_failures_are_retried(tmp_path):
    manifest = tmp_path / "m.parquet"
    pt._record(manifest, [("zerohedge", "zerohedge.com/markets/stellar-2y-auction", pd.Timestamp.now(), "no WI", 1),
                          ("zerohedge", "zerohedge.com/markets/catastrophic-30-year-auction", pd.Timestamp.now(),
                           "passed", 1)])
    fetched = []

    def list_fn(prefix, start, end, url_regex=None, first_per_url=False):
        urls = ["https://www.zerohedge.com/markets/catastrophic-30-year-auction",
                "https://www.zerohedge.com/markets/stellar-2y-auction"] if "markets" in prefix else []
        return pd.DataFrame({"timestamp": pd.to_datetime(["2021-11-10", "2022-02-22"][:len(urls)]), "original": urls,
                             "digest": ["x"] * len(urls)})

    pt.harvest_tails(("zerohedge",), root=tmp_path / "t", manifest=manifest, list_fn=list_fn,
                     fetch_fn=lambda ts, u: (fetched.append(u) or ZH_2Y.encode()), auctions=AUCTIONS, pause_s=0)
    assert fetched == ["https://www.zerohedge.com/markets/stellar-2y-auction"]  # the passed one is never re-fetched


def test_url_dates_and_title_high_yield():
    assert at.slug_date("https://www.zerohedge.com/news/2012-10-23/record-direct-bid-7y") == D("2012-10-23")
    assert at.slug_date("https://www.forexlive.com/news/!/us-sells-5-year-notes-20170125") == D("2017-01-25")
    assert at.slug_date("https://www.zerohedge.com/markets/10y-auction-tails") is None
    x = at.extract("zerohedge", "2 Year Auction Prices At New Record Low Yield Of 0.222%, Well Inside Of 3 Month LIBOR",
                   "nothing in the body")
    assert x["high_yield_reported"] == 0.222



def test_harvest_stops_at_its_time_budget(tmp_path):
    urls = [f"https://www.zerohedge.com/markets/10y-auction-{i}" for i in range(5)]

    def list_fn(prefix, start, end, url_regex=None, first_per_url=False):
        n = len(urls) if "markets" in prefix else 0
        return pd.DataFrame({"timestamp": pd.date_range("2021-11-10", periods=n), "original": urls[:n],
                             "digest": ["x"] * n})

    stats = pt.harvest_tails(("zerohedge",), root=tmp_path / "t", manifest=tmp_path / "m.parquet", list_fn=list_fn,
                             fetch_fn=lambda ts, u: ZH_30Y.encode(), auctions=AUCTIONS, pause_s=0, budget_s=0)
    assert "stopped: time budget" in stats["zerohedge"]  # a spent budget stops before even listing
    assert not (tmp_path / "m.parquet").exists() or pt.read_manifest(tmp_path / "m.parquet").empty  # nothing claimed



def test_stub_pages_and_fetch_errors_are_retried_not_marked_done(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "MIN_PAGE_BYTES", 1000)
    urls = ["https://www.zerohedge.com/markets/catastrophic-30-year-auction",
            "https://www.zerohedge.com/markets/stellar-2y-auction"]

    def list_fn(prefix, start, end, url_regex=None, first_per_url=False):
        n = len(urls) if "markets" in prefix else 0
        return pd.DataFrame({"timestamp": pd.to_datetime(["2021-11-10", "2022-02-22"][:n]), "original": urls[:n],
                             "digest": ["x"] * n})

    def flaky(ts, url):
        if "30-year" in url:
            return b"<html>stub</html>"  # the degraded archive's redirect target
        raise ConnectionError("503")

    kw = dict(root=tmp_path / "t", manifest=tmp_path / "m.parquet", list_fn=list_fn, auctions=AUCTIONS, pause_s=0)
    stats = pt.harvest_tails(("zerohedge",), fetch_fn=flaky, **kw)
    assert sum(stats["zerohedge"].values()) == 2 and all(k.startswith("error") for k in stats["zerohedge"])
    assert pt.read_manifest(tmp_path / "m.parquet").empty  # nothing marked done
    fetched = []
    pt.harvest_tails(("zerohedge",), fetch_fn=lambda ts, u: (fetched.append(u) or
                                                          (ZH_30Y if "30-year" in u else ZH_2Y).encode() * 10), **kw)
    assert len(fetched) == 2  # both retried once the archive is back
