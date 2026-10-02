"""Treasury futures baskets (infra.processing.futures_baskets, infra.pipeline.futures_baskets):
CME file parsing, contract tickers, the conversion-factor formula, eligibility rules, the
raw archive. No network: inline files and fake FTP hooks."""
from __future__ import annotations

import pandas as pd

from infra.config import TREASURY_BASKET_RULES
from infra.pipeline import futures_baskets as pfb
from infra.processing import futures_baskets as fb

FILE = (b"Exch,Period,PFCode,CUSIP,Invoice_Conversion_Factor,Create_Time\n"
        b"CBT,202612,21,91282CQQ7,0.8810,2026-10-01 11:30:21\n"
        b"CBT,202703,17,912810QU5,0.7182,2026-10-01 11:30:21\n"
        b"CBT,202612,ZZZ,999999999,1.0000,2026-10-01 11:30:21\n")


def test_contract_ticker():
    assert fb.contract_ticker("ZN", pd.Timestamp("2026-12-01")) == "ZNZ6"
    assert fb.contract_ticker("Z3N", pd.Timestamp("2027-03-01")) == "Z3NH7"


def test_parse_maps_codes_to_roots_and_drops_unknown_ones():
    df = fb.parse_tcf(FILE, "2026-10-01")
    assert df[["root", "contract", "cusip"]].values.tolist() == [["ZB", "ZBH7", "912810QU5"], ["ZN", "ZNZ6", "91282CQQ7"]]
    assert (df["source"] == "cme").all() and df["timestamp"].iloc[0] == pd.Timestamp("2026-10-01")


def test_a_six_percent_bond_has_factor_one():
    # at the 6% notional coupon every term converts at par
    for months, maturity in ((3, "2036-12-01"), (1, "2028-12-01")):
        assert fb.conversion_factor(6.0, maturity, "2026-12-01", months) == 1.0


def test_conversion_factor_matches_cme_published_values():
    # CME's TCF file of 2026-10-01: 91282CQQ7 (4.375% 2036-05-15) into TNZ6, 912810QU5
    # (3.125% 2042-02-15) into ZBZ6
    assert fb.conversion_factor(4.375, "2036-05-15", "2026-12-01", 3) == 0.8858
    assert fb.conversion_factor(3.125, "2042-02-15", "2026-12-01", 3) == 0.7182


def test_eligibility_boundaries():
    sec = pd.DataFrame({"cusip": ["in", "short", "long", "orig"], "security_type": "Note",
                        "maturity_date": pd.to_datetime(["2033-06-30", "2033-05-31", "2034-12-15", "2033-06-30"]),
                        "term_months": [120.0, 120.0, 120.0, 360.0], "coupon": 4.0,
                        "auction_date": pd.Timestamp("2026-01-01")})
    ok = fb.eligible(sec, "2026-12-01", TREASURY_BASKET_RULES["ZN"])  # 6.5y..8y from Dec 1 2026
    assert sec.loc[ok, "cusip"].tolist() == ["in"]
    assert fb.eligible(sec, "2026-12-01", TREASURY_BASKET_RULES["ZB"]).sum() == 0


def test_computed_basket_adds_a_new_issue_the_day_after_its_auction():
    sec = pd.DataFrame({"cusip": ["old", "new"], "security_type": "Note",
                        "maturity_date": pd.to_datetime(["2033-06-30", "2033-09-30"]), "term_months": 84.0,
                        "coupon": [4.0, 4.25], "auction_date": pd.to_datetime(["2026-06-24", "2026-09-24"])})
    # CME's file adds a new issue the morning AFTER its auction
    before = fb.computed_basket(sec, "ZN", "2026-12-01", TREASURY_BASKET_RULES["ZN"], "2026-09-24")
    after = fb.computed_basket(sec, "ZN", "2026-12-01", TREASURY_BASKET_RULES["ZN"], "2026-09-25")
    assert before["cusip"].tolist() == ["old"] and sorted(after["cusip"]) == ["new", "old"]
    assert (after["source"] == "computed").all() and after["contract"].iloc[0] == "ZNZ6"


def test_archive_keeps_files_raw_and_never_refetches(tmp_path):
    calls = []

    def fetch(name):
        calls.append(name)
        return FILE

    listing = lambda: ["TCF_20261001.csv", "TCF_20260930.csv"]  # noqa: E731
    out = pfb.archive_tcf(root=tmp_path, listing=listing, fetch=fetch, sleep=lambda s: None)
    assert sorted(out["archived"]) == ["TCF_20260930.csv", "TCF_20261001.csv"] and not out["errors"]
    assert pfb.path_for("TCF_20261001.csv", root=tmp_path).read_bytes() == FILE
    pfb.archive_tcf(root=tmp_path, listing=listing, fetch=fetch, sleep=lambda s: None)
    assert len(calls) == 2
    bad = pfb.archive_tcf(root=tmp_path, listing=lambda: ["TCF_20260929.csv"], fetch=lambda n: b"<html>",
                          sleep=lambda s: None)
    assert "TCF_20260929.csv" in bad["errors"] and not pfb.path_for("TCF_20260929.csv", root=tmp_path).exists()


def test_build_from_archive_and_read(tmp_path):
    pfb.archive_tcf(root=tmp_path / "raw", listing=lambda: ["TCF_20261001.csv"], fetch=lambda n: FILE,
                    sleep=lambda s: None)
    assert pfb.build_cme_baskets(archive_root=tmp_path / "raw", root=tmp_path / "ref") == 2
    assert pfb.read_baskets("2026-10-01", "2026-10-02", contract="ZNZ6", root=tmp_path / "ref")["cusip"].tolist() == \
        ["91282CQQ7"]


def test_a_reopening_at_a_shorter_tenor_lifts_the_original_term_cap():
    # 91282CGQ8: 7y of 2023, reopened as the 5y on 2025-02-25; CME put it in ZF on 02-26
    auctions = pd.DataFrame({"cusip": ["91282CGQ8", "91282CGQ8"], "security_term": ["7-Year", "5-Year"],
                             "auction_date": pd.to_datetime(["2023-02-23", "2025-02-25"])})
    sec = pd.DataFrame({"cusip": ["91282CGQ8"], "security_type": "Note", "maturity_date": pd.to_datetime(["2030-02-28"]),
                        "term_months": [84.0], "coupon": [4.0], "auction_date": pd.to_datetime(["2023-02-23"])})
    rule = TREASURY_BASKET_RULES["ZF"]
    before = fb.computed_basket(sec, "ZF", "2025-06-01", rule, "2025-02-25", fb.issued_terms(auctions, "2025-02-25"))
    after = fb.computed_basket(sec, "ZF", "2025-06-01", rule, "2025-02-26", fb.issued_terms(auctions, "2025-02-26"))
    assert before.empty and after["cusip"].tolist() == ["91282CGQ8"]


def test_listed_contracts_keep_the_expiring_month_through_its_last_trade():
    assert fb.listed_contracts("ZN", "2025-12-17") == [pd.Timestamp(m) for m in ("2025-12-01", "2026-03-01", "2026-06-01")]
    assert fb.listed_contracts("ZN", "2025-12-23")[0] == pd.Timestamp("2026-03-01")  # ZNZ5 last traded Dec 19
    assert fb.listed_contracts("ZT", "2025-12-29")[0] == pd.Timestamp("2025-12-01")  # ZT trades to month end
