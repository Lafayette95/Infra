"""Government of Canada auctions (infra.processing.goc_auctions, infra.pipeline.goc_auctions):
Valet parsing (header rows dropped, single-price auctions), the schedule (Valet and page),
the universe and the release rows' times."""
from __future__ import annotations

import json

import pandas as pd

from infra.pipeline import goc_auctions as pg
from infra.processing import goc_auctions as ga

D = pd.Timestamp


def _obs(**kv):
    return {k: {"v": v} for k, v in kv.items()}


RESULTS = {"observations": [
    {"bond_id": "2022-02-02_1", **_obs(AUC_BOND_AUCTION_DATE="2022-02-02", AUC_BOND_BID_DEADLINE="12:00")},   # header
    {"bond_id": "2022-02-02_1-1", **_obs(AUC_BOND_AUCTION_DATE="2022-02-02", AUC_BOND_BID_DEADLINE="12:00",
                                         AUC_BOND_ISIN="CA135087M920", AUC_BOND_TERM_YEARS="2", AUC_BOND_COUPON_RATE="0.750",
                                         AUC_BOND_MATURITY_DATE="2024-02-01", AUC_BOND_AMOUNT="3500.000",
                                         AUC_BOND_HIGH_YIELD="1.294", AUC_BOND_BOC_PURCHASE="240")},
    {"bond_id": "1998-11-25_1", **_obs(AUC_BOND_AUCTION_DATE="1998-11-25", AUC_BOND_BID_DEADLINE="10:30",
                                       AUC_BOND_ISIN="CA135087WN09", AUC_BOND_TERM_YEARS="5")}]}
RRB = {"observations": [{"bond_id": "x-1", **_obs(AUC_BOND_RR_AUCTION_DATE="2022-09-01", AUC_BOND_RR_ISIN="CA135087M433",
                                                  AUC_BOND_RR_TERM_YEARS="30", AUC_BOND_RR_ALLOTMENT_YIELD="1.272")}]}
SCHED = {"observations": [{"bond_sched": "a", **_obs(AUC_SCHED_AUCTION_DATE="2026-10-14", AUC_SCHED_TERM_YEARS="30",
                                                     AUC_SCHED_MATURITY_DATE="2059-06-01", AUC_SCHED_FURTHER_DETAILS="2026-10-08",
                                                     AUC_SCHED_DELIVERED="2026-10-15", AUC_SCHED_AUCTION_TYPE="Bond - Nominal")}]}
PAGE = """<table><tr><th>Auction type</th><th>Term (years)</th><th>Maturity date</th><th>Further details of issue</th>
<th>Auction date</th><th>Delivered</th></tr>
<tr><td>Bond - Real Return</td><td>30</td><td>2054-12-01</td><td>2022-08-25</td><td>2022-09-01</td><td>2022-09-06</td></tr></table>"""


def test_results_drop_header_rows_and_read_single_price_yields():
    b = ga.parse_results(RESULTS, "BOND")
    assert list(b["isin"]) == ["CA135087M920", "CA135087WN09"]          # the header row is gone
    assert b["boc_purchase_m"].iloc[0] == 240 and b["high_yield"].iloc[0] == 1.294
    r = ga.parse_results(RRB, "RRB")
    assert r["high_yield"].iloc[0] == 1.272                               # allotment yield = the single price


def test_schedule_from_valet_and_page():
    v = ga.parse_schedule_valet(SCHED)
    assert (v["auction_date"].iloc[0], v["call_date"].iloc[0], v["kind"].iloc[0]) == (D("2026-10-14"), D("2026-10-08"), "BOND")
    p = ga.parse_schedule_page(PAGE)
    assert (p["auction_date"].iloc[0], p["kind"].iloc[0], p["term_years"].iloc[0]) == (D("2022-09-01"), "RRB", 30)


def test_events():
    assert ga.event_id("BOND", 30.0) == "CA_AUCTION_30Y" and ga.event_id("BOND", 4) == "CA_AUCTION_OTHER"
    assert ga.event_id("TBILL", 0.25) == "CA_AUCTION_TBILL" and ga.event_id("RRB", 30) == "CA_AUCTION_RRB"


def test_held_auctions_sit_at_their_own_deadline(tmp_path):
    pg.store_auctions(ga.parse_results(RESULTS, "BOND"), root=tmp_path / "auc")
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "20261008T040000Z__valet.json").write_text(json.dumps(SCHED), "utf-8")
    rows = pg.release_rows(observed=D("2026-10-08"), root=tmp_path / "plan", auctions_root=tmp_path / "auc")
    held = rows[rows["source"] == pg.HELD_SOURCE].set_index("event")["timestamp"]
    assert held["CA_AUCTION_2Y"] == D("2022-02-02 17:00")      # 12:00 Ottawa (EST) = 17:00 UTC
    assert held["CA_AUCTION_5Y"] == D("1998-11-25 15:30")      # that auction's own 10:30 deadline
    plan = rows[rows["source"] == pg.PLAN_SOURCE]
    assert list(plan["timestamp"]) == [D("2026-10-14 16:00")] and plan["known_from"].iloc[0] == D("2026-10-08")
    universe = pg.read_securities(auctions_root=tmp_path / "auc", outstanding_root=tmp_path / "out")
    assert list(universe["isin"]) == ["CA135087M920", "CA135087WN09"]
