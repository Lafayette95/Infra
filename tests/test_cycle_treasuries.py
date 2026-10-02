"""The ref step (Treasury auctions -> securities -> OTR map, CME baskets) and the Treasury
half of px (FedInvest prices): wiring, ordering without coupling, and their checks. No
network: fake auctions, CME files and FedInvest pages."""
from __future__ import annotations

import pandas as pd

from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.px_treasuries import backfill_daily_treasury_px, plan_daily_treasury_px
from infra.cycle.ref import REF_STEP
from infra.cycle.runner import DEFAULT_STEPS, execute_step
from infra.api.fedinvest_client import parse_page
from infra.pipeline import treasury_otr as potr
from infra.pipeline import treasury_prices as ptp
from infra.pipeline import treasury_ref as pref

D = pd.Timestamp
AUCTION = {"cusip": "91282CQQ7", "auction_date": "2026-05-12", "announcemt_date": "2026-05-06",
           "issue_date": "2026-05-15", "dated_date": "2026-05-15", "maturity_date": "2036-05-15",
           "first_int_payment_date": "2026-11-15", "int_payment_frequency": "Semi-Annual",
           "security_type": "Note", "security_term": "10-Year", "original_security_term": "10-Year",
           "reopening": "No", "inflation_index_security": "No", "floating_rate": "No",
           "cash_management_bill_cmb": "No", "closing_time_comp": "01:00 PM", "offering_amt": "42000000000",
           "high_yield": "4.4", "bid_to_cover_ratio": "2.5", "int_rate": "4.375"}
TCF = (b"Exch,Period,PFCode,CUSIP,Invoice_Conversion_Factor,Create_Time\n"
       b"CBT,202612,TN,91282CQQ7,0.8858,2026-10-01 11:30:21\n")
PAGE = """<table><tr><td>91282CQQ7</td><td>MARKET BASED NOTE</td><td>4.375%</td><td>05/15/2036</td><td></td>
<td>99.0</td><td>98.9</td><td>98.95</td></tr></table>"""


def _ctx(tmp_path, start="2026-09-28", end="2026-09-30", run_day="2026-10-01"):
    return StepContext(D(start), D(end), D(run_day), CyclePaths.under(tmp_path))


def _stub(monkeypatch):
    monkeypatch.setattr("infra.cycle.raw_reference.AUCTIONS_FETCH", lambda since: pd.DataFrame([AUCTION]))
    monkeypatch.setattr("infra.pipeline.futures_baskets.LIST", lambda: ["TCF_20260928.csv", "TCF_20260929.csv",
                                                                         "TCF_20260930.csv"])
    monkeypatch.setattr("infra.pipeline.futures_baskets.FETCH", lambda name: TCF)


def test_ref_runs_first_and_px_does_not_depend_on_it():
    names = [s.name for s in DEFAULT_STEPS]
    assert names.index("ref") < names.index("px")
    px = next(s for s in DEFAULT_STEPS if s.name == "px")
    assert "ref" not in px.depends_on  # a ref failure must never block the settlements


def test_ref_builds_the_reference_from_todays_auctions(tmp_path, monkeypatch):
    _stub(monkeypatch)
    out = execute_step(REF_STEP, _ctx(tmp_path))
    assert out.status == "ok", [(c.name, c.message) for c in out.checks if not c.passed]
    paths = CyclePaths.under(tmp_path)
    assert pref.read_securities(root=paths.treasury_securities_dir)["cusip"].tolist() == ["91282CQQ7"]
    assert potr.otr_cusip("2026-09-30", "10y", root=paths.treasury_otr_dir) == "91282CQQ7"
    checks = {c.name: c for c in out.checks}
    assert checks["auctions_fetch_ok"].passed and checks["cme_tcf_complete"].passed
    assert checks["treasury_otr_complete"].passed


def test_a_down_cme_ftp_only_warns(tmp_path, monkeypatch):
    _stub(monkeypatch)

    def down():
        raise ConnectionError("ftp down")

    monkeypatch.setattr("infra.pipeline.futures_baskets.LIST", down)
    out = execute_step(REF_STEP, _ctx(tmp_path))
    checks = {c.name: c for c in out.checks}
    assert out.status == "ok" and not checks["cme_tcf_fetch_ok"].passed and not checks["cme_tcf_complete"].passed


def test_px_treasuries_store_posted_days_and_retry_pending_ones(tmp_path, monkeypatch):
    _stub(monkeypatch)
    execute_step(REF_STEP, _ctx(tmp_path))  # the reference the yields need
    paths = CyclePaths.under(tmp_path)
    posted, not_yet = parse_page(PAGE), parse_page(PAGE.replace("98.95", "0.000000"))
    fetch = lambda day: not_yet if day == D("2026-09-30") else posted  # noqa: E731
    out = backfill_daily_treasury_px("2026-09-28", "2026-09-30", paths=paths, fetch=fetch, now=D("2026-10-01 10:00"))
    # the window, plus never-covered days in the 30-day lookback before it (all posted here)
    assert {D("2026-09-28"), D("2026-09-29"), D("2026-08-31")} <= set(out["stored"])
    assert out["pending"] == [D("2026-09-30")]
    assert plan_daily_treasury_px("2026-09-28", "2026-09-30", paths=paths, now=D("2026-10-01 10:00")) == \
        [D("2026-09-30")]  # only the pending day is asked again
    px = ptp.read_prices("2026-09-28", "2026-10-01", root=paths.treasury_prices_dir)
    assert px["yield_eod"].notna().all()
