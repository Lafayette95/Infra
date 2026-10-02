"""Repo rates and NY Fed securities lending (CLAUDE.md 19): parsing, coverage/planning,
point-in-time reads, and the px checks. Every network hook is faked."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle import px_repo
from infra.pipeline import repo as prepo
from infra.pipeline import sec_lending as psl
from infra.processing import repo as rp
from infra.processing import sec_lending as sl

NOW = pd.Timestamp("2026-10-02 12:00", tz="UTC")


def _nyfed(rate, start, end):
    days = pd.bdate_range(start, end)
    return [{"effectiveDate": f"{d:%Y-%m-%d}", "type": rate, "percentRate": 3.87, "percentPercentile1": 3.80,
             "percentPercentile25": 3.84, "percentPercentile75": 3.92, "percentPercentile99": 3.97,
             "volumeInBillions": 3000} for d in days]


def _ofr_payload(final_until, prelim_until, start="2026-06-01"):
    out = {}
    days = [f"{d:%Y-%m-%d}" for d in pd.bdate_range(start, prelim_until)]
    finals = [d for d in days if d <= final_until]
    for svc, buckets in prepo.OFR_REPO_BUCKETS.items():
        for b in buckets:
            out[rp.ofr_mnemonic(svc, "AR", b, "P")] = [[d, 3.9] for d in days]
            out[rp.ofr_mnemonic(svc, "TV", b, "P")] = [[d, 2.5e12] for d in days]
            out[rp.ofr_mnemonic(svc, "AR", b, "F")] = [[d, 3.9] for d in finals]
            out[rp.ofr_mnemonic(svc, "TV", b, "F")] = [[d, 2.5e12] for d in finals]
    return out


def _paths(tmp_path) -> CyclePaths:
    return CyclePaths.under(tmp_path / "db")


# ------------------------------------------------------------------- parsing
def test_ofr_rows_keep_both_statuses_and_drop_disclosure_edits():
    payload = {"REPO-DVP_AR_OO-P": [["2026-09-01", 3.9], ["2026-09-02", None]],
               "REPO-DVP_TV_OO-P": [["2026-09-01", 2.5e12], ["2026-09-02", 1e12]],
               "REPO-DVP_AR_OO-F": [["2026-09-01", 3.91]], "REPO-DVP_TV_OO-F": [["2026-09-01", 2.5e12]]}
    df = rp.ofr_rates(payload, {"DVP": ("OO",)})
    assert len(df) == 2 and set(df["status"]) == {"final", "preliminary"}  # 09-02 had no rate
    assert df["volume_bn"].tolist() == [2500.0, 2500.0]
    best = rp.best(df)
    assert len(best) == 1 and best["rate"].iloc[0] == 3.91 and best["status"].iloc[0] == "final"


def test_gcf_workbook_parses_each_collateral():
    sheet = pd.DataFrame([[None] * 4, ["DTCC GCF Repo Index data is provided AS IS", None, None, None],
                          ["Date", "MBS GCF Repo®\nWeighted Average Rate", "Treasury GCF Repo®\nWeighted\nAverage Rate",
                           "Agency GCF Repo®\nWeighted\nAverage Rate"],
                          [pd.Timestamp("2005-01-03"), 2.314, 2.176, 2.269], [pd.Timestamp("2024-12-31"), 4.554, 4.545, None]])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        sheet.to_excel(w, sheet_name=rp.GCF_SHEET, header=False, index=False)
    df = rp.gcf_index(buf.getvalue())
    assert set(df["series"]) == {"DTCC_GCF_MBS", "DTCC_GCF_TSY", "DTCC_GCF_AGENCY"}
    assert len(df) == 5  # Agency has no 2024 value
    assert df.loc[(df["series"] == "DTCC_GCF_TSY") & (df["timestamp"] == "2024-12-31"), "rate"].item() == 4.545


def test_lending_rows_normalise_the_api():
    ops = [{"operationId": "SL 1", "operationType": "Securities Lending", "operationDate": "2005-01-03",
            "maturityDate": "2005-01-04", "auctionStatus": "Results",
            "details": [{"cusip": "912795zj3", "securityDescription": "B 05/31/07", "parAmtSubmitted": 5e6,
                         "parAmtAccepted": 5e6, "weightedAverageRate": 1.2, "somaHoldings": 0, "theoAvailToBorrow": 0,
                         "actualAvailToBorrow": 0, "outstandingLoans": 0},
                        {"cusip": "912828AB1", "securityDescription": "T 4 01/15/10", "parAmtSubmitted": 3e6,
                         "parAmtAccepted": 0, "weightedAverageRate": '"N/A"', "somaHoldings": 0,
                         "theoAvailToBorrow": 0, "actualAvailToBorrow": 0, "outstandingLoans": 0}]},
           {"operationId": "EX 1", "operationType": "Extensions", "operationDate": "2005-01-03",
            "details": [{"cusip": "912828AB1", "securityDescription": "T 4 01/15/10", "parAmtExtended": 6e8}]}]
    df = sl.parse_operations(ops)
    assert "912795ZJ3" in set(df["cusip"])  # lowercase from the API
    assert set(df["security_class"]) == {"treasury"}
    assert sl.security_class(pd.Series(["912828AB1", "3137EAAJ8"])).tolist() == ["treasury", "agency"]
    lend = df[df["operation_type"] == "lending"].set_index("cusip")
    assert lend.loc["912795ZJ3", "fee"] == 1.2 and pd.isna(lend.loc["912828AB1", "fee"])  # nothing accepted
    assert lend[["soma_holdings", "actual_available"]].isna().all().all()  # all-zero op = not reported
    ext = df[df["operation_type"] == "extension"]
    assert ext["par_extended"].item() == 6e8
    back = sl.decode(sl.encode(df))
    assert back.loc[back["cusip"] == "912795ZJ3", "fee"].item() == 1.2 and back["par_extended"].max() == 6e8


def test_fee_floor_follows_the_schedule():
    schedule = (("2007-08-21", 0.50), ("2008-10-08", 0.10))
    floor = sl.fee_floor(["2007-08-20", "2007-08-21", "2008-10-07", "2008-10-08"], schedule)
    assert floor.isna().iloc[0] and floor.iloc[1:].tolist() == [0.5, 0.5, 0.1]
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2008-10-07", "2008-10-08"]), "fee": [0.75, 0.05]})
    assert sl.excess_fee(df, schedule).round(6).tolist() == [0.25, 0.0]


# ------------------------------------------------------------- plan / store
def test_nyfed_days_are_covered_only_once_settled(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.pipeline.repo.NYFED_FETCH", _nyfed)
    cov = tmp_path / "cov.parquet"
    plan = prepo.plan_repo_update("2026-09-21", "2026-10-02", now=NOW, coverage_file=cov, sources=("nyfed",))
    assert plan["nyfed:SOFR"] == (pd.Timestamp("2026-09-21"), pd.Timestamp("2026-10-01"))  # never today
    prepo.fetch_and_store_repo(plan, now=NOW, root=tmp_path / "Repo", coverage_file=cov)
    again = prepo.plan_repo_update("2026-09-21", "2026-10-02", now=NOW, coverage_file=cov, sources=("nyfed",))
    assert again["nyfed:SOFR"] == (pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-01"))  # < REPO_SETTLE_DAYS old
    assert prepo.read_repo("2026-09-21", "2026-10-03", series=["TGCR"], root=tmp_path / "Repo")["rate"].eq(3.87).all()


def test_ofr_is_covered_only_through_its_last_final_day(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.pipeline.repo.OFR_FETCH",
                        lambda mnemonics, start=None: _ofr_payload("2026-06-30", "2026-09-30", start=start))
    cov = tmp_path / "cov.parquet"
    plan = prepo.plan_repo_update("2026-06-01", "2026-10-02", now=NOW, coverage_file=cov, sources=("ofr",))
    prepo.fetch_and_store_repo(plan, now=NOW, root=tmp_path / "Repo", coverage_file=cov)
    again = prepo.plan_repo_update("2026-06-01", "2026-10-02", now=NOW, coverage_file=cov, sources=("ofr",))
    assert again["ofr"][0] == pd.Timestamp("2026-07-01")  # preliminary days asked again until final
    df = prepo.read_repo("2026-06-01", "2026-10-01", series=["OFR_DVP_OO"], status="all", root=tmp_path / "Repo")
    assert df.groupby("status")["timestamp"].max().to_dict() == {
        "final": pd.Timestamp("2026-06-30"), "preliminary": pd.Timestamp("2026-09-30")}


def test_the_frozen_gcf_workbook_is_fetched_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("infra.pipeline.repo.GCF_FETCH", lambda: calls.append(1) or b"")
    monkeypatch.setattr("infra.processing.repo.gcf_index", lambda b: rp._empty())
    cov = tmp_path / "cov.parquet"
    plan = prepo.plan_repo_update("2016-01-01", "2026-10-02", now=NOW, coverage_file=cov, sources=("dtcc_gcf",))
    prepo.fetch_and_store_repo(plan, now=NOW, root=tmp_path / "Repo", coverage_file=cov)
    assert calls == [1]
    assert prepo.plan_repo_update("2016-01-01", "2026-10-02", now=NOW, coverage_file=cov, force_refetch=True,
                                  sources=("dtcc_gcf",)) == {}
    assert prepo.plan_repo_update("2025-01-01", "2026-10-02", now=NOW, coverage_file=tmp_path / "x.parquet",
                                  sources=("dtcc_gcf",)) == {}  # window past the workbook's end


def test_latest_rate_is_point_in_time(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.pipeline.repo.NYFED_FETCH", lambda rate, s, e: [
        {**r, "percentRate": 3.0 + i / 100} for i, r in enumerate(_nyfed(rate, s, e))])
    plan = prepo.plan_repo_update("2026-09-21", "2026-10-02", now=NOW, coverage_file=tmp_path / "c.parquet",
                                  sources=("nyfed",))
    prepo.fetch_and_store_repo(plan, now=NOW, root=tmp_path / "Repo", coverage_file=tmp_path / "c.parquet")
    day, rate = prepo.latest_repo_rate("SOFR", "2026-09-25", root=tmp_path / "Repo")
    assert day == pd.Timestamp("2026-09-24") and rate == pytest.approx(3.03)  # D is published on D+1
    assert prepo.latest_repo_rate("SOFR", "2026-09-25", publication_lag_days=0, root=tmp_path / "Repo")[0] == \
        pd.Timestamp("2026-09-25")


def test_lending_plan_is_chunked_and_settles(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("infra.pipeline.sec_lending.FETCH", lambda s, e: seen.append((s, e)) or [])
    cov = tmp_path / "cov.parquet"
    ranges = psl.plan_sec_lending_update("2024-01-01", "2026-10-02", now=NOW, coverage_file=cov)
    assert len(ranges) == 3 and ranges[-1][1] == pd.Timestamp("2026-10-01")
    psl.fetch_and_store_sec_lending(ranges, now=NOW, root=tmp_path / "SL", coverage_file=cov)
    assert seen == ranges
    assert psl.plan_sec_lending_update("2024-01-01", "2026-10-02", now=NOW, coverage_file=cov) == \
        [(pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-01"))]


# ------------------------------------------------------------------ cycle
def test_the_cycle_rechecks_ofr_days_not_final_yet(tmp_path, monkeypatch):
    """A 3-day scheduled window still asks OFR from its first non-final day, ~3 months back."""
    monkeypatch.setattr("infra.pipeline.repo.OFR_FETCH",
                        lambda mnemonics, start=None: _ofr_payload("2026-06-30", "2026-09-30", start=start))
    paths = _paths(tmp_path)
    prepo.fetch_and_store_repo({"ofr": (pd.Timestamp("2026-03-02"), pd.Timestamp("2026-09-30"))}, now=NOW,
                               root=paths.repo_dir, coverage_file=paths.repo_coverage)
    plan, _ = px_repo.plan_daily_repo_px("2026-09-28", "2026-10-01", paths=paths, force_refetch=True, now=NOW)
    assert plan["ofr"] == (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01"))


def _ctx(paths, start, end, output=None) -> StepContext:
    return StepContext(start=pd.Timestamp(start), end=pd.Timestamp(end), paths=paths, force_refetch=False,
                       run_day=pd.Timestamp(end) + pd.Timedelta(days=1), output=output or {}, options={})


def test_lending_checks_catch_a_fee_below_the_floor_and_an_unknown_cusip(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-30"] * 2), "operation_id": "SL 1",
                         "operation_type": "lending", "cusip": ["912828ZZ9", "91282CAA9"], "security_class": "treasury",
                         "description": "T",
                         "loan_maturity": pd.to_datetime(["2026-10-01"] * 2), "par_submitted": 1e7, "par_accepted": 1e7,
                         "fee": [0.01, 0.05], "soma_holdings": 1e9, "theo_available": 9e8, "actual_available": 9e8,
                         "outstanding_loans": 0.0, "par_extended": float("nan")})
    psl.store_sec_lending(rows, root=paths.sec_lending_dir)
    monkeypatch.setattr("infra.cycle.px_repo.read_auctions", lambda **k: pd.DataFrame({"cusip": ["91282CAA9"]}))
    monkeypatch.setattr("infra.cycle.px_repo.read_securities", lambda **k: pd.DataFrame({"cusip": []}))
    ok, msg, details = px_repo._check_lending_sane(_ctx(paths, "2026-09-30", "2026-09-30"))
    assert not ok and set(details["problem"]) == {"fee below the minimum fee", "CUSIP not a known Treasury"}
    assert set(details["cusip"]) == {"912828ZZ9"}


def test_repo_sane_flags_percentiles_out_of_order(tmp_path):
    paths = _paths(tmp_path)
    df = rp.nyfed_rates(_nyfed("SOFR", "2026-09-29", "2026-09-30"))
    df.loc[1, "p25"] = 3.95  # above the median
    prepo.store_repo(df, root=paths.repo_dir)
    ok, msg, details = px_repo._check_repo_sane(_ctx(paths, "2026-09-29", "2026-09-30"))
    assert not ok and details["problem"].tolist() == ["percentiles out of order"]
