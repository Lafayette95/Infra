"""German Federal securities: the Finanzagentur issuance history (parsing, the securities
view), the Bundesbank's BBSSY prices (parsing, storage, Rule 2.1 coverage) and the px
source with its checks. No network: a fake workbook and fake CSV."""
from __future__ import annotations

import io

import pandas as pd

from infra.cycle import px_bunds
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import bunds
from infra.processing import bunds as pb

D = pd.Timestamp
HEAD = "DATAFLOW;BBK_STD_FREQ;BBK_SEIS_ITEM;BBK_STD_CURRENCY;BBK_SEIS_SECURITY_CLASS;BBK_SEIS_ISIN;BBK_SEIS_RATING;TIME_PERIOD;OBS_VALUE"


def _workbook(rows) -> bytes:
    top = [[None] * 19 for _ in range(5)] + [["Auction Results since 1999"] + [None] * 18]
    body = [[i + 1, D(d), isin, typ, cpn, D(mat), seg, 5000, kind, "Auc", 9000, 8000, 10, 4000, 99.5, 99.6, 2.5, 1000, 2.2]
            for i, (d, isin, typ, cpn, mat, seg, kind) in enumerate(rows)]
    foot = [[None], ["2 ", "N = new issue, R = reopening"]]
    buf = io.BytesIO()
    pd.DataFrame(top + [[None]] * 5 + body + foot).to_excel(buf, header=False, index=False)
    return buf.getvalue()


ROWS = [("2026-01-07", "DE000BU22999", "Schatz", 0.02, "2028-03-10", "2 Y", "N"),
        ("2026-02-04", "DE000BU22999", "Schatz", 0.02, "2028-03-10", "2 Y", "R"),
        ("2026-01-14", "DE000BU2Z999", "Bund", 0.0275, "2036-02-15", "10 Y", "N"),
        ("2026-01-05", "DE000BU0E999", "Bubill", None, "2026-07-15", "6 M", "N")]


def _csv(rows) -> str:
    lines = [HEAD] + [f"BBK:BBSSY(1.0);D;{item};EUR;{cls};{isin};A;{day};{val}" for item, cls, isin, day, val in rows]
    return "\n".join(lines) + "\n"


def test_issuance_history_and_the_securities_view():
    a = pb.parse_issuance_history(_workbook(ROWS))
    assert len(a) == 4 and set(a["issue_kind"]) == {"N", "R"}
    assert abs(a.loc[a["isin"] == "DE000BU2Z999", "coupon"].iloc[0] - 2.75) < 1e-12  # the file gives 0.0275
    s = pb.securities(a).set_index("isin")
    assert s.loc["DE000BU22999", "first_auction"] == D("2026-01-07")  # the first issuance, not the reopening
    assert s.loc["DE000BU22999", "issue_date"] == D("2026-01-09") and s.loc["DE000BU22999", "tenor_years"] == 2.0
    assert bool(s.loc["DE000BU0E999", "bill"]) and s.loc["DE000BU0E999", "tenor_years"] == 0.5


def test_bbssy_csv_pivots_items_and_drops_blank_days():
    df = pb.parse_bbssy_csv(_csv([("KCP", "A610", "DE000BU22999", "2026-10-05", "99.875"),
                                  ("REN", "A610", "DE000BU22999", "2026-10-05", "2.0612"),
                                  ("KCP", "A610", "DE000BU22999", "2026-10-04", ".")]))
    assert len(df) == 1 and df.iloc[0]["price_clean"] == 99.875 and pd.isna(df.iloc[0]["price_dirty"])
    rt = pb.decode_prices(pb.encode_prices(df))
    assert abs(rt.iloc[0]["yield"] - 2.0612) < 1e-9


def test_prices_are_stored_and_a_covered_range_never_asked_again(tmp_path, monkeypatch):
    calls = []
    def fetch(start, end, isin=""):
        calls.append((start, end, isin))
        return _csv([("REN", "A630", "DE000BU2Z999", "2026-10-05", "2.70"), ("KCP", "A630", "DE000BU2Z999", "2026-10-05", "100.5")])
    monkeypatch.setattr(bunds, "FETCH_PRICES", fetch)
    root, cov = tmp_path / "p", tmp_path / "c.parquet"
    r = bunds.plan_prices("2026-10-01", "2026-10-05", now=D("2026-10-07"), coverage_file=cov)
    assert bunds.fetch_and_store_prices(r, root=root, coverage_file=cov) == 1
    assert bunds.plan_prices("2026-10-01", "2026-10-05", now=D("2026-10-07"), coverage_file=cov) == []
    assert bunds.plan_prices("2026-10-01", "2026-10-09", now=D("2026-10-07"), coverage_file=cov)[0][1] == D("2026-10-07")  # never past settled
    assert bunds.read_bund_prices("2026-10-01", "2026-10-06", root=root).iloc[0]["yield"] == 2.70


def test_px_source_and_checks_flag_an_unpriced_outstanding_bond(tmp_path, monkeypatch):
    monkeypatch.setattr(bunds, "FETCH_ISSUANCE", lambda: _workbook(ROWS))
    monkeypatch.setattr(bunds, "FETCH_PRICES", lambda start, end, isin="": _csv(
        [("REN", "A630", "DE000BU2Z999", "2026-10-05", "2.70"), ("KCP", "A630", "DE000BU2Z999", "2026-10-05", "100.5")]))
    paths = CyclePaths.under(tmp_path)
    out = px_bunds.backfill_daily_bund_px("2026-10-05", "2026-10-05", paths=paths)
    assert out["error"] is None and out["auctions"] == 4
    assert len(out["planned"]) == 2  # the window, and the never-covered 30 days before it
    assert len(bunds.read_bund_prices("2026-09-01", "2026-10-06", root=paths.bund_prices_dir)) == 1
    ctx = StepContext(D("2026-10-05"), D("2026-10-05"), D("2026-10-06"), paths, output={"bunds": out})
    checks = {c.name: c.fn(ctx) for c in px_bunds.BUND_CHECKS[:4]}
    assert checks["bunds_fetch_ok"][0] and checks["bunds_sane"][0]
    ok, _, details = checks["bunds_complete"]
    assert not ok and list(details["isin"]) == ["DE000BU22999"]  # the Schatz has no price that day


def test_otr_map_ranks_by_issue_day_and_the_held_bond_pnl(tmp_path, monkeypatch):
    rows = [("2026-01-07", "DE000BU22111", "Schatz", 0.02, "2028-03-10", "2 Y", "N"),
            ("2026-04-08", "DE000BU22222", "Schatz", 0.025, "2028-06-10", "2 Y", "N")]
    monkeypatch.setattr(bunds, "FETCH_ISSUANCE", lambda: _workbook(rows))
    paths = CyclePaths.under(tmp_path)
    bunds.update_auctions(root=paths.de_auctions_dir)
    m = bunds.otr_map("2026-04-08", "2026-04-13", root=paths.de_auctions_dir)
    on = m[m["rank"] == 0].set_index("timestamp")["isin"]
    assert on[D("2026-04-09")] == "DE000BU22111" and on[D("2026-04-10")] == "DE000BU22222"  # issue = auction + 2 weekdays
    # P&L on the bond held the previous day: the switch day measures the OLD bond's move
    px = [("REN", "A610", isin, day, y) for isin, day, y in
          (("DE000BU22111", "2026-04-09", "2.00"), ("DE000BU22111", "2026-04-10", "2.05"), ("DE000BU22222", "2026-04-10", "2.40"))]
    monkeypatch.setattr(bunds, "FETCH_PRICES", lambda start, end, isin="": _csv(px))
    bunds.fetch_and_store_prices([(D("2026-04-09"), D("2026-04-11"))], root=paths.bund_prices_dir,
                                 coverage_file=paths.bund_prices_coverage)
    from infra.cycle.bmk_yields import _de_otr_pnl
    pnl = _de_otr_pnl(D("2026-04-08"), D("2026-04-11"), paths)
    row = pnl.set_index("timestamp").loc[D("2026-04-10")]
    assert row["ticker"] == "DE_BOND_2y" and abs(row["pnl_per_dv01"] + 5.0) < 1e-9 and row["currency"] == "EUR"


# ------------------------------------------------------------------ the fitted curve
def test_long_first_coupon_flows_and_the_implied_interest_start():
    # a new 10y issued 2026-07-10, maturing 2036-08-15: the 2026-08-15 anniversary is skipped,
    # the first coupon (2027-08-15) pays a full coupon plus the 36-day stub
    t, a = pb.first_period_flows(3.0, D("2036-08-15"), D("2026-10-09"), D("2026-07-10"))
    assert abs(a[0] - 3.0 * (1 + 36 / 365)) < 1e-12 and len(a) == 10 and abs(a[-1] - 103.0) < 1e-12
    assert abs(t[0] - (D("2027-08-15") - D("2026-10-09")).days / 365.25) < 1e-12
    short = pb.first_period_flows(3.0, D("2036-08-15"), D("2026-07-20"), D("2026-07-10"), short_first=True)
    assert abs(short[1][0] - 3.0 * 36 / 365) < 1e-12               # a short first coupon pays the stub only
    assert pb.first_period_flows(3.0, D("2036-08-15"), D("2027-09-01"), D("2026-07-10")) is None  # past it: regular
    assert pb.first_period_flows(3.0, D("2036-08-15"), D("2026-08-20"), D("2026-08-14")) is None  # starts on an anniversary
    # accrued 3.0 x (89 + 0) / 365 at settlement 2026-10-09 implies interest from 2026-07-12 on a regular period
    acc = 3.0 * (D("2026-10-09") - D("2026-07-12")).days / 365
    assert pb.implied_commencement(acc, 3.0, D("2026-10-09"), D("2036-08-15")) == D("2026-07-12")


def test_a_bond_whose_published_accrued_disagrees_with_our_settlement_is_left_out():
    from infra.pipeline.bund_curves import settlement_mismatch
    sec = pd.DataFrame({"isin": ["A", "B"], "coupon": [2.5, 2.5], "maturity_date": [D("2028-12-12"), D("2030-08-15")]}).set_index("isin")
    settle = D("2026-12-12")      # A's coupon date: our accrued 0, the published one almost a whole coupon
    g = pd.DataFrame({"isin": ["A", "B"], "accrued": [2.493, 2.5 * (settle - D("2026-08-15")).days / 365]})
    fp = pd.DataFrame(columns=["commencement", "short_first"])
    assert settlement_mismatch(g, sec, settle, fp) == {"A"}
