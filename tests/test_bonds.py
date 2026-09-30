"""Cash-bond par yields: source parsing, the par derivation, the pipeline's caching
(Rule 2.1) and the px step's bond half with its checks. No network - every source is a
fake passed in explicitly (tests/conftest.py blocks the real ones)."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from infra.api import boe_client, bundesbank_client, treasury_client
from infra.config import BOND_CURVES, BondCurve
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.px_bonds import BOND_CHECKS, GAP_LOOKBACK_DAYS, backfill_daily_bond_px
from infra.pipeline import bonds as pb
from infra.processing import bond_curves as bc

TENORS = (2, 3, 5, 7, 10, 20, 30)


# ------------------------------------------------------------------ sources
def test_treasury_csv_reads_tenors_by_name_and_covers_only_published_days():
    text = ('Date,"1 Mo","2 Mo","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"\n'
            "09/29/2026,4.04,4.18,4.89,4.98,5.06,5.16,5.26,5.64,5.59\n"
            "09/25/2026,4.04,4.20,4.81,4.94,4.98,5.06,5.17,,5.49")
    df, covered = treasury_client.fetch_par_curve(pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-01"),
                                                  fetch_year=lambda y: text)
    assert set(df["maturity"]) == {2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0}  # no month tenors
    assert df.loc[(df["timestamp"] == "2026-09-29") & (df["maturity"] == 10.0), "value"].item() == 5.26
    assert len(df[df["timestamp"] == "2026-09-25"]) == 6  # the blank 20y is dropped, not zero
    # 09-30 isn't published yet: covered only up to the last published day
    assert covered == [(pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-30"))]


def test_bundesbank_params_one_row_per_complete_day_covering_only_published_days():
    head = "DATAFLOW;BBK_SEIS_ITEM;BBK_SEIS_VALUATION;TIME_PERIOD;OBS_VALUE\n"
    rows = [f"BBK:BBSIS(1.0);ZST;{p};{d};{v}" for d, v in [("2026-09-29", "1.5"), ("2026-09-30", "2.5")]
            for p in bundesbank_client.PARAMS]
    rows += [f"BBK:BBSIS(1.0);ZST;{p};2026-09-27;." for p in bundesbank_client.PARAMS]  # a weekend, blank
    df, covered = bundesbank_client.fetch_svensson_params(pd.Timestamp("2026-09-27"), pd.Timestamp("2026-10-03"),
                                                          fetch=lambda s, e: head + "\n".join(rows))
    assert list(df["timestamp"].dt.strftime("%m-%d")) == ["09-29", "09-30"] and df["B0"].tolist() == [1.5, 2.5]
    assert covered == [(pd.Timestamp("2026-09-27"), pd.Timestamp("2026-10-01"))]


def test_svensson_reproduces_the_bundesbanks_own_curves():
    """Real Bundesbank parameters for 2026-09-30 and its own published values that day:
    zero curve (ZST) and annual-coupon par curve (ZAR), both 2 decimals."""
    params = pd.DataFrame([{"timestamp": pd.Timestamp("2026-09-30"), "B0": 4.18173, "B1": -1.90569,
                            "B2": 18.84971, "B3": -19.00801, "T1": 1.36147, "T2": 1.50953}])
    curve = bc.svensson_curve(params)
    par = bc.PAR_METHODS["annual_from_annual_spot"](curve, TENORS).set_index("tenor")["par_yield"]
    published_par = {2: 3.24, 3: 3.29, 5: 3.36, 7: 3.46, 10: 3.60, 20: 3.83, 30: 3.90}
    for t, v in published_par.items():
        assert abs(par[t] - v) <= 0.005 + 1e-9, (t, par[t], v)  # within the publication's rounding
    semi = bc.PAR_METHODS["semiannual_from_annual_spot"](curve, TENORS).set_index("tenor")["par_yield"]
    assert ((par - semi) * 100).between(2.5, 4.0).all()  # annual vs semi-annual basis: ~3bp here


def _spot_workbook(days: list[str], maturities=(0.5, 1.0, 1.5, 2.0), value=4.0, holiday=None) -> bytes:
    header = [[None, "UK nominal spot curve"], [None], ["Maturity"], ["years:", *maturities], [None]]
    rows = [[pd.Timestamp(d), *([None] * len(maturities) if d == holiday else [value] * len(maturities))]
            for d in days]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xl:
        pd.DataFrame(header + rows).to_excel(xl, sheet_name=boe_client.SPOT_SHEET, header=False, index=False)
    return buf.getvalue()


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_boe_seam_between_archive_and_current_month_is_covered():
    archive = _zip({"GLC Nominal daily data_2025 to present.xlsx":
                    _spot_workbook(["2026-08-27", "2026-08-28", "2026-08-31"], holiday="2026-08-31")})
    latest = _zip({"GLC Nominal daily data current month.xlsx": _spot_workbook(["2026-09-01", "2026-09-02"])})
    fetch = {boe_client.ARCHIVE_ZIP: archive, boe_client.LATEST_ZIP: latest}.get
    df, covered = boe_client.fetch_spot_curve(pd.Timestamp("2026-08-27"), pd.Timestamp("2026-09-10"),
                                              fetch=fetch, now=pd.Timestamp("2026-09-15"))
    assert sorted(df["timestamp"].unique()) == list(pd.to_datetime(["2026-08-27", "2026-08-28", "2026-09-01", "2026-09-02"]))
    assert covered == [(pd.Timestamp("2026-08-27"), pd.Timestamp("2026-09-01")),
                       (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-03"))]


def test_boe_stale_archive_leaves_a_real_gap_uncovered():
    archive = _zip({"GLC Nominal daily data_2025 to present.xlsx": _spot_workbook(["2026-07-30", "2026-07-31"])})
    latest = _zip({"GLC Nominal daily data current month.xlsx": _spot_workbook(["2026-09-01"])})
    fetch = {boe_client.ARCHIVE_ZIP: archive, boe_client.LATEST_ZIP: latest}.get
    _, covered = boe_client.fetch_spot_curve(pd.Timestamp("2026-07-30"), pd.Timestamp("2026-09-02"),
                                             fetch=fetch, now=pd.Timestamp("2026-09-15"))
    assert covered == [(pd.Timestamp("2026-07-30"), pd.Timestamp("2026-08-01")),
                       (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"))]  # August never claimed


# ------------------------------------------------------------------ par derivation
def _flat_spot(rate: float, days=("2026-09-28",)) -> pd.DataFrame:
    grid = np.arange(0.5, 40.01, 0.5)
    return pd.DataFrame([(pd.Timestamp(d), m, rate) for d in days for m in grid],
                        columns=["timestamp", "maturity", "value"])


def test_par_of_a_flat_semiannual_spot_curve_is_the_same_rate():
    par = bc.PAR_METHODS["semiannual_from_spot"](_flat_spot(4.5), TENORS)
    assert np.allclose(par["par_yield"], 4.5)


def test_par_of_a_flat_continuous_spot_curve_is_its_semiannual_equivalent():
    par = bc.PAR_METHODS["semiannual_from_continuous_spot"](_flat_spot(4.5), TENORS)
    assert np.allclose(par["par_yield"], 2 * (np.exp(0.045 / 2) - 1) * 100)


def test_par_needs_every_coupon_date_on_the_grid():
    spot = _flat_spot(4.0)
    spot = spot[spot["maturity"] != 24.5]  # a hole inside the 30y's coupon dates only
    par = bc.PAR_METHODS["semiannual_from_spot"](spot, TENORS)
    assert set(par["tenor"]) == {2, 3, 5, 7, 10, 20}  # no guessed 30y


def test_rows_are_absolute_bond_tickers_and_round_trip_through_disk_encoding():
    rows = bc.to_par_rows(_flat_spot(4.123456), BOND_CURVES["UK"])
    assert list(rows["ticker"]) == sorted(f"UK_BOND_{t}y" for t in TENORS)
    back = bc.decode_bonds(bc.encode_bonds(rows))
    assert np.allclose(back["par_yield"], 4.1235)  # x10000 fixed point: 0.01bp


# ------------------------------------------------------------------ pipeline + cycle
US = BondCurve("US", "fake", "test curve", "USD", convention="par", history_start="2020-01-01")


def _fake_source(values: dict[str, float] | None = None, published_until="2026-09-30", calls=None):
    """Par yields at every tenor for every weekday up to ``published_until`` (exclusive):
    ``base + tenor/100`` plus an optional override per (day) -> bump on the 10y."""
    values = values or {}

    def fetch(start, end):
        if calls is not None:
            calls.append((start, end))
        last = min(end, pd.Timestamp(published_until))
        days = pd.bdate_range(start, last - pd.Timedelta(days=1))
        rng = np.random.default_rng(0)
        rows = []
        for i, d in enumerate(days):
            level = 4.0 + 0.01 * np.sin(i / 3) + 0.002 * rng.standard_normal()
            for t in TENORS:
                v = level + t / 100 + (values.get(str(d.date()), 0.0) if t == 10 else 0.0)
                rows.append((d, float(t), v))
        df = pd.DataFrame(rows, columns=["timestamp", "maturity", "value"])
        return df, ([(start, last)] if rows else [])

    return fetch


def test_load_bonds_never_refetches_a_covered_range(tmp_path):
    calls = []
    kw = dict(root=tmp_path / "b", coverage_file=tmp_path / "c.parquet", curves={"US": US},
              sources={"fake": _fake_source(calls=calls)})
    first = pb.load_bonds(["US"], "2026-09-01", "2026-09-10", **kw)
    again = pb.load_bonds(["US"], "2026-09-01", "2026-09-10", **kw)
    assert len(calls) == 1 and len(first) == len(again) == 7 * 7


def test_an_unpublished_day_is_not_claimed_covered(tmp_path):
    calls = []
    kw = dict(root=tmp_path / "b", coverage_file=tmp_path / "c.parquet", curves={"US": US})
    pb.load_bonds(["US"], "2026-09-21", "2026-09-26", sources={"fake": _fake_source(published_until="2026-09-24")}, **kw)
    pb.load_bonds(["US"], "2026-09-21", "2026-09-26", sources={"fake": _fake_source(calls=calls)}, **kw)
    assert calls == [(pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-26"))]


def _ctx(paths, start, end, output):
    return StepContext(pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(end), paths, output={"bonds": output})


def _checks(ctx):
    return {c.name: c.run(ctx) for c in BOND_CHECKS}


def test_bond_px_step_stores_every_tenor_and_passes_its_checks(tmp_path):
    paths = CyclePaths.under(tmp_path)
    out = backfill_daily_bond_px("2026-03-02", "2026-09-29", paths=paths, curves={"US": US},
                                 sources={"fake": _fake_source()})
    first = pd.Timestamp("2026-03-02") - pd.Timedelta(days=GAP_LOOKBACK_DAYS)  # never-covered lookback too
    assert out["fetch_errors"] == {} and out["rows"] == len(pd.bdate_range(first, "2026-09-29")) * 7
    results = _checks(_ctx(paths, "2026-03-02", "2026-09-29", out))
    assert all(r.passed for r in results.values()), {n: r.message for n, r in results.items() if not r.passed}


def test_a_bad_print_on_one_tenor_is_na_d_in_the_adjustments_log(tmp_path):
    paths = CyclePaths.under(tmp_path)
    source = _fake_source(values={"2026-09-15": 0.40})  # 10y alone +40bp for one day, gone next day
    backfill_daily_bond_px("2026-03-02", "2026-08-31", paths=paths, curves={"US": US}, sources={"fake": source})
    out = backfill_daily_bond_px("2026-09-01", "2026-09-29", paths=paths, curves={"US": US}, sources={"fake": source})
    treated = out["bad_prints"]
    assert list(zip(treated["key"], treated["timestamp"], treated["action"])) == [
        ("US_BOND_10y", pd.Timestamp("2026-09-15"), "NA")]
    raw = pb.read_bonds_from_disk(["US_BOND_10y"], pd.Timestamp("2026-09-15"), pd.Timestamp("2026-09-16"),
                                  root=paths.daily_bonds_dir, adjusted=False)
    clean = pb.read_bonds_from_disk(["US_BOND_10y"], pd.Timestamp("2026-09-15"), pd.Timestamp("2026-09-16"),
                                    root=paths.daily_bonds_dir, adjustments_dir=paths.adjustments_dir)
    assert raw["par_yield"].notna().all() and clean["par_yield"].isna().all()  # raw kept, reader cleans
    assert not _checks(_ctx(paths, "2026-09-01", "2026-09-29", out))["bonds_outliers"].passed


def test_a_parallel_curve_shift_is_not_a_bad_print(tmp_path):
    paths = CyclePaths.under(tmp_path)

    def shifted(start, end):  # every tenor +40bp on one day and back: a market move, not a print
        df, cov = _fake_source()(start, end)
        df.loc[df["timestamp"] == "2026-09-15", "value"] += 0.40
        return df, cov

    out = backfill_daily_bond_px("2026-03-02", "2026-09-29", paths=paths, curves={"US": US}, sources={"fake": shifted})
    assert out["bad_prints"].empty


def test_a_missing_tenor_and_a_stale_curve_are_reported(tmp_path):
    paths = CyclePaths.under(tmp_path)

    def gappy(start, end):
        df, cov = _fake_source(published_until="2026-09-22")(start, end)
        return df[~((df["timestamp"] == "2026-09-18") & (df["maturity"] == 7.0))], cov

    out = backfill_daily_bond_px("2026-09-01", "2026-09-29", paths=paths, curves={"US": US}, sources={"fake": gappy})
    results = _checks(_ctx(paths, "2026-09-01", "2026-09-29", out))
    assert not results["bonds_complete"].passed and list(results["bonds_complete"].details["ticker"]) == ["US_BOND_7y"]
    assert not results["bonds_fresh"].passed and results["bonds_fresh"].severity.value == "warn"


def test_a_source_failure_is_collected_not_raised(tmp_path):
    def broken(start, end):
        raise ConnectionError("site down")

    out = backfill_daily_bond_px("2026-09-01", "2026-09-29", paths=CyclePaths.under(tmp_path),
                                 curves={"US": US}, sources={"fake": broken})
    assert out["fetch_errors"] == {"US": "ConnectionError: site down"}


@pytest.mark.parametrize("country", ["US", "UK", "DE"])
def test_configured_curves_cover_the_agreed_universe(country):
    spec = BOND_CURVES[country]
    assert spec.tenors == TENORS and spec.source in pb.SOURCES
    assert spec.par_method is None or spec.par_method in bc.PAR_METHODS


def test_short_end_is_held_flat_only_below_the_first_published_point():
    spot = pd.concat([_flat_spot(4.0, days=("2026-09-28",)), _flat_spot(4.0, days=("2026-09-29",))])
    spot.loc[(spot["timestamp"] == "2026-09-29") & (spot["maturity"] == 0.5), "value"] = np.nan  # starts at 1y
    spot.loc[(spot["timestamp"] == "2026-09-28") & (spot["maturity"] == 10.0), "value"] = np.nan  # a hole
    par = bc.PAR_METHODS["semiannual_from_spot"](spot, TENORS)
    day29, day28 = par[par["timestamp"] == "2026-09-29"], par[par["timestamp"] == "2026-09-28"]
    assert set(day29["tenor"]) == set(TENORS) and np.allclose(day29["par_yield"], 4.0)
    assert set(day28["tenor"]) == {2, 3, 5, 7}  # the 10y hole is never filled


def test_a_day_published_late_is_picked_up_by_a_later_scheduled_window(tmp_path):
    paths = CyclePaths.under(tmp_path)
    kw = dict(paths=paths, curves={"US": US}, force_refetch=True)
    pb.load_bonds(["US"], "2026-07-01", "2026-08-17", root=paths.daily_bonds_dir, curves={"US": US},
                  coverage_file=paths.daily_bonds_coverage, sources={"fake": _fake_source()})  # history
    backfill_daily_bond_px("2026-08-17", "2026-08-21", sources={"fake": _fake_source(published_until="2026-08-20")}, **kw)
    calls = []
    backfill_daily_bond_px("2026-08-24", "2026-08-28", sources={"fake": _fake_source(calls=calls)}, **kw)
    # the window, plus the 08-20/08-21 it has moved past (published late), in one request
    assert calls == [(pd.Timestamp("2026-08-20"), pd.Timestamp("2026-08-29"))]


def test_a_quiet_2dp_short_end_moving_with_its_neighbours_is_not_a_bad_print(tmp_path):
    """The DE 2y on 2026-03-09/10: +10/-12bp with the 3y/5y +9/-11 and -10, published to 2
    decimals - its own typical move rounds to ~1bp, so its z-score is inflated vs its
    neighbours'. Only a move out of line in bp too (BOND_MIN_ABS_DEV) may be flagged."""
    paths = CyclePaths.under(tmp_path)
    shock = {"2026-09-14": (0.10, 0.09, 0.09, 0.06, 0.04), "2026-09-15": (-0.12, -0.11, -0.10, -0.07, -0.04)}

    def quantised(start, end):
        days = pd.bdate_range(start, min(end, pd.Timestamp("2026-09-30")) - pd.Timedelta(days=1))
        rng = np.random.default_rng(1)
        typical = {2: 0.006, 3: 0.02, 5: 0.025, 7: 0.025, 10: 0.03, 20: 0.03, 30: 0.03}  # a quiet 2y
        rows = []
        for t in TENORS:
            walk = 2.0 + t / 100 + np.cumsum(rng.normal(0, typical[t], len(days)))
            for d, v in zip(days, walk):
                rows.append((d, float(t), v))
        df = pd.DataFrame(rows, columns=["timestamp", "maturity", "value"])
        for day, moves in shock.items():
            after = df["timestamp"] >= day
            for t, mv in zip((2, 3, 5, 7, 10), moves):
                df.loc[after & (df["maturity"] == t), "value"] += mv
        df["value"] = df["value"].round(2)
        return df, ([(start, days[-1] + pd.Timedelta(days=1))] if len(days) else [])

    out = backfill_daily_bond_px("2026-03-02", "2026-09-29", paths=paths, curves={"US": US}, sources={"fake": quantised})
    assert out["bad_prints"].empty
