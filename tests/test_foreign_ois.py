"""Foreign OIS curves (EUR, GBP, JPY, CAD): per-currency fixed-leg conventions, the ESR-strip
and BoE short ends, the BoE SONIA curve's fetch (both workbook layouts), store and point-in-
time nodes. No network: fake workbooks, temp stores."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from infra.analytics import swap_curve as sc
from infra.api import boe_client
from infra.config import OIS_CURVES, SWAP_CLOSES, SWAP_CURVES
from infra.pipeline import boe_ois
from infra.pipeline.ois_curves import _third_wednesday

D = pd.Timestamp


# ------------------------------------------------------------------ conventions
def test_cad_fixed_leg_is_semiannual_act365_beyond_one_year():
    c = SWAP_CURVES["CAD"]
    s = sc.swap_schedule(D("2026-10-05"), 2, c.spot_lag_days, freq_months=sc.fixed_freq(2, c.fixed_freq_months),
                         basis=c.day_basis, calendar=c.calendar)
    assert len(s.dates) == 4 and s.effective == D("2026-10-06")  # T+1, four semi-annual payments
    assert abs(s.accruals.sum() - 730 / 365) < 0.02
    assert sc.fixed_freq(1, 6) == 12  # one payment up to 1y


@pytest.mark.parametrize("ccy", ["EUR", "GBP", "JPY", "CAD"])
def test_bootstrap_reprices_every_pillar_in_its_own_conventions(ccy):
    c = SWAP_CURVES[ccy]
    quotes = {1: 2.0, 2: 2.1, 5: 2.3, 10: 2.6, 30: 2.9}
    _, nodes = sc.bootstrap_ois(D("2026-10-05"), quotes, spot_lag=c.spot_lag_days, freq_months=c.fixed_freq_months,
                                basis=c.day_basis, calendar=c.calendar)
    assert nodes.loc[nodes.source == "swap", "reprice_bp"].abs().max() < 1e-8


def test_usd_conventions_are_unchanged():
    c = SWAP_CURVES["USD"]
    assert (c.day_basis, c.fixed_freq_months, c.calendar) == (360.0, 12, "us")


def test_short_nodes_stop_three_months_before_the_first_pillar():
    short = [(k / 12, 1 - 0.003 * k) for k in range(1, 13)]
    _, nodes = sc.bootstrap_ois(D("2026-10-05"), {1: 3.6, 2: 3.7, 5: 3.8}, short_nodes=short)
    assert nodes.loc[nodes.source == "short_end", "t_years"].max() < 0.75


# ------------------------------------------------------------------ ESR strip
def test_third_wednesday_and_esr_quarters_run_imm_to_imm():
    assert _third_wednesday(2026, 12) == D("2026-12-16") and _third_wednesday(2027, 3) == D("2027-03-17")


def test_strip_nodes_compound_quarters_and_start_the_live_quarter_today():
    day = D("2026-10-05")
    periods = [(D("2026-09-16"), D("2026-12-16"), 2.0), (D("2026-12-16"), D("2027-03-17"), 1.8),
               (D("2027-03-17"), D("2027-06-16"), 1.7), (D("2027-06-16"), D("2027-09-15"), 1.7)]
    nodes = sc.strip_nodes(day, periods, months=9)
    assert len(nodes) == 3  # the September 2027 quarter end is beyond 9 months
    d1 = 1 / (1 + 0.02 * 72 / 360)
    assert abs(nodes[0][1] - d1) < 1e-12 and abs(nodes[1][1] - d1 / (1 + 0.018 * 91 / 360)) < 1e-12


def test_strip_stops_at_a_hole():
    periods = [(D("2026-09-16"), D("2026-12-16"), 2.0), (D("2027-03-17"), D("2027-06-16"), 1.7)]
    assert len(sc.strip_nodes(D("2026-10-05"), periods, months=12)) == 1


# ------------------------------------------------------------------ BoE SONIA curve
def _sheet(days, maturities, value):
    return [[None, "UK OIS spot curve"], [None], ["months:", *[m * 12 for m in maturities]], ["years:", *maturities],
            [None]] + [[D(d), *[value] * len(maturities)] for d in days]


def _workbook(sheets: dict[str, list]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xl:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(xl, sheet_name=name, header=False, index=False)
    return buf.getvalue()


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    return buf.getvalue()


def test_boe_ois_reads_both_workbook_layouts_and_merges_the_sheets():
    old = _workbook({"info": [["x"]], "1. fwd curve": [["x"]], "2. spot curve": _sheet(["2015-12-30"], [1 / 12, 0.5], 0.5)})
    new = _workbook({"info": [["x"]], "3. spot, short end": _sheet(["2026-08-28"], [1 / 12, 0.5, 1.0], 4.0),
                     "4. spot curve": _sheet(["2026-08-28"], [0.5, 1.0, 1.5, 25.0], 4.1)})
    latest = _workbook({"3. spot, short end": _sheet(["2026-09-01"], [1 / 12, 0.5], 4.0),
                        "4. spot curve": _sheet(["2026-09-01"], [0.5, 2.0], 4.1)})
    files = {boe_client.OIS_ARCHIVE_ZIP: _zip({"OIS daily data_2009 to 2015.xlsx": old,
                                               "OIS daily data_2025 to present.xlsx": new}),
             boe_client.LATEST_ZIP: _zip({"OIS daily data current month.xlsx": latest})}
    df, covered = boe_client.fetch_ois_curve(D("2015-12-01"), D("2026-09-05"), fetch=files.get, now=D("2026-09-15"))
    day = df[df.timestamp == D("2026-08-28")].set_index("maturity")["value"]
    assert list(day.index) == [1 / 12, 0.5, 1.0, 1.5, 25.0]  # short end to 1y, the curve sheet beyond
    assert day[1.0] == 4.0 and day[1.5] == 4.1
    assert D("2015-12-30") in set(df.timestamp)  # the 2009-2015 single-sheet layout


def test_boe_ois_store_and_point_in_time_nodes(tmp_path, monkeypatch):
    rows = pd.DataFrame([(D(d), m, v) for d, v in (("2026-10-01", 4.0), ("2026-10-02", 3.0)) for m in (0.25, 0.5, 2.0)],
                        columns=["timestamp", "maturity", "value"])
    monkeypatch.setattr(boe_ois, "FETCH", lambda s, e: (rows, [(s, e)]))
    root, cov = tmp_path / "BoeOIS", tmp_path / "cov.parquet"
    assert boe_ois.fetch_and_store(boe_ois.plan("2026-10-01", "2026-10-02", coverage_file=cov), root=root,
                                   coverage_file=cov) == 6
    assert boe_ois.plan("2026-10-01", "2026-10-02", coverage_file=cov) == []  # covered: never asked again
    nodes = boe_ois.boe_ois_nodes("2026-10-02", max_years=1.0, root=root)  # the day before only
    assert [t for t, _ in nodes] == [0.25, 0.5] and abs(nodes[1][1] - (1.02) ** -1) < 1e-12


# ------------------------------------------------------------------ config
def test_foreign_curves_and_closes_are_configured():
    assert {"EUR_ESTR", "GBP_SONIA", "JPY_TONA", "CAD_CORRA"} <= set(OIS_CURVES)
    for spec in OIS_CURVES.values():
        assert spec.currency in SWAP_CLOSES[spec.close].currencies
    assert SWAP_CLOSES["LDN1615"].for_currency("USD") is SWAP_CLOSES["LDN1615"]  # USD's calibrated windows
    assert SWAP_CLOSES["LDN1615"].for_currency("EUR").pure_fallback_half_window_min == 240


# ------------------------------------------------------------------ thin-curve fills
def test_fill_ends_fills_the_last_tenor_and_a_run_of_missing_tenors_point_in_time():
    days = pd.bdate_range("2026-09-01", periods=3)
    r = pd.DataFrame({2: [2.0, 2.1, 2.2], 5: [2.3, 2.4, 2.5], 10: [2.6, 2.7, 2.8], 20: [2.9, np.nan, np.nan],
                      30: [2.8, np.nan, np.nan]}, index=days)
    out, filled = sc.fill_missing_tenors(r, fill_ends=True)
    assert abs(out.loc[days[1], 30] - (2.7 + 0.2)) < 1e-12  # 10y + its last observed 30y-10y spread
    assert abs(out.loc[days[2], 20] - (2.8 + 0.3)) < 1e-12 and filled.loc[days[2], [20, 30]].all()
    plain, f0 = sc.fill_missing_tenors(r)  # the USD default: ends never filled
    assert not f0.loc[:, 30].any() and plain.loc[days[1], 30] != plain.loc[days[1], 30]


def test_a_partial_store_replaces_only_its_own_curves(tmp_path):
    from infra.pipeline.ois_curves import COLUMNS, read_ois_curves, store_ois_curves
    def rows(curve, df):
        return pd.DataFrame([{"timestamp": D("2026-10-01 15:15"), "curve": curve, "node": "1Y", "t_years": 1.0, "df": df,
                              "zero_pct": 3.0, "source": "swap", "input_rate": 3.0, "n_trades": 2, "se_bp": 0.5}])[COLUMNS]
    store_ois_curves(pd.concat([rows("USD_SOFR", 0.97), rows("EUR_ESTR", 0.98)]), "2026-10-01", "2026-10-01", root=tmp_path)
    store_ois_curves(rows("EUR_ESTR", 0.99), "2026-10-01", "2026-10-01", root=tmp_path, curves=("EUR_ESTR",))
    got = read_ois_curves("2026-10-01", "2026-10-02", root=tmp_path).set_index("curve")["df"].to_dict()
    assert got == {"EUR_ESTR": 0.99, "USD_SOFR": 0.97}


# ------------------------------------------------------------------ UK / DE yield P&L
def test_official_curve_yield_pnl_in_its_currency(tmp_path):
    from infra.cycle.bmk_yields import compute_yield_pnl
    from infra.cycle.paths import CyclePaths
    from infra.storage import parquet_store
    paths = CyclePaths.under(tmp_path)
    days = pd.bdate_range("2026-09-28", periods=3)
    rows = pd.DataFrame([(d, t, y) for d, base in zip(days, (4.00, 4.10, 4.05)) for t, y in
                         (("UK_BOND_10y", base), ("DE_BOND_10y", base - 1.5))], columns=["timestamp", "ticker", "par_yield"])
    rows = rows.assign(timestamp=rows.timestamp.astype("datetime64[ms]"), par_yield=(rows.par_yield * 1e4).round().astype("Int32"))
    parquet_store.write_partitioned(rows, paths.daily_bonds_dir, ["timestamp", "ticker"])
    df = compute_yield_pnl(days[0], days[-1], paths=paths, sources=())
    uk = df[df.ticker == "UK_BOND_10y"].set_index("timestamp")
    assert set(df.bmk) == {"yield_boe", "yield_bundesbank"} and set(uk.currency) == {"GBP"}
    assert list(uk.pnl_per_dv01.round(6)) == [-10.0, 5.0]  # + = long the bond: yields fell


# ------------------------------------------------------------------ JGB curve (MoF)
def test_mof_csv_parses_tenors_and_skips_dashes_and_covers_only_published_days():
    from infra.api import mof_client
    hist = "Interest Rate,,,,(Unit : %)\nDate,1Y,2Y,10Y,40Y\n2026/9/29,1.6,1.9,3.1,-\n2026/9/30,1.61,1.91,3.09,4.1\n"
    cur = "Interest Rate (October 2026),,,,(Unit : %)\nDate,1Y,2Y,10Y,40Y\n2026/10/1,1.668,1.939,3.092,4.125\n"
    files = {mof_client.HISTORY: hist, mof_client.CURRENT: cur}
    df, covered = mof_client.fetch_curve(D("2026-09-29"), D("2026-10-08"), fetch=files.get, now=D("2026-10-07"))
    assert len(df[df["timestamp"] == D("2026-09-29")]) == 3            # no 40y that day
    assert covered == [(D("2026-09-29"), D("2026-10-02"))]              # never past the last published day
    assert df.loc[(df["timestamp"] == D("2026-10-01")) & (df["maturity"] == 10.0), "value"].iloc[0] == 3.092


def test_yield_pnl_is_written_with_plain_string_columns(tmp_path):
    from infra.cycle.bmk_yields import backfill_daily_yield_pnl
    from infra.cycle.paths import CyclePaths
    from infra.storage import parquet_store
    import pyarrow.parquet as pq
    paths = CyclePaths.under(tmp_path)
    days = pd.bdate_range("2026-09-28", periods=3)
    rows = pd.DataFrame({"timestamp": days.astype("datetime64[ms]"), "ticker": pd.Categorical(["JP_BOND_10y"] * 3),
                         "par_yield": pd.array([31000, 31100, 31050], dtype="Int32")})
    parquet_store.write_partitioned(rows, paths.daily_bonds_dir, ["timestamp", "ticker"])
    backfill_daily_yield_pnl(days[0], days[-1], paths=paths, sources=(), official={"JP": "mof"})
    f = next((paths.bmk_root / "Pnl").rglob("*.parquet"))
    assert not str(pq.read_schema(f).field("ticker").type).startswith("dictionary")


# ------------------------------------------------------------------ Canada (BoC benchmarks), old BoE sheets
def test_boc_benchmark_csv_parses_the_observations_block_and_maps_long_to_30y():
    from infra.api import boc_client
    text = ('"TERMS AND CONDITIONS"\n"https://www.bankofcanada.ca/terms/"\n\n"OBSERVATIONS"\n'
            '"date","BD.CDN.2YR.DQ.YLD","BD.CDN.10YR.DQ.YLD","BD.CDN.LONG.DQ.YLD","BD.CDN.RRB.DQ.YLD"\n'
            '"2026-10-05","3.26","3.96","4.31","2.00"\n"2026-10-06","3.23","","4.28","1.97"\n')
    df, covered = boc_client.fetch_benchmark_yields(D("2026-10-05"), D("2026-10-09"), fetch=lambda s, e: text)
    assert sorted(df["maturity"].unique()) == [2.0, 10.0, 30.0]          # real-return bonds left out
    assert len(df[df["timestamp"] == D("2026-10-06")]) == 2              # a blank value is no row
    assert covered == [(D("2026-10-05"), D("2026-10-07"))]


def test_boe_reads_the_pre_2005_nominal_sheet_name():
    from infra.api import boe_client
    buf = io.BytesIO()
    rows = [[None, "UK nominal spot curve"], [None], ["Maturity"], ["years:", 0.5, 1.0], [None], [D("1992-09-16"), 9.5, 9.6]]
    with pd.ExcelWriter(buf) as xl:
        pd.DataFrame(rows).to_excel(xl, sheet_name="4. nominal spot curve", header=False, index=False)
    df, first, last = boe_client.parse_spot_sheet(buf.getvalue())
    assert len(df) == 2 and first == D("1992-09-16")
