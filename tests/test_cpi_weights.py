"""CPI relative importance: the three published layouts, fetch-once-a-year planning,
publication-day stamping from ALFRED, point-in-time reads, and the raw cycle source.
No network: fake fetchers throughout."""
from __future__ import annotations

import io

import pandas as pd

from infra.api import bls_client
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.raw import RAW_STEP
from infra.cycle.runner import execute_step
from infra.pipeline import cpi_weights as pcw
from infra.processing import cpi_weights as cw
from infra.processing import releases as pr
from infra.storage import parquet_store

D = pd.Timestamp

# Excerpts in the real layouts (verified against the BLS files 2026-09-30)
CODED = (
    "Relative importance of components in the Consumer Price\n"
    "Indexes: U.S. city average, December 1995\n\n"
    "Item     Item and group                               U.S. city average\n"
    "SA0      ALL ITEMS                                     100.000   100.000\n"
    "SA1      FOOD AND BEVERAGES                             17.334    19.370\n"
    "SAS2LSRS HOUSEHOLD SER LESS RENT OF SHELTER              8.681     8.531\n"
)
DOTTED = (
    " Table 1 (2017-2018 Weights). Relative importance of components in the Consumer\n"
    " Price Indexes:  U.S. city average, December 2019\n\n"
    " (Percent of all items)\n"
    "                                                            CPI-U          CPI-W\n"
    "                 Expenditure category\n\n"
    " All items............................................   100.000        100.000\n"
    "  Food and beverages..................................    14.794         16.246\n"
    "         Bacon, breakfast sausage, and related\n"
    "            products..................................      .137           .171\n"
    "  Housing.............................................    42.385         40.337\n"
    " All items............................................   100.000        100.000\n"
    "  Energy..............................................     7.050          8.100\n"
)


def _xlsx() -> bytes:
    rows = [[None, "Table 1 (2024 Weights). Relative importance ..., December 2025", None, None],
            [None, "[Percent of all items]", None, None],
            ["Indent Level", "Item and Group", "U.S. City Average", "U.S. City Average"],
            [None, None, "CPI-U", "CPI-W"],
            [0, "Expenditure category", None, None],
            [0, "All items", 100, 100], [1, "Food and beverages", 14.539, 15.579],
            [0, "All items", 100, 100], [1, "Energy", 6.4, 7.1]]
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, sheet_name="Table 1", header=False, index=False)
    return buf.getvalue()


# ------------------------------------------------------------------ parsing
def test_coded_layout_keeps_codes_even_when_they_fill_their_column():
    df = cw.parse_txt(CODED, 1995)
    assert df["item_code"].tolist() == ["SA0", "SA1", "SAS2LSRS"]
    assert df["item_name"].iloc[2] == "HOUSEHOLD SER LESS RENT OF SHELTER"
    assert df["section"].isna().all() and df["indent"].isna().all()  # flat list, no tree


def test_dotted_layout_indent_wrapped_names_and_sections():
    df = cw.parse_txt(DOTTED, 2019)
    assert df["item_name"].tolist() == ["All items", "Food and beverages",
                                        "Bacon, breakfast sausage, and related products", "Housing", "All items",
                                        "Energy"]
    assert df["indent"].tolist() == [0, 1, 8, 1, 0, 1]  # a wrapped name keeps its FIRST line's indent
    assert df["section"].tolist() == ["expenditure"] * 4 + ["special"] * 2
    assert df.loc[2, "cpi_u"] == 0.137 and df["line"].tolist() == list(range(6))


def test_xlsx_layout():
    df = cw.parse_xlsx(_xlsx(), 2025)
    assert df["item_name"].tolist() == ["All items", "Food and beverages", "All items", "Energy"]
    assert df["indent"].tolist() == [0, 1, 0, 1] and df["section"].tolist()[2:] == ["special"] * 2
    assert df["cpi_w"].iloc[1] == 15.579


def test_item_codes_matched_by_normalized_unique_name():
    items = pd.DataFrame({"item_code": ["SAF1", "SAF1", "SEFA", "X1", "X2"],  # a catalog repeats items per series
                          "item_name": ["Food and Beverages", "Food and Beverages", "Energy", "Dup", "dup"]})
    df = pd.DataFrame({"item_name": ["Food and beverages", "Dup", "Unknown"], "item_code": [None, None, "KEEP"]})
    codes = cw.match_item_codes(df, items)["item_code"]
    assert codes[0] == "SAF1" and pd.isna(codes[1]) and codes[2] == "KEEP"  # ambiguous name -> null


def test_fetch_uses_loose_xlsx_then_the_decade_archive_once():
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("2010-2019_RI_archive/2018.txt", "a")
        zf.writestr("2010-2019_RI_archive/2019.txt", "b")
    calls = []

    def get(name):
        calls.append(name)
        return {"2025.xlsx": b"x", "ri-archive-2010-2019.zip": buf.getvalue()}.get(name)

    got = bls_client.fetch_relative_importance([2018, 2019, 2025, 2026], get=get)
    assert got == {2018: ("txt", b"a"), 2019: ("txt", b"b"), 2025: ("xlsx", b"x")}  # 2026 not out yet
    assert calls.count("ri-archive-2010-2019.zip") == 1


# ------------------------------------------------------------------ planning
def test_a_weight_year_is_due_from_february():
    assert pcw.due_weight_year("2026-01-31") == 2024
    assert pcw.due_weight_year("2026-02-01") == 2025


def test_nothing_is_requested_once_every_due_year_is_on_disk(tmp_path):
    cov = tmp_path / "cov.parquet"
    assert pcw.plan_weights_update("2026-09-30", coverage_file=cov, first_year=2024) == [2024, 2025]
    fake = {2024: ("txt", DOTTED.encode()), 2025: ("xlsx", _xlsx())}
    pcw.store_weights_raw(fake, root=tmp_path / "w", coverage_file=cov, releases_root=tmp_path / "rel",
                          catalog_dir=tmp_path / "cat")
    assert pcw.plan_weights_update("2026-09-30", coverage_file=cov, first_year=2024) == []
    assert pcw.plan_weights_update("2027-02-01", coverage_file=cov, first_year=2024) == [2026]


# ------------------------------------------------------------------ store / point in time
def _alfred(root):
    """ALFRED first prints of January CPI: the publication days of weight years 2024, 2025."""
    rows = pd.DataFrame({"timestamp": [D("2025-02-12"), D("2026-02-13")], "ticker": "CPIAUCNS",
                         "period": [D("2025-01-01"), D("2026-01-01")], "value": [317.7, 325.3]})
    parquet_store.write_partitioned(pr.encode_raw(rows), root, pr.RAW_KEYS)


def test_weights_are_stamped_with_the_january_cpi_day_and_read_point_in_time(tmp_path):
    _alfred(tmp_path / "rel")
    fake = {2024: ("txt", DOTTED.encode()), 2025: ("xlsx", _xlsx())}
    w = pcw.load_cpi_weights(root=tmp_path / "w", coverage_file=tmp_path / "cov.parquet",
                             releases_root=tmp_path / "rel", catalog_dir=tmp_path / "cat",
                             fetch=lambda years: {y: fake[y] for y in years if y in fake})
    assert w["weight_year"].iloc[0] == 2025 and (w["timestamp"] == D("2026-02-13")).all()
    root = tmp_path / "w"
    assert pcw.read_cpi_weights("2026-02-12", root=root)["weight_year"].iloc[0] == 2024  # day before release
    assert pcw.read_cpi_weights("2026-02-13", root=root)["weight_year"].iloc[0] == 2025
    assert pcw.read_cpi_weights("2025-01-01", root=root).empty


# ------------------------------------------------------------------ cycle source
def _ctx(tmp_path, end):
    return StepContext(D(end), D(end), D(end), CyclePaths.under(tmp_path))


def test_raw_step_fetches_due_years_then_nothing(tmp_path, monkeypatch):
    calls = []

    def fetch(years):
        calls.append(list(years))
        return {2025: ("xlsx", _xlsx())} if 2025 in years else {}

    monkeypatch.setattr(pcw, "FETCH", fetch)
    orig = pcw.plan_weights_update
    monkeypatch.setattr(pcw, "plan_weights_update", lambda today=None, **kw: orig(today, first_year=2025, **kw))
    first = execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-29"))
    checks = {c.name: c for c in first.checks}
    assert checks["cpi_weights_fetch_ok"].passed and checks["cpi_weights_sane"].passed
    assert checks["cpi_weights_fresh"].passed and calls == [[2025]]
    execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-30"))
    assert calls == [[2025]]  # the next day: no request at all


def test_an_unpublished_year_is_not_covered_and_turns_overdue(tmp_path, monkeypatch):
    orig = pcw.plan_weights_update
    monkeypatch.setattr(pcw, "plan_weights_update", lambda today=None, **kw: orig(today, first_year=2025, **kw))
    early = execute_step(RAW_STEP, _ctx(tmp_path, "2026-02-05"))  # due, BLS hasn't posted it yet
    checks = {c.name: c for c in early.checks}
    assert checks["cpi_weights_fetch_ok"].passed and checks["cpi_weights_fresh"].passed
    late = execute_step(RAW_STEP, _ctx(tmp_path, "2026-04-01"))
    assert not {c.name: c for c in late.checks}["cpi_weights_fresh"].passed  # warn: overdue
