"""Full-granularity inflation snapshots (BLS / BEA bulk files): file parsing in the real
formats, vintages built from successive snapshots, version-based de-duplication (Rule
2.1), and the raw cycle step. No network: fake sources throughout."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.api import bea_client, bls_client
from infra.config import BulkDataset
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.raw import RAW_STEP
from infra.cycle.runner import execute_step
from infra.pipeline import bulk_series as pbulk
from infra.processing import releases as pr

D = pd.Timestamp

# Excerpts in the real layouts (verified against the live files 2026-09-30): BLS pads every
# field; "-" + footnote X is the 2025 lapse; M13 is the annual average.
BLS_DATA = (
    "series_id        \tyear\tperiod\t       value\tfootnote_codes\n"
    "CUUR0000SA0      \t2025\tM09\t     324.800\t\n"
    "CUUR0000SA0      \t2025\tM10\t           -\tX\n"
    "CUUR0000SA0      \t2025\tM11\t     324.122\t\n"
    "CUUR0000SA0      \t2025\tM13\t     321.943\t\n"
    "CUUR0000SAH1     \t2025\tS01\t     400.000\t\n"
)
BLS_SERIES = (
    "series_id        \tarea_code\titem_code\tseasonal\tseries_title\n"
    "CUUR0000SA0      \t0000\tSA0\tU\tAll items in U.S. city average, all urban consumers, not seasonally adjusted\n"
)
BLS_ITEM = "item_code\titem_name\tdisplay_level\nSA0\tAll items\t0\n"
BLS_AREA = "area_code\tarea_name\tdisplay_level\n0000\tU.S. city average\t0\n"
WP_SERIES = "series_id                     \tgroup_code\titem_code\tseries_title\nWPU011\t01\t1\tFruits\nWPU01\t01\t-\tFarm\n"
WP_ITEM = "group_code\titem_code\titem_name\n01\t-\tFarm products\n01\t1\tFruits & melons\n02\t1\tOther\n"

BEA_DATA = '%SeriesCode,Period,Value\nDPCERG,2026M07,"127.912"\nDPCERG,2026M08,"128.301"\nA015RC,1967M01,"22,068"\n'
BEA_REGISTER = (
    "%SeriesCode,SeriesLabel,MetricName,CalculationType,DefaultScale,TableId:LineNo,SeriesCodeParents\n"
    'DPCERG,"Personal consumption expenditures","Fisher Price Index","Level",0,T20804:1|U20404:1,NONE\n'
    'A015RC,"Something else","Current Dollars","Level",-6,T10705:4,NONE\n'
)


# ------------------------------------------------------------------ clients
def test_bls_parse_keeps_monthly_values_only():
    df = bls_client.parse_data(BLS_DATA)
    assert df["series_id"].tolist() == ["CUUR0000SA0", "CUUR0000SA0"]  # no "-", no M13, no S01
    assert df["date"].tolist() == [D("2025-09-01"), D("2025-11-01")]
    assert df["value"].tolist() == [324.8, 324.122]


def test_bls_catalog_joins_code_names_scoped_by_group():
    cat = bls_client.parse_catalog(BLS_SERIES, {"item": BLS_ITEM, "area": BLS_AREA})
    assert cat.loc[0, "ticker"] == "CUUR0000SA0" and cat.loc[0, "item_name"] == "All items"
    assert cat.loc[0, "area_name"] == "U.S. city average"
    # wp item codes repeat across groups: joined on (group_code, item_code)
    wp = bls_client.parse_catalog(WP_SERIES, {"item": WP_ITEM})
    assert dict(zip(wp["ticker"], wp["item_name"])) == {"WPU011": "Fruits & melons", "WPU01": "Farm products"}


def test_bls_refuses_without_a_contact(monkeypatch):
    monkeypatch.setattr(bls_client, "contact", lambda explicit=None: None)
    with pytest.raises(bls_client.BlsError, match="BLS_CONTACT_EMAIL"):
        bls_client._request("https://download.bls.gov/x")


def test_bls_fetch_refuses_a_file_republished_mid_download():
    stamps = iter([D("2026-09-11 12:30"), D("2026-10-15 12:30")])
    files = {"cu.data.0.Current": BLS_DATA, "cu.series": BLS_SERIES, "cu.item": BLS_ITEM, "cu.area": BLS_AREA}
    with pytest.raises(bls_client.BlsError, match="republished"):
        bls_client.fetch_survey("cu", get=lambda s, n: files[n], modified=lambda s: next(stamps))


def test_bea_parse_periods_thousands_and_table_filter():
    assert bea_client.parse_period(pd.Series(["2026M08", "2026Q3", "2026"])).tolist() == \
        [D("2026-08-01"), D("2026-07-01"), D("2026-01-01")]
    values, catalog, published = bea_client.fetch_nipa(
        "NipaDataM.txt", ("U20404",), get=lambda n: BEA_REGISTER if n == "SeriesRegister.txt" else BEA_DATA,
        modified=lambda n: D("2026-09-30 12:30:02"))
    assert catalog["ticker"].tolist() == ["DPCERG"] and set(values["series_id"]) == {"DPCERG"}
    assert values["value"].tolist() == [127.912, 128.301] and published == D("2026-09-30 12:30:02")
    assert bea_client.parse_data(BEA_DATA)["value"].iloc[-1] == 22068.0  # thousands separator


# ------------------------------------------------------------------ pipeline
PCE = BulkDataset("pce", "PCE", "bea", "NipaDataM.txt", "BEA", tables=("U20404",))


class FakeBulk:
    """A source whose current file is whichever version was set last."""

    def __init__(self):
        self.published, self.values, self.downloads = None, None, 0

    def set(self, published, rows):
        self.published = D(published)
        self.values = pd.DataFrame(rows, columns=["series_id", "date", "value"]).astype({"date": "datetime64[ms]"})

    def fetch(self, dataset):
        self.downloads += 1
        return self.values, pd.DataFrame({"ticker": sorted(set(self.values["series_id"]))}), self.published

    def modified(self, dataset):
        return self.published


def _load(tmp_path, fake, as_of=None, fetch_missing=True):
    return pbulk.load_bulk_series("pce", ["DPCERG"], as_of, fetch_missing=fetch_missing, raw_root=tmp_path / "raw",
                                  coverage_file=tmp_path / "cov.parquet", catalog_dir=tmp_path / "cat",
                                  sources={"bea": fake.fetch}, last_modified={"bea": fake.modified})


def test_snapshots_build_vintages_and_skip_known_versions(tmp_path, monkeypatch):
    monkeypatch.setitem(pbulk.BULK_DATASETS, "pce", PCE)
    fake = FakeBulk()
    fake.set("2026-08-29 12:30", [("DPCERG", D("2026-06-01"), 127.5), ("DPCERG", D("2026-07-01"), 127.9)])
    first = _load(tmp_path, fake)
    assert len(first) == 2 and (first["timestamp"] == D("2026-08-29")).all()  # history stamped at 1st sight
    _load(tmp_path, fake)
    assert fake.downloads == 1  # same Last-Modified: never downloaded again (a HEAD only)

    # next release: July revised, August new, June unchanged
    fake.set("2026-09-30 12:30", [("DPCERG", D("2026-06-01"), 127.5), ("DPCERG", D("2026-07-01"), 127.912),
                                  ("DPCERG", D("2026-08-01"), 128.301)])
    raw = _load(tmp_path, fake)
    assert fake.downloads == 2 and len(raw) == 4  # only what the new version CHANGED was added
    assert pr.snapshot(raw, "2026-09-29")["value"].tolist() == [127.5, 127.9]  # point in time: pre-release
    assert pr.snapshot(raw, "2026-09-30")["value"].tolist() == [127.5, 127.912, 128.301]
    assert pbulk.read_catalog("pce", catalog_dir=tmp_path / "cat")["ticker"].tolist() == ["DPCERG"]


def test_a_past_as_of_never_fetches(tmp_path, monkeypatch):
    monkeypatch.setitem(pbulk.BULK_DATASETS, "pce", PCE)
    fake = FakeBulk()
    fake.set("2026-09-30 12:30", [("DPCERG", D("2026-08-01"), 128.3)])
    assert _load(tmp_path, fake, as_of="2026-09-01").empty and fake.downloads == 0


def test_an_older_or_unpublished_version_is_not_downloaded(tmp_path):
    cov = tmp_path / "cov.parquet"
    assert not pbulk.needs_download("pce", None, coverage_file=cov)
    assert pbulk.needs_download("pce", D("2026-09-30 12:30"), coverage_file=cov)
    pbulk.store_bulk_raw(PCE, pd.DataFrame(columns=["series_id", "date", "value"]), pd.DataFrame({"ticker": []}),
                         D("2026-09-30 12:30"), raw_root=tmp_path, coverage_file=cov, catalog_dir=tmp_path / "c")
    assert pbulk.latest_ingested("pce", coverage_file=cov) == D("2026-09-30 12:30")
    assert not pbulk.needs_download("pce", D("2026-08-29 12:30"), coverage_file=cov)
    assert pbulk.needs_download("pce", D("2026-09-30 12:31"), coverage_file=cov)


# ------------------------------------------------------------------ cycle step
def _ctx(tmp_path, start, end):
    return StepContext(D(start), D(end), D(end), CyclePaths.under(tmp_path))


def test_raw_step_ingests_and_checks(tmp_path, monkeypatch):
    fake = FakeBulk()
    fake.set("2026-09-30 12:30", [("DPCERG", D("2026-08-01"), 128.3)])
    monkeypatch.setattr("infra.cycle.raw_bulk.BULK_DATASETS", {"pce": PCE})
    monkeypatch.setitem(pbulk.BULK_DATASETS, "pce", PCE)
    monkeypatch.setattr(pbulk, "SOURCES", {"bea": fake.fetch})
    monkeypatch.setattr(pbulk, "LAST_MODIFIED", {"bea": fake.modified})
    outcome = execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-28", "2026-09-30"))
    assert outcome.status == "ok", outcome
    checks = {c.name: c for c in outcome.checks}
    assert checks["bulk_fetch_ok"].passed and "pce" in checks["bulk_fetch_ok"].message
    assert checks["bulk_sane"].passed and checks["bulk_fresh"].passed
    paths = CyclePaths.under(tmp_path)
    assert len(pbulk.read_bulk_from_disk("pce", raw_root=paths.raw_data_root)) == 1


def test_raw_step_without_a_bls_contact_warns_and_fetches_nothing(tmp_path, monkeypatch):
    cpi = BulkDataset("cpi", "CPI", "bls", "cu", "BLS")
    monkeypatch.setattr("infra.cycle.raw_bulk.BULK_DATASETS", {"cpi": cpi})
    monkeypatch.setattr(pbulk, "CONFIGURED", {"bls": lambda: False, "bea": lambda: True})
    monkeypatch.setattr(pbulk, "LAST_MODIFIED", {"bls": lambda d: pytest.fail("no request without a contact")})
    outcome = execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-28", "2026-09-30"))
    assert outcome.status == "ok"  # the rest of the cycle is not blocked
    warned = {c.name: c for c in outcome.checks}["bulk_configured"]
    assert not warned.passed and "BLS_CONTACT_EMAIL" in warned.message


def test_raw_step_reports_a_failing_source(tmp_path, monkeypatch):
    def broken(d):
        raise bea_client.BeaError("HTTP 503")

    monkeypatch.setattr("infra.cycle.raw_bulk.BULK_DATASETS", {"pce": PCE})
    monkeypatch.setattr(pbulk, "LAST_MODIFIED", {"bea": broken})
    outcome = execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-28", "2026-09-30"))
    assert outcome.status == "failed"
    failed = {c.name: c for c in outcome.checks if not c.passed}
    assert "BeaError" in failed["bulk_fetch_ok"].details["error"].iloc[0]
