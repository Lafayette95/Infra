"""Macro-release vintages: FRED client parsing/paging, the vintage store (contiguous
coverage, de-duplication, point-in-time reads), unit derivation, and the raw cycle step.
No network: fake ``get`` / ``sources`` throughout."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.api import fred_client
from infra.config import MacroRelease
from infra.reference.events import EventSeries
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.raw import RAW_STEP
from infra.cycle.runner import execute_step
from infra.pipeline import releases as prel
from infra.processing import releases as pr
from infra.storage import coverage_store

D = pd.Timestamp


# ------------------------------------------------------------------ FRED client
def _payload(rows, count=None):
    return {"count": len(rows) if count is None else count, "observations": [
        {"realtime_start": rs, "realtime_end": "9999-12-31", "date": d, "value": v} for rs, d, v in rows]}


def test_parse_drops_missing_values():
    df = fred_client.parse_observations(_payload([("2026-01-09", "2025-12-01", "159000"), ("2026-01-09", "2026-01-01", ".")]))
    assert list(df.columns) == fred_client.VINTAGE_COLUMNS
    assert len(df) == 1 and df["value"].iloc[0] == 159000.0
    assert fred_client.parse_observations({"observations": []}).empty


def test_fetch_vintages_pages_and_never_claims_an_unfinished_day():
    calls = []

    def get(endpoint, params, key=None):
        calls.append(params["offset"])
        rows = [("2026-09-0%d" % (params["offset"] + 1), "2026-08-01", str(params["offset"]))]
        return _payload(rows, count=2)

    old = fred_client.PAGE_LIMIT
    fred_client.PAGE_LIMIT = 1
    try:
        df, covered = fred_client.fetch_vintages("PAYEMS", D("2026-09-01"), D("2026-10-01"), get=get,
                                                 now=D("2026-09-30 05:00"))
    finally:
        fred_client.PAGE_LIMIT = old
    assert calls == [0, 1] and len(df) == 2
    # 05:00 UTC on the 30th: the 29th (US) isn't over yet -> covered only through the 28th
    assert covered == [(D("2026-09-01"), D("2026-09-29"))]


# ------------------------------------------------------------------ processing
def _raw(rows):
    return pd.DataFrame(rows, columns=pr.RAW_COLUMNS).astype({"timestamp": "datetime64[ms]", "period": "datetime64[ms]"})


def test_drop_unchanged_keeps_only_new_information():
    stored = _raw([(D("2026-08-07"), "PAYEMS", D("2026-07-01"), 100.0)])
    incoming = _raw([
        (D("2026-08-07"), "PAYEMS", D("2026-07-01"), 100.0),  # re-fetched duplicate
        (D("2026-09-01"), "PAYEMS", D("2026-07-01"), 100.0),  # clipped "still current" row
        (D("2026-09-05"), "PAYEMS", D("2026-07-01"), 101.0),  # a genuine revision
        (D("2026-09-05"), "PAYEMS", D("2026-08-01"), 102.0),  # a first print
    ])
    new = pr.drop_unchanged(incoming, stored)
    assert list(zip(new["timestamp"], new["period"], new["value"])) == [
        (D("2026-09-05"), D("2026-07-01"), 101.0), (D("2026-09-05"), D("2026-08-01"), 102.0)]


def test_snapshot_and_estimates():
    raw = _raw([(D("2026-07-30"), "G", D("2026-04-01"), 3.0), (D("2026-08-28"), "G", D("2026-04-01"), 3.3),
                (D("2026-09-25"), "G", D("2026-04-01"), 3.8)])
    assert pr.snapshot(raw, "2026-08-27")["value"].tolist() == [3.0]
    assert pr.snapshot(raw, "2026-08-28")["value"].tolist() == [3.3]  # as_of is inclusive
    assert [pr.estimate(raw, k)["value"].iloc[0] for k in range(3)] == [3.0, 3.3, 3.8]  # advance, second, third
    assert pr.estimate(raw, 3).empty


def test_derive_units_dates_a_change_by_its_latest_input():
    s = pd.DataFrame({"period": pd.to_datetime(["2026-05-01", "2026-06-01", "2026-08-01"]),
                      "value": [100.0, 103.0, 110.0],
                      "timestamp": pd.to_datetime(["2026-07-03", "2026-07-03", "2026-09-04"])})
    diff = pr.derive_units(s, "diff", "M")
    assert diff["value"].tolist() == [3.0]  # August has no July: never a change across a gap
    assert diff["timestamp"].iloc[0] == D("2026-07-03")
    pct = pr.derive_units(s, "pct", "M")
    assert pct["value"].iloc[0] == pytest.approx(3.0)
    yoy = pd.DataFrame({"period": pd.date_range("2025-01-01", periods=13, freq="MS"),
                        "value": [100.0] * 12 + [102.0],
                        "timestamp": pd.date_range("2025-02-10", periods=13, freq="MS")})
    out = pr.derive_units(yoy, "yoy", "M")
    assert len(out) == 1 and out["value"].iloc[0] == pytest.approx(2.0)


# ------------------------------------------------------------------ pipeline
class FakeFred:
    """A tiny ALFRED: a list of publications; answers like the real API (values current
    at ``start`` come back with their realtime_start CLIPPED to ``start``)."""

    def __init__(self, pubs):
        self.pubs = pubs  # (realtime_start, date, value)
        self.calls = []

    def __call__(self, series_id, start, end):
        self.calls.append((series_id, start, end))
        df = pd.DataFrame(self.pubs, columns=fred_client.VINTAGE_COLUMNS)
        df = df.sort_values("realtime_start")
        nxt = df.groupby("date")["realtime_start"].shift(-1)
        live = nxt.isna() | (nxt > start)
        df = df[live & (df["realtime_start"] < end)].copy()
        df["realtime_start"] = df["realtime_start"].clip(lower=start)
        return df.reset_index(drop=True), [(start, end)]


def _pubs():
    return [(D("2026-07-03"), D("2026-06-01"), 100.0), (D("2026-08-07"), D("2026-06-01"), 101.0),
            (D("2026-08-07"), D("2026-07-01"), 105.0), (D("2026-09-04"), D("2026-08-01"), 107.0)]


def test_plan_bootstraps_from_the_epoch_then_stays_contiguous(tmp_path):
    cov = tmp_path / "cov.parquet"
    plan = prel.plan_release_update("PAYEMS", "fred", "2026-09-01", "2026-09-10", coverage_file=cov)
    assert plan == [(fred_client.EPOCH, D("2026-09-10"))]
    coverage_store.record_covered(cov, "PAYEMS", [(fred_client.EPOCH, D("2026-08-01"))])
    # a later window never leaves a hole: it starts where coverage ends
    assert prel.plan_release_update("PAYEMS", "fred", "2026-09-01", "2026-09-10", coverage_file=cov) == \
        [(D("2026-08-01"), D("2026-09-10"))]
    assert prel.plan_release_update("PAYEMS", "fred", "2026-07-01", "2026-07-10", coverage_file=cov) == []
    assert prel.plan_release_update("PAYEMS", "fred", "2026-07-01", "2026-07-10", coverage_file=cov,
                                    force_refetch=True) == [(D("2026-07-01"), D("2026-07-10"))]


def test_store_then_refetch_windows_keep_true_publication_days(tmp_path):
    root, cov = tmp_path / "rel", tmp_path / "cov.parquet"
    fake = FakeFred(_pubs())
    prel.store_releases_raw(prel.fetch_releases_raw({"PAYEMS": ("fred", [(fred_client.EPOCH, D("2026-08-10"))])},
                                                    sources={"fred": fake}), root=root, coverage_file=cov)
    # a forced re-fetch from mid-history: clipped rows must not become fake publications
    n = prel.store_releases_raw(prel.fetch_releases_raw({"PAYEMS": ("fred", [(D("2026-08-01"), D("2026-09-10"))])},
                                                        sources={"fred": fake}), root=root, coverage_file=cov)
    assert n == 1  # only August's first print is new
    raw = prel.read_releases_from_disk(["PAYEMS"], root=root)
    assert sorted(zip(raw["timestamp"], raw["period"], raw["value"])) == sorted(_pubs())
    # point in time
    assert prel.read_releases_from_disk(["PAYEMS"], "2026-08-06", root=root)["value"].tolist() == [100.0]


def test_load_releases_fetches_only_the_gap(tmp_path):
    root, cov = tmp_path / "rel", tmp_path / "cov.parquet"
    fake = FakeFred(_pubs())
    rel = {"NFP": MacroRelease("NFP", "n", EventSeries("NFP", "E_NFP", "n", "u", "SA", store_id="PAYEMS", frequency="M", source="fred", derive="diff"), 0, 0, 0, 0, 1, 1, 1, "", "Activity",
                               "Hard", ("Activity_Labor",))}
    prel.load_releases(["PAYEMS"], "2026-08-10", root=root, coverage_file=cov, releases=rel, sources={"fred": fake})
    prel.load_releases(["PAYEMS"], "2026-08-10", root=root, coverage_file=cov, releases=rel, sources={"fred": fake})
    assert len(fake.calls) == 1  # Rule 2.1: covered publication days are never re-asked
    out = prel.load_releases(["PAYEMS"], "2026-09-10", root=root, coverage_file=cov, releases=rel,
                             sources={"fred": fake})
    assert fake.calls[-1][1] == D("2026-08-11") and len(out) == 4


# ------------------------------------------------------------------ cycle step
_FRED_ONLY = {"NFP": MacroRelease("NFP", "n", EventSeries("NFP", "E_NFP", "n", "u", "SA", store_id="PAYEMS", frequency="M", source="fred", derive="diff"), 0, 0, 0, 0, 1, 1, 1, "", "Activity",
                                  "Hard", ("Activity_Labor",))}

def _ctx(tmp_path, start, end, **kw):
    return StepContext(D(start), D(end), D(end), CyclePaths.under(tmp_path), **kw)


def test_raw_step_fetches_and_checks(tmp_path, monkeypatch):
    fake = FakeFred(_pubs())
    monkeypatch.setattr("infra.pipeline.releases.SOURCES", {"fred": fake})
    rel = {"NFP": MacroRelease("NFP", "n", EventSeries("NFP", "E_NFP", "n", "u", "SA", store_id="PAYEMS", frequency="M", source="fred", derive="diff"), 0, 0, 0, 0, 1, 1, 1, "", "Activity",
                               "Hard", ("Activity_Labor",)),
           "ISM": MacroRelease("ISM", "i", EventSeries("ISM", "E_ISM", "i", "u", "SA", store_id=None, frequency="M", source=None, derive="level"), 0, 0, 0, 0, 1, 1, 1, "", "Activity",
                               "Survey", ("Activity_Manufacturing",))}
    monkeypatch.setattr("infra.cycle.raw_releases.MACRO_RELEASES", rel)
    outcome = execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-01", "2026-09-10"))
    assert outcome.status == "ok", outcome
    checks = {c.name: c for c in outcome.checks}
    assert checks["releases_fetch_ok"].passed and checks["releases_sane"].passed
    assert fake.calls[0][0] == "PAYEMS" and fake.calls[0][1] == fred_client.EPOCH  # bootstrap, unavailable skipped
    assert len(prel.read_releases_from_disk(["PAYEMS"], root=CyclePaths.under(tmp_path).releases_dir)) == 4


def test_raw_step_reports_a_failing_source(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.cycle.raw_releases.MACRO_RELEASES", _FRED_ONLY)
    def broken(series_id, start, end):
        raise fred_client.FredError("HTTP 400 Bad Request")

    monkeypatch.setattr("infra.pipeline.releases.SOURCES", {"fred": broken})
    outcome = execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-01", "2026-09-10"))
    assert outcome.status == "failed"
    failed = [c for c in outcome.checks if not c.passed]
    assert failed[0].name == "releases_fetch_ok" and "FredError" in failed[0].details["error"].iloc[0]


def test_raw_step_without_a_key_warns_and_fetches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.cycle.raw_releases.MACRO_RELEASES", _FRED_ONLY)
    fake = FakeFred(_pubs())
    monkeypatch.setattr("infra.pipeline.releases.SOURCES", {"fred": fake})
    monkeypatch.setattr("infra.pipeline.releases.CONFIGURED", {"fred": lambda: False})
    outcome = execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-01", "2026-09-10"))
    assert outcome.status == "ok" and fake.calls == []  # the rest of the cycle is not blocked
    warned = {c.name: c for c in outcome.checks}["releases_configured"]
    assert not warned.passed and "FRED_API_KEY" in warned.message


def test_inflation_components_are_fetched_but_not_nowcast_inputs():
    from infra.config import INFLATION_ALFRED_SERIES
    fetched, inputs = prel.all_series_to_fetch(), prel.series_to_fetch()
    assert set(INFLATION_ALFRED_SERIES) <= set(fetched) and not set(INFLATION_ALFRED_SERIES) & set(inputs)
    assert all(src == "fred" for sid, src in fetched.items() if sid in INFLATION_ALFRED_SERIES)



def test_releases_due_judges_only_when_the_calendar_is_fresh(tmp_path):
    from infra.cycle.raw_releases import due_releases
    from infra.pipeline import release_calendar as prc

    paths = CyclePaths.under(tmp_path)
    rel = {"NFP TCH Index": MacroRelease("NFP TCH Index", "n", EventSeries("NFP TCH Index", "E_NFP TCH Index", "n", "u", "SA", store_id="PAYEMS", frequency="M", source="fred", derive="diff"), 0, 0, 0, 0, 1, 1, 1,
                                         "", "Activity", "Hard", ("Activity_Labor",))}
    assert due_releases("2026-09-01", "2026-09-10", paths=paths, releases=rel)[0] is None  # no calendar
    # history: PAYEMS published on each of the release's past dates (07-03, 08-07) - so it is judged
    fetch = lambda rid: pd.DatetimeIndex(["2026-07-03", "2026-08-07", "2026-09-04"]) if rid == 50 \
        else pd.DatetimeIndex([])  # noqa: E731
    fake0 = FakeFred([p for p in _pubs() if p[0] < D("2026-09-01")])
    prel.store_releases_raw(prel.fetch_releases_raw({"PAYEMS": ("fred", [(fred_client.EPOCH, D("2026-09-01"))])},
                                                    sources={"fred": fake0}),
                            root=paths.releases_dir, coverage_file=paths.releases_coverage)
    prc.refresh_release_calendar(root=paths.release_calendar_dir, observed=D("2026-09-10"), fetch=fetch,
                                 with_calendar=False, with_rules=False, with_agencies=False)
    stale, note = due_releases("2026-09-01", "2026-09-10", paths=paths, releases=rel, now=D("2026-09-30"))
    assert stale is None and "stale" in note
    table, _ = due_releases("2026-09-01", "2026-09-10", paths=paths, releases=rel, now=D("2026-09-10 12:00"))
    assert list(table["arrived"]) == [False]  # Sep 4 payrolls never arrived
    fake = FakeFred(_pubs())  # publishes PAYEMS on 2026-09-04
    prel.store_releases_raw(prel.fetch_releases_raw({"PAYEMS": ("fred", [(fred_client.EPOCH, D("2026-09-10"))])},
                                                    sources={"fred": fake}),
                            root=paths.releases_dir, coverage_file=paths.releases_coverage)
    table, _ = due_releases("2026-09-01", "2026-09-10", paths=paths, releases=rel, now=D("2026-09-10 12:00"))
    assert list(table["arrived"]) == [True]



def test_releases_due_skips_series_not_tied_to_every_release_date(tmp_path):
    """A release id with two dates a month, the series updating on one: never judged."""
    from infra.cycle.raw_releases import due_releases
    from infra.pipeline import release_calendar as prc

    paths = CyclePaths.under(tmp_path)
    rel = {"NFP TCH Index": MacroRelease("NFP TCH Index", "n", EventSeries("NFP TCH Index", "E_NFP TCH Index", "n", "u", "SA", store_id="PAYEMS", frequency="M", source="fred", derive="diff"), 0, 0, 0, 0, 1, 1, 1,
                                         "", "Activity", "Hard", ("Activity_Labor",))}
    dates = pd.DatetimeIndex(["2026-07-03", "2026-07-20", "2026-08-07", "2026-08-21", "2026-09-04", "2026-09-18"])
    prc.refresh_release_calendar(root=paths.release_calendar_dir, observed=D("2026-09-20"),
                                 fetch=lambda rid: dates if rid == 50 else pd.DatetimeIndex([]),
                                 with_calendar=False, with_rules=False, with_agencies=False)
    fake = FakeFred([p for p in _pubs() if p[0] < D("2026-09-01")])  # updates on 07-03 and 08-07 only
    prel.store_releases_raw(prel.fetch_releases_raw({"PAYEMS": ("fred", [(fred_client.EPOCH, D("2026-09-01"))])},
                                                    sources={"fred": fake}),
                            root=paths.releases_dir, coverage_file=paths.releases_coverage)
    table, _ = due_releases("2026-09-01", "2026-09-20", paths=paths, releases=rel, now=D("2026-09-20 12:00"))
    assert table.empty
