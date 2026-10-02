"""DTCC swap-trade archive: planning (UTC days, first day, archived days skipped), verified
atomic storage, per-day failure isolation, the client's 404/503 handling, and the raw
cycle source with its checks. No network: fake fetchers and a fake urlopen throughout."""
from __future__ import annotations

import io
import urllib.error
import zipfile

import pandas as pd
import pytest

from infra.api import dtcc_client
from infra.cycle import raw_dtcc
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.raw import RAW_STEP
from infra.cycle.runner import execute_step
from infra.pipeline import dtcc as pdtcc

D = pd.Timestamp
HEADER = ",".join(f'"{c}"' for c in raw_dtcc.REQUIRED_COLUMNS)
ROW = ",".join(["1"] * len(raw_dtcc.REQUIRED_COLUMNS))


def _zip(csv_text: str = f"{HEADER}\n{ROW}\n", name: str = "CFTC_CUMULATIVE_RATES.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, csv_text)
    return buf.getvalue()


def _fetch_all(calls=None, content=None):
    def fetch(kind, day):
        if calls is not None:
            calls.append(pd.Timestamp(day))
        return _zip() if content is None else content
    return fetch


# ------------------------------------------------------------------ planning
def test_plan_clamps_to_first_day_and_to_the_last_ended_utc_day(tmp_path):
    days = pdtcc.plan_dtcc_update("RATES", "2024-09-28", "2024-10-05", now=D("2024-10-03 12:00"), root=tmp_path)
    assert days == list(pd.date_range("2024-09-30", "2024-10-02"))  # DTCC_FIRST_DAY .. yesterday (UTC)


def test_a_tz_aware_now_is_judged_in_utc(tmp_path):
    # 21:00 New York on Oct 2 is already Oct 3 in UTC, so Oct 2 has ended
    now = pd.Timestamp("2026-10-02 21:00", tz="America/New_York")
    assert pdtcc.plan_dtcc_update("RATES", "2026-10-01", "2026-10-05", now=now, root=tmp_path)[-1] == D("2026-10-02")


def test_archived_days_are_never_planned_again(tmp_path):
    pdtcc.store_dtcc_raw("RATES", "2026-09-29", _zip(), root=tmp_path)
    assert pdtcc.archived_days("RATES", root=tmp_path) == {D("2026-09-29")}
    days = pdtcc.plan_dtcc_update("RATES", "2026-09-28", "2026-09-30", now=D("2026-10-01 12:00"), root=tmp_path)
    assert days == [D("2026-09-28"), D("2026-09-30")]


# ------------------------------------------------------------------ storage
def test_store_writes_the_published_bytes_under_the_year(tmp_path):
    content = _zip()
    path = pdtcc.store_dtcc_raw("RATES", "2026-09-29", content, root=tmp_path)
    assert path == tmp_path / "CFTC_RATES" / "2026" / "CFTC_CUMULATIVE_RATES_2026_09_29.zip"
    assert path.read_bytes() == content
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("content", [b"PK\x03\x04 truncated", b"<html>error</html>", _zip(name="readme.txt")])
def test_a_bad_file_is_never_written(tmp_path, content):
    with pytest.raises(dtcc_client.DtccError):
        pdtcc.store_dtcc_raw("RATES", "2026-09-29", content, root=tmp_path)
    assert not list(tmp_path.rglob("*.zip")) and not list(tmp_path.rglob("*.part"))


def test_one_failed_day_does_not_cost_the_others(tmp_path):
    def fetch(kind, day):
        if day == D("2026-09-28"):
            raise dtcc_client.DtccError("HTTP 503 after 6 attempts")
        return None if day == D("2026-09-30") else _zip()

    days = list(pd.date_range("2026-09-27", "2026-09-30"))
    out = pdtcc.fetch_and_store_dtcc("RATES", days, root=tmp_path, fetch=fetch, sleep=lambda s: None)
    assert out["archived"] == [D("2026-09-27"), D("2026-09-29")]
    assert out["unpublished"] == [D("2026-09-30")]
    assert list(out["errors"]) == [D("2026-09-28")]
    assert pdtcc.archived_days("RATES", root=tmp_path) == {D("2026-09-27"), D("2026-09-29")}


def test_load_parent_archives_a_missing_day_then_reads_it(tmp_path):
    calls = []
    df = pdtcc.load_dtcc_day("RATES", "2026-09-29", root=tmp_path, fetch=_fetch_all(calls))
    assert list(df.columns) == list(raw_dtcc.REQUIRED_COLUMNS) and len(df) == 1
    pdtcc.load_dtcc_day("RATES", "2026-09-29", root=tmp_path, fetch=_fetch_all(calls))
    assert calls == [D("2026-09-29")]  # read from disk the second time


# ------------------------------------------------------------------ client
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _urlopen(answers):
    def urlopen(request, timeout=None):
        answer = answers.pop(0)
        if isinstance(answer, int):
            raise urllib.error.HTTPError(request.full_url, answer, "x", {}, None)
        return _Resp(answer)
    return urlopen


def test_client_404_means_not_published(monkeypatch):
    monkeypatch.setattr(dtcc_client.urllib.request, "urlopen", _urlopen([404]))
    assert dtcc_client.fetch_cumulative("RATES", "2026-10-01", sleep=lambda s: None) is None


def test_client_retries_a_503_burst(monkeypatch):
    monkeypatch.setattr(dtcc_client.urllib.request, "urlopen", _urlopen([503, 503, _zip()]))
    assert dtcc_client.fetch_cumulative("RATES", "2026-09-29", sleep=lambda s: None)[:2] == b"PK"


def test_client_gives_up_after_its_retries(monkeypatch):
    monkeypatch.setattr(dtcc_client.urllib.request, "urlopen", _urlopen([503] * (dtcc_client.RETRIES + 1)))
    with pytest.raises(dtcc_client.DtccError, match="HTTP 503"):
        dtcc_client.fetch_cumulative("RATES", "2026-09-29", sleep=lambda s: None)


# ------------------------------------------------------------------ raw cycle source
def _ctx(tmp_path, start, end):
    return StepContext(D(start), D(end), D(end), CyclePaths.under(tmp_path))


def _checks(outcome):
    return {c.name: c for c in outcome.checks}


def test_raw_step_archives_the_window_and_lookback_then_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pdtcc, "FETCH", _fetch_all(calls))
    monkeypatch.setattr(raw_dtcc, "DTCC_RETENTION_DAYS", 5)
    first = execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-28", "2026-09-30"))
    checks = _checks(first)
    assert all(checks[n].passed for n in ("dtcc_fetch_ok", "dtcc_complete", "dtcc_sane"))
    assert calls == list(pd.date_range("2026-09-25", "2026-09-30"))  # window + 5-day lookback
    execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-28", "2026-09-30"))
    assert len(calls) == 6  # the next run: no request at all


def test_a_day_dtcc_should_have_but_does_not_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(raw_dtcc, "DTCC_RETENTION_DAYS", 5)
    outcome = execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-28", "2026-09-30"))  # conftest: nothing published
    checks = _checks(outcome)
    assert checks["dtcc_fetch_ok"].passed  # a 404 is not a download error...
    assert not checks["dtcc_complete"].passed  # ...but long-ended days still missing are flagged (warn)
    assert outcome.status == "ok"  # warn-level: the day's vintage is not lost over it


def test_a_changed_layout_is_flagged(tmp_path, monkeypatch):
    renamed = _zip(f"{HEADER.replace('UPI FISN', 'Product FISN')}\n{ROW}\n")
    monkeypatch.setattr(pdtcc, "FETCH", _fetch_all(content=renamed))
    monkeypatch.setattr(raw_dtcc, "DTCC_RETENTION_DAYS", 0)
    checks = _checks(execute_step(RAW_STEP, _ctx(tmp_path, "2026-09-29", "2026-09-29")))
    assert not checks["dtcc_sane"].passed and "UPI FISN" in checks["dtcc_sane"].message + str(checks["dtcc_sane"].details)
    assert pdtcc.archived_days("RATES", root=tmp_path / "RawData" / "DTCC") == {D("2026-09-29")}  # still archived raw
