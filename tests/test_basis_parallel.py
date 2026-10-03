"""The parallel basis runner: sampling, chunking, checkpoint/resume - no model run (the
chunk worker is stubbed), no network."""
from __future__ import annotations

import pandas as pd

from infra.models.basis import parallel

D = pd.Timestamp


def test_sample_is_global_not_per_chunk():
    days = parallel.sample_days("2024-01-02", "2024-03-29", every=14)
    ch = parallel.chunks(days)
    assert [lab for lab, _ in ch] == ["2024-01", "2024-02", "2024-03"]
    assert sum(len(d) for _, d in ch) == len(days)
    assert (days[1] - days[0]).days >= 14  # every 14th BUSINESS day


def test_finished_months_are_skipped_on_rerun(tmp_path, monkeypatch):
    ran = []

    def fake_chunk(spec, days, out_dir, label):
        ran.append(label)
        frame = pd.DataFrame({"day": pd.DatetimeIndex(days), "contract": "ZNH4"})
        frame.to_parquet(f"{out_dir}/{label}_bonds.parquet")
        frame.to_parquet(f"{out_dir}/{label}_contracts.parquet")
        return label, len(frame)

    monkeypatch.setattr(parallel, "_run_chunk", fake_chunk)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", _InlinePool)
    spec = parallel.resolve_spec("M0")
    parallel.run_parallel(spec, "2024-01-02", "2024-03-29", tmp_path, every=5, log=lambda *_: None)
    assert sorted(ran) == ["2024-01", "2024-02", "2024-03"]
    (tmp_path / "2024-02_contracts.parquet").unlink()  # "interrupted" mid-run
    ran.clear()
    res = parallel.run_parallel(spec, "2024-01-02", "2024-03-29", tmp_path, every=5, log=lambda *_: None)
    assert ran == ["2024-02"] and res["contracts"]["day"].is_monotonic_increasing


def test_spec_overrides():
    assert parallel.resolve_spec("M2", {"level_betas": True}).level_betas is True
    assert parallel.resolve_spec("M2").level_betas is False


class _InlinePool:
    """A stand-in for ProcessPoolExecutor that runs tasks inline (tests only)."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args):
        from concurrent.futures import Future
        f = Future()
        f.set_result(fn(*args))
        return f
