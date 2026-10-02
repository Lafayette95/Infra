"""infra.storage.parquet_store: reads that race a concurrent writer."""
from __future__ import annotations

import pandas as pd
import pytest


def test_a_read_racing_a_concurrent_rewrite_is_retried(tmp_path, monkeypatch):
    """A file replaced between the dataset's listing and its read fails once with a
    truncated-page error; the read lists the store again and succeeds."""
    from infra.storage import parquet_store as store

    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-02"]).astype("datetime64[ms]"), "ticker": ["A"], "v": [1]})
    store.write_partitioned(df, tmp_path, ["timestamp", "ticker"])
    real, calls = store._dataset, {"n": 0}

    class Flaky:
        def __init__(self, dataset, fail):
            self.dataset, self.fail = dataset, fail

        def to_table(self, *a, **k):
            calls["n"] += 1
            if self.fail:
                raise OSError("Unexpected end of stream: Page was smaller (162) than expected (210)")
            return self.dataset.to_table(*a, **k)

    monkeypatch.setattr(store, "READ_RETRY_PAUSE_S", 0.0)
    monkeypatch.setattr(store, "_dataset", lambda root: Flaky(real(root), fail=calls["n"] == 0))
    assert store.read_partitioned(tmp_path)["v"].tolist() == [1] and calls["n"] == 2

    monkeypatch.setattr(store, "_dataset", lambda root: Flaky(real(root), fail=True))
    with pytest.raises(OSError):
        store.read_partitioned(tmp_path)
