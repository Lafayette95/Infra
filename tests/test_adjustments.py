"""The adjustments log (infra.storage.adjustment_store): record / read / clear / overlay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.storage import adjustment_store as adj

D = pd.Timestamp


def _row(key, day, action, original, adjusted, source="t"):
    return {"store": "S", "timestamp": D(day), "key": key, "column": "px", "action": action,
            "original": original, "adjusted": adjusted, "source": source, "detail": "",
            "run_day": D("2026-09-29")}


def test_record_read_and_overlay(tmp_path):
    adj.record(tmp_path, pd.DataFrame([_row("A", "2026-09-10", "roll", 9.0, 5.0),
                                       _row("B", "2026-09-11", "NA", 7.0, np.nan)]))
    data = pd.DataFrame({"timestamp": [D("2026-09-10"), D("2026-09-11"), D("2026-09-11")],
                         "ticker": ["A", "B", "A"], "px": [9.0, 7.0, 6.0]})
    out = adj.apply(data, adj.read(tmp_path, store="S"), key_column="ticker")
    assert out["px"].tolist()[0] == 5.0 and np.isnan(out["px"].tolist()[1]) and out["px"].tolist()[2] == 6.0
    assert data["px"].tolist() == [9.0, 7.0, 6.0]  # the input is never mutated


def test_a_later_adjustment_of_the_same_value_replaces_the_earlier(tmp_path):
    adj.record(tmp_path, pd.DataFrame([_row("A", "2026-09-10", "roll", 9.0, 5.0)]))
    adj.record(tmp_path, pd.DataFrame([_row("A", "2026-09-10", "NA", 9.0, np.nan)]))
    got = adj.read(tmp_path, store="S")
    assert len(got) == 1 and got["action"].iloc[0] == "NA"


def test_clear_only_touches_its_own_source_keys_and_window(tmp_path):
    adj.record(tmp_path, pd.DataFrame([_row("A", "2026-09-10", "roll", 1, 2, source="mine"),
                                       _row("A", "2026-09-20", "roll", 1, 2, source="mine"),
                                       _row("B", "2026-09-10", "roll", 1, 2, source="mine"),
                                       _row("A", "2026-09-11", "roll", 1, 2, source="other")]))
    removed = adj.clear(tmp_path, store="S", source="mine", keys=["A"], start=D("2026-09-01"), end=D("2026-09-15"))
    left = adj.read(tmp_path, store="S")
    assert removed == 1
    assert sorted(zip(left["key"], left["timestamp"].dt.day, left["source"])) == [
        ("A", 11, "other"), ("A", 20, "mine"), ("B", 10, "mine")]


def test_empty_log_is_a_no_op(tmp_path):
    data = pd.DataFrame({"timestamp": [D("2026-09-10")], "ticker": ["A"], "px": [9.0]})
    assert adj.apply(data, adj.read(tmp_path, store="S"), key_column="ticker") is data
