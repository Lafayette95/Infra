"""Suite-wide safety net: code that reads the adjustments log by DEFAULT (consumers of
cleaned data - infra.pipeline.daily.read_daily_from_disk, infra.pipeline.wirp) must never
see the real ~/Database/_adjustments during tests. Tests use real contract tickers
(SR3Z4, ZNH5, ...), so a real bad print recorded one day would otherwise change unrelated
tests' results. Tests that exercise adjustments pass their own directory explicitly."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hermetic_adjustments_log(tmp_path_factory, monkeypatch):
    empty = tmp_path_factory.mktemp("no_adjustments")
    monkeypatch.setattr("infra.pipeline.daily.ADJUSTMENTS_DIR", empty)
    monkeypatch.setattr("infra.pipeline.wirp.ADJUSTMENTS_DIR", empty)
