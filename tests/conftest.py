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


@pytest.fixture(autouse=True)
def _no_bond_network(monkeypatch):
    """The px step fetches cash-bond curves from public websites (infra.pipeline.bonds.
    SOURCES); no test may reach them. By default every source answers "nothing
    published"; bond tests pass their own fake ``sources`` explicitly."""
    import pandas as pd

    def nothing(start, end):
        return pd.DataFrame(columns=["timestamp", "maturity", "value"]), []

    monkeypatch.setattr("infra.pipeline.bonds.SOURCES", {"treasury": nothing, "boe": nothing, "bundesbank": nothing})


@pytest.fixture(autouse=True)
def _network_always_ready(monkeypatch):
    """The Prefect flows wait for real DNS before running (infra.cycle.network); tests
    must never depend on - or hang for 10 minutes without - a network."""
    monkeypatch.setattr("infra.cycle.flows.wait_for_network", lambda *a, **k: 0.0)
