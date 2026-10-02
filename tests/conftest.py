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


@pytest.fixture(autouse=True)
def _no_release_network(monkeypatch):
    """The raw step fetches macro-release vintages from FRED (infra.pipeline.releases.
    SOURCES); no test may reach it. By default every source answers "nothing published";
    release tests pass their own fake ``sources`` explicitly."""
    import pandas as pd

    def nothing(series_id, start, end):
        return pd.DataFrame(columns=["realtime_start", "date", "value"]), []

    monkeypatch.setattr("infra.pipeline.releases.SOURCES",
                        {"fred": nothing, "calendar": nothing, "fred+prelims": nothing})
    monkeypatch.setattr("infra.pipeline.releases.CONFIGURED",
                        {"fred": lambda: True, "calendar": lambda: True, "fred+prelims": lambda: True})


@pytest.fixture(autouse=True)
def _no_bulk_network(monkeypatch):
    """The raw step snapshots the BLS/BEA bulk files (infra.pipeline.bulk_series); no test
    may reach them. By default no source has published anything (a version of None is
    never downloaded); bulk tests pass their own fakes explicitly."""
    monkeypatch.setattr("infra.pipeline.bulk_series.LAST_MODIFIED", {"bls": lambda d: None, "bea": lambda d: None})
    monkeypatch.setattr("infra.pipeline.bulk_series.CONFIGURED", {"bls": lambda: True, "bea": lambda: True})


@pytest.fixture(autouse=True)
def _no_cpi_weights_network(monkeypatch):
    """The raw step fetches due CPI weight years from bls.gov (infra.pipeline.cpi_weights);
    no test may reach it. By default BLS has published nothing."""
    monkeypatch.setattr("infra.pipeline.cpi_weights.FETCH", lambda years: {})
    monkeypatch.setattr("infra.pipeline.cpi_weights.CONFIGURED", lambda: True)


@pytest.fixture(autouse=True)
def _no_dtcc_network(monkeypatch):
    """The raw step archives DTCC's daily swap-trade reports (infra.pipeline.dtcc.FETCH);
    no test may reach DTCC. By default DTCC has published nothing; DTCC tests pass their
    own fake ``fetch`` explicitly."""
    monkeypatch.setattr("infra.pipeline.dtcc.FETCH", lambda kind, day: None)
    monkeypatch.setattr("infra.pipeline.dtcc.PAUSE_S", 0.0)


@pytest.fixture(autouse=True)
def _no_reference_network(monkeypatch):
    """The raw step refreshes the release calendar (FRED release dates, NAR), Treasury
    auctions (Fiscal Data) and auction tails (the Wayback Machine) - no test may reach
    them. Each hook answers "nothing new"; tests of those pieces pass their own."""
    import pandas as pd

    monkeypatch.setattr("infra.cycle.raw_reference.FRED_DATES_FETCH", lambda release_id: pd.DatetimeIndex([]))
    monkeypatch.setattr("infra.cycle.raw_reference.NAR_FETCH", lambda: "")
    monkeypatch.setattr("infra.cycle.raw_reference.TREASURY_SCHEDULE_FETCH", lambda: (
        "<AuctionCalendar><StartDate>2000-01-01</StartDate></AuctionCalendar>"))
    monkeypatch.setattr("infra.cycle.raw_reference.AUCTIONS_FETCH", lambda since: pd.DataFrame())
    monkeypatch.setattr("infra.cycle.raw_reference.TAILS_LIST", lambda *a, **k: pd.DataFrame(
        {"timestamp": pd.Series(dtype="datetime64[ms]"), "original": pd.Series(dtype=str), "digest": pd.Series(dtype=str)}))
    monkeypatch.setattr("infra.cycle.raw_reference.TAILS_FETCH", lambda *a, **k: b"")


@pytest.fixture(autouse=True)
def _no_cme_ftp_network(monkeypatch):
    """CME's conversion-factor files come from its FTP (infra.pipeline.futures_baskets
    LIST/FETCH); no test may reach it. By default nothing is listed."""
    monkeypatch.setattr("infra.pipeline.futures_baskets.LIST", lambda: [])
    monkeypatch.setattr("infra.pipeline.futures_baskets.PAUSE_S", 0.0)
