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

    monkeypatch.setattr("infra.pipeline.bonds.SOURCES", {"treasury": nothing, "boe": nothing, "bundesbank": nothing,
                                                          "mof": nothing, "boc": nothing})
    monkeypatch.setattr("infra.pipeline.boe_ois.FETCH", nothing)
    monkeypatch.setattr("infra.pipeline.boc_benchmarks.FETCH_PAGE", lambda: None)
    monkeypatch.setattr("infra.pipeline.boc_benchmarks.FETCH_ZERO", lambda start, end: None)
    monkeypatch.setattr("infra.pipeline.bunds.FETCH_PRICES", lambda start, end, isin="": "")
    monkeypatch.setattr("infra.pipeline.bunds.FETCH_ISSUANCE", lambda: None)

    def no_outlook(name):
        raise FileNotFoundError(name)

    monkeypatch.setattr("infra.pipeline.de_issuance.FETCH_PAGE", lambda: "")
    monkeypatch.setattr("infra.pipeline.de_issuance.LAST_MODIFIED", no_outlook)
    monkeypatch.setattr("infra.pipeline.de_issuance.FETCH_OUTLOOK", no_outlook)
    monkeypatch.setattr("infra.pipeline.jgb_auctions.FETCH", no_outlook)
    monkeypatch.setattr("infra.pipeline.jgb_auctions.LAST_MODIFIED", no_outlook)
    monkeypatch.setattr("infra.pipeline.goc_auctions.FETCH_GROUP", no_outlook)
    # the JGB / Canadian refreshes are all network: a test run fetches nothing (a test of a
    # refresh itself must restore ``update`` and stub FETCH / FETCH_GROUP)
    monkeypatch.setattr("infra.pipeline.jgb_auctions.update", lambda **kw: {})
    monkeypatch.setattr("infra.pipeline.goc_auctions.update", lambda **kw: {})


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
def _no_fedinvest_network(monkeypatch):
    """Treasury prices come from FedInvest (infra.pipeline.treasury_prices.FETCH); no test
    may reach it. By default every day is an empty page; price tests pass fakes."""
    import pandas as pd

    from infra.api.fedinvest_client import COLUMNS
    monkeypatch.setattr("infra.pipeline.treasury_prices.FETCH", lambda day: pd.DataFrame(columns=COLUMNS))
    monkeypatch.setattr("infra.pipeline.treasury_prices.PAUSE_S", 0.0)


@pytest.fixture(autouse=True)
def _no_repo_network(monkeypatch):
    """Repo rates (NY Fed, OFR, the DTCC GCF workbook) and the NY Fed's securities
    lending (infra.pipeline.repo / sec_lending hooks); no test may reach them. By default
    nothing is published; repo tests set their own fakes."""
    def no_workbook():
        raise RuntimeError("no GCF workbook in tests")

    monkeypatch.setattr("infra.pipeline.repo.NYFED_FETCH", lambda rate, start, end: [])
    monkeypatch.setattr("infra.pipeline.repo.OFR_FETCH", lambda mnemonics, start=None: {})
    monkeypatch.setattr("infra.pipeline.repo.GCF_FETCH", no_workbook)
    monkeypatch.setattr("infra.pipeline.sec_lending.FETCH", lambda start, end: [])


@pytest.fixture(autouse=True)
def _no_cme_ftp_network(monkeypatch):
    """CME's conversion-factor files come from its FTP (infra.pipeline.futures_baskets
    LIST/FETCH); no test may reach it. By default nothing is listed."""
    monkeypatch.setattr("infra.pipeline.futures_baskets.LIST", lambda: [])
    monkeypatch.setattr("infra.pipeline.futures_baskets.PAUSE_S", 0.0)


@pytest.fixture(autouse=True)
def _no_databento_availability_network(monkeypatch):
    """Ranges ending recently make infra.pipeline.daily.bounded_by_availability ask
    Databento's metadata API how far a dataset is published (api.available_end) - a live
    request with the real key, so a test over recent dates silently hit the network, and
    failed without the key or offline. By default the data counts as available up to now;
    tests of the availability logic set their own stub."""
    import pandas as pd

    monkeypatch.setattr("infra.api.databento_client.available_end",
                        lambda *a, **k: pd.Timestamp.now(tz="UTC").tz_localize(None))


@pytest.fixture(autouse=True)
def _no_external_network(monkeypatch):
    """No test may reach the outside world: every non-loopback connection fails AT ONCE,
    naming the address - so an unstubbed fetch shows up as a clear error instead of a
    hang (seen 2026-10-02: one suite run stalled >10 minutes in a raw-step test; the
    committed code was then verified to make no external call). Loopback stays open -
    Prefect's test harness runs a local server."""
    import ipaddress
    import socket

    def _is_loopback(address) -> bool:
        host = address[0] if isinstance(address, tuple) else address
        if host in ("localhost",):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    connect, create_connection = socket.socket.connect, socket.create_connection

    def guarded_connect(self, address):
        if not _is_loopback(address):
            raise OSError(f"external network blocked in tests: {address} - stub this call (tests/conftest.py)")
        return connect(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        if not _is_loopback(address):
            raise OSError(f"external network blocked in tests: {address} - stub this call (tests/conftest.py)")
        return create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


@pytest.fixture(autouse=True)
def _hermetic_bond_futures_dv01(monkeypatch):
    """The bmk step's bond-futures DV01 reads the real cash, basket and funding stores:
    suite-wide it finds nothing (every US bond row NaN with a reason); tests that want
    values stub ``infra.cycle.bmk.BOND_FUTURES_DV01`` themselves."""
    import pandas as pd
    empty = pd.DataFrame(columns=["timestamp", "ticker", "futures_dv01", "ctd"])
    monkeypatch.setattr("infra.cycle.bmk.BOND_FUTURES_DV01", lambda start, end, tickers: empty)
    monkeypatch.setattr("infra.cycle.bmk.EUREX_FUTURES_DV01", lambda start, end, tickers: empty)
    monkeypatch.setattr("infra.pipeline.eurex_basis.FETCH", lambda: None)


@pytest.fixture(autouse=True)
def _no_positioning_network(monkeypatch):
    """CFTC TFF and NY Fed primary dealer fetches find nothing new suite-wide; tests that
    exercise them pass their own fakes."""
    monkeypatch.setattr("infra.pipeline.cftc_tff.LAST_MODIFIED", lambda dataset: None)
    monkeypatch.setattr("infra.pipeline.cftc_tff.FETCH", lambda dataset, since=None: [])
    monkeypatch.setattr("infra.pipeline.primary_dealer.FETCH",
                        lambda: '"As Of Date","Time Series","Value (millions)"\n')
    monkeypatch.setattr("infra.pipeline.primary_dealer.CATALOG_FETCH", lambda: ([], []))


@pytest.fixture(autouse=True)
def _no_fed_gsw_network(monkeypatch):
    """The Fed GSW curve file is never fetched in tests."""
    def refuse():
        raise RuntimeError("Fed GSW fetch blocked in tests")
    monkeypatch.setattr("infra.pipeline.fed_gsw.FETCH", refuse)
