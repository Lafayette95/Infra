"""infra.cycle.network: a scheduled run waits for DNS before starting (the 2026-09-30
dark-wake failure), requesting a full wake at most once, and fails with ONE clear error."""
from __future__ import annotations

import pytest

from infra.cycle.network import NetworkUnavailable, REQUIRED_HOSTS, wait_for_network


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _resolver(down_until: float, clock: _Clock):
    def resolve(host):
        if clock.t < down_until:
            raise OSError(8, "nodename nor servname provided, or not known")
        return "1.2.3.4"
    return resolve


def test_network_up_returns_immediately_without_waking():
    clock, wakes = _Clock(), []
    assert wait_for_network(resolve=_resolver(0, clock), wake=lambda: wakes.append(1),
                            sleep=clock.sleep, clock=clock) == 0.0
    assert wakes == []


def test_waits_for_dns_after_one_full_wake_request():
    clock, wakes = _Clock(), []
    waited = wait_for_network(resolve=_resolver(40, clock), wake=lambda: wakes.append(1),
                              sleep=clock.sleep, clock=clock, interval_s=15)
    assert waited == 45 and wakes == [1]


def test_gives_up_with_one_clear_error():
    clock = _Clock()
    with pytest.raises(NetworkUnavailable, match="hist.databento.com"):
        wait_for_network(resolve=_resolver(10_000, clock), wake=lambda: None,
                         sleep=clock.sleep, clock=clock, timeout_s=600)
    assert clock.t == 600


def test_every_source_host_is_checked():
    assert set(REQUIRED_HOSTS) == {"hist.databento.com", "home.treasury.gov", "www.bankofengland.co.uk",
                                   "api.statistiken.bundesbank.de"}
