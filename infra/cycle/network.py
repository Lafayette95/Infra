"""Wait for the network before a scheduled run starts.

Found 2026-09-30: the scheduled run fired on time (10:00 UTC) but every request failed
DNS resolution for its whole 19 minutes (``NameResolutionError`` x91, ``[Errno 8]
nodename nor servname``) - the Mac had woken at 04:50 and was, most likely, back in a
low-power DARK WAKE by 06:00 local: processes run, Wi-Fi isn't brought up. Holding off
idle sleep (``caffeinate -i``, infra.cycle.flows) can't fix that; declaring user activity
(``caffeinate -u``) promotes a dark wake to a full wake, which brings the network up.

So a run first checks that every host it needs resolves; if not, it declares user
activity once (the display may light up briefly) and polls until they do - or gives up
with one clear ``NetworkUnavailable`` instead of dozens of per-request failures.
"""
from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import time
from typing import Callable
from urllib.parse import urlparse

from infra.api import boe_client, bundesbank_client, treasury_client

log = logging.getLogger(__name__)

# Every host a scheduled run talks to.
REQUIRED_HOSTS: tuple[str, ...] = (
    "hist.databento.com",
    urlparse(treasury_client.URL).hostname,
    urlparse(boe_client.BASE).hostname,
    urlparse(bundesbank_client.URL).hostname,
)
WAIT_TIMEOUT_S = 600
POLL_INTERVAL_S = 15


class NetworkUnavailable(RuntimeError):
    pass


def unresolved(hosts=REQUIRED_HOSTS, resolve: Callable[[str], object] = socket.gethostbyname) -> list[str]:
    """The hosts that don't resolve right now."""
    bad = []
    for host in hosts:
        try:
            resolve(host)
        except OSError:
            bad.append(host)
    return bad


def request_full_wake(seconds: int = 30) -> None:
    """Declare user activity (``caffeinate -u``) - promotes a dark wake to a full wake."""
    exe = shutil.which("caffeinate")
    if exe:
        subprocess.Popen([exe, "-u", "-t", str(seconds)])


def wait_for_network(
    hosts=REQUIRED_HOSTS,
    *,
    timeout_s: float = WAIT_TIMEOUT_S,
    interval_s: float = POLL_INTERVAL_S,
    resolve: Callable[[str], object] = socket.gethostbyname,
    wake: Callable[[], None] = request_full_wake,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Return once every host resolves (seconds waited); raise ``NetworkUnavailable``
    after ``timeout_s``. Wakes the machine fully at most once, only if needed."""
    start = clock()
    bad = unresolved(hosts, resolve)
    if not bad:
        return 0.0
    log.warning("network not ready (%s unresolved) - requesting a full wake and waiting up to %ds",
                ", ".join(bad), timeout_s)
    wake()
    while True:
        waited = clock() - start
        if waited >= timeout_s:
            raise NetworkUnavailable(
                f"no network after {waited:.0f}s - still unresolved: {', '.join(bad)} "
                "(Mac asleep / in dark wake, Wi-Fi down, or DNS failing)")
        sleep(interval_s)
        bad = unresolved(hosts, resolve)
        if not bad:
            waited = clock() - start
            log.info("network ready after %.0fs", waited)
            return waited
