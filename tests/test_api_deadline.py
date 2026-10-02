"""Hard wall-clock deadline around Databento calls (infra.api.databento_client): a stalled
response must fail loudly, never hang an unattended run. No network - fake callables."""
from __future__ import annotations

import threading
import time

import pytest

from infra.api import databento_client as api

FAST = dict(deadline_s=0.2, pause_s=0.0)


def test_a_call_that_hangs_every_time_raises_api_timeout_after_all_attempts():
    calls = []
    release = threading.Event()

    def hang():
        calls.append(1)
        release.wait(5)  # far longer than the deadline

    t0 = time.monotonic()
    with pytest.raises(api.ApiTimeout):
        api.call_with_deadline(hang, what="t", attempts=3, **FAST)
    assert len(calls) == 3 and time.monotonic() - t0 < 2.0
    release.set()


def test_a_transient_hang_recovers_on_retry():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            time.sleep(1.0)  # first attempt stalls past the deadline
        return "ok"

    assert api.call_with_deadline(flaky, what="t", attempts=3, **FAST) == "ok" and len(calls) == 2


def test_real_api_errors_are_raised_immediately_not_retried():
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("422 data_end_after_available_end")

    with pytest.raises(ValueError):
        api.call_with_deadline(bad, what="t", attempts=3, **FAST)
    assert len(calls) == 1


def test_every_client_call_in_the_api_module_is_wrapped():
    """Guard: a new raw client.metadata/timeseries call added later without the deadline."""
    import ast
    import inspect
    src = inspect.getsource(api)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Attribute) and owner.attr in ("metadata", "timeseries"):
                segment = ast.get_source_segment(src, node)
                line = src.splitlines()[node.lineno - 1]
                assert "lambda" in line or "lambda" in src.splitlines()[node.lineno - 2], (
                    f"unwrapped Databento call: {segment}")


# ------------------------------------------------------------ transient-error retries
from databento.common.error import BentoClientError, BentoServerError  # noqa: E402


def _flaky(exc, fail_times):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= fail_times:
            raise exc
        return "ok"
    return fn, calls


def test_a_transient_server_error_is_retried_and_recovers():
    fn, calls = _flaky(BentoServerError(http_status=503, message="unavailable"), fail_times=2)
    assert api.call_with_deadline(fn, what="t", attempts=3, **FAST) == "ok" and len(calls) == 3


def test_rate_limiting_429_is_retried():
    fn, calls = _flaky(BentoClientError(http_status=429, message="too many requests"), fail_times=1)
    assert api.call_with_deadline(fn, what="t", attempts=3, **FAST) == "ok" and len(calls) == 2


def test_a_dropped_connection_is_retried():
    import requests
    fn, calls = _flaky(requests.ConnectionError("reset by peer"), fail_times=1)
    assert api.call_with_deadline(fn, what="t", attempts=3, **FAST) == "ok"


def test_client_errors_are_never_retried():
    fn, calls = _flaky(BentoClientError(http_status=422, message="dataset_unavailable_range"), fail_times=5)
    with pytest.raises(BentoClientError):
        api.call_with_deadline(fn, what="t", attempts=3, **FAST)
    assert len(calls) == 1


def test_a_transient_error_that_persists_is_raised_after_the_last_attempt():
    fn, calls = _flaky(BentoServerError(http_status=502, message="bad gateway"), fail_times=9)
    with pytest.raises(BentoServerError):
        api.call_with_deadline(fn, what="t", attempts=3, **FAST)
    assert len(calls) == 3


def test_a_stream_that_breaks_midway_is_retried_but_other_bento_errors_are_not():
    from databento.common.error import BentoError
    from infra.api.databento_client import is_transient
    assert is_transient(BentoError("Error streaming response: Response ended prematurely"))
    assert is_transient(BentoError("Error streaming response: HTTPSConnectionPool(...): Read timed out."))
    assert not is_transient(BentoError("Symbol not found"))
