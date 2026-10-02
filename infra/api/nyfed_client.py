"""New York Fed Markets Data API (markets.newyorkfed.org/api) - NETWORK ONLY, no files.

Free, no key. Two endpoints are used (verified 2026-10-02):
* secured reference rates ``rates/secured/<rate>/search.json`` - SOFR, TGCR and BGCR per
  business day from 2018-04-02 (when they started), each with its 1st/25th/75th/99th
  volume-weighted percentiles and volume ($bn). Day D is published ~08:00 New York on
  D+1 and can be revised once, the same afternoon (``revisionIndicator``);
* securities lending results ``seclending/all/results/details/search.json`` - every
  operation of the Fed's lending of its own Treasury holdings (SOMA), per CUSIP, from
  1999. Two operation types: the daily ``Securities Lending`` auction (12:00-12:15 New
  York; fee, amounts submitted/accepted, SOMA holdings, amounts available) and
  ``Extensions`` (existing loans rolled; only the par extended, posted the next morning).
  Only CUSIPs that were actually borrowed are listed. Nine months is one ~1s request.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://markets.newyorkfed.org/api"
TIMEOUT_S = 120
RETRIES = 4
BACKOFF_S = 2.0


class NyFedError(RuntimeError):
    pass


def get_json(path: str, params: dict, *, sleep=time.sleep) -> dict:
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "infra-data-pipeline"})
    last = None
    for attempt in range(RETRIES + 1):
        if attempt:
            sleep(BACKOFF_S * 2 ** (attempt - 1))
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise NyFedError(f"{path}: {last}")


def _dates(start, end) -> dict:
    return {"startDate": f"{pd.Timestamp(start):%Y-%m-%d}", "endDate": f"{pd.Timestamp(end):%Y-%m-%d}"}


def fetch_reference_rates(rate: str, start, end) -> list[dict]:
    """The published records of one secured rate (``"sofr"``, ``"tgcr"``, ``"bgcr"``) with
    effective dates in ``[start, end]`` (inclusive), as the API returns them."""
    return get_json(f"rates/secured/{rate.lower()}/search.json", _dates(start, end)).get("refRates", [])


def fetch_sec_lending(start, end) -> list[dict]:
    """Every securities-lending operation (both types) dated in ``[start, end]``
    (inclusive), each with its per-CUSIP ``details``."""
    return get_json("seclending/all/results/details/search.json", _dates(start, end)) \
        .get("seclending", {}).get("operations", [])
