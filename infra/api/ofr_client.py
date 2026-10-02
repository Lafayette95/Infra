"""OFR Short-term Funding Monitor API (data.financialresearch.gov/v1) - NETWORK ONLY, no files.

Free, no key. The U.S. Repo Markets release (dataset ``repo``, verified 2026-10-02):
average rates (``AR``) and transaction volumes (``TV``, dollars) per segment - ``DVP``
(FICC's cleared bilateral service), ``GCF`` (FICC's general-collateral service), ``TRI``
(tri-party) and ``TRIV1`` (tri-party excluding the Fed's own trades) - and per term
bucket (``OO`` overnight/open, ``B27`` 2-7 days, ``B830`` 8-30 days, ``LE30`` <=30 days -
replaced by B27/B830 from 2025-08-13 - ``G30`` >30 days, ``TOT``) or collateral (``T``
Treasuries, GCF and tri-party only). Mnemonics look like ``REPO-DVP_AR_OO-F``; each comes
as ``-P`` (preliminary, ~1 business day after the trade date) and ``-F`` (final, ~3
months later). ``series/multifull`` returns many mnemonics in one request - but asking for
~100 mnemonics' full history at once fails (HTTP 502, 2026-10-02), so requests are batched
(``BATCH``).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://data.financialresearch.gov/v1"
TIMEOUT_S = 120
RETRIES = 4
BACKOFF_S = 2.0
BATCH = 16  # mnemonics per request


class OfrError(RuntimeError):
    pass


def get_json(path: str, params: dict, *, sleep=time.sleep):
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
    raise OfrError(f"{path}: {last}")


def fetch_series(mnemonics, start=None) -> dict[str, list]:
    """``{mnemonic: [[date, value|None], ...]}`` for every mnemonic, observations from
    ``start`` (inclusive; None = full history), ``BATCH`` mnemonics per request."""
    mnemonics = list(mnemonics)
    out = {}
    for i in range(0, len(mnemonics), BATCH):
        chunk = mnemonics[i:i + BATCH]
        params = {"mnemonics": ",".join(chunk)}
        if start is not None:
            params["start_date"] = f"{pd.Timestamp(start):%Y-%m-%d}"
        payload = get_json("series/multifull", params)
        out.update({m: (payload.get(m) or {}).get("timeseries", {}).get("aggregation", []) for m in chunk})
    return out
