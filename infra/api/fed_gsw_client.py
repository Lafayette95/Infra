"""Federal Reserve Board: the Gurkaynak-Sack-Wright (GSW) nominal Treasury curve - NETWORK
ONLY, no files. One CSV (``feds200628.csv``, ~16 MB) with every day since 1961: Svensson
parameters (BETA0-3 in %, TAU1-2 in years), zero yields SVENY01-30 (continuously
compounded, %), par yields SVENPY01-30 (coupon-equivalent, %), forwards. Free, no key;
a staff research product, published with ~a week's lag (verified 2026-10-04: latest row
2026-09-25). The header has ~9 note lines before the ``Date,...`` row.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request

URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
# the TIPS (real) curve, same methodology and layout family (BETA0-3, TAU1-2, TIPSY.., BKEVEN..)
TIPS_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv"
TIMEOUT_S = 120
RETRIES = 3


class FedGswError(RuntimeError):
    pass


def fetch_csv(*, url: str = URL, sleep=time.sleep) -> str:
    last = None
    for attempt in range(RETRIES + 1):
        if attempt:
            sleep(2.0 * 2 ** (attempt - 1))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "infra-data-pipeline"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.read().decode("utf-8-sig")
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise FedGswError(last)
