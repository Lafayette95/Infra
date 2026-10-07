"""Bank of Canada benchmark bonds - pure (no I/O): the current-benchmark list on the BoC's
"Selected benchmark bond yields" page, and the benchmark-switch correction of the
benchmark yield P&L.

The page states, per term, the benchmark issue and the date it became the benchmark:
"2 year - 2028.08.01, 2.75% (2026.08.06); ... Long - 2057.12.01, 3.50% (2026.03.06)" (verified
2026-10-07; the series behind ``CA_BOND_<t>y``, ``infra.api.boc_client``).

Switch correction: on a switch day the benchmark series moves from the old bond's yield to
the new one's, so its change = the old bond's move + (new - old) spread. The P&L of the bond
held the day before is the change minus that spread, estimated by pricing both bonds off the
BoC's fitted zero curve (continuous compounding, Bolder-Johnson-Metzler) of the latest day at
least ``SPREAD_LAG_DAYS`` before - a fixed lag, so the result never depends on when it is
computed (the curve is published weekly, two weeks late).
"""
from __future__ import annotations

import html as _html
import re

import numpy as np
import pandas as pd

TENORS = {"2 year": 2, "3 year": 3, "5 year": 5, "7 year": 7, "10 year": 10, "Long": 30}
COLUMNS = ["tenor", "maturity_date", "coupon", "effective"]
SPREAD_LAG_DAYS = 21
# tenors whose switch days are corrected. Checked 2026-10-07 on the page's own switch days
# 2011-2026: the excess move on the effective day (2-6bp median) falls to ~1bp once the
# predicted spread is taken out, and the days either side carry no jump (0.3-0.5bp) - the
# effective date IS the series' switch day. Not the 30y: its switches go between bonds of
# near-equal yield, and the correction made the median error worse (1.0 -> 1.6bp, 6 cases).
CORRECTED_TENORS = (2, 3, 5, 7, 10)
_ENTRY = re.compile(r"(2 year|3 year|5 year|7 year|10 year|Long)\s*-\s*(\d{4})\.(\d{2})\.(\d{2}),\s*([\d.]+)\s*%\s*"
                    r"\((\d{4})\.(\d{2})\.(\d{2})\)")


def parse_benchmark_page(page: bytes | str) -> pd.DataFrame:
    """``tenor, maturity_date, coupon, effective`` for each benchmark the page lists (the
    real-return bond is left out)."""
    text = page.decode("utf-8", "replace") if isinstance(page, bytes) else page
    text = _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))).replace("‑", "-")
    rows = {}
    for m in _ENTRY.finditer(text):
        rows[TENORS[m.group(1)]] = {"tenor": TENORS[m.group(1)],
                                    "maturity_date": pd.Timestamp(f"{m.group(2)}-{m.group(3)}-{m.group(4)}"),
                                    "coupon": float(m.group(5)),
                                    "effective": pd.Timestamp(f"{m.group(6)}-{m.group(7)}-{m.group(8)}")}
    return pd.DataFrame(list(rows.values()), columns=COLUMNS)


def benchmark_history(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Every (tenor, bond) seen across snapshots, with its effective date - one row per
    benchmark period, ``tenor, maturity_date, coupon, effective``, ordered."""
    if snapshots.empty:
        return pd.DataFrame(columns=COLUMNS)
    h = snapshots.drop_duplicates(["tenor", "maturity_date", "coupon", "effective"])
    return h.sort_values(["tenor", "effective"]).reset_index(drop=True)[COLUMNS]


def bond_yield_from_zero(coupon: float, maturity, day, mats: np.ndarray, zero: np.ndarray) -> float:
    """Semi-annual yield of a GoC bond priced off a zero curve (continuously compounded,
    ``mats`` years) - the street yield, settlement T+1."""
    from infra.analytics import treasury_curve as tc
    ok = ~np.isnan(zero)
    t, a = tc.cash_flows(float(coupon), pd.Timestamp(maturity), pd.Timestamp(day) + pd.offsets.BDay(1), 2)
    if t.size == 0 or ok.sum() < 10:
        return float("nan")
    r = np.interp(t, mats[ok], zero[ok])
    return tc.ytm(float((a * np.exp(-r * t)).sum()), t, a, 2, guess=0.03)
