"""CFTC Traders in Financial Futures rows -> a flat frame. Pure, no I/O.

The API returns every value as a string. Kept: the market's identity (code, names, units)
as strings, and EVERY numeric field (positions, changes, % of open interest, trader counts,
concentration ratios) - counts as nullable ``Int64``, ``pct_*`` / ``conc_*`` as ``float64``. Each row is
point in time: ``known_from`` = the scheduled release, ``CFTC_TFF_RELEASE`` (Friday 15:30
New York after the Tuesday it reports) - a shutdown-delayed release is LATER than that
(``TOFIX.md``).
"""
from __future__ import annotations

import pandas as pd

from infra.trading_calendar import snap_instants

KEYS = ["timestamp", "report", "market_code"]
_RENAME = {"report_date_as_yyyy_mm_dd": "timestamp", "cftc_contract_market_code": "market_code",
           "market_and_exchange_names": "market_name"}
_DROP = {"id", "yyyy_report_week_ww", "futonly_or_combined"}
# strings even when they look numeric (codes keep leading zeros)
# the only fractional fields (verified 2026-10-02 on both reports): % of open interest and
# the concentration ratios; everything else is a count (contracts or traders)
_FRACTIONAL = ("pct_", "conc_")
_TEXT = {"market_code", "market_name", "contract_market_name", "commodity_name", "contract_units",
         "commodity", "commodity_subgroup_name", "commodity_group_name"}


def parse_tff(rows: list[dict], report: str, release: tuple[str, str, int]) -> pd.DataFrame:
    """API rows of one report (``"futures"`` / ``"combined"``) -> one row per (report day,
    market), sorted. ``release`` = (local time, zone, days after the report day)."""
    if not rows:
        return pd.DataFrame(columns=[*KEYS, "known_from"])
    df = pd.DataFrame(rows).rename(columns=_RENAME).drop(columns=[c for c in _DROP if c in rows[0]], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize().astype("datetime64[ms]")
    for c in df.columns:
        if c == "timestamp":
            continue
        if c in _TEXT or c.endswith("_code"):
            df[c] = df[c].astype("string").str.strip()
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        # by NAME, never by the batch's values (a batch of whole percentages would otherwise
        # store a column as Int64 in one file and float64 in the next)
        df[c] = x.astype("float64") if c.startswith(_FRACTIONAL) else x.round().astype("Int64")
    df["report"] = report
    local_time, zone, lag = release
    df["known_from"] = snap_instants(df["timestamp"] + pd.Timedelta(days=lag), local_time, zone).astype("datetime64[ms]")
    df = df.drop_duplicates(KEYS, keep="last").sort_values(KEYS, ignore_index=True)
    return df[[*KEYS, "known_from", *[c for c in df.columns if c not in (*KEYS, "known_from")]]]


def position_identity_gaps(df: pd.DataFrame, tolerance: int = 1) -> pd.DataFrame:
    """Rows where reportable + non-reportable longs (or shorts) miss open interest by more
    than ``tolerance`` contracts - the report's own accounting identity; a sanity check on
    parsing. Measured 2026-10-02 over all 46,838 futures rows: exact except +-1 contract on
    942 rows, all in CONSOLIDATED markets (e.g. S&P 500 Consolidated, which mixes contract
    sizes) - the source's rounding, hence the default tolerance of 1."""
    long_ = df["tot_rept_positions_long_all"] + df["nonrept_positions_long_all"]
    short = df["tot_rept_positions_short"] + df["nonrept_positions_short_all"]
    oi = df["open_interest_all"]
    bad = ((long_ - oi).abs() > tolerance) | ((short - oi).abs() > tolerance)
    return df.loc[bad.fillna(False), [*KEYS, "open_interest_all"]]
