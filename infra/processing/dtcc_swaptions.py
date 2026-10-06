"""DTCC public swaption reports (CFTC Part 43) -> typed records, an open-interest ledger
and executed prints. Pure, no I/O. Root CLAUDE.md 16; spec ``infra.config.SwaptionSpec``.

How a swaption's life shows up in the RATES files (verified on two years of USD SOFR
swaptions, 2026-10-06):
* ``NEWT`` starts a trade under its own ``Dissemination Identifier``; every later record
  (``CORR`` / ``MODI`` / ``REVI`` / ``TERM`` / ``EROR``) points back to it through
  ``Original Dissemination Identifier`` (the rest point to trades executed before the
  archive began, 2024-09-30);
* **the identifiers changed format on 2025-11-02** (``trade_key``): from then on an id is
  ``base x 10^9 + suffix``, a NEWT carrying suffix 101 / 201 / ... and later records
  pointing to the same base with another suffix (0 / 256 / 512 ...) - matching ids
  exactly links NONE of them, matching the base links 88% (the rest predate the archive).
  Bases overlap the old ids' range, so keys are namespaced (``S<id>`` / ``L<base>``);
* ``NEWT`` with event ``TRAD`` is an execution; ``NEWT`` ``NOVA`` / ``COMP`` / ``EXER``
  re-report existing risk under a new id (a novation's step-in, a compression's
  replacement) - for open interest they are trades like any other, their old side
  closed by its own ``TERM`` / ``MODI``;
* ``MODI`` / ``CORR`` carry the trade's full current terms (a partial unwind lowers the
  notional); ``TERM`` closes (``ETRM`` unwind, ``EXER`` exercise, ``COMP`` compression,
  ``NOVA`` novation); ``EROR`` cancels; ``REVI`` revives.
* The Call/Put label doesn't say payer or receiver (``infra.config.SwaptionSpec``).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

RAW_COLUMNS = {
    "Dissemination Identifier": "diss_id",
    "Original Dissemination Identifier": "orig_id",
    "Action type": "action",
    "Event type": "event",
    "Event timestamp": "event_ts",
    "Execution Timestamp": "executed",
    "Expiration Date": "expiry",
    "Maturity date of the underlier": "maturity",
    "Notional amount-Leg 1": "notional_raw",
    "Strike Price": "strike_raw",
    "Strike price notation": "strike_notation",
    "Option Premium Amount": "premium_raw",
    "Option Premium Currency": "premium_ccy",
    "Platform identifier": "platform",
    "Package indicator": "package",
    "Block trade election indicator": "block",
    "UPI FISN": "fisn",
}
RECORD_COLUMNS = ["file_day", "diss_id", "trade_id", "action", "event", "event_ts", "executed", "expiry", "maturity",
                  "strike", "notional", "capped", "premium", "label", "package", "block", "platform"]
STATE_FIELDS = ["executed", "expiry", "maturity", "strike", "notional", "capped", "premium", "label", "package",
                "platform"]
_ORDER = {"NEWT": 0, "MODI": 1, "CORR": 2, "REVI": 3, "TERM": 4, "EROR": 5}
DECIMAL_NOTATION = "3"  # CFTC price notation 3 = decimal (0.0425 = 4.25%)


LONG_ID = 10 ** 12  # ids at or above: the post-2025-11-02 format
ID_SUFFIX = 10 ** 9


def trade_key(ids: pd.Series) -> pd.Series:
    """Dissemination ids (Int64) -> the key every record of one trade shares: ``S<id>``
    for the old format, ``L<id // 10^9>`` for the new one."""
    x = ids.astype("Int64")
    out = pd.Series(pd.NA, index=ids.index, dtype="string")
    long_ = x >= LONG_ID
    out[long_.fillna(False)] = "L" + (x[long_.fillna(False)] // ID_SUFFIX).astype(str)
    short = (x < LONG_ID).fillna(False)
    out[short] = "S" + x[short].astype(str)
    return out


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False).str.rstrip("+"), errors="coerce")


def normalize(raw: pd.DataFrame, file_day, fisn_pattern: str) -> pd.DataFrame:
    """One file's rows (text, as read) -> the swaptions matching ``fisn_pattern``, typed.
    ``strike`` in PERCENT (NaN unless decimal notation); ``notional`` at the cap floor when
    ``capped``; times tz-naive UTC; ``trade_id`` = the NEWT's dissemination id."""
    df = raw[[c for c in RAW_COLUMNS if c in raw.columns]].rename(columns=RAW_COLUMNS)
    for c in RAW_COLUMNS.values():
        if c not in df:
            df[c] = pd.NA
    df = df[df["fisn"].astype("string").str.match(fisn_pattern, na=False)].copy()
    if df.empty:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    ids = lambda s: pd.to_numeric(s, errors="coerce").astype("Int64")
    df["diss_id"], df["orig_id"] = ids(df["diss_id"]), ids(df["orig_id"])
    df["trade_id"] = trade_key(df["orig_id"]).fillna(trade_key(df["diss_id"]))
    df["file_day"] = pd.Timestamp(file_day).normalize()
    for c in ("event_ts", "executed"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_localize(None)
    for c in ("expiry", "maturity"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    notation = df["strike_notation"].astype("string").str.replace(".0", "", regex=False)
    df["strike"] = (_num(df["strike_raw"]) * 100.0).where(notation == DECIMAL_NOTATION)
    df["capped"] = df["notional_raw"].astype("string").str.strip().str.endswith("+").fillna(False).astype(bool)
    df["notional"] = _num(df["notional_raw"])
    df["premium"] = _num(df["premium_raw"]).where(df["premium_ccy"].astype("string").isin(["USD"]) | df["premium_raw"].isna())
    df["label"] = np.where(df["fisn"].astype(str).str.contains(" Call "), "Call", "Put")
    for c in ("package", "block"):
        df[c] = df[c].astype("string").str.strip().str.lower().eq("true").fillna(False).astype(bool)
    df["action"], df["event"] = df["action"].astype("string"), df["event"].astype("string")
    df["platform"] = df["platform"].astype("string")
    out = df[RECORD_COLUMNS].copy()
    for c in ("file_day", "event_ts", "executed", "expiry", "maturity"):
        out[c] = out[c].astype("datetime64[ms]")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------- open interest
def ledger_versions(records: pd.DataFrame) -> pd.DataFrame:
    """Each trade's state per dissemination day: one row per (``trade_id``, ``valid_from`` =
    file day) with ``status`` (open / closed / cancelled) and the terms as last reported,
    ``valid_to`` = the next version's day (NaT = still current), ``origin`` = "trade" (its
    NEWT is in the archive) or "lifecycle" (first seen through a later record: a trade
    executed before the archive, open by then). Point in time by construction: a version
    only uses records disseminated by its day."""
    if records.empty:
        return pd.DataFrame(columns=["trade_id", "valid_from", "valid_to", "status", "origin"] + STATE_FIELDS)
    r = records.assign(_o=records["action"].map(_ORDER).fillna(9)).sort_values(
        ["trade_id", "file_day", "event_ts", "_o", "diss_id"], kind="stable")
    status = r["action"].map({"TERM": "closed", "EROR": "cancelled"}).fillna("open")
    r = r.assign(status=status)
    # terms carry forward: a later record without a field keeps the earlier value
    r[STATE_FIELDS] = r.groupby("trade_id")[STATE_FIELDS].ffill()
    newt = set(records.loc[records["action"] == "NEWT", "trade_id"].dropna())
    last = r.groupby(["trade_id", "file_day"], sort=False).tail(1)  # the day's final state
    v = last.rename(columns={"file_day": "valid_from"})[["trade_id", "valid_from", "status"] + STATE_FIELDS].copy()
    v["valid_to"] = v.groupby("trade_id")["valid_from"].shift(-1)
    v["origin"] = np.where(v["trade_id"].isin(newt), "trade", "lifecycle")
    return v.reset_index(drop=True)


def open_at(versions: pd.DataFrame, day, *, max_notional: float) -> pd.DataFrame:
    """Trades open after day ``day``'s file: current version, status open, expiry after
    the day, a usable notional."""
    d = pd.Timestamp(day).normalize()
    v = versions
    cur = (v["valid_from"] <= d) & (v["valid_to"].isna() | (v["valid_to"] > d))
    ok = cur & (v["status"] == "open") & (v["expiry"] > d) & (v["notional"] > 0) & (v["notional"] <= max_notional)
    return v[ok].copy()


# ------------------------------------------------------------------------ executions
def executions(records: pd.DataFrame, *, as_of, fresh_days: int) -> pd.DataFrame:
    """Executed trades (``NEWT`` / ``TRAD`` disseminated within ``fresh_days`` of the
    execution) in their latest state known by file day ``as_of``: a ``CORR`` replaces the
    terms, an ``EROR`` drops the trade. ``MODI`` / ``TERM`` are later life, not the
    executed price, and are ignored."""
    r = records[records["file_day"] <= pd.Timestamp(as_of)]
    newt = r[(r["action"] == "NEWT") & (r["event"] == "TRAD")]
    lag = (newt["file_day"] - newt["executed"].dt.normalize()).dt.days
    newt = newt[lag.between(0, fresh_days)]
    fix = r[r["action"].isin(["CORR", "EROR"]) & r["trade_id"].isin(newt["trade_id"])]
    ev = pd.concat([newt, fix]).assign(_o=lambda x: x["action"].map(_ORDER))
    last = ev.sort_values(["trade_id", "file_day", "event_ts", "_o"], kind="stable").groupby("trade_id").tail(1)
    return last[last["action"] != "EROR"].drop(columns="_o").reset_index(drop=True)
