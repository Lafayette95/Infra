"""NY Fed securities lending results (CLAUDE.md 19): operations -> one typed row per
(``timestamp`` = operation day, ``operation_id``, ``cusip``). Pure, no I/O.

``operation_type`` is ``lending`` (the daily auction: amounts submitted/accepted, the
accepted bids' weighted average ``fee`` in percent, SOMA holdings and the amounts
available) or ``extension`` (an existing loan rolled: ``par_extended`` only). Amounts in
dollars (Int64 on disk: they overflow Int32), ``fee`` x10000 nullable Int32 (Rule 6b).

The fee is a lending fee, roughly the bond's SPECIALNESS (general-collateral repo minus
its own special repo rate) - but only ABOVE the program's minimum fee, which bids can't
go below and which the Fed has changed over time (``fee_floor``). Only CUSIPs actually
borrowed are listed: a bond the Fed held but nobody borrowed was not special beyond the
floor that day (censored, not missing). Holdings and availability are reported from
2005-11-07 (NaN before); the Fed lends at most 90% of its holding of an issue
(``theo_available``), less what is already out on loan (``actual_available``, rounded to
$1m). ``security_class``: ``treasury`` (CUSIP 912...) or ``agency`` - from 2009-07-09 to
2023-06-27 the program also lent the Fed's FHLB / Freddie Mac / Fannie Mae debt.
"""
from __future__ import annotations

import pandas as pd

LENDING_COLUMNS = ["timestamp", "operation_id", "operation_type", "cusip", "security_class", "description",
                   "loan_maturity",
                   "par_submitted", "par_accepted", "fee", "soma_holdings", "theo_available", "actual_available",
                   "outstanding_loans", "par_extended"]
LENDING_KEYS = ["timestamp", "operation_id", "cusip"]
AMOUNT_COLUMNS = ("par_submitted", "par_accepted", "soma_holdings", "theo_available", "actual_available",
                  "outstanding_loans", "par_extended")
_HOLDING_COLUMNS = ("soma_holdings", "theo_available", "actual_available", "outstanding_loans")
_TYPES = {"Securities Lending": "lending", "Extensions": "extension"}
_FIELDS = {"parAmtSubmitted": "par_submitted", "parAmtAccepted": "par_accepted", "weightedAverageRate": "fee",
           "somaHoldings": "soma_holdings", "theoAvailToBorrow": "theo_available",
           "actualAvailToBorrow": "actual_available", "outstandingLoans": "outstanding_loans",
           "parAmtExtended": "par_extended"}
_SCALE = 10_000


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in LENDING_COLUMNS}).astype(
        {"timestamp": "datetime64[ms]", "loan_maturity": "datetime64[ms]", "operation_id": object,
         "operation_type": object, "cusip": object, "security_class": object, "description": object})


def parse_operations(operations: list[dict]) -> pd.DataFrame:
    """API operations (with ``details``) -> rows. Operations without results are skipped."""
    rows = []
    for op in operations:
        if op.get("auctionStatus", "Results") != "Results":
            continue
        base = {"timestamp": op["operationDate"], "operation_id": op["operationId"],
                "operation_type": _TYPES.get(op.get("operationType"), op.get("operationType")),
                "loan_maturity": op.get("maturityDate")}
        for d in op.get("details", []):
            row = dict(base, cusip=str(d["cusip"]).strip().upper(),  # the API sometimes sends lowercase
                       description=d.get("securityDescription"))
            row.update({dst: d.get(src) for src, dst in _FIELDS.items()})
            rows.append(row)
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df["security_class"] = security_class(df["cusip"])
    df = df.reindex(columns=LENDING_COLUMNS)
    for c in ("timestamp", "loan_maturity"):
        df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ms]")
    for c in AMOUNT_COLUMNS + ("fee",):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    # bids submitted but none accepted come with fee 0: no fee was set, not a free loan
    df.loc[df["operation_type"].eq("lending") & df["par_accepted"].eq(0), "fee"] = float("nan")
    # holdings and availability weren't reported before 2005-11-07: an operation whose
    # every row says 0 for all four means "not reported", not "the Fed held none"
    held = list(_HOLDING_COLUMNS)
    lending = df["operation_type"].eq("lending")
    unreported = df[lending].groupby("operation_id")[held].transform(lambda c: c.fillna(0).eq(0).all()).all(axis=1)
    df.loc[unreported[unreported].index, held] = float("nan")
    return df.sort_values(LENDING_KEYS).reset_index(drop=True)


def security_class(cusips: pd.Series) -> pd.Series:
    """``treasury`` for Treasury CUSIPs (all start 912), else ``agency``."""
    return pd.Series(cusips).astype(str).str.startswith("912").map({True: "treasury", False: "agency"})


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df[LENDING_COLUMNS].copy()
    for c in ("timestamp", "loan_maturity"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
    out["fee"] = (out["fee"].astype("float64") * _SCALE).round().astype("Int32")
    for c in AMOUNT_COLUMNS:
        out[c] = out[c].astype("float64").round().astype("Int64")
    return out


def decode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fee"] = out["fee"].astype("float64") / _SCALE
    for c in AMOUNT_COLUMNS:
        out[c] = out[c].astype("float64")
    for c in ("operation_id", "operation_type", "cusip", "security_class", "description"):
        out[c] = out[c].astype(str)
    return out


def fee_floor(days, schedule: tuple[tuple[str, float], ...]) -> pd.Series:
    """The program's minimum fee (percent) in force on each day, from a schedule of
    ``(effective_from, minimum_fee)`` - sorted, each entry valid until the next."""
    days = pd.to_datetime(pd.Series(days)).reset_index(drop=True)
    starts = pd.to_datetime([s for s, _ in schedule])
    idx = starts.searchsorted(days, side="right") - 1
    return pd.Series([schedule[i][1] if i >= 0 else float("nan") for i in idx], index=days.index, dtype="float64")


def excess_fee(df: pd.DataFrame, schedule: tuple[tuple[str, float], ...]) -> pd.Series:
    """Lending fee above the minimum in force that day, in percent (>= 0 where the fee is
    known; NaN for extensions). The specialness signal: the floor alone carries none."""
    floor = fee_floor(df["timestamp"], schedule).to_numpy()
    return pd.Series(df["fee"].to_numpy() - floor, index=df.index).clip(lower=0.0)
