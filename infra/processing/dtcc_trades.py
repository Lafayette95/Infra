"""DTCC public swap reports (CFTC Part 43) -> clean OIS par trades. Pure functions, no I/O.

How the reports encode a trade's life (verified on the RATES files, 2026-09):
* every record has its OWN ``Dissemination Identifier``; any action after the first
  points back to the trade through ``Original Dissemination Identifier`` - possibly into
  an earlier day's file (a trade executed 09-23 corrected 09-28). Compare them through
  ``trade_key``, never as raw text: the ids can be read as float text ("1174107000.0"),
  and since 2025-11-02 an id is ``base x 10^9 + suffix`` with a DIFFERENT suffix on the
  later records (found 2026-10-06: raw-text linking matched none of them, so until then
  no correction or cancellation ever reached the swap closes);
* ``NEWT`` + event ``TRAD`` is an executed trade. Other ``NEWT`` events (clearing,
  novation, compression, exercise) re-report EXISTING risk - counting them would count a
  trade twice;
* ``CORR`` is a complete replacement record for the trade (same execution time);
  ``EROR`` cancels it; ``MODI`` / ``TERM`` are later lifecycle changes that don't change
  the price agreed at execution, so a price snap ignores them;
* cancellations arrive fast (95% within 17h, 99% within 33 days, September 2026);
  corrections mostly within minutes, though ~17% are fixes to trades executed months or
  years earlier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.config import (
    SWAP_EXCLUDED_PLATFORMS,
    SWAP_OFF_MARKET_BP,
    SWAP_OFF_MARKET_CURVE_BP,
    SWAP_OFF_MARKET_WINDOW_HOURS,
    SWAP_SPOT_TOLERANCE_DAYS,
    SWAP_TENOR_TOLERANCE_DAYS,
    SwapCurveSpec,
)

# Report column -> our name. Only these are read.
RAW_COLUMNS = {
    "Dissemination Identifier": "diss_id",
    "Original Dissemination Identifier": "orig_id",
    "Action type": "action",
    "Event type": "event",
    "Event timestamp": "event_ts",
    "Execution Timestamp": "executed",
    "Effective Date": "effective",
    "Expiration Date": "maturity",
    "Fixed rate-Leg 1": "rate_leg1",
    "Fixed rate-Leg 2": "rate_leg2",
    "Notional amount-Leg 1": "notional_raw",
    "Notional currency-Leg 1": "currency",
    "Cleared": "cleared",
    "Platform identifier": "platform",
    "Block trade election indicator": "block",
    "Package indicator": "package",
    "Non-standardized term indicator": "non_standard",
    "Fixed rate payment frequency period-Leg 1": "fixed_freq_leg1",
    "Fixed rate payment frequency period-Leg 2": "fixed_freq_leg2",
    "Other payment amount": "upfront_raw",
    "UPI FISN": "fisn",
    "UPI Underlier Name": "underlier",
}
TRADE_COLUMNS = ["executed", "currency", "tenor", "rate", "notional", "notional_capped", "cleared", "platform",
                 "block", "trade_id"]


LONG_ID = 10 ** 12  # ids at or above: the post-2025-11-02 format, base x 10^9 + suffix
ID_SUFFIX = 10 ** 9


def trade_key(ids: pd.Series) -> pd.Series:
    """Dissemination ids -> the key every record of one trade shares: ``S<id>`` for the
    old format, ``L<id // 10^9>`` for the new one (the bases overlap the old ids' range,
    hence the namespaces). Numeric text in any form ("1174107000.0") is parsed; a
    non-numeric id is kept as it is; missing stays missing."""
    raw = ids.astype("string")
    x = pd.to_numeric(raw, errors="coerce")
    out = raw.copy()
    long_ = (x >= LONG_ID).fillna(False)
    short = (x < LONG_ID).fillna(False)
    # a long id's base comes from its TEXT: ids reach ~5e18, float64 is exact only to ~9e15
    digits = raw.str.replace(r"\.0+$", "", regex=True)
    out[long_] = "L" + digits[long_].str[:-9]
    out[short] = "S" + x[short].astype("int64").astype(str)
    return out


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None).astype("datetime64[ms]")


def _flag(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Report rows (as read, text) -> typed columns. Rates in PERCENT; times tz-naive UTC."""
    df = raw[[c for c in RAW_COLUMNS if c in raw.columns]].rename(columns=RAW_COLUMNS).copy()
    for c in RAW_COLUMNS.values():
        if c not in df:
            df[c] = pd.NA
    df["diss_id"] = df["diss_id"].astype("string")
    df["orig_id"] = df["orig_id"].astype("string")
    df["event_ts"] = _utc(df["event_ts"])
    df["executed"] = _utc(df["executed"])
    df["effective"] = pd.to_datetime(df["effective"], errors="coerce").astype("datetime64[ms]")
    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce").astype("datetime64[ms]")
    leg1 = pd.to_numeric(df["rate_leg1"], errors="coerce")
    df["rate"] = leg1.fillna(pd.to_numeric(df["rate_leg2"], errors="coerce")) * 100.0
    notional = df["notional_raw"].astype("string").str.replace(",", "", regex=False).str.strip()
    df["notional_capped"] = notional.str.endswith("+").fillna(False).astype(bool)
    df["notional"] = pd.to_numeric(notional.str.rstrip("+"), errors="coerce")
    df["fixed_freq"] = df["fixed_freq_leg1"].fillna(df["fixed_freq_leg2"])
    upfront = pd.to_numeric(df["upfront_raw"].astype("string").str.replace(",", "", regex=False), errors="coerce")
    df["upfront"] = upfront.fillna(0.0) != 0.0
    for c in ("block", "package", "non_standard"):
        df[c] = _flag(df[c])
    return df.drop(columns=["rate_leg1", "rate_leg2", "notional_raw", "fixed_freq_leg1", "fixed_freq_leg2",
                            "upfront_raw"])


def product_rows(df: pd.DataFrame, spec: SwapCurveSpec) -> pd.DataFrame:
    """Every record (any action) of the currency's OIS product."""
    return df[(df["fisn"] == spec.fisn) & df["underlier"].astype(str).str.contains(spec.underlier, regex=False)]


def trade_events(df: pd.DataFrame) -> pd.DataFrame:
    """The records a price snap cares about: executed trades (``NEWT``/``TRAD``) and their
    corrections and cancellations, each keyed by ``trade_id`` (the original trade's
    Dissemination Identifier)."""
    keep = ((df["action"] == "NEWT") & (df["event"] == "TRAD")) | df["action"].isin(["CORR", "EROR"])
    out = df[keep].copy()
    out["trade_id"] = trade_key(out["orig_id"]).fillna(trade_key(out["diss_id"])) if len(out) else out["diss_id"]
    return out


def current_trades(events: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Each executed trade's latest state known by ``as_of`` (an event-time cutoff, None =
    everything): cancelled trades dropped, corrected ones replaced by their correction.
    Corrections of trades whose execution we never saw (executed before the archive, or a
    non-TRAD event) are ignored - there is no executed trade for them to correct."""
    ev = events if as_of is None else events[events["event_ts"] <= pd.Timestamp(as_of)]
    executed = set(ev.loc[ev["action"] == "NEWT", "trade_id"])
    ev = ev[ev["trade_id"].isin(executed)]
    # NEWT first among same-time records, so a same-second correction still wins
    order = ev["action"].map({"NEWT": 0, "CORR": 1, "EROR": 2})
    last = ev.assign(_o=order).sort_values(["event_ts", "_o"], kind="stable").groupby("trade_id").tail(1)
    return last[last["action"] != "EROR"].drop(columns="_o")


def tenor_years(effective: pd.Series, maturity: pd.Series, tenors, tolerance_days: int) -> pd.Series:
    """Whole-year tenor N if ``maturity`` is within ``tolerance_days`` of effective + N
    years, else NaN."""
    out = pd.Series(np.nan, index=effective.index)
    for n in tenors:
        target = effective + pd.DateOffset(years=n)
        out[(maturity - target).abs() <= pd.Timedelta(days=tolerance_days)] = n
    return out


def par_trades(trades: pd.DataFrame, spec: SwapCurveSpec, currency: str, *, allow_packages: bool = False) -> pd.DataFrame:
    """Plain spot-starting par swaps on a standard tenor: not a package leg (unless
    ``allow_packages``: USD CPI swap package legs trade at the market - median 0.0 to
    -0.05bp from plain trades within 30 min, 2026-03..09 - and more than double the
    5-30y prints), not
    non-standard, no upfront fee, not declared uncleared, not on an excluded platform
    (``SWAP_EXCLUDED_PLATFORMS``), annual fixed leg, a rate, and not off-market
    (``drop_off_market``). Columns TRADE_COLUMNS.

    Spot is judged from the UTC trade date, which is the local one for every snap window
    (all inside US/European business hours)."""
    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    trade_day = trades["executed"].dt.normalize()
    spot = trade_day + pd.offsets.BDay(spec.spot_lag_days) if spec.spot_lag_days else trade_day
    eff = trades["effective"]
    ok = (eff >= spot) & (eff <= spot + pd.Timedelta(days=SWAP_SPOT_TOLERANCE_DAYS))
    ok &= ~trades["non_standard"] & ~trades["upfront"] & (trades["cleared"].astype(str) != "N")
    if not allow_packages:
        ok &= ~trades["package"]
    ok &= ~trades["platform"].isin(SWAP_EXCLUDED_PLATFORMS)
    ok &= trades["fixed_freq"].isin(spec.fixed_frequencies) & trades["rate"].notna()
    out = trades[ok].copy()
    out["tenor"] = tenor_years(out["effective"], out["maturity"], spec.tenors, SWAP_TENOR_TOLERANCE_DAYS)
    out = out[out["tenor"].notna()]
    out["tenor"] = out["tenor"].astype(int)
    out["currency"] = currency
    return drop_off_market(out[TRADE_COLUMNS].sort_values("executed").reset_index(drop=True))


def drop_off_market(trades: pd.DataFrame) -> pd.DataFrame:
    """Drop prints far from where their tenor traded (config ``SWAP_OFF_MARKET_*``): an
    off-market coupon is a real trade but not a par rate. Reference per print, in order:
    the median of OTHER same-tenor prints within +-SWAP_OFF_MARKET_WINDOW_HOURS (at least
    3), else of the day's other same-tenor prints (at least 3), else the neighbouring
    tenors' day medians interpolated (limit SWAP_OFF_MARKET_CURVE_BP). A print with no
    reference at all is kept - nothing to judge it by."""
    if trades.empty:
        return trades
    t = trades.reset_index(drop=True)
    ts = t["executed"].to_numpy()
    rate, tenor = t["rate"].to_numpy(dtype=float), t["tenor"].to_numpy()
    window = np.timedelta64(int(SWAP_OFF_MARKET_WINDOW_HOURS * 3600), "s")
    day_median = t.groupby("tenor")["rate"].median()
    keep = np.ones(len(t), dtype=bool)
    for i in range(len(t)):
        same = (tenor == tenor[i])
        same[i] = False
        near = same & (np.abs(ts - ts[i]) <= window)
        if near.sum() >= 3:
            ref, limit = np.median(rate[near]), SWAP_OFF_MARKET_BP
        elif same.sum() >= 3:
            ref, limit = np.median(rate[same]), SWAP_OFF_MARKET_BP
        else:
            others = day_median.drop(index=tenor[i], errors="ignore")
            if others.empty:
                continue
            ref = float(np.interp(tenor[i], others.index.to_numpy(dtype=float), others.to_numpy()))
            limit = SWAP_OFF_MARKET_CURVE_BP
        keep[i] = abs(rate[i] - ref) * 100.0 <= limit
    return t[keep].reset_index(drop=True)
