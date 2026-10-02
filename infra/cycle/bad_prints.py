"""The bad-print rule and its treatment, shared by every px source (futures settlements,
cash-bond yields): instrument-agnostic - callers supply values, each instrument's curve
and position on it, and a per-instrument policy. CLAUDE.md section 12.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from infra.config import FOMC_MEETINGS, BadPrintRules

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)

# Custom outlier check (test c): a BAD PRINT, not a big market day. Each move is first put
# in units of its own contract's typical move (z = move / median |move| over the prior
# OUTLIER_LOOKBACK moves), then compared with its peers' z the same day. A contract is
# flagged when it (1) moved a lot by its own standard (|z| >= OUTLIER_Z_MIN), (2) was out
# of line with its peers (|z - median peer z| >= OUTLIER_DEV_MIN), and (3) that deviation
# REVERSED at its next session by at least OUTLIER_REVERSAL of its size. Calibrated
# 2026-09-28 on a year of real settlements (2025-07 .. 2026-09, 19,324 contract-days): it
# flags exactly three prints - ESRM7 and ESRU7 on 2026-09-11 (a ~40bp kink mid-strip,
# gone next session) and ESRZ6 on 2025-10-31 (-6bp while every neighbour rose, +5.75bp
# back) - and none of NFP 2025-08-01, the ECB-dated EUR days, 2026-04-08, or FOMC
# 2026-06-17/07-29. Each condition is load-bearing: without (1) a nearly-expired contract
# that barely moved on a shock day looks "out of line"; without (3) an FOMC meeting-month
# contract repricing alone looks like a bad print. Cost of (3): a print can only be judged
# once its next session exists - the latest day is reported as pending (warn), not judged.
OUTLIER_Z_MIN = 5.0
OUTLIER_DEV_MIN = 6.0
OUTLIER_REVERSAL = 0.5
OUTLIER_PEERS = 4  # nearest instruments on the same curve (futures: by expiry; bonds: by tenor)
OUTLIER_LOOKBACK = 60
OUTLIER_MIN_HISTORY = 20


def fomc_decision_sessions(meetings=FOMC_MEETINGS) -> pd.DatetimeIndex:
    """The first futures session whose settlement follows each FOMC decision: the
    announcement day, or the next weekday when it fell on a weekend (Sunday 2020-03-15)."""
    days = pd.to_datetime([m.end_date for m in meetings])
    return pd.DatetimeIndex([d + pd.offsets.BDay(0) if d.weekday() >= 5 else d for d in days])


# Event-date sources for BadPrintRules.exempt_events: name -> () -> sessions
EXEMPTION_EVENTS = {"fomc": fomc_decision_sessions}


def exempt_sessions(rules: BadPrintRules | None) -> set[tuple[pd.Timestamp, str]]:
    """``{(session, root)}`` on which ``rules`` exempt a root's candidates."""
    if rules is None:
        return set()
    return {(pd.Timestamp(d), root) for name, roots in rules.exempt_events
            for d in EXEMPTION_EVENTS[name]() for root in roots}


def market_factor(day: pd.DataFrame, rules: BadPrintRules) -> pd.Series:
    """Per row of one day's frame (``z``, ``root``, ``market``): the median |z| of the
    OTHER curves in the same market, floored at ``rules.market_scaling_floor``; the floor
    alone when fewer than ``rules.market_scaling_min_instruments`` such instruments moved."""
    out = pd.Series(rules.market_scaling_floor, index=day.index, dtype="float64")
    for idx, r in day.iterrows():
        other = day[(day["market"] == r["market"]) & (day["root"] != r["root"])]["z"].abs()
        if len(other) >= rules.market_scaling_min_instruments:
            out[idx] = max(rules.market_scaling_floor, float(other.median()))
    return out


def last_weekday(day: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(day).normalize()
    while day.weekday() >= 5:
        day -= _ONE_DAY
    return day


def peer_outliers_frame(
    values: pd.DataFrame, meta: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
    *, group_label: str = "group", min_abs_dev: float | None = None, rules: BadPrintRules | None = None,
    exempt: set[tuple[pd.Timestamp, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The instrument-agnostic bad-print rule (see the OUTLIER_* constants). ``values``:
    ``ticker, timestamp, value`` (with history before ``start``, for each instrument's own
    scale); ``meta`` indexed by ticker: ``root`` (its curve), ``order`` (position on that
    curve - expiry for futures, tenor for bonds) and ``group`` (the fallback peer pool).
    Peers: the OUTLIER_PEERS nearest same-root instruments by ``order``; a root with too
    few instruments that day falls back to its whole ``group``; fewer than 2 peers -> not
    judged. ``min_abs_dev`` (optional, in ``value`` units) additionally requires the move
    to be out of line with the same peers' median move IN RAW UNITS - for sources quoted so
    coarsely that each instrument's own scale is quantised (see the cash-bond caller).
    ``rules`` (optional, ``infra.config.BadPrintRules``): with ``market_scaling`` and a
    ``market`` column in ``meta``, both thresholds are multiplied by the day's market
    factor (``market_factor``); ``exempt``: ``{(day, root)}`` whose candidates are logged
    and dropped, never flagged. Returns ``(flagged, pending)`` for days in ``[start, end]``,
    each with ``market_k`` (1.0 when not scaled)."""
    df = values[values["ticker"].isin(meta.index)].sort_values(["ticker", "timestamp"]).copy()
    df["move"] = df.groupby("ticker")["value"].diff()
    df["scale"] = df.groupby("ticker")["move"].transform(
        lambda m: m.abs().shift(1).rolling(OUTLIER_LOOKBACK, min_periods=OUTLIER_MIN_HISTORY).median())
    df["z"] = df["move"] / df["scale"]
    df = df[np.isfinite(df["z"]) & (df["timestamp"] >= start)]
    cols = ["ticker", "timestamp", "move", "z", "dev", "peers", "market_k"]
    if df.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    df["root"] = df["ticker"].map(meta["root"])
    df["expiry"] = df["ticker"].map(meta["order"])
    df["group"] = df["ticker"].map(meta["group"])
    scale = rules is not None and rules.market_scaling and "market" in meta.columns
    if scale:
        df["market"] = df["ticker"].map(meta["market"])
        df["market_k"] = pd.concat([market_factor(day, rules) for _, day in df.groupby("timestamp")])
    else:
        df["market_k"] = 1.0

    devs, pools, raw_devs = [], [], []
    for _, day in df.groupby("timestamp"):
        for idx, r in day.iterrows():
            peers = day[(day["root"] == r["root"]) & (day.index != idx)]
            if len(peers) >= 2:
                peers = peers.iloc[(peers["expiry"] - r["expiry"]).abs().argsort()[:OUTLIER_PEERS]]
                pool = "root"
            else:
                peers = day[(day["group"] == r["group"]) & (day.index != idx)]
                pool = group_label
            devs.append((idx, r["z"] - peers["z"].median() if len(peers) >= 2 else np.nan))
            raw_devs.append((idx, r["move"] - peers["move"].median() if len(peers) >= 2 else np.nan))
            pools.append((idx, pool))
    df["dev"] = pd.Series(dict(devs))
    df["peers"] = pd.Series(dict(pools))
    df["raw_dev"] = pd.Series(dict(raw_devs))
    df = df.dropna(subset=["dev"])
    df["next_dev"] = df.groupby("ticker")["dev"].shift(-1)
    k = df["market_k"]
    candidate = (df["z"].abs() >= OUTLIER_Z_MIN * k) & (df["dev"].abs() >= OUTLIER_DEV_MIN * k) & (df["timestamp"] <= end)
    if exempt:
        on_event = pd.Series([(t, r) in exempt for t, r in zip(df["timestamp"], df["root"])], index=df.index)
        for r in df[candidate & on_event].itertuples():
            log.info("bad-print candidate %s %s exempt (policy-decision session): z %.1f, dev %.1f",
                     r.ticker, pd.Timestamp(r.timestamp).date(), r.z, r.dev)
        candidate &= ~on_event
    if min_abs_dev is not None:
        candidate &= df["raw_dev"].abs() >= min_abs_dev
    reverted = (np.sign(df["next_dev"]) == -np.sign(df["dev"])) & (df["next_dev"].abs() >= OUTLIER_REVERSAL * df["dev"].abs())
    return (df[candidate & reverted][cols + ["next_dev"]].reset_index(drop=True),
            df[candidate & df["next_dev"].isna()][cols].reset_index(drop=True))


# The reversal test needs each judged day's NEXT session, which for the window's last day
# lies after the window. Read it if it's already on disk - only days inside the window are
# ever JUDGED, so this is a data-quality lookup, not look-ahead in any value computed.
# Without it, non-overlapping history windows (e.g. monthly) never judged a boundary day:
# ESRZ6's bad print on 2025-10-31 slipped through an Oct-2025 window this way.
_NEXT_SESSION_LOOKAHEAD = pd.Timedelta(days=10)
BAD_PRINT_SOURCE = "px_bad_print"  # this process's name in the adjustments log


def treatment_rows(
    flagged: pd.DataFrame, hist: pd.DataFrame, *, policy, store: str, column: str, run_day: pd.Timestamp,
) -> list[dict]:
    """Adjustment-log rows for confirmed bad prints, any instrument. ``hist``: ``ticker,
    timestamp, value`` (raw); ``policy(r) -> (action, detail prefix)`` with action "NA"
    (dropped) or "roll" (last GOOD earlier value - consecutive bad prints roll from the
    last good one; no earlier value falls back to NA)."""
    rows, done = [], {}  # done: (ticker, day) -> adjusted value, for consecutive bad prints
    for r in flagged.sort_values(["ticker", "timestamp"]).itertuples(index=False):
        action, prefix = policy(r)
        series = hist.loc[hist["ticker"] == r.ticker].set_index("timestamp")["value"]
        original, adjusted, note = float(series[r.timestamp]), np.nan, ""
        if action == "roll":
            for day in reversed(series.index[series.index < r.timestamp]):
                value = done.get((r.ticker, day), series[day])
                if np.isfinite(value):
                    adjusted = float(value)
                    break
            else:
                action, note = "NA", " (roll had no earlier good value - NA instead)"
        done[(r.ticker, r.timestamp)] = adjusted
        rows.append({
            "store": store, "timestamp": r.timestamp, "key": r.ticker, "column": column,
            "action": action, "original": original, "adjusted": adjusted, "source": BAD_PRINT_SOURCE,
            "detail": (f"{prefix}off-peer move (z {r.z:.1f}, peer deviation {r.dev:.1f}"
                       + (f", market x{r.market_k:.1f}" if getattr(r, "market_k", 1.0) > 1.0 else "")
                       + f") reversed next session ({r.next_dev:.1f}){note}"),
            "run_day": run_day,
        })
    return rows
