"""OIS discount curves from the stored swap closes (root CLAUDE.md 16): bootstrap, store,
read. Disk only - the closes come from ``Derived/SwapCloses``, the short end from the
SR1-fitted overnight path (infra.pipeline.financing.sofr_path).

Store ``Derived/OisCurves``: one row per NODE - keys ``timestamp`` (the close's snap
instant, UTC: the curve is known at that instant), ``curve`` (an ``OIS_CURVES`` name),
``node`` (``1M``.. short end, ``1Y``.. swap pillars) - with ``t_years`` (ACT/365.25 from the
trade day), ``df``, ``zero_pct``, ``source`` (``short_end`` / ``swap`` / ``filled`` - a missing
interior tenor, ``fill_missing_tenors``; ``n_trades`` 0), ``input_rate`` and the input close's
``n_trades`` / ``se_bp``. A recomputed day replaces its rows. The nodes ARE the curve
(log-linear discount factors between them): ``ois_curve`` rebuilds it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import swap_curve as sc
from infra.config import (
    DAILY_BOE_OIS_DIR,
    DAILY_FUTURES_DIR,
    FUTURES_CONTRACTS_FILE,
    OIS_CURVES,
    OIS_CURVES_DIR,
    REPO_DIR,
    SWAP_CLOSES,
    SWAP_CLOSES_DIR,
    SWAP_CURVES,
    OisCurveSpec,
)
from infra.pipeline.swap_closes import read_swap_closes
from infra.storage import parquet_store
from infra.trading_calendar import snap_instants

# closes read before the window, so a fill's residual history - and so every recomputed
# window - is the same as in a full build (must cover ``fill_max_age_days``)
FILL_LOOKBACK_DAYS = 31

log = logging.getLogger(__name__)
KEYS = ["timestamp", "curve", "node"]
COLUMNS = KEYS + ["t_years", "df", "zero_pct", "source", "input_rate", "n_trades", "se_bp"]
_ONE_DAY = pd.Timedelta(days=1)


ESR_ROOT = "ESR"


def _third_wednesday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(2 - first.dayofweek) % 7 + 14)


def esr_periods(day, *, futures_root: Path = DAILY_FUTURES_DIR, contracts_file: Path = FUTURES_CONTRACTS_FILE,
                lookback_days: int = 10) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """The ESR strip as ``(quarter start, quarter end, rate %)``, from the latest settlement
    day STRICTLY BEFORE ``day``. A contract's reference quarter runs IMM to IMM: it ends on
    the 3rd Wednesday after its last trading day (a Tuesday - checked on every stored
    contract) and starts on the 3rd Wednesday three months earlier."""
    from infra.pipeline.daily import read_daily_from_disk
    from infra.storage import contract_store
    d = pd.Timestamp(day).normalize()
    k = contract_store.read_contracts(contracts_file, ESR_ROOT)
    k = k[pd.to_datetime(k["expiry"]) >= d - pd.Timedelta(days=lookback_days)]
    if k.empty:
        return []
    px = read_daily_from_disk(list(k["ticker"].astype(str)), d - pd.Timedelta(days=lookback_days), d, root=futures_root)
    px = px.dropna(subset=["settlement_price"])
    px = px[(100.0 - px["settlement_price"].astype(float)).between(-1.0, 25.0)]
    if px.empty:
        return []
    px = px[px["timestamp"] == px["timestamp"].max()]
    expiry = dict(zip(k["ticker"].astype(str), pd.to_datetime(k["expiry"])))
    out = []
    for t, p in zip(px["ticker"].astype(str), px["settlement_price"].astype(float)):
        end = expiry[t] + pd.Timedelta(days=1)
        s = end - pd.DateOffset(months=3)
        out.append((_third_wednesday(s.year, s.month), end, 100.0 - p))
    return sorted(out, key=lambda x: x[1])


def pillar_schedule(day, tenor: int, currency: str) -> sc.SwapSchedule:
    """A ``tenor``-year par OIS's fixed-leg schedule in ``currency``'s conventions."""
    conv = SWAP_CURVES[currency]
    return sc.swap_schedule(day, tenor, conv.spot_lag_days, freq_months=sc.fixed_freq(tenor, conv.fixed_freq_months),
                            basis=conv.day_basis, calendar=conv.calendar)


def _short_nodes(spec: OisCurveSpec, day, *, futures_root, contracts_file, repo_root, boe_root=DAILY_BOE_OIS_DIR):
    if spec.short_end == "none":
        return []
    if spec.short_end == "esr_futures":
        return sc.strip_nodes(day, esr_periods(day, futures_root=futures_root, contracts_file=contracts_file),
                              spec.short_end_months)
    if spec.short_end == "boe_ois":
        from infra.pipeline.boe_ois import boe_ois_nodes
        return boe_ois_nodes(day, max_years=spec.short_end_months / 12.0, root=boe_root)
    if spec.short_end == "sofr_path":
        from infra.pipeline.financing import sofr_path
        path = sofr_path(day, futures_root=futures_root, contracts_file=contracts_file, repo_root=repo_root)
        return sc.short_end_nodes(path.compounded, day, spec.short_end_months)
    raise ValueError(f"unknown short_end {spec.short_end!r}")


def compute_ois_curves(start, end, *, curves: tuple[str, ...] | None = None, swap_closes_root: Path = SWAP_CLOSES_DIR,
                       futures_root: Path = DAILY_FUTURES_DIR, contracts_file: Path = FUTURES_CONTRACTS_FILE,
                       repo_root: Path = REPO_DIR, boe_root: Path = DAILY_BOE_OIS_DIR) -> tuple[pd.DataFrame, dict]:
    """Every configured curve on every day in ``[start, end]`` with swap closes. No writes.
    Returns the node rows and ``{"days": input days, "empty_days": {day: reason}}``."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    frames, days, empty = [], set(), {}
    for name in curves or tuple(OIS_CURVES):
        spec = OIS_CURVES[name]
        assert spec.fill_max_age_days < FILL_LOOKBACK_DAYS
        closes = read_swap_closes(start - pd.Timedelta(days=FILL_LOOKBACK_DAYS), end + _ONE_DAY, close=spec.close,
                                  currency=spec.currency, method=spec.method, root=swap_closes_root)
        if closes.empty:
            continue
        closes["day"] = pd.to_datetime(closes["timestamp"]).dt.normalize()
        rates = closes.pivot(index="day", columns="tenor", values="rate").sort_index()
        filled = pd.DataFrame(False, index=rates.index, columns=rates.columns)
        if spec.fill_missing:
            rates, filled = sc.fill_missing_tenors(rates, max_age_days=spec.fill_max_age_days, fill_ends=spec.fill_ends)
        for day, g in closes[closes["day"] >= start].groupby("day"):
            days.add(day)
            if len(g) < spec.min_tenors:
                empty[day] = f"{name}: {len(g)} tenor(s) with a close, fewer than {spec.min_tenors}"
                continue
            q = rates.loc[day].dropna()
            try:
                short = _short_nodes(spec, day, futures_root=futures_root, contracts_file=contracts_file,
                                     repo_root=repo_root, boe_root=boe_root)
                conv = SWAP_CURVES[spec.currency]
                _, nodes = sc.bootstrap_ois(day, {int(t): float(r) for t, r in q.items()},
                                            spot_lag=conv.spot_lag_days, freq_months=conv.fixed_freq_months,
                                            basis=conv.day_basis, calendar=conv.calendar, short_nodes=short)
            except Exception as e:  # noqa: BLE001 - recorded with the reason, judged by the presence check
                empty[day] = f"{name}: {type(e).__name__}: {e}"
                continue
            inputs = g.assign(node=g["tenor"].astype(int).astype(str) + "Y").set_index("node")
            nodes["n_trades"] = nodes["node"].map(inputs["n_trades"]).fillna(0).astype("Int32")
            nodes["se_bp"] = nodes["node"].map(inputs["se_bp"]).astype(float)
            was_filled = {f"{int(t)}Y" for t, f in filled.loc[day].items() if f}
            nodes.loc[nodes["node"].isin(was_filled), "source"] = "filled"
            nodes["timestamp"] = snap_instants([day], SWAP_CLOSES[spec.close].local_time, SWAP_CLOSES[spec.close].timezone)[0]
            nodes["curve"] = name
            frames.append(nodes)
    df = pd.concat(frames, ignore_index=True)[COLUMNS] if frames else pd.DataFrame(columns=COLUMNS)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
    return df, {"days": sorted(days), "empty_days": empty}


def store_ois_curves(df: pd.DataFrame, start, end, *, root: Path = OIS_CURVES_DIR,
                     curves: tuple[str, ...] | None = None) -> int:
    """Replace every row with ``timestamp`` in the days ``[start, end]`` by ``df`` - of the
    ``curves`` named only, if given (a hand build of some curves leaves the others)."""
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    parquet_store.delete_where(root, lambda part: pd.to_datetime(part["timestamp"]).ge(lo)
                               & pd.to_datetime(part["timestamp"]).lt(hi)
                               & (True if curves is None else part["curve"].astype(str).isin(curves)))
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, KEYS)
    return len(df)


def build_ois_curves(start, end, **kw) -> dict:
    """Compute and store ``[start, end]`` (a hand build; the daily cycle's ``derived`` step
    runs the same compute)."""
    root = kw.pop("root", OIS_CURVES_DIR)
    df, diag = compute_ois_curves(start, end, **kw)
    n = store_ois_curves(df, start, end, root=root, curves=kw.get("curves"))
    log.info("ois curves %s..%s: %d day(s), %d node row(s), %d empty", pd.Timestamp(start).date(),
             pd.Timestamp(end).date(), len(diag["days"]), n, len(diag["empty_days"]))
    return {"rows": n, **diag}


def read_ois_curves(start=None, end=None, *, curve: str | None = None, root: Path = OIS_CURVES_DIR) -> pd.DataFrame:
    """Stored node rows with ``timestamp`` in ``[start, end)``."""
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end),
                                        equals_in={"curve": [curve]} if curve else None)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["curve"] = df["curve"].astype(str)
    df["node"] = df["node"].astype(str)
    return df[COLUMNS].sort_values(["timestamp", "curve", "t_years"]).reset_index(drop=True)


def ois_curve(as_of, curve: str = "USD_SOFR", *, root: Path = OIS_CURVES_DIR,
              lookback_days: int = 10) -> tuple[pd.Timestamp, sc.OisCurve] | None:
    """The latest stored curve KNOWN by ``as_of`` (its snap instant <= ``as_of``; a bare day
    means the end of that day), with its instant. None if nothing in ``lookback_days``."""
    as_of = pd.Timestamp(as_of)
    cutoff = as_of + _ONE_DAY if as_of == as_of.normalize() else as_of
    df = read_ois_curves(cutoff - pd.Timedelta(days=lookback_days), cutoff, curve=curve, root=root)
    df = df[pd.to_datetime(df["timestamp"]) <= cutoff]
    if df.empty:
        return None
    ts = df["timestamp"].max()
    return pd.Timestamp(ts), sc.curve_from_nodes(df[df["timestamp"] == ts])


def par_curve(curve: sc.OisCurve, trade_day, tenors=range(1, 31), *, spot_lag: int | None = None,
              currency: str = "USD") -> pd.Series:
    """Par OIS rates (%) at whole-year tenors on a curve, in ``currency``'s conventions
    (``SWAP_CURVES``; ``spot_lag`` overrides its spot lag)."""
    if spot_lag is not None:
        conv = SWAP_CURVES[currency]
        sched = lambda t: sc.swap_schedule(trade_day, t, spot_lag, freq_months=sc.fixed_freq(t, conv.fixed_freq_months),
                                           basis=conv.day_basis, calendar=conv.calendar)
    else:
        sched = lambda t: pillar_schedule(trade_day, t, currency)
    return pd.Series({t: sc.par_rate(curve, trade_day, sched(t)) for t in tenors}, name="par_rate")
