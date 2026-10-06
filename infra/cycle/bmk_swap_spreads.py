"""Benchmark SWAP-SPREAD P&L, part of the ``bmk_pnl`` step (root CLAUDE.md 12/16; config
``SWAP_SPREAD_*``): per ``US_SWSP_<t>y`` and source, the day's price-action P&L in bp of
LONG THE TREASURY AGAINST SWAPS (long the bond, pay fixed, DV01-matched;
``infra.processing.swap_spreads``), stored in ``Bmk/Pnl`` under bmk ``swsp_cmt`` /
``swsp_otr`` next to the futures and yield rows (``pnl_per_dv01`` bp for every bmk). Reads
the persisted spreads (Derived/SwapSpreads, the ``derived`` step), no API. No carry yet.
Upsert by ``timestamp, ticker, bmk``.
"""
from __future__ import annotations

import pandas as pd

from infra.config import SWAP_SPREAD_AGREE_BP, SWAP_SPREAD_MAX_GAP_DAYS, SWAP_SPREAD_SOURCES
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.processing import swap_spreads as ssp
from infra.storage import parquet_store

_LOOKBACK = pd.Timedelta(days=14)
_ONE_DAY = pd.Timedelta(days=1)
PNL_KEYS = ["timestamp", "ticker", "bmk"]
MAX_DAILY_BP = 25.0


def _pnl_dir(paths: CyclePaths):
    return paths.bmk_root / "Pnl"


def compute_swap_spread_pnl(start, end, *, paths: CyclePaths | None = None,
                            sources=SWAP_SPREAD_SOURCES) -> pd.DataFrame:
    from infra.pipeline.swap_spreads import read_swap_spreads
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    s = read_swap_spreads(start - _LOOKBACK, end + _ONE_DAY, root=paths.swap_spreads_dir)
    parts = []
    if "cmt" in sources:
        parts.append(ssp.level_pnl(s[s["source"] == "cmt"], "swsp_cmt", max_gap_days=SWAP_SPREAD_MAX_GAP_DAYS))
    if "otr" in sources:
        parts.append(ssp.held_bond_pnl(s[s["source"] == "otr"], "swsp_otr", max_gap_days=SWAP_SPREAD_MAX_GAP_DAYS))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=ssp.PNL_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def backfill_daily_swap_spread_pnl(start, end, *, paths: CyclePaths | None = None, **kw) -> dict:
    from infra.pipeline.swap_spreads import read_swap_spreads
    paths = paths or CyclePaths.default()
    df = compute_swap_spread_pnl(start, end, paths=paths, **kw)
    if len(df):
        out = df.copy()
        for col in ("timestamp", "prev_timestamp"):
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ms]")
        parquet_store.write_partitioned(out, _pnl_dir(paths), PNL_KEYS)
    s = read_swap_spreads(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY,
                          root=paths.swap_spreads_dir)
    expected = {f"swsp_{src}": str(pd.Timestamp(g["timestamp"].max()).date()) for src, g in s.groupby("source")}
    last = df.groupby("bmk")["timestamp"].max().dt.date.astype(str).to_dict() if len(df) else {}
    return {"rows": len(df), "by_bmk": df.groupby("bmk").size().to_dict() if len(df) else {},
            "last_day": last, "expected_last": expected}


# ------------------------------------------------------------------------ checks
def _read(ctx: StepContext) -> pd.DataFrame:
    df = parquet_store.read_partitioned(_pnl_dir(ctx.paths), start=ctx.start, end=ctx.end + _ONE_DAY)
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "bmk", "pnl_per_dv01"])
    df["bmk"], df["ticker"] = df["bmk"].astype(str), df["ticker"].astype(str)
    return df[df["bmk"].str.startswith("swsp_")]


def check_present(ctx: StepContext):
    """Each source's P&L reaches the latest day its stored spreads reach (warn)."""
    out = ctx.output.get("swap_spreads", {})
    last, expected = out.get("last_day", {}), out.get("expected_last", {})
    if not expected:
        return True, "no swap spreads in the window", None
    behind = {k: f"{last.get(k, 'none')} < {v}" for k, v in expected.items() if last.get(k, "") < v}
    if not behind:
        return True, "swap-spread P&L reaches its spreads' latest day: " + \
            ", ".join(f"{k} {v}" for k, v in sorted(expected.items())), None
    return False, "behind their spreads: " + "; ".join(f"{k} {v}" for k, v in sorted(behind.items())), None


def check_sane(ctx: StepContext):
    df = _read(ctx)
    bad = df[df["pnl_per_dv01"].abs() > MAX_DAILY_BP]
    return (bad.empty, f"every daily swap-spread move within {MAX_DAILY_BP:.0f}bp" if bad.empty
            else f"{len(bad)} move(s) beyond {MAX_DAILY_BP:.0f}bp",
            None if bad.empty else bad[["timestamp", "ticker", "bmk", "pnl_per_dv01"]])


def cmt_switch_days(start, end, *, paths: CyclePaths) -> set:
    """(day, ``US_SWSP_<t>``) where CMT's on-the-run input changed: CMT moves to a new issue
    at its AUCTION (the auction-convention map), so the CMT series jumps by the new note's
    yield difference that day - a property of CMT (``yield_cmt`` has it too), not a market
    move. The on-the-run series holds the previous day's bond and has no such jump."""
    from infra.pipeline.treasury_otr import read_otr
    a = read_otr(pd.Timestamp(start) - _LOOKBACK, pd.Timestamp(end) + _ONE_DAY, rank=0, convention="auction",
                 root=paths.treasury_otr_dir).sort_values(["tenor", "timestamp"])
    if a.empty:
        return set()
    prev = a.groupby("tenor")["cusip"].shift(1)
    a = a[prev.notna() & prev.ne(a["cusip"])]
    return {(pd.Timestamp(r.timestamp), f"US_SWSP_{r.tenor}") for r in a.itertuples()}


def check_sources_agree(ctx: StepContext):
    """CMT and on-the-run moves within ``SWAP_SPREAD_AGREE_BP`` (warn): a wide gap is a bad
    swap close, a bad price or a roll handled wrong. CMT's auction-switch days are skipped
    (``cmt_switch_days``): over 2024-09..2026-10 they held 16 of the 2y's 23 gaps > 3bp and
    all 7 of the 3y's."""
    df = _read(ctx)
    if df.empty:
        return True, "no swap-spread rows", None
    w = df.pivot_table(index=["timestamp", "ticker"], columns="bmk", values="pnl_per_dv01")
    if not {"swsp_cmt", "swsp_otr"} <= set(w.columns):
        return True, "fewer than two sources in the window", None
    w = w.dropna()
    switch = cmt_switch_days(ctx.start, ctx.end, paths=ctx.paths)
    skip = pd.Series([k in switch for k in w.index], index=w.index, dtype=bool)
    w = w[~skip]
    gap = (w["swsp_cmt"] - w["swsp_otr"]).abs()
    bad = gap[gap > SWAP_SPREAD_AGREE_BP]
    note = f" ({int(skip.sum())} CMT auction-switch ticker-day(s) skipped)"
    if bad.empty:
        return True, f"sources agree within {SWAP_SPREAD_AGREE_BP}bp on {len(w)} ticker-days" + note, None
    return False, f"{len(bad)} ticker-day(s) where sources disagree by > {SWAP_SPREAD_AGREE_BP}bp" + note, \
        w.loc[bad.index].assign(gap=bad).reset_index()
