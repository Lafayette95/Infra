"""Run a basis model over a long range in PARALLEL, with checkpoints - for history runs
and benches (infra/models/basis/CLAUDE.md). Reads disk only, never fetches.

The range is cut into calendar-month CHUNKS; each runs in its own process
(``validate.run`` on that month's days) and writes ``<out_dir>/<YYYY-MM>_contracts.parquet``
and ``_bonds.parquet`` when done. A chunk whose files exist is skipped, so an interrupted
run (a Spot instance reclaimed, a laptop asleep) resumes where it stopped. ``every`` keeps
every Nth business day of the WHOLE range (the benches' sample), so a sample is the same
whatever the chunking.

Memory: ~0.3-0.5 GB per worker - on an 8 GB laptop already deep in swap, 2 workers is the
safe ceiling (found 2026-10-03: two concurrent 0.4 GB benches pushed it into constant
swapping, ~12x slower); a 32 GB machine takes ~12-16.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import multiprocessing as mp
import pandas as pd

from infra.analytics.sofr_curve import business_days
from infra.models.basis.config import BASIS_MODELS, BasisSpec


def sample_days(start, end, every: int = 1) -> pd.DatetimeIndex:
    """Every ``every``-th business day of ``[start, end]``."""
    return business_days(pd.Timestamp(start), pd.Timestamp(end))[::max(int(every), 1)]


def chunks(days: pd.DatetimeIndex, period: str = "M") -> list[tuple[str, pd.DatetimeIndex]]:
    """``(label, days)`` per calendar ``period`` ("M" month -> "YYYY-MM", "Y" year -> "YYYY").
    Each chunk loads its inputs once, which costs about as much as running a few days, so
    a SPARSE sample (``every`` >= 5) wants year chunks: found 2026-10-03, every 5th day in
    month chunks spent ~75s a day on loading (93 chunks, ~4 hours) for ~4 days each."""
    s = pd.Series(days, index=days)
    return [(str(p), pd.DatetimeIndex(g.values)) for p, g in s.groupby(s.index.to_period(period))]


def resolve_spec(name: str, overrides: dict | None = None) -> BasisSpec:
    """A registered spec, with field overrides (e.g. ``{"level_betas": True}``)."""
    spec = BASIS_MODELS[name]
    return replace(spec, **(overrides or {})) if overrides else spec


def _run_chunk(spec: BasisSpec, days: list, out_dir: str, label: str) -> tuple[str, int]:
    from infra.models.basis import validate
    days = pd.DatetimeIndex(days)
    res = validate.run(spec, days.min(), days.max(), days=days)
    out = Path(out_dir)
    # bonds first, contracts last: the contracts file is the chunk's "done" marker
    res["bonds"].to_parquet(out / f"{label}_bonds.parquet")
    res["contracts"].to_parquet(out / f"{label}_contracts.parquet")
    return label, len(res["contracts"])


def done(out_dir: Path, label: str) -> bool:
    return (Path(out_dir) / f"{label}_contracts.parquet").exists()


def run_parallel(spec: BasisSpec, start, end, out_dir, *, every: int = 1, workers: int = 2,
                 log=print) -> dict[str, pd.DataFrame]:
    """Run ``spec`` over ``[start, end]`` (every ``every``-th business day) in ``workers``
    processes, one month per task (one YEAR when ``every`` >= 5), skipping chunks already
    in ``out_dir``; returns the combined ``{"contracts", "bonds"}``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    period = "Y" if every >= 5 else "M"
    todo = [(lab, d) for lab, d in chunks(sample_days(start, end, every), period) if not done(out, lab)]
    log(f"{spec.name}: {len(todo)} chunk(s) to run ({period}), {workers} worker(s), out {out}")
    if todo:
        ctx = mp.get_context("spawn")  # a clean interpreter per worker (no forked pandas/numpy state)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futs = [pool.submit(_run_chunk, spec, list(d), str(out), lab) for lab, d in todo]
            for f in as_completed(futs):
                lab, n = f.result()
                log(f"  {lab}: {n} contract-days")
    return collect(out)


def collect(out_dir) -> dict[str, pd.DataFrame]:
    """Every finished chunk in ``out_dir``, combined."""
    out = Path(out_dir)
    labs = sorted(p.name[: -len("_contracts.parquet")] for p in out.glob("*_contracts.parquet"))
    if not labs:
        return {"contracts": pd.DataFrame(), "bonds": pd.DataFrame()}
    return {"contracts": pd.concat([pd.read_parquet(out / f"{l}_contracts.parquet") for l in labs], ignore_index=True),
            "bonds": pd.concat([pd.read_parquet(out / f"{l}_bonds.parquet") for l in labs], ignore_index=True)}
