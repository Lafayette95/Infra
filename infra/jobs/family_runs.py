"""Operating an event FAMILY as runs: one ordinary single-code run per code (``<name>/c000`` ...,
spec ``family:<family>#<i>``, on the root CLAUDE.md 3b machinery unchanged) plus the family's
point-in-time false-discovery table (``<name>/fdr.parquet``, recomputed from the codes' stored
params after every job - so it is always consistent with them). The panel, the occurrences
and the condition timeline are read ONCE per job and shared by every code."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from infra.config import MODEL_RUNS_DIR
from infra.jobs import model_runs as jobs
from infra.models import runs
from infra.models.event_study.config import EVENT_FAMILIES, family_member_name
from infra.models.event_study.family import fdr_table
from infra.models.runs import RunConfig
from infra.storage import model_runs as store

log = logging.getLogger(__name__)
FAMILY_META = "family.json"


def code_name(name: str, i: int) -> str:
    return f"{name}/c{i:03d}"


def create(name: str, family: str, *, start, history_start, refit="W-FRI", root: Path = MODEL_RUNS_DIR,
           force: bool = False) -> list[RunConfig]:
    fam = EVENT_FAMILIES[family]
    configs = [RunConfig(name=code_name(name, i), kind="event_study", spec=family_member_name(family, i),
                         series=tuple(fam.study.instruments), start=str(pd.Timestamp(start).date()),
                         history_start=str(pd.Timestamp(history_start).date()), refit=refit)
               for i in range(len(fam.codes()))]
    folder = root / name
    if (folder / FAMILY_META).exists() and not force:
        raise FileExistsError(f"family run {name!r} exists")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / FAMILY_META).write_text(json.dumps({"family": family, "codes": fam.codes(), "start": str(start),
                                                  "history_start": str(history_start), "refit": refit}, indent=1))
    for c in configs:
        jobs.create(c, root=root, force=force)
    return configs


def load(name: str, root: Path = MODEL_RUNS_DIR) -> tuple[dict, list[RunConfig]]:
    meta = json.loads((root / name / FAMILY_META).read_text())
    configs = [jobs.load(code_name(name, i), root)[0] for i in range(len(meta["codes"]))]
    return meta, configs


def shared_inputs(configs: list[RunConfig], through) -> tuple[pd.DataFrame, dict]:
    """The panel (one read for every code), and prepare kwargs shared per event set /
    condition (occurrences and condition timeline read once)."""
    first = configs[0].make_model()
    panel = first.read_panel(configs[0].history_start, through)
    return panel, {}


def _prepare(config: RunConfig, panel, cache: dict):
    model = config.make_model()
    key = tuple(e for e, _ in model.code.dt_refs)
    if key not in cache:
        cache[key] = model.occurrences(panel.index.max())
    kw = {"occurrences": cache[key]}
    if model.spec.condition is not None:
        ckey = ("cond", model.spec.condition.series, model.spec.condition.history)
        if ckey not in cache:
            cache[ckey] = model.condition_timeline(panel.index.min(), panel.index.max() + pd.Timedelta(days=60))
        kw["condition_timeline"] = cache[ckey]
    return model.prepare(panel, **kw), kw


def write_fdr(name: str, family: str, configs: list[RunConfig], root: Path = MODEL_RUNS_DIR) -> pd.DataFrame:
    params = {i: store.read_params(c.name, root=root) for i, c in enumerate(configs)}
    t = fdr_table(params, EVENT_FAMILIES[family].fdr_q)
    t.to_parquet(root / name / "fdr.parquet", index=False, compression="zstd", compression_level=5)
    return t


def read_fdr(name: str, root: Path = MODEL_RUNS_DIR) -> pd.DataFrame:
    f = root / name / "fdr.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["fit_as_of", "code", "row", "q", "passed_fdr"])


def run_daily(name: str, through, *, root: Path = MODEL_RUNS_DIR, panel: pd.DataFrame | None = None) -> dict:
    meta, configs = load(name, root)
    panel = shared_inputs(configs, through)[0] if panel is None else panel
    cache, out = {}, {}
    for i, c in enumerate(configs):
        prepared, _ = _prepare(c, panel, cache)
        out[i] = jobs.run_daily(c, prepared, through, root=root)
    fdr = write_fdr(name, meta["family"], configs, root)
    return {"codes": len(configs), "fitted": sorted({d for r in out.values() for d in r["fit"]["fitted"]}),
            "fdr_rows": len(fdr)}


def rebuild(name: str, through, *, root: Path = MODEL_RUNS_DIR, promote: bool = False,
            panel: pd.DataFrame | None = None) -> dict:
    meta, configs = load(name, root)
    panel = shared_inputs(configs, through)[0] if panel is None else panel
    cache, reports = {}, {}
    for i, c in enumerate(configs):
        _, kw = _prepare(c, panel, cache)
        reports[i] = jobs.rebuild(c, panel, through, root=root, promote=promote, prepare_kwargs=kw)
    if promote:
        write_fdr(name, meta["family"], configs, root)
    ident = [r.get("identical") for r in reports.values()]
    return {"codes": len(configs), "identical": None if all(x is None for x in ident) else all(x is True for x in ident
                                                                                              if x is not None),
            "reports": reports}


def upcoming(name: str, through, horizon_days: int, *, root: Path = MODEL_RUNS_DIR,
             panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every code's scheduled events after ``through`` within the horizon, with the latest fit
    (``runs.predict_upcoming``) and the family's latest FDR verdict; ``code`` column."""
    meta, configs = load(name, root)
    panel = shared_inputs(configs, through)[0] if panel is None else panel
    cache, parts = {}, []
    for i, c in enumerate(configs):
        prepared, _ = _prepare(c, panel, cache)
        up = runs.predict_upcoming(c, prepared, store.read_params(c.name, root=root), through, horizon_days)
        if len(up):
            parts.append(up.assign(code=i))
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()


def predictions(name: str, *, root: Path = MODEL_RUNS_DIR) -> pd.DataFrame:
    """Every code's stored predictions, ``code`` column (the walk-forward record)."""
    meta = json.loads((root / name / FAMILY_META).read_text())
    parts = [store.read_predictions(code_name(name, i), root=root).assign(code=i) for i in range(len(meta["codes"]))]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()
