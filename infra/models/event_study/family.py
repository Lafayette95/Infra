"""Families of event studies: many window codes generated from rules (``FamilySpec``: a
cartesian product of leg values), each fitted like a single study on ONE shared P&L panel,
then judged together.

Testing dozens of windows x instruments makes some pass by luck, so a family adds
**Benjamini-Hochberg** false-discovery-rate control over all its (code, instrument) tests
(``q`` = the BH-adjusted p-value of the mean's t-test; ``passed_fdr`` = the study's own tests
AND ``q <= fdr_q``). With a condition, the bucket rows (``instrument|-1`` ...) are a second
family: ``q_did`` over their difference-in-differences p-values, ``passed_fdr`` = the
bucket's own and conditioning tests AND ``q_did <= fdr_q``. Codes whose every window is illegal (an end before its start, a lag off
the grid) are kept with ``n = 0`` and the reasons, never silently dropped.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from infra.models.event_study.config import EVENT_FAMILIES, FamilySpec
from infra.models.event_study.model import EventStudy


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q); NaN stays NaN."""
    p = np.asarray(p, dtype="float64")
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(m)
    out[order] = adj
    q[ok] = out
    return q


def family_studies(family: FamilySpec | str) -> list[EventStudy]:
    fam = EVENT_FAMILIES[family] if isinstance(family, str) else family
    return [EventStudy(replace(fam.study, name=f"{fam.name}#{i}", code=c)) for i, c in enumerate(fam.codes())]


def run_family(family: FamilySpec | str, panel: pd.DataFrame, as_of=None, *, occurrences=None) -> pd.DataFrame:
    """Fit every code of the family at ``as_of`` on ``panel`` (one grid, the family's
    instruments): long table ``code, instrument`` + statistics + tests + ``q`` + ``passed_fdr``,
    and per code the count of legal windows and the illegal reasons."""
    fam = EVENT_FAMILIES[family] if isinstance(family, str) else family
    rows = []
    for study in family_studies(fam):
        occ = occurrences if occurrences is not None else study.occurrences(as_of)
        data = study.prepare(panel, occurrences=occ)
        ev = data[data["kind"] == "event"]
        legal = ev["legal"].astype(bool)
        reasons = ev.loc[~legal, "reason"].value_counts().to_dict()
        study.fit(data, as_of=as_of)
        t = study.fitted_.table.reset_index()
        t.insert(0, "code", study.code.encode())
        t["legal_windows"] = int(legal.sum())
        t["illegal_reasons"] = ";".join(f"{k}={v}" for k, v in reasons.items())
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    overall = ~out["instrument"].astype(str).str.contains("|", regex=False)
    out["q"] = np.nan
    out.loc[overall, "q"] = benjamini_hochberg(out.loc[overall, "p_t"].to_numpy())
    out["passed_fdr"] = ((out["passed"] == 1.0) & (out["q"] <= fam.fdr_q)).astype(float)
    if "p_did" in out:
        # conditional buckets: their own family of tests - the conditioning's added value (DiD)
        out["q_did"] = np.nan
        out.loc[~overall, "q_did"] = benjamini_hochberg(out.loc[~overall, "p_did"].to_numpy())
        out.loc[~overall, "passed_fdr"] = ((out.loc[~overall, "cond_passed"] == 1.0)
                                           & (out.loc[~overall, "q_did"] <= fam.fdr_q)).astype(float)
    return out
