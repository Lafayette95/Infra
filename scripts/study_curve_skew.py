"""The 63-day skew of curve structures as a reversal signal: horizons, overlap with plain
mean reversion, and other countries' curves (positioning CLAUDE.md 8-9).

Structures (+ = a LOSS for a long holder, bp, duration-neutral from yield changes):
  CURVE a_b (long a, short b: a steepener):  pain = dy_a - dy_b
  FLY a_b_c (long the belly b vs 50/50 wings): pain = dy_b - (dy_a + dy_c) / 2
Signal: rolling 63-day skew of the daily pain moves. Controls: the past 63-day pain (the
structure's own recent move) and the level z (cumulative pain vs its trailing 252-day mean
/ sd - the fade-the-level-z of mean-reversion models, infra/models/meanrev "exante").
Forward: pain over the next h days, sampled every h days (non-overlapping).

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/study_curve_skew.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from infra.analytics.positioning.asymmetry import rolling_skew  # noqa: E402
from infra.pipeline.series_panel import read_panel  # noqa: E402

STRUCTURES = {"2s10s": ("curve", 2, 10), "5s30s": ("curve", 5, 30), "2s5s": ("curve", 2, 5),
              "10s30s": ("curve", 10, 30), "2/5/10": ("fly", 2, 5, 10), "5/10/30": ("fly", 5, 10, 30)}
SOURCES = {"US OTR": ("bmk", "US", "2008-09-02"), "US CMT": ("bond", "US", "2016-01-04"),
           "DE": ("bond", "DE", "2016-01-04"), "UK": ("bond", "UK", "2016-01-04")}
HORIZONS = (5, 10, 20, 40, 60)
SKEW_WINDOW, PAST, Z_WINDOW = 63, 63, 252


def bmk_pain(start: str) -> pd.DataFrame:
    """US on-the-run structures from the bmk yield P&L, which follows the SAME bond across an
    on-the-run switch. Differencing the otr: yield series instead puts a fake move on every
    switch day (a new bond, another maturity) - found 2026-10-08: it cut the skew effect from
    -0.10 to -0.03."""
    ids = {}
    for name, (kind, *t) in STRUCTURES.items():
        legs = "__".join(f"US_BOND_{x}y" for x in t)
        ids[name] = f"bmk:otr:{'CURVE' if kind == 'curve' else 'FLY'}__{legs}"
    p = read_panel(list(ids.values()), start, "2026-10-08")
    return -p.rename(columns={v: k for k, v in ids.items()}).dropna(how="all")   # long P&L -> pain


def structure_pain(source: str, country: str, start: str) -> pd.DataFrame:
    if source == "bmk":
        return bmk_pain(start)
    tenors = sorted({t for s in STRUCTURES.values() for t in s[1:]})
    y = read_panel([f"{source}:{country}_BOND_{t}y" for t in tenors], start, "2026-10-08")
    dy = (y.diff() * 100).rename(columns=lambda c: int(c.split("_")[-1][:-1]))
    out = {}
    for name, (kind, *t) in STRUCTURES.items():
        out[name] = dy[t[0]] - dy[t[1]] if kind == "curve" else dy[t[1]] - (dy[t[0]] + dy[t[2]]) / 2
    return pd.DataFrame(out).dropna(how="all")


def panel_for(pain: pd.DataFrame, h: int) -> pd.DataFrame:
    """Standardised (signal, controls, forward) per structure, sampled every h days, stacked."""
    rows = []
    for s in pain.columns:
        p = pain[s].dropna()
        lvl = p.cumsum()
        d = pd.DataFrame({
            "skew": rolling_skew(p, SKEW_WINDOW, SKEW_WINDOW),
            "past": p.rolling(PAST).sum(),
            "z": (lvl - lvl.rolling(Z_WINDOW).mean()) / lvl.rolling(Z_WINDOW).std(),
            "fwd": p.rolling(h).sum().shift(-h),
        }).dropna().iloc[::h]
        rows.append(((d - d.mean()) / d.std()).assign(structure=s))
    return pd.concat(rows)


def pooled(d: pd.DataFrame, cols: list[str], n_struct: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = d[cols].to_numpy(), d["fwd"].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    # the structures are correlated: count them as ~half independent (variance x 2)
    cov = np.linalg.inv(X.T @ X) * (e @ e) / (len(y) - len(cols)) * 2
    return b, b / np.sqrt(np.diag(cov))


def main() -> int:
    print("Signal: 63-day skew of a structure's daily moves (+ = skewed towards sharp losses for a long holder).")
    print("Coefficient on the standardised forward loss; NEGATIVE = reversal (the long recovers). t counts")
    print("the 6 structures as ~3 independent.\n")
    us_otr = None
    for label, (src, ctry, start) in SOURCES.items():
        pain = structure_pain(src, ctry, start)
        if label == "US OTR":
            us_otr = pain
        if label == "US CMT" and us_otr is not None:
            j = pd.concat([us_otr.add_suffix("_otr"), pain.add_suffix("_cmt")], axis=1).dropna()
            c = np.mean([j[f"{s}_otr"].corr(j[f"{s}_cmt"]) for s in STRUCTURES])
            print(f"  (US OTR vs CMT structure moves, daily correlation: {c:.2f})")
        print(f"== {label} ({pain.index.min().date()}..{pain.index.max().date()})")
        for h in HORIZONS:
            d = panel_for(pain, h)
            b1, t1 = pooled(d, ["skew"], 6)
            b3, t3 = pooled(d, ["skew", "past", "z"], 6)
            print(f"  h={h:2d}d  skew alone {b1[0]:+.3f} (t {t1[0]:+.1f})   with controls: skew {b3[0]:+.3f} "
                  f"(t {t3[0]:+.1f}), past move {b3[1]:+.3f} (t {t3[1]:+.1f}), level z {b3[2]:+.3f} "
                  f"(t {t3[2]:+.1f})   n {len(d)}")
        d = panel_for(pain, 20)
        per = {s: np.corrcoef(g["skew"], g["fwd"])[0, 1] for s, g in d.groupby("structure")}
        print("  per structure, h=20, skew alone: " + "  ".join(f"{s} {r:+.2f}" for s, r in per.items()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
