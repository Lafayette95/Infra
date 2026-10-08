"""Does the 63-day skew (positioning CLAUDE.md 8-9) add to infra/models/meanrev's signal on
the PCA residuals it actually trades? Research only - runs meanrev's own walk-forward, never
changes it (positioning CLAUDE.md 10).

Per residual and day (out of sample, monthly refits): its daily move (the day-to-day change of
meanrev's `level:` within a fit - the level is re-based at each refit, whose first day is
dropped), the 63-day skew of those moves, meanrev's OU s-score `s:`, the past 63-day move;
forward = the residual's move over the next h days (sampled every h). Pooled, standardised.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/study_meanrev_skew.py
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
from infra.models.meanrev.config import get_meanrev_spec  # noqa: E402
from infra.models.meanrev.evaluate import walk  # noqa: E402
from infra.pipeline.series_panel import read_panel  # noqa: E402

K = tuple(f"bmk:otr:US_BOND_{t}y" for t in (2, 3, 5, 7, 10, 30))


def ols(d: pd.DataFrame, cols: list[str], n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = d[cols].to_numpy(), d["fwd"].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    cov = np.linalg.inv(X.T @ X) * (e @ e) / (len(y) - len(cols)) * 2   # residuals ~half independent
    return b, b / np.sqrt(np.diag(cov))


def main() -> int:
    panel = read_panel(list(K), "2009-03-02", "2026-09-30").dropna()
    spec = get_meanrev_spec("prior", columns=K, window=250)
    res = walk(spec, panel, "2012-01-03", "2026-09-30")
    p = res.predictions
    fit = p["fit_as_of"] if "fit_as_of" in p else None
    residuals = sorted(c.split(":", 1)[1] for c in p.columns if c.startswith("level:"))
    print(f"meanrev walk-forward (spec prior, K = US OTR 2/3/5/7/10/30y, window 250): {len(p)} days, "
          f"{len(residuals)} residuals; fit column {'present' if fit is not None else 'absent'}")
    new_fit = (fit != fit.shift(1)) if fit is not None else pd.Series(False, index=p.index)
    for h in (5, 10, 20):
        rows = []
        for r in residuals:
            lvl = p[f"level:{r}"]
            m = lvl.diff().where(~new_fit)                       # daily residual move within a fit
            d = pd.DataFrame({"skew": rolling_skew(m, 63, 50), "s": p[f"s:{r}"],
                              "past": m.rolling(63, min_periods=50).sum(),
                              "fwd": m.rolling(h, min_periods=h - 2).sum().shift(-h)}).dropna().iloc[::h]
            rows.append(((d - d.mean()) / d.std()).assign(res=r, s_raw=d["s"], skew_raw=d["skew"]))
        d = pd.concat(rows)
        b1, t1 = ols(d, ["skew"], len(residuals))
        b2, t2 = ols(d, ["s"], len(residuals))
        b3, t3 = ols(d, ["skew", "s", "past"], len(residuals))
        print(f"\nh={h:2d}d (n {len(d)}): skew alone {b1[0]:+.3f} (t {t1[0]:+.1f}) | meanrev s alone {b2[0]:+.3f} "
              f"(t {t2[0]:+.1f}) | together: skew {b3[0]:+.3f} (t {t3[0]:+.1f}), s {b3[1]:+.3f} (t {t3[1]:+.1f}), "
              f"past {b3[2]:+.3f} (t {t3[2]:+.1f})")
        same = np.sign(d["skew_raw"]) == np.sign(d["s_raw"])
        for name, m_ in (("skew on the side of the dislocation", same), ("skew against it", ~same)):
            b, t = ols(d[m_], ["s"], len(residuals))
            print(f"   s-score's power when {name:36s}: {b[0]:+.3f} (t {t[0]:+.1f}, n {int(m_.sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
