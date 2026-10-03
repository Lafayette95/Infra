"""The short's delivery TIMING options: the wild card and the end-of-month (EOM) option
(infra/models/basis/CLAUDE.md). Pure functions, no I/O. Values in FUTURES points.

Mechanics (a DV01-hedged short holds h = 1/CF face of the CTD per futures contract):
* WILD CARD - on each intention day up to the last trading day, the invoice is fixed at
  the 15:00 New York settlement while bonds trade until the ~19:00 notice deadline.
  Delivering then locks the futures leg while the (h - 1) face TAIL keeps the window's
  move: payoff (h - 1) x dP. CF < 1 (long tail) pays on a rally after the settlement, CF > 1
  on a sell-off - the value scales with |1/CF - 1|, largest for UB (CFs 0.6-0.75).
  Exercising ends the position, so it's a Bermudan choice: deliver in a window only if
  that beats waiting. Windows are iid normal, sd = the CTD's daily price vol x
  sqrt(``window_var_share``) - measured 2024-2026: 6.1-6.7% of a day's futures variance
  falls between 15:00 and 19:00 New York, every root.
* END OF MONTH - after the last trading day the futures price is FROZEN while delivery stays
  open (ZB/UB/ZN/TN: 7 business days; ZT/ZF/Z3N: 3). The tail itself is plain exposure (no
  option: it can be sold any time); the option is SWITCHING bonds against the frozen
  price: on the delivery day the short delivers argmin_i (P_i - CF_i x F_frozen), a ranking
  that level moves now change at first order (bonds' DV01s differ while F stays put).
A gain of V bond points per contract lowers the fair futures price by V / CF_ctd (the
invoice is CF x F).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_max_normal(a: float, v: float) -> float:
    """E[max(a Z, v)] for Z ~ N(0, 1), a >= 0."""
    if a <= 0:
        return max(v, 0.0)
    x = v / a
    return float(v * norm.cdf(x) + a * norm.pdf(x))


def wildcard_value(cf: float, window_sd, n_windows: int | None = None, continuation: float = 0.0,
                   wait_cost: float = 0.0) -> float:
    """Value (bond points per contract) of the wild-card windows by backward induction,
    ``continuation`` = the value of waiting past the last window (the EOM option).
    ``window_sd``: one sd per window in CHRONOLOGICAL order (event days carry more), or a
    single sd repeated ``n_windows`` times. ``wait_cost``: what passing a window costs
    (bond points) - a day's NEGATIVE carry on the bond to deliver, when delivering early is
    the baseline: at each window deliver (the tail's move) or pay it and keep the later
    windows. Returns the value ABOVE ``continuation``."""
    sds = np.atleast_1d(np.asarray(window_sd, dtype="float64"))
    if n_windows is not None and sds.size == 1:
        sds = np.repeat(sds, max(n_windows, 0))
    tail = abs(1.0 / cf - 1.0)
    v = continuation
    for sd in sds[::-1]:  # backward from the last window
        v = expected_max_normal(tail * sd, v - wait_cost)
    return v - continuation


def window_kinds(days, fomc_days) -> list[str]:
    """Each wild-card window day's kind, for its variance: ``fomc`` (a decision day - the
    press conference straddles the 15:00 settlement), else ``quarter_end``, else
    ``month_end`` (index-extension and balance-sheet flows around the close), else
    ``ordinary``."""
    import pandas as pd
    days = pd.DatetimeIndex(days)
    fomc = set(pd.DatetimeIndex(fomc_days).normalize())
    nxt = days + pd.offsets.BDay(1)
    return ["fomc" if d in fomc else "quarter_end" if n.quarter != d.quarter else "month_end" if n.month != d.month
            else "ordinary" for d, n in zip(days, nxt)]


def eom_switch_value(prices_at_ld: np.ndarray, cf: np.ndarray, frozen_futures: float, ctd: int,
                     dv01: np.ndarray, level_sd_bp: float, z: np.ndarray) -> float:
    """Value (bond points per contract) of switching bonds after the last trading day:
    ``prices_at_ld`` = each bond's forward to the delivery day, ``dv01`` its price change
    per 1bp (points per 100 face), one level shock per path (``level_sd_bp`` x ``z``,
    first-order in price). Savings per path = cost of the LTD-CTD minus the cheapest cost,
    cost_i = P_i - CF_i x F_frozen."""
    shocks = level_sd_bp * np.asarray(z)
    prices = prices_at_ld[None, :] - shocks[:, None] * dv01[None, :]
    cost = prices - cf[None, :] * frozen_futures
    return float((cost[:, ctd] - cost.min(axis=1)).mean())
