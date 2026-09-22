"""Implied forward price and discount factor from put-call parity.

No risk-free-rate curve pipeline needed - both fall out of the option chain itself.
See CLAUDE.md section 9.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_forward_and_discount(chain: pd.DataFrame) -> tuple[float, float]:
    """(forward, discount_factor) from a SINGLE (underlying, expiry, day) option chain.

    Put-call parity for options on a future: ``C - P = discount * (forward - strike)``,
    linear in strike. Regressing ``C - P`` against strike gives slope = -discount,
    intercept = discount * forward. Verified against real SR3U6 data 2026-09-21
    (CLAUDE.md section 9): the implied forward matched the future's own settlement
    price to 4 decimal places, and the fit was line-straight across ~100 strikes.

    ``chain`` needs a ``strike``, ``option_type`` ('C'/'P') and ``settlement_price``
    column; rows missing either side at a given strike are dropped before fitting.
    """
    piv = chain.pivot_table(index="strike", columns="option_type", values="settlement_price")
    if "C" not in piv.columns or "P" not in piv.columns:
        raise ValueError("chain must include both calls ('C') and puts ('P')")
    piv = piv.dropna(subset=["C", "P"])
    if len(piv) < 2:
        raise ValueError(
            f"need at least 2 strikes with both a call and a put settlement to fit "
            f"parity, got {len(piv)}"
        )
    x = piv.index.to_numpy(dtype=float)
    y = (piv["C"] - piv["P"]).to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    discount = -slope
    if discount <= 0:
        raise ValueError(f"fitted discount factor {discount:.4f} is non-positive - degenerate or bad chain")
    forward = intercept / discount
    return float(forward), float(discount)
