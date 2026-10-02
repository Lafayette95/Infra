"""US Treasury on/off-the-run map (CLAUDE.md 18). Pure functions, no I/O.

Per day and tenor (``TREASURY_OTR_TENORS``: security type + original term), the original
issues of that term ranked newest first - 0 = on the run, 1 = 1-old, ... - counting an
issue from its ``issue_date`` ("issue" convention) or its ``auction_date`` ("auction"),
and dropping it once matured. A reopening never starts a new series: only original
issues are ranked (the reference table holds exactly those).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OTR_COLUMNS = ["timestamp", "tenor", "rank", "convention", "cusip", "issue_date", "auction_date", "maturity_date",
               "coupon"]
OTR_KEYS = ["timestamp", "tenor", "rank", "convention"]
_SWITCH = {"issue": "issue_date", "auction": "auction_date"}


def otr_map(securities: pd.DataFrame, days, tenors: dict[str, tuple[str, str]], depth: int,
            conventions=("issue", "auction")) -> pd.DataFrame:
    """Rows OTR_COLUMNS for every day in ``days`` x tenor x rank (0..depth) x convention
    that has an issue. ``securities``: the reference table."""
    days = pd.DatetimeIndex(days).normalize()
    out = []
    for tenor, (stype, term) in tenors.items():
        series = securities[(securities["security_type"] == stype) & (securities["original_term"] == term)]
        for conv in conventions:
            col = _SWITCH[conv]
            s = series.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
            if s.empty:
                continue
            start = s[col].to_numpy(dtype="datetime64[ns]")
            mat = s["maturity_date"].to_numpy(dtype="datetime64[ns]")
            # newest issue counted by each day: last start <= day
            newest = np.searchsorted(start, days.to_numpy(dtype="datetime64[ns]"), side="right") - 1
            for rank in range(depth + 1):
                idx = newest - rank
                ok = idx >= 0
                ok[ok] &= mat[idx[ok]] > days.to_numpy(dtype="datetime64[ns]")[ok]
                if not ok.any():
                    continue
                pick = s.iloc[idx[ok]]
                out.append(pd.DataFrame({
                    "timestamp": days[ok], "tenor": tenor, "rank": rank, "convention": conv,
                    "cusip": pick["cusip"].to_numpy(), "issue_date": pick["issue_date"].to_numpy(),
                    "auction_date": pick["auction_date"].to_numpy(), "maturity_date": pick["maturity_date"].to_numpy(),
                    "coupon": pick["coupon"].to_numpy(),
                }))
    if not out:
        return pd.DataFrame(columns=OTR_COLUMNS)
    df = pd.concat(out, ignore_index=True)[OTR_COLUMNS]
    df["rank"] = df["rank"].astype("int8")
    for c in ("timestamp", "issue_date", "auction_date", "maturity_date"):
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ms]")
    return df.sort_values(OTR_KEYS).reset_index(drop=True)
