"""Display-only timezone conversion for chart x-axes.

This is the ONLY place in the codebase allowed to localize/convert a timestamp (see
CLAUDE.md section 7). Everything upstream - infra.pipeline, infra.processing,
infra.relative, infra.storage, infra.api - stays tz-naive UTC always. Nothing here may
mutate a DataFrame coming from the pipeline layer; it only builds a separate index for
Plotly to render, on a copy the caller supplies.
"""
from __future__ import annotations

import pandas as pd

# label (dropdown) -> IANA zone. UTC first and default, so no selection means no
# conversion. The rest cover the exchanges this project trades.
DISPLAY_TIMEZONES: dict[str, str] = {
    "UTC": "UTC",
    "New York (CME/CBOT)": "America/New_York",
    "Chicago (CME/CBOT floor)": "America/Chicago",
    "London (ICE)": "Europe/London",
    "Frankfurt (Eurex)": "Europe/Berlin",
}
DEFAULT_TIMEZONE = "UTC"


def to_display_index(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """UTC tz-naive index -> a NEW tz-naive index showing wall-clock time in ``tz``.

    Returns the input unchanged (same object) for UTC/empty so the common case is free.
    Never mutates ``index``; the caller must treat the result as display-only and never
    write it back to a DataFrame that flows anywhere outside the chart being built.
    """
    if not tz or tz == "UTC":
        return index
    return index.tz_localize("UTC").tz_convert(tz).tz_localize(None)
