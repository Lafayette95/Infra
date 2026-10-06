"""Volatility risk premium maths and the point-in-time reader. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.analytics import vrp as va
from infra.pipeline import vrp
from infra.storage import parquet_store


def test_realised_vols_annualise_daily_changes():
    ch = pd.Series(np.tile([5.0, -5.0], 40), index=pd.bdate_range("2026-01-01", periods=80))
    assert np.isclose(va.trailing_vol(ch, 21).iloc[-1], 5.0 * np.sqrt(252))
    assert np.isclose(va.ewma_vol(ch, 0.94).iloc[-1], 5.0 * np.sqrt(252), rtol=1e-3)
    assert np.isnan(va.trailing_vol(ch, 21).iloc[5])  # fewer than min_obs
    path = ch.cumsum()
    assert np.isclose(va.realised_over(path), 5.0 * np.sqrt(252))


def test_read_vrp_hides_ex_post_values_not_yet_known(tmp_path):
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-01"]).astype("datetime64[ms]"), "instrument": ["SWPT_1m_10y"],
                       "iv": [86.0], "rv_life": [71.0], "vrp_life": [15.0],
                       "life_end": pd.to_datetime(["2026-10-01"]).astype("datetime64[ms]")})
    parquet_store.write_partitioned(df, tmp_path, vrp.KEYS)
    assert np.isnan(vrp.read_vrp(as_of="2026-09-30", root=tmp_path)["vrp_life"].iloc[0])
    assert vrp.read_vrp(as_of="2026-10-01", root=tmp_path)["vrp_life"].iloc[0] == 15.0
