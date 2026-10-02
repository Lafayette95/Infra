"""The UBS note's own results for rates bond futures, as published - the comparison target.

UBS Q-Series, "CTAs: How $375 bln Influences Global Assets?", dated 2022-09-05 (US Labor
Day), so the data date is taken as the prior business day, 2022-09-02, and "2 weeks ago"
as 2022-08-19. Transcribed 2026-10-02 from a scan. Only values that were LEGIBLE are
here; the rest are None (a clearer scan of Fig. 107 would fill them):

* Fig. 63 / 65 / 67 (clear): current signal, current vol (bp, yield space), liquidity
  factor, current position and its 10-year max ($DV01 mn);
* Fig. 102 (clear): every metric for US10Y.

UBS's asset-class weight 30% and portfolio vol scaling 809.9 (Fig. 63) reproduce its
positions as 0.30 x liquidity x signal x 809.9 / vol up to a common factor ~1.075 (US2Y
1.83 vs 2.0, US10Y 18.4 vs 19.7, EU10Y 11.0 vs 11.9): the formula is as stated, with one
unexplained constant (an asset-class weight nearer 32%, or a unit).
"""
from __future__ import annotations

import pandas as pd

AS_OF = pd.Timestamp("2022-09-02")
TWO_WEEKS_AGO = pd.Timestamp("2022-08-19")
HORIZON = 10  # "2 weeks", in observations

# name: (signal, vol_bp, liquidity, position_dv01mn, max_abs_position_10y)
FIG63 = {
    "US2Y": (-1.00, 132.9, 1, -2.0, 7.9),
    "US5Y": (-1.00, 133.0, 4, -7.8, 31.2),
    "US10Y": (-0.98, 103.4, 8, -19.7, 67.4),
    "US20Y": (-0.85, 98.6, 4, -9.0, 29.2),
    "US30Y": (-1.00, 102.5, 4, -10.2, 26.3),
    "EU2Y": (-0.76, 148.0, 1, -1.3, 8.3),
    "EU5Y": (-0.78, 157.4, 4, -5.2, 42.6),
    "EU10Y": (-0.74, 130.3, 8, -11.9, 78.3),
    "EU30Y": (-0.79, 96.2, 4, -8.6, 35.2),
    "FR10Y": (-0.81, 130.3, 2, -3.2, 20.9),
    "IT10Y": (-0.99, 192.6, 2, -2.7, 12.4),
    "GB10Y": (-1.00, 138.4, 4, -7.5, 25.8),
    "JP10Y": (0.12, 43.4, 4, 3.0, 41.8),
    "CA10Y": (-0.56, 111.6, 2, -2.6, 13.7),
    "AD10Y": (-0.66, 132.6, 2, -2.6, 12.5),
    "KR10Y": (-0.86, 105.0, 1, -2.1, 9.4),
}
ASSET_CLASS_WEIGHT = 0.30
PORTFOLIO_VOL_SCALING = 809.9

# Fig. 102, US10Y, complete.
US10Y = {
    "signal": -0.98, "signal_2w_ago": -0.46, "exp_signal_2w": -0.94,
    "vol": 103.4, "vol_2w_ago": 106.1, "exp_vol_2w": 101.0,
    "position": -19.74, "position_2w_ago": -8.24, "exp_position_2w": -19.32,
    "max_abs_position_10y": 67.21, "flow_2w_from_signal_adv": 0.54, "flow_2w_from_vol_adv": -0.28,
    "adv_dv01mn": 163.0,
}


def reference() -> pd.DataFrame:
    """One row per UBS asset: its published values, plus ``position_unit`` = position /
    10y max, the same [-1, 1] unit our model reports."""
    df = pd.DataFrame.from_dict(FIG63, orient="index",
                                columns=["signal", "vol_bp", "liquidity", "position", "max_abs_position"])
    df["position_unit"] = df["position"] / df["max_abs_position"]
    for col in ("signal_2w_ago", "exp_signal_2w", "vol_2w_ago", "exp_vol_2w", "position_2w_ago",
                "exp_position_2w"):
        df[col] = None
    for col in ("signal_2w_ago", "exp_signal_2w", "vol_2w_ago", "exp_vol_2w"):
        df.loc["US10Y", col] = US10Y[col]
    df.loc["US10Y", "position_2w_ago"] = US10Y["position_2w_ago"]
    df.loc["US10Y", "exp_position_2w"] = US10Y["exp_position_2w"]
    df["position_unit_2w_ago"] = pd.to_numeric(df["position_2w_ago"]) / df["max_abs_position"]
    df["exp_position_unit_2w"] = pd.to_numeric(df["exp_position_2w"]) / df["max_abs_position"]
    df.index.name = "asset"
    return df
