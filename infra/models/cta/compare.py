"""Our model against UBS's own published snapshot (2022-09-02), in UBS's units where
they can be matched:

* signals are unit-free: compared as is;
* positions: UBS's $DV01 / its own 10y max = our [-1, 1] ``position``;
* vols are in different spaces (UBS: yield bp; ours: futures price points), so only
  their RATIOS are compared (vol 2w ago / now, expected / now);
* expected flows: UBS's %ADV x ADV / 10y max -> our position units (US10Y only).

Reads stored futures (disk only), so it needs the US bond futures history from 2014-12.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from infra.models.cta import paper
from infra.models.cta.config import CTASpec, get_spec, get_universe
from infra.models.cta.inputs import universe_prices
from infra.models.cta.model import CTAModel


def compare_with_paper(spec: CTASpec | str = "ubs2022", universe: str = "ubs_us_bonds",
                       prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per asset of ``universe`` that UBS reports: ``<metric>_ubs`` vs ``<metric>``."""
    spec = get_spec(spec) if isinstance(spec, str) else spec
    spec = replace(spec, flow_horizons=tuple(sorted(set(spec.flow_horizons) | {paper.HORIZON})))
    uni = get_universe(universe)
    if prices is None:
        prices = universe_prices(uni, "2014-12-01", paper.AS_OF)
    model = CTAModel(spec)
    model.fit(model.prepare(prices, universe=uni), as_of=paper.AS_OF)
    res = model.predict()
    frame = res.frame.set_index(["timestamp", "asset"])
    fc = res.forecast[res.forecast["horizon"] == paper.HORIZON].set_index("asset")
    ref = paper.reference()
    rows = []
    for asset in [a.name for a in uni.assets if a.name in ref.index]:
        if (paper.AS_OF, asset) not in frame.index:
            continue
        now, ago = frame.loc[(paper.AS_OF, asset)], frame.loc[(paper.TWO_WEEKS_AGO, asset)]
        r = ref.loc[asset]
        row = {
            "asset": asset,
            "signal_ubs": r["signal"], "signal": now["signal"],
            "signal_2w_ago_ubs": r["signal_2w_ago"], "signal_2w_ago": ago["signal"],
            "exp_signal_2w_ubs": r["exp_signal_2w"], "exp_signal_2w": fc["exp_signal"].get(asset),
            "position_ubs": r["position_unit"], "position": now["position"],
            "position_2w_ago_ubs": r["position_unit_2w_ago"], "position_2w_ago": ago["position"],
            "exp_position_2w_ubs": r["exp_position_unit_2w"], "exp_position_2w": fc["exp_position"].get(asset),
            "vol_ratio_2w_ago_ubs": (r["vol_2w_ago"] / r["vol_bp"]) if r["vol_2w_ago"] is not None else None,
            "vol_ratio_2w_ago": ago["vol"] / now["vol"],
            "exp_vol_ratio_ubs": (r["exp_vol_2w"] / r["vol_bp"]) if r["exp_vol_2w"] is not None else None,
            "exp_vol_ratio": now["forecast_vol"] / now["vol"],
            "flow_from_signal": fc["flow_from_signal"].get(asset),
            "flow_from_vol": fc["flow_from_vol"].get(asset),
        }
        if asset == "US10Y":
            to_units = paper.US10Y["adv_dv01mn"] / paper.US10Y["max_abs_position_10y"] / 100.0
            row["flow_from_signal_ubs"] = paper.US10Y["flow_2w_from_signal_adv"] * to_units
            row["flow_from_vol_ubs"] = paper.US10Y["flow_2w_from_vol_adv"] * to_units
        rows.append(row)
    return pd.DataFrame(rows).set_index("asset").apply(pd.to_numeric)
