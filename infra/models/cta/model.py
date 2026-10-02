"""The CTA trend-following positioning model (UBS Q-Series 2022 methodology).

    model = CTAModel("ubs2022")                    # parameters by name (config.CTA_MODELS)
    data = model.prepare(prices, universe=uni)     # 1. data prep (any wide price frame)
    model.fit(data, as_of="2022-09-02")            # 2. fit: slow parameters, point in time
    res = model.predict()                          # 3. outputs at the fit date, or
    res = model.predict(new_prices)                #    on new data only (daily or intraday)
    model.reaction(res, "price")                   # spot reaction function

What ``fit`` estimates and freezes, per asset: each EWMA pair's scale (RMS of its
crossover), the response function (ECDF of the combined score, by default), the position
scale (max |raw position| over 10y); and once for the portfolio: the vol scaling (3y).
What ``predict`` updates from the data: the EWMAs and all rolling vols, continued exactly
from the fit date's state (EWMA values and the trailing returns are stored).

Intraday: pass the daily observations after the fit date with the intraday price as the
LAST row; it is read as a provisional close (one full EWMA step). Rows before it must be
daily, since every row is one step of the daily recursion.

Outputs (``CTAResult``):
* ``frame``: per (timestamp, asset): ``signal`` and ``position`` in [-1, 1], the vols,
  ``raw_position`` (weight x liquidity x signal x pvs / vol) and ``chg_<h>`` = position
  change over the past h observations;
* ``forecast``: per (timestamp, asset, horizon): expected signal / vol / position and the
  expected flow split into its signal and vol parts (``forecast.expected_flows``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from infra.models.base import Model
from infra.models.cta import signal as sig
from infra.models.cta.config import CTASpec, CTAUniverse, get_spec
from infra.models.cta.forecast import expected_flows
from infra.models.cta.prep import CTAData, prepare
from infra.models.cta.reaction import REACTIONS
from infra.models.cta.state import AssetParams, AssetState, raw_position, to_position, to_signal

FRAME_COLUMNS = ["timestamp", "asset", "level", "ret", "norm_vol", "vol", "forecast_vol", "score",
                 "signal", "pvs", "raw_position", "position"]


@dataclass
class Fitted:
    as_of: pd.Timestamp
    params: dict[str, AssetParams]
    pvs: float                               # frozen for predict
    frames: dict[str, pd.DataFrame]          # per asset, the fit sample's full frame
    ret_tail: dict[str, np.ndarray]          # trailing returns, so rolling vols continue
    data: CTAData                            # metadata (and the fit sample) for re-preparing


@dataclass
class CTAResult:
    spec: str
    fit_as_of: pd.Timestamp
    frame: pd.DataFrame
    forecast: pd.DataFrame
    states: dict[str, AssetState] = field(default_factory=dict)

    def latest(self) -> pd.DataFrame:
        """The last row per asset, with its expected flows as columns ``flow_<h>``."""
        last = self.frame.sort_values("timestamp").groupby("asset").tail(1).set_index("asset")
        if not self.forecast.empty:
            wide = self.forecast.pivot(index="asset", columns="horizon",
                                       values=["exp_signal", "exp_position", "flow"])
            wide.columns = [f"{name}_{h}" for name, h in wide.columns]
            last = last.join(wide)
        return last


def _asset_frame(level: pd.Series, ret: pd.Series, spec: CTASpec, *, ema_init=None, tail=None,
                 obs_offset: int = 0) -> pd.DataFrame:
    """EWMAs, crossovers and vols of one series (on its own calendar)."""
    df = pd.DataFrame({"level": level, "ret": ret})
    for n in spec.spans:
        df[f"ema_{n}"] = sig.ewma(level, n, None if ema_init is None else ema_init[n])
    df["norm_vol"] = sig.rolling_vol(ret, spec.norm_vol_window, spec.vol_min_obs, tail)
    annual = np.sqrt(spec.periods_per_year)
    df["vol"] = sig.rolling_vol(ret, spec.sizing_vol_window, spec.vol_min_obs, tail) * annual
    df["forecast_vol"] = sig.rolling_vol(ret, spec.forecast_vol_window, spec.vol_min_obs, tail) * annual
    z = sig.crossover_z({n: df[f"ema_{n}"] for n in spec.spans}, spec.ewma_pairs, df["norm_vol"])
    for k, zk in enumerate(z):
        df[f"z{k}"] = zk
    df["obs"] = obs_offset + np.arange(len(df))
    return df


def _changes(position: pd.Series, horizons) -> pd.DataFrame:
    return pd.DataFrame({f"chg_{h}": position - position.shift(h) for h in horizons})


class CTAModel(Model):
    """See the module docstring. ``spec``: a ``CTASpec`` or its name in ``CTA_MODELS``."""

    def __init__(self, spec: CTASpec | str = "ubs2022"):
        self.spec = get_spec(spec) if isinstance(spec, str) else spec
        self.fitted_: Fitted | None = None

    # ------------------------------------------------------------------ 1. prepare
    def prepare(self, raw: pd.DataFrame, *, universe: CTAUniverse | None = None, **kwargs) -> CTAData:
        return prepare(raw, universe=universe, **kwargs)

    def _as_data(self, data) -> CTAData:
        if isinstance(data, CTAData):
            return data
        f = self.fitted_
        anchor = {a: float(fr["level"].iloc[-1]) for a, fr in f.frames.items() if len(fr)}
        return prepare(data, assets=f.data.assets, class_weights=f.data.class_weights, anchor=anchor)

    # ------------------------------------------------------------------ 2. fit
    def fit(self, prepared: CTAData, as_of=None) -> "CTAModel":
        spec = self.spec
        as_of = prepared.levels.index.max() if as_of is None else pd.Timestamp(as_of)
        weights = spec.weights()
        frames, scales, scores = {}, {}, {}
        for asset in prepared.levels.columns:
            level, ret = prepared.series(asset)
            level, ret = level[level.index <= as_of], ret[ret.index <= as_of]
            df = _asset_frame(level, ret, spec)
            valid = (df["obs"] >= spec.warmup) & df["norm_vol"].notna()
            scales[asset] = np.array([sig.rms(df.loc[valid, f"z{k}"]) for k in range(len(spec.ewma_pairs))])
            df["score"] = sig.combined_score([df[f"z{k}"] for k in range(len(spec.ewma_pairs))],
                                             scales[asset], weights).where(valid)
            frames[asset], scores[asset] = df, df["score"].dropna().to_numpy()

        responses = self._fit_responses(scores)
        unit = {}
        for asset, df in frames.items():
            df["signal"] = to_signal(_partial(responses[asset]), df["score"].to_numpy())
            cw, liq = prepared.class_weight(asset), prepared.assets[asset].liquidity
            unit[asset] = cw * liq * df["signal"] / df["vol"]
        unit_wide = pd.DataFrame(unit).sort_index()
        rets_wide = pd.DataFrame({a: f["ret"] for a, f in frames.items()}).reindex(unit_wide.index)
        pvs = sig.portfolio_vol_scaling(unit_wide, rets_wide, spec.portfolio_vol_target, spec.pvs_window,
                                        spec.pvs_min_obs, spec.periods_per_year)
        pvs_now = pvs.dropna()
        pvs_frozen = float(pvs_now.iloc[-1]) if len(pvs_now) else np.nan

        params, tails = {}, {}
        for asset, df in frames.items():
            df["pvs"] = pvs.reindex(df.index)
            df["raw_position"] = unit[asset] * df["pvs"]
            recent = df["raw_position"].iloc[-spec.position_scale_window:].abs()
            pos_scale = float(recent.max()) if recent.notna().any() else np.nan
            params[asset] = AssetParams(
                pair_scale=scales[asset], response=responses[asset], pos_scale=pos_scale,
                class_weight=prepared.class_weight(asset), liquidity=prepared.assets[asset].liquidity,
            )
            df["position"] = to_position(params[asset], df["raw_position"].to_numpy(), spec.clip_position)
            tails[asset] = df["ret"].iloc[-spec.tail_length:].to_numpy()
        self.fitted_ = Fitted(as_of=as_of, params=params, pvs=pvs_frozen, frames=frames,
                              ret_tail=tails, data=prepared)
        return self

    def _fit_responses(self, scores: dict[str, np.ndarray]) -> dict:
        spec = self.spec
        if spec.response_pool == "pooled":
            pooled = np.concatenate([s for s in scores.values()]) if scores else np.array([])
            enough = len(pooled) >= spec.response_min_obs
            shared = sig.fit_response(spec.response, pooled, spec.response_gain) if enough else None
            return {a: (shared if len(s) else None) for a, s in scores.items()}
        if spec.response_pool != "asset":
            raise ValueError(f"unknown response_pool {spec.response_pool!r} (asset | pooled)")
        return {a: (sig.fit_response(spec.response, s, spec.response_gain) if len(s) >= spec.response_min_obs else None)
                for a, s in scores.items()}

    # ------------------------------------------------------------------ 3. predict
    def predict(self, prepared=None, *, forecast_at: str = "last",
                horizons: tuple[int, ...] | None = None) -> CTAResult:
        """Outputs on ``prepared`` (a ``CTAData`` or a raw price frame) - rows AFTER the fit
        date only - or, with ``prepared=None``, on the fit sample itself (the last row is
        the fit date). ``forecast_at``: "last" (expected flows at each asset's last row,
        the default), "none"."""
        self.check_fitted()
        spec, f = self.spec, self.fitted_
        new_frames = dict(f.frames) if prepared is None else self._step(self._as_data(prepared))
        rows, states, forecasts = [], {}, []
        for asset, df in new_frames.items():
            if df.empty:
                continue
            hist = f.frames[asset]["position"]
            pos_all = df["position"] if prepared is None else pd.concat([hist, df["position"]])
            chg = _changes(pos_all, spec.change_horizons).reindex(df.index)
            out = df.assign(asset=asset).join(chg)
            out.index.name = "timestamp"
            rows.append(out.reset_index()[FRAME_COLUMNS + list(chg.columns)])
            states[asset] = self._state(asset, df.iloc[-1], df.index[-1])
            if forecast_at == "last":
                forecasts.append(expected_flows(f.params[asset], states[asset], spec, horizons))
            elif forecast_at != "none":
                raise ValueError(f"forecast_at {forecast_at!r} (last | none)")
        frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=FRAME_COLUMNS)
        forecast = pd.concat([x for x in forecasts if not x.empty], ignore_index=True) \
            if any(not x.empty for x in forecasts) else pd.DataFrame()
        return CTAResult(spec=spec.name, fit_as_of=f.as_of, frame=frame, forecast=forecast, states=states)

    def _step(self, data: CTAData) -> dict[str, pd.DataFrame]:
        """Run the frozen model forward over observations after each asset's fit date."""
        spec, f = self.spec, self.fitted_
        weights = spec.weights()
        out = {}
        for asset, fitted in f.frames.items():
            if asset not in data.levels.columns:
                continue
            level, ret = data.series(asset)
            last = fitted.index[-1] if len(fitted) else pd.Timestamp.min
            keep = level.index > last
            level, ret = level[keep], ret[keep]
            if level.empty:
                out[asset] = fitted.iloc[0:0]
                continue
            end = fitted.iloc[-1]
            df = _asset_frame(level, ret, spec, ema_init={n: end[f"ema_{n}"] for n in spec.spans},
                              tail=f.ret_tail[asset], obs_offset=int(end["obs"]) + 1)
            p = f.params[asset]
            valid = (df["obs"] >= spec.warmup) & df["norm_vol"].notna()
            df["score"] = sig.combined_score([df[f"z{k}"] for k in range(len(spec.ewma_pairs))],
                                             p.pair_scale, weights).where(valid)
            df["signal"] = to_signal(p, df["score"].to_numpy())
            df["pvs"] = f.pvs
            df["raw_position"] = raw_position(p, df["signal"], f.pvs, df["vol"])
            df["position"] = to_position(p, df["raw_position"].to_numpy(), spec.clip_position)
            out[asset] = df
        return out

    def _state(self, asset: str, row: pd.Series, ts) -> AssetState:
        return AssetState(
            asset=asset, timestamp=pd.Timestamp(ts), level=float(row["level"]),
            emas={n: float(row[f"ema_{n}"]) for n in self.spec.spans},
            norm_vol=float(row["norm_vol"]), vol=float(row["vol"]), forecast_vol=float(row["forecast_vol"]),
            signal=float(row["signal"]), raw_position=float(row["raw_position"]),
            position=float(row["position"]), pvs=float(row["pvs"]),
        )

    # ------------------------------------------------------------------ reaction function
    def reaction(self, result: CTAResult | None = None, axis: str = "price", *, assets=None,
                 grid=None) -> pd.DataFrame:
        """Spot reaction function at each asset's latest state in ``result`` (default: the
        fit date). ``axis``: a key of ``reaction.REACTIONS``."""
        self.check_fitted()
        try:
            fn = REACTIONS[axis]
        except KeyError:
            raise KeyError(f"unknown reaction axis {axis!r}; known: {sorted(REACTIONS)}") from None
        result = self.predict(forecast_at="none") if result is None else result
        names = assets or list(result.states)
        frames = [fn(self.fitted_.params[a], result.states[a], self.spec, grid) for a in names
                  if a in result.states and np.isfinite(result.states[a].signal)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _partial(response):
    """``AssetParams`` stand-in during fit, before the position scale exists."""
    return AssetParams(pair_scale=np.array([]), response=response, pos_scale=np.nan,
                       class_weight=1.0, liquidity=1.0)


def walk_forward(spec: CTASpec | str, data: CTAData, start, end, *, refit: str = "W-FRI") -> pd.DataFrame:
    """The history as it would have been produced live: fit on each ``refit`` date (data up
    to that date only; that date's row comes from that fit), predict every observation
    until the next fit, stitch. Past
    changes are recomputed on the stitched positions. No forecasts (cheap to add per fit
    date if wanted)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    idx = data.levels.index
    fit_dates = [d for d in pd.date_range(start, end, freq=refit)]
    if not fit_dates or fit_dates[0] > start:
        fit_dates = [start] + fit_dates
    frames = []
    for i, d in enumerate(fit_dates):
        model = CTAModel(spec).fit(data, as_of=d)
        at_fit = model.predict(forecast_at="none").frame
        at_fit = at_fit[at_fit["timestamp"] == at_fit.groupby("asset")["timestamp"].transform("max")]
        # until the next fit, which then produces its own date's row
        window = idx[(idx > d) & ((idx < fit_dates[i + 1]) if i + 1 < len(fit_dates) else (idx <= end))]
        new = data.levels.loc[window]
        seg = [at_fit[at_fit["timestamp"] >= start]]
        if len(window):
            sub = CTAData(levels=new, returns=data.returns.loc[window], assets=data.assets,
                          class_weights=data.class_weights)
            seg.append(model.predict(sub, forecast_at="none").frame)
        frames.append(pd.concat(seg, ignore_index=True))
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["timestamp", "asset"], keep="first")
    out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)].sort_values(["asset", "timestamp"])
    horizons = CTAModel(spec).spec.change_horizons
    out = out.drop(columns=[c for c in out.columns if c.startswith("chg_")])
    chg = out.groupby("asset", group_keys=False)["position"].apply(lambda p: _changes(p, horizons))
    return out.join(chg).reset_index(drop=True)
