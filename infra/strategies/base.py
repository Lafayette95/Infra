"""The strategy layer: model predictions -> raw VIEWS ("signals") -> POSITIONS (root CLAUDE.md 27).

A strategy class per strategy type (``infra.strategies.cevt.CEVT`` ...), named configurations
per class in ``infra/strategies/config/<class>.py``. Everything here is in RELATIVE tickers
(``ZN.v.0``); ``to_absolute`` is the last step (the contract each relative ticker maps to).
Strategies read stored data and model outputs but never write: ``infra.jobs.strategy_runs``
operates them and writes ``Database/Strategies/<name>``.

**Labels are DECISION times** (user decision 2026-10-05): a signal / position labelled T uses
only what is known at T and is held from T to the next grid point. Step P&L is stamped at the
END of its interval, so the two conventions meet in ONE place, ``pnl``: the P&L at label L is
the position decided at L - (1 + lag) steps times the step ending at L (``lag`` = execution
delay in grid steps).

**P&L and costs** are a separate layer on the ABSOLUTE positions: ``infra.strategies.accounting``
(execution checks, executed positions, P&L, costs, checks), operated by
``infra.jobs.strategy_runs.account``. ``pnl`` below is the instrument-level research shift.

**Positions** (shared; a class may override ``positions``): each instrument's view divided by
its ex-ante $ volatility per contract (EWMA of daily changes, ``vol_span`` days, x point value
x sqrt 252, as available at the label - instruments treated as INDEPENDENT for now), then scaled
to ``target_vol_usd`` a year:

* ``scaling="realised"`` (the base default, for always-on strategies): the scale that makes the
  strategy's trailing realised $ P&L volatility hit the target (``realised_window`` days);
* ``scaling="full_strength"`` (sparse strategies, e.g. CEVT): a fixed scale - the target is
  reached when every instrument holds a full-strength (|1|) view; weaker views trade smaller.
  Vol-targeting a mostly-flat strategy's INSTANTANEOUS position would lever any weak view up to
  the full target, erasing the meaning of its size.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

ANNUAL = np.sqrt(252.0)


@dataclass(frozen=True)
class StrategySpec:
    name: str = "custom"
    description: str = ""
    frequency: str | None = None                 # REQUIRED: "intraday" | "daily" (validated with the cycle and
                                                 # the accounting marks; stores live under Strategies/<frequency>)
    instruments: tuple[str, ...] = ()           # relative tickers
    target_vol_usd: float = 1_000_000.0          # annual $ volatility
    vol_span: int = 60                           # EWMA span of daily changes, days
    scaling: str = "realised"                    # realised | full_strength
    realised_window: int = 252                   # days of strategy P&L for "realised"
    realised_min_obs: int = 60
    exec_lag: int = 0                            # grid steps between decision and execution (pnl)
    horizon_days: int = 10                       # the forward plan
    plan_vintages: bool = True
    cycle: str = "DEFAULT_CYCLE"
    accounting: str = "bbo_mid"                  # ACCOUNTING_MODELS (infra/strategies/config/accounting.py)
    layers: str | None = None                    # LAYER_SIZINGS name: views are on curve STRUCTURES, sized per
                                                 # layer and netted into futures (infra.strategies.layered);
                                                 # None = views per future, sized per instrument

    def __post_init__(self):
        from infra.reference.event_grid import resolve_cycle
        from infra.strategies.config.accounting import get_accounting_spec
        if self.frequency not in ("intraday", "daily"):
            raise ValueError(f"strategy {self.name!r}: frequency is required, 'intraday' or 'daily' "
                             f"(got {self.frequency!r})")
        cf = resolve_cycle(self.cycle).frequency
        if cf != self.frequency:
            raise ValueError(f"strategy {self.name!r} is {self.frequency} but its cycle {self.cycle!r} is {cf}")
        marks = get_accounting_spec(self.accounting).marks
        if (marks == "settlement") != (self.frequency == "daily"):
            raise ValueError(f"strategy {self.name!r} is {self.frequency} but its accounting {self.accounting!r} "
                             f"marks with {marks!r} (daily <-> settlement, intraday <-> bbo)")


# Presets: what a frequency implies; a spec states only what differs (root CLAUDE.md 29)
INTRADAY = dict(frequency="intraday")
DAILY = dict(frequency="daily", cycle="DAILY_SETTLE", accounting="settlement")


def strategy_registry(frequency: str, specs) -> dict:
    """A frequency's registry of specs: rejects a spec of the other frequency (the guard)."""
    out = {}
    for s in specs:
        if s.frequency != frequency:
            raise ValueError(f"strategy {s.name!r} is {s.frequency}: it can't go in the {frequency} registry")
        if s.name in out:
            raise ValueError(f"duplicate strategy name {s.name!r}")
        out[s.name] = s
    return out


def pnl(positions: pd.DataFrame, steps: pd.DataFrame, lag: int = 0) -> pd.DataFrame:
    """THE label shift (and the only one): positions labelled by decision time, ``steps`` (P&L
    per contract) stamped at the END of their interval on the same grid -> P&L per label =
    position decided ``1 + lag`` grid points earlier x the step ending at the label."""
    steps = steps.reindex(columns=positions.columns)
    held = positions.reindex(steps.index).shift(1 + lag)
    return held * steps


def instrument_vol_usd(instrument: str, start, end, *, span: int = 60) -> pd.Series:
    """Ex-ante annual $ vol per contract of a relative futures ticker, indexed by WHEN it was
    known (``available_at``: each daily settlement counts from the session close): EWMA std of
    the back-adjusted daily settlement change x point value x sqrt 252."""
    from infra.config import FUTURES_ROOTS
    from infra.pipeline.series_panel import read_available
    from infra.relative.symbology import parse_relative
    rows = read_available(f"fut:{instrument}", pd.Timestamp(start) - pd.Timedelta(days=3 * span), end)
    if rows.empty:
        return pd.Series(dtype="float64")
    pv = FUTURES_ROOTS[parse_relative(instrument).root].point_value
    d = rows["value"].diff()
    vol = d.ewm(span=span, min_periods=max(span // 2, 10)).std() * pv * ANNUAL
    return pd.Series(vol.to_numpy(), index=pd.DatetimeIndex(rows["available_at"])).dropna()


def asof(series: pd.Series, labels: pd.DatetimeIndex) -> np.ndarray:
    """``series`` (indexed by availability) at each label: the latest value known at it."""
    if series.empty:
        return np.full(len(labels), np.nan)
    s = series.sort_index()
    pos = s.index.searchsorted(pd.DatetimeIndex(labels), side="right") - 1
    out = np.full(len(labels), np.nan)
    ok = pos >= 0
    out[ok] = s.to_numpy()[pos[ok]]
    return out


class Strategy(ABC):
    """See the module docstring."""

    def __init__(self, spec: StrategySpec):
        self.spec = spec

    @property
    def instruments(self) -> list[str]:
        return list(self.spec.instruments)

    @abstractmethod
    def views(self, *args, **kwargs) -> pd.DataFrame:
        """The class's raw views per label (the "signals"), wide."""

    def vols(self, labels: pd.DatetimeIndex, vol_series: dict[str, pd.Series] | None = None) -> pd.DataFrame:
        """Ex-ante $ vol per contract of each instrument at each label (point in time)."""
        out = {}
        for inst in self.instruments:
            s = vol_series[inst] if vol_series is not None else instrument_vol_usd(
                inst, labels.min(), labels.max(), span=self.spec.vol_span)
            out[inst] = asof(s, labels)
        return pd.DataFrame(out, index=labels)

    def positions(self, signal: pd.DataFrame, vols: pd.DataFrame, steps: pd.DataFrame | None = None) -> pd.DataFrame:
        """Contracts per instrument and label from the views ``signal`` (columns = instruments)."""
        n = max(len(self.instruments), 1)
        unit = (self.spec.target_vol_usd / np.sqrt(n)) / vols.reindex(columns=signal.columns)
        raw = signal * unit
        if self.spec.scaling == "full_strength":
            return raw
        if self.spec.scaling != "realised":
            raise ValueError(f"scaling {self.spec.scaling!r} (realised | full_strength)")
        if steps is None:
            raise ValueError("scaling='realised' needs the step P&L ($ per contract) to measure realised vol")
        p = pnl(raw, steps, self.spec.exec_lag).sum(axis=1, min_count=1)
        daily = p.groupby(p.index.normalize()).sum(min_count=1)
        realised = daily.rolling(self.spec.realised_window, min_periods=self.spec.realised_min_obs).std() * ANNUAL
        # the scale for day d uses the P&L of days BEFORE d (shifted one row: known from d 00:00)
        scale = (self.spec.target_vol_usd / realised).shift(1).dropna()
        k = asof(scale, signal.index)
        return raw.mul(k, axis=0)

    def size(self, signal: pd.DataFrame, labels, *, vols: dict | None = None, state=None) -> pd.DataFrame:
        """Contracts per relative future from the views: per instrument (``positions``) or, with
        ``spec.layers``, per structure layer, netted (``infra.strategies.layered``)."""
        if self.spec.layers:
            from infra.strategies.layered import layered_positions
            contracts, self.sizing_diagnostics_ = layered_positions(signal, self.spec.layers, state=state)
            return contracts
        from infra.relative.symbology import parse_relative
        untradeable = [i for i in self.instruments if parse_relative(i) is None]
        if untradeable and vols is None:            # vols passed in = the caller defines the risk unit
            raise NotImplementedError(
                f"strategy {self.spec.name!r}: views on {untradeable} (yields, not futures) need an execution map "
                "into futures - not built yet (TOFIX); use futures or curve structures (layers)")
        return self.positions(signal, self.vols(pd.DatetimeIndex(labels), vols))

    def futures(self) -> list[str]:
        """The relative futures positions are held in."""
        if self.spec.layers:
            from infra.reference.structures import STRUCTURE_SETS
            from infra.strategies.config.layers import get_layer_sizing
            return STRUCTURE_SETS[get_layer_sizing(self.spec.layers).structure_set].legs()
        return self.instruments

    def to_absolute(self, positions: pd.DataFrame, contracts: dict[str, pd.Series]) -> pd.DataFrame:
        """Long ``label, instrument, contract, position``: each relative ticker's position on
        the contract it maps to at that label (``contracts[inst]``: per label)."""
        rows = []
        for inst in positions.columns:
            con = contracts[inst].reindex(positions.index)
            rows.append(pd.DataFrame({"label": positions.index, "instrument": inst,
                                      "contract": con.to_numpy(), "position": positions[inst].to_numpy()}))
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["label", "instrument", "contract", "position"])
        return out.dropna(subset=["position"])


def contracts_at(instrument: str, labels: pd.DatetimeIndex) -> pd.Series:
    """The contract a relative ticker maps to at each label (its CME trading day's), the last
    known mapping carried forward for labels past the stored data (a provisional plan)."""
    from infra.config import FUTURES_ROOTS
    from infra.pipeline.event_pnl import contract_map
    from infra.relative.symbology import parse_relative
    from infra.trading_calendar import trading_day
    labels = pd.DatetimeIndex(labels)
    if labels.empty:
        return pd.Series(dtype="object")
    root = parse_relative(instrument).root
    tday = trading_day(labels, FUTURES_ROOTS[root].dataset)
    m = contract_map(instrument, pd.DatetimeIndex(tday.unique()))
    return pd.Series(m.reindex(tday).to_numpy(), index=labels).ffill()
