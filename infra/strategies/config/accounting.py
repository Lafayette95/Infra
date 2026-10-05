"""Accounting configurations (``infra.strategies.accounting``): how a strategy's absolute
positions become executed trades, P&L and costs. Named, toggled by ``StrategySpec.accounting``."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AccountingSpec:
    name: str
    description: str = ""
    marks: str = "bbo"                  # bbo (bbo-1m mids at each label) | settlement (daily)
    policy: str = "defer"               # defer: a non-executable trade waits for the next executable label
                                        # ignore: trade at the mark regardless (research)
    spread_paid: float = 0.5            # share of the half-spread paid per trade: 1 = always cross,
                                        # 0 = always passively filled; 0.5 = filled passively on half
                                        # the trades (user decision 2026-10-05, to be researched)
    max_quote_age_min: float = 2.0      # bbo: the last quote must be this fresh to trade
    wide_spread_k: float = 4.0          # bbo: spread > k x the contract's trailing median -> no trade
    spread_lookback: int = 500          # labels of fresh two-sided quotes behind that median
    spread_min_obs: int = 20
    cost_fallback_days: int = 20        # settlement: no snap that day -> trailing median half-spread
                                        # (the contract's, else its root's)
    defer_warn_labels: int = 8          # a trade deferred longer than this is a warning
    mark_jump_k: float = 15.0           # a held contract's mark move > k x its trailing median move ...
    mark_jump_reversal: float = 0.5     # ... AND at least this share of it reversed at the next label (a bad
                                        # quote, not news: every unreversed NFP jump was real, 2026-10-05)
    mark_jump_lookback: int = 500


ACCOUNTING_MODELS: dict[str, AccountingSpec] = {s.name: s for s in (
    AccountingSpec("bbo_mid", "intraday: bbo-1m mids, trades only when open, fresh, two-sided and not wide"),
    AccountingSpec("bbo_mid_ignore", "intraday: bbo-1m mids, every trade at the mark (research)", policy="ignore"),
    AccountingSpec("settlement", "daily: trade at the execution day's settlement, NY1500 spread as cost",
                   marks="settlement"),
)}


def get_accounting_spec(spec: AccountingSpec | str, **overrides) -> AccountingSpec:
    base = ACCOUNTING_MODELS[spec] if isinstance(spec, str) else spec
    return replace(base, **overrides) if overrides else base
