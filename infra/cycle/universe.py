"""The daily cycle's contract universe: for every enabled root in ``DAILY_BACKFILL``, the
``n_contracts`` nearest unexpired contracts on its expiry cycle - evaluated PER DAY
(point-in-time), so a multi-year backfill follows the curve as contracts expire and roll.

Deliberately a thin wrapper: "the N nearest contracts" is exactly relative ranks
``c.0 .. c.N-1``, so this reuses infra.pipeline.relative.needed_contract_windows (and
through it infra.relative.rolls.calendar_mapping) rather than a second ranking rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from infra.config import DAILY_BACKFILL, DEFINITION_SNAPSHOT_DAYS, FUTURES_ROOTS, MAX_COST_USD, DailyBackfillSpec
from infra.cycle.paths import CyclePaths
from infra.pipeline import contracts as contracts_pipe
from infra.pipeline.relative import needed_contract_windows
from infra.relative.symbology import RelativeSpec

_ONE_DAY = pd.Timedelta(days=1)
_GRID_EPOCH = pd.Timestamp("2000-01-01")


@dataclass(frozen=True)
class UniverseMember:
    ticker: str
    root: str
    dataset: str
    first: pd.Timestamp  # first day it is in the universe (inclusive)
    last: pd.Timestamp  # last day it is in the universe (inclusive) - NOT its expiry: a
    #                     contract still alive at the window's end has last == window end
    activation: pd.Timestamp | None  # listing date, from the definitions snapshot
    expiry: pd.Timestamp | None = None  # real expiry (contracts table) - orders the curve

    @property
    def curve_order(self) -> pd.Timestamp:
        """Where the contract sits on its root's curve. Its expiry, never ``last``: on a
        short window every live contract shares the same ``last`` (the window end), which
        would make "nearest contracts" and ranks arbitrary."""
        return self.expiry if self.expiry is not None else self.last

    def expected_on(self, day: pd.Timestamp) -> bool:
        """In the universe on ``day`` AND already listed then."""
        listed = self.activation is None or pd.isna(self.activation) or self.activation.normalize() <= day
        return self.first <= day <= self.last and listed


def snapshot_grid_floor(day: pd.Timestamp, step_days: int = DEFINITION_SNAPSHOT_DAYS) -> pd.Timestamp:
    """Align definition snapshots to a fixed grid rather than to each run's start date:
    the scheduled run's window moves every day, and snapshot days anchored to its start
    would pull a brand-new (paid) snapshot per root per day. On a grid, consecutive runs
    reuse the same snapshot until the grid advances - one per root per ``step_days``."""
    n = (pd.Timestamp(day).normalize() - _GRID_EPOCH).days // step_days
    return _GRID_EPOCH + pd.Timedelta(days=n * step_days)


def daily_universe(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    paths: CyclePaths,
    specs: dict[str, DailyBackfillSpec] = DAILY_BACKFILL,
    refresh_contracts: bool = True,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> tuple[dict[str, UniverseMember], dict[str, str]]:
    """``({ticker: member}, {root: error})`` over ``[start, end]`` (inclusive). A root that
    fails (e.g. a definitions snapshot before its dataset's history begins) is reported
    in the second dict rather than aborting every other root."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    members: dict[str, UniverseMember] = {}
    errors: dict[str, str] = {}
    for root, spec in specs.items():
        if not spec.enabled or spec.n_contracts <= 0:
            continue
        cfg = FUTURES_ROOTS[root]
        try:
            contracts = contracts_pipe.ensure_contracts(
                cfg, snapshot_grid_floor(start), end + _ONE_DAY, fetch_missing=refresh_contracts,
                contracts_file=paths.contracts_file, coverage_file=paths.defs_coverage,
                max_cost_usd=max_cost_usd, client=client,
            )
        except Exception as exc:
            errors[root] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            continue
        if contracts.empty:
            errors[root] = "no contracts known for this root"
            continue
        ranks = [RelativeSpec(root, "c", r) for r in range(spec.n_contracts)]
        windows, _ = needed_contract_windows(ranks, cfg, contracts, start, end + _ONE_DAY)
        meta = contracts.drop_duplicates("ticker").set_index("ticker")
        for ticker, (w0, w1) in windows.items():
            act, exp = meta["activation"].get(ticker), meta["expiry"].get(ticker)
            members[ticker] = UniverseMember(
                ticker, root, cfg.dataset, pd.Timestamp(w0), pd.Timestamp(w1) - _ONE_DAY,
                None if act is None or pd.isna(act) else pd.Timestamp(act),
                None if exp is None or pd.isna(exp) else pd.Timestamp(exp),
            )
    return members, errors


def rank_on(ticker: str, day: pd.Timestamp, members: dict[str, UniverseMember]) -> int:
    """The contract's rank on its root's curve that day (0 = front): its position, by
    expiry, among the root's universe members alive that day - i.e. its relative
    calendar rank c.N, which is exactly how the universe was built."""
    m = members[ticker]
    alive = sorted((o for o in members.values() if o.root == m.root and o.first <= day <= o.last),
                   key=lambda o: (o.curve_order, o.ticker))
    return [o.ticker for o in alive].index(ticker)
