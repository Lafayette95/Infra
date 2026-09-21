"""Options for the Ticker Root and Expiry dropdowns (no Dash imports; unit-testable)."""
from __future__ import annotations

from pathlib import Path

from infra.config import (
    DEFAULT_RELATIVE_RANKS,
    FUTURES_CONTRACTS_FILE,
    FUTURES_DIR,
    FUTURES_ROOTS,
)
from infra.relative.symbology import parse_relative
from infra.storage import contract_store, parquet_store

DEFAULT_KIND_RANK = ("v", 0)  # most active contract: the sensible default for every root


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _relative_label(root: str, kind: str, rank: int) -> str:
    if kind == "c":
        what = f"{_ordinal(rank + 1)} by expiry"
    else:
        what = "most active" if rank == 0 else f"{_ordinal(rank + 1)} most active"
    return f"{root}.{kind}.{rank} · {what}"


def root_options() -> list[dict[str, str]]:
    """One entry per configured root, in config order (STIR first, then bonds)."""
    return [
        {"label": f"{cfg.root} · {cfg.name} ({cfg.category})", "value": cfg.root}
        for cfg in FUTURES_ROOTS.values()
    ]


def expiry_options(
    root: str,
    *,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    futures_dir: Path = FUTURES_DIR,
) -> list[dict[str, str]]:
    """Relative tickers first, then the root's known absolute contracts by expiry.

    Absolute contracts come from the contracts table (fetchable); a trailing ● marks the
    ones that already have bars on disk.
    """
    options = [
        {"label": _relative_label(root, kind, rank), "value": f"{root}.{kind}.{rank}"}
        for kind, rank in DEFAULT_RELATIVE_RANKS
    ]
    contracts = contract_store.read_contracts(contracts_file, root)
    on_disk = set(parquet_store.list_values(futures_dir, "ticker"))
    for ticker, expiry in zip(contracts["ticker"], contracts["expiry"]):
        dot = " ●" if ticker in on_disk else ""
        options.append({"label": f"{ticker} · expires {expiry:%Y-%m-%d}{dot}", "value": ticker})
    return options


def default_expiry(root: str, current: str | None = None) -> str:
    """Expiry value to select after the root changes.

    Keeps the same relative choice (e.g. ``c.1``) when the previous selection was relative,
    otherwise falls back to the most active contract (``v.0``).
    """
    spec = parse_relative(current) if current else None
    kind, rank = (spec.kind, spec.rank) if spec and (spec.kind, spec.rank) in DEFAULT_RELATIVE_RANKS \
        else DEFAULT_KIND_RANK
    return f"{root}.{kind}.{rank}"
