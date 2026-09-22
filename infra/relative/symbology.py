"""Parsing of relative ticker notation: ``<root>.<kind>.<rank>``.

kind: ``c`` = calendar (Nth contract by expiry), ``v`` = volume (Nth by trailing average
daily volume - see infra.config.VOLUME_LOOKBACK_DAYS).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RELATIVE_RE = re.compile(r"^([A-Z0-9]+)\.([cv])\.(\d+)$")


@dataclass(frozen=True)
class RelativeSpec:
    root: str
    kind: str
    rank: int

    @property
    def label(self) -> str:
        return f"{self.root}.{self.kind}.{self.rank}"


def parse_relative(ticker: str) -> RelativeSpec | None:
    """``RelativeSpec`` for ``SR3.c.0``-style tickers, ``None`` for anything else."""
    match = RELATIVE_RE.match(ticker)
    if not match:
        return None
    return RelativeSpec(match.group(1), match.group(2), int(match.group(3)))


def is_relative(ticker: str) -> bool:
    return RELATIVE_RE.match(ticker) is not None
