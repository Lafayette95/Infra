"""Building blocks of the daily cycle: steps, their checks, and what a run reports.

A ``Step`` wraps one ``backfill_daily_*`` function plus the regression checks that run
right after it. Every check has a severity - ``fail`` (default: the step fails and every
step depending on it is skipped), ``warn`` (logged as a warning, run continues) or
``info`` (logged, run continues).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.config import DEFAULT_REVISION_WINDOW_DAYS
from infra.cycle.paths import CyclePaths

log = logging.getLogger(__name__)


class Severity(str, Enum):
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"


@dataclass
class StepContext:
    """Everything a step and its checks see. ``start``/``end`` are INCLUSIVE days."""
    start: pd.Timestamp
    end: pd.Timestamp
    run_day: pd.Timestamp
    paths: CyclePaths
    force_refetch: bool = False
    reference_vintage: Path | None = None  # latest vintage strictly before run_day
    options: dict = field(default_factory=dict)  # per-run knobs a step may read (client, ...)
    output: dict = field(default_factory=dict)  # what the step returned, for its checks


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: Severity
    message: str
    details: pd.DataFrame | None = None

    @property
    def blocking(self) -> bool:
        return not self.passed and self.severity is Severity.FAIL


# A check function returns (passed, message, details-or-None).
CheckFn = Callable[[StepContext], "tuple[bool, str, pd.DataFrame | None]"]


@dataclass(frozen=True)
class Check:
    name: str
    fn: CheckFn
    severity: Severity = Severity.FAIL

    def run(self, ctx: StepContext) -> CheckResult:
        try:
            passed, message, details = self.fn(ctx)
        except Exception as exc:  # a crashing check is a failed check, never a silent pass
            log.exception("check %s raised", self.name)
            passed, message, details = False, f"check raised {type(exc).__name__}: {exc}", None
        return CheckResult(self.name, passed, self.severity, message, details)


@dataclass(frozen=True)
class Step:
    name: str
    fn: Callable[[StepContext], dict]
    depends_on: tuple[str, ...] = ()
    checks: tuple[Check, ...] = ()
    # business days re-fetched by the SCHEDULED run; per step, global default 3
    revision_window_days: int = DEFAULT_REVISION_WINDOW_DAYS


@dataclass
class StepOutcome:
    name: str
    status: str  # "ok" | "failed" (a blocking check) | "error" (step raised) | "skipped"
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    checks: list[CheckResult] = field(default_factory=list)
    output: dict = field(default_factory=dict)
    error: str | None = None


class CycleFailed(RuntimeError):
    pass


@dataclass
class CycleReport:
    run_day: pd.Timestamp
    outcomes: list[StepOutcome]

    @property
    def ok(self) -> bool:
        return all(o.status == "ok" for o in self.outcomes)

    def outcome(self, name: str) -> StepOutcome:
        return next(o for o in self.outcomes if o.name == name)

    def summary(self) -> str:
        lines = [f"daily cycle, run day {self.run_day.date()}: {'OK' if self.ok else 'FAILED'}"]
        for o in self.outcomes:
            window = f" [{o.start.date()} .. {o.end.date()}]" if o.start is not None else ""
            lines.append(f"  {o.name:10s} {o.status}{window}" + (f" - {o.error}" if o.error else ""))
            for c in o.checks:
                mark = "pass" if c.passed else c.severity.value.upper()
                lines.append(f"      [{mark}] {c.name}: {c.message}")
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise CycleFailed(self.summary())
