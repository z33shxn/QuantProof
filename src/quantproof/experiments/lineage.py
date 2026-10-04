"""Lightweight lineage records: which inputs produced which outputs, step by step."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LineageStep:
    """One processing step with input/output content hashes."""

    step: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Lineage:
    """Ordered list of :class:`LineageStep` objects."""

    steps: list[LineageStep] = field(default_factory=list)

    def record(
        self,
        step: str,
        *,
        inputs: dict[str, str] | None = None,
        outputs: dict[str, str] | None = None,
        parameters: dict[str, Any] | None = None,
        notes: list[str] | None = None,
    ) -> LineageStep:
        entry = LineageStep(step, inputs or {}, outputs or {}, parameters or {}, notes or [])
        self.steps.append(entry)
        return entry

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.steps]
