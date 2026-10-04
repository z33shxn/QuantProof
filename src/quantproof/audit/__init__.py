"""Audit engine, result model and severity rules.

``audit`` and ``LIMITATIONS`` are imported lazily: low-level modules (e.g. data
validation) import :mod:`quantproof.audit.models` and must not pull in the engine.
"""

from typing import Any

from quantproof.audit.models import AuditResult, AuditSummary, Category, Finding, Location
from quantproof.audit.severity import VERDICT_RULES, Confidence, Severity


def __getattr__(name: str) -> Any:
    if name in ("audit", "LIMITATIONS"):
        from quantproof.audit import engine

        return getattr(engine, name)
    raise AttributeError(f"module 'quantproof.audit' has no attribute {name!r}")


__all__ = [
    "LIMITATIONS",
    "VERDICT_RULES",
    "AuditResult",
    "AuditSummary",
    "Category",
    "Confidence",
    "Finding",
    "Location",
    "Severity",
    "audit",
]
