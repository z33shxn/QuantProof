"""Severity levels and the explicit verdict rules.

The overall verdict is *not* a score. It is derived from individual findings by
three ordered rules (see :func:`determine_verdict`):

1. ``FAIL`` if at least one finding has severity ``FAIL``.
2. ``WARN`` if no finding is ``FAIL`` but at least one is ``WARN``.
3. ``PASS`` otherwise (only ``PASS`` and ``INFO`` findings).

Analyzers decide the severity of each finding using thresholds that are recorded
in the finding's ``evidence``, so a reader can always trace why the verdict
occurred.
"""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    """Severity of a single finding (and of the overall verdict)."""

    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"

    @property
    def rank(self) -> int:
        """Ordering used for sorting: PASS < INFO < WARN < FAIL."""
        return _RANK[self]

    @property
    def symbol(self) -> str:
        """Single-character marker used by text reports."""
        return _SYMBOL[self]

    def __str__(self) -> str:
        return self.value


_RANK = {Severity.PASS: 0, Severity.INFO: 1, Severity.WARN: 2, Severity.FAIL: 3}
_SYMBOL = {Severity.PASS: "✓", Severity.INFO: "·", Severity.WARN: "⚠", Severity.FAIL: "✗"}


class Confidence(str, Enum):
    """How certain an analyzer is that a finding reflects a real problem.

    ``HIGH``   the pattern is unambiguous (e.g. ``rolling(center=True)``).
    ``MEDIUM`` the pattern is usually a problem but context can make it valid.
    ``LOW``    a heuristic signal that deserves a human look.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def __str__(self) -> str:
        return self.value


VERDICT_RULES: tuple[str, ...] = (
    "FAIL if at least one finding has severity FAIL (a critical research-validity violation).",
    "WARN if no finding is FAIL and at least one finding has severity WARN.",
    "PASS if every executed check produced only PASS or INFO findings.",
)
