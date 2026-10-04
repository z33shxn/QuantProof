"""Structured result model shared by every analyzer.

All models are Pydantic models so they serialize to plain JSON without any
QuantProof object being required to read the output.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantproof._utils import to_jsonable
from quantproof.severity import VERDICT_RULES, Confidence, Severity


class Category:
    """Well-known finding categories.

    Categories are plain strings so third-party analyzers can introduce new
    ones without changing this module.
    """

    DATA = "data"
    STATIC = "static"
    CAUSALITY = "causality"
    LEAKAGE = "leakage"
    EXECUTION = "execution"
    STATISTICS = "statistics"
    VALIDATION = "validation"
    REGIME = "regime"
    SENSITIVITY = "sensitivity"
    REPRODUCIBILITY = "reproducibility"

    ORDER: tuple[str, ...] = (
        DATA,
        STATIC,
        CAUSALITY,
        LEAKAGE,
        EXECUTION,
        STATISTICS,
        VALIDATION,
        REGIME,
        SENSITIVITY,
        REPRODUCIBILITY,
    )


class Location(BaseModel):
    """Where a finding was observed (source file position or data range)."""

    model_config = ConfigDict(frozen=True)

    file: str | None = None
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    snippet: str | None = None
    start: str | None = Field(default=None, description="First affected timestamp, if any.")
    end: str | None = Field(default=None, description="Last affected timestamp, if any.")

    def describe(self) -> str:
        """Compact human-readable location."""
        if self.file:
            out = self.file
            if self.line is not None:
                out += f":{self.line}"
                if self.column is not None:
                    out += f":{self.column}"
            return out
        if self.start or self.end:
            return f"{self.start or '?'} → {self.end or '?'}"
        return ""


class Finding(BaseModel):
    """A single piece of audit evidence."""

    id: str = Field(description="Stable rule identifier, e.g. 'QP001' or 'QP-DATA-004'.")
    category: str
    severity: Severity
    title: str
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    location: Location | None = None
    why_it_matters: str | None = None
    recommendation: str | None = None
    confidence: Confidence | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def _jsonable_evidence(cls, value: Any) -> Any:
        return to_jsonable(value) if value is not None else {}

    @property
    def is_issue(self) -> bool:
        """True for WARN and FAIL findings."""
        return self.severity in (Severity.WARN, Severity.FAIL)


class AuditSummary(BaseModel):
    """Counts by severity, plus the explicit reasons behind the verdict."""

    failures: int = 0
    warnings: int = 0
    passed: int = 0
    info: int = 0
    verdict_rules: list[str] = Field(default_factory=lambda: list(VERDICT_RULES))
    verdict_reasons: list[str] = Field(default_factory=list)


def determine_verdict(findings: list[Finding]) -> tuple[Severity, AuditSummary]:
    """Apply the documented verdict rules to a list of findings."""
    counts = dict.fromkeys(Severity, 0)
    for f in findings:
        counts[f.severity] += 1

    if counts[Severity.FAIL]:
        status = Severity.FAIL
        drivers = [f for f in findings if f.severity is Severity.FAIL]
    elif counts[Severity.WARN]:
        status = Severity.WARN
        drivers = [f for f in findings if f.severity is Severity.WARN]
    else:
        status = Severity.PASS
        drivers = []

    reasons = [f"{f.severity.value} {f.id}: {f.title}" for f in drivers]
    if status is Severity.PASS:
        reasons = ["No WARN or FAIL findings were produced by the executed checks."]
    summary = AuditSummary(
        failures=counts[Severity.FAIL],
        warnings=counts[Severity.WARN],
        passed=counts[Severity.PASS],
        info=counts[Severity.INFO],
        verdict_reasons=reasons,
    )
    return status, summary


class AuditResult(BaseModel):
    """Complete audit output.

    ``sections`` holds the machine-readable evidence produced by each analyzer
    (statistics, validation, execution, causality, regimes, sensitivity, data);
    ``manifest`` holds the reproducibility manifest.
    """

    status: Severity
    summary: AuditSummary
    findings: list[Finding]
    sections: dict[str, Any] = Field(default_factory=dict)
    manifest: dict[str, Any] = Field(default_factory=dict)
    quantproof_version: str
    limitations: list[str] = Field(default_factory=list)

    @field_validator("sections", "manifest", mode="before")
    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        return to_jsonable(value) if value is not None else {}

    @classmethod
    def from_findings(
        cls,
        findings: list[Finding],
        *,
        sections: dict[str, Any] | None = None,
        manifest: dict[str, Any] | None = None,
        limitations: list[str] | None = None,
    ) -> AuditResult:
        """Build a result, computing the verdict from ``findings``."""
        from quantproof._version import __version__

        ordered = sort_findings(findings)
        status, summary = determine_verdict(ordered)
        return cls(
            status=status,
            summary=summary,
            findings=ordered,
            sections=sections or {},
            manifest=manifest or {},
            quantproof_version=__version__,
            limitations=limitations or [],
        )

    @property
    def issues(self) -> list[Finding]:
        """Only WARN and FAIL findings."""
        return [f for f in self.findings if f.is_issue]

    def by_category(self, category: str) -> list[Finding]:
        """Findings belonging to one category."""
        return [f for f in self.findings if f.category == category]

    def ids(self, *severities: Severity) -> set[str]:
        """Rule ids present, optionally filtered by severity."""
        wanted = set(severities) if severities else set(Severity)
        return {f.id for f in self.findings if f.severity in wanted}

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON dictionary representation."""
        from quantproof.reports.json import result_to_dict

        return result_to_dict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        from quantproof.reports.json import render_json

        return render_json(self, indent=indent)


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Stable ordering: by category order, then rule id, then location."""
    cat_index = {c: i for i, c in enumerate(Category.ORDER)}

    def key(f: Finding) -> tuple[int, str, str, int]:
        loc = f.location
        return (
            cat_index.get(f.category, len(cat_index)),
            f.id,
            (loc.file or loc.start or "") if loc else "",
            (loc.line or 0) if loc else 0,
        )

    return sorted(findings, key=key)
