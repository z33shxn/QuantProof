"""Structured result model shared by every analyzer.

All models are Pydantic models so they serialize to plain JSON without any
QuantProof object being required to read the output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class Usage(str, Enum):
    """How future-dated information is used, when a finding concerns timing."""

    LIVE_DECISION = "live-decision"
    LABEL_OR_ANALYSIS = "label-or-analysis"

    @property
    def label(self) -> str:
        return {
            Usage.LIVE_DECISION: "FORBIDDEN IN LIVE DECISION",
            Usage.LABEL_OR_ANALYSIS: "LEGITIMATE FOR LABEL / ANALYSIS",
        }[self]


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
    impact: str | None = Field(default=None, description="Potential impact if the issue is real.")
    investigate: str | None = Field(default=None, description="How to investigate the finding.")
    confidence: Confidence | None = Field(
        default=None,
        description="Analysis confidence that the matched pattern means what the rule says "
        "(not a statistical probability).",
    )
    usage: Usage | None = Field(
        default=None,
        description="Where future information is used: 'live-decision' (forbidden in a live "
        "decision) or 'label-or-analysis' (legitimate for labels or offline analysis).",
    )

    @model_validator(mode="after")
    def _fill_from_registry(self) -> Finding:
        """Default why / impact / investigate / recommendation text from the rule registry.

        Every WARN and FAIL finding therefore answers: what happened (``message``), why it
        matters, its potential impact, how to investigate, and what to do.
        """
        if self.is_issue:
            from quantproof.rules import REGISTRY

            spec = REGISTRY.get(self.id)
            if spec is not None:
                defaults = {
                    "why_it_matters": spec.rationale,
                    "impact": spec.impact,
                    "investigate": spec.investigate,
                    "recommendation": spec.remediation,
                }
                for name, text in defaults.items():
                    if getattr(self, name) is None and text:
                        object.__setattr__(self, name, text)
        return self

    @field_validator("evidence", mode="before")
    @classmethod
    def _jsonable_evidence(cls, value: Any) -> Any:
        return to_jsonable(value) if value is not None else {}

    @property
    def is_issue(self) -> bool:
        """True for WARN and FAIL findings."""
        return self.severity in (Severity.WARN, Severity.FAIL)


class Narrative(BaseModel):
    """Evidence-first explanation of the verdict (no scores)."""

    primary_reason: str
    supporting_evidence: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class AuditSummary(BaseModel):
    """Counts by severity, plus the explicit reasons behind the verdict."""

    failures: int = 0
    warnings: int = 0
    passed: int = 0
    info: int = 0
    verdict_rules: list[str] = Field(default_factory=lambda: list(VERDICT_RULES))
    verdict_reasons: list[str] = Field(default_factory=list)
    narrative: Narrative | None = None


# Issues in these categories invalidate everything computed downstream of the signals.
_UPSTREAM = ("causality", "static", "leakage", "data")
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2, None: 3}


def _first_sentence(text: str, limit: int = 220) -> str:
    head = text.split(". ")[0].strip()
    head = head if head.endswith(".") else head + "."
    return head if len(head) <= limit else head[: limit - 1] + "…"


def _issue_rank(f: Finding) -> tuple[int, int, int, int]:
    sev = 0 if f.severity is Severity.FAIL else 1
    live = 0 if f.usage is Usage.LIVE_DECISION else 1
    upstream = _UPSTREAM.index(f.category) if f.category in _UPSTREAM else len(_UPSTREAM)
    conf = _CONFIDENCE_RANK.get(f.confidence.value if f.confidence else None, 3)
    return (sev, live, upstream, conf)


def build_narrative(status: Severity, findings: list[Finding]) -> Narrative:
    """Primary reason, supporting evidence, recommendations and next steps for a verdict."""
    issues = sorted((f for f in findings if f.is_issue), key=_issue_rank)
    if not issues:
        ran = sorted({f.category for f in findings if f.severity is Severity.PASS})
        return Narrative(
            primary_reason="No executed check produced a WARN or FAIL finding.",
            supporting_evidence=[f"Checks passed in: {', '.join(ran)}."] if ran else [],
            recommendations=[],
            next_steps=[
                "A PASS is not evidence of profitability: confirm on data the research never "
                "touched (paper trading or a later period).",
                "Review the limitations section; checks that did not run are not covered.",
            ],
        )
    top = issues[0]
    primary = f"{top.severity.value} {top.id} — {top.title}: {_first_sentence(top.message)}"
    evidence = [
        f"{f.severity.value} {f.id} — {f.title}: {_first_sentence(f.message)}" for f in issues[1:6]
    ]
    if len(issues) > 6:
        evidence.append(f"… and {len(issues) - 6} more WARN/FAIL finding(s).")
    recs: list[str] = []
    for f in issues:
        if f.recommendation and f.recommendation not in recs:
            recs.append(f.recommendation)
    steps: list[str] = []
    if any(f.category in ("causality", "static") and f.severity is Severity.FAIL for f in issues):
        steps.append(
            "Fix look-ahead first: while signals use future data, every performance, cost and "
            "overfitting statistic in this report is computed on contaminated returns."
        )
    if any(f.category == "data" for f in issues):
        steps.append("Resolve data-quality issues and re-run; they affect every later check.")
    if any(f.category == "execution" for f in issues):
        steps.append("Re-run with lagged fills and realistic costs and report those numbers.")
    if any(f.category in ("validation", "statistics") for f in issues):
        steps.append(
            "Declare every variant tried (statistics.trials or PARAM_GRID) and judge the "
            "strategy on out-of-sample evidence, not the in-sample headline."
        )
    steps.append("Re-run the audit after each fix; the verdict is recomputed from scratch.")
    return Narrative(
        primary_reason=primary,
        supporting_evidence=evidence,
        recommendations=recs[:8],
        next_steps=steps,
    )


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
        narrative=build_narrative(status, findings),
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

    @property
    def narrative(self) -> Narrative:
        """Primary reason, supporting evidence, recommendations and next steps."""
        return self.summary.narrative or build_narrative(self.status, self.findings)

    @property
    def statistics(self) -> dict[str, Any]:
        """Sharpe, PSR, DSR (with trial provenance), bootstrap CI, MinTRL."""
        return dict(self.sections.get("statistics", {}))

    @property
    def validation(self) -> dict[str, Any]:
        """Walk-forward / OOS evidence, PBO, CPCV and data-snooping results."""
        return dict(self.sections.get("validation", {}))

    @property
    def execution(self) -> dict[str, Any]:
        """Execution scenarios, cost attribution, cost multipliers and turnover."""
        return dict(self.sections.get("execution", {}))

    @property
    def causality(self) -> dict[str, Any]:
        """Future-perturbation test report."""
        return dict(self.sections.get("causality", {}))

    @property
    def metrics(self) -> dict[str, Any]:
        """Headline metrics: naive/supplied (``headline``) and audited (``realistic``)."""
        ov = self.sections.get("overview", {})
        return {k: ov[k] for k in ("headline", "realistic") if k in ov}

    @property
    def reproducibility(self) -> dict[str, Any]:
        """The reproducibility manifest (fingerprints, configuration, environment)."""
        return dict(self.manifest)

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

    def to_markdown(self) -> str:
        """Markdown report."""
        from quantproof.reports.markdown import render_markdown

        return render_markdown(self)

    def to_html(self) -> str:
        """Self-contained HTML report (no external resources)."""
        from quantproof.reports.html import render_html

        return render_html(self)


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
