"""JSON serialization of audit results (strict JSON: NaN/inf become null)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quantproof._utils import to_jsonable

if TYPE_CHECKING:
    from quantproof.results import AuditResult

SCHEMA_VERSION = "1.0"


def result_to_dict(result: AuditResult) -> dict[str, Any]:
    """Self-describing dictionary: no QuantProof object is needed to interpret it."""
    s = result.sections
    out = {
        "schema_version": SCHEMA_VERSION,
        "quantproof_version": result.quantproof_version,
        "overall_status": result.status.value,
        "summary": {
            "passed": result.summary.passed,
            "warnings": result.summary.warnings,
            "failures": result.summary.failures,
            "info": result.summary.info,
            "verdict_rules": result.summary.verdict_rules,
            "verdict_reasons": result.summary.verdict_reasons,
            "narrative": result.narrative.model_dump(mode="json"),
        },
        "overview": s.get("overview", {}),
        "findings": [f.model_dump(mode="json") for f in result.findings],
        "data": s.get("data", {}),
        "static": s.get("static", {}),
        "causality": s.get("causality", {}),
        "leakage": s.get("leakage", {}),
        "statistics": s.get("statistics", {}),
        "validation": s.get("validation", {}),
        "execution": s.get("execution", {}),
        "regimes": s.get("regimes", {}),
        "sensitivity": s.get("sensitivity", {}),
        "charts": s.get("charts", {}),
        "reproducibility": result.manifest,
        "limitations": result.limitations,
    }
    extra = {k: v for k, v in s.items() if k not in out}
    if extra:
        out["extra"] = extra
    return to_jsonable(out)


def render_json(result: AuditResult, indent: int = 2) -> str:
    """JSON string of :func:`result_to_dict`."""
    return json.dumps(result_to_dict(result), indent=indent, ensure_ascii=False, allow_nan=False)


def load_result(path: str | Path) -> AuditResult:
    """Rebuild an :class:`AuditResult` from a JSON report written by QuantProof."""
    from quantproof.results import AuditResult, AuditSummary, Finding, Narrative
    from quantproof.severity import Severity

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "overall_status" not in data or "findings" not in data:
        raise ValueError(f"{path} is not a QuantProof JSON report.")
    keys = (
        "overview",
        "data",
        "static",
        "causality",
        "leakage",
        "statistics",
        "validation",
        "execution",
        "regimes",
        "sensitivity",
        "charts",
    )
    sections = {k: data.get(k, {}) for k in keys if data.get(k)}
    sections.update(data.get("extra", {}))
    summary = data["summary"]
    return AuditResult(
        status=Severity(data["overall_status"]),
        summary=AuditSummary(
            failures=summary.get("failures", 0),
            warnings=summary.get("warnings", 0),
            passed=summary.get("passed", 0),
            info=summary.get("info", 0),
            verdict_rules=summary.get("verdict_rules", []),
            verdict_reasons=summary.get("verdict_reasons", []),
            narrative=Narrative.model_validate(summary["narrative"])
            if summary.get("narrative")
            else None,
        ),
        findings=[Finding.model_validate(f) for f in data["findings"]],
        sections=sections,
        manifest=data.get("reproducibility", {}),
        quantproof_version=data.get("quantproof_version", "unknown"),
        limitations=data.get("limitations", []),
    )
