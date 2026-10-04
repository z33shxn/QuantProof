"""Plain-text (terminal) rendering of an audit result."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantproof._utils import format_number
from quantproof.results import Category
from quantproof.severity import Severity

if TYPE_CHECKING:
    from quantproof.results import AuditResult

SECTION_TITLES = {
    Category.DATA: "DATA VALIDATION",
    Category.STATIC: "STATIC ANALYSIS",
    Category.CAUSALITY: "CAUSALITY",
    Category.LEAKAGE: "LEAKAGE",
    Category.EXECUTION: "EXECUTION",
    Category.STATISTICS: "STATISTICS",
    Category.VALIDATION: "VALIDATION",
    Category.REGIME: "REGIMES",
    Category.SENSITIVITY: "PARAMETER SENSITIVITY",
}


def _box(title: str, width: int = 60) -> list[str]:
    inner = width - 2
    return ["╔" + "═" * inner + "╗", "║" + title.center(inner) + "║", "╚" + "═" * inner + "╝"]


def _metric_line(label: str, block: dict[str, Any]) -> str:
    return (
        f"  {label:<28} Sharpe {format_number(block.get('sharpe'), 2):>6}   "
        f"CAGR {_pct(block.get('cagr')):>7}   MaxDD {_pct(block.get('max_drawdown')):>7}"
    )


def _pct(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:
        return "n/a"
    return "0.0%" if f == 0 else f"{f:+.1%}"


def render_text(result: AuditResult, *, verbose: bool = False, color: bool = False) -> str:
    """Render a readable terminal report. Every number comes from ``result``."""

    def paint(text: str, sev: Severity) -> str:
        if not color:
            return text
        codes = {Severity.FAIL: "31", Severity.WARN: "33", Severity.PASS: "32", Severity.INFO: "90"}
        return f"\x1b[{codes[sev]}m{text}\x1b[0m"

    lines = _box("QUANTPROOF AUDIT")
    ov = result.sections.get("overview", {})
    if ov:
        lines.append("")
        lines.append(f"Strategy: {ov.get('strategy')}   Observations: {ov.get('n_observations')}")
        if ov.get("evaluation_start"):
            lines.append(f"Window:   {ov.get('evaluation_start')} → {ov.get('evaluation_end')}")
        if "headline" in ov:
            lines.append("")
            lines.append(_metric_line(ov["headline"].get("label", "Headline")[:28], ov["headline"]))
        if "realistic" in ov:
            lines.append(
                _metric_line(ov["realistic"].get("label", "Realistic")[:28], ov["realistic"])
            )

    order = list(SECTION_TITLES)
    for cat in order:
        items = [f for f in result.findings if f.category == cat]
        if not items:
            continue
        if not verbose:
            shown = [f for f in items if f.severity is not Severity.INFO or f.id.startswith("QP0")]
        else:
            shown = items
        if not shown:
            continue
        lines.append("")
        lines.append(SECTION_TITLES[cat])
        seen_pass: set[str] = set()
        for f in shown:
            if f.severity is Severity.PASS and f.id in seen_pass:
                continue
            seen_pass.add(f.id)
            loc = f" ({f.location.describe()})" if f.location and f.location.line else ""
            usage = f" [{f.usage.label}]" if f.usage is not None and f.is_issue else ""
            lines.append(paint(f"{f.severity.symbol} {f.id:<13} {f.title}{loc}{usage}", f.severity))
            if f.is_issue or verbose:
                msg = f.message if len(f.message) < 160 or verbose else f.message[:157] + "..."
                lines.append(f"    {msg}")
            if verbose and f.is_issue:
                for label, text in (
                    ("Why it matters", f.why_it_matters),
                    ("Potential impact", f.impact),
                    ("How to investigate", f.investigate),
                    ("Recommendation", f.recommendation),
                ):
                    if text:
                        lines.append(f"    {label}: {text}")

    stats = result.sections.get("statistics", {})
    val = result.sections.get("validation", {})
    if stats or val:
        lines.append("")
        lines.append("KEY STATISTICS (realistic net returns)")
        if stats:
            lines.append(
                f"  Sharpe (annualized):          {format_number(stats.get('sharpe', {}).get('sharpe_annualized'), 2)}"
            )
            lines.append(
                f"  Probabilistic Sharpe Ratio:   {format_number(stats.get('psr', {}).get('psr'), 3)}"
            )
            dsr = stats.get("dsr", {})
            lines.append(
                f"  Deflated Sharpe Ratio:        {format_number(dsr.get('dsr'), 3)}  "
                f"(trials: {dsr.get('n_trials')}, {dsr.get('trials_source', 'n/a')})"
            )
        if isinstance(val.get("pbo"), dict) and "pbo" in val["pbo"]:
            lines.append(f"  Prob. of Backtest Overfitting: {format_number(val['pbo']['pbo'], 2)}")
        wf = val.get("walk_forward", {})
        if isinstance(wf, dict) and "oos_sharpe" in wf:
            lines.append(f"  Walk-forward OOS Sharpe:      {format_number(wf['oos_sharpe'], 2)}")

    n = result.narrative
    lines.append("")
    lines.append("WHY THIS VERDICT")
    lines.append(f"  Primary reason: {n.primary_reason}")
    for e in n.supporting_evidence:
        lines.append(f"  - {e}")
    if n.next_steps:
        lines.append("  Next steps:")
        lines += [f"    {i}. {step}" for i, step in enumerate(n.next_steps, 1)]
    lines.append("")
    s = result.summary
    lines.append(
        f"Critical failures: {s.failures}   Warnings: {s.warnings}   Passed checks: {s.passed}"
    )
    lines.append(paint(f"OVERALL: {result.status.value}", result.status))
    return "\n".join(lines)
