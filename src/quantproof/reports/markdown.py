"""Markdown report for pull requests, issues and research notes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantproof._utils import format_number
from quantproof.audit.models import Category
from quantproof.audit.severity import Severity
from quantproof.experiments.manifest import manifest_to_yaml
from quantproof.reports.text import SECTION_TITLES

if TYPE_CHECKING:
    from quantproof.audit.models import AuditResult

_BADGE = {
    Severity.FAIL: "🔴 FAIL",
    Severity.WARN: "🟠 WARN",
    Severity.PASS: "🟢 PASS",
    Severity.INFO: "⚪ INFO",
}


def _esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _pct(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:
        return "n/a"
    return "0.0%" if f == 0 else f"{f:+.1%}"


def render_markdown(result: AuditResult, *, include_passes: bool = True) -> str:
    """Render a Markdown audit report."""
    ov = result.sections.get("overview", {})
    s = result.summary
    out = [
        f"# QuantProof audit — {_BADGE[result.status]}",
        "",
        f"**Strategy:** `{ov.get('strategy', 'n/a')}` · **Observations:** {ov.get('n_observations', 'n/a')}"
        + (
            f" · **Window:** {ov.get('evaluation_start')} → {ov.get('evaluation_end')}"
            if ov.get("evaluation_start")
            else ""
        ),
        "",
        "| Critical failures | Warnings | Passed checks | Info |",
        "|---:|---:|---:|---:|",
        f"| {s.failures} | {s.warnings} | {s.passed} | {s.info} |",
        "",
        "**Why this verdict:**",
        "",
    ]
    out += [f"- {_esc(r)}" for r in s.verdict_reasons]
    if "headline" in ov:
        out += [
            "",
            "## Headline vs audited performance",
            "",
            "| Scenario | Sharpe | CAGR | Max drawdown |",
            "|---|---:|---:|---:|",
        ]
        for key in ("headline", "realistic"):
            if key in ov:
                b = ov[key]
                out.append(
                    f"| {_esc(b.get('label', key))} | {format_number(b.get('sharpe'), 2)} | "
                    f"{_pct(b.get('cagr'))} | {_pct(b.get('max_drawdown'))} |"
                )
    out += ["", "## Findings", ""]
    for cat, title in SECTION_TITLES.items():
        items = [f for f in result.findings if f.category == cat]
        if not include_passes:
            items = [f for f in items if f.is_issue]
        if not items:
            continue
        out += [
            f"### {title.title()}",
            "",
            "| Status | Rule | Finding | Location | Detail |",
            "|---|---|---|---|---|",
        ]
        for f in items:
            loc = f.location.describe() if f.location else ""
            out.append(
                f"| {_BADGE[f.severity]} | `{f.id}` | {_esc(f.title)} | {_esc(loc)} | {_esc(f.message)} |"
            )
        out.append("")
    issues = [f for f in result.findings if f.is_issue]
    if issues:
        out += ["## Recommendations", ""]
        for f in issues:
            if f.recommendation:
                out.append(f"- **{f.id}** — {_esc(f.recommendation)}")
        out.append("")
    stats = result.sections.get("statistics", {})
    if stats:
        dsr = stats.get("dsr", {})
        psr = stats.get("psr", {})
        out += [
            "## Statistical diagnostics",
            "",
            "| Statistic | Value | Inputs |",
            "|---|---:|---|",
            f"| Sharpe (annualized, net) | {format_number(stats.get('sharpe', {}).get('sharpe_annualized'), 2)} | n={stats.get('sharpe', {}).get('n_observations')} |",
            f"| Probabilistic Sharpe Ratio | {format_number(psr.get('psr'), 3)} | skew={format_number(psr.get('skewness'), 2)}, kurt={format_number(psr.get('kurtosis'), 2)} |",
            f"| Deflated Sharpe Ratio | {format_number(dsr.get('dsr'), 3)} | trials={dsr.get('n_trials')}, E[max SR]={format_number(dsr.get('expected_max_sharpe_annualized'), 2)} |",
        ]
        val = result.sections.get("validation", {})
        if isinstance(val.get("pbo"), dict) and "pbo" in val["pbo"]:
            p = val["pbo"]
            out.append(
                f"| Probability of Backtest Overfitting | {format_number(p['pbo'], 2)} | S={p['n_partitions']}, N={p['n_strategies']}, combinations={p['n_combinations']} |"
            )
        wf = val.get("walk_forward", {})
        if isinstance(wf, dict) and "oos_sharpe" in wf:
            out.append(
                f"| Walk-forward OOS Sharpe | {format_number(wf['oos_sharpe'], 2)} | IS mean={format_number(wf.get('is_sharpe_mean'), 2)} |"
            )
        out.append("")
    sens = result.sections.get("sensitivity", {})
    if sens.get("ascii"):
        out += ["## Parameter surface (Sharpe)", "", "```text", sens["ascii"], "```", ""]
    if result.manifest:
        out += [
            "<details><summary>Reproducibility manifest</summary>",
            "",
            "```yaml",
            manifest_to_yaml(result.manifest).rstrip(),
            "```",
            "",
            "</details>",
            "",
        ]
    out += ["## Limitations", ""] + [f"- {_esc(x)}" for x in result.limitations]
    out += ["", f"_Generated by QuantProof {result.quantproof_version}._", ""]
    return "\n".join(out)


__all__ = ["Category", "render_markdown"]
