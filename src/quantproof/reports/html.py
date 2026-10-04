"""Self-contained HTML report (inline CSS/SVG, no network access needed to view)."""

from __future__ import annotations

import math
from importlib import resources
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, Undefined, select_autoescape

from quantproof.experiments.manifest import manifest_to_yaml
from quantproof.reports.charts import bar_chart, heatmap, histogram, line_chart
from quantproof.reports.json import render_json
from quantproof.results import Category

if TYPE_CHECKING:
    from quantproof.results import AuditResult

SYMBOLS = {"PASS": "✓", "INFO": "·", "WARN": "⚠", "FAIL": "✗"}
SECTION_KEYS = (
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


def _num(v: Any, digits: int = 2) -> str:
    if isinstance(v, Undefined):
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if math.isinf(f):
        return "∞" if f > 0 else "−∞"
    return "n/a" if math.isnan(f) else f"{f:,.{digits}f}"


def _pct(v: Any) -> str:
    if isinstance(v, Undefined):
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(f) else f"{f:+.1%}"


def _pct_plain(v: Any, digits: int = 0) -> str:
    if isinstance(v, Undefined):
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(f) else f"{f:.{digits}%}"


def _env() -> Environment:
    env = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
    env.filters["num"] = _num
    env.filters["pct"] = _pct
    env.filters["pctplain"] = _pct_plain
    return env


def _template() -> str:
    return (resources.files("quantproof.reports") / "templates" / "report.html.j2").read_text(
        encoding="utf-8"
    )


def render_html(result: AuditResult) -> str:
    """Render the full HTML report."""
    # Every section key the template reads exists (empty when that analysis did not run),
    # so static-only and results-mode audits render the same report structure.
    s = {key: {} for key in SECTION_KEYS} | dict(result.sections)
    ov = s.get("overview", {})
    charts = s.get("charts", {})
    series = []
    results_mode = ov.get("mode") == "results"
    for key, name, short, slot in (
        (
            "realistic_equity",
            "Supplied returns" if results_mode else "Audit (lag + costs)",
            "Returns" if results_mode else "Audit",
            1,
        ),
        ("naive_equity", "Naive (same-bar, no costs)", "Naive", 2),
        ("benchmark_equity", "Benchmark", "Bench", 3),
        ("asset_equity", "Buy & hold asset", "B&H", 3),
    ):
        if key in charts and charts[key].get("y"):
            series.append(
                {
                    "name": name,
                    "short": short,
                    "x": charts[key]["x"],
                    "y": charts[key]["y"],
                    "slot": slot,
                }
            )
    equity_chart = line_chart(series, y_label="Growth of 1", chart_id="equity") if series else ""
    ex = s.get("execution", {})
    lag_chart = cost_chart = ""
    if ex.get("lag_sensitivity"):
        rows = ex["lag_sensitivity"]
        lag_chart = bar_chart(
            [str(r["signal_lag"]) for r in rows],
            [r.get("sharpe") for r in rows],
            y_label="Sharpe by lag",
        )
    if ex.get("cost_sensitivity"):
        rows = ex["cost_sensitivity"]
        cost_chart = bar_chart(
            [f"{r['multiplier']:g}x" for r in rows],
            [r.get("sharpe") for r in rows],
            y_label="Net Sharpe by cost multiplier",
        )
    pbo = s.get("validation", {}).get("pbo", {})
    pbo_chart = (
        histogram(pbo.get("logits", []), label="PBO logits") if isinstance(pbo, dict) else ""
    )
    sens = s.get("sensitivity", {})
    heat_chart = ""
    if sens.get("heatmap"):
        hm = sens["heatmap"]
        heat_chart = heatmap(
            hm["x_values"], hm["y_values"], hm["values"], x_label=hm["x"], y_label=hm["y"]
        )
    by_cat: dict[str, list[Any]] = {c: [] for c in Category.ORDER}
    for f in result.findings:
        by_cat.setdefault(f.category, []).append(f)
    issues = [f for f in result.findings if f.is_issue]
    issues.sort(key=lambda f: -f.severity.rank)
    # "<" as \u003c keeps the embedded JSON valid and makes "</script>" / "<!--" impossible.
    result_json = render_json(result, indent=0).replace("<", "\\u003c")
    m = result.manifest or {}
    git = (m.get("experiment") or {}).get("git") or {}
    data = m.get("data") or {}
    footer = {
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "data_hash": data.get("content_sha256")
        or data.get("file_sha256")
        or data.get("returns_sha256")
        or (m.get("fingerprints") or {}).get("data"),
        "config_hash": (m.get("configuration") or {}).get("sha256"),
    }
    return (
        _env()
        .from_string(_template())
        .render(
            version=result.quantproof_version,
            created_at=result.manifest.get("created_at", "")[:19],
            status=result.status.value,
            summary=result.summary,
            narrative=result.narrative,
            footer=footer,
            symbols=SYMBOLS,
            ov=ov,
            sections=s,
            by_cat=by_cat,
            issues=issues,
            equity_chart=equity_chart,
            equity_series=series,
            lag_chart=lag_chart,
            cost_chart=cost_chart,
            pbo_chart=pbo_chart,
            heat_chart=heat_chart,
            manifest=result.manifest,
            manifest_yaml=manifest_to_yaml(result.manifest) if result.manifest else "",
            limitations=result.limitations,
            result_json=result_json,
        )
    )
