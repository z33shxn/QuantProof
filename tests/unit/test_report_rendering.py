"""Pathological report rendering, the Python result API, narrative and fingerprints."""

from __future__ import annotations

import json
import math
import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from quantproof import AuditConfig, audit
from quantproof.reports import render
from quantproof.results import AuditResult, Category, Finding, Location, Usage
from quantproof.severity import Confidence, Severity


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.thead_th_without_scope = 0
        self._in_thead = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append((tag, a))
        if tag == "thead":
            self._in_thead = True
        if tag == "th" and self._in_thead and a.get("scope") != "col":
            self.thead_th_without_scope += 1

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False


def _check_html(html: str) -> _Collector:
    parser = _Collector()
    parser.feed(html)
    tags = [t for t, _ in parser.tags]
    assert tags.count("h1") == 1
    assert "footer" in tags and "nav" in tags
    assert parser.thead_th_without_scope == 0
    for tag, attrs in parser.tags:
        # Offline: no external scripts, stylesheets, images or fonts.
        for key in ("src", "href"):
            value = attrs.get(key) or ""
            assert not value.startswith(("http://", "https://", "//")), (tag, value)
        assert not (tag == "link" and attrs.get("rel") == "stylesheet")
    embedded = re.search(
        r'<script type="application/json" id="qp-result">(.*?)</script>', html, re.S
    )
    assert embedded is not None
    json.loads(embedded.group(1))
    return parser


def _finding(i: int, sev: Severity = Severity.WARN, message: str | None = None) -> Finding:
    return Finding(
        id="QP001" if i % 2 else "QP-STAT-002",
        category=Category.STATIC if i % 2 else Category.STATISTICS,
        severity=sev,
        title=f"Finding {i}",
        message=message or f"Message {i}.",
        location=Location(file="strategy.py", line=i + 1) if i % 2 else None,
        confidence=Confidence.MEDIUM,
        usage=Usage.LIVE_DECISION if i % 2 else None,
    )


def _all_formats(result: AuditResult) -> dict[str, str]:
    out = {fmt: render(result, fmt) for fmt in ("html", "markdown", "text", "json")}
    _check_html(out["html"])
    json.loads(out["json"])
    return out


def test_empty_result_renders():
    out = _all_formats(AuditResult.from_findings([]))
    assert "No executed check produced a WARN or FAIL" in out["text"]
    assert "PASS" in out["html"]


def test_single_finding_renders_all_explanations():
    result = AuditResult.from_findings([_finding(1, Severity.FAIL)])
    out = _all_formats(result)
    f = result.findings[0]
    assert f.why_it_matters and f.impact and f.investigate and f.recommendation
    for label in ("Why it matters", "Potential impact", "How to investigate", "Recommendation"):
        assert label in out["html"] and label in out["markdown"]
    assert "FORBIDDEN IN LIVE DECISION" in out["html"]
    assert result.narrative.primary_reason.startswith("FAIL QP001")


def test_hundreds_of_findings_render():
    findings = [_finding(i) for i in range(400)]
    out = _all_formats(AuditResult.from_findings(findings))
    assert out["html"].count('class="finding WARN"') == 400
    assert "and 394 more WARN/FAIL" in out["markdown"]


def test_long_and_hostile_messages_are_escaped():
    hostile = "<script>alert('x')</script> " + "very long message " * 800 + "</script><!--"
    out = _all_formats(AuditResult.from_findings([_finding(0, message=hostile)]))
    assert "<script>alert" not in out["html"]
    assert "&lt;script&gt;alert" in out["html"]
    assert out["html"].count("</script>") == out["html"].count("<script")


def test_nan_and_missing_sections_render():
    sections = {
        "overview": {"strategy": "x", "headline": {"label": "h", "sharpe": math.nan}},
        "statistics": {
            "sharpe": {"sharpe_annualized": math.nan, "n_observations": 0},
            "psr": {"psr": math.inf},
            "dsr": {"dsr": None},
        },
        "validation": {"pbo": {"error": "not enough data"}},
        "execution": {},
    }
    out = _all_formats(AuditResult.from_findings([_finding(2)], sections=sections))
    assert "n/a" in out["html"]
    assert "NaN" not in out["json"] and "Infinity" not in out["json"]


@pytest.mark.parametrize("n_charts", [0, 1, 3])
def test_chart_counts(n_charts):
    idx = pd.date_range("2024-01-01", periods=50).strftime("%Y-%m-%d").tolist()
    keys = ["naive_equity", "realistic_equity", "asset_equity"][:n_charts]
    charts = {k: {"x": idx, "y": list(np.linspace(1, 1.2, 50))} for k in keys}
    out = _all_formats(
        AuditResult.from_findings([], sections={"overview": {"strategy": "x"}, "charts": charts})
    )
    assert out["html"].count('<svg class="qp-chart') == (1 if n_charts else 0)
    if n_charts:
        assert out["html"].count('class="legend"') == 1


def test_result_api_and_round_trip(tmp_path):
    from quantproof.data.synthetic import generate_prices
    from quantproof.reports import load_result, write_report

    prices = generate_prices(400, seed=11)

    def strat(d, lookback=10):
        return np.sign(d["close"].pct_change(lookback)).fillna(0.0)

    res = audit(strat, prices, config=AuditConfig(profile="quick"))
    for prop in (
        "statistics",
        "validation",
        "execution",
        "causality",
        "metrics",
        "reproducibility",
    ):
        assert isinstance(getattr(res, prop), dict) and getattr(res, prop), prop
    assert "dsr" in res.statistics and "cost_attribution" in res.execution
    assert set(res.metrics) == {"headline", "realistic"}
    assert res.to_html().startswith("<!doctype html>")
    assert res.to_markdown().startswith("# QuantProof audit")
    path = write_report(res, tmp_path / "r.json")
    back = load_result(path)
    assert back.narrative == res.narrative
    assert back.status is res.status


def test_fingerprints_are_separate_and_deterministic():
    from quantproof.data.synthetic import generate_prices

    prices = generate_prices(300, seed=2)

    def strat(d):
        return np.sign(d["close"].pct_change(5)).fillna(0.0)

    cfg = AuditConfig(profile="quick")
    a = audit(strat, prices, config=cfg).manifest
    b = audit(strat, prices, config=cfg).manifest
    assert a["fingerprints"] == b["fingerprints"]
    assert a["content_hash"] == b["content_hash"]
    assert set(a["fingerprints"]) == {"code", "data", "config", "parameters", "environment"}

    c = audit(strat, prices, config=AuditConfig(profile="quick", seed=7)).manifest
    changed = {k for k in a["fingerprints"] if a["fingerprints"][k] != c["fingerprints"][k]}
    assert changed == {"config"}

    d = audit(strat, generate_prices(300, seed=3), config=cfg).manifest
    changed = {k for k in a["fingerprints"] if a["fingerprints"][k] != d["fingerprints"][k]}
    assert changed == {"data"}


@pytest.fixture(scope="module")
def lookahead_result():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return audit(
        root / "examples/lookahead_strategy/strategy.py",
        root / "examples/data/prices.parquet",
        config=AuditConfig(profile="quick"),
    )


def test_narrative_puts_lookahead_first(lookahead_result):
    n = lookahead_result.narrative
    assert n.primary_reason.startswith("FAIL")
    assert any(k in n.primary_reason for k in ("QP-CAUSAL", "QP0"))
    assert any("Fix look-ahead first" in s for s in n.next_steps)
