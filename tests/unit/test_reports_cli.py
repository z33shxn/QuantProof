"""Report renderers and the command-line interface."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantproof import AuditConfig, audit
from quantproof.cli import app
from quantproof.reports import (
    load_result,
    render,
    render_html,
    render_json,
    render_markdown,
    render_text,
    write_report,
)

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


@pytest.fixture(scope="module")
def lookahead_result():
    cfg = AuditConfig(quick=True)
    return audit(
        ROOT / "examples/lookahead_strategy/strategy.py",
        ROOT / "examples/data/prices.parquet",
        config=cfg,
    )


def test_json_is_strict_and_self_describing(lookahead_result):
    text = render_json(lookahead_result)
    data = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert data["overall_status"] == "FAIL"
    assert set(data["summary"]) >= {
        "passed",
        "warnings",
        "failures",
        "verdict_rules",
        "verdict_reasons",
    }
    for key in (
        "findings",
        "statistics",
        "validation",
        "execution",
        "reproducibility",
        "causality",
    ):
        assert key in data
    f = data["findings"][0]
    assert set(f) >= {
        "id",
        "category",
        "severity",
        "title",
        "message",
        "evidence",
        "location",
        "recommendation",
        "confidence",
    }


def test_json_round_trip(lookahead_result, tmp_path):
    p = write_report(lookahead_result, tmp_path / "r.json")
    back = load_result(p)
    assert back.status == lookahead_result.status
    assert [f.id for f in back.findings] == [f.id for f in lookahead_result.findings]
    assert render_markdown(back) == render_markdown(lookahead_result)
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError, match="not a QuantProof JSON report"):
        load_result(tmp_path / "bad.json")


def test_markdown_and_text(lookahead_result):
    md = render_markdown(lookahead_result)
    assert md.startswith("# QuantProof audit — 🔴 FAIL")
    assert "`QP001`" in md and "Reproducibility manifest" in md and "## Limitations" in md
    txt = render_text(lookahead_result)
    assert "QUANTPROOF AUDIT" in txt and "OVERALL: FAIL" in txt and "✗ QP001" in txt
    sharpe = lookahead_result.sections["overview"]["headline"]["sharpe"]
    assert f"{sharpe:.2f}" in txt  # numbers come from the analysis


def test_html_contains_all_sections_and_embedded_json(lookahead_result):
    html = render_html(lookahead_result)
    for i, title in enumerate(
        [
            "Overall verdict",
            "Executive summary",
            "Data validation and static code findings",
            "Causality test results",
            "Statistical diagnostics",
            "Validation methodology",
            "Execution assumptions",
            "Cost sensitivity",
            "Regime analysis",
            "Parameter sensitivity",
            "Reproducibility manifest",
            "Methodology and limitations",
            "Machine-readable results",
        ],
        start=1,
    ):
        assert f"{i}. {title}" in html
    assert 'id="qp-result"' in html and "<svg" in html
    assert ">nan<" not in html.lower()
    assert "<script src=" not in html  # self-contained


def test_render_dispatch(lookahead_result):
    assert render(lookahead_result, "md").startswith("#")
    with pytest.raises(ValueError, match="Unknown report format"):
        render(lookahead_result, "pdf")


def test_cli_version_and_rules():
    assert runner.invoke(app, ["version"]).stdout.startswith("quantproof ")
    out = runner.invoke(app, ["rules"]).stdout
    assert "QP001" in out and "QP015" in out


def test_cli_audit_exit_codes_and_outputs(tmp_path):
    base = [
        "audit",
        "-s",
        str(ROOT / "examples/lookahead_strategy/strategy.py"),
        "-d",
        str(ROOT / "examples/data/prices.parquet"),
        "--quick",
    ]
    r = runner.invoke(app, base)
    assert r.exit_code == 1 and "OVERALL: FAIL" in r.stdout
    r = runner.invoke(app, [*base, "--fail-on", "never", "-o", str(tmp_path / "a.html")])
    assert r.exit_code == 0 and (tmp_path / "a.html").exists()
    r = runner.invoke(app, [*base, "--format", "markdown", "--fail-on", "never"])
    assert r.stdout.startswith("# QuantProof audit")
    r = runner.invoke(app, [*base, "-q", "-o", str(tmp_path / "a.json")])
    assert (
        r.exit_code == 1
        and json.loads((tmp_path / "a.json").read_text())["overall_status"] == "FAIL"
    )
    r = runner.invoke(app, ["report", str(tmp_path / "a.json"), "-f", "text"])
    assert "OVERALL: FAIL" in r.stdout
    r = runner.invoke(app, ["report", str(tmp_path / "a.json"), "-o", str(tmp_path / "b.html")])
    assert (tmp_path / "b.html").exists()


def test_cli_errors():
    r = runner.invoke(app, ["audit"])
    assert r.exit_code == 2
    r = runner.invoke(app, ["audit", "-s", "missing.py", "-d", "x.csv"])
    assert r.exit_code == 2 and "not found" in r.stderr
    r = runner.invoke(app, ["report", "missing.json"])
    assert r.exit_code == 2


def test_cli_results_mode(tmp_path):
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2020", periods=300, freq="B")
    rng = np.random.default_rng(0)
    pd.DataFrame({"date": idx, "returns": rng.normal(0.0005, 0.01, 300)}).to_csv(
        tmp_path / "r.csv", index=False
    )
    trials = pd.DataFrame(rng.normal(0, 0.01, (300, 6)), columns=[f"v{i}" for i in range(6)])
    trials.insert(0, "date", idx)
    trials.to_csv(tmp_path / "t.csv", index=False)
    r = runner.invoke(
        app,
        [
            "audit",
            "--returns",
            str(tmp_path / "r.csv"),
            "--trials",
            str(tmp_path / "t.csv"),
            "--quick",
            "--fail-on",
            "never",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    assert "Deflated Sharpe Ratio" in r.stdout


def test_cli_validate_scan_statistics(tmp_path):
    r = runner.invoke(app, ["validate", str(ROOT / "examples/data/prices.csv")])
    assert r.exit_code == 0 and "OVERALL: PASS" in r.stdout
    r = runner.invoke(app, ["scan", str(ROOT / "tests/fixtures/broken")])
    assert r.exit_code == 1 and "QP012" in r.stdout
    r = runner.invoke(app, ["scan", str(ROOT / "tests/fixtures/clean"), "--show-passes"])
    assert r.exit_code == 0
    import numpy as np
    import pandas as pd

    pd.DataFrame(
        {
            "date": pd.date_range("2020", periods=500, freq="B"),
            "returns": np.random.default_rng(1).normal(0.0005, 0.01, 500),
        }
    ).to_csv(tmp_path / "r.csv", index=False)
    r = runner.invoke(
        app, ["statistics", str(tmp_path / "r.csv"), "--trials", "20", "--bootstrap", "200"]
    )
    assert r.exit_code == 0 and "Deflated Sharpe Ratio" in r.stdout
    r = runner.invoke(app, ["statistics", str(tmp_path / "r.csv"), "--json", "--bootstrap", "200"])
    assert math.isfinite(json.loads(r.stdout)["sharpe"]["sharpe_annualized"])


def test_cli_init_config_and_generate_data(tmp_path):
    cfg = tmp_path / "q.yaml"
    assert runner.invoke(app, ["init-config", str(cfg)]).exit_code == 0
    assert AuditConfig.from_file(cfg) == AuditConfig()
    assert runner.invoke(app, ["init-config", str(cfg)]).exit_code == 2
    assert (
        runner.invoke(app, ["generate-data", str(tmp_path / "p.csv"), "--n", "50"]).exit_code == 0
    )
    assert (
        runner.invoke(app, ["generate-data", str(tmp_path / "p.parquet"), "--n", "50"]).exit_code
        == 0
    )
    assert runner.invoke(app, ["generate-data", str(tmp_path / "p.xlsx")]).exit_code == 2


def test_html_renders_in_every_audit_mode(tmp_path, prices):
    """Regression: results-mode and static-only audits lack some sections."""
    import numpy as np
    import pandas as pd

    cfg = AuditConfig(quick=True)
    idx = pd.date_range("2020", periods=400, freq="B")
    trials = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (400, 6)), index=idx)
    results_mode = audit(returns=trials.iloc[:, 0], trial_returns=trials, config=cfg)
    p = tmp_path / "s.py"
    p.write_text("def generate_signals(d):\n    return d['close'].rolling(5).mean()\n")
    static_only = audit(p, None, config=cfg)
    for res in (results_mode, static_only):
        html = render_html(res)
        assert "13. Machine-readable results" in html
        assert render_markdown(res).startswith("# QuantProof audit")
        assert "OVERALL" in render_text(res)
