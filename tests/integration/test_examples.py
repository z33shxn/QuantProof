"""End-to-end audits of the deliberately constructed examples.

These tests are the core acceptance criteria: QuantProof must catch every intentional
flaw and must not raise critical findings on the clean example.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantproof import AuditConfig, Severity, audit

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "examples/data/prices.parquet"
NOISE = ROOT / "examples/data/noise.parquet"


def run(name: str, data: Path = PRICES, **cfg_overrides):
    cfg = AuditConfig(**cfg_overrides)
    return audit(ROOT / "examples" / name / "strategy.py", data, config=cfg)


@pytest.fixture(scope="module")
def clean():
    return run("clean_strategy")


def test_clean_strategy_has_no_critical_findings(clean):
    assert clean.status is not Severity.FAIL
    assert clean.ids(Severity.FAIL) == set()
    static_issues = [f for f in clean.by_category("static") if f.is_issue]
    assert static_issues == []
    assert "QP-CAUSAL-001" in clean.ids(Severity.PASS)
    assert "QP-EXEC-001" in clean.ids(Severity.PASS)
    assert "QP-LEAK-003" in clean.ids(Severity.PASS)


def test_lookahead_strategy_is_caught():
    res = run("lookahead_strategy")
    assert res.status is Severity.FAIL
    fails = res.ids(Severity.FAIL)
    assert {"QP001", "QP002", "QP-CAUSAL-001"} <= fails
    assert {"QP-LEAK-003", "QP-STAT-004"} <= res.ids(Severity.WARN)
    headline = res.sections["overview"]["headline"]["sharpe"]
    realistic = res.sections["overview"]["realistic"]["sharpe"]
    assert headline > 4 and realistic < headline


def test_leakage_strategy_is_caught():
    pytest.importorskip("sklearn")
    res = run("leakage_strategy")
    assert res.status is Severity.FAIL
    assert {"QP006", "QP012", "QP-CAUSAL-001"} <= res.ids(Severity.FAIL)
    assert "QP004" in res.ids(Severity.WARN)


def test_overfit_strategy_is_exposed():
    res = run("overfit_strategy", NOISE)
    warns = res.ids(Severity.WARN)
    assert {"QP013", "QP-STAT-003", "QP-VAL-001", "QP-VAL-002"} <= warns
    assert res.ids(Severity.FAIL) == set()  # no validity violation, only weak evidence
    val = res.sections["validation"]
    assert val["pbo"]["pbo"] >= 0.5
    assert val["walk_forward"]["oos_sharpe"] < val["walk_forward"]["is_sharpe_mean"]
    assert res.sections["statistics"]["dsr"]["n_trials"] == 240
    assert res.sections["sensitivity"]["surface"]["classification"] in {"fragile", "moderate"}


def test_unrealistic_execution_is_caught():
    res = run("unrealistic_execution")
    assert res.status is Severity.FAIL
    assert "QP-EXEC-001" in res.ids(Severity.FAIL)
    assert {"QP009", "QP011", "QP-EXEC-002"} <= res.ids(Severity.WARN)
    ex = res.sections["execution"]
    assert ex["headline_retained_under_audit"] < 0.25


UNIVERSE = ROOT / "examples/data/universe.parquet"


def test_cross_sectional_momentum_has_no_validity_failures():
    res = run("cross_sectional_momentum", UNIVERSE, profile="quick")
    assert res.ids(Severity.FAIL) == set()
    assert [f for f in res.by_category("static") if f.is_issue] == []
    assert "QP-CAUSAL-001" in res.ids(Severity.PASS)
    assert res.sections["overview"]["n_symbols"] == 10
    assert res.statistics["trials"]["source"] == "PARAM_GRID size"
    assert res.statistics["dsr"]["n_trials"] == 36


def test_proof_of_value_flaws_are_caught_and_fixed():
    flawed = audit(
        ROOT / "examples/proof_of_value/flawed_strategy.py",
        UNIVERSE,
        config=AuditConfig(profile="quick"),
    )
    assert {"QP002", "QP-CAUSAL-001"} <= flawed.ids(Severity.FAIL)
    assert {"QP009", "QP011"} <= flawed.ids(Severity.WARN)
    assert flawed.narrative.primary_reason.startswith("FAIL")


def test_clean_audit_is_reproducible(clean):
    again = run("clean_strategy")
    assert again.manifest["content_hash"] == clean.manifest["content_hash"]
    a = again.to_dict()
    b = clean.to_dict()
    for d in (a, b):
        d["reproducibility"].pop("created_at")
    assert a == b


def test_csv_and_parquet_give_same_results():
    a = run("clean_strategy", profile="quick")
    b = run("clean_strategy", ROOT / "examples/data/prices.csv", profile="quick")
    sa = a.sections["statistics"]["sharpe"]["sharpe_annualized"]
    sb = b.sections["statistics"]["sharpe"]["sharpe_annualized"]
    assert sa == pytest.approx(sb, rel=1e-4)  # CSV is rounded to 6 decimals
