"""The rule registry is the single source of truth for rule ids and documentation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from quantproof.errors import QuantProofInputError
from quantproof.rules import REGISTRY, get_rule, list_rules, rules_markdown
from quantproof.severity import Severity

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "quantproof"


def test_generated_rules_doc_is_up_to_date():
    doc = (ROOT / "docs/api/rules.md").read_text(encoding="utf-8")
    assert doc == rules_markdown(), "run: quantproof rules --markdown > docs/api/rules.md"


def test_every_emitted_rule_id_is_registered():
    pattern = re.compile(
        r"""(?:id\s*=\s*|"id"\s*:\s*|_not_computed\(\s*|_not_run\(\s*)"(QP[-A-Z0-9]+)\""""
    )
    emitted: set[str] = set()
    for path in SRC.rglob("*.py"):
        if path.name == "rules.py" and path.parent == SRC:
            continue
        emitted |= set(pattern.findall(path.read_text(encoding="utf-8")))
    # Rule classes declare their id as a class attribute.
    for path in (SRC / "analyzers").rglob("*.py"):
        emitted |= set(re.findall(r'^\s+id\s*=\s*"(QP[-A-Z0-9]+)"', path.read_text(), re.M))
    assert len(emitted) >= 50, f"pattern found only {len(emitted)} ids"
    unknown = sorted(emitted - set(REGISTRY))
    assert unknown == [], f"emitted but not registered: {unknown}"


def test_registry_entries_are_complete():
    for spec in list_rules():
        assert spec.name and spec.category and spec.severity_policy and spec.description
        assert spec.max_severity in set(Severity)
        if spec.analysis != "meta" and spec.max_severity in (Severity.WARN, Severity.FAIL):
            assert spec.rationale and spec.impact and spec.investigate and spec.remediation, spec.id


def test_get_rule_suggestions():
    assert get_rule(" qp001 ").id == "QP001"
    with pytest.raises(QuantProofInputError, match="Did you mean"):
        get_rule("QP-STAT-03")


def test_static_rule_classes_match_registry():
    from quantproof.analyzers.static import list_rules as static_rules

    # QP000 (unparseable source) is emitted by the analyzer itself, not by a rule class.
    registered = {r.id for r in list_rules() if r.analysis == "static"} - {"QP000"}
    assert {r.id for r in static_rules()} == registered
