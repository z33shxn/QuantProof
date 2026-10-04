"""Proof of value: flawed research -> QuantProof -> fix -> re-run (deterministic).

    python examples/proof_of_value/run.py

Audits the first-draft momentum strategy (flawed_strategy.py) and the corrected one
(../cross_sectional_momentum/strategy.py) on the same synthetic universe, and prints
what changed. Output is deterministic for a given QuantProof/NumPy/pandas version.
"""

from __future__ import annotations

from pathlib import Path

from quantproof import AuditConfig, audit

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "universe.parquet"


def describe(label: str, path: Path) -> dict[str, object]:
    result = audit(path, DATA, config=AuditConfig(profile="quick"))
    m = result.metrics
    print(f"\n=== {label}: {result.status.value} ===")
    print(f"Headline Sharpe (as the researcher would report it): {m['headline']['sharpe']:.2f}")
    print(f"Audited Sharpe (lagged fills, realistic costs):      {m['realistic']['sharpe']:.2f}")
    print(f"Primary reason: {result.narrative.primary_reason}")
    for f in result.issues:
        usage = f" [{f.usage.label}]" if f.usage else ""
        print(f"  {f.severity.value:<4} {f.id:<14} {f.title}{usage}")
    return {"status": result.status.value, "issues": {f.id for f in result.issues}}


def main() -> None:
    before = describe("Before (first draft)", HERE / "flawed_strategy.py")
    after = describe("After (fixed)", HERE.parent / "cross_sectional_momentum" / "strategy.py")
    fixed = sorted(before["issues"] - after["issues"])  # type: ignore[operator]
    remaining = sorted(after["issues"])  # type: ignore[call-overload]
    print("\nResolved by the fix:", ", ".join(fixed) or "none")
    print("Still open after the fix:", ", ".join(remaining) or "none")
    print(
        "\nThe fixed strategy is not 'approved': the remaining warnings say the evidence for "
        "an edge is weak once all 36 variants are accounted for. That is the honest answer."
    )


if __name__ == "__main__":
    main()
