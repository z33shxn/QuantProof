"""Audit every example and write HTML/JSON/Markdown reports to examples/reports/.

python examples/run_examples.py            # full configuration
python examples/run_examples.py --quick    # smaller bootstrap/CSCV sizes
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantproof import AuditConfig, audit
from quantproof.reports import write_report

HERE = Path(__file__).resolve().parent
EXAMPLES = {
    "clean_strategy": "prices.parquet",
    "lookahead_strategy": "prices.parquet",
    "leakage_strategy": "prices.parquet",
    "overfit_strategy": "noise.parquet",
    "unrealistic_execution": "prices.parquet",
    "cross_sectional_momentum": "universe.parquet",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = HERE / "reports"
    print(f"{'example':<24}{'verdict':<9}{'naive SR':>9}{'audit SR':>9}  issues")
    for name, data in EXAMPLES.items():
        result = audit(
            HERE / name / "strategy.py",
            HERE / "data" / data,
            config=AuditConfig(profile="quick" if args.quick else "standard"),
        )
        for ext in ("html", "json", "md"):
            write_report(result, out / f"{name}.{ext}")
        ov = result.sections["overview"]
        issues = ", ".join(sorted(f.id for f in result.issues))
        print(
            f"{name:<24}{result.status.value:<9}{ov['headline']['sharpe']:>9.2f}"
            f"{ov['realistic']['sharpe']:>9.2f}  {issues}"
        )
    print(f"\nReports written to {out}")


if __name__ == "__main__":
    main()
