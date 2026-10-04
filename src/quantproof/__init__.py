"""QuantProof — trust your backtest before you trust your strategy.

Research-grade validation and auditing for quantitative trading strategies.

Quick start::

    from quantproof import AuditConfig, audit

    result = audit(strategy="strategy.py", data="prices.parquet", config=AuditConfig())
    print(result.status)
    for finding in result.issues:
        print(finding.id, finding.title)

Sub-packages expose the building blocks directly::

    from quantproof.statistics import deflated_sharpe_ratio
    from quantproof.validation import PurgedKFold, CPCV, WalkForward
    from quantproof.execution import TransactionCostModel
"""

from quantproof._version import __version__
from quantproof.adapters import PandasAdapter, ResearchArtifacts
from quantproof.config import AuditConfig
from quantproof.engine import audit
from quantproof.errors import (
    QuantProofConfigError,
    QuantProofDataError,
    QuantProofError,
    QuantProofInputError,
    QuantProofStrategyError,
)
from quantproof.results import AuditResult, Category, Finding
from quantproof.severity import Confidence, Severity
from quantproof.strategy import StrategySpec, load_strategy

__all__ = [
    "AuditConfig",
    "AuditResult",
    "Category",
    "Confidence",
    "Finding",
    "PandasAdapter",
    "QuantProofConfigError",
    "QuantProofDataError",
    "QuantProofError",
    "QuantProofInputError",
    "QuantProofStrategyError",
    "ResearchArtifacts",
    "Severity",
    "StrategySpec",
    "__version__",
    "audit",
    "load_strategy",
]
