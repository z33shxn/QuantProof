"""Statistical diagnostics: Sharpe, PSR, DSR, PBO, bootstrap, data-snooping tests."""

from quantproof.statistics.bootstrap import (
    BootstrapResult,
    bootstrap_indices,
    bootstrap_sharpe,
    bootstrap_statistic,
    optimal_block_length,
)
from quantproof.statistics.deflated_sharpe import (
    DSRResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from quantproof.statistics.pbo import PBOResult, probability_of_backtest_overfitting
from quantproof.statistics.probabilistic_sharpe import (
    PSRResult,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)
from quantproof.statistics.reality_check import RealityCheckResult, adjust_pvalues, reality_check
from quantproof.statistics.sharpe import (
    annualized_return,
    max_drawdown,
    return_moments,
    sharpe_ratio,
    sharpe_standard_error,
    sharpe_summary,
)

__all__ = [
    "BootstrapResult",
    "DSRResult",
    "PBOResult",
    "PSRResult",
    "RealityCheckResult",
    "adjust_pvalues",
    "annualized_return",
    "bootstrap_indices",
    "bootstrap_sharpe",
    "bootstrap_statistic",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "max_drawdown",
    "minimum_track_record_length",
    "optimal_block_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "reality_check",
    "return_moments",
    "sharpe_ratio",
    "sharpe_standard_error",
    "sharpe_summary",
]
