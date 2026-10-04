"""Regime definitions and per-regime performance analysis."""

from quantproof.regimes.analysis import performance_by_regime, regime_analysis
from quantproof.regimes.drawdown import drawdown_regimes, drawdown_series, trend_regimes
from quantproof.regimes.volatility import trailing_volatility, volatility_regimes

__all__ = [
    "drawdown_regimes",
    "drawdown_series",
    "performance_by_regime",
    "regime_analysis",
    "trailing_volatility",
    "trend_regimes",
    "volatility_regimes",
]
