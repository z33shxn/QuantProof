"""Square-root market-impact approximation.

Impact per unit traded is modelled as ``Y * sigma * sqrt(Q / V)`` where ``sigma`` is
the per-bar volatility, ``Q`` the traded quantity and ``V`` the average volume per
bar; ``Y`` is of order one in empirical studies (Torre, 1997; Almgren et al., 2005;
Tóth et al., 2011). The cost as a fraction of equity is
``|Δw| * Y * sigma * sqrt(Q / V)``.

This is a *coarse approximation*: it ignores order slicing, temporary vs permanent
impact, intraday liquidity patterns and impact decay. It is meant to show whether
a strategy's capacity is plausibly sensitive to size, not to predict fills.
"""

from __future__ import annotations

import numpy as np

from quantproof.errors import QuantProofInputError
from quantproof.execution.costs import CostComponent, CostContext, _check_non_negative


class SquareRootImpact(CostComponent):
    """Square-root impact cost component."""

    name = "impact"

    def __init__(self, coefficient: float = 1.0) -> None:
        _check_non_negative(coefficient, "coefficient")
        self.coefficient = float(coefficient)

    def cost(self, ctx: CostContext) -> np.ndarray:
        if ctx.volatility is None or ctx.adv is None:
            raise QuantProofInputError(
                "SquareRootImpact needs volatility and average volume; provide a 'volume' column."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(ctx.adv > 0, ctx.shares / ctx.adv, 0.0)
        participation = np.nan_to_num(participation, nan=0.0, posinf=0.0)
        vol = np.nan_to_num(ctx.volatility)
        return ctx.traded_fraction * self.coefficient * vol * np.sqrt(participation)
