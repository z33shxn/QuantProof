"""Slippage models (execution price worse than the reference price)."""

from __future__ import annotations

import numpy as np

from quantproof.errors import QuantProofInputError
from quantproof.execution.costs import CostComponent, CostContext, _check_non_negative


class FixedSlippage(CostComponent):
    """Constant slippage in bps of traded notional."""

    name = "slippage"

    def __init__(self, bps: float) -> None:
        _check_non_negative(bps, "bps")
        self.bps = float(bps)

    def cost(self, ctx: CostContext) -> np.ndarray:
        return ctx.traded_fraction * self.bps / 1e4


class VolatilitySlippage(CostComponent):
    """Slippage proportional to recent volatility: ``k * sigma_t`` per unit traded.

    ``sigma_t`` is the causal per-bar return volatility provided by the simulator
    (trailing window, lagged one bar).
    """

    name = "slippage"

    def __init__(self, k: float = 0.1) -> None:
        _check_non_negative(k, "k")
        self.k = float(k)

    def cost(self, ctx: CostContext) -> np.ndarray:
        if ctx.volatility is None:
            raise QuantProofInputError("VolatilitySlippage requires per-bar volatility estimates.")
        return ctx.traded_fraction * self.k * np.nan_to_num(ctx.volatility)


class SizeSlippage(CostComponent):
    """Slippage linear in participation: ``coef_bps * (shares / ADV)`` bps per unit traded.

    Requires average volume (``ADV``) per bar. Linear participation models are crude;
    prefer :class:`~quantproof.execution.impact.SquareRootImpact` for large orders.
    """

    name = "slippage"

    def __init__(self, coef_bps: float = 10.0) -> None:
        _check_non_negative(coef_bps, "coef_bps")
        self.coef_bps = float(coef_bps)

    def cost(self, ctx: CostContext) -> np.ndarray:
        if ctx.adv is None:
            raise QuantProofInputError("SizeSlippage requires average volume (a 'volume' column).")
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(ctx.adv > 0, ctx.shares / ctx.adv, 0.0)
        return ctx.traded_fraction * self.coef_bps * np.nan_to_num(participation) / 1e4
