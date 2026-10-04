"""Transaction-cost components and the composite :class:`TransactionCostModel`.

Conventions
-----------
QuantProof's reference simulator works with *portfolio weights* (fraction of
equity). A trade at bar ``t`` changes the weight by ``Δw_t``. Every cost component
returns the cost at each bar **as a fraction of equity**, so net returns are
``gross_t - cost_t``.

Notional-dependent components (per-share and fixed-per-order commissions, size-based
slippage, impact) assume a constant notional ``capital``: traded value is
``|Δw_t| * capital``. This ignores compounding of position size and is documented
as an approximation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofInputError


@dataclass
class CostContext:
    """Inputs available to cost components, one entry per bar.

    ``volatility`` and ``adv`` must be *causal* estimates (known before the trade).
    """

    trades: np.ndarray  # signed weight change Δw executed at each bar
    prices: np.ndarray  # execution price at each bar
    capital: float
    timestamps: pd.DatetimeIndex
    volatility: np.ndarray | None = None  # per-bar return std (fraction)
    adv: np.ndarray | None = None  # average volume in shares per bar

    @property
    def traded_fraction(self) -> np.ndarray:
        return np.abs(np.nan_to_num(self.trades))

    @property
    def shares(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self.traded_fraction * self.capital / self.prices
        return np.nan_to_num(out, nan=0.0, posinf=0.0)


class CostComponent(ABC):
    """A single cost source. Subclasses implement :meth:`cost`."""

    name: str = "cost"

    @abstractmethod
    def cost(self, ctx: CostContext) -> np.ndarray:
        """Cost at each bar as a fraction of equity (non-negative)."""

    def describe(self) -> dict[str, Any]:
        """Parameters for reports and manifests."""
        return {
            "component": type(self).__name__,
            **{k: v for k, v in vars(self).items() if not k.startswith("_")},
        }


def _check_non_negative(value: float, name: str) -> None:
    if value < 0 or not np.isfinite(value):
        raise QuantProofInputError(f"{name} must be a finite non-negative number, got {value}.")


class BpsCommission(CostComponent):
    """Commission in basis points of traded notional."""

    name = "commission"

    def __init__(self, bps: float) -> None:
        _check_non_negative(bps, "bps")
        self.bps = float(bps)

    def cost(self, ctx: CostContext) -> np.ndarray:
        return ctx.traded_fraction * self.bps / 1e4


class PercentageCommission(CostComponent):
    """Commission as a fraction of traded notional (``rate=0.001`` = 0.1 %)."""

    name = "commission"

    def __init__(self, rate: float) -> None:
        _check_non_negative(rate, "rate")
        self.rate = float(rate)

    def cost(self, ctx: CostContext) -> np.ndarray:
        return ctx.traded_fraction * self.rate


class PerShareCommission(CostComponent):
    """Commission per share traded, with an optional minimum per order."""

    name = "commission"

    def __init__(self, per_share: float, min_per_order: float = 0.0) -> None:
        _check_non_negative(per_share, "per_share")
        _check_non_negative(min_per_order, "min_per_order")
        self.per_share = float(per_share)
        self.min_per_order = float(min_per_order)

    def cost(self, ctx: CostContext) -> np.ndarray:
        fee = ctx.shares * self.per_share
        traded = ctx.traded_fraction > 0
        fee = np.where(traded, np.maximum(fee, self.min_per_order), 0.0)
        return fee / ctx.capital


class FixedPerOrderCommission(CostComponent):
    """Fixed fee for every bar with a non-zero trade."""

    name = "commission"

    def __init__(self, fee: float) -> None:
        _check_non_negative(fee, "fee")
        self.fee = float(fee)

    def cost(self, ctx: CostContext) -> np.ndarray:
        return np.where(ctx.traded_fraction > 0, self.fee / ctx.capital, 0.0)


class ScaledCost(CostComponent):
    """A component whose cost is multiplied by ``multiplier`` (for sensitivity analysis)."""

    def __init__(self, inner: CostComponent, multiplier: float) -> None:
        _check_non_negative(multiplier, "multiplier")
        self.inner = inner
        self.multiplier = float(multiplier)
        self.name = inner.name

    def cost(self, ctx: CostContext) -> np.ndarray:
        return self.multiplier * np.asarray(self.inner.cost(ctx), dtype=float)

    def describe(self) -> dict[str, Any]:
        return {**self.inner.describe(), "multiplier": self.multiplier}


class TransactionTax(CostComponent):
    """Transaction tax / stamp duty in bps of traded notional.

    ``side`` restricts the tax to purchases (``"buy"``: weight increases) or sales
    (``"sell"``: weight decreases); ``"both"`` taxes every trade. For short positions the
    side is defined by the sign of the weight change, which is a simplification.
    """

    name = "taxes"

    def __init__(self, bps: float, side: str = "both") -> None:
        _check_non_negative(bps, "bps")
        if side not in ("both", "buy", "sell"):
            raise QuantProofInputError(f"side must be 'both', 'buy' or 'sell', got {side!r}.")
        self.bps = float(bps)
        self.side = side

    def cost(self, ctx: CostContext) -> np.ndarray:
        trades = np.nan_to_num(ctx.trades)
        if self.side == "buy":
            traded = np.clip(trades, 0.0, None)
        elif self.side == "sell":
            traded = np.clip(-trades, 0.0, None)
        else:
            traded = np.abs(trades)
        return traded * self.bps / 1e4


COMPONENT_ORDER = ("commission", "spread", "slippage", "impact", "taxes", "statutory")


@dataclass
class TransactionCostModel:
    """Composite cost model: the sum of its components.

    Example
    -------
    >>> from quantproof.execution import TransactionCostModel
    >>> model = TransactionCostModel.from_bps(commission_bps=1, spread_bps=2, slippage_bps=2)
    """

    components: list[CostComponent] = field(default_factory=list)

    @classmethod
    def from_bps(
        cls,
        commission_bps: float = 0.0,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
        impact_coefficient: float = 0.0,
        tax_bps: float = 0.0,
    ) -> TransactionCostModel:
        """Common model: commission + half spread + fixed slippage (+ optional impact, taxes)."""
        from quantproof.execution.impact import SquareRootImpact
        from quantproof.execution.slippage import FixedSlippage
        from quantproof.execution.spread import FixedSpread

        comps: list[CostComponent] = []
        if commission_bps:
            comps.append(BpsCommission(commission_bps))
        if spread_bps:
            comps.append(FixedSpread(spread_bps))
        if slippage_bps:
            comps.append(FixedSlippage(slippage_bps))
        if impact_coefficient:
            comps.append(SquareRootImpact(impact_coefficient))
        if tax_bps:
            comps.append(TransactionTax(tax_bps))
        return cls(comps)

    @classmethod
    def compose(
        cls,
        *,
        commission: CostComponent | None = None,
        spread: CostComponent | None = None,
        slippage: CostComponent | None = None,
        impact: CostComponent | None = None,
        taxes: CostComponent | None = None,
    ) -> TransactionCostModel:
        """Explicit composition; ``None`` disables a component.

        >>> from quantproof.execution import BpsCommission, FixedSpread
        >>> TransactionCostModel.compose(commission=BpsCommission(1), spread=FixedSpread(2)).names
        ['commission', 'spread']
        """
        return cls([c for c in (commission, spread, slippage, impact, taxes) if c is not None])

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.components]

    def without(self, *names: str) -> TransactionCostModel:
        """Copy with the named components removed (e.g. ``model.without("impact")``)."""
        return TransactionCostModel([c for c in self.components if c.name not in set(names)])

    def scaled(self, multiplier: float) -> TransactionCostModel:
        """Copy with every component's cost multiplied by ``multiplier``."""
        return TransactionCostModel([ScaledCost(c, multiplier) for c in self.components])

    @property
    def is_zero(self) -> bool:
        return not self.components

    def breakdown(self, ctx: CostContext) -> pd.DataFrame:
        """Per-component costs (fraction of equity) indexed by timestamp."""
        data: dict[str, np.ndarray] = {}
        for comp in self.components:
            values = np.asarray(comp.cost(ctx), dtype=float)
            if values.shape != ctx.trades.shape:
                raise QuantProofInputError(f"{type(comp).__name__} returned a mis-shaped array.")
            if np.any(values < -1e-15):
                raise QuantProofInputError(f"{type(comp).__name__} produced negative costs.")
            key = comp.name
            data[key] = data[key] + values if key in data else values
        if not data:
            data["none"] = np.zeros_like(ctx.trades, dtype=float)
        return pd.DataFrame(data, index=ctx.timestamps)

    def total(self, ctx: CostContext) -> np.ndarray:
        """Total cost per bar (fraction of equity)."""
        return self.breakdown(ctx).sum(axis=1).to_numpy()

    def describe(self) -> list[dict[str, Any]]:
        return [c.describe() for c in self.components]
