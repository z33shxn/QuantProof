"""Effective-dated, jurisdiction-specific cost providers.

Statutory charges (transaction taxes, exchange fees, stamp duty, regulator fees,
indirect taxes on fees) change over time and differ by instrument, exchange,
broker and side. A :class:`CostProvider` therefore stores a list of
:class:`FeeSchedule` objects, each valid over ``[effective_from, effective_to]``,
and looks up the schedule in force on each trade date. Dates outside the
provider's coverage raise an error rather than silently applying the wrong rates.

The core package never hard-codes jurisdiction-specific rates; providers live in
this sub-package and carry their sources.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofInputError
from quantproof.execution.costs import CostComponent, CostContext


@dataclass(frozen=True)
class FeeSchedule:
    """Rates (as fractions of traded value) in force over a date range.

    ``tax_on_fees`` is applied to ``brokerage + exchange + regulatory`` charges (e.g. GST).
    """

    segment: str
    effective_from: dt.date
    effective_to: dt.date | None
    buy_tax: float = 0.0
    sell_tax: float = 0.0
    exchange_fee: float = 0.0
    regulatory_fee: float = 0.0
    stamp_duty_buy: float = 0.0
    tax_on_fees: float = 0.0
    notes: str = ""

    def covers(self, day: dt.date) -> bool:
        return self.effective_from <= day and (
            self.effective_to is None or day <= self.effective_to
        )


@dataclass(frozen=True)
class CostBreakdown:
    """Cost of a single trade in currency units."""

    transaction_tax: float
    exchange_fee: float
    regulatory_fee: float
    stamp_duty: float
    brokerage: float
    tax_on_fees: float

    @property
    def total(self) -> float:
        return float(sum(asdict(self).values()))

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["total"] = self.total
        return d


@dataclass
class CostProvider:
    """A named set of effective-dated fee schedules."""

    name: str
    jurisdiction: str
    schedules: list[FeeSchedule]
    sources: list[str] = field(default_factory=list)
    last_reviewed: str = ""
    disclaimer: str = ""

    def segments(self) -> list[str]:
        return sorted({s.segment for s in self.schedules})

    def schedule_for(self, segment: str, day: dt.date | pd.Timestamp | str) -> FeeSchedule:
        d = pd.Timestamp(day).date()
        matches = [s for s in self.schedules if s.segment == segment and s.covers(d)]
        if not matches:
            known = self.segments()
            if segment not in known:
                raise QuantProofInputError(
                    f"Unknown segment {segment!r} for provider {self.name}; known: {known}."
                )
            first = min(s.effective_from for s in self.schedules if s.segment == segment)
            raise QuantProofInputError(
                f"No {self.name} schedule for segment {segment!r} on {d}. Coverage starts "
                f"{first}; supply your own FeeSchedule for earlier dates."
            )
        if len(matches) > 1:
            raise QuantProofInputError(f"Overlapping schedules for {segment!r} on {d}.")
        return matches[0]

    def trade_cost(
        self,
        value: float,
        side: str,
        segment: str,
        day: dt.date | pd.Timestamp | str,
        *,
        brokerage: float = 0.0,
    ) -> CostBreakdown:
        """Charges for one trade of traded ``value`` (premium value for options)."""
        if side not in {"buy", "sell"}:
            raise QuantProofInputError("side must be 'buy' or 'sell'.")
        if value < 0:
            raise QuantProofInputError("value must be non-negative.")
        s = self.schedule_for(segment, day)
        tax = value * (s.buy_tax if side == "buy" else s.sell_tax)
        exch = value * s.exchange_fee
        reg = value * s.regulatory_fee
        stamp = value * s.stamp_duty_buy if side == "buy" else 0.0
        indirect = (brokerage + exch + reg) * s.tax_on_fees
        return CostBreakdown(tax, exch, reg, stamp, brokerage, indirect)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "jurisdiction": self.jurisdiction,
            "segments": self.segments(),
            "sources": self.sources,
            "last_reviewed": self.last_reviewed,
        }


class ProviderCost(CostComponent):
    """Cost component applying a :class:`CostProvider` bar by bar (effective-dated).

    Buys are positive weight changes and sells negative. ``brokerage_bps`` models the
    broker's own fee (which varies by broker and is not part of statutory schedules);
    ``brokerage_cap`` optionally caps it per order in currency units (e.g. "0.03% or
    ₹20, whichever is lower" → ``brokerage_bps=3, brokerage_cap=20``).
    """

    name = "statutory"

    def __init__(
        self,
        provider: CostProvider,
        segment: str,
        *,
        brokerage_bps: float = 0.0,
        brokerage_cap: float | None = None,
    ) -> None:
        self.provider = provider
        self.segment = segment
        self.brokerage_bps = float(brokerage_bps)
        self.brokerage_cap = brokerage_cap

    def cost(self, ctx: CostContext) -> np.ndarray:
        out = np.zeros(len(ctx.trades))
        for i, (trade, ts) in enumerate(zip(ctx.trades, ctx.timestamps, strict=True)):
            if not np.isfinite(trade) or trade == 0:
                continue
            value = abs(trade) * ctx.capital
            brokerage = value * self.brokerage_bps / 1e4
            if self.brokerage_cap is not None:
                brokerage = min(brokerage, self.brokerage_cap)
            side = "buy" if trade > 0 else "sell"
            out[i] = (
                self.provider.trade_cost(value, side, self.segment, ts, brokerage=brokerage).total
                / ctx.capital
            )
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "component": "ProviderCost",
            "segment": self.segment,
            "brokerage_bps": self.brokerage_bps,
            "brokerage_cap": self.brokerage_cap,
            **self.provider.describe(),
        }
