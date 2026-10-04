"""Bid/ask spread costs and a simple spread estimator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantproof.execution.costs import CostComponent, CostContext, _check_non_negative


class FixedSpread(CostComponent):
    """Pay half of a constant quoted spread (in bps of price) on every trade."""

    name = "spread"

    def __init__(self, spread_bps: float) -> None:
        _check_non_negative(spread_bps, "spread_bps")
        self.spread_bps = float(spread_bps)

    def cost(self, ctx: CostContext) -> np.ndarray:
        return ctx.traded_fraction * (self.spread_bps / 2.0) / 1e4


class SeriesSpread(CostComponent):
    """Half of a time-varying spread supplied per bar (bps), e.g. from quote data."""

    name = "spread"

    def __init__(self, spread_bps: pd.Series) -> None:
        self.spread_bps = spread_bps

    def cost(self, ctx: CostContext) -> np.ndarray:
        s = self.spread_bps.reindex(ctx.timestamps).ffill().fillna(0.0).to_numpy(dtype=float)
        return ctx.traded_fraction * s / 2.0 / 1e4

    def describe(self) -> dict[str, Any]:
        return {"component": "SeriesSpread", "mean_spread_bps": float(self.spread_bps.mean())}


def roll_spread(prices: Any) -> float:
    """Roll (1984) effective spread estimate, as a fraction of price.

    ``spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1}))`` using log-price changes. Returns 0
    when the first-order autocovariance is non-negative (the estimator is undefined;
    this is common for daily data and does **not** mean the spread is zero).
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size < 4:
        return float("nan")
    dp = np.diff(np.log(p))
    cov = float(np.cov(dp[1:], dp[:-1], ddof=1)[0, 1])
    return 2.0 * math.sqrt(-cov) if cov < 0 else 0.0
