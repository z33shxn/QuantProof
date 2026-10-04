"""Realistic research example: cross-sectional momentum on a 10-symbol universe.

Data: ``examples/data/universe.parquet`` (long format: timestamp, symbol, OHLCV).
QuantProof turns it into a ``(timestamp, symbol)`` panel; the strategy returns a wide
DataFrame of portfolio weights (timestamps x symbols).

Rule: rank symbols by their trailing return from ``t - lookback`` to ``t - skip`` (the
most recent ``skip`` days are excluded, a common short-term-reversal guard), go long the
top ``n_side`` and short the bottom ``n_side`` with equal weights (gross exposure 1),
and rebalance every ``rebalance`` bars, holding weights in between.

Everything uses closes up to and including bar t; the weights decided at t are traded
one bar later under the declared execution assumptions.
"""

import pandas as pd

NAME = "cross_sectional_momentum"
PARAMETERS = {"lookback": 120, "skip": 5, "n_side": 2, "rebalance": 5}
PARAM_GRID = {
    "lookback": [60, 120, 250],
    "skip": [0, 5, 20],
    "n_side": [2, 3],
    "rebalance": [5, 20],
}
EXECUTION = {
    "signal_lag": 1,
    "fill": "close",
    "commission_bps": 1.0,
    "spread_bps": 4.0,
    "slippage_bps": 2.0,
}


def generate_signals(
    data: pd.DataFrame, lookback: int = 120, skip: int = 5, n_side: int = 2, rebalance: int = 5
) -> pd.DataFrame:
    close = data["close"].unstack("symbol")
    past = close.shift(skip)
    momentum = past / past.shift(lookback - skip) - 1.0
    ranks = momentum.rank(axis=1, method="first")
    n = ranks.notna().sum(axis=1)
    long = ranks.gt(n - n_side, axis=0)
    short = ranks.le(n_side, axis=0)
    weights = (long.astype(float) - short.astype(float)) / (2.0 * n_side)
    weights = weights.where(momentum.notna())
    # Rebalance on a fixed schedule; hold the last decided weights in between.
    schedule = pd.Series(range(len(weights)), index=weights.index) % rebalance == 0
    return weights.where(schedule, axis=0).ffill()
