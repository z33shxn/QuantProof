"""Causal signal construction with documented, lagged execution and costs."""

import pandas as pd

EXECUTION = {"signal_lag": 1, "commission_bps": 1.0, "spread_bps": 2.0, "slippage_bps": 1.0}


def generate_signals(data: pd.DataFrame) -> pd.Series:
    close = data["close"]
    fast = close.ewm(span=10).mean()
    slow = close.rolling(50).mean()
    z = (close - close.rolling(20).mean()) / close.rolling(20).std()
    prior = close.shift(1)
    signal = ((fast > slow) & (z < 2) & (close > prior * 0.9)).astype(float)
    return signal.where(slow.notna())
