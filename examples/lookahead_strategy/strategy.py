"""DELIBERATELY BROKEN example: look-ahead bias.

Two classic mistakes:
1. "Tomorrow's return" is computed with shift(-1) and used to decide today's position.
2. The trend filter uses a centered rolling window, which averages future prices.

The backtest looks spectacular. QuantProof should flag QP001, QP002 and fail the
future-perturbation test.
"""

import pandas as pd

NAME = "lookahead_momentum"
PARAMETERS = {"window": 10}
EXECUTION = {"signal_lag": 1, "commission_bps": 1.0, "spread_bps": 2.0, "slippage_bps": 2.0}


def generate_signals(data: pd.DataFrame, window: int = 10) -> pd.Series:
    close = data["close"]
    data["next_return"] = close.pct_change().shift(-1)  # BUG: tomorrow's return
    trend = close.rolling(window=window, center=True).mean()  # BUG: centered window
    signal = (data["next_return"] > 0).astype(float)
    signal = signal.where(close > trend * 0.98, 0.0)
    return signal
