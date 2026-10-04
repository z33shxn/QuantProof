"""Clean example: dual moving-average trend filter (long/flat).

Every input at bar t uses closes up to and including t (trailing windows only).
Execution assumptions are declared explicitly: fill one bar after the signal, with
commission, spread and slippage. The parameter grid that was considered is declared
so QuantProof can account for selection.
"""

import numpy as np
import pandas as pd

NAME = "dual_ma_trend"
PARAMETERS = {"fast": 20, "slow": 100}
PARAM_GRID = {"fast": [10, 20, 30, 40], "slow": [60, 100, 140, 180]}
EXECUTION = {
    "signal_lag": 1,
    "fill": "close",
    "commission_bps": 1.0,
    "spread_bps": 2.0,
    "slippage_bps": 2.0,
}


def generate_signals(data: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    close = data["close"]
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()
    signal = pd.Series(np.where(fast_ma > slow_ma, 1.0, 0.0), index=data.index)
    return signal.where(slow_ma.notna())
