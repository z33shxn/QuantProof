"""The cross-sectional momentum idea as it often looks in a first notebook — with bugs.

Three common mistakes, each realistic and easy to miss:

1. The momentum score is smoothed with a *centered* rolling mean, so the score at t
   averages in returns from t+1 and t+2 (look-ahead).
2. Execution is declared as filling at the same close that produced the signal, with
   no costs.
3. Only the best parameter set is shown (no PARAM_GRID), so selection bias cannot be
   measured — the 36 variants tried are not declared.

The fixed version is ``examples/cross_sectional_momentum/strategy.py``.
"""

import pandas as pd

NAME = "xs_momentum_first_draft"
PARAMETERS = {"lookback": 120, "n_side": 2}
EXECUTION = {"signal_lag": 0, "fill": "close", "commission_bps": 0.0}


def generate_signals(data: pd.DataFrame, lookback: int = 120, n_side: int = 2) -> pd.DataFrame:
    close = data["close"].unstack("symbol")
    momentum = close.pct_change(lookback)
    smooth = momentum.rolling(5, center=True).mean()  # BUG: uses t+1, t+2
    ranks = smooth.rank(axis=1, method="first")
    n = ranks.notna().sum(axis=1)
    long = ranks.gt(n - n_side, axis=0)
    short = ranks.le(n_side, axis=0)
    weights = (long.astype(float) - short.astype(float)) / (2.0 * n_side)
    return weights.where(smooth.notna())
