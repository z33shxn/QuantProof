"""DELIBERATELY UNREALISTIC example: same-bar fills and no costs.

A one-day reversal rule: after an up close, go short; after a down close, go long.
The research assumes the trade fills at the very close that produced the signal
(signal_lag = 0) and that trading is free. On data with bid-ask-bounce-like negative
autocorrelation this looks excellent; with a one-bar delay and realistic costs the
edge disappears. QuantProof should flag QP009/QP011 statically and fail QP-EXEC-001.
"""

import numpy as np
import pandas as pd

NAME = "one_day_reversal"
PARAMETERS = {}
EXECUTION = {"signal_lag": 0, "fill": "close", "commission_bps": 0.0, "slippage_bps": 0.0}


def generate_signals(data: pd.DataFrame) -> pd.Series:
    ret = data["close"].pct_change()
    return pd.Series(-np.sign(ret), index=data.index)
