"""DELIBERATELY OVERFIT example: the best of a large grid on a random walk.

The data (examples/data/noise.parquet) is a driftless random walk, so no rule has
genuine skill. Searching 4 x 5 x 4 x 3 = 240 combinations of an RSI band rule and
reporting the best full-sample result produces an attractive headline Sharpe.

PARAMETERS below were chosen as the configuration with the highest full-sample
Sharpe in PARAM_GRID (see `select_best` at the bottom). QuantProof should expose
the selection bias through the Deflated Sharpe Ratio, PBO, walk-forward/CPCV
out-of-sample results and the parameter surface.
"""

import numpy as np
import pandas as pd

NAME = "rsi_band_overfit"
PARAMETERS = {"lookback": 6, "lower": 35, "upper": 65, "hold": 3}
PARAM_GRID = {
    "lookback": [4, 6, 10, 14],
    "lower": [20, 25, 30, 35, 40],
    "upper": [55, 60, 65, 70],
    "hold": [1, 3, 6],
}
EXECUTION = {"signal_lag": 1, "commission_bps": 0.5, "spread_bps": 1.0, "slippage_bps": 0.5}


def rsi(close: pd.Series, lookback: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(lookback).mean()
    loss = (-delta.clip(upper=0)).rolling(lookback).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def generate_signals(
    data: pd.DataFrame, lookback: int = 6, lower: int = 25, upper: int = 55, hold: int = 6
) -> pd.Series:
    r = rsi(data["close"], lookback)
    raw = pd.Series(np.where(r < lower, 1.0, np.where(r > upper, -1.0, np.nan)), index=data.index)
    signal = raw.ffill(limit=hold).fillna(0.0)
    return signal.where(r.notna())


def select_best(data: pd.DataFrame) -> dict:
    """The (flawed) research step: pick the best full-sample gross Sharpe."""
    import itertools

    best, best_sr = None, -np.inf
    ret = data["close"].pct_change()
    keys = sorted(PARAM_GRID)
    for values in itertools.product(*(PARAM_GRID[k] for k in keys)):
        params = dict(zip(keys, values, strict=False))
        pnl = generate_signals(data, **params).shift(1) * ret
        sr = pnl.mean() / pnl.std() * np.sqrt(252)
        if sr > best_sr:
            best, best_sr = params, sr
    return best
