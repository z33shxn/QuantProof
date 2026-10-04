"""Seeded synthetic market data for examples and tests.

The generator is intentionally simple and transparent. It is *not* a realistic
market model; it exists so every example in this repository runs offline,
deterministically, and without redistributing third-party market data.

Model
-----
Daily log returns follow a two-state Markov-switching process:

* state 0 ("calm"):     drift ``+mu``, volatility ``sigma``
* state 1 ("stressed"): drift ``-mu``, volatility ``stress_vol_multiplier * sigma``

with persistence ``p_stay`` in each state, plus an optional first-order
autocorrelation term ``ar1`` (negative values imitate bid-ask bounce / short-term
reversal). Open/high/low are generated around the close path so that the
usual OHLC inequalities hold by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_prices(
    n: int = 1500,
    *,
    seed: int = 42,
    start: str = "2015-01-01",
    freq: str = "B",
    mu: float = 0.0006,
    sigma: float = 0.009,
    stress_vol_multiplier: float = 2.2,
    p_stay: float = 0.985,
    ar1: float = 0.0,
    initial_price: float = 100.0,
    tz: str | None = None,
) -> pd.DataFrame:
    """Generate an OHLCV DataFrame indexed by timestamp.

    Parameters
    ----------
    n: number of bars.
    seed: random seed (results are bit-for-bit reproducible for a given NumPy version).
    start, freq, tz: index construction (``freq="B"`` = business days).
    mu, sigma: per-bar drift and volatility of the calm regime.
    stress_vol_multiplier: volatility multiplier in the stressed regime.
    p_stay: probability of staying in the current regime each bar.
    ar1: first-order autocorrelation coefficient applied to the return innovations.
    initial_price: starting close.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=np.int64)
    states[0] = 0
    switches = rng.random(n) > p_stay
    for t in range(1, n):
        states[t] = 1 - states[t - 1] if switches[t] else states[t - 1]
    drift = np.where(states == 0, mu, -mu)
    vol = np.where(states == 0, sigma, sigma * stress_vol_multiplier)
    eps = rng.standard_normal(n) * vol
    if ar1:
        shocks = np.empty(n)
        shocks[0] = eps[0]
        for t in range(1, n):
            shocks[t] = eps[t] + ar1 * shocks[t - 1]
        eps = shocks
    log_ret = drift + eps
    close = initial_price * np.exp(np.cumsum(log_ret))
    prev_close = np.r_[initial_price, close[:-1]]
    gap = rng.standard_normal(n) * vol * 0.3
    open_ = prev_close * np.exp(gap)
    hi_ext = np.abs(rng.standard_normal(n)) * vol * 0.5
    lo_ext = np.abs(rng.standard_normal(n)) * vol * 0.5
    high = np.maximum(open_, close) * np.exp(hi_ext)
    low = np.minimum(open_, close) * np.exp(-lo_ext)
    volume = np.round(1_000_000 * np.exp(rng.standard_normal(n) * 0.3) * (1 + states)).astype(
        np.int64
    )
    index = pd.date_range(start=start, periods=n, freq=freq, tz=tz, name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def generate_benchmark(prices: pd.DataFrame, *, column: str = "close") -> pd.Series:
    """Benchmark returns (buy-and-hold of the synthetic asset)."""
    return prices[column].pct_change().fillna(0.0).rename("benchmark")
