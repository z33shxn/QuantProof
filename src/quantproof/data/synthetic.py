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


def generate_universe(
    n_symbols: int = 12,
    n: int = 1500,
    *,
    seed: int = 7,
    start: str = "2019-01-01",
    market_vol: float = 0.008,
    idio_vol: float = 0.012,
    drift_persistence: float = 0.995,
    drift_vol: float = 0.00004,
) -> pd.DataFrame:
    """Long-format multi-asset OHLCV data (``timestamp``, ``symbol``, OHLCV columns).

    Each symbol's log return is ``beta_i · market_t + alpha_i,t + e_i,t`` where ``alpha_i,t``
    is a slowly varying AR(1) drift (``drift_persistence``, innovation ``drift_vol``). A
    persistent drift makes past relative returns weakly predictive of future relative
    returns, i.e. a mild, noisy cross-sectional momentum effect exists by construction.
    With these defaults it is small relative to noise and costs.
    """
    if n_symbols < 2 or n < 2:
        raise ValueError("need at least 2 symbols and 2 bars")
    rng = np.random.default_rng(seed)
    market = rng.standard_normal(n) * market_vol
    index = pd.date_range(start=start, periods=n, freq="B", name="timestamp")
    frames = []
    for i in range(n_symbols):
        beta = 0.7 + 0.6 * rng.random()
        alpha = np.empty(n)
        alpha[0] = rng.standard_normal() * drift_vol / np.sqrt(1 - drift_persistence**2)
        shocks = rng.standard_normal(n) * drift_vol
        for t in range(1, n):
            alpha[t] = drift_persistence * alpha[t - 1] + shocks[t]
        log_ret = beta * market + alpha + rng.standard_normal(n) * idio_vol
        close = 50.0 * (1 + i / n_symbols) * np.exp(np.cumsum(log_ret))
        prev = np.r_[close[0], close[:-1]]
        open_ = prev * np.exp(rng.standard_normal(n) * idio_vol * 0.3)
        high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(n)) * idio_vol * 0.5)
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(n)) * idio_vol * 0.5)
        volume = np.round(500_000 * np.exp(rng.standard_normal(n) * 0.3)).astype(np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": index,
                    "symbol": f"SYM{i:02d}",
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp", "symbol"], ignore_index=True
    )
