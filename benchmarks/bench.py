"""Timing of the expensive components (informational; not part of the test suite).

python benchmarks/bench.py
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from quantproof.audit.causal import run_causality_test
from quantproof.data import generate_prices
from quantproof.statistics import (
    bootstrap_sharpe,
    probability_of_backtest_overfitting,
    reality_check,
)
from quantproof.validation import CPCV


def timed(label: str, fn: Callable[[], object], repeat: int = 3) -> None:
    best = min(_once(fn) for _ in range(repeat))
    print(f"{label:<55}{best * 1000:>10.1f} ms")


def _once(fn: Callable[[], object]) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 2520)
    m = rng.normal(0, 0.01, (2520, 200))
    prices = generate_prices(2520, seed=1)

    def strat(d):
        return np.sign(d["close"].rolling(20).mean() - d["close"].rolling(100).mean())

    timed(
        "bootstrap Sharpe, 10y daily, B=1000 (stationary)", lambda: bootstrap_sharpe(r, n_boot=1000)
    )
    timed(
        "PBO, 10y x 200 configs, S=10 (252 combinations)",
        lambda: probability_of_backtest_overfitting(m, n_partitions=10),
    )
    timed(
        "PBO, 10y x 200 configs, S=16 (12,870 combinations)",
        lambda: probability_of_backtest_overfitting(m, n_partitions=16, max_combinations=None),
    )
    timed("Reality Check + SPA, 200 configs, B=500", lambda: reality_check(m, n_bootstrap=500))
    timed("CPCV N=10 k=2 splits, 10y", lambda: list(CPCV(10, 2, embargo=0.01).split(r)))
    timed(
        "causality test, 32 perturbations, 10y", lambda: run_causality_test(strat, prices), repeat=1
    )


if __name__ == "__main__":
    main()
