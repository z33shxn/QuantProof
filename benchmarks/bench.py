"""Timing of the expensive components (informational; not part of the test suite).

python benchmarks/bench.py
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from quantproof import AuditConfig, audit
from quantproof.analyzers.causal import run_causality_test
from quantproof.analyzers.static import analyze_source
from quantproof.data import generate_prices, validate_data
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
        "causality test, default config, 10y", lambda: run_causality_test(strat, prices), repeat=1
    )
    panel = (
        pd.concat({f"S{i:02d}": generate_prices(1260, seed=i) for i in range(20)}, names=["symbol"])
        .swaplevel()
        .sort_index()
    )

    def xs(d):
        c = d["close"].unstack()
        rk = c.pct_change(20).rank(axis=1)
        return rk.sub(rk.mean(axis=1), axis=0)

    timed("causality test, panel 5y x 20 symbols", lambda: run_causality_test(xs, panel), repeat=1)
    timed("data validation, 10y daily", lambda: validate_data(prices))
    timed("data validation, panel 5y x 20 symbols", lambda: validate_data(panel.reset_index()))
    source = "\n\n".join(
        f"def f{i}(df):\n    x = df['close'].shift(1).rolling(5).mean()\n"
        f"    y = helper{i}(x)\n    return y.pct_change()\n\n"
        f"def helper{i}(s):\n    return s.diff()"
        for i in range(300)
    )
    timed(f"static analysis, {source.count(chr(10))} lines", lambda: analyze_source(source))
    res = audit(strat, prices, config=AuditConfig(profile="quick"))
    timed("report generation (HTML)", res.to_html)
    timed("report generation (Markdown)", res.to_markdown)
    timed(
        "full audit, quick profile, 10y",
        lambda: audit(strat, prices, config=AuditConfig(profile="quick")),
        repeat=1,
    )


if __name__ == "__main__":
    main()
