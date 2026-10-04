"""Work-count safeguards: expensive procedures do exactly as much work as configured.

These tests count calls and combinations instead of measuring wall-clock time, so they are
deterministic on any machine. Timings live in ``benchmarks/bench.py``.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from quantproof import AuditConfig
from quantproof.analyzers.causal import run_causality_test
from quantproof.config import CausalityConfig
from quantproof.data.synthetic import generate_prices
from quantproof.statistics import probability_of_backtest_overfitting
from quantproof.strategy import StrategySpec
from quantproof.validation import CPCV


def test_causality_strategy_calls_are_bounded():
    prices = generate_prices(300, seed=0)
    calls = {"n": 0}

    def strat(d):
        calls["n"] += 1
        return np.sign(d["close"].pct_change(5)).fillna(0.0)

    cfg = CausalityConfig(n_timestamps=6, schemes=["additive", "extreme", "permutation"])
    rep = run_causality_test(strat, prices, cfg)
    # 2 baseline runs (determinism check) + one per executed (timestamp, scheme) + 1 control.
    executed = [t for t in rep.trials if t.status != "skipped"]
    assert calls["n"] == 2 + len(executed) + 1
    assert len(rep.trials) <= 6 * 3


def test_quick_profile_caps_the_work():
    cfg = AuditConfig(profile="quick").effective()
    assert cfg.causality.n_timestamps <= 4
    assert cfg.statistics.n_bootstrap <= 200
    assert cfg.validation.pbo_max_combinations <= 500
    assert cfg.validation.max_trials <= 50
    strict = AuditConfig(profile="strict").effective()
    assert strict.statistics.n_bootstrap >= 2000 and strict.causality.n_timestamps >= 16


def test_pbo_combinations_are_capped():
    m = np.random.default_rng(0).normal(size=(400, 5))
    res = probability_of_backtest_overfitting(m, n_partitions=16, max_combinations=300)
    assert res.n_combinations == 300 and res.n_combinations_total == comb(16, 8)
    assert len(res.logits) == 300


def test_cpcv_split_count_is_combinatorial():
    for n, k in [(6, 2), (8, 3), (10, 2)]:
        cv = CPCV(n, k)
        assert sum(1 for _ in cv.split(np.zeros(200))) == comb(n, k) == cv.n_splits


def test_grid_is_subsampled_deterministically():
    def strat(d, a=1, b=1):
        return pd.Series(0.0, index=d.index)

    spec = StrategySpec(
        func=strat, name="s", param_grid={"a": list(range(30)), "b": list(range(30))}
    )
    first = spec.grid(max_trials=50, seed=1)
    assert len(first) == 50 and spec.grid_size == 900
    assert first == spec.grid(max_trials=50, seed=1)
