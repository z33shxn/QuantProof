"""PBO via CSCV, White's Reality Check, Hansen SPA, multiple-testing corrections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantproof.errors import QuantProofInputError
from quantproof.statistics import adjust_pvalues, probability_of_backtest_overfitting, reality_check
from quantproof.statistics.pbo import interpret_pbo


def test_pbo_one_when_is_winner_always_loses_oos():
    # Two strategies, two halves: each wins one half and loses the other.
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.001, size=(200, 2))
    m = noise.copy()
    m[:100, 0] += 0.01
    m[100:, 1] += 0.01
    res = probability_of_backtest_overfitting(m, n_partitions=2)
    assert res.n_combinations == 2
    assert res.pbo == 1.0
    assert all(lg < 0 for lg in res.logits)


def test_pbo_zero_for_dominant_strategy():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.01, size=(800, 20))
    m[:, 7] += 0.004
    res = probability_of_backtest_overfitting(m, n_partitions=8)
    assert res.pbo == 0.0
    assert set(res.selected_index) == {7}
    assert res.n_combinations == 70


def test_pbo_near_half_for_noise():
    vals = [
        probability_of_backtest_overfitting(
            np.random.default_rng(s).normal(0, 0.01, (400, 16)), n_partitions=8
        ).pbo
        for s in range(30)
    ]
    assert 0.35 < np.mean(vals) < 0.65


def test_pbo_logit_definition():
    rng = np.random.default_rng(2)
    m = rng.normal(0, 0.01, (300, 9))
    res = probability_of_backtest_overfitting(m, n_partitions=6)
    w = np.array(res.oos_rank_selected) / (9 + 1)
    assert np.allclose(res.logits, np.log(w / (1 - w)))
    assert res.pbo == pytest.approx(np.mean(np.array(res.logits) <= 0))


def test_pbo_sharpe_from_block_sums_matches_direct():
    rng = np.random.default_rng(3)
    m = rng.normal(0.0005, 0.01, (120, 4))
    res = probability_of_backtest_overfitting(m, n_partitions=4, periods_per_year=1)
    # First combination is blocks (0, 1) in-sample → rows 0..59.
    is_rows = m[:60]
    sr = is_rows.mean(axis=0) / is_rows.std(axis=0, ddof=1)
    assert res.is_sharpe_selected[0] == pytest.approx(sr.max())
    assert res.selected_index[0] == int(np.argmax(sr))


def test_pbo_subsampling_and_validation():
    m = np.random.default_rng(4).normal(0, 0.01, (400, 5))
    res = probability_of_backtest_overfitting(m, n_partitions=12, max_combinations=100)
    assert res.n_combinations == 100 and res.n_combinations_total == 924
    again = probability_of_backtest_overfitting(m, n_partitions=12, max_combinations=100)
    assert res.pbo == again.pbo
    with pytest.raises(QuantProofInputError):
        probability_of_backtest_overfitting(m, n_partitions=5)
    with pytest.raises(QuantProofInputError):
        probability_of_backtest_overfitting(m[:, :1])
    with pytest.raises(QuantProofInputError):
        probability_of_backtest_overfitting(m[:10], n_partitions=10)
    bad = m.copy()
    bad[3, 2] = np.nan
    with pytest.raises(QuantProofInputError, match="NaN"):
        probability_of_backtest_overfitting(bad)
    assert "overfit" in interpret_pbo(0.7)
    assert probability_of_backtest_overfitting(pd.DataFrame(m), n_partitions=4).n_strategies == 5


def test_reality_check_noise_not_significant_and_skill_significant():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.01, (750, 30))
    rc = reality_check(noise, n_bootstrap=400, seed=1)
    assert rc.p_value > 0.05 and rc.spa_p_value > 0.05
    skill = noise.copy()
    skill[:, 4] += 0.003
    rc2 = reality_check(skill, n_bootstrap=400, seed=1)
    assert rc2.best_index == 4
    assert rc2.p_value < 0.05 and rc2.spa_p_value < 0.05


def test_spa_less_conservative_than_rc_with_many_poor_strategies():
    rng = np.random.default_rng(6)
    m = rng.normal(0, 0.01, (750, 60))
    m[:, 1:] -= 0.002  # many clearly poor strategies
    m[:, 0] += 0.0009
    rc = reality_check(m, n_bootstrap=500, seed=3)
    assert rc.spa_p_value <= rc.p_value


def test_reality_check_benchmark_and_validation():
    rng = np.random.default_rng(7)
    m = rng.normal(0.001, 0.01, (300, 3))
    bench = m[:, 0]
    rc = reality_check(m, bench, n_bootstrap=200)
    assert rc.n_strategies == 3
    with pytest.raises(QuantProofInputError):
        reality_check(m[:5])
    with pytest.raises(QuantProofInputError):
        reality_check(m, bench[:10])
    bad = m.copy()
    bad[0, 0] = np.nan
    with pytest.raises(QuantProofInputError):
        reality_check(bad)


def test_adjust_pvalues_reference_values():
    p = [0.01, 0.04, 0.03, 0.2]
    assert np.allclose(adjust_pvalues(p, "bonferroni"), [0.04, 0.16, 0.12, 0.8])
    assert np.allclose(adjust_pvalues(p, "holm"), [0.04, 0.09, 0.09, 0.2])
    assert np.allclose(adjust_pvalues(p, "bh"), [0.04, 0.0533333, 0.0533333, 0.2])
    assert adjust_pvalues([], "holm").size == 0
    with pytest.raises(QuantProofInputError):
        adjust_pvalues([1.5])
    with pytest.raises(QuantProofInputError):
        adjust_pvalues([0.1], "nope")  # type: ignore[arg-type]
