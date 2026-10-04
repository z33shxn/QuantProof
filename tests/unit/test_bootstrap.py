"""Bootstrap index generators, automatic block length, and CIs."""

from __future__ import annotations

import numpy as np
import pytest

from quantproof.errors import QuantProofInputError
from quantproof.statistics import bootstrap_sharpe, bootstrap_statistic, optimal_block_length
from quantproof.statistics.bootstrap import (
    bootstrap_indices,
    circular_block_indices,
    iid_indices,
    moving_block_indices,
    stationary_indices,
)


def ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.parametrize(
    "fn",
    [
        lambda: iid_indices(50, 7, 1),
        lambda: circular_block_indices(50, 7, 5, 1),
        lambda: moving_block_indices(50, 7, 5, 1),
        lambda: stationary_indices(50, 7, 4.0, 1),
    ],
)
def test_index_shapes_and_range(fn):
    idx = fn()
    assert idx.shape == (7, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_circular_blocks_are_contiguous_mod_n():
    idx = circular_block_indices(20, 3, 5, seed=2)
    for row in idx:
        for b in range(4):
            block = row[b * 5 : (b + 1) * 5]
            assert np.all(np.diff(block) % 20 == 1)


def test_moving_blocks_do_not_wrap():
    idx = moving_block_indices(20, 50, 5, seed=3)
    for row in idx:
        for b in range(4):
            assert np.all(np.diff(row[b * 5 : (b + 1) * 5]) == 1)


def test_stationary_mean_block_length():
    idx = stationary_indices(5000, 4, 10.0, seed=4)
    breaks = (np.diff(idx, axis=1) % 5000 != 1).sum()
    mean_len = idx.size / (breaks + idx.shape[0])
    assert mean_len == pytest.approx(10.0, rel=0.15)
    with pytest.raises(QuantProofInputError):
        stationary_indices(10, 2, 0.5)


def test_determinism():
    a, _ = bootstrap_indices(100, 10, method="stationary", block_length=5, seed=9)
    b, _ = bootstrap_indices(100, 10, method="stationary", block_length=5, seed=9)
    assert np.array_equal(a, b)
    with pytest.raises(QuantProofInputError):
        bootstrap_indices(10, 2, method="bogus", block_length=2)  # type: ignore[arg-type]


def test_optimal_block_length_grows_with_dependence():
    iid = optimal_block_length(np.random.default_rng(0).standard_normal(2000))
    dep = optimal_block_length(ar1(2000, 0.8))
    assert iid["stationary"] < 3
    assert dep["stationary"] > 5 * iid["stationary"]
    assert dep["circular"] > 0
    assert optimal_block_length([0.1, 0.2]) == {"stationary": 1.0, "circular": 1.0}


def test_block_bootstrap_preserves_autocorrelation_better_than_iid():
    x = ar1(1000, 0.7, seed=5)

    def lag1(sample):
        a = sample - sample.mean(axis=1, keepdims=True)
        return (a[:, 1:] * a[:, :-1]).sum(axis=1) / (a**2).sum(axis=1)

    iid = bootstrap_statistic(x, lag1, method="iid", n_boot=200, seed=1)
    block = bootstrap_statistic(x, lag1, method="stationary", block_length=30, n_boot=200, seed=1)
    assert abs(iid.mean) < 0.15
    assert block.mean > 0.5


def test_sharpe_ci_contains_estimate_and_is_reproducible(rng):
    x = rng.normal(0.0005, 0.01, 750)
    a = bootstrap_sharpe(x, n_boot=300, seed=3)
    b = bootstrap_sharpe(x, n_boot=300, seed=3)
    assert a == b
    assert a.ci_lower < a.estimate < a.ci_upper
    assert 0 <= a.prob_below_zero <= 1


def test_bootstrap_validates_inputs():
    with pytest.raises(QuantProofInputError):
        bootstrap_sharpe([0.1])
    with pytest.raises(QuantProofInputError):
        bootstrap_sharpe([0.1, 0.2, 0.3], confidence=1.2)


def test_constant_series_gives_nan_interval():
    res = bootstrap_sharpe([0.01] * 50, n_boot=100)
    assert np.isnan(res.ci_lower) and np.isnan(res.estimate)


@pytest.mark.parametrize(
    ("seed", "phi", "stationary", "circular"),
    [
        # Reference values computed with arch 8.0.0 `arch.bootstrap.optimal_block_length`
        # (Politis & White 2004 with the Patton, Politis & White 2009 correction) on the
        # same AR(1) series. arch is not a dependency; the values are recorded here.
        (1, 0.5, 14.062282, 16.097295),
        (2, -0.3, 10.353872, 11.852225),
        (3, 0.9, 40.905134, 46.824689),
    ],
)
def test_block_length_matches_arch_reference(seed, phi, stationary, circular):
    from quantproof.statistics.bootstrap import optimal_block_length

    rng = np.random.default_rng(seed)
    e = rng.standard_normal(2000)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = phi * x[t - 1] + e[t]
    out = optimal_block_length(x)
    assert out["stationary"] == pytest.approx(stationary, abs=1e-5)
    assert out["circular"] == pytest.approx(circular, abs=1e-5)
