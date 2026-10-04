"""PSR, MinTRL and DSR: formula checks, Monte Carlo calibration, edge cases."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.stats import norm

from quantproof.errors import QuantProofInputError
from quantproof.statistics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)
from quantproof.statistics.deflated_sharpe import EULER_MASCHERONI
from quantproof.statistics.probabilistic_sharpe import psr_from_moments


def reference_psr(sr, n, g3, g4, bench):
    return norm.cdf((sr - bench) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))


def test_psr_formula_by_hand():
    psr, z = psr_from_moments(0.1, 253, -0.5, 6.0, 0.02)
    assert psr == pytest.approx(reference_psr(0.1, 253, -0.5, 6.0, 0.02))
    assert z == pytest.approx(0.08 * math.sqrt(252) / math.sqrt(1 + 0.05 + 1.25 * 0.01))


def test_psr_from_returns_uses_sample_moments(rng):
    from scipy import stats

    x = rng.normal(0.0008, 0.01, 750)
    res = probabilistic_sharpe_ratio(x, benchmark_sharpe=0.5, periods_per_year=252)
    sr = x.mean() / x.std(ddof=1)
    expected = reference_psr(
        sr, 750, stats.skew(x), stats.kurtosis(x, fisher=False), 0.5 / math.sqrt(252)
    )
    assert res.psr == pytest.approx(expected, rel=1e-10)
    assert res.to_dict()["benchmark_sharpe_annualized"] == pytest.approx(0.5)


def test_psr_is_calibrated_under_the_null():
    """With zero true Sharpe, P(PSR > 0.95) should be close to 5 %."""
    rng = np.random.default_rng(2024)
    psrs = np.array([probabilistic_sharpe_ratio(rng.normal(0, 0.01, 250)).psr for _ in range(2000)])
    assert 0.035 < np.mean(psrs > 0.95) < 0.065


def test_psr_monotone_in_sample_size_and_benchmark():
    a, _ = psr_from_moments(0.1, 100, 0, 3, 0)
    b, _ = psr_from_moments(0.1, 1000, 0, 3, 0)
    c, _ = psr_from_moments(0.1, 1000, 0, 3, 0.05)
    assert a < b and c < b


def test_negative_skew_and_fat_tails_lower_psr():
    base, _ = psr_from_moments(0.1, 500, 0, 3, 0)
    skewed, _ = psr_from_moments(0.1, 500, -2, 3, 0)
    fat, _ = psr_from_moments(0.1, 500, 0, 15, 0)
    assert skewed < base and fat < base


def test_psr_edge_cases():
    assert math.isnan(psr_from_moments(0.1, 1, 0, 3)[0])
    assert math.isnan(psr_from_moments(float("nan"), 100, 0, 3)[0])
    assert math.isnan(probabilistic_sharpe_ratio([0.01] * 30).psr)  # zero variance
    assert math.isnan(probabilistic_sharpe_ratio([]).psr)


def test_min_track_record_length():
    sr, g3, g4 = 0.1, -0.5, 5.0
    mtrl = minimum_track_record_length(sr, g3, g4, benchmark_sharpe=0.0, confidence=0.95)
    expected = 1 + (1 - g3 * sr + (g4 - 1) / 4 * sr**2) * (norm.ppf(0.95) / sr) ** 2
    assert mtrl == pytest.approx(expected)
    # At exactly MinTRL observations, PSR equals the confidence level.
    psr, _ = psr_from_moments(sr, mtrl, g3, g4, 0.0)
    assert psr == pytest.approx(0.95, abs=1e-9)
    assert minimum_track_record_length(0.0, 0, 3) == float("inf")
    with pytest.raises(QuantProofInputError):
        minimum_track_record_length(0.1, 0, 3, confidence=1.5)


def test_expected_max_formula():
    n, v = 100, 0.25
    g = EULER_MASCHERONI
    expected = math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e)))
    assert expected_max_sharpe(n, v) == pytest.approx(expected)
    assert expected_max_sharpe(1, 1.0) == 0.0
    with pytest.raises(QuantProofInputError):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(QuantProofInputError):
        expected_max_sharpe(10, -1.0)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_expected_max_matches_monte_carlo(n):
    rng = np.random.default_rng(n)
    sims = rng.standard_normal((20000, n)).max(axis=1).mean()
    assert expected_max_sharpe(n, 1.0) == pytest.approx(sims, rel=0.03)


def test_dsr_equals_psr_for_one_trial(rng):
    x = rng.normal(0.0005, 0.01, 500)
    assert deflated_sharpe_ratio(x, n_trials=1).dsr == pytest.approx(
        probabilistic_sharpe_ratio(x).psr
    )


def test_dsr_decreases_with_trials(rng):
    x = rng.normal(0.0008, 0.01, 1000)
    values = [deflated_sharpe_ratio(x, n_trials=n).dsr for n in (1, 10, 100, 1000)]
    assert all(a > b for a, b in itertools.pairwise(values))


def test_dsr_variance_sources(rng):
    x = rng.normal(0.0008, 0.01, 1000)
    trial = rng.normal(0, 0.03, 50)
    a = deflated_sharpe_ratio(x, n_trials=50, trial_sharpes=trial)
    assert a.variance_source == "trial_sharpes"
    assert a.sharpe_variance == pytest.approx(np.var(trial, ddof=1))
    b = deflated_sharpe_ratio(x, n_trials=50)
    assert b.sharpe_variance == pytest.approx(1 / 999)
    c = deflated_sharpe_ratio(x, n_trials=50, sharpe_variance=0.001)
    assert c.variance_source == "explicit"
    assert c.expected_max_sharpe_per_period == pytest.approx(expected_max_sharpe(50, 0.001))
    with pytest.raises(QuantProofInputError):
        deflated_sharpe_ratio(x, n_trials=5, trial_sharpes=[0.1])


def test_dsr_rejects_best_of_noise():
    """Selecting the best of 200 noise strategies should rarely yield DSR > 0.95."""
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(40):
        m = rng.normal(0, 0.01, size=(500, 200))
        sr = m.mean(axis=0) / m.std(axis=0, ddof=1)
        best = int(np.argmax(sr))
        if deflated_sharpe_ratio(m[:, best], n_trials=200, trial_sharpes=sr).dsr > 0.95:
            hits += 1
    assert hits <= 4  # ~5 % nominal; allow sampling error
