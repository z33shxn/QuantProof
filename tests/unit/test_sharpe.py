"""Sharpe ratio, moments and standard error — checked against independent references."""

from __future__ import annotations

import math
import statistics as pystats

import numpy as np
import pandas as pd
import pytest

from quantproof.errors import QuantProofInputError
from quantproof.statistics import (
    annualized_return,
    max_drawdown,
    return_moments,
    sharpe_ratio,
    sharpe_standard_error,
    sharpe_summary,
)
from quantproof.statistics.sharpe import periodic_risk_free, sharpe_ratios


def reference_sharpe(xs: list[float], ppy: float) -> float:
    """Pure-Python reference (statistics module, sample stdev)."""
    return pystats.mean(xs) / pystats.stdev(xs) * math.sqrt(ppy)


def test_known_value_by_hand():
    r = [0.01, -0.01, 0.02, 0.0]
    # mean 0.005, sample sd sqrt(0.0005/3)
    expected = 0.005 / math.sqrt(0.0005 / 3)
    assert sharpe_ratio(r, annualize=False) == pytest.approx(expected, rel=1e-12)
    assert sharpe_ratio(r, periods_per_year=252) == pytest.approx(
        expected * math.sqrt(252), rel=1e-12
    )


def test_matches_pure_python_reference(rng):
    for _ in range(20):
        xs = list(rng.normal(0.0004, 0.01, size=int(rng.integers(5, 500))))
        assert sharpe_ratio(xs, periods_per_year=52) == pytest.approx(
            reference_sharpe(xs, 52), rel=1e-10
        )


def test_risk_free_conversion_is_geometric():
    assert periodic_risk_free(0.05, 12) == pytest.approx(1.05 ** (1 / 12) - 1)
    r = np.full(24, 0.01) + np.tile([0.001, -0.001], 12)
    rf_p = 1.03 ** (1 / 12) - 1
    expected = reference_sharpe(list(r - rf_p), 12)
    assert sharpe_ratio(r, periods_per_year=12, risk_free_rate=0.03) == pytest.approx(expected)


def test_nan_handling():
    r = [0.01, np.nan, -0.01, 0.02, 0.0]
    assert sharpe_ratio(r, annualize=False) == pytest.approx(
        sharpe_ratio([0.01, -0.01, 0.02, 0.0], annualize=False)
    )
    with pytest.raises(QuantProofInputError, match="NaN"):
        sharpe_ratio(r, nan_policy="raise")


def test_infinite_values_raise_with_guidance():
    with pytest.raises(QuantProofInputError, match="zero or missing price"):
        sharpe_ratio([0.01, np.inf, 0.0])


@pytest.mark.parametrize("values", [[], [0.01], [np.nan, np.nan]])
def test_empty_and_tiny_inputs_return_nan(values):
    assert math.isnan(sharpe_ratio(values))


def test_constant_returns_have_undefined_sharpe():
    assert math.isnan(sharpe_ratio([0.001] * 50))
    assert math.isnan(sharpe_ratio([0.0] * 50))


def test_extreme_values_finite():
    r = np.array([1e6, -1e6, 2e6, -0.5e6])
    assert math.isfinite(sharpe_ratio(r))
    tiny = np.array([1e-12, -1e-12, 2e-12, 0.0])
    assert sharpe_ratio(tiny, annualize=False) == pytest.approx(
        sharpe_ratio(tiny * 1e12, annualize=False)
    )


def test_accepts_series_and_single_column_frame(rng):
    x = rng.normal(0, 0.01, 100)
    s = pd.Series(x)
    assert sharpe_ratio(s) == pytest.approx(sharpe_ratio(x))
    assert sharpe_ratio(s.to_frame()) == pytest.approx(sharpe_ratio(x))
    with pytest.raises(QuantProofInputError):
        sharpe_ratio(pd.DataFrame({"a": x, "b": x}))


def test_vectorized_matches_scalar(rng):
    m = rng.normal(0.001, 0.02, size=(300, 5))
    m[:, 3] = 0.002  # zero variance column
    vec = sharpe_ratios(m)
    for j in range(5):
        expected = sharpe_ratio(m[:, j], annualize=False)
        if math.isnan(expected):
            assert math.isnan(vec[j])
        else:
            assert vec[j] == pytest.approx(expected)


def test_moments_match_scipy_biased_estimators(rng):
    from scipy import stats

    x = rng.standard_t(5, size=2000) * 0.01
    m = return_moments(x)
    assert m.skewness == pytest.approx(stats.skew(x, bias=True))
    assert m.kurtosis == pytest.approx(stats.kurtosis(x, fisher=False, bias=True))
    assert m.std == pytest.approx(np.std(x, ddof=1))


def test_standard_error_reduces_to_lo_formula_under_normality():
    sr, n = 0.1, 500
    assert sharpe_standard_error(sr, n, 0.0, 3.0) == pytest.approx(
        math.sqrt((1 + sr**2 / 2) / (n - 1))
    )
    assert math.isnan(sharpe_standard_error(sr, 1))


def test_standard_error_covers_simulated_dispersion():
    # Monte Carlo: std of estimated per-period Sharpe across samples ≈ analytic s.e.
    rng = np.random.default_rng(7)
    n, true_sr = 250, 0.05
    est = [sharpe_ratio(rng.normal(true_sr * 0.01, 0.01, n), annualize=False) for _ in range(3000)]
    assert np.std(est) == pytest.approx(sharpe_standard_error(true_sr, n), rel=0.08)


def test_summary_and_drawdown_and_cagr():
    r = [0.10, -0.50, 0.20]
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert annualized_return([0.01] * 252, periods_per_year=252) == pytest.approx(1.01**252 - 1)
    assert annualized_return([-1.0, 0.5]) == -1.0
    s = sharpe_summary([0.01, -0.01, 0.02, 0.0], periods_per_year=4)
    assert s.n_observations == 4
    assert s.sharpe_annualized == pytest.approx(s.sharpe_per_period * 2)


def test_reproducible():
    x = np.random.default_rng(1).normal(size=100)
    assert sharpe_ratio(x) == sharpe_ratio(x.copy())


def test_sharpe_is_stable_for_tiny_and_huge_magnitudes():
    """Regression (found by hypothesis): squared deviations underflowed for ~1e-161."""
    import numpy as np

    from quantproof.statistics.sharpe import sharpe_ratio, sharpe_ratios

    r = np.array([6.66554818e-161, 0.0, 0.0])
    assert sharpe_ratio(r) == pytest.approx(sharpe_ratio(r * 0.25), rel=1e-12)
    assert sharpe_ratio(r) == pytest.approx(sharpe_ratio(r * 1e300), rel=1e-12)
    np.testing.assert_allclose(sharpe_ratios(np.c_[r, r * 1e-5]), sharpe_ratios(np.c_[r, r])[0])
