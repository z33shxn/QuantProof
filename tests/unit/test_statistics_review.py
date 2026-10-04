"""Independent checks of DSR inputs, PBO behaviour regimes and SPA edge cases."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from quantproof.analyzers.statistical.selection import analyze_selection
from quantproof.config import AuditConfig
from quantproof.errors import QuantProofInputError
from quantproof.severity import Severity
from quantproof.statistics import probability_of_backtest_overfitting, reality_check
from quantproof.statistics.deflated_sharpe import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    expected_max_standard_normal,
)

# --- expected maximum ------------------------------------------------------------------


def test_exact_expected_max_matches_closed_forms():
    assert expected_max_standard_normal(1) == 0.0
    assert expected_max_standard_normal(2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-9)
    assert expected_max_standard_normal(3) == pytest.approx(1.5 / math.sqrt(math.pi), abs=1e-9)


def test_exact_expected_max_matches_monte_carlo():
    rng = np.random.default_rng(7)
    sims = rng.standard_normal((40_000, 50)).max(axis=1)
    se = sims.std() / math.sqrt(sims.size)
    assert abs(expected_max_standard_normal(50) - sims.mean()) < 4 * se


@pytest.mark.parametrize(("n", "rel"), [(2, -0.079), (10, 0.023), (100, 0.009), (1000, 0.004)])
def test_documented_approximation_error(n, rel):
    approx = expected_max_sharpe(n, 1.0)
    exact = expected_max_sharpe(n, 1.0, method="exact")
    assert approx / exact - 1 == pytest.approx(rel, abs=0.001)


def test_expected_max_scales_with_variance_and_validates():
    assert expected_max_sharpe(20, 4.0, method="exact") == pytest.approx(
        2 * expected_max_sharpe(20, 1.0, method="exact")
    )
    with pytest.raises(QuantProofInputError):
        expected_max_sharpe(10, 1.0, method="simulated")
    with pytest.raises(QuantProofInputError):
        expected_max_sharpe(0, 1.0)


# --- effective trials ------------------------------------------------------------------


def test_effective_trials_limits():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1000, 1))
    identical = np.repeat(base, 25, axis=1) + 1e-12 * rng.standard_normal((1000, 25))
    assert effective_number_of_trials(identical) == pytest.approx(1.0, abs=0.05)
    independent = rng.standard_normal((5000, 25))
    assert 22 <= effective_number_of_trials(independent) <= 25
    two_clusters = np.hstack(
        [np.repeat(base, 10, axis=1), np.repeat(rng.standard_normal((1000, 1)), 10, axis=1)]
    )
    two_clusters += 1e-9 * rng.standard_normal(two_clusters.shape)
    # Li & Ji credit fractional eigenvalue parts, so two equal clusters score between 2 and 4
    # (here ~3); the estimate is a heuristic upper bound on the dimension, as documented.
    assert 2.0 <= effective_number_of_trials(two_clusters) <= 4.0


def test_dsr_reports_trial_provenance():
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 750))
    out = deflated_sharpe_ratio(
        r,
        n_trials=50,
        expected_max_method="exact",
        trials_source="PARAM_GRID size",
        effective_trials=12.5,
    ).to_dict()
    assert out["n_trials"] == 50
    assert out["trials_source"] == "PARAM_GRID size"
    assert out["effective_trials"] == 12.5
    assert out["expected_max_method"] == "exact"
    more = deflated_sharpe_ratio(r, n_trials=500, expected_max_method="exact")
    assert more.dsr < out["dsr"], "more trials must raise the hurdle"


# --- PBO regimes -----------------------------------------------------------------------


def _pbo(m, s=10):
    return probability_of_backtest_overfitting(m, n_partitions=s, max_combinations=None).pbo


def test_pbo_noise_is_roughly_neutral():
    vals = [_pbo(np.random.default_rng(seed).normal(0, 0.01, (1000, 10))) for seed in range(8)]
    assert 0.3 <= float(np.mean(vals)) <= 0.7


def test_pbo_real_edge_is_low():
    rng = np.random.default_rng(3)
    m = rng.normal(0, 0.01, (1000, 10))
    m[:, 4] += 0.003  # one configuration with a large, persistent edge
    assert _pbo(m) < 0.1


def test_pbo_anti_persistent_is_high():
    # Each configuration is good in a different half of the blocks: whatever wins in-sample
    # is, by construction, poor out of sample.
    rng = np.random.default_rng(4)
    s, t, n = 10, 1000, 10
    blocks = np.array_split(np.arange(t), s)
    m = rng.normal(0, 0.002, (t, n))
    for j in range(n):
        good = rng.choice(s, size=s // 2, replace=False)
        for b in range(s):
            m[blocks[b], j] += 0.002 if b in good else -0.002
    assert _pbo(m, s) > 0.9


def test_pbo_matches_naive_reference():
    rng = np.random.default_rng(5)
    t, n, s = 240, 6, 6
    m = rng.normal(0.0002, 0.01, (t, n))
    blocks = np.array_split(np.arange(t), s)
    logits = []
    for combo in combinations(range(s), s // 2):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in range(s) if b not in combo])
        is_sr = m[is_rows].mean(0) / m[is_rows].std(0, ddof=1)
        oos_sr = m[oos_rows].mean(0) / m[oos_rows].std(0, ddof=1)
        w = rankdata(oos_sr)[int(np.argmax(is_sr))] / (n + 1)
        logits.append(math.log(w / (1 - w)))
    res = probability_of_backtest_overfitting(m, n_partitions=s, max_combinations=None)
    np.testing.assert_allclose(res.logits, logits, rtol=1e-9)
    assert res.pbo == pytest.approx(np.mean(np.array(logits) <= 0))


def test_pbo_small_sample_errors_and_notes():
    rng = np.random.default_rng(6)
    with pytest.raises(QuantProofInputError, match="at least 32 observations"):
        probability_of_backtest_overfitting(rng.normal(size=(30, 4)), n_partitions=16)
    res = probability_of_backtest_overfitting(rng.normal(size=(100, 5)), n_partitions=10)
    joined = " ".join(res.notes)
    assert "noisy" in joined and "odd number" in joined
    sampled = probability_of_backtest_overfitting(
        rng.normal(size=(400, 4)), n_partitions=16, max_combinations=100
    )
    assert any("sampled" in note for note in sampled.notes)


def test_selection_reports_insufficient_data_as_info():
    rng = np.random.default_rng(8)
    m = pd.DataFrame(rng.normal(size=(15, 4)), index=pd.bdate_range("2024-01-01", periods=15))
    cfg = AuditConfig()
    _, findings = analyze_selection(m, cfg.validation, cfg.statistics)
    not_run = [f for f in findings if f.title.endswith("not computed")]
    assert not_run and all(f.severity is Severity.INFO for f in not_run)
    assert any(f.id == "QP-VAL-002" for f in not_run)


# --- Reality Check / SPA ---------------------------------------------------------------


def test_spa_is_undefined_for_constant_differentials():
    m = np.full((200, 3), 0.001)
    rc = reality_check(m, n_bootstrap=200)
    assert math.isnan(rc.spa_p_value)


def test_selection_does_not_pass_undefined_spa():
    m = pd.DataFrame(np.full((300, 4), 0.001), index=pd.bdate_range("2023-01-02", periods=300))
    cfg = AuditConfig()
    cfg.validation.pbo = cfg.validation.cpcv = cfg.validation.walk_forward = False
    _, findings = analyze_selection(m, cfg.validation, cfg.statistics)
    spa = next(f for f in findings if f.id == "QP-VAL-004")
    assert spa.severity is Severity.INFO and "undefined" in spa.message


def test_reality_check_noise_vs_edge():
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 0.01, (750, 20))
    edge = noise.copy()
    edge[:, 0] += 0.004
    assert reality_check(noise, n_bootstrap=400, seed=1).spa_p_value > 0.05
    assert reality_check(edge, n_bootstrap=400, seed=1).spa_p_value < 0.05
