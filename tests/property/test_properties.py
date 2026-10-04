"""Property-based tests (Hypothesis) for invariants."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from quantproof.analyzers.causal import perturb_future
from quantproof.data import generate_prices, validate_data
from quantproof.experiments import hash_dataframe
from quantproof.statistics import sharpe_ratio
from quantproof.statistics.probabilistic_sharpe import psr_from_moments
from quantproof.validation import CPCV, PurgedKFold, WalkForward, label_intervals, purge

finite = st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False)
returns_arrays = arrays(np.float64, st.integers(3, 200), elements=finite)


@given(returns_arrays, st.floats(0.01, 100))
def test_sharpe_scale_invariant(r, k):
    a = sharpe_ratio(r)
    b = sharpe_ratio(r * k)
    if math.isnan(a):
        assert math.isnan(b)
    else:
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


@given(returns_arrays)
def test_sharpe_sign_flips_with_returns(r):
    a, b = sharpe_ratio(r), sharpe_ratio(-r)
    if not math.isnan(a):
        assert math.isclose(a, -b, rel_tol=1e-9, abs_tol=1e-12)


@given(returns_arrays)
def test_sharpe_permutation_invariant(r):
    a, b = sharpe_ratio(r), sharpe_ratio(r[::-1])
    assert (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


@given(st.floats(-0.5, 0.5), st.integers(2, 5000), st.floats(-3, 3), st.floats(1, 30))
def test_psr_in_unit_interval(sr, n, skew, kurt):
    p, _ = psr_from_moments(sr, n, skew, kurt, 0.0)
    assert math.isnan(p) or 0.0 <= p <= 1.0


@settings(max_examples=60)
@given(st.integers(20, 300), st.integers(2, 8), st.integers(0, 5), st.integers(0, 10))
def test_purged_kfold_invariants(n, k, horizon, embargo):
    if n < k:
        return
    s, e = label_intervals(n, label_horizon=horizon)
    seen = []
    for train, test in PurgedKFold(k, label_horizon=horizon, embargo=embargo).split(np.zeros(n)):
        assert np.intersect1d(train, test).size == 0
        assert purge(train, test, s, e).size == train.size  # nothing left to purge
        seen.append(test)
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(n))


@settings(max_examples=40)
@given(st.integers(3, 8), st.integers(1, 3), st.integers(40, 200))
def test_cpcv_paths_partition_sample(n_groups, k, n):
    if k >= n_groups:
        return
    cv = CPCV(n_groups, k)
    x = np.arange(n, dtype=float)
    preds = [x[test] for _, test in cv.split(x)]
    paths = cv.assemble_paths(x, preds)
    assert paths.shape[0] == cv.n_paths
    assert np.all(paths == x)


@settings(max_examples=50)
@given(
    st.integers(30, 300), st.floats(0.2, 0.8), st.integers(2, 6), st.integers(0, 5), st.booleans()
)
def test_walk_forward_is_strictly_forward(n, frac, splits, gap, expanding):
    wf = WalkForward(frac, n_splits=splits, gap=gap, expanding=expanding)
    try:
        folds = list(wf.split(n))
    except Exception:
        return
    prev_end = -1
    for train, test in folds:
        assert train.max() + gap < test.min()
        assert test.min() > prev_end
        prev_end = test.max()


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(0, 2**31 - 1))
def test_hash_deterministic_and_order_sensitive(seed):
    df = generate_prices(30, seed=seed)
    assert hash_dataframe(df) == hash_dataframe(df.copy(deep=True))
    swapped = df.iloc[[1, 0, *range(2, 30)]]
    assert hash_dataframe(df) != hash_dataframe(swapped)


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(0, 10_000),
    st.sampled_from(["additive", "multiplicative", "permutation", "extreme"]),
    st.integers(0, 37),
)
def test_perturbation_leaves_past_untouched(seed, scheme, k):
    df = generate_prices(40, seed=seed)
    out = perturb_future(df, k, scheme, seed=seed)
    assert out.iloc[: k + 1].equals(df.iloc[: k + 1].astype(float))


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(0, 10_000), st.integers(20, 120))
def test_generated_data_is_always_valid(seed, n):
    findings = validate_data(generate_prices(n, seed=seed))
    issues = [f for f in findings if f.is_issue and f.id not in {"QP-DATA-013"}]
    assert issues == []


@settings(max_examples=30)
@given(st.lists(st.integers(0, 50), min_size=1, max_size=30, unique=True))
def test_duplicate_detection_counts_exactly(dups):
    df = generate_prices(60, seed=1)
    extra = df.iloc[dups]
    out = pd.concat([df, extra]).sort_index(kind="mergesort")
    f = next(x for x in validate_data(out) if x.id == "QP-DATA-004")
    assert f.evidence["count"] == len(dups)
