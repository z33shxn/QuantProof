"""Walk-forward, purging, embargo, purged K-fold and CPCV."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from quantproof.errors import QuantProofInputError
from quantproof.validation import (
    CPCV,
    PurgedKFold,
    WalkForward,
    apply_embargo,
    assert_no_leakage,
    embargo_size,
    label_intervals,
    purge,
    temporal_train_test_split,
)
from quantproof.validation.embargo import contiguous_blocks


# ----------------------------------------------------------------- walk-forward
def test_expanding_walk_forward_windows():
    wf = WalkForward(train_size=50, test_size=10, gap=2)
    w = wf.windows(100)
    assert [(x.train_start, x.train_end, x.test_start, x.test_end) for x in w] == [
        (0, 50, 52, 62),
        (0, 60, 62, 72),
        (0, 70, 72, 82),
        (0, 80, 82, 92),
    ]
    for train, test in wf.split(np.zeros(100)):
        assert train.max() + 2 < test.min()  # gap respected, strictly before


def test_rolling_walk_forward_fixed_train_length():
    wf = WalkForward(train_size=30, test_size=10, expanding=False, step=5)
    for train, test in wf.split(np.zeros(100)):
        assert len(train) == 30 and len(test) == 10
        assert train.max() < test.min()


def test_walk_forward_n_splits_and_fractions():
    wf = WalkForward(train_size=0.5, n_splits=5)
    folds = list(wf.split(np.zeros(103)))
    assert len(folds) == 5 == wf.get_n_splits(np.zeros(103))
    tests = np.concatenate([t for _, t in folds])
    assert len(np.unique(tests)) == len(tests)  # test windows never overlap
    assert "fold  0" in wf.diagram(np.zeros(103))


@pytest.mark.parametrize(
    ("kwargs", "n"),
    [
        ({"train_size": 0.5}, 10),
        ({"train_size": 200, "test_size": 10}, 100),
        ({"train_size": 5, "test_size": 0}, 50),
    ],
)
def test_walk_forward_errors(kwargs, n):
    with pytest.raises(QuantProofInputError):
        WalkForward(**kwargs).windows(n)


def test_walk_forward_bad_constructor_args():
    with pytest.raises(QuantProofInputError):
        WalkForward(10, 5, gap=-1)
    with pytest.raises(QuantProofInputError):
        WalkForward(10, 5, step=0)
    with pytest.raises(QuantProofInputError):
        WalkForward(10).get_n_splits()


# ----------------------------------------------------------------- embargo/purge
def test_embargo_size_conventions():
    assert embargo_size(1000, 0.01) == 10
    assert embargo_size(1000, 0.0101) == 11  # ceil
    assert embargo_size(1000, 5) == 5
    assert embargo_size(1000, 0) == 0
    with pytest.raises(QuantProofInputError):
        embargo_size(100, -1)
    with pytest.raises(QuantProofInputError):
        embargo_size(100, 2.5)


def test_embargo_boundaries():
    train = np.r_[0:10, 20:40]
    test = np.arange(10, 20)
    kept = apply_embargo(train, test, 40, 3)
    assert 20 not in kept and 22 not in kept and 23 in kept
    assert 9 in kept  # nothing embargoed before the test block


def test_contiguous_blocks():
    assert contiguous_blocks(np.array([5, 1, 2, 3, 7, 8])) == [(1, 3), (5, 5), (7, 8)]
    assert contiguous_blocks(np.array([], dtype=int)) == []


def test_purge_with_label_horizon_boundaries():
    start, end = label_intervals(30, label_horizon=3)
    test = np.arange(10, 15)
    train = np.setdiff1d(np.arange(30), test)
    kept = purge(train, test, start, end)
    # Labels of obs 7..9 end at 10..12 (overlap test); obs 15..17 start inside test span [10, 17].
    assert set(range(7, 10)).isdisjoint(kept)
    assert set(range(15, 18)).isdisjoint(kept)
    assert 6 in kept and 18 in kept


def test_purge_with_datetime_t1():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    t1 = pd.Series(idx + pd.Timedelta(days=2), index=idx)
    start, end = label_intervals(10, t1)
    kept = purge(np.r_[0:4, 6:10], np.array([4, 5]), start, end)
    # test span: [day4, day7]; obs 2,3 end at day4/5 → purged; obs 6,7 start ≤ day7 → purged.
    assert kept.tolist() == [0, 1, 8, 9]


def test_label_interval_validation():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    with pytest.raises(QuantProofInputError):
        label_intervals(3, pd.Series(idx - pd.Timedelta(days=1), index=idx))
    with pytest.raises(QuantProofInputError):
        label_intervals(4, pd.Series(idx, index=idx))
    with pytest.raises(QuantProofInputError):
        label_intervals(3, label_horizon=-1)


def test_temporal_train_test_split():
    train, test = temporal_train_test_split(100, test_size=0.2, gap=5)
    assert test.tolist() == list(range(80, 100)) and train.max() == 74
    with pytest.raises(QuantProofInputError):
        temporal_train_test_split(10, test_size=10)


def test_assert_no_leakage():
    s, e = label_intervals(20, label_horizon=2)
    with pytest.raises(AssertionError):
        assert_no_leakage(np.arange(0, 10), np.arange(9, 15))
    with pytest.raises(AssertionError):
        assert_no_leakage(np.arange(0, 10), np.arange(10, 15), s, e)
    assert_no_leakage(np.arange(0, 8), np.arange(10, 15), s, e)


# ----------------------------------------------------------------- purged k-fold
def test_purged_kfold_no_overlap_and_coverage():
    X = np.zeros(100)
    cv = PurgedKFold(5, label_horizon=3, embargo=0.02)
    s, e = label_intervals(100, label_horizon=3)
    tests = []
    for train, test in cv.split(X):
        assert_no_leakage(train, test, s, e)
        after = train[train > test.max()]
        if after.size:
            assert after.min() > test.max() + 3 + 2  # purge (3) then embargo (2)
        tests.append(test)
    assert np.array_equal(np.sort(np.concatenate(tests)), np.arange(100))
    assert cv.get_n_splits() == 5


def test_purged_kfold_with_t1_series_on_dataframe():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    t1 = pd.Series(idx + pd.Timedelta(days=5), index=idx)
    X = pd.DataFrame({"x": np.arange(60)}, index=idx)
    for train, test in PurgedKFold(3, t1=t1).split(X):
        s, e = label_intervals(60, t1)
        assert_no_leakage(train, test, s, e)


def test_purged_kfold_errors():
    with pytest.raises(QuantProofInputError):
        PurgedKFold(1)
    with pytest.raises(QuantProofInputError):
        list(PurgedKFold(5).split(np.zeros(3)))


# ----------------------------------------------------------------- CPCV
@pytest.mark.parametrize(("n", "k"), [(6, 2), (8, 2), (5, 1), (6, 3)])
def test_cpcv_counts(n, k):
    cv = CPCV(n, k)
    assert cv.n_splits == comb(n, k) == len(list(cv.split(np.zeros(120))))
    assert cv.n_paths == comb(n, k) * k // n


def test_cpcv_paths_cover_each_group_once():
    cv = CPCV(6, 2)
    combos = cv.group_combinations()
    for path in cv.paths():
        assert sorted(g for g, _ in path) == list(range(6))
        for g, s in path:
            assert g in combos[s]
    used = [pair for path in cv.paths() for pair in path]
    assert len(set(used)) == len(used)  # each (group, split) used once


def test_cpcv_purge_and_embargo_around_every_test_block():
    X = np.zeros(120)
    cv = CPCV(6, 2, embargo=3, label_horizon=2)
    s, e = label_intervals(120, label_horizon=2)
    for train, test in cv.split(X):
        assert_no_leakage(train, test, s, e)
        for _a, b in contiguous_blocks(test):
            assert not np.any((train > b) & (train <= b + 3))


def test_cpcv_assemble_paths_reconstructs_full_series():
    X = np.arange(60, dtype=float)
    cv = CPCV(5, 2)
    preds = [X[test] for _, test in cv.split(X)]
    paths = cv.assemble_paths(X, preds)
    assert paths.shape == (cv.n_paths, 60)
    for p in paths:
        assert np.array_equal(p, X)


def test_cpcv_deterministic_and_errors():
    a = [t.tolist() for _, t in CPCV(6, 2).split(np.zeros(50))]
    b = [t.tolist() for _, t in CPCV(6, 2).split(np.zeros(50))]
    assert a == b
    with pytest.raises(QuantProofInputError):
        CPCV(1, 1)
    with pytest.raises(QuantProofInputError):
        CPCV(4, 4)
    with pytest.raises(QuantProofInputError):
        list(CPCV(6, 2).split(np.zeros(4)))
