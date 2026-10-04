"""Exact boundary behaviour of purging/embargo, input validation, and brute-force references."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from quantproof.errors import QuantProofInputError
from quantproof.validation import CPCV, PurgedKFold, label_intervals
from quantproof.validation.embargo import apply_embargo


def _brute_force_train(test, start, end, embargo_bars=0, n=None):
    """Reference: keep j outside test whose interval intersects no test interval (inclusive),
    then drop the ``embargo_bars`` observations after each test block's label span."""
    test = {int(t) for t in test}
    n = len(start) if n is None else n
    train = []
    for j in range(n):
        if j in test:
            continue
        if any(start[j] <= end[i] and end[j] >= start[i] for i in test):
            continue
        train.append(j)
    if embargo_bars:
        blocks, cur = [], []
        for t in sorted(test):
            if cur and t != cur[-1] + 1:
                blocks.append(cur)
                cur = []
            cur.append(t)
        blocks.append(cur)
        for blk in blocks:
            span_end = max(end[i] for i in blk)
            anchor = max(j for j in range(n) if start[j] <= span_end)
            anchor = max(anchor, blk[-1])
            train = [j for j in train if not (anchor < j <= anchor + embargo_bars)]
    return np.array(train, dtype=np.int64)


# --- sample-time boundaries -----------------------------------------------------------


def test_label_horizon_boundary_is_inclusive():
    # 12 bars, 3 folds of 4. Labels span [i, i+2].
    folds = list(PurgedKFold(3, label_horizon=2).split(np.zeros(12)))
    train0, test0 = folds[0]
    assert test0.tolist() == [0, 1, 2, 3]
    # test span [0, 5]; bars 4 and 5 start inside it -> purged; bar 6 starts after -> kept.
    assert train0.tolist() == [6, 7, 8, 9, 10, 11]
    train1, test1 = folds[1]
    assert test1.tolist() == [4, 5, 6, 7]
    # bar 2's label [2, 4] touches the test start 4 -> purged (inclusive); bar 1 [1, 3] kept.
    assert train1.tolist() == [0, 1, 10, 11]


def test_bar_embargo_boundary():
    folds = list(PurgedKFold(3, embargo=2).split(np.zeros(12)))
    train0, _ = folds[0]
    # test [0..3]; embargo removes exactly bars 4 and 5.
    assert train0.tolist() == [6, 7, 8, 9, 10, 11]
    train2, _ = folds[2]
    # last fold: nothing after it to embargo; nothing before it is removed.
    assert train2.tolist() == list(range(8))


def test_embargo_starts_after_label_span_not_after_test_block():
    # label horizon 2 + embargo 1: span of test [0..3] ends at bar 5 -> purge 4, 5; embargo 6.
    train0, _ = next(iter(PurgedKFold(3, label_horizon=2, embargo=1).split(np.zeros(12))))
    assert train0.tolist() == [7, 8, 9, 10, 11]


# --- event-time boundaries --------------------------------------------------------------


@pytest.fixture
def event_data():
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    lengths = [1, 3, 0, 2, 5, 1, 1, 4, 0, 2, 1, 1]
    ends = pd.Series(idx + pd.to_timedelta(lengths, unit="D"), index=idx)
    return pd.Series(np.arange(12.0), index=idx), ends


def test_event_time_purging_matches_brute_force(event_data):
    X, ends = event_data
    start, end = label_intervals(len(X), ends, index=X.index)
    for train, test in PurgedKFold(4, event_end=ends).split(X):
        np.testing.assert_array_equal(train, _brute_force_train(test, start, end))


def test_event_time_label_ending_at_test_start_is_purged(event_data):
    X, _ = event_data
    idx = X.index
    # obs 3's label ends exactly when the test block (obs 4..7) starts.
    ends = pd.Series(idx, index=idx)
    ends.iloc[3] = idx[4]
    folds = list(PurgedKFold(3, event_end=ends).split(X))
    train1, test1 = folds[1]
    assert test1.tolist() == [4, 5, 6, 7]
    assert 3 not in train1 and 2 in train1


def test_duration_embargo(event_data):
    X, _ = event_data
    idx = X.index
    ends = pd.Series(idx, index=idx)  # zero-length labels
    train0, _ = next(iter(PurgedKFold(3, event_end=ends, embargo="2D").split(X)))
    # test 0..3 ends on day 3; obs starting on days 4 and 5 (positions 4, 5) are embargoed.
    assert train0.tolist() == [6, 7, 8, 9, 10, 11]
    with pytest.raises(QuantProofInputError, match="event-time"):
        list(PurgedKFold(3, embargo="2D").split(X))


def test_event_start_later_than_observation_time(event_data):
    X, ends = event_data
    starts = pd.Series(X.index + pd.Timedelta(days=1), index=X.index)
    with pytest.raises(QuantProofInputError, match="precedes"):
        # obs 2 has a zero-length label; starting it a day later puts start after end.
        list(PurgedKFold(3, event_end=ends, event_start=starts).split(X))
    ok_ends = ends + pd.Timedelta(days=1)
    s, _e = label_intervals(len(X), ok_ends, index=X.index, event_start=starts)
    assert s[0] == pd.Timestamp("2024-01-02").value


# --- regressions: T1-T4 -----------------------------------------------------------------


def test_t1_timezone_mismatch_is_rejected(event_data):
    X, ends = event_data
    aware = X.copy()
    aware.index = aware.index.tz_localize("UTC")
    with pytest.raises(QuantProofInputError, match="not indexed like the data"):
        list(PurgedKFold(3, event_end=ends).split(aware))
    ends_aware_idx = pd.Series(ends.to_numpy(), index=aware.index)  # naive values, aware index
    with pytest.raises(QuantProofInputError, match="timezone-aware and naive"):
        list(PurgedKFold(3, event_end=ends_aware_idx).split(aware))


def test_t1_timezones_are_compared_in_utc():
    idx = pd.date_range("2024-01-01 09:00", periods=8, freq="h", tz="America/New_York")
    ends = pd.Series(idx.tz_convert("Asia/Tokyo") + pd.Timedelta(hours=1), index=idx)
    X = pd.Series(range(8), index=idx)
    s, e = label_intervals(8, ends, index=X.index)
    np.testing.assert_array_equal(e - s, np.full(8, 3_600_000_000_000))


def test_t2_unsorted_observations_are_rejected(event_data):
    X, ends = event_data
    shuffled = X.iloc[[1, 0, *range(2, 12)]]
    with pytest.raises(QuantProofInputError, match="not sorted"):
        list(PurgedKFold(3, event_end=ends.reindex(shuffled.index)).split(shuffled))


def test_t3_event_end_index_must_match_data(event_data):
    X, ends = event_data
    shifted = ends.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    with pytest.raises(QuantProofInputError, match="Reindex"):
        list(PurgedKFold(3, event_end=shifted).split(X))
    with pytest.raises(QuantProofInputError, match="one label end time per observation"):
        list(PurgedKFold(3, event_end=ends.iloc[:-1]).split(X))
    missing = ends.copy()
    missing.iloc[4] = pd.NaT
    with pytest.raises(QuantProofInputError, match="missing timestamps"):
        list(PurgedKFold(3, event_end=missing).split(X))


def test_t4_empty_training_set_raises():
    with pytest.raises(QuantProofInputError, match="removed every training observation"):
        list(PurgedKFold(2, label_horizon=50).split(np.zeros(20)))
    with pytest.raises(QuantProofInputError, match="removed every training observation"):
        list(CPCV(4, 3, label_horizon=30).split(np.zeros(40)))


def test_conflicting_label_arguments():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    ends = pd.Series(idx, index=idx)
    with pytest.raises(QuantProofInputError, match="not both"):
        PurgedKFold(2, event_end=ends, t1=ends)
    with pytest.raises(QuantProofInputError, match="not both"):
        list(PurgedKFold(2, event_end=ends, label_horizon=2).split(pd.Series(range(6), index=idx)))
    with pytest.raises(QuantProofInputError, match="requires event end"):
        label_intervals(6, None, event_start=idx)
    assert PurgedKFold(2, t1=ends).event_end is ends


# --- CPCV reference ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_groups", "k", "horizon", "embargo"), [(6, 2, 0, 0), (6, 2, 3, 2), (5, 3, 1, 1)]
)
def test_cpcv_matches_brute_force(n_groups, k, horizon, embargo):
    n = 60
    cv = CPCV(n_groups, k, label_horizon=horizon, embargo=embargo)
    start = np.arange(n)
    end = start + horizon
    groups = np.array_split(np.arange(n), n_groups)
    combos = list(combinations(range(n_groups), k))
    splits = list(cv.split(np.zeros(n)))
    assert len(splits) == len(combos) == cv.n_splits
    for (train, test), combo in zip(splits, combos, strict=True):
        expected_test = np.concatenate([groups[g] for g in combo])
        np.testing.assert_array_equal(test, expected_test)
        np.testing.assert_array_equal(
            train, _brute_force_train(test, start, end, embargo_bars=embargo, n=n)
        )
        assert not set(train) & set(test)


def test_cpcv_paths_cover_every_observation_once():
    n = 60
    cv = CPCV(6, 2)
    preds = [np.asarray(test, dtype=float) for _, test in cv.split(np.zeros(n))]
    paths = cv.assemble_paths(np.zeros(n), preds)
    assert paths.shape == (cv.n_paths, n)
    for p in paths:
        np.testing.assert_array_equal(p, np.arange(n, dtype=float))


def test_apply_embargo_rejects_non_numeric():
    with pytest.raises(QuantProofInputError):
        apply_embargo(np.arange(5), np.arange(5, 10), 10, [1])  # type: ignore[arg-type]


def test_methodology_diagram_is_exact():
    """docs/methodology/purging-and-embargo.md draws exactly these splits."""
    folds = list(PurgedKFold(3, label_horizon=2, embargo=1).split(np.zeros(12)))
    assert folds[0][1].tolist() == [0, 1, 2, 3]
    assert folds[0][0].tolist() == [7, 8, 9, 10, 11]
    assert folds[1][1].tolist() == [4, 5, 6, 7]
    assert folds[1][0].tolist() == [0, 1, 11]


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_event_time_purging_is_resolution_independent(event_data, unit):
    X, ends = event_data
    ref = [tr.tolist() for tr, _ in PurgedKFold(4, event_end=ends, embargo="1D").split(X)]
    Xu = X.copy()
    Xu.index = Xu.index.as_unit(unit)
    ends_u = pd.Series(pd.DatetimeIndex(ends.to_numpy()).as_unit(unit), index=Xu.index)
    got = [tr.tolist() for tr, _ in PurgedKFold(4, event_end=ends_u, embargo="1D").split(Xu)]
    assert got == ref


@pytest.mark.parametrize("unit", ["s", "us"])
def test_perturbation_cutoff_is_resolution_independent(unit):
    from quantproof.analyzers.causal import perturb_after
    from quantproof.data.synthetic import generate_prices

    px = generate_prices(50, seed=1)
    pu = px.copy()
    pu.index = pu.index.as_unit(unit)
    cutoff = px.index[20]
    a = perturb_after(px, cutoff, "extreme", seed=3)
    b = perturb_after(pu, cutoff, "extreme", seed=3)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
