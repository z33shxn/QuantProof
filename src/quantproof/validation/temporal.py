"""Shared helpers for temporal splitting: label intervals, purging, simple splits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantproof._utils import datetime_ns
from quantproof.errors import QuantProofInputError
from quantproof.validation.embargo import contiguous_blocks


def num_samples(X: Any) -> int:
    """Length of an array-like (DataFrame, Series, ndarray, list, or an int)."""
    if isinstance(X, (int, np.integer)):
        return int(X)
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)


def label_intervals(
    n_samples: int,
    t1: pd.Series | np.ndarray | None = None,
    label_horizon: int = 0,
    index: pd.Index | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(start, end)`` arrays describing each observation's label interval.

    * With ``t1`` (a Series indexed by observation start time whose values are the time at
      which the label is resolved), intervals are in nanoseconds since epoch.
    * Otherwise intervals are in bar positions: ``[i, i + label_horizon]``.
    """
    if t1 is None:
        if label_horizon < 0:
            raise QuantProofInputError("label_horizon must be >= 0.")
        start = np.arange(n_samples, dtype=np.int64)
        return start, start + int(label_horizon)
    if isinstance(t1, pd.Series):
        if len(t1) != n_samples:
            raise QuantProofInputError(
                f"t1 has {len(t1)} entries but the data has {n_samples} rows; t1 must contain one "
                "label end time per observation."
            )
        start_idx = pd.DatetimeIndex(t1.index if index is None else index)
        end_idx = pd.DatetimeIndex(t1.to_numpy())
        start = datetime_ns(start_idx)
        end = datetime_ns(end_idx)
    else:
        arr = np.asarray(t1)
        if arr.shape[0] != n_samples:
            raise QuantProofInputError("t1 must have one entry per observation.")
        if index is None:
            start = np.arange(n_samples, dtype=np.int64)
            end = arr.astype(np.int64)
        else:
            start = datetime_ns(index)
            end = datetime_ns(arr)
    if np.any(end < start):
        bad = int(np.argmax(end < start))
        raise QuantProofInputError(
            f"Label end precedes label start at position {bad}; t1 must be >= the observation time."
        )
    return start, end


def purge(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Drop training observations whose label interval overlaps any contiguous test block.

    For each contiguous test block ``[a, b]`` (positions), the test span is
    ``[start[a], max(end[a..b])]``; training observation ``j`` is purged when
    ``start[j] <= span_end`` and ``end[j] >= span_start``.
    """
    train = np.asarray(train_indices, dtype=np.int64)
    if train.size == 0:
        return train
    keep = np.ones(train.size, dtype=bool)
    for a, b in contiguous_blocks(test_indices):
        span_start = start[a]
        span_end = end[a : b + 1].max()
        overlap = (start[train] <= span_end) & (end[train] >= span_start)
        keep &= ~overlap
    return train[keep]


def temporal_train_test_split(
    data: Any, test_size: float | int = 0.25, gap: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split: the first rows train, the last rows test, optional gap.

    Returns integer position arrays ``(train, test)``.
    """
    n = num_samples(data)
    n_test = (
        round(test_size * n) if isinstance(test_size, float) and test_size < 1 else int(test_size)
    )
    if not 0 < n_test < n:
        raise QuantProofInputError(f"test_size={test_size} leaves no train or test data (n={n}).")
    if gap < 0 or n - n_test - gap <= 0:
        raise QuantProofInputError(f"gap={gap} leaves no training data.")
    test = np.arange(n - n_test, n)
    train = np.arange(0, n - n_test - gap)
    return train, test


def assert_no_leakage(
    train: np.ndarray,
    test: np.ndarray,
    start: np.ndarray | None = None,
    end: np.ndarray | None = None,
) -> None:
    """Raise if train and test share positions or (given intervals) overlapping labels."""
    shared = np.intersect1d(train, test)
    if shared.size:
        raise AssertionError(f"{shared.size} positions are in both train and test.")
    if start is not None and end is not None:
        remaining = purge(train, test, start, end)
        if remaining.size != np.asarray(train).size:
            raise AssertionError("Training labels overlap test labels; purging is required.")
