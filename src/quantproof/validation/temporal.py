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


def _as_utc_ns(values: Any, name: str) -> tuple[np.ndarray, bool]:
    """Nanoseconds since the epoch and whether the input was timezone-aware."""
    idx = pd.DatetimeIndex(values)
    if idx.hasnans:
        raise QuantProofInputError(
            f"{name} contains missing timestamps; drop observations whose label is unresolved."
        )
    aware = idx.tz is not None
    if aware:
        idx = idx.tz_convert("UTC")
    return datetime_ns(idx), aware


def label_intervals(
    n_samples: int,
    t1: pd.Series | np.ndarray | None = None,
    label_horizon: int = 0,
    index: pd.Index | None = None,
    *,
    event_start: pd.Series | pd.Index | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(start, end)`` arrays describing each observation's label interval.

    Two modes, never mixed:

    * **Sample-time** (``t1`` is None): intervals are bar positions ``[i, i + label_horizon]``.
      Use when every label spans the same number of bars.
    * **Event-time** (``t1`` = event end times): intervals are ``[event_start_i, t1_i]`` in
      nanoseconds since the epoch (UTC for timezone-aware input). ``event_start`` defaults to
      the observation timestamps (``t1.index`` or ``index``). Use when labels have variable
      length (e.g. triple-barrier labels).

    Validation (each raises :class:`QuantProofInputError`): one entry per observation; if
    both ``t1`` (Series) and ``index`` are given, their indexes must match exactly; start and
    end must both be timezone-aware or both naive; no missing times; observation start times
    must be non-decreasing (data sorted by time); every end ≥ its start.
    """
    if t1 is None:
        if event_start is not None:
            raise QuantProofInputError("event_start requires event end times (event_end / t1).")
        if label_horizon < 0:
            raise QuantProofInputError("label_horizon must be >= 0.")
        start = np.arange(n_samples, dtype=np.int64)
        return start, start + int(label_horizon)
    if label_horizon:
        raise QuantProofInputError(
            "Give either label_horizon (sample-time labels) or event end times, not both."
        )
    if len(t1) != n_samples:
        raise QuantProofInputError(
            f"Event end times have {len(t1)} entries but the data has {n_samples} rows; supply "
            "one label end time per observation."
        )
    if isinstance(t1, pd.Series):
        if index is not None and not t1.index.equals(pd.Index(index)):
            raise QuantProofInputError(
                "The event end Series is not indexed like the data (different timestamps or "
                "order). Reindex it to the data's index first: t1 = t1.reindex(X.index)."
            )
        obs_times: Any = t1.index if index is None else index
        end_values: Any = t1.to_numpy()
    else:
        obs_times = index
        end_values = np.asarray(t1)
    if event_start is not None:
        if len(event_start) != n_samples:
            raise QuantProofInputError("event_start must have one entry per observation.")
        obs_times = event_start.to_numpy() if isinstance(event_start, pd.Series) else event_start
    if obs_times is None:
        # Integer label ends without timestamps: positions.
        start = np.arange(n_samples, dtype=np.int64)
        end = np.asarray(end_values).astype(np.int64)
    else:
        start, start_aware = _as_utc_ns(obs_times, "event_start")
        end, end_aware = _as_utc_ns(end_values, "event_end")
        if start_aware != end_aware:
            raise QuantProofInputError(
                "Observation times and event end times mix timezone-aware and naive timestamps. "
                "Localize both to the same timezone (e.g. tz_localize('UTC'))."
            )
    if n_samples > 1 and np.any(np.diff(start) < 0):
        bad = int(np.argmax(np.diff(start) < 0)) + 1
        raise QuantProofInputError(
            f"Observation times are not sorted (position {bad} is earlier than {bad - 1}). "
            "Sort the data by time before splitting; purging assumes time order."
        )
    if np.any(end < start):
        bad = int(np.argmax(end < start))
        raise QuantProofInputError(
            f"Label end precedes label start at position {bad}; event end times must be >= "
            "the observation time."
        )
    return start, end


def resolve_event_end(event_end: Any, t1: Any) -> Any:
    """``event_end`` or its alias ``t1`` (giving both is an error)."""
    if event_end is not None and t1 is not None:
        raise QuantProofInputError("Pass event_end or its alias t1, not both.")
    return event_end if event_end is not None else t1


def require_training_data(train: np.ndarray, split: int, what: str) -> np.ndarray:
    """Raise a helpful error when purging/embargo removed every training observation."""
    if train.size == 0:
        raise QuantProofInputError(
            f"{what} split {split}: purging and embargo removed every training observation. "
            "Label spans or the embargo are too long relative to the fold size; use fewer "
            "folds, a shorter label horizon, or a smaller embargo."
        )
    return train


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
