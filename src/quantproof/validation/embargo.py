"""Embargo: removing training observations that immediately follow a test block.

Even after purging overlapping labels, features computed on rolling windows and
serially correlated returns mean that observations just *after* a test block
carry information about it. López de Prado (2018, ch. 7) therefore drops a
fraction ``h`` of observations after each test block from the training set.

QuantProof expresses the embargo as

* a fraction of the sample (``0 <= embargo < 1``, converted with ``ceil(embargo * n)``),
* an integer number of bars (``embargo >= 1``), or
* a duration (``pd.Timedelta`` / ``"5D"``), only with event-time labels: training
  observations that *start* within that duration after the end of the test block's label
  span are removed.

Boundaries: with a bar embargo of ``h`` and a test block whose label span ends at
position ``T``, positions ``T+1 … T+h`` (inclusive) are removed. With a duration ``d``,
observations with ``span_end < start <= span_end + d`` are removed.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofInputError

EmbargoSpec = float | int | pd.Timedelta | _dt.timedelta | str


def is_time_embargo(embargo: Any) -> bool:
    """True for duration embargoes (Timedelta, timedelta, numpy timedelta64 or a string)."""
    return isinstance(embargo, (pd.Timedelta, _dt.timedelta, np.timedelta64, str))


def embargo_ns(embargo: Any) -> int:
    """A duration embargo in nanoseconds."""
    try:
        td = pd.Timedelta(embargo)
    except (ValueError, TypeError) as exc:
        raise QuantProofInputError(f"Cannot interpret embargo={embargo!r} as a duration.") from exc
    if td < pd.Timedelta(0):
        raise QuantProofInputError(f"embargo must be non-negative, got {embargo}.")
    return int(td.value)


def embargo_size(n_samples: int, embargo: EmbargoSpec) -> int:
    """Number of bars embargoed after each test block.

    Fractions in ``[0, 1)`` are interpreted as a share of ``n_samples``; integers
    ``>= 1`` as a bar count.
    """
    if is_time_embargo(embargo):
        raise QuantProofInputError(
            "A duration embargo has no bar count; it needs event-time labels (event_end)."
        )
    if isinstance(embargo, bool) or not isinstance(embargo, (int, float, np.integer, np.floating)):
        raise QuantProofInputError(f"embargo must be a number or a duration, got {embargo!r}.")
    if embargo < 0:
        raise QuantProofInputError(f"embargo must be non-negative, got {embargo}.")
    if embargo == 0:
        return 0
    if embargo < 1:
        return math.ceil(embargo * n_samples)
    if float(embargo).is_integer():
        return int(embargo)
    raise QuantProofInputError(
        f"embargo={embargo} is ambiguous: use a fraction in [0, 1) or an integer number of bars."
    )


def contiguous_blocks(indices: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted integer positions into inclusive ``(start, end)`` contiguous runs."""
    idx = np.unique(np.asarray(indices, dtype=np.int64))
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def apply_embargo(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
    embargo: EmbargoSpec,
    *,
    label_start: np.ndarray | None = None,
    label_end: np.ndarray | None = None,
    time_based: bool = False,
) -> np.ndarray:
    """Remove ``h`` training positions after each contiguous test block.

    The embargo window starts after the *end of the test block's label span*, following
    López de Prado (2018, snippet 7.3): if the last test label is resolved at time
    ``T``, positions whose observation time is ``<= T`` are handled by purging and the
    next ``h`` positions are embargoed. Without label intervals the window starts right
    after the last test position. ``time_based`` states that ``label_start``/``label_end``
    are timestamps (ns), which a duration embargo requires.
    """
    train = np.asarray(train_indices, dtype=np.int64)
    if is_time_embargo(embargo):
        if not time_based or label_start is None or label_end is None:
            raise QuantProofInputError(
                "A duration embargo needs event-time labels (event_end); with sample-time "
                "labels use a bar count or a fraction of the sample."
            )
        d = embargo_ns(embargo)
        if d == 0 or train.size == 0:
            return train
        keep = np.ones(train.size, dtype=bool)
        for a, b in contiguous_blocks(test_indices):
            span_end = label_end[a : b + 1].max()
            st = label_start[train]
            keep &= ~((st > span_end) & (st <= span_end + d))
        return train[keep]
    h = embargo_size(n_samples, embargo)
    if h == 0 or train.size == 0:
        return train
    mask = np.ones(train.size, dtype=bool)
    for a, b in contiguous_blocks(test_indices):
        anchor = b
        if label_start is not None and label_end is not None:
            span_end = label_end[a : b + 1].max()
            anchor = max(b, int(np.searchsorted(label_start, span_end, side="right")) - 1)
        mask &= ~((train > anchor) & (train <= anchor + h))
    return train[mask]
