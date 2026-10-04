"""Embargo: removing training observations that immediately follow a test block.

Even after purging overlapping labels, features computed on rolling windows and
serially correlated returns mean that observations just *after* a test block
carry information about it. López de Prado (2018, ch. 7) therefore drops a
fraction ``h`` of observations after each test block from the training set.

QuantProof expresses the embargo either as a fraction of the sample
(``0 <= embargo < 1``, converted with ``ceil(embargo * n)``) or as an integer
number of bars (``embargo >= 1``).
"""

from __future__ import annotations

import math

import numpy as np

from quantproof.errors import QuantProofInputError


def embargo_size(n_samples: int, embargo: float | int) -> int:
    """Number of bars embargoed after each test block.

    Fractions in ``[0, 1)`` are interpreted as a share of ``n_samples``; integers
    ``>= 1`` as a bar count.
    """
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
    embargo: float | int,
    *,
    label_start: np.ndarray | None = None,
    label_end: np.ndarray | None = None,
) -> np.ndarray:
    """Remove ``h`` training positions after each contiguous test block.

    The embargo window starts after the *end of the test block's label span*, following
    López de Prado (2018, snippet 7.3): if the last test label is resolved at time
    ``T``, positions whose observation time is ``<= T`` are handled by purging and the
    next ``h`` positions are embargoed. Without label intervals the window starts right
    after the last test position.
    """
    h = embargo_size(n_samples, embargo)
    train = np.asarray(train_indices, dtype=np.int64)
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
