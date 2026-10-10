"""Purged K-Fold cross-validation (López de Prado, 2018, ch. 7).

Folds are contiguous blocks in time. For each test fold, training observations
whose *label interval* overlaps the test fold's label span are purged, and an
embargo removes observations immediately after the test fold.

Sample-time vs event-time
-------------------------
Folds are always contiguous blocks of *observations* (sample-time splitting). What
differs is how label overlap is measured for purging:

* sample-time labels, ``label_horizon=h``: the label at bar ``i`` uses data through bar
  ``i + h``, interval ``[i, i + h]`` in positions;
* event-time labels, ``event_end`` (alias ``t1``): a Series mapping each observation's
  timestamp to the time its label is resolved (e.g. a barrier touch), optionally with
  ``event_start`` when the label starts later than the observation. Intervals are compared
  in time, so variable-length labels are purged correctly.

Boundary convention
-------------------
A training observation is purged when its interval ``[s_j, e_j]`` intersects the test
block's span ``[s_a, max e_{a..b}]``, with both ends **inclusive** (a label that ends exactly
when the test span starts is purged). The embargo is applied after purging (see
:mod:`quantproof.validation.embargo`).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofInputError
from quantproof.validation.embargo import EmbargoSpec, apply_embargo
from quantproof.validation.temporal import (
    label_intervals,
    num_samples,
    purge,
    require_training_data,
    resolve_event_end,
)


class PurgedKFold:
    """K-fold splitter with purging and embargo for overlapping labels.

    Parameters
    ----------
    n_splits: number of contiguous folds (>= 2).
    event_end: Series of label end times indexed by observation time (event-time labels).
    event_start: optional label start times (default: the observation times).
    t1: alias of ``event_end`` (López de Prado's notation).
    label_horizon: label length in bars for sample-time labels.
    embargo: fraction of the sample (``< 1``), number of bars (``>= 1``) or, with
        event-time labels, a duration such as ``"5D"``.

    Example
    -------
    >>> import pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=10, freq="D")
    >>> ends = pd.Series(idx + pd.Timedelta(days=2), index=idx)
    >>> cv = PurgedKFold(2, event_end=ends)
    >>> [tr.tolist() for tr, te in cv.split(pd.Series(range(10), index=idx))]
    [[7, 8, 9], [0, 1, 2]]
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        event_end: pd.Series | None = None,
        event_start: pd.Series | None = None,
        t1: pd.Series | None = None,
        label_horizon: int = 0,
        embargo: EmbargoSpec = 0.0,
    ) -> None:
        if n_splits < 2:
            raise QuantProofInputError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.event_end = resolve_event_end(event_end, t1)
        self.event_start = event_start
        self.label_horizon = label_horizon
        self.embargo = embargo

    @property
    def t1(self) -> pd.Series | None:
        """Alias of :attr:`event_end`."""
        return self.event_end

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_positions, test_positions)``."""
        n = num_samples(X)
        if n < self.n_splits:
            raise QuantProofInputError(f"Cannot make {self.n_splits} folds from {n} samples.")
        timed = self.event_end is not None
        index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) and timed else None
        start, end = label_intervals(
            n, self.event_end, self.label_horizon, index=index, event_start=self.event_start
        )
        positions = np.arange(n)
        for k, test in enumerate(np.array_split(positions, self.n_splits)):
            train = np.setdiff1d(positions, test, assume_unique=True)
            train = purge(train, test, start, end)
            train = apply_embargo(
                train,
                test,
                n,
                self.embargo,
                label_start=start,
                label_end=end,
                time_based=timed,
            )
            yield require_training_data(train, k, "PurgedKFold"), test

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of folds."""
        return self.n_splits
