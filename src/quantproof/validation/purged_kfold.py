"""Purged K-Fold cross-validation (López de Prado, 2018, ch. 7).

Folds are contiguous blocks in time. For each test fold, training observations
whose *label interval* overlaps the test fold's label span are purged, and an
embargo removes observations immediately after the test fold.

Label intervals are given either by ``t1`` (a Series mapping each observation's
timestamp to the time its label is resolved, e.g. the barrier-touch time) or by a
fixed ``label_horizon`` in bars (a label at bar i that uses returns through
i + h has interval ``[i, i + h]``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofInputError
from quantproof.validation.embargo import apply_embargo
from quantproof.validation.temporal import label_intervals, num_samples, purge


class PurgedKFold:
    """K-fold splitter with purging and embargo for overlapping labels.

    Parameters
    ----------
    n_splits: number of contiguous folds (>= 2).
    t1: optional Series of label end times indexed by observation time.
    label_horizon: label length in bars when ``t1`` is not supplied.
    embargo: fraction of the sample (``< 1``) or number of bars (``>= 1``).
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        t1: pd.Series | None = None,
        label_horizon: int = 0,
        embargo: float | int = 0.0,
    ) -> None:
        if n_splits < 2:
            raise QuantProofInputError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.t1 = t1
        self.label_horizon = label_horizon
        self.embargo = embargo

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_positions, test_positions)``."""
        n = num_samples(X)
        if n < self.n_splits:
            raise QuantProofInputError(f"Cannot make {self.n_splits} folds from {n} samples.")
        index = (
            X.index if isinstance(X, (pd.DataFrame, pd.Series)) and self.t1 is not None else None
        )
        start, end = label_intervals(n, self.t1, self.label_horizon, index=index)
        positions = np.arange(n)
        for test in np.array_split(positions, self.n_splits):
            train = np.setdiff1d(positions, test, assume_unique=True)
            train = purge(train, test, start, end)
            train = apply_embargo(train, test, n, self.embargo, label_start=start, label_end=end)
            yield train, test

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of folds."""
        return self.n_splits
