"""Combinatorial Purged Cross-Validation (López de Prado, 2018, ch. 12).

The sample is divided into ``N = n_groups`` contiguous groups. Every combination
of ``k = n_test_groups`` groups forms one test set, giving ``C(N, k)`` splits.
Training sets are the remaining groups after purging and embargo around *each*
contiguous test block (adjacent test groups form one block).

Every group is a test group in ``C(N-1, k-1)`` splits, so the out-of-sample
predictions can be stitched into ``phi = C(N, k) * k / N`` complete backtest
paths, each covering the full sample exactly once.

Complexity: the number of splits grows as ``C(N, k)``; ``N=6, k=2`` gives 15
splits and 5 paths, ``N=10, k=2`` gives 45 splits and 9 paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from math import comb
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


class CPCV:
    """Combinatorial purged cross-validation splitter.

    Example
    -------
    >>> cv = CPCV(n_groups=6, n_test_groups=2, embargo=0.01)
    >>> cv.get_n_splits(), cv.n_paths
    (15, 5)
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        *,
        embargo: EmbargoSpec = 0.0,
        event_end: pd.Series | None = None,
        event_start: pd.Series | None = None,
        t1: pd.Series | None = None,
        label_horizon: int = 0,
    ) -> None:
        if n_groups < 2:
            raise QuantProofInputError("n_groups must be >= 2.")
        if not 1 <= n_test_groups < n_groups:
            raise QuantProofInputError("n_test_groups must satisfy 1 <= k < n_groups.")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo = embargo
        self.event_end = resolve_event_end(event_end, t1)
        self.event_start = event_start
        self.label_horizon = label_horizon

    @property
    def t1(self) -> pd.Series | None:
        """Alias of :attr:`event_end`."""
        return self.event_end

    @property
    def n_splits(self) -> int:
        """``C(N, k)``."""
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """``phi = C(N, k) * k / N`` complete backtest paths."""
        return self.n_splits * self.n_test_groups // self.n_groups

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of train/test splits."""
        return self.n_splits

    def group_combinations(self) -> list[tuple[int, ...]]:
        """Test-group tuples in the deterministic (lexicographic) split order."""
        return list(combinations(range(self.n_groups), self.n_test_groups))

    def groups(self, X: Any) -> list[np.ndarray]:
        """Positions of each contiguous group."""
        n = num_samples(X)
        if n < self.n_groups:
            raise QuantProofInputError(f"Cannot form {self.n_groups} groups from {n} samples.")
        return list(np.array_split(np.arange(n), self.n_groups))

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_positions, test_positions)`` for each group combination."""
        n = num_samples(X)
        timed = self.event_end is not None
        index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) and timed else None
        start, end = label_intervals(
            n, self.event_end, self.label_horizon, index=index, event_start=self.event_start
        )
        grp = self.groups(X)
        positions = np.arange(n)
        for s, combo in enumerate(self.group_combinations()):
            test = np.concatenate([grp[g] for g in combo])
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
            yield require_training_data(train, s, "CPCV"), test

    def paths(self) -> list[list[tuple[int, int]]]:
        """Backtest paths as lists of ``(group, split_index)`` pairs.

        Path ``p`` uses, for group ``g``, the ``p``-th split (in split order) in which ``g`` is
        a test group. Each path therefore covers every group exactly once.
        """
        combos = self.group_combinations()
        per_group: dict[int, list[int]] = {g: [] for g in range(self.n_groups)}
        for s, combo in enumerate(combos):
            for g in combo:
                per_group[g].append(s)
        return [[(g, per_group[g][p]) for g in range(self.n_groups)] for p in range(self.n_paths)]

    def assemble_paths(
        self, X: Any, split_predictions: list[np.ndarray] | dict[int, np.ndarray]
    ) -> np.ndarray:
        """Stitch per-split OOS outputs into a ``(n_paths, n_samples)`` array.

        ``split_predictions[s]`` must be aligned with the test positions yielded by split ``s``
        (i.e. concatenated test groups in ascending group order).
        """
        n = num_samples(X)
        grp = self.groups(X)
        combos = self.group_combinations()
        out = np.full((self.n_paths, n), np.nan)
        for p, path in enumerate(self.paths()):
            for g, s in path:
                combo = combos[s]
                preds = np.asarray(split_predictions[s])
                offset = sum(len(grp[c]) for c in combo if c < g)
                out[p, grp[g]] = preds[offset : offset + len(grp[g])]
        return out
