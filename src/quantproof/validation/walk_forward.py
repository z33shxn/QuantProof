"""Walk-forward (forward-chaining) validation.

::

    expanding                         rolling
    TRAIN TRAIN | TEST                TRAIN TRAIN | TEST
    TRAIN TRAIN TRAIN | TEST                TRAIN TRAIN | TEST
    TRAIN TRAIN TRAIN TRAIN | TEST                TRAIN TRAIN | TEST

Each test window starts ``gap`` bars after the end of its training window. Test
windows never overlap and are always strictly after their training data.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantproof.errors import QuantProofInputError
from quantproof.validation.temporal import num_samples


@dataclass(frozen=True)
class WalkForwardWindow:
    """Positions of one walk-forward fold (half-open ``[start, end)`` ranges)."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class WalkForward:
    """Walk-forward splitter with expanding or rolling training windows.

    Parameters
    ----------
    train_size:
        Initial (expanding) or fixed (rolling) training length. A float in (0, 1) is a
        fraction of the sample.
    test_size:
        Length of each test window (int or fraction). If omitted, ``n_splits`` must be given
        and the remaining data is divided into that many equal test windows.
    step:
        Bars between consecutive test-window starts (default: ``test_size``).
    gap:
        Bars dropped between the end of training and the start of testing.
    expanding:
        Expanding (anchored) training windows when True, rolling when False.
    n_splits:
        Alternative to ``test_size``.
    """

    def __init__(
        self,
        train_size: int | float = 0.5,
        test_size: int | float | None = None,
        *,
        step: int | None = None,
        gap: int = 0,
        expanding: bool = True,
        n_splits: int | None = None,
    ) -> None:
        if test_size is None and n_splits is None:
            raise QuantProofInputError("Provide test_size or n_splits.")
        if gap < 0:
            raise QuantProofInputError("gap must be >= 0.")
        if step is not None and step <= 0:
            raise QuantProofInputError("step must be positive.")
        if n_splits is not None and n_splits < 1:
            raise QuantProofInputError("n_splits must be >= 1.")
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
        self.gap = gap
        self.expanding = expanding
        self.n_splits = n_splits

    @staticmethod
    def _resolve(size: int | float, n: int, name: str) -> int:
        value = int(np.floor(size * n)) if isinstance(size, float) and 0 < size < 1 else int(size)
        if value <= 0:
            raise QuantProofInputError(f"{name}={size} resolves to {value} bars for n={n}.")
        return value

    def windows(self, X: Any) -> list[WalkForwardWindow]:
        """All folds as explicit position ranges."""
        n = num_samples(X)
        train = self._resolve(self.train_size, n, "train_size")
        if self.test_size is not None:
            test = self._resolve(self.test_size, n, "test_size")
        else:
            assert self.n_splits is not None
            test = (n - train - self.gap) // self.n_splits
            if test <= 0:
                raise QuantProofInputError(
                    f"Not enough data for {self.n_splits} test windows after train={train}, "
                    f"gap={self.gap} (n={n})."
                )
        step = self.step or test
        out: list[WalkForwardWindow] = []
        train_end = train
        fold = 0
        while True:
            test_start = train_end + self.gap
            test_end = test_start + test
            if test_end > n:
                break
            train_start = 0 if self.expanding else max(0, train_end - train)
            out.append(WalkForwardWindow(fold, train_start, train_end, test_start, test_end))
            fold += 1
            if self.n_splits is not None and fold >= self.n_splits:
                break
            train_end += step
        if not out:
            raise QuantProofInputError(
                f"No walk-forward fold fits: n={n}, train={train}, test={test}, gap={self.gap}."
            )
        return out

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_positions, test_positions)`` (scikit-learn compatible)."""
        for w in self.windows(X):
            yield np.arange(w.train_start, w.train_end), np.arange(w.test_start, w.test_end)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of folds (requires ``X`` unless ``n_splits`` was given)."""
        if X is None:
            if self.n_splits is None:
                raise QuantProofInputError("Pass X to compute the number of splits.")
            return self.n_splits
        return len(self.windows(X))

    def diagram(self, X: Any, width: int = 60) -> str:
        """ASCII diagram of the folds (``=`` train, ``#`` test, `` `` unused/gap)."""
        n = num_samples(X)
        lines = []
        for w in self.windows(X):
            row = [" "] * width
            for pos_range, ch in (
                ((w.train_start, w.train_end), "="),
                ((w.test_start, w.test_end), "#"),
            ):
                a = int(pos_range[0] * width / n)
                b = max(a + 1, int(np.ceil(pos_range[1] * width / n)))
                for i in range(a, min(b, width)):
                    row[i] = ch
            lines.append(f"fold {w.fold:>2} |{''.join(row)}|")
        return "\n".join(lines)
