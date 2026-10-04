"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017). "The Probability
of Backtest Overfitting." Journal of Computational Finance 20(4).

Procedure implemented
---------------------
1. ``M`` is a ``T x N`` matrix of per-period returns of ``N`` strategy configurations.
2. Rows are split into ``S`` (even) contiguous blocks.
3. For every combination ``c`` of ``S/2`` blocks: in-sample = those blocks,
   out-of-sample = the complement (time order inside each set is irrelevant to the
   Sharpe metric).
4. ``n*`` = configuration with the best in-sample Sharpe.
5. ``w = rank_OOS(n*) / (N + 1)`` where rank 1 is the worst OOS Sharpe and ties use
   average ranks; ``lambda = ln(w / (1 - w))``.
6. ``PBO = share of combinations with lambda <= 0`` — i.e. the in-sample winner
   performs at or below the out-of-sample median. Counting ``lambda == 0`` as
   overfit is a slightly conservative choice for odd ``N``.

Also reported: the regression slope of OOS on IS Sharpe of the selected
configuration (performance degradation) and the probability that the selected
configuration has a negative OOS Sharpe ("probability of loss").

Complexity: ``C(S, S/2)`` combinations, each ``O(N)`` after pre-computing block sums.
``S = 16`` gives 12,870 combinations; ``max_combinations`` sub-samples
deterministically above a cap.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quantproof.errors import QuantProofInputError


@dataclass(frozen=True)
class PBOResult:
    """Outcome of CSCV."""

    pbo: float
    n_partitions: int
    n_combinations: int
    n_combinations_total: int
    n_strategies: int
    n_observations: int
    logits: list[float] = field(repr=False)
    is_sharpe_selected: list[float] = field(repr=False)
    oos_sharpe_selected: list[float] = field(repr=False)
    oos_rank_selected: list[float] = field(repr=False)
    selected_index: list[int] = field(repr=False)
    degradation_slope: float
    degradation_intercept: float
    prob_oos_loss: float
    median_logit: float
    periods_per_year: float

    def to_dict(self, include_arrays: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_arrays:
            for k in (
                "logits",
                "is_sharpe_selected",
                "oos_sharpe_selected",
                "oos_rank_selected",
                "selected_index",
            ):
                d.pop(k)
        d["interpretation"] = interpret_pbo(self.pbo)
        return d


def interpret_pbo(pbo: float) -> str:
    """Plain-language reading of a PBO value."""
    if not math.isfinite(pbo):
        return "PBO could not be computed."
    if pbo <= 0.1:
        return "The in-sample selection usually remains above the out-of-sample median."
    if pbo < 0.5:
        return (
            "The in-sample winner falls to the bottom half out of sample in a minority of "
            "splits; some selection bias is present."
        )
    return (
        "The in-sample winner is more likely than not to rank in the bottom half out of "
        "sample: the selection process is likely overfit."
    )


def probability_of_backtest_overfitting(
    returns: Any,
    *,
    n_partitions: int = 10,
    max_combinations: int | None = 5000,
    periods_per_year: float = 252.0,
    seed: int = 42,
) -> PBOResult:
    """Estimate PBO with CSCV on a ``T x N`` returns matrix (rows = time)."""
    m = (
        returns.to_numpy(dtype=float)
        if isinstance(returns, pd.DataFrame)
        else np.asarray(returns, dtype=float)
    )
    if m.ndim != 2:
        raise QuantProofInputError("PBO needs a 2-D (time x strategies) returns matrix.")
    t, n = m.shape
    if n < 2:
        raise QuantProofInputError("PBO needs at least two strategy configurations.")
    if n_partitions < 2 or n_partitions % 2:
        raise QuantProofInputError("n_partitions must be an even number >= 2.")
    if t < 2 * n_partitions:
        raise QuantProofInputError(
            f"Need at least {2 * n_partitions} observations for {n_partitions} partitions; got {t}."
        )
    if not np.isfinite(m).all():
        raise QuantProofInputError(
            "Returns matrix contains NaN/inf. Align the configurations on a common sample "
            "(e.g. drop warm-up rows) before computing PBO."
        )
    blocks = np.array_split(np.arange(t), n_partitions)
    cnt = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.stack([m[b].sum(axis=0) for b in blocks])  # S x N
    s2 = np.stack([(m[b] ** 2).sum(axis=0) for b in blocks])

    all_combos = list(combinations(range(n_partitions), n_partitions // 2))
    total = len(all_combos)
    if max_combinations is not None and total > max_combinations:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(total, size=max_combinations, replace=False))
        combos = [all_combos[i] for i in chosen]
    else:
        combos = all_combos
    ind = np.zeros((len(combos), n_partitions))
    for i, c in enumerate(combos):
        ind[i, list(c)] = 1.0

    def _sharpe(w: np.ndarray) -> np.ndarray:
        k = w @ cnt
        mean = (w @ s1) / k[:, None]
        var = ((w @ s2) - k[:, None] * mean**2) / (k[:, None] - 1)
        sd = np.sqrt(np.maximum(var, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(sd > 0, mean / sd, np.nan)
        return sr

    is_sr = _sharpe(ind)
    oos_sr = _sharpe(1.0 - ind)
    # Configurations with undefined Sharpe never win in-sample and rank lowest OOS.
    is_filled = np.where(np.isfinite(is_sr), is_sr, -np.inf)
    oos_filled = np.where(np.isfinite(oos_sr), oos_sr, -np.inf)
    best = np.argmax(is_filled, axis=1)
    ranks = np.apply_along_axis(rankdata, 1, oos_filled)  # 1 = worst, average ties
    sel_rank = ranks[np.arange(len(combos)), best]
    w = sel_rank / (n + 1.0)
    logits = np.log(w / (1.0 - w))
    pbo = float(np.mean(logits <= 0))
    is_sel = is_sr[np.arange(len(combos)), best]
    oos_sel = oos_sr[np.arange(len(combos)), best]
    ok = np.isfinite(is_sel) & np.isfinite(oos_sel)
    if ok.sum() >= 2 and np.ptp(is_sel[ok]) > 0:
        slope, intercept = (float(v) for v in np.polyfit(is_sel[ok], oos_sel[ok], 1))
    else:
        slope = intercept = float("nan")
    root = math.sqrt(periods_per_year)
    return PBOResult(
        pbo=pbo,
        n_partitions=n_partitions,
        n_combinations=len(combos),
        n_combinations_total=total,
        n_strategies=n,
        n_observations=t,
        logits=logits.tolist(),
        is_sharpe_selected=(is_sel * root).tolist(),
        oos_sharpe_selected=(oos_sel * root).tolist(),
        oos_rank_selected=sel_rank.tolist(),
        selected_index=best.tolist(),
        degradation_slope=slope,
        degradation_intercept=intercept * root if math.isfinite(intercept) else intercept,
        prob_oos_loss=float(np.mean(oos_sel[ok] < 0)) if ok.any() else float("nan"),
        median_logit=float(np.median(logits)),
        periods_per_year=periods_per_year,
    )
