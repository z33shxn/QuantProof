"""Data-snooping tests and multiple-testing corrections.

White's Reality Check
    White, H. (2000). "A Reality Check for Data Snooping." Econometrica 68(5).
    H0: no strategy outperforms the benchmark, ``max_k E[d_k] <= 0``, where ``d_k,t`` is
    strategy ``k``'s return minus the benchmark return at ``t``. Statistic
    ``V = max_k sqrt(n) * mean(d_k)``; its null distribution is approximated with the
    stationary bootstrap, recentring every strategy at its sample mean (the least
    favourable configuration — conservative when many strategies are poor).

Hansen's Superior Predictive Ability (consistent version, SPA_c)
    Hansen, P. R. (2005). "A Test for Superior Predictive Ability." JBES 23(4).
    Studentized statistic; strategies that are clearly worse than the benchmark are not
    recentred, which makes the test less sensitive to the inclusion of poor strategies.

p-values use ``(1 + #{T* >= T}) / (1 + B)``, a standard finite-sample correction.

``ω_k`` (the standard deviation used to studentize SPA) is estimated from the same
stationary bootstrap draws rather than a HAC kernel estimator; Hansen (2005) allows
either. If every ``ω_k`` is zero (constant differentials) the studentized statistic is
undefined and the SPA p-value is reported as NaN rather than a spuriously small number.

What these tests do **not** show: a small p-value says the best strategy's mean
differential is unlikely to be zero *given the strategies supplied*; it does not
correct for strategies you tried but did not include, and it says nothing about future
performance or economic significance after costs.

Assumptions: the return differentials are strictly stationary and weakly dependent
(so the stationary bootstrap is valid); the set of tested strategies is the *full*
set that was searched. Omitting tried strategies invalidates the correction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofInputError
from quantproof.statistics.bootstrap import optimal_block_length, stationary_indices


@dataclass(frozen=True)
class RealityCheckResult:
    """White's Reality Check and Hansen's SPA_c p-values."""

    statistic: float
    p_value: float
    spa_statistic: float
    spa_p_value: float
    best_index: int
    best_mean: float
    n_strategies: int
    n_observations: int
    n_bootstrap: int
    block_length: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_matrix(x: Any) -> np.ndarray:
    m = (
        x.to_numpy(dtype=float)
        if isinstance(x, (pd.DataFrame, pd.Series))
        else np.asarray(x, dtype=float)
    )
    if m.ndim == 1:
        m = m[:, None]
    return m


def reality_check(
    strategy_returns: Any,
    benchmark_returns: Any = None,
    *,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    seed: int = 42,
) -> RealityCheckResult:
    """White's Reality Check and Hansen's SPA_c on a ``T x K`` returns matrix.

    ``benchmark_returns`` defaults to zero (i.e. "does any strategy have positive mean
    return?"). Pass a benchmark series for relative tests.
    """
    m = _as_matrix(strategy_returns)
    n, k = m.shape
    if n < 10:
        raise QuantProofInputError("Reality Check needs at least 10 observations.")
    if benchmark_returns is not None:
        b = _as_matrix(benchmark_returns)
        if b.shape[0] != n:
            raise QuantProofInputError("benchmark_returns must have the same length as strategies.")
        d = m - b[:, :1]
    else:
        d = m
    if not np.isfinite(d).all():
        raise QuantProofInputError("Return differentials contain NaN/inf; align the data first.")
    dbar = d.mean(axis=0)
    root_n = math.sqrt(n)
    stat = float(np.max(root_n * dbar))
    if block_length is None:
        best_series = d[:, int(np.argmax(dbar))]
        block_length = optimal_block_length(best_series)["stationary"]
    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_bootstrap, k))
    chunk = 128
    for s in range(0, n_bootstrap, chunk):
        size = min(chunk, n_bootstrap - s)
        idx = stationary_indices(n, size, block_length, rng)
        boot_means[s : s + size] = d[idx].mean(axis=1)
    centered = root_n * (boot_means - dbar)
    v_star = centered.max(axis=1)
    p_rc = float((1 + np.sum(v_star >= stat)) / (1 + n_bootstrap))

    omega = centered.std(axis=0, ddof=1)
    # Constant differentials give omega ≈ 0 up to floating-point noise, which would make the
    # studentized statistic explode; treat them as having no variance.
    degenerate = d.std(axis=0) <= 1e-12 * np.maximum(np.abs(dbar), 1e-300)
    omega = np.where((omega > 0) & ~degenerate, omega, np.nan)
    t_k = root_n * dbar / omega
    spa_stat = float(max(np.nanmax(t_k), 0.0)) if np.isfinite(t_k).any() else float("nan")
    threshold = -math.sqrt(2.0 * math.log(math.log(n))) if n > math.e else -np.inf
    keep = t_k >= threshold
    # Hansen's SPA_c: strategies far below the benchmark keep their (negative) mean so they
    # cannot drive the bootstrap maximum; all others are recentred at zero.
    mu_c = np.where(keep, dbar, 0.0)
    z = root_n * (boot_means - mu_c) / omega
    if math.isfinite(spa_stat):
        t_star = np.maximum(np.nanmax(z, axis=1), 0.0)
        p_spa = float((1 + np.sum(t_star >= spa_stat)) / (1 + n_bootstrap))
    else:
        # Previously this produced p = 1/(1+B) (every comparison with NaN is False),
        # i.e. a "significant" result for degenerate input.
        p_spa = float("nan")
    best = int(np.argmax(dbar))
    return RealityCheckResult(
        statistic=stat,
        p_value=p_rc,
        spa_statistic=spa_stat,
        spa_p_value=p_spa,
        best_index=best,
        best_mean=float(dbar[best]),
        n_strategies=k,
        n_observations=n,
        n_bootstrap=n_bootstrap,
        block_length=float(block_length),
        seed=seed,
    )


def adjust_pvalues(
    p_values: Any, method: Literal["bonferroni", "holm", "bh"] = "holm"
) -> np.ndarray:
    """Multiple-testing adjusted p-values.

    * ``bonferroni`` — family-wise error rate, any dependence.
    * ``holm`` — step-down FWER control (Holm, 1979); uniformly more powerful than Bonferroni.
    * ``bh`` — Benjamini-Hochberg (1995) false discovery rate under independence or positive
      dependence.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise QuantProofInputError("p_values must be one-dimensional.")
    if np.any((p < 0) | (p > 1)):
        raise QuantProofInputError("p-values must lie in [0, 1].")
    m = p.size
    if m == 0:
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method == "bh":
        adj = ranked * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
    else:
        raise QuantProofInputError(f"Unknown method {method!r}.")
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out
