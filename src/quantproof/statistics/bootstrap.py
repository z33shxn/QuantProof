"""Bootstrap resampling for return series.

Methods
-------
* ``iid``:        Efron's bootstrap. Valid only if observations are independent;
                   it destroys autocorrelation and volatility clustering and usually
                   understates uncertainty for financial returns.
* ``circular``:   circular block bootstrap (Politis & Romano, 1992) with fixed block
                   length ``b``; preserves dependence within blocks.
* ``moving``:     moving block bootstrap (Künsch, 1989), non-circular.
* ``stationary``: stationary bootstrap (Politis & Romano, 1994): geometric block
                   lengths with mean ``b``; resamples are stationary.

Automatic block length follows Politis & White (2004) with the correction of
Patton, Politis & White (2009).

Confidence intervals are percentile intervals. They are approximate and can be
poor for small samples or heavy tails; QuantProof reports them as uncertainty
evidence, not exact coverage guarantees.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from quantproof._utils import clean_returns
from quantproof.errors import QuantProofInputError

Method = Literal["iid", "circular", "moving", "stationary"]


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def iid_indices(n: int, n_boot: int, seed: int | np.random.Generator | None = None) -> np.ndarray:
    """``(n_boot, n)`` i.i.d. resampling indices."""
    return _rng(seed).integers(0, n, size=(n_boot, n))


def circular_block_indices(
    n: int, n_boot: int, block_length: int, seed: int | np.random.Generator | None = None
) -> np.ndarray:
    """``(n_boot, n)`` circular-block indices with fixed block length."""
    b = max(1, min(int(block_length), n))
    n_blocks = math.ceil(n / b)
    starts = _rng(seed).integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def moving_block_indices(
    n: int, n_boot: int, block_length: int, seed: int | np.random.Generator | None = None
) -> np.ndarray:
    """``(n_boot, n)`` moving-block (non-circular) indices."""
    b = max(1, min(int(block_length), n))
    n_blocks = math.ceil(n / b)
    starts = _rng(seed).integers(0, n - b + 1, size=(n_boot, n_blocks))
    idx = starts[:, :, None] + np.arange(b)[None, None, :]
    return idx.reshape(n_boot, -1)[:, :n]


def stationary_indices(
    n: int, n_boot: int, mean_block_length: float, seed: int | np.random.Generator | None = None
) -> np.ndarray:
    """``(n_boot, n)`` stationary-bootstrap indices (geometric block lengths, mean ``L``)."""
    if mean_block_length < 1:
        raise QuantProofInputError("mean_block_length must be >= 1.")
    rng = _rng(seed)
    p = 1.0 / mean_block_length
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    new_block = rng.random((n_boot, n)) < p
    fresh = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def optimal_block_length(returns: Any) -> dict[str, float]:
    """Politis-White (2004) automatic block lengths, corrected by Patton et al. (2009).

    Returns ``{"stationary": b_sb, "circular": b_cb}``. Values are clipped to
    ``[1, min(3*sqrt(n), n/3)]``.
    """
    x = clean_returns(returns)
    n = x.size
    if n < 8:
        return {"stationary": 1.0, "circular": 1.0}
    x = x - x.mean()
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = math.ceil(math.sqrt(n)) + k_n
    m_max = min(m_max, n - 1)
    var0 = float(np.dot(x, x) / n)
    if var0 <= 0:
        return {"stationary": 1.0, "circular": 1.0}
    acov = np.array([float(np.dot(x[: n - k], x[k:]) / n) for k in range(m_max + 1)])
    rho = acov / var0
    crit = 2.0 * math.sqrt(math.log10(n) / n)
    m_hat = None
    for m in range(1, m_max - k_n + 2):
        window = np.abs(rho[m : m + k_n])
        if window.size == k_n and np.all(window < crit):
            m_hat = m
            break
    if m_hat is None:
        m_hat = m_max
    big_m = min(2 * m_hat, m_max)
    lags = np.arange(-big_m, big_m + 1)
    t = np.abs(lags) / big_m if big_m > 0 else np.zeros_like(lags, dtype=float)
    lam = np.where(t <= 0.5, 1.0, np.where(t <= 1.0, 2.0 * (1.0 - t), 0.0))
    r = acov[np.abs(lags)]
    g_hat = float(np.sum(lam * np.abs(lags) * r))
    g0 = float(np.sum(lam * r))
    upper = min(3.0 * math.sqrt(n), n / 3.0)
    if g0 <= 0 or g_hat == 0:
        return {"stationary": 1.0, "circular": 1.0}
    d_sb = 2.0 * g0**2
    d_cb = (4.0 / 3.0) * g0**2
    b_sb = (2.0 * g_hat**2 / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_cb = (2.0 * g_hat**2 / d_cb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return {
        "stationary": float(np.clip(b_sb, 1.0, upper)),
        "circular": float(np.clip(b_cb, 1.0, upper)),
    }


def bootstrap_indices(
    n: int,
    n_boot: int,
    *,
    method: Method = "stationary",
    block_length: float | None = None,
    seed: int | np.random.Generator | None = 42,
    data_for_block_length: Any = None,
) -> tuple[np.ndarray, float]:
    """Resampling indices and the block length actually used (1 for i.i.d.)."""
    rng = _rng(seed)
    if method == "iid":
        return iid_indices(n, n_boot, rng), 1.0
    if block_length is None:
        if data_for_block_length is None:
            block_length = max(1.0, n ** (1.0 / 3.0))
        else:
            key = "circular" if method in ("circular", "moving") else "stationary"
            block_length = optimal_block_length(data_for_block_length)[key]
    if method == "stationary":
        return stationary_indices(n, n_boot, block_length, rng), float(block_length)
    b = max(1, round(block_length))
    if method == "circular":
        return circular_block_indices(n, n_boot, b, rng), float(b)
    if method == "moving":
        return moving_block_indices(n, n_boot, b, rng), float(b)
    raise QuantProofInputError(f"Unknown bootstrap method {method!r}.")


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap distribution summary of a statistic."""

    estimate: float
    mean: float
    std_error: float
    ci_lower: float
    ci_upper: float
    confidence: float
    method: str
    block_length: float
    n_boot: int
    seed: int | None
    prob_below_zero: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_statistic(
    returns: Any,
    statistic: Callable[[np.ndarray], np.ndarray],
    *,
    method: Method = "stationary",
    n_boot: int = 1000,
    block_length: float | None = None,
    confidence: float = 0.95,
    seed: int | None = 42,
    chunk_size: int = 256,
) -> BootstrapResult:
    """Bootstrap a statistic of a single return series.

    ``statistic`` must be *vectorized over rows*: it receives a 2-D array of shape
    ``(k, n)`` (``k`` resamples) and returns ``k`` values. The point estimate is the
    statistic applied to the original series.
    """
    x = clean_returns(returns)
    n = x.size
    if n < 2:
        raise QuantProofInputError("Bootstrap requires at least two observations.")
    if not 0 < confidence < 1:
        raise QuantProofInputError("confidence must be in (0, 1).")
    estimate = float(np.asarray(statistic(x[None, :]))[0])
    rng = np.random.default_rng(seed)
    if block_length is None and method != "iid":
        key = "circular" if method in ("circular", "moving") else "stationary"
        block_length = optimal_block_length(x)[key]
    values = np.empty(n_boot)
    used_b = 1.0
    for start in range(0, n_boot, chunk_size):
        k = min(chunk_size, n_boot - start)
        idx, used_b = bootstrap_indices(n, k, method=method, block_length=block_length, seed=rng)
        values[start : start + k] = statistic(x[idx])
    finite = values[np.isfinite(values)]
    alpha = 1.0 - confidence
    if finite.size == 0:
        lo = hi = mean = sd = float("nan")
        p0 = float("nan")
    else:
        lo, hi = (float(v) for v in np.quantile(finite, [alpha / 2, 1 - alpha / 2]))
        mean, sd = float(finite.mean()), float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        p0 = float(np.mean(finite <= 0))
    return BootstrapResult(
        estimate, mean, sd, lo, hi, confidence, method, float(used_b), n_boot, seed, p0
    )


def sharpe_statistic(periods_per_year: float = 252.0) -> Callable[[np.ndarray], np.ndarray]:
    """Vectorized annualized Sharpe for :func:`bootstrap_statistic`."""
    root = math.sqrt(periods_per_year)

    def _stat(sample: np.ndarray) -> np.ndarray:
        mean = sample.mean(axis=1)
        sd = sample.std(axis=1, ddof=1)
        out = np.full(sample.shape[0], np.nan)
        ok = sd > 0
        out[ok] = mean[ok] / sd[ok] * root
        return out

    return _stat


def bootstrap_sharpe(
    returns: Any,
    *,
    periods_per_year: float = 252.0,
    method: Method = "stationary",
    n_boot: int = 1000,
    block_length: float | None = None,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> BootstrapResult:
    """Bootstrap confidence interval for the annualized Sharpe ratio."""
    return bootstrap_statistic(
        returns,
        sharpe_statistic(periods_per_year),
        method=method,
        n_boot=n_boot,
        block_length=block_length,
        confidence=confidence,
        seed=seed,
    )
