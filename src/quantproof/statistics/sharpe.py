"""Sharpe ratio and its sampling uncertainty.

Conventions (used consistently across QuantProof)
-------------------------------------------------
* Inputs are *simple* per-period returns (e.g. daily).
* The per-period Sharpe ratio is ``SR = mean(r - rf_p) / std(r - rf_p, ddof=1)``.
* ``rf_p`` converts an annual risk-free rate geometrically:
  ``rf_p = (1 + rf_annual) ** (1 / periods_per_year) - 1``.
* Annualization multiplies by ``sqrt(periods_per_year)``. This is exact only for
  i.i.d. returns; with autocorrelation the true annual Sharpe differs (Lo, 2002).
* Skewness and kurtosis are the (biased) sample moments ``m3/m2**1.5`` and
  ``m4/m2**2``; kurtosis is *non-excess* (3 for a normal distribution), matching
  Bailey & López de Prado (2012).
* Fewer than two finite observations, or zero variance, yield ``nan`` — a Sharpe
  ratio is undefined without dispersion; QuantProof never reports ``inf``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from quantproof._utils import clean_returns
from quantproof.errors import QuantProofInputError

_ZERO_VAR_RTOL = 1e-14


def periodic_risk_free(risk_free_rate: float, periods_per_year: float) -> float:
    """Convert an annual risk-free rate into a per-period rate (geometric)."""
    if periods_per_year <= 0:
        raise QuantProofInputError("periods_per_year must be positive.")
    if risk_free_rate <= -1:
        raise QuantProofInputError("risk_free_rate must be greater than -100%.")
    return float((1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0)


def _excess(
    returns: Any, risk_free_rate: float, periods_per_year: float, nan_policy: str
) -> np.ndarray:
    r = clean_returns(returns, nan_policy=nan_policy)
    if risk_free_rate:
        r = r - periodic_risk_free(risk_free_rate, periods_per_year)
    return r


def _std_is_zero(r: np.ndarray, sd: float) -> bool:
    scale = max(float(np.max(np.abs(r))), 1e-300) if r.size else 1.0
    return not math.isfinite(sd) or sd <= _ZERO_VAR_RTOL * scale


def sharpe_ratio(
    returns: Any,
    *,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    nan_policy: str = "omit",
) -> float:
    """Sharpe ratio of a return series.

    Parameters
    ----------
    returns: simple per-period returns.
    periods_per_year: e.g. 252 (daily), 52 (weekly), 12 (monthly).
    risk_free_rate: annual risk-free rate (decimal).
    annualize: multiply by ``sqrt(periods_per_year)`` when True.
    nan_policy: ``"omit"`` drops NaN, ``"raise"`` errors.

    Returns ``nan`` for fewer than two observations or zero variance.
    """
    r = _excess(returns, risk_free_rate, periods_per_year, nan_policy)
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if _std_is_zero(r, sd):
        return float("nan")
    sr = float(np.mean(r) / sd)
    return sr * math.sqrt(periods_per_year) if annualize else sr


def sharpe_ratios(matrix: Any, *, ddof: int = 1) -> np.ndarray:
    """Per-period Sharpe ratio of each column of a 2-D array (NaN for zero variance)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    mean = m.mean(axis=0)
    sd = m.std(axis=0, ddof=ddof)
    out = np.full(m.shape[1], np.nan)
    ok = sd > _ZERO_VAR_RTOL * np.maximum(np.abs(m).max(axis=0), 1e-300)
    out[ok] = mean[ok] / sd[ok]
    return out


@dataclass(frozen=True)
class ReturnMoments:
    """Sample moments used by PSR/DSR."""

    n: int
    mean: float
    std: float
    skewness: float
    kurtosis: float  # non-excess

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def return_moments(returns: Any, *, nan_policy: str = "omit") -> ReturnMoments:
    """Mean, sample std (ddof=1), skewness and non-excess kurtosis (biased moments)."""
    r = clean_returns(returns, nan_policy=nan_policy)
    n = int(r.size)
    if n < 2:
        return ReturnMoments(
            n, float(np.mean(r)) if n else float("nan"), float("nan"), float("nan"), float("nan")
        )
    mean = float(np.mean(r))
    dev = r - mean
    m2 = float(np.mean(dev**2))
    std = float(np.std(r, ddof=1))
    if _std_is_zero(r, std):
        return ReturnMoments(n, mean, 0.0, float("nan"), float("nan"))
    skew = float(np.mean(dev**3) / m2**1.5)
    kurt = float(np.mean(dev**4) / m2**2)
    return ReturnMoments(n, mean, std, skew, kurt)


def sharpe_standard_error(
    sharpe: float, n: int, skewness: float = 0.0, kurtosis: float = 3.0
) -> float:
    """Standard error of a *per-period* Sharpe estimate under non-normal i.i.d. returns.

    ``se = sqrt((1 - skew*SR + (kurt - 1)/4 * SR**2) / (n - 1))`` (Mertens, 2002;
    Bailey & López de Prado, 2012). With ``skew=0, kurt=3`` this reduces to Lo's (2002)
    i.i.d.-normal result ``sqrt((1 + SR**2/2)/(n-1))``.
    """
    if n < 2 or not math.isfinite(sharpe):
        return float("nan")
    var = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if not math.isfinite(var) or var <= 0:
        return float("nan")
    return math.sqrt(var / (n - 1))


@dataclass(frozen=True)
class SharpeSummary:
    """Sharpe ratio together with the inputs that determine it."""

    sharpe_annualized: float
    sharpe_per_period: float
    periods_per_year: float
    risk_free_rate: float
    n_observations: int
    mean_return: float
    volatility_annualized: float
    skewness: float
    kurtosis: float
    standard_error_annualized: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sharpe_summary(
    returns: Any, *, periods_per_year: float = 252.0, risk_free_rate: float = 0.0
) -> SharpeSummary:
    """Sharpe ratio with moments and non-normal standard error, for reports."""
    r = _excess(returns, risk_free_rate, periods_per_year, "omit")
    mom = return_moments(r)
    sr = sharpe_ratio(r, periods_per_year=periods_per_year, annualize=False)
    se = sharpe_standard_error(sr, mom.n, mom.skewness, mom.kurtosis)
    root = math.sqrt(periods_per_year)
    return SharpeSummary(
        sharpe_annualized=sr * root,
        sharpe_per_period=sr,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        n_observations=mom.n,
        mean_return=mom.mean,
        volatility_annualized=mom.std * root if math.isfinite(mom.std) else float("nan"),
        skewness=mom.skewness,
        kurtosis=mom.kurtosis,
        standard_error_annualized=se * root,
    )


def annualized_return(returns: Any, *, periods_per_year: float = 252.0) -> float:
    """Geometric annualized return (CAGR) of simple returns."""
    r = clean_returns(returns)
    if r.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / r.size) - 1.0


def max_drawdown(returns: Any) -> float:
    """Maximum peak-to-trough decline of the compounded equity curve (negative number)."""
    r = clean_returns(returns)
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float(np.min(equity / peak - 1.0))
