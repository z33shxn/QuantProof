"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality." Journal of Portfolio
Management 40(5).

Formulation implemented
-----------------------
The hurdle is the expected maximum Sharpe ratio among ``N`` independent trials
with zero true Sharpe and cross-trial Sharpe variance ``V`` (the "False Strategy
Theorem")::

    SR0 = sqrt(V) * ( (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)) )
    DSR = PSR(SR0)

where ``gamma`` is the Euler-Mascheroni constant. All Sharpe quantities are
per-period.

Choice of ``V``:

1. If the Sharpe ratios of all trials are supplied, ``V`` is their sample variance
   (as in the paper).
2. Otherwise ``V = 1 / (T - 1)``: the approximate sampling variance of a Sharpe
   estimate when the true Sharpe is zero and returns are normal. This assumes the
   trials are independent draws of noise and is reported as such.

``N`` should be the number of *effectively independent* trials. Correlated variants
(e.g. neighbouring parameter values) make the raw count conservative (too high a
hurdle); the paper suggests clustering to estimate the effective count.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from quantproof.errors import QuantProofInputError
from quantproof.statistics.probabilistic_sharpe import psr_from_moments
from quantproof.statistics.sharpe import return_moments, sharpe_ratio

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum of ``n_trials`` zero-mean Sharpe estimates with variance ``V``.

    Returns 0 for a single trial (no selection).
    """
    if n_trials < 1:
        raise QuantProofInputError("n_trials must be >= 1.")
    if sharpe_variance < 0 or not math.isfinite(sharpe_variance):
        raise QuantProofInputError("sharpe_variance must be a finite non-negative number.")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    return math.sqrt(sharpe_variance) * (
        (1 - g) * float(norm.ppf(1 - 1.0 / n_trials))
        + g * float(norm.ppf(1 - 1.0 / (n_trials * math.e)))
    )


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe Ratio with all inputs exposed."""

    dsr: float
    sharpe_per_period: float
    expected_max_sharpe_per_period: float
    n_trials: int
    sharpe_variance: float
    variance_source: str
    n_observations: int
    skewness: float
    kurtosis: float
    z_score: float
    periods_per_year: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        root = math.sqrt(self.periods_per_year)
        d["sharpe_annualized"] = self.sharpe_per_period * root
        d["expected_max_sharpe_annualized"] = self.expected_max_sharpe_per_period * root
        return d


def deflated_sharpe_ratio(
    returns: Any,
    *,
    n_trials: int,
    trial_sharpes: Any = None,
    sharpe_variance: float | None = None,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> DSRResult:
    """Deflated Sharpe Ratio of the *selected* strategy's returns.

    Parameters
    ----------
    returns: per-period returns of the selected strategy.
    n_trials: number of (effectively independent) strategy variants tried.
    trial_sharpes: optional per-period Sharpe ratios of all trials (used to estimate ``V``).
    sharpe_variance: optional explicit per-period cross-trial variance ``V``.
    """
    mom = return_moments(returns)
    sr = sharpe_ratio(
        returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        annualize=False,
    )
    if sharpe_variance is not None:
        var, source = float(sharpe_variance), "explicit"
    elif trial_sharpes is not None:
        ts = np.asarray(trial_sharpes, dtype=float)
        ts = ts[np.isfinite(ts)]
        if ts.size < 2:
            raise QuantProofInputError("trial_sharpes needs at least two finite values.")
        var, source = float(np.var(ts, ddof=1)), "trial_sharpes"
    else:
        var = 1.0 / (mom.n - 1) if mom.n > 1 else float("nan")
        source = "null_sampling_variance_1/(T-1)"
    sr0 = float("nan") if not math.isfinite(var) else expected_max_sharpe(n_trials, var)
    dsr, z = psr_from_moments(sr, mom.n, mom.skewness, mom.kurtosis, sr0)
    return DSRResult(
        dsr=dsr,
        sharpe_per_period=sr,
        expected_max_sharpe_per_period=sr0,
        n_trials=n_trials,
        sharpe_variance=var,
        variance_source=source,
        n_observations=mom.n,
        skewness=mom.skewness,
        kurtosis=mom.kurtosis,
        z_score=z,
        periods_per_year=periods_per_year,
    )
