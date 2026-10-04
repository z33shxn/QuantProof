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

Expected maximum: two methods
-----------------------------
``"approximation"`` is the closed form above (the paper's). ``"exact"`` integrates
``E[max_{i≤N} Z_i] = ∫ x · N φ(x) Φ(x)^{N−1} dx`` numerically for i.i.d. standard normal
``Z_i`` and multiplies by ``sqrt(V)``. Relative error of the approximation (verified in the
tests against ``1/√π`` for ``N = 2`` and ``3/(2√π)`` for ``N = 3``): −7.9 % at ``N = 2``,
+0.8 % at 3, +2.3 % at 10, +0.9 % at 100, +0.4 % at 1000. The audit uses the exact value
and reports which method was used.

What "trials" means
-------------------
``N`` is the number of strategy variants whose results could have been selected —
every parameter set, feature set, universe or rule change that was backtested,
including ones that were discarded. QuantProof cannot observe trials that were not
declared: entering ``trials=5000`` makes the hurdle *consistent with* 5000 independent
tries, but it is only as honest as the number entered.

``N`` should be the number of *effectively independent* trials. Correlated variants
(e.g. neighbouring parameter values) make the raw count conservative (too high a
hurdle). When trial returns are available, :func:`effective_number_of_trials` estimates
an effective count from the eigenvalues of their correlation matrix (Li & Ji, 2005);
the audit reports it next to the declared count but uses the declared count for the
DSR unless configured otherwise.

What DSR does **not** show: a high DSR says the selected Sharpe is unlikely under the
null of zero true Sharpe *given N trials and the moments assumed*; it does not prove
future profitability, does not correct for undeclared trials, and inherits the PSR
assumptions (stationary returns, asymptotic normality of the Sharpe estimator).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import integrate
from scipy.stats import norm

from quantproof.errors import QuantProofInputError
from quantproof.statistics.probabilistic_sharpe import psr_from_moments
from quantproof.statistics.sharpe import return_moments, sharpe_ratio

EULER_MASCHERONI = 0.5772156649015329


def expected_max_standard_normal(n: int) -> float:
    """Exact ``E[max of n i.i.d. N(0, 1)]`` by numerical integration (``0`` for ``n = 1``)."""
    if n < 1:
        raise QuantProofInputError("n must be >= 1.")
    if n == 1:
        return 0.0

    def integrand(x: float) -> float:
        # n φ(x) Φ(x)^(n-1), computed in log space for stability at large n.
        return x * math.exp(math.log(n) + norm.logpdf(x) + (n - 1) * norm.logcdf(x))

    value, _ = integrate.quad(integrand, -12.0, 12.0, limit=200, epsabs=1e-12, epsrel=1e-10)
    return float(value)


def expected_max_sharpe(
    n_trials: int, sharpe_variance: float, *, method: str = "approximation"
) -> float:
    """Expected maximum of ``n_trials`` zero-mean Sharpe estimates with variance ``V``.

    ``method`` is ``"approximation"`` (Bailey & López de Prado's closed form) or
    ``"exact"`` (numerical integration). Returns 0 for a single trial (no selection).
    """
    if n_trials < 1:
        raise QuantProofInputError("n_trials must be >= 1.")
    if sharpe_variance < 0 or not math.isfinite(sharpe_variance):
        raise QuantProofInputError("sharpe_variance must be a finite non-negative number.")
    if method not in ("approximation", "exact"):
        raise QuantProofInputError("method must be 'approximation' or 'exact'.")
    if n_trials == 1:
        return 0.0
    if method == "exact":
        return math.sqrt(sharpe_variance) * expected_max_standard_normal(n_trials)
    g = EULER_MASCHERONI
    return math.sqrt(sharpe_variance) * (
        (1 - g) * float(norm.ppf(1 - 1.0 / n_trials))
        + g * float(norm.ppf(1 - 1.0 / (n_trials * math.e)))
    )


def effective_number_of_trials(trial_returns: Any) -> float:
    """Effective number of independent trials from return correlations (Li & Ji, 2005).

    ``M_eff = Σ_i [ 1(|λ_i| ≥ 1) + (|λ_i| − floor(|λ_i|)) ]`` over the eigenvalues ``λ_i`` of
    the trials' correlation matrix. Perfectly correlated trials give 1; uncorrelated
    trials give ``N``. Columns with zero variance are dropped. This is a heuristic for
    the *dimension* of the search, not an exact multiple-testing correction: fractional
    eigenvalue parts are credited, so e.g. two independent clusters of identical trials
    score between 2 and 4 rather than exactly 2.
    """
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim != 2 or m.shape[1] < 1:
        raise QuantProofInputError("trial_returns must be a 2-D (time x trials) array.")
    m = m[np.isfinite(m).all(axis=1)]
    m = m[:, np.nanstd(m, axis=0) > 0] if m.size else m
    if m.shape[1] <= 1 or m.shape[0] < 3:
        return float(m.shape[1])
    corr = np.corrcoef(m, rowvar=False)
    eig = np.abs(np.linalg.eigvalsh(corr))
    # Round away floating-point noise first: an eigenvalue of 24.9999999999 must count as 25
    # (contributing 1), not as 24 + 0.9999999999 (contributing ~2).
    eig = np.round(eig, 8)
    return float(np.sum((eig >= 1).astype(float) + (eig - np.floor(eig))))


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
    expected_max_method: str = "approximation"
    trials_source: str = "caller"
    effective_trials: float | None = None

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
    expected_max_method: str = "approximation",
    trials_source: str = "caller",
    effective_trials: float | None = None,
) -> DSRResult:
    """Deflated Sharpe Ratio of the *selected* strategy's returns.

    Parameters
    ----------
    returns: per-period returns of the selected strategy.
    n_trials: number of (effectively independent) strategy variants tried.
    trial_sharpes: optional per-period Sharpe ratios of all trials (used to estimate ``V``).
    sharpe_variance: optional explicit per-period cross-trial variance ``V``.
    expected_max_method: ``"approximation"`` (paper) or ``"exact"`` (numerical integration).
    trials_source / effective_trials: provenance of ``n_trials`` and an optional effective
        count, recorded in the result for reporting (they do not change the computation).
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
    sr0 = (
        float("nan")
        if not math.isfinite(var)
        else expected_max_sharpe(n_trials, var, method=expected_max_method)
    )
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
        expected_max_method=expected_max_method,
        trials_source=trials_source,
        effective_trials=effective_trials,
    )
