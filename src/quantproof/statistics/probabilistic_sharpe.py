"""Probabilistic Sharpe Ratio and Minimum Track Record Length.

Bailey, D. H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
Journal of Risk 15(2).

Formulation implemented
-----------------------
With per-period estimated Sharpe ``SR``, benchmark ``SR*`` (same frequency),
``T`` observations, skewness ``g3`` and non-excess kurtosis ``g4``::

    PSR(SR*) = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR**2) )

PSR is the probability that the true Sharpe exceeds ``SR*`` *under the asymptotic
normal approximation of the Sharpe estimator for i.i.d. (but non-normal) returns*.
Serial correlation is not modelled; positively autocorrelated returns make PSR
over-confident.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from scipy.stats import norm

from quantproof.errors import QuantProofInputError
from quantproof.statistics.sharpe import return_moments, sharpe_ratio


@dataclass(frozen=True)
class PSRResult:
    """PSR value and every input that determined it."""

    psr: float
    sharpe_per_period: float
    benchmark_sharpe_per_period: float
    n_observations: int
    skewness: float
    kurtosis: float
    z_score: float
    periods_per_year: float

    @property
    def sharpe_annualized(self) -> float:
        return self.sharpe_per_period * math.sqrt(self.periods_per_year)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sharpe_annualized"] = self.sharpe_annualized
        d["benchmark_sharpe_annualized"] = self.benchmark_sharpe_per_period * math.sqrt(
            self.periods_per_year
        )
        return d


def psr_from_moments(
    sharpe: float,
    n: int,
    skewness: float,
    kurtosis: float,
    benchmark_sharpe: float = 0.0,
) -> tuple[float, float]:
    """``(PSR, z)`` from per-period inputs. Returns ``(nan, nan)`` if undefined."""
    if n < 2 or not all(math.isfinite(v) for v in (sharpe, skewness, kurtosis, benchmark_sharpe)):
        return float("nan"), float("nan")
    denom_sq = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if denom_sq <= 0:
        return float("nan"), float("nan")
    z = (sharpe - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z)), float(z)


def probabilistic_sharpe_ratio(
    returns: Any,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> PSRResult:
    """Probabilistic Sharpe Ratio of a return series.

    ``benchmark_sharpe`` is *annualized* and converted to per-period with
    ``/ sqrt(periods_per_year)``.
    """
    mom = return_moments(returns)
    sr = sharpe_ratio(
        returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        annualize=False,
    )
    bench = benchmark_sharpe / math.sqrt(periods_per_year)
    psr, z = psr_from_moments(sr, mom.n, mom.skewness, mom.kurtosis, bench)
    return PSRResult(psr, sr, bench, mom.n, mom.skewness, mom.kurtosis, z, periods_per_year)


def minimum_track_record_length(
    sharpe: float,
    skewness: float,
    kurtosis: float,
    *,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed for PSR(benchmark) to reach ``confidence`` (per-period inputs).

    ``MinTRL = 1 + (1 - g3*SR + (g4-1)/4*SR**2) * (z_conf / (SR - SR*))**2``

    Returns ``inf`` when ``SR <= SR*`` (no track record length is sufficient).
    """
    if not 0 < confidence < 1:
        raise QuantProofInputError("confidence must be in (0, 1).")
    if not all(math.isfinite(v) for v in (sharpe, skewness, kurtosis)):
        return float("nan")
    if sharpe <= benchmark_sharpe:
        return float("inf")
    var = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    z = float(norm.ppf(confidence))
    return 1.0 + var * (z / (sharpe - benchmark_sharpe)) ** 2
