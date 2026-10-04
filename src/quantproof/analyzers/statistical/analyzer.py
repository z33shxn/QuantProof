"""Statistical diagnostics of the (realistic, net) strategy returns.

Verdict rules (explicit)
------------------------
* ``QP-STAT-001`` sample size: WARN if fewer than ``min_observations`` returns.
* ``QP-STAT-002`` PSR: WARN if PSR(benchmark) < ``confidence``.
* ``QP-STAT-003`` DSR: WARN if DSR < ``confidence``.
* ``QP-STAT-004`` implausible Sharpe: WARN if the annualized Sharpe of the audited returns
  *or of any simulated scenario* (naive same-bar/no-cost, declared) exceeds
  ``max_plausible_sharpe``.
* ``QP-STAT-005`` trials: INFO when the number of trials was not declared.
* ``QP-STAT-006`` non-normality: INFO if |skew| > 1 or excess kurtosis > 3.
* ``QP-STAT-007`` serial correlation: INFO if |rho_1| > 2/sqrt(n); reports Lo (2002)
  adjusted annual Sharpe using the first 10 autocorrelations.

Statistical findings never produce FAIL on their own: weak evidence is not proof
of invalid research.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantproof.config import StatisticsConfig
from quantproof.results import Category, Finding
from quantproof.severity import Confidence, Severity
from quantproof.statistics.bootstrap import bootstrap_sharpe
from quantproof.statistics.deflated_sharpe import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
)
from quantproof.statistics.probabilistic_sharpe import (
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)
from quantproof.statistics.sharpe import annualized_return, max_drawdown, sharpe_summary


def autocorrelations(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelations rho_1..rho_max_lag."""
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([float(np.dot(x[:-k], x[k:]) / denom) for k in range(1, max_lag + 1)])


def lo_adjusted_sharpe(sharpe_per_period: float, rho: np.ndarray, q: float) -> float:
    """Lo (2002) annualization ``SR * q / sqrt(q + 2 * sum_{k<q} (q - k) rho_k)``.

    Autocorrelations beyond ``len(rho)`` are assumed zero.
    """
    k = np.arange(1, len(rho) + 1)
    mask = k < q
    denom = q + 2.0 * float(np.sum((q - k[mask]) * rho[mask]))
    if denom <= 0 or not math.isfinite(sharpe_per_period):
        return float("nan")
    return sharpe_per_period * q / math.sqrt(denom)


def analyze_statistics(
    returns: pd.Series,
    cfg: StatisticsConfig,
    *,
    n_trials: int,
    trials_declared: bool,
    trial_sharpes: np.ndarray | None = None,
    seed: int = 42,
    scenario_sharpes: dict[str, float] | None = None,
    trials_source: str = "caller",
    trial_returns: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], list[Finding]]:
    """Compute Sharpe, PSR, DSR, bootstrap CI, MinTRL and related findings."""
    r = returns.dropna()
    ppy = cfg.periods_per_year
    findings: list[Finding] = []
    summary = sharpe_summary(r, periods_per_year=ppy, risk_free_rate=cfg.risk_free_rate)
    psr = probabilistic_sharpe_ratio(
        r,
        benchmark_sharpe=cfg.benchmark_sharpe,
        periods_per_year=ppy,
        risk_free_rate=cfg.risk_free_rate,
    )
    effective = (
        effective_number_of_trials(trial_returns.to_numpy(dtype=float))
        if trial_returns is not None and trial_returns.shape[1] >= 2
        else None
    )
    dsr = deflated_sharpe_ratio(
        r,
        n_trials=max(1, n_trials),
        expected_max_method="exact",
        trials_source=trials_source,
        effective_trials=effective,
        trial_sharpes=trial_sharpes
        if trial_sharpes is not None and len(trial_sharpes) >= 2
        else None,
        periods_per_year=ppy,
        risk_free_rate=cfg.risk_free_rate,
    )
    section: dict[str, Any] = {
        "sharpe": summary.to_dict(),
        "cagr": annualized_return(r, periods_per_year=ppy),
        "max_drawdown": max_drawdown(r),
        "psr": psr.to_dict(),
        "dsr": dsr.to_dict(),
        "trials_declared": trials_declared,
        "trials": {
            "declared": n_trials if trials_declared else None,
            "used_for_dsr": dsr.n_trials,
            "source": trials_source,
            "effective_estimate": effective,
            "effective_method": "Li & Ji (2005) eigenvalue count of trial-return correlations"
            if effective is not None
            else None,
            "meaning": (
                "Number of strategy variants whose results could have been selected, including "
                "discarded ones. QuantProof cannot observe undeclared trials."
            ),
        },
    }
    if len(r) >= 10:
        boot = bootstrap_sharpe(
            r,
            periods_per_year=ppy,
            method=cfg.bootstrap_method,
            n_boot=cfg.n_bootstrap,
            block_length=cfg.block_length,
            confidence=cfg.confidence,
            seed=seed,
        )
        section["bootstrap_sharpe"] = boot.to_dict()
    bench_pp = cfg.benchmark_sharpe / math.sqrt(ppy)
    mintrl = (
        minimum_track_record_length(
            summary.sharpe_per_period,
            summary.skewness,
            summary.kurtosis,
            benchmark_sharpe=bench_pp,
            confidence=cfg.confidence,
        )
        if math.isfinite(summary.sharpe_per_period)
        else float("nan")
    )
    section["min_track_record_length"] = mintrl
    rho = autocorrelations(r.to_numpy(dtype=float), 10) if len(r) > 20 else np.zeros(0)
    section["autocorrelation"] = rho.tolist()
    section["lo_adjusted_sharpe"] = (
        lo_adjusted_sharpe(summary.sharpe_per_period, rho, ppy) if rho.size else float("nan")
    )

    n = summary.n_observations
    findings.append(
        Finding(
            id="QP-STAT-001",
            category=Category.STATISTICS,
            severity=Severity.WARN if n < cfg.min_observations else Severity.PASS,
            title="Small sample" if n < cfg.min_observations else "Sample size adequate",
            message=f"{n} return observations (minimum {cfg.min_observations}).",
            evidence={"n": n, "minimum": cfg.min_observations},
            why_it_matters="Sharpe estimates from short samples have very wide confidence intervals.",
            recommendation="Extend the evaluation period or treat results as anecdotal.",
            confidence=Confidence.HIGH,
        )
    )
    psr_ok = math.isfinite(psr.psr) and psr.psr >= cfg.confidence
    findings.append(
        Finding(
            id="QP-STAT-002",
            category=Category.STATISTICS,
            severity=Severity.PASS if psr_ok else Severity.WARN,
            title="Probabilistic Sharpe Ratio",
            message=(
                f"PSR = {psr.psr:.3f}: estimated probability that the true Sharpe exceeds "
                f"{cfg.benchmark_sharpe:.2f} (annualized) given {n} observations, skew "
                f"{psr.skewness:.2f}, kurtosis {psr.kurtosis:.2f}. Threshold {cfg.confidence:.2f}."
                + (
                    f" Minimum track record ≈ {mintrl:.0f} observations."
                    if math.isfinite(mintrl)
                    else " The Sharpe ratio does not exceed the benchmark."
                )
            ),
            evidence={**psr.to_dict(), "min_track_record_length": mintrl},
            why_it_matters="A positive Sharpe can easily arise by chance in short or fat-tailed samples.",
            recommendation="Collect more out-of-sample data before trusting the strategy.",
            confidence=Confidence.MEDIUM,
        )
    )
    dsr_ok = math.isfinite(dsr.dsr) and dsr.dsr >= cfg.confidence
    findings.append(
        Finding(
            id="QP-STAT-003",
            category=Category.STATISTICS,
            severity=Severity.PASS if dsr_ok else Severity.WARN,
            title="Deflated Sharpe Ratio",
            message=(
                f"DSR = {dsr.dsr:.3f} with {dsr.n_trials} trial(s): the expected maximum Sharpe "
                f"of {dsr.n_trials} skill-less trials ({trials_source}) is "
                f"{dsr.expected_max_sharpe_per_period * math.sqrt(ppy):.2f} (annualized, "
                f"{dsr.expected_max_method}) vs observed "
                f"{dsr.sharpe_per_period * math.sqrt(ppy):.2f}. Variance source: "
                f"{dsr.variance_source}."
                + (
                    f" Estimated effective number of independent trials: {effective:.1f}."
                    if effective is not None
                    else ""
                )
            ),
            evidence=dsr.to_dict(),
            why_it_matters=(
                "Selecting the best of many backtests inflates the Sharpe ratio; DSR asks whether "
                "the selected result beats what luck alone would produce."
            ),
            recommendation="Report the number of trials honestly; reduce the search space.",
            confidence=Confidence.MEDIUM,
        )
    )
    candidates = [("audited returns", summary.sharpe_annualized)]
    candidates += [(f"{k} backtest", float(v)) for k, v in (scenario_sharpes or {}).items()]
    implausible = [
        (k, v) for k, v in candidates if math.isfinite(v) and v > cfg.max_plausible_sharpe
    ]
    if implausible:
        findings.append(
            Finding(
                id="QP-STAT-004",
                category=Category.STATISTICS,
                severity=Severity.WARN,
                title="Suspiciously strong backtest statistics",
                message=(
                    "; ".join(f"{k}: annualized Sharpe {v:.2f}" for k, v in implausible)
                    + f" — above the plausibility threshold {cfg.max_plausible_sharpe:.1f}."
                ),
                evidence={
                    "sharpe": summary.sharpe_annualized,
                    "scenario_sharpes": scenario_sharpes,
                    "threshold": cfg.max_plausible_sharpe,
                },
                why_it_matters=(
                    "Sharpe ratios this high on liquid markets are rare and usually come from "
                    "look-ahead, data errors, or unrealistic execution."
                ),
                recommendation="Treat the result as a bug until proven otherwise.",
                confidence=Confidence.MEDIUM,
            )
        )
    else:
        findings.append(
            Finding(
                id="QP-STAT-004",
                category=Category.STATISTICS,
                severity=Severity.PASS,
                title="Backtest statistics within plausible range",
                message=(
                    "Annualized Sharpe "
                    + ", ".join(f"{k} {v:.2f}" for k, v in candidates)
                    + f" ≤ {cfg.max_plausible_sharpe:.1f}."
                ),
                evidence={
                    "sharpe": summary.sharpe_annualized,
                    "scenario_sharpes": scenario_sharpes,
                },
            )
        )
    if not trials_declared:
        findings.append(
            Finding(
                id="QP-STAT-005",
                category=Category.STATISTICS,
                severity=Severity.WARN if cfg.require_declared_trials else Severity.INFO,
                title="Number of trials not declared",
                message=(
                    "No PARAM_GRID and no statistics.trials were given, so the DSR assumes a single "
                    "trial. If other variants were tried, the DSR overstates significance."
                ),
                recommendation="Set statistics.trials to the number of variants evaluated.",
            )
        )
    if math.isfinite(summary.skewness) and (abs(summary.skewness) > 1 or summary.kurtosis - 3 > 3):
        findings.append(
            Finding(
                id="QP-STAT-006",
                category=Category.STATISTICS,
                severity=Severity.INFO,
                title="Non-normal returns",
                message=(
                    f"Skewness {summary.skewness:.2f}, excess kurtosis {summary.kurtosis - 3:.2f}. "
                    "PSR/DSR account for this; i.i.d.-normal Sharpe confidence intervals do not."
                ),
                evidence={"skewness": summary.skewness, "kurtosis": summary.kurtosis},
            )
        )
    if rho.size and abs(rho[0]) > 2 / math.sqrt(max(n, 1)):
        findings.append(
            Finding(
                id="QP-STAT-007",
                category=Category.STATISTICS,
                severity=Severity.INFO,
                title="Serially correlated returns",
                message=(
                    f"Lag-1 autocorrelation {rho[0]:.3f} (|ρ| > 2/√n). sqrt(T) annualization is "
                    f"biased; Lo (2002)-adjusted Sharpe ≈ {section['lo_adjusted_sharpe']:.2f} vs "
                    f"{summary.sharpe_annualized:.2f}."
                ),
                evidence={"rho": rho.tolist(), "lo_adjusted_sharpe": section["lo_adjusted_sharpe"]},
            )
        )
    return section, findings
