"""Performance breakdown by regime."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantproof.audit.models import Category, Finding
from quantproof.audit.severity import Confidence, Severity
from quantproof.config import RegimeConfig
from quantproof.regimes.drawdown import drawdown_regimes, trend_regimes
from quantproof.regimes.volatility import volatility_regimes
from quantproof.statistics.sharpe import sharpe_ratio


def performance_by_regime(
    strategy_returns: pd.Series,
    regimes: pd.Series,
    *,
    periods_per_year: float = 252.0,
    min_observations: int = 40,
) -> list[dict[str, Any]]:
    """Per-regime statistics. Regimes with too few observations report Sharpe as None."""
    df = pd.DataFrame({"r": strategy_returns, "regime": regimes}).dropna()
    rows = []
    total_pnl = float(df["r"].sum()) if len(df) else 0.0
    order = list(dict.fromkeys(regimes.dropna().tolist()))
    for label in order:
        r = df.loc[df["regime"] == label, "r"]
        n = len(r)
        enough = n >= min_observations
        rows.append(
            {
                "regime": str(label),
                "n_observations": n,
                "share_of_time": n / len(df) if len(df) else float("nan"),
                "mean_return_annualized": float(r.mean() * periods_per_year) if n else float("nan"),
                "volatility_annualized": float(r.std(ddof=1) * math.sqrt(periods_per_year))
                if n > 1
                else float("nan"),
                "sharpe": sharpe_ratio(r, periods_per_year=periods_per_year) if enough else None,
                "hit_rate": float((r > 0).mean()) if n else float("nan"),
                "share_of_pnl": float(r.sum() / total_pnl) if total_pnl else float("nan"),
                "sufficient_data": enough,
            }
        )
    return rows


def regime_analysis(
    strategy_returns: pd.Series,
    reference_returns: pd.Series,
    config: RegimeConfig | None = None,
    *,
    periods_per_year: float = 252.0,
    benchmark_supplied: bool = False,
) -> tuple[dict[str, Any], list[Finding]]:
    """Volatility, drawdown and (with a benchmark) bull/bear breakdowns plus findings."""
    cfg = config or RegimeConfig()
    ref = reference_returns.reindex(strategy_returns.index)
    definitions = {
        "volatility": (
            f"Trailing {cfg.vol_window}-bar realized volatility of the "
            f"{'benchmark' if benchmark_supplied else 'traded asset'}, lagged one bar; cut at "
            f"full-sample quantiles {[round(q, 3) for q in cfg.vol_quantiles]} → {cfg.vol_labels}."
        ),
        "drawdown": (
            f"Drawdown of the {'benchmark' if benchmark_supplied else 'traded asset'} from its "
            f"running peak at the previous close; thresholds {cfg.drawdown_thresholds}."
        ),
    }
    labels = {
        "volatility": volatility_regimes(
            ref,
            window=cfg.vol_window,
            quantiles=cfg.vol_quantiles,
            labels=cfg.vol_labels,
            periods_per_year=periods_per_year,
        ),
        "drawdown": drawdown_regimes(ref, thresholds=cfg.drawdown_thresholds),
    }
    if benchmark_supplied:
        labels["trend"] = trend_regimes(ref, window=cfg.trend_window)
        definitions["trend"] = (
            f"bull if the benchmark's trailing {cfg.trend_window}-bar cumulative return at the "
            "previous close is positive, else bear."
        )
    section: dict[str, Any] = {"definitions": definitions, "tables": {}}
    findings: list[Finding] = []
    overall = sharpe_ratio(strategy_returns, periods_per_year=periods_per_year)
    section["overall_sharpe"] = overall
    fragile: list[str] = []
    for name, lab in labels.items():
        table = performance_by_regime(
            strategy_returns,
            lab,
            periods_per_year=periods_per_year,
            min_observations=cfg.min_observations,
        )
        section["tables"][name] = table
        for row in table:
            s = row["sharpe"]
            if s is not None and np.isfinite(s) and s < 0 and np.isfinite(overall) and overall > 0:
                fragile.append(
                    f"{name}={row['regime']} (Sharpe {s:.2f}, {row['n_observations']} obs)"
                )
    if fragile:
        findings.append(
            Finding(
                id="QP-REGIME-001",
                category=Category.REGIME,
                severity=Severity.WARN,
                title="Regime-dependent performance",
                message=(
                    f"Overall Sharpe {overall:.2f} but negative Sharpe in: {'; '.join(fragile)}."
                ),
                evidence={"overall_sharpe": overall, "negative_regimes": fragile},
                why_it_matters=(
                    "Performance concentrated in some regimes may disappear if the regime mix "
                    "changes; the full-sample Sharpe hides this dependence."
                ),
                recommendation=(
                    "Check whether the strategy's economic rationale explains the dependence and "
                    "size the strategy for the adverse regimes."
                ),
                confidence=Confidence.MEDIUM,
            )
        )
    else:
        findings.append(
            Finding(
                id="QP-REGIME-001",
                category=Category.REGIME,
                severity=Severity.PASS if np.isfinite(overall) and overall > 0 else Severity.INFO,
                title="No regime with negative performance",
                message=(
                    "No regime with sufficient data has a negative Sharpe ratio while the overall "
                    "Sharpe is positive."
                ),
                evidence={"overall_sharpe": overall},
            )
        )
    return section, findings
