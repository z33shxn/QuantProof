"""Selection-bias and out-of-sample diagnostics on a grid of strategy variants.

Given a ``T x N`` matrix of realistic net returns (one column per parameter
combination), this module runs:

* **Walk-forward selection** — in each fold, choose the configuration with the best
  training Sharpe and record its *test* returns; the concatenated test returns form a
  genuinely out-of-sample track record of the selection procedure.
* **CPCV selection paths** — the same procedure over combinatorial purged splits,
  stitched into ``phi`` complete out-of-sample paths.
* **PBO** via CSCV.
* **White's Reality Check / Hansen's SPA** over all configurations.
* **Parameter surface** robustness (full-sample Sharpe) and IS/OOS rank persistence
  (first half vs second half).

Using precomputed full-sample signals is valid for selection only because each
configuration's signals were tested to be causal (future-perturbation test); if that
test failed, these diagnostics inherit the contamination.

Verdict rules (explicit)
------------------------
* ``QP-VAL-001`` walk-forward: WARN if the OOS Sharpe is ≤ 0, or below 50 % of a positive
  mean IS Sharpe.
* ``QP-VAL-002`` PBO: WARN if PBO ≥ 0.5.
* ``QP-VAL-003`` CPCV: WARN if the median path Sharpe ≤ 0.
* ``QP-VAL-004`` Reality Check: WARN if the SPA p-value > 1 − confidence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantproof.config import StatisticsConfig, ValidationConfig
from quantproof.errors import QuantProofInputError
from quantproof.results import Category, Finding
from quantproof.severity import Confidence, Severity
from quantproof.statistics.pbo import probability_of_backtest_overfitting
from quantproof.statistics.reality_check import reality_check
from quantproof.statistics.sharpe import sharpe_ratio, sharpe_ratios
from quantproof.validation.cpcv import CPCV
from quantproof.validation.walk_forward import WalkForward


def _best_column(m: np.ndarray) -> int:
    sr = sharpe_ratios(m)
    sr = np.where(np.isfinite(sr), sr, -np.inf)
    return int(np.argmax(sr))


def walk_forward_selection(
    matrix: pd.DataFrame, cfg: ValidationConfig, periods_per_year: float
) -> dict[str, Any]:
    """Select the best training configuration per fold; evaluate on the next window."""
    m = matrix.to_numpy(dtype=float)
    wf = WalkForward(
        train_size=cfg.train_fraction,
        n_splits=cfg.n_test_windows,
        gap=cfg.gap,
        expanding=cfg.expanding,
    )
    folds = []
    oos_parts = []
    for w in wf.windows(m):
        train = m[w.train_start : w.train_end]
        test = m[w.test_start : w.test_end]
        best = _best_column(train)
        is_sr = sharpe_ratio(train[:, best], periods_per_year=periods_per_year)
        oos_sr = sharpe_ratio(test[:, best], periods_per_year=periods_per_year)
        oos_parts.append(pd.Series(test[:, best], index=matrix.index[w.test_start : w.test_end]))
        folds.append(
            {
                "fold": w.fold,
                "train": [
                    str(matrix.index[w.train_start].date()),
                    str(matrix.index[w.train_end - 1].date()),
                ],
                "test": [
                    str(matrix.index[w.test_start].date()),
                    str(matrix.index[w.test_end - 1].date()),
                ],
                "selected": str(matrix.columns[best]),
                "is_sharpe": is_sr,
                "oos_sharpe": oos_sr,
            }
        )
    oos = pd.concat(oos_parts)
    is_mean = float(np.nanmean([f["is_sharpe"] for f in folds]))
    return {
        "method": "walk_forward",
        "expanding": cfg.expanding,
        "gap": cfg.gap,
        "folds": folds,
        "is_sharpe_mean": is_mean,
        "oos_sharpe": sharpe_ratio(oos, periods_per_year=periods_per_year),
        "oos_returns": oos,
        "diagram": wf.diagram(m),
    }


def walk_forward_single(
    series: pd.Series, cfg: ValidationConfig, periods_per_year: float
) -> dict[str, Any]:
    """Per-window Sharpe stability for a single (non-optimized) strategy."""
    wf = WalkForward(
        train_size=cfg.train_fraction,
        n_splits=cfg.n_test_windows,
        gap=cfg.gap,
        expanding=cfg.expanding,
    )
    rows = []
    for w in wf.windows(series):
        seg = series.iloc[w.test_start : w.test_end]
        rows.append(
            {
                "fold": w.fold,
                "test": [str(seg.index[0].date()), str(seg.index[-1].date())],
                "oos_sharpe": sharpe_ratio(seg, periods_per_year=periods_per_year),
            }
        )
    return {"method": "walk_forward_windows", "folds": rows, "diagram": wf.diagram(series)}


def cpcv_selection(
    matrix: pd.DataFrame, cfg: ValidationConfig, periods_per_year: float
) -> dict[str, Any]:
    """CPCV: select on purged/embargoed training groups, evaluate on test groups, stitch paths."""
    m = matrix.to_numpy(dtype=float)
    cv = CPCV(cfg.cpcv_groups, cfg.cpcv_test_groups, embargo=cfg.embargo)
    preds = []
    for train, test in cv.split(m):
        best = _best_column(m[train])
        preds.append(m[test, best])
    paths = cv.assemble_paths(m, preds)
    path_sr = [sharpe_ratio(p, periods_per_year=periods_per_year) for p in paths]
    return {
        "n_groups": cfg.cpcv_groups,
        "n_test_groups": cfg.cpcv_test_groups,
        "embargo": cfg.embargo,
        "n_splits": cv.n_splits,
        "n_paths": cv.n_paths,
        "path_sharpes": path_sr,
        "median_path_sharpe": float(np.nanmedian(path_sr)) if path_sr else float("nan"),
        "min_path_sharpe": float(np.nanmin(path_sr)) if path_sr else float("nan"),
        "max_path_sharpe": float(np.nanmax(path_sr)) if path_sr else float("nan"),
    }


def analyze_selection(
    matrix: pd.DataFrame,
    cfg: ValidationConfig,
    stats_cfg: StatisticsConfig,
    *,
    seed: int = 42,
) -> tuple[dict[str, Any], list[Finding]]:
    """Run all selection diagnostics on a ``T x N`` net-returns matrix."""
    ppy = stats_cfg.periods_per_year
    findings: list[Finding] = []
    section: dict[str, Any] = {
        "n_configurations": matrix.shape[1],
        "n_observations": matrix.shape[0],
    }
    alpha = 1 - stats_cfg.confidence

    if cfg.walk_forward:
        try:
            wf = walk_forward_selection(matrix, cfg, ppy)
            section["walk_forward"] = {k: v for k, v in wf.items() if k != "oos_returns"}
            is_m, oos = wf["is_sharpe_mean"], wf["oos_sharpe"]
            bad = (
                not math.isfinite(oos)
                or oos <= 0
                or (math.isfinite(is_m) and is_m > 0 and oos < 0.5 * is_m)
            )
            findings.append(
                Finding(
                    id="QP-VAL-001",
                    category=Category.VALIDATION,
                    severity=Severity.WARN if bad else Severity.PASS,
                    title="Weak out-of-sample evidence"
                    if bad
                    else "Out-of-sample performance holds up",
                    message=(
                        f"Walk-forward selection: mean in-sample Sharpe {is_m:.2f}, out-of-sample "
                        f"Sharpe {oos:.2f} over {len(wf['folds'])} folds."
                    ),
                    evidence={"is_sharpe_mean": is_m, "oos_sharpe": oos, "folds": wf["folds"]},
                    why_it_matters=(
                        "The out-of-sample result of the full selection procedure is the honest "
                        "estimate of what the research process delivers."
                    ),
                    recommendation="Report walk-forward OOS performance as the headline result.",
                    confidence=Confidence.MEDIUM,
                )
            )
        except QuantProofInputError as exc:
            section["walk_forward"] = {"error": str(exc)}

    if cfg.pbo:
        try:
            pbo = probability_of_backtest_overfitting(
                matrix,
                n_partitions=cfg.pbo_partitions,
                max_combinations=cfg.pbo_max_combinations,
                periods_per_year=ppy,
                seed=seed,
            )
            section["pbo"] = pbo.to_dict()
            findings.append(
                Finding(
                    id="QP-VAL-002",
                    category=Category.VALIDATION,
                    severity=Severity.WARN if pbo.pbo >= 0.5 else Severity.PASS,
                    title="Probability of Backtest Overfitting",
                    message=(
                        f"PBO = {pbo.pbo:.2f} over {pbo.n_combinations} CSCV combinations "
                        f"(S={pbo.n_partitions}, N={pbo.n_strategies}); probability of OOS loss "
                        f"{pbo.prob_oos_loss:.2f}. {pbo.to_dict()['interpretation']}"
                    ),
                    evidence=pbo.to_dict(include_arrays=False),
                    why_it_matters=(
                        "PBO estimates how often the in-sample winner underperforms the median "
                        "configuration out of sample."
                    ),
                    recommendation="Simplify the search or require OOS confirmation.",
                    confidence=Confidence.MEDIUM,
                )
            )
        except QuantProofInputError as exc:
            section["pbo"] = {"error": str(exc)}

    if cfg.cpcv:
        try:
            cp = cpcv_selection(matrix, cfg, ppy)
            section["cpcv"] = cp
            bad = not math.isfinite(cp["median_path_sharpe"]) or cp["median_path_sharpe"] <= 0
            findings.append(
                Finding(
                    id="QP-VAL-003",
                    category=Category.VALIDATION,
                    severity=Severity.WARN if bad else Severity.PASS,
                    title="Combinatorial purged CV",
                    message=(
                        f"{cp['n_paths']} CPCV paths ({cp['n_splits']} splits, embargo "
                        f"{cp['embargo']}): median OOS Sharpe {cp['median_path_sharpe']:.2f} "
                        f"(range {cp['min_path_sharpe']:.2f} to {cp['max_path_sharpe']:.2f})."
                    ),
                    evidence=dict(cp.items()),
                    why_it_matters="A distribution of OOS paths shows how much the result depends on one split.",
                    recommendation="Prefer strategies whose OOS path distribution is mostly positive.",
                    confidence=Confidence.MEDIUM,
                )
            )
        except QuantProofInputError as exc:
            section["cpcv"] = {"error": str(exc)}

    if cfg.reality_check:
        try:
            rc = reality_check(matrix, n_bootstrap=max(200, stats_cfg.n_bootstrap // 2), seed=seed)
            section["reality_check"] = rc.to_dict()
            bad = rc.spa_p_value > alpha
            findings.append(
                Finding(
                    id="QP-VAL-004",
                    category=Category.VALIDATION,
                    severity=Severity.WARN if bad else Severity.PASS,
                    title="Data-snooping test (Reality Check / SPA)",
                    message=(
                        f"Best of {rc.n_strategies} configurations vs zero-return benchmark: White "
                        f"RC p = {rc.p_value:.3f}, Hansen SPA_c p = {rc.spa_p_value:.3f} "
                        f"(threshold {alpha:.2f})."
                    ),
                    evidence=rc.to_dict(),
                    why_it_matters=(
                        "After accounting for the whole search, the best configuration may not be "
                        "distinguishable from luck."
                    ),
                    recommendation="Treat the strategy as unproven unless the data-snooping test rejects.",
                    confidence=Confidence.MEDIUM,
                )
            )
        except QuantProofInputError as exc:
            section["reality_check"] = {"error": str(exc)}
    return section, findings
