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
* ``QP-VAL-001`` out-of-sample evidence (no composite score; each criterion is reported):
  WARN if the concatenated OOS Sharpe is ≤ 0, fewer than half of the folds have a positive
  OOS Sharpe, the OOS Sharpe is below 50 % of a positive mean IS Sharpe, or the OOS sample
  has fewer than ``min_observations`` bars. Fold dispersion (std of fold Sharpes) is
  reported as evidence but is not a criterion on its own.
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


OOS_DEGRADATION_WARN = 0.5


def oos_evidence(
    folds: list[dict[str, Any]],
    oos_returns: pd.Series,
    *,
    is_sharpe_mean: float | None,
    periods_per_year: float,
    min_observations: int,
) -> dict[str, Any]:
    """Out-of-sample evidence and the criteria (if any) that make it weak.

    Returns observation count, concatenated OOS Sharpe, number and share of positive
    folds, fold Sharpe median and dispersion, IS→OOS degradation (OOS / mean IS), and
    ``weak_reasons`` — human-readable criteria that failed. No score is computed.
    """
    fold_sr = np.array([f["oos_sharpe"] for f in folds], dtype=float)
    finite = fold_sr[np.isfinite(fold_sr)]
    n_pos = int((finite > 0).sum())
    oos_sr = sharpe_ratio(oos_returns, periods_per_year=periods_per_year)
    n_obs = int(oos_returns.notna().sum())
    degradation = (
        oos_sr / is_sharpe_mean
        if is_sharpe_mean is not None
        and math.isfinite(is_sharpe_mean)
        and is_sharpe_mean > 0
        and math.isfinite(oos_sr)
        else float("nan")
    )
    reasons: list[str] = []
    if not math.isfinite(oos_sr) or oos_sr <= 0:
        reasons.append(f"out-of-sample Sharpe is {oos_sr:.2f} (≤ 0)")
    if len(fold_sr) and n_pos < len(fold_sr) / 2:
        reasons.append(f"only {n_pos} of {len(fold_sr)} folds have a positive OOS Sharpe")
    if math.isfinite(degradation) and degradation < OOS_DEGRADATION_WARN:
        reasons.append(
            f"OOS Sharpe is {degradation:.0%} of the mean in-sample Sharpe "
            f"(< {OOS_DEGRADATION_WARN:.0%})"
        )
    if n_obs < min_observations:
        reasons.append(f"only {n_obs} out-of-sample observations (< {min_observations})")
    return {
        "n_observations": n_obs,
        "oos_sharpe": oos_sr,
        "n_folds": len(fold_sr),
        "n_positive_folds": n_pos,
        "share_positive_folds": n_pos / len(fold_sr) if len(fold_sr) else float("nan"),
        "fold_sharpe_median": float(np.median(finite)) if finite.size else float("nan"),
        "fold_sharpe_std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        "is_sharpe_mean": is_sharpe_mean,
        "is_to_oos_ratio": degradation,
        "weak_reasons": reasons,
        "criteria": {
            "oos_sharpe_min": 0.0,
            "min_share_positive_folds": 0.5,
            "min_is_to_oos_ratio": OOS_DEGRADATION_WARN,
            "min_observations": min_observations,
        },
    }


def oos_finding(evidence: dict[str, Any], *, selection: bool) -> Finding:
    """QP-VAL-001 from :func:`oos_evidence`."""
    ev = evidence
    reasons = ev["weak_reasons"]
    what = (
        "walk-forward selection (best training configuration per fold)"
        if selection
        else ("sequential test windows of the single strategy")
    )
    summary = (
        f"{ev['n_positive_folds']} of {ev['n_folds']} folds positive; OOS Sharpe "
        f"{ev['oos_sharpe']:.2f} over {ev['n_observations']} observations; fold Sharpe median "
        f"{ev['fold_sharpe_median']:.2f}, dispersion (std) {ev['fold_sharpe_std']:.2f}"
        + (
            f"; OOS/IS ratio {ev['is_to_oos_ratio']:.2f}"
            if math.isfinite(ev["is_to_oos_ratio"])
            else ""
        )
        + "."
    )
    if reasons:
        message = f"Weak out-of-sample evidence from {what}: " + "; ".join(reasons) + ". " + summary
    else:
        message = f"Out-of-sample evidence from {what}: " + summary
    if not selection:
        message += " Without a parameter grid, QuantProof cannot measure selection bias."
    return Finding(
        id="QP-VAL-001",
        category=Category.VALIDATION,
        severity=Severity.WARN if reasons else Severity.PASS,
        title="Weak out-of-sample evidence" if reasons else "Out-of-sample evidence holds up",
        message=message,
        evidence=ev,
        recommendation=None
        if selection
        else "Declare PARAM_GRID (or statistics.trials) to enable PBO/DSR/CPCV diagnostics.",
        confidence=Confidence.MEDIUM,
    )


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
    parts = []
    for w in wf.windows(series):
        seg = series.iloc[w.test_start : w.test_end]
        parts.append(seg)
        rows.append(
            {
                "fold": w.fold,
                "test": [str(seg.index[0].date()), str(seg.index[-1].date())],
                "oos_sharpe": sharpe_ratio(seg, periods_per_year=periods_per_year),
            }
        )
    return {
        "method": "walk_forward_windows",
        "folds": rows,
        "oos_returns": pd.concat(parts) if parts else series.iloc[:0],
        "diagram": wf.diagram(series),
    }


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
            ev = oos_evidence(
                wf["folds"],
                wf["oos_returns"],
                is_sharpe_mean=wf["is_sharpe_mean"],
                periods_per_year=ppy,
                min_observations=stats_cfg.min_observations,
            )
            section["oos_evidence"] = ev
            findings.append(oos_finding(ev, selection=True))
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
