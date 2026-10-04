"""Runtime leakage diagnostics on features, targets, and signals.

These checks look at *values*, complementing the static analyzer (which looks at
code) and the perturbation test (which looks at behaviour).

* ``QP-LEAK-001`` — a feature is (almost) identical to the target: |corr| ≥ threshold.
* ``QP-LEAK-002`` — a feature at ``t`` is (almost) identical to a *future* asset return
  ``r[t+h]`` for some ``1 <= h <= max_lead``.
* ``QP-LEAK-003`` — signal direction predicts the next return with implausible accuracy
  (hit rate above ``max_hit_rate`` with a one-sided binomial p-value below 1e-6).

Thresholds are deliberately extreme: they target unambiguous leakage, not genuinely
predictive features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from quantproof.audit.models import Category, Finding
from quantproof.audit.severity import Confidence, Severity


def _corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 10:
        return float("nan")
    x, y = df.iloc[:, 0].to_numpy(dtype=float), df.iloc[:, 1].to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def feature_leakage_findings(
    features: pd.DataFrame,
    *,
    target: pd.Series | None = None,
    asset_returns: pd.Series | None = None,
    max_lead: int = 5,
    threshold: float = 0.98,
) -> list[Finding]:
    """Detect features that copy the target or future returns."""
    out: list[Finding] = []
    numeric = features.select_dtypes(include=[np.number])
    copies: list[dict[str, Any]] = []
    futures: list[dict[str, Any]] = []
    for col in numeric.columns:
        x = numeric[col]
        if target is not None:
            c = _corr(x, target.reindex(x.index))
            if np.isfinite(c) and abs(c) >= threshold:
                copies.append({"feature": str(col), "corr": c})
        if asset_returns is not None:
            r = asset_returns.reindex(x.index)
            for h in range(1, max_lead + 1):
                c = _corr(x, r.shift(-h))
                if np.isfinite(c) and abs(c) >= threshold:
                    futures.append({"feature": str(col), "lead": h, "corr": c})
                    break
    if target is not None:
        out.append(
            Finding(
                id="QP-LEAK-001",
                category=Category.LEAKAGE,
                severity=Severity.FAIL if copies else Severity.PASS,
                title="Feature replicates the target"
                if copies
                else "No feature replicates the target",
                message=(
                    "Features nearly identical to the target: "
                    + ", ".join(f"{d['feature']} (corr {d['corr']:.3f})" for d in copies)
                    if copies
                    else f"No feature has |corr| ≥ {threshold} with the target."
                ),
                evidence={"features": copies, "threshold": threshold},
                why_it_matters="A model given a copy of its target learns nothing transferable.",
                recommendation="Remove target-derived columns from the feature set.",
                confidence=Confidence.HIGH,
            )
        )
    if asset_returns is not None:
        out.append(
            Finding(
                id="QP-LEAK-002",
                category=Category.LEAKAGE,
                severity=Severity.FAIL if futures else Severity.PASS,
                title="Feature encodes future returns"
                if futures
                else "No feature encodes future returns",
                message=(
                    "Features nearly identical to future returns: "
                    + ", ".join(
                        f"{d['feature']} ≈ r[t+{d['lead']}] (corr {d['corr']:.3f})" for d in futures
                    )
                    if futures
                    else f"No feature has |corr| ≥ {threshold} with returns 1–{max_lead} bars ahead."
                ),
                evidence={"features": futures, "threshold": threshold, "max_lead": max_lead},
                why_it_matters="Such a feature is unavailable at decision time.",
                recommendation="Rebuild the feature from data at or before t.",
                confidence=Confidence.HIGH,
            )
        )
    return out


def signal_foresight_finding(
    signals: pd.Series,
    asset_returns: pd.Series,
    *,
    max_hit_rate: float = 0.75,
    min_trades: int = 50,
) -> tuple[dict[str, Any], Finding]:
    """Directional hit rate of ``sign(signal[t])`` against ``sign(r[t+1])``."""
    df = pd.DataFrame({"s": signals, "r_next": asset_returns.shift(-1)}).dropna()
    df = df[(df["s"] != 0) & (df["r_next"] != 0)]
    n = len(df)
    hits = int((np.sign(df["s"]) == np.sign(df["r_next"])).sum())
    rate = hits / n if n else float("nan")
    p = float(binomtest(hits, n, 0.5, alternative="greater").pvalue) if n else float("nan")
    evidence = {"n": n, "hits": hits, "hit_rate": rate, "p_value": p, "threshold": max_hit_rate}
    flagged = n >= min_trades and rate > max_hit_rate and p < 1e-6
    finding = Finding(
        id="QP-LEAK-003",
        category=Category.LEAKAGE,
        severity=Severity.WARN if flagged else Severity.PASS,
        title="Implausible directional accuracy" if flagged else "Directional accuracy plausible",
        message=(
            f"Signal direction matches the next bar's return {rate:.1%} of the time over {n} "
            f"bars (p = {p:.1e}). Accuracy this high on market data almost always indicates "
            "look-ahead."
            if flagged
            else f"Directional hit rate vs next-bar return: {rate:.1%} over {n} bars."
        ),
        evidence=evidence,
        why_it_matters=(
            "Even strong predictive signals rarely exceed ~55–60% directional accuracy at short "
            "horizons; near-perfect accuracy suggests the signal sees the outcome."
        ),
        recommendation="Check that every input to the signal is known before the bar it trades.",
        confidence=Confidence.MEDIUM,
    )
    return evidence, finding
