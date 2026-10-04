"""Runtime causality test: does perturbing the future change past decisions?

Procedure
---------
For a strategy ``f`` and data ``X``:

1. Run ``f(X)`` twice. If the outputs differ, the strategy is non-deterministic and the
   test is inconclusive (QP-CAUSAL-002).
2. Choose decision positions ``k`` evenly spaced between ``min_history_fraction × n`` and
   ``n - 2`` (deterministic).
3. For each ``k`` and each scheme, build ``X'`` equal to ``X`` on rows ``<= k`` and
   perturbed on rows ``> k``; run ``f(X')``.
4. A **violation** occurs if any output at a timestamp ``<= X.index[k]`` differs from the
   baseline beyond ``rtol``/``atol`` (NaN equals NaN).
5. A control run perturbs *all* rows; if the output never changes, the strategy ignores
   its input and the test carries no information (QP-CAUSAL-004).

Interpretation
--------------
A violation is direct evidence that information from after ``t`` influenced the
decision at ``t`` (look-ahead, centered windows, full-sample normalization, models fitted
on the whole sample, backward fills, ...). Passing is evidence *against* this class of
violation for the tested timestamps and schemes only; it cannot prove the absence of
look-ahead (e.g. leakage baked into the input data itself, such as survivorship bias or
restated fundamentals, is invisible to this test).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantproof.audit.causal.perturbation import as_float_frame, perturb_future
from quantproof.audit.models import Category, Finding, Location
from quantproof.audit.severity import Confidence, Severity
from quantproof.config import CausalityConfig


@dataclass
class PerturbationTrial:
    """Outcome of one (timestamp, scheme) perturbation."""

    timestamp: str
    position: int
    scheme: str
    status: str  # "pass", "fail", "error", "skipped"
    n_changed: int = 0
    first_changed: str | None = None
    max_abs_diff: float = 0.0
    error: str | None = None


@dataclass
class CausalityReport:
    """Aggregate evidence from the future-perturbation test."""

    trials: list[PerturbationTrial] = field(default_factory=list)
    deterministic: bool = True
    control_sensitive: bool | None = None
    n_observations: int = 0
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def n_fail(self) -> int:
        return sum(t.status == "fail" for t in self.trials)

    @property
    def n_pass(self) -> int:
        return sum(t.status == "pass" for t in self.trials)

    @property
    def n_error(self) -> int:
        return sum(t.status == "error" for t in self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trials": len(self.trials),
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_error": self.n_error,
            "deterministic": self.deterministic,
            "control_sensitive": self.control_sensitive,
            "n_observations": self.n_observations,
            "config": self.config,
            "trials": [t.__dict__ for t in self.trials],
        }


def _to_frame(out: Any, index: pd.Index) -> pd.DataFrame:
    if isinstance(out, pd.DataFrame):
        return out.astype(float)
    if isinstance(out, pd.Series):
        return out.astype(float).to_frame("output")
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return pd.DataFrame(arr, index=index[: arr.shape[0]])


def _compare(
    base: pd.DataFrame, new: pd.DataFrame, cutoff: Any, rtol: float, atol: float
) -> tuple[int, str | None, float]:
    b = base.loc[base.index <= cutoff]
    n = new.reindex(index=b.index, columns=b.columns)
    bv = b.to_numpy(dtype=float)
    nv = n.to_numpy(dtype=float)
    both_nan = np.isnan(bv) & np.isnan(nv)
    close = np.isclose(bv, nv, rtol=rtol, atol=atol) | both_nan
    changed_rows = ~close.all(axis=1)
    n_changed = int(changed_rows.sum())
    if n_changed == 0:
        return 0, None, 0.0
    first_label = b.index[int(np.argmax(changed_rows))]
    first = (
        pd.Timestamp(first_label).isoformat()
        if isinstance(b.index, pd.DatetimeIndex)
        else str(first_label)
    )
    with np.errstate(invalid="ignore"):
        diff = np.abs(bv - nv)
    diff = np.where(np.isnan(diff), np.inf, diff)
    max_diff = float(np.max(diff[changed_rows]))
    return n_changed, first, max_diff


def run_causality_test(
    func: Callable[[pd.DataFrame], Any],
    data: pd.DataFrame,
    config: CausalityConfig | None = None,
    *,
    seed: int = 42,
) -> CausalityReport:
    """Run the future-perturbation test on ``func`` (which maps data → decisions)."""
    cfg = config or CausalityConfig()
    frame = as_float_frame(data)
    n = len(frame)
    report = CausalityReport(n_observations=n, config=cfg.model_dump())
    base = _to_frame(func(frame.copy()), frame.index)
    again = _to_frame(func(frame.copy()), frame.index)
    if base.shape != again.shape or not np.allclose(
        base.to_numpy(dtype=float),
        again.to_numpy(dtype=float),
        rtol=cfg.rtol,
        atol=cfg.atol,
        equal_nan=True,
    ):
        report.deterministic = False
        return report
    if n < 4:
        return report
    first = max(1, int(np.floor(cfg.min_history_fraction * n)))
    last = n - 2
    if first > last:
        first = last
    positions = np.unique(np.linspace(first, last, num=cfg.n_timestamps).round().astype(int))
    rng = np.random.default_rng(seed)
    for k in positions:
        cutoff = frame.index[int(k)]
        for scheme in cfg.schemes:
            ts = (
                pd.Timestamp(cutoff).isoformat()
                if isinstance(frame.index, pd.DatetimeIndex)
                else str(cutoff)
            )
            if scheme == "permutation" and n - k - 1 < 2:
                report.trials.append(PerturbationTrial(ts, int(k), scheme, "skipped"))
                continue
            perturbed = perturb_future(frame, int(k), scheme, magnitude=cfg.magnitude, seed=rng)
            try:
                out = _to_frame(func(perturbed.copy()), frame.index)
            except Exception as exc:  # strategy failed on perturbed data
                report.trials.append(
                    PerturbationTrial(
                        ts, int(k), scheme, "error", error=f"{type(exc).__name__}: {exc}"
                    )
                )
                continue
            n_changed, first_changed, max_diff = _compare(base, out, cutoff, cfg.rtol, cfg.atol)
            report.trials.append(
                PerturbationTrial(
                    ts,
                    int(k),
                    scheme,
                    "fail" if n_changed else "pass",
                    n_changed,
                    first_changed,
                    max_diff,
                )
            )
    # Control: perturb everything after the first row; the output should change.
    try:
        control = perturb_future(frame, 0, "shock", seed=np.random.default_rng(seed + 1))
        out = _to_frame(func(control.copy()), frame.index)
        changed, _, _ = _compare(base, out, frame.index[-1], cfg.rtol, cfg.atol)
        report.control_sensitive = changed > 0
    except Exception:
        report.control_sensitive = None
    return report


def causality_findings(report: CausalityReport, *, source: str | None = None) -> list[Finding]:
    """Translate a :class:`CausalityReport` into findings."""
    loc = Location(file=source) if source else None
    findings: list[Finding] = []
    if not report.deterministic:
        findings.append(
            Finding(
                id="QP-CAUSAL-002",
                category=Category.CAUSALITY,
                severity=Severity.WARN,
                title="Strategy output is non-deterministic",
                message=(
                    "Two runs on identical data produced different outputs, so the future-"
                    "perturbation test is inconclusive."
                ),
                location=loc,
                why_it_matters="Unseeded randomness makes results irreproducible and untestable.",
                recommendation="Seed every random number generator used by the strategy.",
                confidence=Confidence.HIGH,
            )
        )
        return findings
    fails = [t for t in report.trials if t.status == "fail"]
    tested = [t for t in report.trials if t.status in {"pass", "fail"}]
    if fails:
        earliest = min(fails, key=lambda t: t.position)
        by_scheme = sorted({t.scheme for t in fails})
        findings.append(
            Finding(
                id="QP-CAUSAL-001",
                category=Category.CAUSALITY,
                severity=Severity.FAIL,
                title="Future perturbation changed historical decisions",
                message=(
                    f"FAIL: historical decision changed after future-data perturbation in "
                    f"{len(fails)} of {len(tested)} trials (schemes: {', '.join(by_scheme)}). "
                    f"Example: perturbing data after {earliest.timestamp} changed "
                    f"{earliest.n_changed} earlier decision(s), first at {earliest.first_changed}."
                ),
                evidence={
                    "n_fail": len(fails),
                    "n_tested": len(tested),
                    "schemes_failed": by_scheme,
                    "examples": [t.__dict__ for t in fails[:5]],
                },
                location=loc,
                why_it_matters=(
                    "A decision at time t that depends on data after t could not have been made "
                    "in real time; the backtest performance is contaminated by look-ahead."
                ),
                recommendation=(
                    "Inspect the static findings for negative shifts, centered windows, "
                    "full-sample normalization or models fitted on the full sample. Every "
                    "transformation must use data up to t only."
                ),
                confidence=Confidence.HIGH,
            )
        )
    elif tested:
        findings.append(
            Finding(
                id="QP-CAUSAL-001",
                category=Category.CAUSALITY,
                severity=Severity.PASS,
                title="Past decisions invariant to future perturbations",
                message=(
                    f"No decision at or before t changed in {len(tested)} perturbation trials "
                    f"across {len({t.position for t in tested})} timestamps "
                    f"({', '.join(sorted({t.scheme for t in tested}))}). This is evidence "
                    "against, not proof of absence of, look-ahead."
                ),
                evidence={"n_tested": len(tested)},
                location=loc,
            )
        )
    if report.n_error:
        errs = [t for t in report.trials if t.status == "error"]
        findings.append(
            Finding(
                id="QP-CAUSAL-003",
                category=Category.CAUSALITY,
                severity=Severity.INFO,
                title="Strategy raised errors on perturbed data",
                message=(
                    f"{len(errs)} perturbation trial(s) raised an exception and were not "
                    f"evaluated; first: {errs[0].error}"
                ),
                evidence={"errors": [t.__dict__ for t in errs[:5]]},
                location=loc,
            )
        )
    if report.control_sensitive is False:
        findings.append(
            Finding(
                id="QP-CAUSAL-004",
                category=Category.CAUSALITY,
                severity=Severity.INFO,
                title="Strategy output insensitive to input data",
                message=(
                    "Perturbing the entire sample did not change the output; the causality test "
                    "carries no information for this strategy."
                ),
                location=loc,
            )
        )
    return findings
