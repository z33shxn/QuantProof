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

from quantproof.analyzers.causal.perturbation import as_float_frame, perturb_after
from quantproof.config import CausalityConfig
from quantproof.errors import QuantProofInputError
from quantproof.results import Category, Finding, Location, Usage
from quantproof.severity import Confidence, Severity


@dataclass
class PerturbationTrial:
    """Outcome of one (decision time, scheme) perturbation.

    ``decision_time`` is the cutoff: every row after it was perturbed. For failed trials,
    ``changed_at`` is the first decision at or before the cutoff that changed, with its
    ``original`` and ``perturbed`` values (and the output ``column`` for multi-column or
    multi-asset outputs); ``first_modified`` is the first future observation that was
    perturbed.
    """

    timestamp: str
    scheme: str
    status: str  # "pass", "fail", "error", "skipped"
    n_changed: int = 0
    first_changed: str | None = None
    column: str | None = None
    original: float | None = None
    perturbed: float | None = None
    first_modified: str | None = None
    max_abs_diff: float = 0.0
    error: str | None = None

    @property
    def decision_time(self) -> str:
        return self.timestamp


@dataclass
class CausalityReport:
    """Aggregate evidence from the future-perturbation test."""

    trials: list[PerturbationTrial] = field(default_factory=list)
    deterministic: bool = True
    control_sensitive: bool | None = None
    n_observations: int = 0
    n_symbols: int = 1
    seed: int = 0
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
            "n_symbols": self.n_symbols,
            "seed": self.seed,
            "config": self.config,
            "trials": [t.__dict__ for t in self.trials],
        }


def _iso(value: Any) -> str:
    return (
        pd.Timestamp(value).isoformat()
        if isinstance(value, (pd.Timestamp, np.datetime64))
        else str(value)
    )


def outputs_frame(out: Any, data: pd.DataFrame) -> pd.DataFrame:
    """Normalize strategy output to a frame indexed by timestamp.

    Series/DataFrames indexed by ``(timestamp, symbol)`` are unstacked to one column per
    symbol; 1-D arrays must have one value per row of single-asset data.
    """
    if isinstance(out, pd.Series):
        if isinstance(out.index, pd.MultiIndex):
            return out.astype(float).unstack(level=1)
        return out.astype(float).to_frame("output")
    if isinstance(out, pd.DataFrame):
        if isinstance(out.index, pd.MultiIndex):
            return out.astype(float).unstack(level=1)
        return out.astype(float)
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if isinstance(data.index, pd.MultiIndex):
        raise QuantProofInputError("Panel strategies must return a Series/DataFrame, not an array.")
    return pd.DataFrame(arr, index=data.index[: arr.shape[0]])


def _compare(
    base: pd.DataFrame, new: pd.DataFrame, cutoff: Any, rtol: float, atol: float
) -> dict[str, Any] | None:
    """Evidence about decisions at or before ``cutoff`` that differ; ``None`` if none do."""
    b = base.loc[base.index <= cutoff]
    n = new.reindex(index=b.index, columns=b.columns)
    bv = b.to_numpy(dtype=float)
    nv = n.to_numpy(dtype=float)
    same = np.isclose(bv, nv, rtol=rtol, atol=atol, equal_nan=True)
    changed_rows = ~same.all(axis=1)
    if not changed_rows.any():
        return None
    r = int(np.argmax(changed_rows))
    c = int(np.argmax(~same[r]))
    with np.errstate(invalid="ignore"):
        diff = np.abs(bv - nv)
    diff = np.where(np.isnan(diff), np.inf, diff)
    return {
        "n_changed": int(changed_rows.sum()),
        "first_changed": _iso(b.index[r]),
        "column": str(b.columns[c]),
        "original": float(bv[r, c]),
        "perturbed": float(nv[r, c]),
        "max_abs_diff": float(np.max(diff[~same])),
    }


def run_causality_test(
    func: Callable[[pd.DataFrame], Any],
    data: pd.DataFrame,
    config: CausalityConfig | None = None,
    *,
    seed: int = 42,
) -> CausalityReport:
    """Run the future-perturbation test on ``func`` (which maps data → decisions).

    ``data`` may be single-asset (DatetimeIndex) or a panel ((timestamp, symbol) MultiIndex);
    decision times are distinct timestamps, and a perturbation changes every symbol's rows
    after the decision time. The perturbation seed is ``config.seed`` if set, else ``seed``.
    """
    cfg = config or CausalityConfig()
    seed = cfg.seed if cfg.seed is not None else seed
    frame = as_float_frame(data)
    times = (
        pd.DatetimeIndex(
            frame.index.get_level_values(0)
            if isinstance(frame.index, pd.MultiIndex)
            else frame.index
        )
        .unique()
        .sort_values()
    )
    n = len(times)
    n_symbols = (
        frame.index.get_level_values(1).nunique() if isinstance(frame.index, pd.MultiIndex) else 1
    )
    report = CausalityReport(
        n_observations=n, n_symbols=int(n_symbols), seed=seed, config=cfg.model_dump()
    )
    base = outputs_frame(func(frame.copy()), frame)
    again = outputs_frame(func(frame.copy()), frame)
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
    first = min(first, last)
    positions = np.unique(np.linspace(first, last, num=cfg.n_timestamps).round().astype(int))
    rng = np.random.default_rng(seed)
    for k in positions:
        cutoff = times[int(k)]
        first_modified = _iso(times[int(k) + 1])
        for scheme in cfg.schemes:
            ts = _iso(cutoff)
            if scheme == "permutation" and n - k - 1 < 2:
                report.trials.append(PerturbationTrial(ts, scheme, "skipped"))
                continue
            perturbed = perturb_after(frame, cutoff, scheme, magnitude=cfg.magnitude, seed=rng)
            try:
                out = outputs_frame(func(perturbed.copy()), frame)
            except Exception as exc:  # the strategy failed on perturbed data
                report.trials.append(
                    PerturbationTrial(ts, scheme, "error", error=f"{type(exc).__name__}: {exc}")
                )
                continue
            ev = _compare(base, out, cutoff, cfg.rtol, cfg.atol)
            if ev is None:
                report.trials.append(
                    PerturbationTrial(ts, scheme, "pass", first_modified=first_modified)
                )
            else:
                report.trials.append(
                    PerturbationTrial(ts, scheme, "fail", first_modified=first_modified, **ev)
                )
    # Control: perturb everything after the first timestamp; the output should change.
    try:
        control = perturb_after(frame, times[0], "extreme", seed=np.random.default_rng(seed + 1))
        out = outputs_frame(func(control.copy()), frame)
        report.control_sensitive = _compare(base, out, times[-1], cfg.rtol, cfg.atol) is not None
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
        earliest = min(fails, key=lambda t: t.timestamp)
        by_scheme = sorted({t.scheme for t in fails})
        col = f" [{earliest.column}]" if earliest.column not in (None, "output") else ""
        findings.append(
            Finding(
                id="QP-CAUSAL-001",
                category=Category.CAUSALITY,
                severity=Severity.FAIL,
                title="Future perturbation changed historical decisions",
                message=(
                    f"Historical decisions changed after future-data perturbation in "
                    f"{len(fails)} of {len(tested)} trials (schemes: {', '.join(by_scheme)}). "
                    f"Example: data from {earliest.first_modified} onward was perturbed "
                    f"({earliest.scheme}); the decision at {earliest.first_changed}{col} changed "
                    f"from {earliest.original:.6g} to {earliest.perturbed:.6g} "
                    f"({earliest.n_changed} decision(s) at or before {earliest.timestamp} changed). "
                    "FORBIDDEN IN LIVE DECISION."
                ),
                evidence={
                    "n_fail": len(fails),
                    "n_tested": len(tested),
                    "schemes_failed": by_scheme,
                    "examples": [t.__dict__ for t in fails[:5]],
                },
                location=loc,
                usage=Usage.LIVE_DECISION,
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
                    f"across {len({t.timestamp for t in tested})} decision times "
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
