"""Parameter-surface robustness.

Given one row per parameter combination with a performance metric (e.g. Sharpe),
the surface is analysed on the grid implied by each parameter's sorted unique
values.

* **Neighbours** of a point are the grid points at Chebyshev distance 1 in index
  space (all parameters may move by one step).
* **Plateau ratio** = median(neighbour metric) / best metric (defined when the best
  metric is positive). Near 1: the optimum sits on a plateau. Near 0 or negative:
  the optimum is an isolated spike.
* **Spike z-score** = (best − mean of neighbours) / std of the whole surface.
* Classification: ``robust`` if plateau ratio ≥ ``robust_ratio``; ``fragile`` if
  < ``fragile_ratio``; ``moderate`` in between; ``undetermined`` if the best metric
  is not positive or the optimum has no neighbours.

A fragile optimum is *evidence* of overfitting risk, not proof that the strategy is
invalid: some genuine effects are narrow.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quantproof.errors import QuantProofInputError
from quantproof.results import Category, Finding
from quantproof.severity import Confidence, Severity


@dataclass
class SurfaceAnalysis:
    """Result of :func:`analyze_parameter_surface`."""

    params: list[str]
    metric: str
    n_points: int
    best: dict[str, Any]
    best_value: float
    neighbour_median: float
    neighbour_mean: float
    n_neighbours: int
    plateau_ratio: float
    spike_z: float
    classification: str
    local_maxima: list[dict[str, Any]] = field(default_factory=list)
    share_above_half_best: float = float("nan")
    oos_metric: str | None = None
    is_oos_spearman: float | None = None
    oos_at_is_best: float | None = None
    oos_median: float | None = None
    oos_best_matches_is_best: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _grid_index(df: pd.DataFrame, params: list[str]) -> np.ndarray:
    coords = []
    for p in params:
        values = sorted(df[p].unique().tolist())
        lookup = {v: i for i, v in enumerate(values)}
        coords.append(df[p].map(lookup).to_numpy())
    return np.column_stack(coords).astype(int)


def neighbours(coords: np.ndarray, i: int) -> np.ndarray:
    """Indices of rows at Chebyshev distance exactly 1 from row ``i``."""
    dist = np.abs(coords - coords[i]).max(axis=1)
    return np.flatnonzero(dist == 1)


def analyze_parameter_surface(
    results: pd.DataFrame,
    params: list[str],
    metric: str = "sharpe",
    *,
    oos_metric: str | None = None,
    robust_ratio: float = 0.7,
    fragile_ratio: float = 0.4,
) -> SurfaceAnalysis:
    """Analyse a parameter surface (one row per combination)."""
    missing = [c for c in [*params, metric] if c not in results.columns]
    if missing:
        raise QuantProofInputError(f"Columns missing from results: {missing}")
    df = results.dropna(subset=[metric]).reset_index(drop=True)
    if len(df) < 2:
        raise QuantProofInputError("Need at least two finite metric values to analyse a surface.")
    coords = _grid_index(df, params)
    values = df[metric].to_numpy(dtype=float)
    best_i = int(np.argmax(values))
    best_value = float(values[best_i])
    nb = neighbours(coords, best_i)
    nb_vals = values[nb]
    nb_median = float(np.median(nb_vals)) if nb.size else float("nan")
    nb_mean = float(np.mean(nb_vals)) if nb.size else float("nan")
    sd = float(np.std(values))
    spike_z = (best_value - nb_mean) / sd if nb.size and sd > 0 else float("nan")
    if best_value > 0 and nb.size:
        ratio = nb_median / best_value
        if ratio >= robust_ratio:
            cls = "robust"
        elif ratio < fragile_ratio:
            cls = "fragile"
        else:
            cls = "moderate"
    else:
        ratio, cls = float("nan"), "undetermined"
    local_max = []
    for i in range(len(df)):
        nbi = neighbours(coords, i)
        if nbi.size and values[i] >= values[nbi].max():
            local_max.append({**{p: df.loc[i, p] for p in params}, metric: float(values[i])})
    local_max.sort(key=lambda d: -d[metric])
    out = SurfaceAnalysis(
        params=list(params),
        metric=metric,
        n_points=len(df),
        best={p: df.loc[best_i, p] for p in params},
        best_value=best_value,
        neighbour_median=nb_median,
        neighbour_mean=nb_mean,
        n_neighbours=int(nb.size),
        plateau_ratio=ratio,
        spike_z=spike_z,
        classification=cls,
        local_maxima=local_max[:10],
        share_above_half_best=float(np.mean(values >= 0.5 * best_value))
        if best_value > 0
        else float("nan"),
    )
    if oos_metric is not None and oos_metric in df.columns:
        oos = df[oos_metric].to_numpy(dtype=float)
        ok = np.isfinite(oos)
        if ok.sum() >= 3 and np.ptp(values[ok]) > 0 and np.ptp(oos[ok]) > 0:
            out.is_oos_spearman = float(spearmanr(values[ok], oos[ok]).statistic)
        out.oos_metric = oos_metric
        out.oos_at_is_best = float(oos[best_i])
        out.oos_median = float(np.nanmedian(oos))
        out.oos_best_matches_is_best = bool(int(np.nanargmax(oos)) == best_i)
    return out


def surface_matrix(results: pd.DataFrame, x: str, y: str, metric: str) -> pd.DataFrame:
    """2-D pivot (rows = ``y`` values, columns = ``x`` values); other params at their best."""
    df = results.copy()
    others = [c for c in df.columns if c not in {x, y, metric} and df[c].nunique() > 1 and c in df]
    if others:
        best = df.loc[df[metric].idxmax()]
        for c in others:
            df = df[df[c] == best[c]]
    return df.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")


def render_ascii_surface(
    results: pd.DataFrame, x: str, y: str, metric: str, digits: int = 2
) -> str:
    """Text heat-map of a 2-D slice; ``★`` marks the maximum."""
    mat = surface_matrix(results, x, y, metric)
    if mat.empty:
        return ""
    best = np.nanmax(mat.to_numpy())
    width = max(7, max(len(str(c)) for c in mat.columns) + 2)
    header = f"{y}\\{x}".ljust(10) + "".join(str(c).rjust(width) for c in mat.columns)
    lines = [header]
    for row_label, row in mat.iterrows():
        cells = []
        for v in row.to_numpy():
            if not np.isfinite(v):
                cells.append(".".rjust(width))
            else:
                txt = f"{v:.{digits}f}" + ("★" if v == best else "")
                cells.append(txt.rjust(width))
        lines.append(str(row_label).ljust(10) + "".join(cells))
    return "\n".join(lines)


def sensitivity_findings(analysis: SurfaceAnalysis) -> list[Finding]:
    """Findings QP-SENS-001 (optimum shape) and QP-SENS-002 (IS/OOS consistency)."""
    out: list[Finding] = []
    ev = {
        "best": analysis.best,
        "best_value": analysis.best_value,
        "plateau_ratio": analysis.plateau_ratio,
        "neighbour_median": analysis.neighbour_median,
        "n_neighbours": analysis.n_neighbours,
        "spike_z": analysis.spike_z,
    }
    cls = analysis.classification
    sev = {"robust": Severity.PASS, "moderate": Severity.INFO, "fragile": Severity.WARN}.get(
        cls, Severity.INFO
    )
    msg = {
        "robust": "Neighbouring parameter combinations perform similarly to the optimum (robust region).",
        "moderate": "Neighbouring parameter combinations perform noticeably worse than the optimum.",
        "fragile": "The optimum is an isolated spike: neighbouring combinations perform much worse (fragile optimum).",
        "undetermined": "Optimum shape could not be classified (non-positive best metric or no neighbours).",
    }[cls]
    if math.isfinite(analysis.plateau_ratio):
        msg += f" Plateau ratio {analysis.plateau_ratio:.2f} (neighbour median / best)."
    out.append(
        Finding(
            id="QP-SENS-001",
            category=Category.SENSITIVITY,
            severity=sev,
            title={"robust": "Robust parameter region", "fragile": "Fragile parameter optimum"}.get(
                cls, "Parameter optimum shape"
            ),
            message=msg,
            evidence=ev,
            why_it_matters=(
                "An optimum surrounded by poor neighbours is more likely to reflect noise fitted "
                "by the search than a persistent effect. Narrow optima are a warning, not a verdict."
            ),
            recommendation="Prefer parameters in the middle of a plateau; report the full surface.",
            confidence=Confidence.MEDIUM,
        )
    )
    if analysis.is_oos_spearman is not None:
        rho = analysis.is_oos_spearman
        bad = rho < 0
        out.append(
            Finding(
                id="QP-SENS-002",
                category=Category.SENSITIVITY,
                severity=Severity.WARN if bad else Severity.PASS,
                title="In-sample ranking does not persist out of sample"
                if bad
                else "In-sample ranking persists out of sample",
                message=(
                    f"Spearman correlation between in-sample and out-of-sample performance "
                    f"across the grid is {rho:.2f}; OOS value at the IS optimum is "
                    f"{analysis.oos_at_is_best:.2f} vs OOS median {analysis.oos_median:.2f}."
                ),
                evidence={
                    "spearman": rho,
                    "oos_at_is_best": analysis.oos_at_is_best,
                    "oos_median": analysis.oos_median,
                },
                why_it_matters=(
                    "If parameter rankings reverse out of sample, selecting the in-sample best "
                    "adds no information."
                ),
                recommendation="Use validation-based selection and report OOS results only.",
                confidence=Confidence.MEDIUM,
            )
        )
    return out
