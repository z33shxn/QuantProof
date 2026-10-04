"""Strategy contract and loader.

A strategy is a Python file (or callable) exposing::

    def generate_signals(data: pd.DataFrame, **params) -> pd.Series:
        '''Target weight for each bar, decided using data up to and including that bar.'''

Optional module-level metadata (plain literals, so the static analyzer can read them):

``PARAMETERS``  default keyword arguments for ``generate_signals``.
``PARAM_GRID``  ``{name: [values, ...]}`` — the variants that were (or would be) searched.
                QuantProof evaluates the grid to measure selection bias (DSR, PBO,
                Reality Check, parameter-surface stability).
``EXECUTION``   the strategy's *own* execution assumptions, e.g.
                ``{"signal_lag": 1, "fill": "close", "commission_bps": 1.0,
                "spread_bps": 2.0, "slippage_bps": 2.0}``.
``NAME``        a human-readable identifier.

Security boundary
-----------------
Loading a strategy file **executes it** with the privileges of the current Python
process. QuantProof does not sandbox strategy code. Only audit code you trust, or
run audits inside an isolated environment (container, VM, CI runner without
secrets).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofStrategyError

ENTRY_POINT = "generate_signals"
_EXECUTION_KEYS = {
    "signal_lag",
    "fill",
    "commission_bps",
    "spread_bps",
    "slippage_bps",
    "impact_coefficient",
    "capital",
}


@dataclass
class StrategySpec:
    """A loaded strategy and its declared metadata."""

    name: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    execution: dict[str, Any] | None = None
    source_path: Path | None = None

    def __call__(self, data: pd.DataFrame, **overrides: Any) -> pd.Series:
        params = {**self.parameters, **overrides}
        return normalize_signals(self.func(data.copy(), **params), data.index, self.name)

    def grid(self, max_trials: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
        """All parameter combinations (deterministically sub-sampled above ``max_trials``)."""
        if not self.param_grid:
            return []
        keys = sorted(self.param_grid)
        combos = [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(self.param_grid[k] for k in keys))
        ]
        if max_trials is not None and len(combos) > max_trials:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(combos), size=max_trials, replace=False))
            combos = [combos[i] for i in idx]
        return combos

    @property
    def grid_size(self) -> int:
        size = 1
        for v in self.param_grid.values():
            size *= len(v)
        return size if self.param_grid else 0


def normalize_signals(raw: Any, index: pd.Index, name: str = "strategy") -> pd.Series:
    """Coerce a strategy output into a float Series aligned to ``index``."""
    if isinstance(raw, pd.DataFrame):
        if raw.shape[1] != 1:
            raise QuantProofStrategyError(
                f"{name}: generate_signals returned a DataFrame with {raw.shape[1]} columns. "
                "Single-asset audits expect one target-weight series."
            )
        raw = raw.iloc[:, 0]
    if isinstance(raw, pd.Series):
        s = raw.astype(float)
        if not s.index.equals(index):
            if not s.index.isin(index).all():
                raise QuantProofStrategyError(
                    f"{name}: signal index contains timestamps that are not in the data index."
                )
            s = s.reindex(index)
        return s
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != len(index):
        raise QuantProofStrategyError(
            f"{name}: generate_signals must return one value per row ({len(index)}), got shape "
            f"{arr.shape}."
        )
    return pd.Series(arr, index=index, name="signal")


def _validate_execution(execution: Any, origin: str) -> dict[str, Any] | None:
    if execution is None:
        return None
    if not isinstance(execution, dict):
        raise QuantProofStrategyError(f"{origin}: EXECUTION must be a dict.")
    unknown = set(execution) - _EXECUTION_KEYS
    if unknown:
        raise QuantProofStrategyError(
            f"{origin}: unknown EXECUTION keys {sorted(unknown)}; allowed: {sorted(_EXECUTION_KEYS)}."
        )
    return dict(execution)


def _from_module(module: ModuleType, origin: str, path: Path | None) -> StrategySpec:
    func = getattr(module, ENTRY_POINT, None)
    if func is None or not callable(func):
        raise QuantProofStrategyError(
            f"{origin} does not define a callable `{ENTRY_POINT}(data, **params)`. See the "
            "strategy contract in quantproof.strategy."
        )
    params = getattr(module, "PARAMETERS", {}) or {}
    grid = getattr(module, "PARAM_GRID", {}) or {}
    if not isinstance(params, dict) or not isinstance(grid, dict):
        raise QuantProofStrategyError(f"{origin}: PARAMETERS and PARAM_GRID must be dicts.")
    for k, v in grid.items():
        if not isinstance(v, (list, tuple)) or not v:
            raise QuantProofStrategyError(f"{origin}: PARAM_GRID[{k!r}] must be a non-empty list.")
    return StrategySpec(
        name=str(getattr(module, "NAME", path.stem if path else origin)),
        func=func,
        parameters=dict(params),
        param_grid={k: list(v) for k, v in grid.items()},
        execution=_validate_execution(getattr(module, "EXECUTION", None), origin),
        source_path=path,
    )


def load_strategy(
    source: str | Path | Callable[..., Any] | ModuleType | StrategySpec,
) -> StrategySpec:
    """Load a strategy from a file path, module, callable, or existing spec.

    Loading a file executes it (see the module docstring on the security boundary).
    """
    if isinstance(source, StrategySpec):
        return source
    if isinstance(source, ModuleType):
        file = getattr(source, "__file__", None)
        path = Path(file) if file else None
        return _from_module(source, source.__name__, path)
    if callable(source) and not isinstance(source, (str, Path)):
        name = getattr(source, "__name__", "strategy")
        path = None
        try:
            path = Path(inspect.getfile(source))
        except (TypeError, OSError):
            path = None
        return StrategySpec(name=name, func=source, source_path=path)
    path = Path(source)
    if not path.exists():
        raise QuantProofStrategyError(f"Strategy file not found: {path}")
    if path.suffix != ".py":
        raise QuantProofStrategyError(f"Strategy must be a .py file, got {path.name}.")
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    module_name = f"quantproof_strategy_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise QuantProofStrategyError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise QuantProofStrategyError(
            f"Importing {path} failed: {type(exc).__name__}: {exc}"
        ) from exc
    return _from_module(module, str(path), path)
