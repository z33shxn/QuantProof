"""Reference adapter for pandas objects and CSV/Parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quantproof.adapters.base import ResearchArtifacts
from quantproof.data.loaders import load_frame, load_returns, prepare_prices
from quantproof.errors import QuantProofInputError

_FIELDS = (
    "returns",
    "trial_returns",
    "trial_params",
    "benchmark",
    "prices",
    "signals",
    "positions",
    "trades",
    "features",
    "target",
)


def _as_series(obj: Any, name: str) -> pd.Series:
    if isinstance(obj, (str, Path)):
        return load_returns(obj)
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise QuantProofInputError(f"{name} must be a single column; got {obj.shape[1]}.")
        return obj.iloc[:, 0]
    if isinstance(obj, pd.Series):
        return obj
    raise QuantProofInputError(f"{name}: expected a Series, single-column DataFrame, or file path.")


def _as_frame(obj: Any, name: str, *, time_indexed: bool = True) -> pd.DataFrame:
    if isinstance(obj, (str, Path)):
        frame = load_frame(obj)
        return prepare_prices(frame).prices if time_indexed else frame
    if isinstance(obj, pd.DataFrame):
        return obj
    raise QuantProofInputError(f"{name}: expected a DataFrame or file path.")


class PandasAdapter:
    """Build :class:`ResearchArtifacts` from pandas objects or CSV/Parquet paths.

    Example
    -------
    >>> artifacts = PandasAdapter().load(  # doctest: +SKIP
    ...     {"returns": returns_series, "benchmark": spy_returns}
    ... )
    """

    name = "pandas"

    def load(self, source: dict[str, Any] | None = None, **kwargs: Any) -> ResearchArtifacts:
        items = {**(source or {}), **kwargs}
        unknown = set(items) - set(_FIELDS) - {"metadata"}
        if unknown:
            raise QuantProofInputError(
                f"Unknown artifact fields {sorted(unknown)}; allowed: {_FIELDS}."
            )
        out = ResearchArtifacts(metadata=dict(items.get("metadata", {}), adapter=self.name))
        for key in ("returns", "benchmark", "signals", "target"):
            if items.get(key) is not None:
                setattr(out, key, _as_series(items[key], key).astype(float))
        if items.get("positions") is not None:
            pos = items["positions"]
            out.positions = (
                pos if isinstance(pos, (pd.Series, pd.DataFrame)) else _as_frame(pos, "positions")
            )
        for key in ("trial_returns", "prices", "features"):
            if items.get(key) is not None:
                setattr(out, key, _as_frame(items[key], key))
        for key in ("trial_params", "trades"):
            if items.get(key) is not None:
                setattr(out, key, _as_frame(items[key], key, time_indexed=False))
        out.validate()
        return out
