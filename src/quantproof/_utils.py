"""Small internal helpers shared across modules (not public API)."""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy/Pandas/Enum objects into plain JSON types.

    Non-finite floats become ``None`` so the output is strict JSON.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(row) for row in obj.to_dict(orient="records")]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump(mode="json"))
    return str(obj)


def as_float_array(values: Any, *, name: str = "returns") -> np.ndarray:
    """Convert array-like input into a 1-D float64 array without copying semantics."""
    from quantproof.errors import QuantProofInputError

    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise QuantProofInputError(
                f"Expected a single series for `{name}`, got a DataFrame with "
                f"{values.shape[1]} columns. Select one column first."
            )
        values = values.iloc[:, 0]
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise QuantProofInputError(
            f"Expected a one-dimensional `{name}` input, got shape {arr.shape}."
        )
    return arr


def clean_returns(values: Any, *, name: str = "returns", nan_policy: str = "omit") -> np.ndarray:
    """Return finite observations according to ``nan_policy``.

    ``nan_policy="omit"`` drops NaN; ``"raise"`` raises if any NaN is present.
    Infinite values always raise: they almost always indicate a data error
    (e.g. a division by a zero price) and silently dropping them would hide it.
    """
    from quantproof.errors import QuantProofInputError

    arr = as_float_array(values, name=name)
    if np.isinf(arr).any():
        n_inf = int(np.isinf(arr).sum())
        raise QuantProofInputError(
            f"`{name}` contains {n_inf} infinite value(s). Infinite returns usually come from "
            "a zero or missing price; fix the price series before computing statistics."
        )
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        if nan_policy == "raise":
            raise QuantProofInputError(
                f"`{name}` contains {int(nan_mask.sum())} NaN value(s). Pass nan_policy='omit' "
                "to drop them or fill them explicitly before calling."
            )
        if nan_policy != "omit":
            raise QuantProofInputError(
                f"Unknown nan_policy={nan_policy!r}; expected 'omit' or 'raise'."
            )
        arr = arr[~nan_mask]
    return arr


def format_number(value: float | None, digits: int = 3) -> str:
    """Human-friendly number formatting that tolerates None/NaN."""
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.{digits}f}"


def datetime_ns(values: Any) -> np.ndarray:
    """Int64 nanoseconds since the epoch (UTC for tz-aware values), regardless of resolution.

    ``DatetimeIndex.asi8`` is expressed in the index's own unit (pandas ≥ 2 may infer
    seconds or microseconds), so it must never be used directly as nanoseconds.
    """
    idx = pd.DatetimeIndex(values)
    return np.asarray(idx.as_unit("ns").asi8, dtype=np.int64)


def timedelta_ns(values: Any) -> np.ndarray:
    """Int64 nanoseconds of timedelta values regardless of resolution."""
    idx = pd.TimedeltaIndex(values)
    return np.asarray(idx.as_unit("ns").asi8, dtype=np.int64)
