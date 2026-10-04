"""Deterministic content hashes for data, files, and configuration.

DataFrame hashing is defined explicitly (not via pickle or Parquet bytes, which
embed library versions): for every column, in order, the hash consumes the column
name, a normalized dtype tag, and the values —

* floats as little-endian float64 with all NaNs canonicalized,
* integers/booleans as little-endian int64,
* datetimes as int64 nanoseconds since epoch (UTC for tz-aware values),
* everything else as UTF-8 strings separated by ``\\x1f``.

The index is hashed the same way. The digest is therefore stable across pandas,
NumPy and PyArrow versions and across platforms.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantproof._utils import datetime_ns, to_jsonable

_CANONICAL_NAN = np.array([np.nan], dtype="<f8").tobytes()


def _hash_values(h: Any, values: pd.Index | pd.Series) -> None:
    s = pd.Series(values, copy=False) if isinstance(values, pd.Index) else values
    dtype = s.dtype
    if isinstance(dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(dtype):
        idx = pd.DatetimeIndex(s)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC")
        h.update(b"datetime\x00")
        h.update(datetime_ns(idx).astype("<i8").tobytes())
        h.update(np.asarray(idx.isna(), dtype=np.uint8).tobytes())
    elif pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
        if s.isna().any():
            h.update(b"float\x00")
            arr = s.astype("float64").to_numpy(dtype="<f8", na_value=np.nan)
            _update_float(h, arr)
        else:
            h.update(b"int\x00")
            h.update(s.to_numpy(dtype="<i8").tobytes())
    elif pd.api.types.is_float_dtype(dtype):
        h.update(b"float\x00")
        _update_float(h, s.to_numpy(dtype="<f8", na_value=np.nan))
    else:
        h.update(b"str\x00")
        h.update("\x1f".join("\x00" if pd.isna(v) else str(v) for v in s.tolist()).encode("utf-8"))


def _update_float(h: Any, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype="<f8")
    nan = np.isnan(arr)
    if nan.any():
        arr = arr.copy()
        arr[nan] = 0.0
    h.update(arr.tobytes())
    h.update(np.packbits(nan).tobytes())


def hash_dataframe(df: pd.DataFrame | pd.Series) -> str:
    """SHA-256 of a DataFrame/Series' schema and contents (see module docstring)."""
    frame = df.to_frame() if isinstance(df, pd.Series) else df
    h = hashlib.sha256()
    h.update(f"shape={frame.shape}".encode())
    h.update(b"index\x00")
    _hash_values(h, frame.index)
    for col in frame.columns:
        h.update(b"\x1ecol\x00" + str(col).encode("utf-8") + b"\x00")
        _hash_values(h, frame[col])
    return h.hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Canonical JSON (sorted keys, no whitespace) for hashing."""
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_config(obj: Any) -> str:
    """SHA-256 of the canonical JSON representation of a configuration object."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def schema_of(df: pd.DataFrame) -> dict[str, Any]:
    """Column names, dtypes, row count and time span (for manifests)."""
    out: dict[str, Any] = {
        "rows": len(df),
        "columns": {str(c): str(df[c].dtype) for c in df.columns},
        "index_dtype": str(df.index.dtype),
    }
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        out["start"] = df.index.min().isoformat()
        out["end"] = df.index.max().isoformat()
        out["timezone"] = str(df.index.tz) if df.index.tz is not None else None
    return out
