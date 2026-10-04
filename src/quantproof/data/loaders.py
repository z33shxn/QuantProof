"""Loading tabular market data from CSV, Parquet, or in-memory objects.

Loading is deliberately separated from cleaning: :func:`load_frame` returns the
raw table (with normalized column names) so that :mod:`quantproof.data.validation`
can report problems such as unsorted or duplicated timestamps *before*
:func:`prepare_prices` fixes them, and every fix is reported as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantproof.data.panel import find_symbol_column, flatten_multiindex
from quantproof.data.schemas import find_timestamp_column, normalize_column_name
from quantproof.errors import QuantProofDataError

DataSource = str | Path | pd.DataFrame | pd.Series

_OFFSET_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


@dataclass
class ParsedTimestamps:
    """Result of timestamp parsing, with diagnostics used by data validation."""

    values: pd.DatetimeIndex
    n_invalid: int
    invalid_examples: list[str]
    n_naive: int
    n_aware: int
    offsets: list[str]
    tz_aware: bool


@dataclass
class PreparedData:
    """Analysis-ready prices plus a log of every transformation applied."""

    prices: pd.DataFrame
    actions: list[str] = field(default_factory=list)
    timestamp_column: str | None = None


def load_frame(source: DataSource, *, timestamp_column: str | None = None) -> pd.DataFrame:
    """Read a CSV/Parquet file or accept a DataFrame/Series; normalize column names.

    The timestamp column (if not already the index) is kept as a column so data
    validation can inspect it in raw form.
    """
    if isinstance(source, pd.Series):
        frame = source.to_frame(name=source.name if source.name is not None else "close")
    elif isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise QuantProofDataError(
                f"Data file not found: {path}. Pass a CSV/Parquet path or a pandas DataFrame."
            )
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".parquet", ".pq"}:
            raise QuantProofDataError(
                f"Unsupported data format {suffix!r} for {path}. Use .csv or .parquet."
            )
        try:
            frame = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise QuantProofDataError(
                "Reading Parquet requires pyarrow. Install with: pip install 'quantproof[parquet]'"
            ) from exc
        except (ValueError, OSError) as exc:
            # pandas/pyarrow parse errors (ParserError, EmptyDataError, ArrowInvalid and
            # UnicodeDecodeError are ValueErrors) are input problems, not QuantProof bugs.
            raise QuantProofDataError(
                f"Could not read {path} as {suffix.lstrip('.')}: {type(exc).__name__}: {exc}"
            ) from exc
    if isinstance(frame.index, pd.MultiIndex):
        frame = flatten_multiindex(frame)
    frame.columns = [normalize_column_name(c) for c in frame.columns]
    if timestamp_column is not None:
        timestamp_column = normalize_column_name(timestamp_column)
        if timestamp_column not in frame.columns:
            raise QuantProofDataError(
                f"Timestamp column {timestamp_column!r} not found. Available columns: "
                f"{list(frame.columns)}"
            )
    return frame


def resolve_timestamp_column(
    frame: pd.DataFrame, timestamp_column: str | None = None
) -> str | None:
    """Return the timestamp column name, or ``None`` if the index already holds timestamps."""
    if timestamp_column is not None:
        return normalize_column_name(timestamp_column)
    if isinstance(frame.index, pd.DatetimeIndex):
        return None
    col = find_timestamp_column(frame.columns)
    if col is None:
        raise QuantProofDataError(
            "Could not find a timestamp column. Expected a DatetimeIndex or a column named one "
            f"of: timestamp, datetime, date, time, ts. Found columns: {list(frame.columns)}. "
            "Pass timestamp_column=... to name it explicitly."
        )
    return col


def parse_timestamps(raw: pd.Series | pd.Index) -> ParsedTimestamps:
    """Parse timestamps, tolerating mixed offsets, and record diagnostics.

    * Values carrying an explicit UTC offset are converted to UTC.
    * If *all* values are naive they stay naive (assumed exchange-local time).
    * If naive and aware values are mixed, naive values are interpreted as UTC and the
      mixture is reported by data validation (QP-DATA-010).
    """
    series = pd.Series(raw, copy=False)
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        idx = pd.DatetimeIndex(series)
        return ParsedTimestamps(idx, 0, [], 0, len(idx), [str(series.dtype.tz)], True)
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        idx = pd.DatetimeIndex(series)
        n_invalid = int(idx.isna().sum())
        return ParsedTimestamps(idx, n_invalid, [], len(idx) - n_invalid, 0, [], False)
    if pd.api.types.is_numeric_dtype(series.dtype):
        raise QuantProofDataError(
            "Timestamp column is numeric. Convert epoch values explicitly, e.g. "
            "pd.to_datetime(df['ts'], unit='s', utc=True), so the unit is unambiguous."
        )
    text = series.astype("string")
    offsets = text.str.strip().str.extract(_OFFSET_RE, expand=False)
    has_offset = offsets.notna() & text.notna()
    n_aware = int(has_offset.sum())
    n_nonnull = int(text.notna().sum())
    n_naive = n_nonnull - n_aware
    distinct = sorted({str(o) for o in offsets.dropna().unique()})
    tz_aware = n_aware > 0
    parsed = pd.to_datetime(text, errors="coerce", utc=tz_aware, format="mixed")
    idx = pd.DatetimeIndex(parsed)
    invalid_mask = idx.isna() & text.notna().to_numpy()
    n_invalid = int(invalid_mask.sum()) + int(text.isna().sum())
    examples = [str(v) for v in text[invalid_mask].head(5).tolist()]
    return ParsedTimestamps(idx, n_invalid, examples, n_naive, n_aware, distinct, tz_aware)


def _coerce_numeric(data: pd.DataFrame) -> pd.DataFrame:
    for col in data.columns:
        dtype = data[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() == data[col].notna().sum() and converted.notna().any():
                data[col] = converted
    return data


def prepare_panel(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    symbol_column: str | None = None,
    keep: str = "last",
) -> PreparedData:
    """Prepare long-format multi-asset data into a ``(timestamp, symbol)`` MultiIndex panel."""
    actions: list[str] = []
    frame = flatten_multiindex(frame)
    sym_col = (
        normalize_column_name(symbol_column) if symbol_column else find_symbol_column(frame.columns)
    )
    if sym_col is None or sym_col not in frame.columns:
        raise QuantProofDataError("No symbol column found for panel data (expected e.g. 'symbol').")
    ts_col = resolve_timestamp_column(frame, timestamp_column)
    if ts_col is None:
        raise QuantProofDataError(
            "Panel data needs a timestamp column (or a (timestamp, symbol) MultiIndex)."
        )
    parsed = parse_timestamps(frame[ts_col])
    data = frame.drop(columns=[ts_col]).copy()
    data["__ts__"] = parsed.values
    invalid = data["__ts__"].isna().to_numpy()
    if invalid.any():
        data = data.loc[~invalid]
        actions.append(f"Dropped {int(invalid.sum())} row(s) with invalid timestamps.")
    data[sym_col] = data[sym_col].astype(str)
    data = data.set_index(["__ts__", sym_col])
    data.index = data.index.set_names(["timestamp", "symbol"])
    if not data.index.is_monotonic_increasing:
        data = data.sort_index(kind="mergesort")
        actions.append("Sorted rows by (timestamp, symbol).")
    dup = data.index.duplicated(keep="first" if keep == "first" else "last")
    if dup.any():
        data = data.loc[~dup]
        actions.append(
            f"Removed {int(dup.sum())} duplicated (timestamp, symbol) row(s), keeping the {keep}."
        )
    data = _coerce_numeric(data)
    if data.select_dtypes(include=[np.number]).shape[1] == 0:
        raise QuantProofDataError(
            "No numeric columns found in the panel; a 'close' column is required."
        )
    if data.index.get_level_values(0).nunique() < 2:
        raise QuantProofDataError("Panel data needs at least 2 distinct timestamps.")
    return PreparedData(prices=data, actions=actions, timestamp_column=ts_col)


def prepare_prices(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    symbol_column: str | None = None,
    keep: str = "last",
) -> PreparedData:
    """Turn a raw frame into a sorted, de-duplicated price table.

    Single-asset data get a DatetimeIndex; data with a symbol column (or a
    ``(timestamp, symbol)`` MultiIndex) become a panel (see :mod:`quantproof.data.panel`).
    Every modification is recorded in :attr:`PreparedData.actions`; nothing is changed
    silently.
    """
    if (
        symbol_column is not None
        or isinstance(frame.index, pd.MultiIndex)
        or find_symbol_column(frame.columns)
    ):
        return prepare_panel(
            frame, timestamp_column=timestamp_column, symbol_column=symbol_column, keep=keep
        )
    actions: list[str] = []
    ts_col = resolve_timestamp_column(frame, timestamp_column)
    if ts_col is None:
        parsed = parse_timestamps(frame.index)
        data = frame.copy()
    else:
        parsed = parse_timestamps(frame[ts_col])
        data = frame.drop(columns=[ts_col])
    data.index = parsed.values
    data.index.name = "timestamp"

    invalid = data.index.isna()
    if invalid.any():
        data = data.loc[~invalid]
        actions.append(f"Dropped {int(invalid.sum())} row(s) with invalid timestamps.")
    if not data.index.is_monotonic_increasing:
        data = data.sort_index(kind="mergesort")
        actions.append("Sorted rows by timestamp (input was not monotonically increasing).")
    dup = data.index.duplicated(keep="first" if keep == "first" else "last")
    if dup.any():
        data = data.loc[~dup]
        actions.append(f"Removed {int(dup.sum())} duplicated timestamp(s), keeping the {keep}.")

    data = _coerce_numeric(data)
    numeric = data.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise QuantProofDataError(
            "No numeric columns found after parsing. QuantProof needs at least a price column "
            "(e.g. 'close')."
        )
    if len(data) < 2:
        raise QuantProofDataError(
            f"Only {len(data)} usable row(s) after cleaning; at least 2 are required. Check the "
            "file and the timestamp column (data validation reports QP-DATA-014)."
        )
    return PreparedData(prices=data, actions=actions, timestamp_column=ts_col)


def load_prices(
    source: DataSource, *, timestamp_column: str | None = None, symbol_column: str | None = None
) -> pd.DataFrame:
    """Convenience: load and prepare price data in one call.

    Raises :class:`QuantProofDataError` if the data cannot be used.
    """
    frame = load_frame(source, timestamp_column=timestamp_column)
    return prepare_prices(
        frame, timestamp_column=timestamp_column, symbol_column=symbol_column
    ).prices


def load_returns(source: DataSource, *, column: str | None = None) -> pd.Series:
    """Load a returns series from CSV/Parquet/Series/DataFrame.

    If ``column`` is not given, a column named ``returns``/``return``/``ret`` is used,
    falling back to the only numeric column.
    """
    if isinstance(source, pd.Series):
        return source.astype(float)
    frame = load_frame(source)
    try:
        prepared = prepare_prices(frame)
        data = prepared.prices
    except QuantProofDataError:
        data = frame
    if column is not None:
        col = normalize_column_name(column)
        if col not in data.columns:
            raise QuantProofDataError(f"Column {col!r} not found; available: {list(data.columns)}")
        return data[col].astype(float)
    for cand in ("returns", "return", "ret", "strategy_returns", "pnl"):
        if cand in data.columns:
            return data[cand].astype(float)
    numeric = data.select_dtypes(include=[np.number])
    if numeric.shape[1] == 1:
        return numeric.iloc[:, 0].astype(float)
    raise QuantProofDataError(
        "Could not identify the returns column; pass column=... explicitly. "
        f"Numeric columns: {list(numeric.columns)}"
    )


def describe_source(source: Any) -> str:
    """Short identifier of a data source for manifests."""
    if isinstance(source, (str, Path)):
        return str(source)
    if isinstance(source, pd.DataFrame):
        return f"<DataFrame {source.shape[0]}x{source.shape[1]}>"
    if isinstance(source, pd.Series):
        return f"<Series {len(source)}>"
    return type(source).__name__
